"""Reload a trained decoder and evaluate real validation RGB through VACE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import smplx
import torch
from torch.utils.data import DataLoader

from models.human_decoder import HumanDecoderConfig, HumanQueryTransformer
from real_3dpw_dataset import Real3DPWDataset
from real_vace_feature_encoder import FrozenVaceFeatureEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vace-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--body-model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("feature_backend") != "vace":
        raise ValueError("This evaluator requires a frozen-VACE decoder checkpoint")

    decoder = HumanQueryTransformer(
        HumanDecoderConfig(**checkpoint["decoder_config"])
    ).to(device).eval()
    decoder.load_state_dict(checkpoint["decoder"], strict=True)

    training_args = checkpoint["args"]
    image_size = int(training_args["image_size"])
    feature_layers = tuple(checkpoint["feature_layers"])
    timestep = int(checkpoint["vace_timestep"])
    dataset = Real3DPWDataset(
        root=args.data_root,
        split="validation",
        image_size=image_size,
        frame_stride=1,
        max_samples=args.max_samples,
    )
    if not dataset:
        raise RuntimeError("The real 3DPW validation dataset is empty")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    encoder = FrozenVaceFeatureEncoder(
        args.vace_checkpoint_dir,
        device=device,
        image_size=image_size,
        feature_layers=feature_layers,
        timestep=timestep,
    )
    grid_size = (1, image_size // 16, image_size // 16)
    sums = {
        name: 0.0
        for name in (
            "body_pose_squared_error",
            "body_pose_absolute_error",
            "global_orient_squared_error",
            "global_orient_absolute_error",
            "betas_squared_error",
            "betas_absolute_error",
            "transl_squared_error",
            "transl_absolute_error",
        )
    }
    sample_count = 0
    first = None

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        features = encoder.encode(images)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = decoder(features, grid_size=grid_size, target_frames=1)
        predicted = {
            "body_pose": prediction["body_pose"][:, 0].float(),
            "global_orient": prediction["global_orient"][:, 0].float(),
            "betas": prediction["betas"].float(),
            "transl": prediction["transl"][:, 0].float(),
        }
        batch_size = images.shape[0]
        for name, value in predicted.items():
            target = batch[name].to(device).float()
            error = value - target
            sums[f"{name}_squared_error"] += float(error.square().mean()) * batch_size
            sums[f"{name}_absolute_error"] += float(error.abs().mean()) * batch_size
        sample_count += batch_size

        if first is None:
            first = {
                "sequence": batch["sequence"][0],
                "image_path": batch["image_path"][0],
                "frame_id": int(batch["frame_id"][0]),
                "feature_shape": tuple(features[:1].shape),
                "global_orient": predicted["global_orient"][:1].cpu(),
                "body_pose": predicted["body_pose"][:1].cpu(),
                "betas": predicted["betas"][:1].cpu(),
                "transl": predicted["transl"][:1].cpu(),
            }

    encoder.close()
    if first is None:
        raise RuntimeError("No validation samples were evaluated")

    body = smplx.create(
        str(args.body_model_root),
        model_type="smplx",
        gender="neutral",
        num_betas=10,
        use_pca=False,
        batch_size=1,
    ).eval().to(device)
    zeros45 = torch.zeros(1, 45, device=device)
    zeros3 = torch.zeros(1, 3, device=device)
    zeros10 = torch.zeros(1, 10, device=device)
    with torch.inference_mode():
        mesh = body(
            global_orient=first["global_orient"].to(device),
            body_pose=first["body_pose"].to(device),
            left_hand_pose=zeros45,
            right_hand_pose=zeros45,
            jaw_pose=zeros3,
            expression=zeros10,
            betas=first["betas"].to(device),
            transl=first["transl"].to(device),
        )
    if not torch.isfinite(mesh.vertices).all():
        raise AssertionError("Reloaded checkpoint produced a non-finite SMPL-X mesh")

    metrics = {}
    for name in ("body_pose", "global_orient", "betas", "transl"):
        mse = sums[f"{name}_squared_error"] / sample_count
        metrics[f"{name}_rmse"] = mse**0.5
        metrics[f"{name}_mae"] = sums[f"{name}_absolute_error"] / sample_count

    result = {
        "status": "DIVIBODY_REAL_CHECKPOINT_RELOAD_OK",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_best_validation_total": float(checkpoint["best_val"]),
        "real_validation_samples": sample_count,
        "feature_backend": "official frozen VACE",
        "feature_layers": feature_layers,
        "supervised_outputs": ["body_pose", "global_orient", "betas", "transl"],
        "metrics": metrics,
        "first_validation_sample": {
            "sequence": first["sequence"],
            "frame_id": first["frame_id"],
            "image_path": first["image_path"],
            "feature_shape": first["feature_shape"],
        },
        "smplx_vertices_shape": tuple(mesh.vertices.shape),
        "smplx_joints_shape": tuple(mesh.joints.shape),
        "smplx_mesh_finite": True,
        "note": "3DPW has no hand/face ground truth; zero hand/face pose was used only for mesh validation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "result": result,
            "vertices": mesh.vertices.cpu(),
            "joints": mesh.joints.cpu(),
            "first_prediction": {
                key: value for key, value in first.items() if torch.is_tensor(value)
            },
        },
        args.output.with_suffix(".pt"),
    )
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
