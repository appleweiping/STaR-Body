"""Train the additive high-resolution RGB-anchor + continuous-VACE v7 model.

This is a new training path; no v3-v6 source is modified.  The cache is
expected to contain official HMR image tokens and base SMPL predictions for the
same eight center frames as the continuous T17 VACE cache.  Test caches are
explicitly rejected so model selection remains validation-only.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from divibody_training.evaluate_standard_3dpw import create_smpl_models
from divibody_training.geometry_metrics_3dpw import (
    hip_centered_j14,
    mean_euclidean_error,
    pa_mpjpe_per_sample,
    pelvis_aligned_j14,
)
from divibody_training.train_divibody_continuous_geometry_v5 import (
    camera_targets,
    gendered_vertices,
    masked_mean,
    smpl_vertices,
)
from models.human_decoder_rgb_vace_v7 import (
    HumanRGBVaceGeometryTransformerV7,
    RGBVaceGeometryConfig,
)


FORMAT = "divibody-rgb-vace-geometry-v7"
CACHE_FORMAT = "divibody-rgb-vace-cache-v7"
TARGET_FRAMES = 8
REQUIRED_INPUTS = (
    "rgb_tokens",
    "vace_features",
    "base_rotmat",
    "base_betas",
    "base_camera",
)
REQUIRED_TARGETS = ("full_pose", "betas", "gender_code", "cam_pose")
REQUIRED_MASKS = ("frame_owner", "velocity_owner", "acceleration_owner")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--smpl-model-root", type=Path, required=True)
    parser.add_argument("--joint-regressor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--joint-weight", type=float, default=1.0)
    parser.add_argument("--vertex-weight", type=float, default=0.5)
    parser.add_argument("--rotation-weight", type=float, default=0.05)
    parser.add_argument("--shape-weight", type=float, default=0.001)
    parser.add_argument("--velocity-weight", type=float, default=0.5)
    parser.add_argument("--acceleration-weight", type=float, default=0.25)
    parser.add_argument("--anchor-weight", type=float, default=0.02)
    parser.add_argument("--amp-dtype", choices=("bf16", "none"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _finite_tensor(payload: dict[str, Any], name: str) -> torch.Tensor:
    value = payload.get(name)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Cache field {name!r} must be a tensor")
    if not value.is_floating_point() or not bool(torch.isfinite(value.float()).all()):
        raise ValueError(f"Cache field {name!r} must contain finite floats")
    return value


def validate_cache(payload: dict[str, Any], expected_split: str) -> dict[str, Any]:
    """Validate a v7 cache before allocating a model or touching test data."""

    if payload.get("format") != CACHE_FORMAT:
        raise ValueError(f"Expected cache format {CACHE_FORMAT!r}")
    signature = payload.get("signature")
    if not isinstance(signature, dict) or signature.get("split") != expected_split:
        raise ValueError(f"Expected a {expected_split!r} v7 cache")
    if expected_split == "test" or signature.get("contains_test_ground_truth"):
        raise ValueError("v7 training refuses caches containing test ground truth")
    inputs = payload.get("inputs")
    targets = payload.get("targets")
    masks = payload.get("masks")
    if not isinstance(inputs, dict) or not isinstance(targets, dict):
        raise TypeError("Cache inputs and targets must be dictionaries")
    if not isinstance(masks, dict):
        raise TypeError("Cache masks must be a dictionary")
    tensors = {name: _finite_tensor(inputs, name) for name in REQUIRED_INPUTS}
    count = tensors["rgb_tokens"].shape[0]
    if count < 1:
        raise ValueError("v7 cache cannot be empty")
    if tensors["rgb_tokens"].ndim != 4:
        raise ValueError("Cached rgb_tokens must be [W,T,N,C]")
    if tensors["rgb_tokens"].shape[1] != TARGET_FRAMES:
        raise ValueError("Cached RGB tokens must contain eight target frames")
    if tensors["vace_features"].ndim != 5:
        raise ValueError("Cached vace_features must be [W,F,H,W,C]")
    if tensors["base_rotmat"].shape != (count, TARGET_FRAMES, 24, 3, 3):
        raise ValueError("Invalid cached HMR rotation shape")
    if tensors["base_betas"].shape not in (
        (count, TARGET_FRAMES, 10),
        (count, 10),
    ):
        raise ValueError("Invalid cached HMR shape shape")
    if tensors["base_camera"].shape != (count, TARGET_FRAMES, 3):
        raise ValueError("Invalid cached HMR camera shape")
    if any(value.shape[0] != count for value in tensors.values()):
        raise ValueError("v7 input tensors have different window counts")
    for name in REQUIRED_TARGETS:
        tensor = targets.get(name)
        if not isinstance(tensor, torch.Tensor) or tensor.shape[0] != count:
            raise ValueError(f"Invalid target tensor {name!r}")
    for name in REQUIRED_MASKS:
        tensor = masks.get(name)
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.bool:
            raise TypeError(f"Mask {name!r} must be boolean")
        expected_width = (
            TARGET_FRAMES
            - {"frame_owner": 0, "velocity_owner": 1, "acceleration_owner": 2}[name]
        )
        if tensor.shape != (count, expected_width):
            raise ValueError(f"Invalid mask shape for {name!r}")
    return {
        "split": expected_split,
        "windows": count,
        "rgb_token_shape": list(tensors["rgb_tokens"].shape[1:]),
        "vace_feature_shape": list(tensors["vace_features"].shape[1:]),
        "hmr_protocol": signature.get("hmr_protocol"),
        "vace_protocol": signature.get("vace_protocol"),
    }


class RGBVaceGeometryDataset(Dataset):
    def __init__(self, payload: dict[str, Any]) -> None:
        self.inputs = payload["inputs"]
        self.targets = payload["targets"]
        self.masks = payload["masks"]
        self.length = self.inputs["rgb_tokens"].shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            **{name: self.inputs[name][index] for name in REQUIRED_INPUTS},
            **{name: self.targets[name][index] for name in REQUIRED_TARGETS},
            **{name: self.masks[name][index] for name in REQUIRED_MASKS},
        }


def model_config(payload: dict[str, Any]) -> RGBVaceGeometryConfig:
    rgb = payload["inputs"]["rgb_tokens"]
    vace = payload["inputs"]["vace_features"]
    return RGBVaceGeometryConfig(
        rgb_input_dim=rgb.shape[-1],
        vace_input_dim=vace.shape[-1],
        output_frames=TARGET_FRAMES,
    )


def forward_batch(
    model: HumanRGBVaceGeometryTransformerV7,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    amp_dtype: str,
) -> dict[str, torch.Tensor]:
    values = [
        batch[name].to(device=device, non_blocking=True) for name in REQUIRED_INPUTS
    ]
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and amp_dtype == "bf16",
    ):
        return model(*values)


def geometric_losses(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    models: dict[int, torch.nn.Module],
    regressor: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    with torch.autocast(device_type=device.type, enabled=False):
        pred_rot = prediction["rotmat"].float()
        target_rot = camera_targets(batch, device)
        target_betas = batch["betas"].to(device=device, dtype=torch.float32)
        genders = batch["gender_code"].to(device=device).reshape(-1)
        pred_betas = prediction["betas"].float()
        pred_vertices = smpl_vertices(
            models[2], pred_rot.reshape(-1, 24, 3, 3), pred_betas.reshape(-1, 10)
        )
        gt_vertices = gendered_vertices(
            models,
            target_rot.reshape(-1, 24, 3, 3),
            target_betas.reshape(-1, 10),
            genders,
        )
        pred_joints, _ = pelvis_aligned_j14(pred_vertices, regressor)
        gt_joints, _ = pelvis_aligned_j14(gt_vertices, regressor)
        batch_size = pred_rot.shape[0]
        frame_mask = batch["frame_owner"].to(device)
        joint = masked_mean(
            mean_euclidean_error(pred_joints, gt_joints).reshape(batch_size, -1),
            frame_mask,
            "joint",
        )
        vertex = masked_mean(
            mean_euclidean_error(pred_vertices, gt_vertices).reshape(batch_size, -1),
            frame_mask,
            "vertex",
        )
        rotation = masked_mean(
            (pred_rot - target_rot).square().mean(dim=(-1, -2, -3)),
            frame_mask,
            "rotation",
        )
        shape = masked_mean(
            (pred_betas - target_betas).square().mean(dim=-1), frame_mask, "shape"
        )
        pred_motion = hip_centered_j14(pred_joints).reshape(
            batch_size, TARGET_FRAMES, 14, 3
        )
        gt_motion = hip_centered_j14(gt_joints).reshape(
            batch_size, TARGET_FRAMES, 14, 3
        )
        motion_error = pred_motion - gt_motion
        velocity = masked_mean(
            torch.linalg.vector_norm(torch.diff(motion_error, dim=1), dim=-1).mean(-1),
            batch["velocity_owner"].to(device),
            "velocity",
        )
        acceleration = masked_mean(
            torch.linalg.vector_norm(torch.diff(motion_error, n=2, dim=1), dim=-1).mean(
                -1
            ),
            batch["acceleration_owner"].to(device),
            "acceleration",
        )
        anchor = (
            (
                prediction["rotation_residual_6d"]
                - prediction["rotation_residual_6d"].new_tensor(
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
                )
            )
            .square()
            .mean()
            + (
                prediction["betas"] - batch["base_betas"].to(device)[:, None]
                if batch["base_betas"].ndim == 2
                else prediction["betas"] - batch["base_betas"].to(device)
            )
            .square()
            .mean()
            + (prediction["camera"] - batch["base_camera"].to(device)).square().mean()
        )
        total = (
            args.joint_weight * joint
            + args.vertex_weight * vertex
            + args.rotation_weight * rotation
            + args.shape_weight * shape
            + args.velocity_weight * velocity
            + args.acceleration_weight * acceleration
            + args.anchor_weight * anchor
        )
    return {
        "total": total,
        "joint_m": joint,
        "vertex_m": vertex,
        "rotation_chordal": rotation,
        "shape_mse": shape,
        "velocity_m_per_frame": velocity,
        "acceleration_m_per_frame2": acceleration,
        "anchor": anchor,
    }


@torch.no_grad()
def validate(
    loader: DataLoader,
    model: HumanRGBVaceGeometryTransformerV7,
    models: dict[int, torch.nn.Module],
    regressor: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    sums = {name: 0.0 for name in ("mpjpe_mm", "pa_mpjpe_mm", "pve_mm")}
    count = 0
    for batch in loader:
        prediction = forward_batch(model, batch, device, args.amp_dtype)
        pred_rot = prediction["rotmat"].float()
        target_rot = camera_targets(batch, device)
        target_betas = batch["betas"].to(device=device, dtype=torch.float32)
        genders = batch["gender_code"].to(device=device).reshape(-1)
        pred_vertices = smpl_vertices(
            models[2],
            pred_rot.reshape(-1, 24, 3, 3),
            prediction["betas"].float().reshape(-1, 10),
        )
        gt_vertices = gendered_vertices(
            models,
            target_rot.reshape(-1, 24, 3, 3),
            target_betas.reshape(-1, 10),
            genders,
        )
        pred_joints, _ = pelvis_aligned_j14(pred_vertices, regressor)
        gt_joints, _ = pelvis_aligned_j14(gt_vertices, regressor)
        mask = batch["frame_owner"].to(device).reshape(-1)
        arrays = {
            "mpjpe_mm": mean_euclidean_error(pred_joints, gt_joints) * 1000.0,
            "pa_mpjpe_mm": pa_mpjpe_per_sample(pred_joints, gt_joints) * 1000.0,
            "pve_mm": mean_euclidean_error(pred_vertices, gt_vertices) * 1000.0,
        }
        for name, values in arrays.items():
            selected = values.masked_select(mask)
            sums[name] += float(selected.double().sum())
        count += int(mask.sum())
    if count < 1:
        raise RuntimeError("Validation has no owned target frames")
    return {**{name: value / count for name, value in sums.items()}, "frames": count}


def save_checkpoint(
    path: Path,
    model: HumanRGBVaceGeometryTransformerV7,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation: dict[str, float],
    args: argparse.Namespace,
) -> None:
    payload = {
        "format": FORMAT,
        "epoch": epoch,
        "selection_metric": "validation_mpjpe_mm",
        "validation": validation,
        "model_config": vars(model.config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
        "test_used_for_model_selection": False,
        "architecture": "HMR RGB anchor + zero-gated continuous VACE residual",
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if any(
        value <= 0
        for value in (
            args.epochs,
            args.batch_size,
            args.val_batch_size,
            args.lr,
            args.grad_clip,
        )
    ):
        raise ValueError("Invalid positive training argument")
    if args.num_workers < 0 or any(
        value < 0
        for value in (
            args.joint_weight,
            args.vertex_weight,
            args.rotation_weight,
            args.shape_weight,
            args.velocity_weight,
            args.acceleration_weight,
            args.anchor_weight,
        )
    ):
        raise ValueError("Workers and loss weights cannot be negative")
    if any((args.output_dir / name).exists() for name in ("best.pt", "latest.pt")):
        raise FileExistsError("Refusing to overwrite an existing v7 experiment")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    train_payload = load_payload(args.train_cache)
    val_payload = load_payload(args.val_cache)
    train_audit = validate_cache(train_payload, "train")
    val_audit = validate_cache(val_payload, "validation")
    config = model_config(train_payload)
    if model_config(val_payload) != config:
        raise ValueError("Train/validation v7 feature dimensions differ")
    model = HumanRGBVaceGeometryTransformerV7(config).to(device)
    smpl_models = create_smpl_models(args.smpl_model_root, device)
    for smpl_model in smpl_models.values():
        smpl_model.requires_grad_(False)
    regressor_np = np.load(args.joint_regressor)
    if regressor_np.shape != (17, 6890) or not np.isfinite(regressor_np).all():
        raise ValueError("Invalid H36M joint regressor")
    regressor = torch.as_tensor(regressor_np, dtype=torch.float32, device=device)
    options = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        RGBVaceGeometryDataset(train_payload),
        batch_size=args.batch_size,
        shuffle=True,
        **options,
    )
    val_loader = DataLoader(
        RGBVaceGeometryDataset(val_payload),
        batch_size=args.val_batch_size,
        shuffle=False,
        **options,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "format": FORMAT,
                "train_cache": train_audit,
                "validation_cache": val_audit,
                "model_config": vars(config),
                "test_data_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    baseline = validate(val_loader, model, smpl_models, regressor, args, device)
    best_value = baseline["mpjpe_mm"]
    save_checkpoint(args.output_dir / "best.pt", model, optimizer, 0, baseline, args)
    metrics_path = args.output_dir / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"epoch": 0, "validation": baseline}) + "\n", encoding="utf-8"
    )
    print(json.dumps({"epoch": 0, "validation": baseline}))
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals: dict[str, float] = {}
        batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = forward_batch(model, batch, device, args.amp_dtype)
            losses = geometric_losses(
                prediction, batch, smpl_models, regressor, args, device
            )
            if not bool(torch.isfinite(losses["total"])):
                raise FloatingPointError("Non-finite v7 training loss")
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
            batches += 1
        validation = validate(val_loader, model, smpl_models, regressor, args, device)
        record = {
            "epoch": epoch,
            "train": {name: value / batches for name, value in totals.items()},
            "validation": validation,
            "temporal_gate": model.temporal_gates.detach().tanh().cpu().tolist(),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        save_checkpoint(
            args.output_dir / "latest.pt", model, optimizer, epoch, validation, args
        )
        if validation["mpjpe_mm"] < best_value:
            best_value = validation["mpjpe_mm"]
            save_checkpoint(
                args.output_dir / "best.pt", model, optimizer, epoch, validation, args
            )
        print(json.dumps(record))


if __name__ == "__main__":
    main()
