"""Evaluate a native v27 checkpoint on the complete cached PoseTrack18 val split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from divibody_training.evaluate_standard_3dpw import create_smpl_models
from divibody_training.ram_parts_runtime_v27 import evaluate_pose, model_from_payload
from divibody_training.ram_pose_stream_v10 import ShuffledPoseDatasetV10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vace-dir", type=Path, required=True)
    parser.add_argument("--hmr-dir", type=Path, required=True)
    parser.add_argument("--smpl-model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp-dtype", choices=("bf16", "none"), default="bf16")
    args = parser.parse_args()

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") != "ram-v9-geometry-coupled-v27":
        raise ValueError("This evaluator requires a native v27 checkpoint")

    dataset = ShuffledPoseDatasetV10(
        args.vace_dir, args.hmr_dir, args.seed, max_windows=0
    )
    if dataset.split != "val":
        raise ValueError(f"Expected PoseTrack validation split, got {dataset.split!r}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = model_from_payload(payload).to(device)
    geometry_models = create_smpl_models(args.smpl_model_root, device)
    for value in geometry_models.values():
        value.requires_grad_(False)
    model.bind_geometry(geometry_models[2])
    from hmr2.models.smpl_wrapper import SMPL as OfficialSMPL

    official_smpl = OfficialSMPL(
        model_path=str(args.smpl_model_root),
        gender="neutral",
        num_betas=10,
        joint_regressor_extra=str(args.smpl_model_root.parent / "SMPL_to_J19.pkl"),
    ).to(device).requires_grad_(False)

    metrics = evaluate_pose(loader, model, official_smpl, args, device)
    result = {
        "status": "RAM_V27_FULL_POSETRACK18_VAL_OK",
        "model": "DiviBody v27 geometry-coupled body reconstruction",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_variant": payload.get("variant"),
        "metrics": metrics,
        "coverage": {
            "windows": len(dataset),
            "visible_keypoints": int(metrics["visible_keypoints"]),
        },
        "protocol": (
            "complete cached PoseTrack18 validation split; visible owned keypoints; "
            "PCK normalized by the 1.25-expanded 96-pixel person crop; direct SMPL "
            "projection is primary and assisted residual output is reported separately"
        ),
        "test_used_for_model_selection": False,
        "checkpoint_development_only": bool(payload.get("development_only", True)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
