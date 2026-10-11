"""Create a strict per-frame VACE feature cache for temporal 3DPW inference."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from evaluate_standard_3dpw import (
    Standard3DPWDataset,
    audit_split,
    enforce_complete_test,
)
from real_vace_feature_encoder import FrozenVaceFeatureEncoder
from train_divibody_real_3dpw import metadata_item


TARGET_NAMES = (
    "global_orient",
    "body_pose",
    "betas",
    "transl",
    "full_pose",
    "gender_code",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache strict 3DPW VACE frame features"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), required=True
    )
    parser.add_argument("--vace-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--feature-layers", type=int, nargs="+", default=(7, 14, 22))
    parser.add_argument("--vace-timestep", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-interval", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite feature cache: {args.output}")
    if args.image_size <= 0 or args.image_size % 16:
        raise ValueError("image-size must be a positive multiple of 16")
    if args.batch_size <= 0 or args.num_workers < 0 or args.log_interval <= 0:
        raise ValueError("Invalid batch/worker/log interval")

    audit = audit_split(args.data_root, args.split)
    if args.split == "test":
        enforce_complete_test(audit, allow_incomplete=False)
    dataset = Standard3DPWDataset(
        root=args.data_root,
        split=args.split,
        image_size=args.image_size,
        frame_stride=1,
        max_samples=None,
        crop_to_person=True,
    )
    if len(dataset) != int(audit["available_valid_person_frames"]):
        raise RuntimeError(
            f"Strict dataset has {len(dataset)} frames; audit has "
            f"{audit['available_valid_person_frames']}"
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    device = torch.device(args.device)
    encoder = FrozenVaceFeatureEncoder(
        checkpoint_dir=args.vace_checkpoint_dir,
        device=device,
        image_size=args.image_size,
        feature_layers=args.feature_layers,
        timestep=args.vace_timestep,
    )
    feature_chunks: list[torch.Tensor] = []
    target_chunks: Dict[str, list[torch.Tensor]] = {name: [] for name in TARGET_NAMES}
    metadata: list[dict[str, Any]] = []
    excluded = {"image", *TARGET_NAMES}
    started = time.monotonic()
    seen = 0
    try:
        for step, batch in enumerate(loader, start=1):
            images = batch["image"].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            features = encoder.encode(images)
            expected_tokens = (args.image_size // 16) ** 2
            if tuple(features.shape[1:]) != (expected_tokens, 1536):
                raise RuntimeError(
                    f"Unexpected VACE feature shape {tuple(features.shape)}"
                )
            feature_chunks.append(features.to(device="cpu", dtype=torch.bfloat16))
            for name in TARGET_NAMES:
                value = batch[name].detach().cpu()
                target_chunks[name].append(
                    value.float() if value.is_floating_point() else value
                )
            for index in range(images.shape[0]):
                metadata.append(
                    {
                        key: metadata_item(value, index)
                        for key, value in batch.items()
                        if key not in excluded
                    }
                )
            seen += images.shape[0]
            if step % args.log_interval == 0 or step == len(loader):
                print(
                    json.dumps(
                        {
                            "phase": "cache_strict_3dpw_vace_features",
                            "split": args.split,
                            "step": step,
                            "steps": len(loader),
                            "samples": seen,
                            "seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
    finally:
        encoder.close()

    payload = {
        "signature": {
            "format": "divibody-strict-vace-features-v2",
            "split": args.split,
            "dataset_root": str(args.data_root.expanduser().resolve()),
            "dataset_size": len(dataset),
            "dataset_preflight": audit,
            "strict_rgb_resolver": "image_{annotation_frame_index:05d}",
            "crop": (
                "oracle poses2d person crop with 25 percent margin when valid; "
                "deterministic whole-image fallback otherwise"
            ),
            "image_size": args.image_size,
            "feature_backend": "official frozen Wan2.1-VACE",
            "feature_layers": list(args.feature_layers),
            "vace_timestep": args.vace_timestep,
            "checkpoint_dir": str(args.vace_checkpoint_dir.expanduser().resolve()),
            "tokens_per_frame": (args.image_size // 16) ** 2,
            "feature_width": 1536,
            "token_layout": "one strict RGB frame per contiguous spatial token block",
            "crop_applied_person_frames": sum(
                bool(item.get("crop_applied", False)) for item in metadata
            ),
            "whole_image_fallback_person_frames": sum(
                not bool(item.get("crop_applied", False)) for item in metadata
            ),
        },
        "features": torch.cat(feature_chunks),
        "targets": {name: torch.cat(chunks) for name, chunks in target_chunks.items()},
        "metadata": metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, args.output)
    result = {
        "status": "DIVIBODY_STRICT_3DPW_VACE_CACHE_OK",
        "split": args.split,
        "samples": len(dataset),
        "feature_shape": list(payload["features"].shape),
        "output": str(args.output.expanduser().resolve()),
        "seconds": time.monotonic() - started,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
