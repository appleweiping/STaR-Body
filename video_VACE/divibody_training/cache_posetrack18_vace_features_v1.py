"""Build an additive PoseTrack18 continuous-person VACE cache.

The existing 3DPW cache builders and model files are imported read-only.  This
adapter preserves their 17-RGB/8-target temporal contract while adding public
PoseTrack 2D supervision and deterministic person crops.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from divibody_training.cache_continuous_3dpw_vace_features import (
    FrameIdentity,
    checkpoint_tree_fingerprint,
    sha256_file,
)
from divibody_training.cache_continuous_3dpw_vace_features_v5 import (
    CONTEXT_FRAMES,
    FEATURE_WIDTH,
    IMAGE_SIZE,
    LATENT_FRAMES,
    LATENT_SOURCE_RANGES,
    TARGET_FRAMES,
    TOKEN_GRID,
    build_v5_window_plan,
    development_window_subset,
    stable_noise_seed,
    v5_metadata,
    validate_v5_window_plan,
)
from divibody_training.ram_ready_adapter_v1 import PoseTrack18PersonDataset
from divibody_training.real_vace_temporal_feature_encoder import wan_latent_frames
from divibody_training.real_vace_temporal_feature_encoder_v2 import (
    ProtocolAwareTemporalVaceFeatureEncoder,
)


FORMAT = "ram-posetrack18-contiguous-vace-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--vace-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-layers", type=int, nargs="+", default=(7, 14, 22))
    parser.add_argument("--vace-timestep", type=int, default=0)
    parser.add_argument(
        "--diffusion-input-protocol",
        choices=("clean", "flow_noised"),
        default="clean",
    )
    parser.add_argument("--text-embedding-path", type=Path)
    parser.add_argument("--noise-seed-base", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--development-max-windows", type=int, default=0)
    return parser.parse_args()


def identities(dataset: PoseTrack18PersonDataset) -> list[FrameIdentity]:
    result = [
        FrameIdentity(
            dataset_index=index,
            sequence=str(sample["video_id"]),
            person_id=int(sample["track_id"]),
            frame_id=int(sample["frame_id"]),
            source_frame_id=int(sample["image_id"]),
            image_path=str(sample["image_path"]),
        )
        for index, sample in enumerate(dataset.samples)
    ]
    keys = [item.key for item in result]
    if len(keys) != len(set(keys)):
        raise ValueError("PoseTrack contains duplicate (video, track, frame) identities")
    return result


class PoseTrackWindowDataset(Dataset):
    def __init__(self, base: PoseTrack18PersonDataset, plan: Any) -> None:
        self.base = base
        self.plan = plan

    def __len__(self) -> int:
        return len(self.plan.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        window = self.plan.windows[index]
        unique = {
            frame.dataset_index
            for frame in (*window.context_frames, *window.target_frames)
        }
        loaded = {source: self.base[source] for source in unique}
        context = [loaded[frame.dataset_index] for frame in window.context_frames]
        targets = [loaded[frame.dataset_index] for frame in window.target_frames]
        return {
            "window_index": torch.tensor(index, dtype=torch.long),
            "images": torch.stack([item["image"] for item in context]),
            "target_keypoints": torch.stack([item["keypoints"] for item in targets]),
            "target_crop_center": torch.stack([item["crop_center"] for item in targets]),
            "target_crop_side": torch.stack([item["crop_side"] for item in targets]),
        }


def validate_payload(payload: dict[str, Any]) -> None:
    signature = payload.get("signature", {})
    if signature.get("format") != FORMAT:
        raise ValueError("Unexpected PoseTrack VACE cache format")
    count = len(payload.get("metadata", []))
    expected = (count, LATENT_FRAMES, *TOKEN_GRID, FEATURE_WIDTH)
    features = payload.get("features")
    if not isinstance(features, torch.Tensor) or tuple(features.shape) != expected:
        raise ValueError(f"Invalid VACE feature shape: expected {expected}")
    if features.dtype != torch.bfloat16 or not bool(torch.isfinite(features.float()).all()):
        raise ValueError("PoseTrack VACE features must be finite BF16")
    shapes = {
        "context_edge_repeat_mask": (count, CONTEXT_FRAMES),
        "target_valid_mask": (count, TARGET_FRAMES),
        "frame_owner_mask": (count, TARGET_FRAMES),
        "velocity_owner_mask": (count, TARGET_FRAMES - 1),
        "acceleration_owner_mask": (count, TARGET_FRAMES - 2),
        "target_keypoints": (count, TARGET_FRAMES, 17, 3),
        "target_crop_center": (count, TARGET_FRAMES, 2),
        "target_crop_side": (count, TARGET_FRAMES),
    }
    for name, shape in shapes.items():
        value = payload.get(name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"Invalid {name} shape; expected {shape}")
        if value.is_floating_point() and not bool(torch.isfinite(value.float()).all()):
            raise ValueError(f"{name} contains non-finite values")


def main() -> None:
    args = parse_args()
    if args.output.suffix != ".pt":
        raise ValueError("output must end in .pt")
    summary_path = args.output.with_suffix(".json")
    if args.output.exists() or summary_path.exists():
        raise FileExistsError("Refusing to overwrite an existing cache")
    if args.batch_size <= 0 or args.num_workers < 0 or args.development_max_windows < 0:
        raise ValueError("Invalid batch/worker/development arguments")
    if wan_latent_frames(CONTEXT_FRAMES) != LATENT_FRAMES:
        raise RuntimeError("Wan temporal geometry changed: expected 17 RGB -> 5 latents")

    started = time.monotonic()
    dataset = PoseTrack18PersonDataset(
        args.ready_root,
        split=args.split,
        image_size=IMAGE_SIZE,
        bbox_scale=1.25,
        min_keypoints=1,
    )
    plan = build_v5_window_plan(identities(dataset))
    if args.development_max_windows:
        plan = development_window_subset(plan, args.development_max_windows)
    validate_v5_window_plan(plan)
    if not plan.windows:
        raise RuntimeError("PoseTrack split has no temporal windows")

    encoder = ProtocolAwareTemporalVaceFeatureEncoder(
        checkpoint_dir=args.vace_checkpoint_dir,
        device=args.device,
        image_size=IMAGE_SIZE,
        feature_layers=args.feature_layers,
        timestep=args.vace_timestep,
        diffusion_input_protocol=args.diffusion_input_protocol,
        text_embedding_path=args.text_embedding_path,
    )
    loader = DataLoader(
        PoseTrackWindowDataset(dataset, plan),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    feature_chunks: list[torch.Tensor] = []
    keypoint_chunks: list[torch.Tensor] = []
    center_chunks: list[torch.Tensor] = []
    side_chunks: list[torch.Tensor] = []
    expected_window = 0
    try:
        for step, batch in enumerate(loader, start=1):
            indices = batch["window_index"].tolist()
            expected = list(range(expected_window, expected_window + len(indices)))
            if indices != expected:
                raise RuntimeError("DataLoader changed deterministic window order")
            expected_window += len(indices)
            seeds = [stable_noise_seed(plan.windows[i].uid, args.noise_seed_base) for i in indices]
            noise = seeds if args.diffusion_input_protocol == "flow_noised" else None
            clips = batch["images"].to(args.device, dtype=torch.float32, non_blocking=True)
            encoded = encoder.encode(clips, noise_seeds=noise)
            if tuple(encoded.shape[1:]) != (LATENT_FRAMES, *TOKEN_GRID, FEATURE_WIDTH):
                raise RuntimeError(f"Unexpected VACE feature shape {tuple(encoded.shape)}")
            feature_chunks.append(encoded.cpu().to(torch.bfloat16))
            keypoint_chunks.append(batch["target_keypoints"].float())
            center_chunks.append(batch["target_crop_center"].float())
            side_chunks.append(batch["target_crop_side"].float())
            if step % args.log_interval == 0 or step == len(loader):
                print(json.dumps({"phase": "posetrack18_vace", "split": args.split,
                                  "windows": expected_window, "total": len(plan.windows),
                                  "seconds": time.monotonic() - started}), flush=True)
    finally:
        encoder.close()

    encoder_path = Path(__file__).with_name("real_vace_temporal_feature_encoder_v2.py")
    payload = {
        "signature": {
            "format": FORMAT,
            "split": args.split,
            "development_only": bool(plan.stats.get("development_only")),
            "ready_root": str(args.ready_root.resolve()),
            "dataset_audit": dataset.audit(),
            "window_plan": plan.stats,
            "context_frames_rgb": CONTEXT_FRAMES,
            "target_frames": TARGET_FRAMES,
            "target_stride": 8,
            "crop_protocol": "PoseTrack GT square bbox, scale=1.25, per-person per-frame",
            "feature_layers": list(args.feature_layers),
            "encoder_protocol": encoder.protocol_signature(),
            "checkpoint": checkpoint_tree_fingerprint(args.vace_checkpoint_dir),
            "builder_source_sha256": sha256_file(Path(__file__)),
            "encoder_source_sha256": sha256_file(encoder_path),
        },
        "features": torch.cat(feature_chunks),
        "context_edge_repeat_mask": torch.tensor(
            [window.context_edge_repeat for window in plan.windows], dtype=torch.bool
        ),
        "target_valid_mask": torch.tensor(
            [window.target_valid for window in plan.windows], dtype=torch.bool
        ),
        "frame_owner_mask": torch.tensor(
            [window.frame_owner for window in plan.windows], dtype=torch.bool
        ),
        "velocity_owner_mask": torch.tensor(
            [window.velocity_owner for window in plan.windows], dtype=torch.bool
        ),
        "acceleration_owner_mask": torch.tensor(
            [window.acceleration_owner for window in plan.windows], dtype=torch.bool
        ),
        "target_keypoints": torch.cat(keypoint_chunks),
        "target_crop_center": torch.cat(center_chunks),
        "target_crop_side": torch.cat(side_chunks),
        "metadata": v5_metadata(plan),
    }
    validate_payload(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, args.output)
    result = {
        "status": "RAM_POSETRACK18_VACE_V1_OK",
        "split": args.split,
        "development_only": payload["signature"]["development_only"],
        "windows": len(plan.windows),
        "feature_shape": list(payload["features"].shape),
        "owned_person_frames": int(payload["frame_owner_mask"].sum()),
        "output": str(args.output.resolve()),
        "output_bytes": args.output.stat().st_size,
        "output_sha256": sha256_file(args.output),
        "seconds": time.monotonic() - started,
    }
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
