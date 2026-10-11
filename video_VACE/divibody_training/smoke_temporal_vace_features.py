"""Real-3DPW smoke test for the additive continuous-video VACE encoder."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from evaluate_standard_3dpw import Standard3DPWDataset
from real_vace_temporal_feature_encoder import (
    FrozenTemporalVaceFeatureEncoder,
    wan_latent_frames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--vace-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"),
                        default="validation")
    parser.add_argument("--clip-length", type=int, default=9)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--feature-layers", type=int, nargs="+", default=(7, 14, 22))
    parser.add_argument("--vace-timestep", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def first_consecutive_clip(dataset: Standard3DPWDataset,
                           clip_length: int) -> list[int]:
    tracks: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, sample in enumerate(dataset.samples):
        tracks[(sample.sequence, sample.person_index)].append(index)
    for _, indices in sorted(tracks.items()):
        ordered = sorted(indices, key=lambda index: dataset.samples[index].frame_index)
        for start in range(0, len(ordered) - clip_length + 1):
            selection = ordered[start:start + clip_length]
            frames = [dataset.samples[index].frame_index for index in selection]
            if all(right == left + 1 for left, right in zip(frames, frames[1:])):
                return selection
    raise RuntimeError(f"No consecutive {clip_length}-frame person track exists")


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu())


def main() -> None:
    args = parse_args()
    latent_frames = wan_latent_frames(args.clip_length)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite smoke result: {args.output}")
    if args.image_size <= 0 or args.image_size % 16:
        raise ValueError("image-size must be a positive multiple of 16")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The real VACE smoke test requires CUDA")

    started = time.monotonic()
    dataset = Standard3DPWDataset(
        root=args.data_root,
        split=args.split,
        image_size=args.image_size,
        frame_stride=1,
        max_samples=None,
        crop_to_person=True,
    )
    indices = first_consecutive_clip(dataset, args.clip_length)
    items = [dataset[index] for index in indices]
    clip = torch.stack([item["image"] for item in items])
    reverse = clip.flip(0)
    tail_ablated = clip.clone()
    tail_ablated[-1].zero_()
    clips = torch.stack((clip, reverse, tail_ablated))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    encoder = FrozenTemporalVaceFeatureEncoder(
        checkpoint_dir=args.vace_checkpoint_dir,
        device=device,
        image_size=args.image_size,
        feature_layers=args.feature_layers,
        timestep=args.vace_timestep,
    )
    try:
        features = encoder.encode(clips)
    finally:
        encoder.close()
    expected_shape = (
        3,
        latent_frames,
        args.image_size // 16,
        args.image_size // 16,
        1536,
    )
    if tuple(features.shape) != expected_shape:
        raise RuntimeError(
            f"Expected continuous VACE features {expected_shape}, got "
            f"{tuple(features.shape)}"
        )
    if not bool(torch.isfinite(features).all()):
        raise FloatingPointError("Continuous VACE features are non-finite")

    true_features, reverse_features, tail_features = features
    reverse_aligned = reverse_features.flip(0)
    reversed_difference = scalar((true_features - reverse_aligned).abs().mean())
    tail_difference = [
        scalar((true_features[index] - tail_features[index]).abs().mean())
        for index in range(latent_frames)
    ]
    temporal_means = true_features.mean(dim=(1, 2))
    adjacent_difference = [
        scalar((temporal_means[right] - temporal_means[left]).abs().mean())
        for left, right in zip(range(latent_frames - 1), range(1, latent_frames))
    ]
    if reversed_difference <= 1.0e-6:
        raise RuntimeError("VACE features are unexpectedly insensitive to frame order")
    if max(tail_difference) <= 1.0e-7:
        raise RuntimeError("VACE features are unexpectedly insensitive to the final frame")

    source: list[dict[str, Any]] = []
    for index, item in zip(indices, items):
        sample = dataset.samples[index]
        source.append(
            {
                "sequence": sample.sequence,
                "person_id": sample.person_index,
                "frame_id": sample.frame_index,
                "source_frame_id": sample.source_frame_id,
                "image_path": sample.image_path,
                "bbox_xyxy": [float(value) for value in item["bbox_xyxy"]],
                "crop_applied": bool(item["crop_applied"]),
            }
        )
    result = {
        "status": "DIVIBODY_CONTINUOUS_VACE_SMOKE_OK",
        "split": args.split,
        "input_shape": list(clips.shape),
        "feature_shape": list(features.shape),
        "clip_length": args.clip_length,
        "latent_frames": latent_frames,
        "token_layout": "batch, latent_time, token_row, token_column, channel",
        "feature_layers": list(args.feature_layers),
        "vace_timestep": args.vace_timestep,
        "time_sensitivity": {
            "reverse_then_latent_reverse_mean_absolute_difference": reversed_difference,
            "last_rgb_frame_zeroed_mean_absolute_difference_by_latent": tail_difference,
            "real_clip_adjacent_latent_mean_absolute_difference": adjacent_difference,
        },
        "peak_cuda_memory_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 1024**3
        ),
        "peak_cuda_memory_reserved_gib": (
            torch.cuda.max_memory_reserved(device) / 1024**3
        ),
        "source": source,
        "runtime_seconds": time.monotonic() - started,
        "limitations": [
            "Oracle 3DPW person crops are used independently at each frame.",
            "This is a feature-contract smoke test, not a trained model metric.",
            "The final-frame ablation is deliberately out of distribution and only tests sensitivity.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
