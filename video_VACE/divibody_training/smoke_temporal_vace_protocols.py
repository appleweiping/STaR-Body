"""Compare coherent and legacy diffusion states on one real T17 3DPW clip."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from evaluate_standard_3dpw import Standard3DPWDataset
from real_vace_temporal_feature_encoder_v2 import (
    ProtocolAwareTemporalVaceFeatureEncoder,
)
from smoke_temporal_vace_features import first_consecutive_clip
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--vace-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--text-embedding-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--noise-seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def difference(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    delta = (left.float() - right.float()).abs()
    return {
        "mean_absolute": float(delta.mean()),
        "root_mean_square": float(delta.square().mean().sqrt()),
        "maximum_absolute": float(delta.max()),
        "relative_l2": float(
            torch.linalg.vector_norm(delta)
            / torch.linalg.vector_norm(left.float()).clamp_min(1.0e-12)
        ),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite protocol smoke: {args.output}")
    if args.noise_seed < 0 or args.noise_seed >= 2**63:
        raise ValueError("noise-seed must lie in [0,2**63)")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Protocol smoke requires CUDA")
    started = time.monotonic()
    dataset = Standard3DPWDataset(
        root=args.data_root,
        split="validation",
        image_size=args.image_size,
        frame_stride=1,
        max_samples=None,
        crop_to_person=True,
    )
    indices = first_consecutive_clip(dataset, 17)
    items = [dataset[index] for index in indices]
    clip = torch.stack([item["image"] for item in items]).unsqueeze(0)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    encoder = ProtocolAwareTemporalVaceFeatureEncoder(
        checkpoint_dir=args.vace_checkpoint_dir,
        device=device,
        image_size=args.image_size,
        feature_layers=(7, 14, 22),
        timestep=0,
        diffusion_input_protocol="clean",
        text_embedding_path=args.text_embedding_path,
    )
    try:
        clean_t0 = encoder.encode(clip)

        # Reproduce the old shape-valid but distribution-incoherent protocol,
        # now with the official empty prompt so text is held constant.
        encoder.timestep = 500
        legacy_clean_t500 = encoder.encode(clip)

        encoder.diffusion_input_protocol = "flow_noised"
        encoder.scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=1,
            use_dynamic_shifting=False,
        )
        noised_t500 = encoder.encode(clip, noise_seeds=(args.noise_seed,))
    finally:
        encoder.close()

    expected = (1, 5, args.image_size // 16, args.image_size // 16, 1536)
    for name, value in (
        ("clean_t0", clean_t0),
        ("legacy_clean_t500", legacy_clean_t500),
        ("flow_noised_t500", noised_t500),
    ):
        if tuple(value.shape) != expected or not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"Invalid {name} features {tuple(value.shape)}")
    text_payload = torch.load(
        args.text_embedding_path, map_location="cpu", weights_only=False
    )
    text = text_payload["embedding"].float()
    if float(text.abs().mean()) <= 0:
        raise RuntimeError("Official empty-prompt embedding is unexpectedly zero")

    result = {
        "status": "DIVIBODY_TEMPORAL_VACE_PROTOCOL_SMOKE_OK",
        "feature_shape": list(expected),
        "source": {
            "sequence": dataset.samples[indices[0]].sequence,
            "person_id": dataset.samples[indices[0]].person_index,
            "frame_ids": [dataset.samples[index].frame_index for index in indices],
        },
        "official_empty_prompt": {
            "shape": list(text.shape),
            "mean_absolute": float(text.abs().mean()),
            "root_mean_square": float(text.square().mean().sqrt()),
            "path": str(args.text_embedding_path.expanduser().resolve()),
        },
        "comparisons": {
            "clean_t0_vs_legacy_clean_t500": difference(
                clean_t0, legacy_clean_t500
            ),
            "clean_t0_vs_flow_noised_t500": difference(clean_t0, noised_t500),
            "legacy_clean_t500_vs_flow_noised_t500": difference(
                legacy_clean_t500, noised_t500
            ),
        },
        "peak_cuda_memory_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 1024**3
        ),
        "peak_cuda_memory_reserved_gib": (
            torch.cuda.max_memory_reserved(device) / 1024**3
        ),
        "noise_seed": args.noise_seed,
        "runtime_seconds": time.monotonic() - started,
        "selection_warning": (
            "Feature distance does not select a regression protocol; final choice "
            "must use a fixed train/validation ablation without test feedback."
        ),
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
