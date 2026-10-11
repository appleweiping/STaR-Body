"""Cache unique-frame HMR 2.0a RGB tokens and SMPL anchors for DiviBody v8.

This additive sidecar deliberately keeps the existing VACE cache untouched.
Window targets/masks/features remain in that cache; this file stores each HMR
frame once plus an exact window-to-frame lookup table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from .evaluate_unified_hmr2_divibody_3dpw import UnifiedHMR3DPWDataset
    from .hmr2a_anchor import HMR2AAnchorExtractor, HMR2A_SHA256, sha256_file
except ImportError:
    from evaluate_unified_hmr2_divibody_3dpw import UnifiedHMR3DPWDataset
    from hmr2a_anchor import HMR2AAnchorExtractor, HMR2A_SHA256, sha256_file


FORMAT = "divibody-hmr2a-anchor-sidecar-v8"
TARGET_FRAMES = 8
RGB_TOKENS = 192
RGB_WIDTH = 1280
VACE_FORMATS = {
    "divibody-contiguous-vace-video-features-v5",
    "divibody-contiguous-vace-video-features-v6-stride4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vace-cache", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--hmr-source-root", type=Path, required=True)
    parser.add_argument("--hmr2a-checkpoint", type=Path, required=True)
    parser.add_argument("--hmr-model-config", type=Path, required=True)
    parser.add_argument("--hmr-data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--skip-vace-sha256", action="store_true")
    return parser.parse_args()


def uid(sequence: str, person_id: int, frame_id: int) -> str:
    return f"{sequence}/person_{int(person_id)}/frame_{int(frame_id):05d}"


@dataclass(frozen=True)
class AnchorPlan:
    split: str
    frame_uids: list[str]
    window_frame_indices: torch.Tensor
    window_uids: list[str]
    development_only: bool


def build_anchor_plan(
    metadata: list[dict[str, Any]], split: str, max_windows: int = 0
) -> AnchorPlan:
    if split == "test":
        raise ValueError("HMR2a v8 training sidecars must never use the test split")
    if split not in ("train", "validation"):
        raise ValueError(f"Unsupported training split: {split!r}")
    if max_windows < 0:
        raise ValueError("max_windows must be non-negative")
    selected = metadata[:max_windows] if max_windows else metadata
    if not selected:
        raise ValueError("No VACE windows selected")

    frame_uids: list[str] = []
    frame_index: dict[str, int] = {}
    rows: list[list[int]] = []
    window_uids: list[str] = []
    for record in selected:
        values = list(record.get("target_frame_uids", ()))
        if len(values) != TARGET_FRAMES:
            raise ValueError("Every VACE window must identify exactly eight targets")
        row = []
        for value in values:
            if value not in frame_index:
                frame_index[value] = len(frame_uids)
                frame_uids.append(value)
            row.append(frame_index[value])
        rows.append(row)
        window_uids.append(str(record["window_uid"]))
    return AnchorPlan(
        split=split,
        frame_uids=frame_uids,
        window_frame_indices=torch.tensor(rows, dtype=torch.long),
        window_uids=window_uids,
        development_only=bool(max_windows and max_windows < len(metadata)),
    )


def semantic_vace_digest(payload: dict[str, Any], count: int) -> str:
    records = payload["metadata"][:count]
    value = {
        "signature": payload["signature"],
        "window_uids": [record["window_uid"] for record in records],
        "target_frame_uids": [record["target_frame_uids"] for record in records],
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_vace_source(payload: dict[str, Any]) -> None:
    """Validate only the immutable contract consumed by this sidecar.

    The legacy v5 module cannot be package-imported because it intentionally
    uses direct-script imports, so v8 owns this small independent boundary.
    """

    signature = payload.get("signature", {})
    if signature.get("format") not in VACE_FORMATS:
        raise ValueError("Unsupported continuous VACE cache format")
    if signature.get("split") not in ("train", "validation"):
        raise ValueError("VACE source must be a train or validation cache")
    if bool(signature.get("feature_only", False)):
        raise ValueError("v8 training requires VACE targets")
    metadata = payload.get("metadata")
    features = payload.get("features")
    if not isinstance(metadata, list) or not metadata:
        raise ValueError("VACE metadata is empty")
    if not isinstance(features, torch.Tensor) or tuple(features.shape[1:]) != (
        5, 6, 6, 1536
    ) or len(features) != len(metadata):
        raise ValueError("Invalid continuous VACE features")
    if features.dtype != torch.bfloat16:
        raise ValueError("Continuous VACE features must be BF16")
    targets = payload.get("targets", {})
    target_shapes = {
        "full_pose": (len(metadata), 8, 72),
        "betas": (len(metadata), 8, 10),
        "gender_code": (len(metadata), 8),
        "cam_pose": (len(metadata), 8, 4, 4),
    }
    for name, shape in target_shapes.items():
        value = targets.get(name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"Invalid VACE target {name}")
    for name, width in (
        ("target_valid_mask", 8),
        ("frame_owner_mask", 8),
        ("velocity_owner_mask", 7),
        ("acceleration_owner_mask", 6),
    ):
        value = payload.get(name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (
            len(metadata), width
        ) or value.dtype != torch.bool:
            raise ValueError(f"Invalid VACE mask {name}")


def _finite_in_chunks(value: torch.Tensor, chunk: int = 256) -> bool:
    return all(bool(torch.isfinite(value[start : start + chunk]).all())
               for start in range(0, len(value), chunk))


def validate_sidecar(
    payload: dict[str, Any], vace_payload: dict[str, Any] | None = None
) -> None:
    signature = payload.get("signature", {})
    if signature.get("format") != FORMAT:
        raise ValueError("Not a DiviBody HMR2a v8 sidecar")
    if signature.get("split") == "test":
        raise ValueError("Test leakage in HMR2a v8 sidecar")
    if signature.get("hmr2a_checkpoint_sha256") != HMR2A_SHA256:
        raise ValueError("Wrong HMR 2.0a checkpoint provenance")
    frame_uids = payload.get("frame_uids")
    if not isinstance(frame_uids, list) or len(frame_uids) != len(set(frame_uids)):
        raise ValueError("frame_uids must be a unique list")
    count = len(frame_uids)
    shapes = {
        "rgb_tokens": (count, RGB_TOKENS, RGB_WIDTH),
        "base_rotmat": (count, 24, 3, 3),
        "base_betas": (count, 10),
        "base_camera": (count, 3),
    }
    for name, shape in shapes.items():
        value = payload.get(name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if not value.is_floating_point() or not _finite_in_chunks(value):
            raise ValueError(f"{name} must be finite floating point")
    if payload["rgb_tokens"].dtype != torch.bfloat16:
        raise ValueError("RGB tokens must be stored as BF16")
    lookup = payload.get("window_frame_indices")
    window_uids = payload.get("window_uids")
    if not isinstance(lookup, torch.Tensor) or lookup.dtype != torch.long:
        raise ValueError("window_frame_indices must be a LongTensor")
    if tuple(lookup.shape) != (len(window_uids), TARGET_FRAMES):
        raise ValueError("Invalid window-to-frame lookup shape")
    if int(lookup.min()) < 0 or int(lookup.max()) >= count:
        raise ValueError("Window-to-frame lookup is out of range")

    if vace_payload is not None:
        metadata = vace_payload["metadata"][: len(window_uids)]
        if signature["split"] != vace_payload["signature"]["split"]:
            raise ValueError("HMR sidecar and VACE cache splits differ")
        if window_uids != [record["window_uid"] for record in metadata]:
            raise ValueError("HMR sidecar and VACE window order differ")
        expected = [record["target_frame_uids"] for record in metadata]
        actual = [[frame_uids[index] for index in row] for row in lookup.tolist()]
        if actual != expected:
            raise ValueError("HMR sidecar and VACE frame identities differ")
        digest = semantic_vace_digest(vace_payload, len(window_uids))
        if signature["vace_semantic_sha256"] != digest:
            raise ValueError("VACE semantic provenance mismatch")


class _ImageSubset(Dataset):
    def __init__(self, dataset: UnifiedHMR3DPWDataset, indices: Iterable[int]):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.dataset[self.indices[index]]["image"]


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Invalid batch size or worker count")
    vace_cache = args.vace_cache.expanduser().resolve()
    vace = torch.load(vace_cache, map_location="cpu", weights_only=False, mmap=True)
    signature = vace.get("signature", {})
    validate_vace_source(vace)
    split = str(signature.get("split"))
    plan = build_anchor_plan(vace["metadata"], split, args.max_windows)

    dataset = UnifiedHMR3DPWDataset(
        root=args.data_root,
        split=split,
        image_size=256,
        frame_stride=1,
        max_samples=None,
        crop_to_person=True,
        crop_margin=0.25,
    )
    dataset_lookup = {
        uid(sample.sequence, sample.person_index, sample.frame_index): index
        for index, sample in enumerate(dataset.samples)
    }
    missing = [value for value in plan.frame_uids if value not in dataset_lookup]
    if missing:
        raise ValueError(f"VACE identities missing from strict HMR dataset: {missing[:5]}")
    indices = [dataset_lookup[value] for value in plan.frame_uids]
    loader = DataLoader(
        _ImageSubset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    count = len(plan.frame_uids)
    rgb_tokens = torch.empty((count, RGB_TOKENS, RGB_WIDTH), dtype=torch.bfloat16)
    base_rotmat = torch.empty((count, 24, 3, 3), dtype=torch.float32)
    base_betas = torch.empty((count, 10), dtype=torch.float32)
    base_camera = torch.empty((count, 3), dtype=torch.float32)
    offset = 0
    with HMR2AAnchorExtractor(
        source_root=args.hmr_source_root,
        checkpoint=args.hmr2a_checkpoint,
        model_config=args.hmr_model_config,
        hmr2_data_root=args.hmr_data_root,
        device=torch.device(args.device),
    ) as extractor:
        for images in loader:
            encoded = extractor.encode(images)
            size = len(images)
            target = slice(offset, offset + size)
            rgb_tokens[target].copy_(encoded["rgb_tokens"].to("cpu", torch.bfloat16))
            base_rotmat[target].copy_(encoded["base_rotmat"].float().cpu())
            base_betas[target].copy_(encoded["base_betas"].float().cpu())
            base_camera[target].copy_(encoded["base_camera"].float().cpu())
            offset += size
            print(f"HMR2a frames {offset}/{count}", flush=True)
    if offset != count:
        raise RuntimeError(f"Incomplete HMR extraction: {offset}/{count}")

    vace_stat = vace_cache.stat()
    window_count = len(plan.window_uids)
    output = {
        "signature": {
            "format": FORMAT,
            "split": split,
            "development_only": plan.development_only,
            "unique_frames": count,
            "windows": window_count,
            "hmr2a_checkpoint": str(args.hmr2a_checkpoint.resolve()),
            "hmr2a_checkpoint_sha256": HMR2A_SHA256,
            "vace_cache": str(vace_cache),
            "vace_cache_bytes": vace_stat.st_size,
            "vace_cache_sha256": None if args.skip_vace_sha256 else sha256_file(vace_cache),
            "vace_semantic_sha256": semantic_vace_digest(vace, window_count),
            "crop_protocol": (
                "oracle 3DPW poses2d bbox + 25% margin; expand to 192:256 HMR2 "
                "aspect; square 256 pre-crop; model centre-width crop to 192"
            ),
            "rgb_token_dtype": "bfloat16",
            "anchor_dtype": "float32",
        },
        "frame_uids": plan.frame_uids,
        "window_uids": plan.window_uids,
        "window_frame_indices": plan.window_frame_indices,
        "rgb_tokens": rgb_tokens,
        "base_rotmat": base_rotmat,
        "base_betas": base_betas,
        "base_camera": base_camera,
    }
    validate_sidecar(output, vace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(output, temporary)
    os.replace(temporary, args.output)
    print(json.dumps({
        "status": "DIVIBODY_HMR2A_ANCHOR_SIDECAR_V8_OK",
        "split": split,
        "windows": window_count,
        "unique_frames": count,
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
