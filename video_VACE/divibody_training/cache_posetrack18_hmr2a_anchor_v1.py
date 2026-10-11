"""Build an additive HMR 2.0a sidecar for a PoseTrack18 VACE cache."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from divibody_training.hmr2a_anchor import HMR2AAnchorExtractor, HMR2A_SHA256, sha256_file
from divibody_training.ram_ready_adapter_v1 import PoseTrack18PersonDataset


FORMAT = "ram-posetrack18-hmr2a-anchor-v1"
VACE_FORMAT = "ram-posetrack18-contiguous-vace-v1"
RGB_TOKENS = 192
RGB_WIDTH = 1280


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vace-cache", type=Path, required=True)
    parser.add_argument("--ready-root", type=Path, required=True)
    parser.add_argument("--hmr-source-root", type=Path, required=True)
    parser.add_argument("--hmr2a-checkpoint", type=Path, required=True)
    parser.add_argument("--hmr-model-config", type=Path, required=True)
    parser.add_argument("--hmr-data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def sample_uid(sample: dict[str, Any]) -> str:
    return f"{sample['video_id']}/person_{int(sample['track_id'])}/frame_{int(sample['frame_id']):05d}"


class Images(Dataset):
    def __init__(self, base: PoseTrack18PersonDataset, indices: list[int]) -> None:
        self.base = base
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.base[self.indices[index]]["image"]


def validate(payload: dict[str, Any], vace: dict[str, Any]) -> None:
    count = len(payload.get("frame_uids", []))
    windows = len(vace["metadata"])
    if payload.get("signature", {}).get("format") != FORMAT:
        raise ValueError("Unexpected PoseTrack HMR sidecar format")
    expected = {
        "rgb_tokens": (count, RGB_TOKENS, RGB_WIDTH),
        "base_rotmat": (count, 24, 3, 3),
        "base_betas": (count, 10),
        "base_camera": (count, 3),
    }
    for name, shape in expected.items():
        value = payload.get(name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"Invalid {name} shape; expected {shape}")
        if not bool(torch.isfinite(value.float()).all()):
            raise ValueError(f"{name} contains non-finite values")
    lookup = payload.get("window_frame_indices")
    if not isinstance(lookup, torch.Tensor) or tuple(lookup.shape) != (windows, 8):
        raise ValueError("window_frame_indices must be [windows,8]")
    for row, record in enumerate(vace["metadata"]):
        reconstructed = [payload["frame_uids"][i] for i in lookup[row].tolist()]
        if reconstructed != record["target_frame_uids"]:
            raise ValueError("HMR sidecar and VACE target identities disagree")


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an existing HMR sidecar")
    vace = load(args.vace_cache)
    signature = vace.get("signature", {})
    if signature.get("format") != VACE_FORMAT:
        raise ValueError("Input is not a PoseTrack18 VACE v1 cache")
    split = signature.get("split")
    if split not in ("train", "val"):
        raise ValueError("PoseTrack HMR cache supports only train/val")

    frame_uids: list[str] = []
    seen: set[str] = set()
    for record in vace["metadata"]:
        for uid, valid in zip(record["target_frame_uids"], record["target_valid"]):
            if valid and uid not in seen:
                seen.add(uid)
                frame_uids.append(uid)
    frame_index = {uid: index for index, uid in enumerate(frame_uids)}
    window_frame_indices = torch.tensor(
        [[frame_index[uid] for uid in record["target_frame_uids"]] for record in vace["metadata"]],
        dtype=torch.long,
    )

    dataset = PoseTrack18PersonDataset(
        args.ready_root,
        split=split,
        image_size=256,
        bbox_scale=1.25,
        min_keypoints=1,
    )
    dataset_lookup = {sample_uid(sample): index for index, sample in enumerate(dataset.samples)}
    missing = [uid for uid in frame_uids if uid not in dataset_lookup]
    if missing:
        raise ValueError(f"PoseTrack source images missing for identities: {missing[:5]}")
    loader = DataLoader(
        Images(dataset, [dataset_lookup[uid] for uid in frame_uids]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    count = len(frame_uids)
    rgb = torch.empty(count, RGB_TOKENS, RGB_WIDTH, dtype=torch.bfloat16)
    rot = torch.empty(count, 24, 3, 3, dtype=torch.float32)
    betas = torch.empty(count, 10, dtype=torch.float32)
    camera = torch.empty(count, 3, dtype=torch.float32)
    offset = 0
    with HMR2AAnchorExtractor(
        source_root=args.hmr_source_root,
        checkpoint=args.hmr2a_checkpoint,
        model_config=args.hmr_model_config,
        hmr2_data_root=args.hmr_data_root,
        device=torch.device(args.device),
    ) as extractor:
        for images in loader:
            output = extractor.encode(images)
            size = len(images)
            target = slice(offset, offset + size)
            rgb[target].copy_(output["rgb_tokens"].cpu().to(torch.bfloat16))
            rot[target].copy_(output["base_rotmat"].float().cpu())
            betas[target].copy_(output["base_betas"].float().cpu())
            camera[target].copy_(output["base_camera"].float().cpu())
            offset += size
            print(json.dumps({"phase": "posetrack18_hmr2a", "frames": offset, "total": count}), flush=True)
    payload = {
        "signature": {
            "format": FORMAT,
            "split": split,
            "development_only": bool(signature.get("development_only")),
            "anchor_revision": "posetrack18-gt-crop-v1",
            "unique_frames": count,
            "windows": len(vace["metadata"]),
            "crop_protocol": "PoseTrack GT square bbox, scale=1.25; one HMR2 aspect crop",
            "hmr2a_checkpoint": str(args.hmr2a_checkpoint.resolve()),
            "hmr2a_checkpoint_sha256": HMR2A_SHA256,
            "vace_cache": str(args.vace_cache.resolve()),
            "vace_cache_sha256": sha256_file(args.vace_cache),
            "builder_source_sha256": sha256_file(Path(__file__)),
        },
        "frame_uids": frame_uids,
        "window_uids": [record["window_uid"] for record in vace["metadata"]],
        "window_frame_indices": window_frame_indices,
        "rgb_tokens": rgb,
        "base_rotmat": rot,
        "base_betas": betas,
        "base_camera": camera,
    }
    validate(payload, vace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, args.output)
    print(json.dumps({"status": "RAM_POSETRACK18_HMR2A_V1_OK", "split": split,
                      "frames": count, "windows": len(vace["metadata"]),
                      "output": str(args.output.resolve()),
                      "output_bytes": args.output.stat().st_size}), flush=True)


if __name__ == "__main__":
    main()
