"""Build resumable per-VACE-shard HMR 2.0a PoseTrack sidecars.

The original monolithic v1 builder remains unchanged.  This v2 collection
builder loads HMR once, processes each committed VACE shard independently, and
atomically records SHA-256 checksums for safe restart after interruption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from divibody_training.cache_posetrack18_hmr2a_anchor_v1 import (
    FORMAT,
    Images,
    load,
    sample_uid,
    validate,
)
from divibody_training.cache_posetrack18_vace_sharded_v2 import COLLECTION_FORMAT
from divibody_training.hmr2a_anchor import HMR2AAnchorExtractor, HMR2A_SHA256, sha256_file
from divibody_training.ram_ready_adapter_v1 import PoseTrack18PersonDataset


HMR_COLLECTION_FORMAT = "ram-posetrack18-hmr2a-sharded-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vace-collection-dir", type=Path, required=True)
    parser.add_argument("--ready-root", type=Path, required=True)
    parser.add_argument("--hmr-source-root", type=Path, required=True)
    parser.add_argument("--hmr2a-checkpoint", type=Path, required=True)
    parser.add_argument("--hmr-model-config", type=Path, required=True)
    parser.add_argument("--hmr-data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def canonical_hash(value: Any) -> str:
    value_bytes = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(value_bytes).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def verify(output_dir: Path, record: dict[str, Any]) -> None:
    path = output_dir / record["file"]
    if not path.is_file() or path.stat().st_size != int(record["bytes"]):
        raise RuntimeError(f"HMR shard missing or size changed: {path}")
    if sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"HMR shard checksum changed: {path}")


def frame_contract(vace: dict[str, Any]) -> tuple[list[str], torch.Tensor]:
    frame_uids: list[str] = []
    seen: set[str] = set()
    for record in vace["metadata"]:
        for uid, valid in zip(record["target_frame_uids"], record["target_valid"]):
            if valid and uid not in seen:
                seen.add(uid)
                frame_uids.append(uid)
    frame_index = {uid: index for index, uid in enumerate(frame_uids)}
    lookup = torch.tensor(
        [[frame_index[uid] for uid in record["target_frame_uids"]] for record in vace["metadata"]],
        dtype=torch.long,
    )
    return frame_uids, lookup


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Invalid batch/worker arguments")
    source_manifest_path = args.vace_collection_dir / "manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError("VACE collection must be COMPLETE before HMR extraction")
    source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source.get("status") != "COMPLETE" or source.get("collection_format") != COLLECTION_FORMAT:
        raise ValueError("Input is not a complete PoseTrack VACE sharded-v2 collection")
    split = source["configuration"]["split"]
    if split not in ("train", "val"):
        raise ValueError("PoseTrack HMR collection supports only train/val")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = args.output_dir / "manifest.partial.json"
    complete_path = args.output_dir / "manifest.json"
    if list(args.output_dir.glob("*.tmp")):
        raise RuntimeError("Remove incomplete temporary HMR files before resume")
    configuration = {
        "collection_format": HMR_COLLECTION_FORMAT,
        "split": split,
        "source_vace_collection_id": source["collection_id"],
        "source_vace_manifest_sha256": sha256_file(source_manifest_path),
        "hmr2a_checkpoint": str(args.hmr2a_checkpoint.resolve()),
        "hmr2a_checkpoint_sha256": HMR2A_SHA256,
        "ready_root": str(args.ready_root.resolve()),
        "shards": len(source["shards"]),
    }
    collection_id = canonical_hash(configuration)
    if complete_path.exists():
        manifest = json.loads(complete_path.read_text(encoding="utf-8"))
        if manifest.get("collection_id") != collection_id:
            raise RuntimeError("Existing complete HMR manifest belongs to another configuration")
        for record in manifest["shards"]:
            verify(args.output_dir, record)
        print(json.dumps({"status": "RAM_POSETRACK18_HMR2A_SHARDED_V2_ALREADY_COMPLETE",
                          "split": split, "shards": len(manifest["shards"])}, indent=2), flush=True)
        return

    if partial_path.exists():
        manifest = json.loads(partial_path.read_text(encoding="utf-8"))
        if manifest.get("collection_id") != collection_id:
            raise RuntimeError("Existing partial HMR manifest belongs to another configuration")
    else:
        if list(args.output_dir.glob("hmr2a_shard_*.pt")):
            raise RuntimeError("HMR shard files exist without a manifest")
        manifest = {"status": "IN_PROGRESS", "collection_format": HMR_COLLECTION_FORMAT,
                    "collection_id": collection_id, "configuration": configuration,
                    "total_shards": len(source["shards"]), "shards": []}
        atomic_json(partial_path, manifest)
    completed = {int(item["index"]): item for item in manifest.get("shards", [])}
    for item in completed.values():
        verify(args.output_dir, item)

    dataset = PoseTrack18PersonDataset(
        args.ready_root, split=split, image_size=256, bbox_scale=1.25, min_keypoints=1
    )
    dataset_lookup = {sample_uid(sample): index for index, sample in enumerate(dataset.samples)}
    started = time.monotonic()
    with HMR2AAnchorExtractor(
        source_root=args.hmr_source_root,
        checkpoint=args.hmr2a_checkpoint,
        model_config=args.hmr_model_config,
        hmr2_data_root=args.hmr_data_root,
        device=torch.device(args.device),
    ) as extractor:
        for vace_record in source["shards"]:
            shard_index = int(vace_record["index"])
            if shard_index in completed:
                continue
            vace_path = args.vace_collection_dir / vace_record["file"]
            if sha256_file(vace_path) != vace_record["sha256"]:
                raise RuntimeError(f"Source VACE shard checksum changed: {vace_path}")
            vace = load(vace_path)
            frame_uids, window_frame_indices = frame_contract(vace)
            missing = [uid for uid in frame_uids if uid not in dataset_lookup]
            if missing:
                raise ValueError(f"PoseTrack source images missing: {missing[:5]}")
            loader = DataLoader(
                Images(dataset, [dataset_lookup[uid] for uid in frame_uids]),
                batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                pin_memory=True, persistent_workers=args.num_workers > 0,
            )
            count = len(frame_uids)
            rgb = torch.empty(count, 192, 1280, dtype=torch.bfloat16)
            rot = torch.empty(count, 24, 3, 3, dtype=torch.float32)
            betas = torch.empty(count, 10, dtype=torch.float32)
            camera = torch.empty(count, 3, dtype=torch.float32)
            offset = 0
            for images in loader:
                output = extractor.encode(images)
                size = len(images)
                target = slice(offset, offset + size)
                rgb[target].copy_(output["rgb_tokens"].cpu().to(torch.bfloat16))
                rot[target].copy_(output["base_rotmat"].float().cpu())
                betas[target].copy_(output["base_betas"].float().cpu())
                camera[target].copy_(output["base_camera"].float().cpu())
                offset += size
                print(json.dumps({"phase": "posetrack18_hmr2a_sharded", "split": split,
                                  "shard": shard_index, "frames": offset, "total": count}), flush=True)
            payload = {
                "signature": {"format": FORMAT, "split": split,
                    "development_only": bool(vace["signature"].get("development_only")),
                    "sharded_collection": True, "collection_id": collection_id,
                    "shard_index": shard_index, "anchor_revision": "posetrack18-gt-crop-v1",
                    "unique_frames": count, "windows": len(vace["metadata"]),
                    "crop_protocol": "PoseTrack GT square bbox, scale=1.25; one HMR2 aspect crop",
                    "hmr2a_checkpoint": str(args.hmr2a_checkpoint.resolve()),
                    "hmr2a_checkpoint_sha256": HMR2A_SHA256,
                    "vace_cache": str(vace_path.resolve()),
                    "vace_cache_sha256": vace_record["sha256"],
                    "builder_source_sha256": sha256_file(Path(__file__))},
                "frame_uids": frame_uids,
                "window_uids": [record["window_uid"] for record in vace["metadata"]],
                "window_frame_indices": window_frame_indices,
                "rgb_tokens": rgb, "base_rotmat": rot,
                "base_betas": betas, "base_camera": camera,
            }
            validate(payload, vace)
            filename = f"hmr2a_shard_{shard_index:05d}.pt"
            output_path = args.output_dir / filename
            temporary = output_path.with_name(output_path.name + ".tmp")
            torch.save(payload, temporary)
            os.replace(temporary, output_path)
            record = {"index": shard_index, "file": filename,
                      "vace_file": vace_record["file"], "windows": len(vace["metadata"]),
                      "unique_frames": count, "bytes": output_path.stat().st_size,
                      "sha256": sha256_file(output_path)}
            completed[shard_index] = record
            manifest["shards"] = [completed[index] for index in sorted(completed)]
            manifest["completed_shards"] = len(completed)
            manifest["updated_unix"] = time.time()
            atomic_json(partial_path, manifest)
            print(json.dumps({"phase": "commit_hmr", **record,
                              "total_shards": len(source["shards"]),
                              "seconds": time.monotonic() - started}), flush=True)
            del vace, payload, rgb, rot, betas, camera

    if len(completed) != len(source["shards"]):
        raise RuntimeError("HMR extraction ended without completing every shard")
    manifest["status"] = "COMPLETE"
    manifest["completed_unix"] = time.time()
    atomic_json(complete_path, manifest)
    partial_path.unlink()
    print(json.dumps({"status": "RAM_POSETRACK18_HMR2A_SHARDED_V2_OK",
                      "split": split, "collection_id": collection_id,
                      "shards": len(completed), "manifest": str(complete_path.resolve()),
                      "seconds": time.monotonic() - started}, indent=2), flush=True)


if __name__ == "__main__":
    main()
