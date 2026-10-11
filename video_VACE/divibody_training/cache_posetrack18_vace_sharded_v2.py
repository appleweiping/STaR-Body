"""Build a resumable, bounded-memory PoseTrack18 VACE cache collection.

This is an additive replacement for the monolithic v1 cache builder.  It keeps
the exact v1 per-window payload contract, but commits independent atomic shard
files and a SHA-256 manifest so no final all-windows concatenation is needed.
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
from torch.utils.data import DataLoader, Subset

from divibody_training.cache_continuous_3dpw_vace_features import (
    checkpoint_tree_fingerprint,
    sha256_file,
)
from divibody_training.cache_continuous_3dpw_vace_features_v5 import (
    CONTEXT_FRAMES,
    FEATURE_WIDTH,
    IMAGE_SIZE,
    LATENT_FRAMES,
    TARGET_FRAMES,
    TOKEN_GRID,
    build_v5_window_plan,
    development_window_subset,
    stable_noise_seed,
    v5_metadata,
    validate_v5_window_plan,
)
from divibody_training.cache_posetrack18_vace_features_v1 import (
    FORMAT,
    PoseTrackWindowDataset,
    identities,
    validate_payload,
)
from divibody_training.ram_ready_adapter_v1 import PoseTrack18PersonDataset
from divibody_training.real_vace_temporal_feature_encoder import wan_latent_frames
from divibody_training.real_vace_temporal_feature_encoder_v2 import (
    ProtocolAwareTemporalVaceFeatureEncoder,
)


COLLECTION_FORMAT = "ram-posetrack18-contiguous-vace-sharded-v2"
MANIFEST_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--vace-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-windows", type=int, default=512)
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


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def shard_bounds(count: int, width: int) -> list[tuple[int, int]]:
    return [(start, min(start + width, count)) for start in range(0, count, width)]


def verify_shard(output_dir: Path, record: dict[str, Any]) -> None:
    path = output_dir / record["file"]
    if not path.is_file():
        raise FileNotFoundError(f"Completed shard is missing: {path}")
    if path.stat().st_size != int(record["bytes"]):
        raise RuntimeError(f"Completed shard size changed: {path}")
    if sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"Completed shard SHA-256 changed: {path}")


def masks_for(windows: list[Any], name: str) -> torch.Tensor:
    return torch.tensor([getattr(window, name) for window in windows], dtype=torch.bool)


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("v2 currently requires --batch-size 1 for exact shard boundaries")
    if args.shard_windows <= 0 or args.num_workers < 0 or args.development_max_windows < 0:
        raise ValueError("Invalid shard/worker/development arguments")
    if wan_latent_frames(CONTEXT_FRAMES) != LATENT_FRAMES:
        raise RuntimeError("Wan temporal geometry changed: expected 17 RGB -> 5 latents")

    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = args.output_dir / "manifest.partial.json"
    complete_path = args.output_dir / "manifest.json"
    unexpected_tmp = list(args.output_dir.glob("*.tmp"))
    if unexpected_tmp:
        raise RuntimeError(f"Remove incomplete temporary files before resume: {unexpected_tmp}")

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

    metadata = v5_metadata(plan)
    checkpoint = checkpoint_tree_fingerprint(args.vace_checkpoint_dir)
    configuration = {
        "collection_format": COLLECTION_FORMAT,
        "manifest_version": MANIFEST_VERSION,
        "split": args.split,
        "ready_root": str(args.ready_root.resolve()),
        "windows": len(plan.windows),
        "shard_windows": args.shard_windows,
        "development_only": bool(plan.stats.get("development_only")),
        "feature_layers": list(args.feature_layers),
        "vace_timestep": args.vace_timestep,
        "diffusion_input_protocol": args.diffusion_input_protocol,
        "noise_seed_base": args.noise_seed_base,
        "text_embedding_path": (
            str(args.text_embedding_path.resolve()) if args.text_embedding_path else None
        ),
        "checkpoint": checkpoint,
        "window_uids_sha256": canonical_hash([item["window_uid"] for item in metadata]),
    }
    collection_id = canonical_hash(configuration)
    bounds = shard_bounds(len(plan.windows), args.shard_windows)

    if complete_path.exists():
        manifest = json.loads(complete_path.read_text(encoding="utf-8"))
        if manifest.get("collection_id") != collection_id:
            raise RuntimeError("Existing complete manifest belongs to another configuration")
        if len(manifest.get("shards", [])) != len(bounds):
            raise RuntimeError("Complete manifest has the wrong shard count")
        for record in manifest["shards"]:
            verify_shard(args.output_dir, record)
        print(json.dumps({"status": "RAM_POSETRACK18_VACE_SHARDED_V2_ALREADY_COMPLETE",
                          "collection_id": collection_id, "shards": len(bounds),
                          "windows": len(plan.windows)}, indent=2), flush=True)
        return

    if partial_path.exists():
        manifest = json.loads(partial_path.read_text(encoding="utf-8"))
        if manifest.get("collection_id") != collection_id:
            raise RuntimeError("Existing partial manifest belongs to another configuration")
    else:
        existing_shards = list(args.output_dir.glob("vace_shard_*.pt"))
        if existing_shards:
            raise RuntimeError("Shard files exist without a manifest; refusing ambiguous resume")
        manifest = {
            "status": "IN_PROGRESS",
            "collection_format": COLLECTION_FORMAT,
            "manifest_version": MANIFEST_VERSION,
            "collection_id": collection_id,
            "configuration": configuration,
            "dataset_audit": dataset.audit(),
            "window_plan": plan.stats,
            "total_shards": len(bounds),
            "shards": [],
        }
        atomic_json(partial_path, manifest)

    completed = {int(record["index"]): record for record in manifest.get("shards", [])}
    for shard_index, record in completed.items():
        if shard_index >= len(bounds):
            raise RuntimeError("Partial manifest contains an out-of-range shard")
        start, end = bounds[shard_index]
        if [record.get("window_start"), record.get("window_end")] != [start, end]:
            raise RuntimeError("Partial manifest shard boundaries changed")
        verify_shard(args.output_dir, record)

    missing_indices = [
        index
        for shard_index, (start, end) in enumerate(bounds)
        if shard_index not in completed
        for index in range(start, end)
    ]
    encoder = ProtocolAwareTemporalVaceFeatureEncoder(
        checkpoint_dir=args.vace_checkpoint_dir,
        device=args.device,
        image_size=IMAGE_SIZE,
        feature_layers=args.feature_layers,
        timestep=args.vace_timestep,
        diffusion_input_protocol=args.diffusion_input_protocol,
        text_embedding_path=args.text_embedding_path,
    )
    encoder_signature = encoder.protocol_signature()
    loader = DataLoader(
        Subset(PoseTrackWindowDataset(dataset, plan), missing_indices),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    encoder_path = Path(__file__).with_name("real_vace_temporal_feature_encoder_v2.py")
    builder_path = Path(__file__)
    chunks: dict[str, list[torch.Tensor]] = {
        "features": [], "target_keypoints": [],
        "target_crop_center": [], "target_crop_side": [],
    }
    active_shard: int | None = None
    processed = 0

    def commit(shard_index: int) -> None:
        start, end = bounds[shard_index]
        windows = list(plan.windows[start:end])
        payload = {
            "signature": {
                "format": FORMAT,
                "split": args.split,
                "development_only": configuration["development_only"],
                "sharded_collection": True,
                "collection_format": COLLECTION_FORMAT,
                "collection_id": collection_id,
                "shard_index": shard_index,
                "window_start": start,
                "window_end": end,
                "global_windows": len(plan.windows),
                "ready_root": configuration["ready_root"],
                "dataset_audit": dataset.audit(),
                "window_plan": plan.stats,
                "context_frames_rgb": CONTEXT_FRAMES,
                "target_frames": TARGET_FRAMES,
                "target_stride": 8,
                "crop_protocol": "PoseTrack GT square bbox, scale=1.25, per-person per-frame",
                "feature_layers": list(args.feature_layers),
                "encoder_protocol": encoder_signature,
                "checkpoint": checkpoint,
                "builder_source_sha256": sha256_file(builder_path),
                "encoder_source_sha256": sha256_file(encoder_path),
            },
            "features": torch.cat(chunks["features"]),
            "context_edge_repeat_mask": masks_for(windows, "context_edge_repeat"),
            "target_valid_mask": masks_for(windows, "target_valid"),
            "frame_owner_mask": masks_for(windows, "frame_owner"),
            "velocity_owner_mask": masks_for(windows, "velocity_owner"),
            "acceleration_owner_mask": masks_for(windows, "acceleration_owner"),
            "target_keypoints": torch.cat(chunks["target_keypoints"]),
            "target_crop_center": torch.cat(chunks["target_crop_center"]),
            "target_crop_side": torch.cat(chunks["target_crop_side"]),
            "metadata": metadata[start:end],
        }
        validate_payload(payload)
        filename = f"vace_shard_{shard_index:05d}.pt"
        output = args.output_dir / filename
        temporary = output.with_name(output.name + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, output)
        record = {
            "index": shard_index,
            "file": filename,
            "window_start": start,
            "window_end": end,
            "windows": end - start,
            "owned_person_frames": int(payload["frame_owner_mask"].sum()),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        }
        completed[shard_index] = record
        manifest["shards"] = [completed[index] for index in sorted(completed)]
        manifest["completed_windows"] = sum(item["windows"] for item in manifest["shards"])
        manifest["updated_unix"] = time.time()
        atomic_json(partial_path, manifest)
        print(json.dumps({"phase": "commit", "split": args.split, **record,
                          "completed_shards": len(completed), "total_shards": len(bounds),
                          "seconds": time.monotonic() - started}), flush=True)
        for values in chunks.values():
            values.clear()

    try:
        for batch in loader:
            index = int(batch["window_index"].item())
            shard_index = index // args.shard_windows
            if active_shard is None:
                active_shard = shard_index
            elif shard_index != active_shard:
                commit(active_shard)
                active_shard = shard_index
            seed = stable_noise_seed(plan.windows[index].uid, args.noise_seed_base)
            noise = [seed] if args.diffusion_input_protocol == "flow_noised" else None
            clips = batch["images"].to(args.device, dtype=torch.float32, non_blocking=True)
            encoded = encoder.encode(clips, noise_seeds=noise)
            if tuple(encoded.shape[1:]) != (LATENT_FRAMES, *TOKEN_GRID, FEATURE_WIDTH):
                raise RuntimeError(f"Unexpected VACE feature shape {tuple(encoded.shape)}")
            chunks["features"].append(encoded.cpu().to(torch.bfloat16))
            chunks["target_keypoints"].append(batch["target_keypoints"].float())
            chunks["target_crop_center"].append(batch["target_crop_center"].float())
            chunks["target_crop_side"].append(batch["target_crop_side"].float())
            processed += 1
            if processed % args.log_interval == 0:
                print(json.dumps({"phase": "encode", "split": args.split,
                                  "new_windows": processed, "remaining": len(missing_indices) - processed,
                                  "global_window": index, "seconds": time.monotonic() - started}), flush=True)
        if active_shard is not None and chunks["features"]:
            commit(active_shard)
    finally:
        encoder.close()

    if len(completed) != len(bounds):
        raise RuntimeError("Encoding ended without completing every shard")
    manifest["status"] = "COMPLETE"
    manifest["completed_windows"] = len(plan.windows)
    manifest["completed_unix"] = time.time()
    atomic_json(complete_path, manifest)
    partial_path.unlink()
    print(json.dumps({"status": "RAM_POSETRACK18_VACE_SHARDED_V2_OK",
                      "collection_id": collection_id, "split": args.split,
                      "windows": len(plan.windows), "shards": len(bounds),
                      "newly_encoded_windows": processed,
                      "manifest": str(complete_path.resolve()),
                      "seconds": time.monotonic() - started}, indent=2), flush=True)


if __name__ == "__main__":
    main()
