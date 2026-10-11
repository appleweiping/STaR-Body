"""Validation-only feature-reliance diagnostics for the frozen camera-v2 model.

This additive diagnostic never trains a model, modifies a checkpoint, or reads
the test split.  Its balanced validation subset is not an official benchmark.
The three conditions keep the same targets and original centered contexts:
real visual features, one fixed permutation over each complete person track,
and a per-spatial-token mean computed only from the training cache.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from evaluate_temporal_3dpw import (
    CenteredFrameClipDataset,
    compute_geometry_metrics,
    run_temporal_predictions,
    sha256_file,
    validate_cache_and_alignment,
    validate_feature_compatibility,
)
from models.human_decoder import HumanDecoderConfig, HumanQueryTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--smpl-model-root", "--smpl-path", type=Path, required=True)
    parser.add_argument("--joint-regressor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=768, help="0 means all validation frames")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--smpl-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hash-caches", action="store_true")
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError):
        return torch.load(path, map_location="cpu", weights_only=False)


def track_indices(metadata: list[dict[str, Any]]) -> dict[tuple[str, int], list[int]]:
    tracks: dict[tuple[str, int], list[int]] = defaultdict(list)
    identities: set[tuple[str, int, int]] = set()
    for index, record in enumerate(metadata):
        key = (str(record["sequence"]), int(record["person_id"]), int(record["frame_id"]))
        if key in identities:
            raise ValueError(f"Duplicate cached frame: {key}")
        identities.add(key)
        tracks[key[:2]].append(index)
    return {
        key: sorted(indices, key=lambda index: int(metadata[index]["frame_id"]))
        for key, indices in sorted(tracks.items())
    }


def choose_balanced_subset(metadata: list[dict[str, Any]], limit: int) -> list[int]:
    """Allocate equal track quotas, then sample uniformly along each track."""

    tracks = track_indices(metadata)
    if not tracks:
        raise ValueError("Validation cache is empty")
    count = len(metadata) if limit == 0 else min(limit, len(metadata))
    if count < len(tracks):
        raise ValueError("limit must permit at least one frame from each validation track")
    quotas = {key: 0 for key in tracks}
    remaining = count
    while remaining:
        for key, indices in tracks.items():
            if quotas[key] < len(indices):
                quotas[key] += 1
                remaining -= 1
                if not remaining:
                    break
    selected = []
    for key, indices in tracks.items():
        positions = np.linspace(0, len(indices) - 1, quotas[key], dtype=np.int64)
        selected.extend(indices[int(position)] for position in positions)
    selected.sort()
    if len(selected) != count or len(set(selected)) != count:
        raise RuntimeError("Balanced validation selection has duplicates or wrong length")
    return selected


def make_track_permutation(metadata: list[dict[str, Any]], seed: int) -> torch.Tensor:
    """Map each original index to a fixed source index in the same full track."""

    generator = np.random.default_rng(seed)
    permutation = np.arange(len(metadata), dtype=np.int64)
    for indices in track_indices(metadata).values():
        permutation[np.asarray(indices)] = generator.permutation(indices)
    if sorted(permutation.tolist()) != list(range(len(metadata))):
        raise RuntimeError("Track permutation is not one-to-one")
    return torch.from_numpy(permutation)


def training_mean_feature(payload: dict[str, Any], chunk_size: int = 512) -> torch.Tensor:
    features = payload["features"]
    if features.ndim != 3 or not len(features) or features.shape[-1] != 1536:
        raise ValueError(f"Invalid training feature array: {tuple(features.shape)}")
    total = torch.zeros(features.shape[1:], dtype=torch.float64)
    for start in range(0, len(features), chunk_size):
        chunk = features[start : start + chunk_size].float()
        if not bool(torch.isfinite(chunk).all()):
            raise ValueError("Training cache contains non-finite features")
        total += chunk.sum(dim=0, dtype=torch.float64)
    return (total / len(features)).float()


class DiagnosticSubset(Dataset):
    """Use original temporal contexts, with subset-local output indices."""

    def __init__(
        self,
        base: CenteredFrameClipDataset,
        selected: list[int],
        condition: str,
        permutation: torch.Tensor,
        mean_feature: torch.Tensor,
    ) -> None:
        if condition not in ("real", "within_track_shuffle", "training_mean"):
            raise ValueError(f"Unknown feature condition: {condition}")
        self.base = base
        self.selected = selected
        self.condition = condition
        self.permutation = permutation
        self.mean_feature = mean_feature
        self.request_by_target = {request[1]: request for request in base.requests}

    def __len__(self) -> int:
        return len(self.selected)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        frame_indices, _, local_position = self.request_by_target[self.selected[index]]
        if self.condition == "training_mean":
            features = self.mean_feature.unsqueeze(0).expand(len(frame_indices), -1, -1)
        else:
            source_indices = (
                self.permutation.index_select(0, frame_indices)
                if self.condition == "within_track_shuffle"
                else frame_indices
            )
            features = self.base.features.index_select(0, source_indices)
        return {
            "features": features,
            "target_index": torch.tensor(index, dtype=torch.long),
            "local_position": torch.tensor(local_position, dtype=torch.long),
        }


def select_geometry_payload(payload: dict[str, Any], selected: list[int]) -> dict[str, Any]:
    indices = torch.as_tensor(selected, dtype=torch.long)
    return {
        "metadata": [payload["metadata"][index] for index in selected],
        "targets": {
            name: torch.as_tensor(payload["targets"][name]).index_select(0, indices)
            for name in ("full_pose", "gender_code", "betas")
        },
    }


def pose_variability(
    prediction: np.ndarray, target: np.ndarray, metadata: list[dict[str, Any]]
) -> dict[str, Any]:
    """Coordinate std over frames, not std across the joints of one skeleton."""

    pred_std = prediction.std(axis=0, ddof=0) * 1000.0
    target_std = target.std(axis=0, ddof=0) * 1000.0
    track_values = []
    pred_weighted = 0.0
    gt_weighted = 0.0
    for (sequence, person_id), indices in track_indices(metadata).items():
        pred_track_std = float(prediction[indices].std(axis=0, ddof=0).mean() * 1000.0)
        gt_track_std = float(target[indices].std(axis=0, ddof=0).mean() * 1000.0)
        pred_weighted += pred_track_std * len(indices)
        gt_weighted += gt_track_std * len(indices)
        track_values.append(
            {
                "sequence": sequence,
                "person_id": person_id,
                "frames": len(indices),
                "pred_mean_coordinate_std_mm": pred_track_std,
                "gt_mean_coordinate_std_mm": gt_track_std,
                "pred_to_gt_std_ratio": pred_track_std / gt_track_std if gt_track_std else None,
            }
        )
    return {
        "coordinate_system": "camera, hip-midpoint-rooted J14",
        "formula": "mean over 14 joints and xyz of std over selected frames (ddof=0)",
        "pred_mean_coordinate_std_mm": float(pred_std.mean()),
        "gt_mean_coordinate_std_mm": float(target_std.mean()),
        "pred_to_gt_std_ratio": float(pred_std.mean() / target_std.mean()) if target_std.mean() else None,
        "pred_coordinate_std_mm_j14_xyz": pred_std.tolist(),
        "gt_coordinate_std_mm_j14_xyz": target_std.tolist(),
        "weighted_within_track_pred_std_mm": pred_weighted / len(metadata),
        "weighted_within_track_gt_std_mm": gt_weighted / len(metadata),
        "weighted_within_track_std_ratio": pred_weighted / gt_weighted if gt_weighted else None,
        "per_track": track_values,
        "interpretation": "Low variability can indicate mean-pose collapse; this is not acceleration or motion accuracy.",
    }


def spatial_permutation_check(
    decoder: HumanQueryTransformer,
    dataset: DiagnosticSubset,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    features = torch.stack([dataset[index]["features"] for index in range(min(2, len(dataset)))])
    features = features.to(device=device, dtype=torch.float32)
    batch_size, frames, tokens, channels = features.shape
    side = math.isqrt(tokens)
    if side * side != tokens:
        raise ValueError("Spatial token count is not a square")
    generator = torch.Generator(device="cpu").manual_seed(seed + 1009)
    permutation = torch.randperm(tokens, generator=generator).to(device)
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            original = decoder(
                features.reshape(batch_size, frames * tokens, channels),
                grid_size=(frames, side, side),
                target_frames=frames,
            )
            shuffled = decoder(
                features.index_select(2, permutation).reshape(batch_size, frames * tokens, channels),
                grid_size=(frames, side, side),
                target_frames=frames,
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_tf32
    differences = {
        name: float((original[name].float() - shuffled[name].float()).abs().max())
        for name in original
    }
    maximum = max(differences.values())
    return {
        "batch_clips": batch_size,
        "dtype": "float32, autocast disabled, CUDA matmul TF32 disabled",
        "permutation": permutation.cpu().tolist(),
        "max_abs_output_difference": maximum,
        "per_output_max_abs_difference": differences,
        "numerically_invariant_at_1e_minus4": maximum <= 1.0e-4,
        "interpretation": "This isolates the decoder's missing explicit x/y indexing. The frozen VACE features can still contain upstream positional information.",
    }


def main() -> None:
    args = parse_args()
    if args.output.suffix.lower() != ".json" or args.output.exists():
        raise ValueError("output must be a new .json path; existing reports are not overwritten")
    if args.limit < 0 or args.batch_size <= 0 or args.smpl_batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Invalid sample limit, batch size, or worker count")
    started = time.monotonic()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    checkpoint = load_payload(args.checkpoint)
    if checkpoint.get("format") != "divibody-real-3dpw-temporal-camera-v2":
        raise ValueError("This diagnostic only accepts the frozen camera-v2 checkpoint")
    if checkpoint.get("target_coordinate_system") != "camera" or checkpoint.get("camera_transform") != "R_camera_from_world @ R_global_world":
        raise ValueError("Checkpoint camera-frame protocol is missing or inconsistent")
    config = HumanDecoderConfig(**checkpoint["decoder_config"])
    if config.body_pose_dim != 63 or config.input_dim != 1536:
        raise ValueError("Expected the original 63D camera-v2 decoder")
    decoder = HumanQueryTransformer(config).to(device).eval().requires_grad_(False)
    decoder.load_state_dict(checkpoint["decoder"], strict=True)

    payload = load_payload(args.val_cache)
    signature = payload.get("signature", {})
    if signature.get("split") != "validation":
        raise ValueError("Only the validation split is permitted; test is never read")
    data_root = args.data_root or Path(signature["dataset_root"])
    validate_feature_compatibility(checkpoint, payload)
    audit = validate_cache_and_alignment(payload, data_root, "validation")
    train_payload = load_payload(args.train_cache)
    train_signature = train_payload.get("signature", {})
    if train_signature.get("split") != "train" or train_signature.get("format") != "divibody-frozen-vace-features-v1":
        raise ValueError("Mean features must come exclusively from the training split")
    validate_feature_compatibility(checkpoint, train_payload)
    if len(train_payload.get("metadata", [])) != len(train_payload["features"]):
        raise ValueError("Training feature and metadata lengths differ")
    if train_signature.get("dataset_size") != len(train_payload["features"]):
        raise ValueError("Training feature count differs from the cache signature")
    train_sequences = {str(item["sequence"]) for item in train_payload["metadata"]}
    val_sequences = {str(item["sequence"]) for item in payload["metadata"]}
    if train_sequences & val_sequences:
        raise ValueError("Training and validation cache sequences overlap")
    mean_feature = training_mean_feature(train_payload)
    if tuple(mean_feature.shape) != tuple(payload["features"].shape[1:]):
        raise ValueError("Train mean and validation feature spatial grids differ")
    training_count = len(train_payload["features"])
    del train_payload
    gc.collect()

    base = CenteredFrameClipDataset(payload, config.max_latent_frames)
    selected = choose_balanced_subset(payload["metadata"], args.limit)
    permutation = make_track_permutation(payload["metadata"], args.seed)
    geometry_payload = select_geometry_payload(payload, selected)
    identity_keys = [
        [str(item["sequence"]), int(item["person_id"]), int(item["frame_id"])]
        for item in geometry_payload["metadata"]
    ]
    conditions = {}
    true_joints = None
    true_gt_joints = None
    permutation_check = None
    for condition in ("real", "within_track_shuffle", "training_mean"):
        print(json.dumps({"phase": "feature_use_diagnostic", "condition": condition, "frames": len(selected)}), flush=True)
        dataset = DiagnosticSubset(base, selected, condition, permutation, mean_feature)
        if condition == "real":
            permutation_check = spatial_permutation_check(decoder, dataset, device, args.seed)
        pred_pose, pred_betas = run_temporal_predictions(
            decoder, dataset, args.batch_size, args.num_workers, device
        )
        metric_arrays, pred_joints, gt_joints = compute_geometry_metrics(
            pred_pose,
            pred_betas,
            geometry_payload,
            args.smpl_model_root,
            args.joint_regressor,
            args.smpl_batch_size,
            device,
            "camera",
        )
        conditions[condition] = {
            "metrics": {name: float(values.mean()) for name, values in metric_arrays.items()},
            "pose_variability": pose_variability(pred_joints, gt_joints, geometry_payload["metadata"]),
        }
        if condition == "real":
            true_joints = pred_joints.copy()
            true_gt_joints = gt_joints.copy()
        else:
            if not np.array_equal(gt_joints, true_gt_joints):
                raise RuntimeError("Ground-truth geometry changed between diagnostic conditions")
            conditions[condition]["prediction_change_from_real_mpjpe_mm"] = float(
                np.linalg.norm(pred_joints - true_joints, axis=-1).mean() * 1000.0
            )
            conditions[condition]["metric_change_from_real"] = {
                name: value - conditions["real"]["metrics"][name]
                for name, value in conditions[condition]["metrics"].items()
            }

    report = {
        "status": "DIVIBODY_VALIDATION_FEATURE_USE_DIAGNOSTIC_OK",
        "split": "validation",
        "coverage": "balanced_diagnostic_subset_not_an_official_benchmark",
        "test_read_or_used_for_selection": False,
        "model_training_or_checkpoint_modification": False,
        "seed": args.seed,
        "selected_frames": len(selected),
        "total_validation_frames": len(payload["metadata"]),
        "selection": "Equal track quotas (capacity constrained), uniformly spaced frame indices per track",
        "selected_identity_keys": identity_keys,
        "selected_identity_keys_sha256": hashlib.sha256(json.dumps(identity_keys, separators=(",", ":")).encode()).hexdigest(),
        "conditions": conditions,
        "spatial_token_permutation": permutation_check,
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256_file(args.checkpoint),
            "epoch": checkpoint["epoch"],
            "format": checkpoint["format"],
        },
        "validation_cache": {
            "path": str(args.val_cache.resolve()),
            "sha256": sha256_file(args.val_cache) if args.hash_caches else None,
            "signature": signature,
        },
        "training_mean": {
            "cache_path": str(args.train_cache.resolve()),
            "cache_sha256": sha256_file(args.train_cache) if args.hash_caches else None,
            "signature": train_signature,
            "frames": training_count,
            "definition": "Per-spatial-token/channel arithmetic mean of training frame features, repeated across time",
        },
        "shuffle": {
            "definition": "One seed-fixed bijection over each complete validation person track, applied globally before building any diagnostic clip",
            "unchanged_feature_indices": int((permutation == torch.arange(len(permutation))).sum()),
            "mapping_sha256": hashlib.sha256(permutation.numpy().tobytes()).hexdigest(),
            "context_and_targets": "Original centered context identities and target labels never change; only the source visual feature index changes",
        },
        "dataset_preflight": audit,
        "limitations": [
            "Not a complete 3DPW test benchmark and not a model-selection criterion for test.",
            "The validation targets are used only to measure a fixed checkpoint; no fitting or parameter update occurs.",
            "Track shuffling destroys motion order intentionally while keeping scene/person identity; it is not a temporal accuracy evaluation.",
            "An error increase under shuffling shows feature reliance, not that the frozen representation is sufficiently strong.",
            "Numerical decoder permutation invariance does not imply VACE tokens contain no upstream positional information.",
            "Camera-v2 and original metric protocol are unchanged; two final SMPL end-joint rotations remain zero.",
        ],
        "runtime_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "selected_frames": len(selected), "metrics": {name: value["metrics"] for name, value in conditions.items()}, "spatial_token_max_difference": permutation_check["max_abs_output_difference"], "output": str(args.output.resolve())}, indent=2), flush=True)


if __name__ == "__main__":
    main()
