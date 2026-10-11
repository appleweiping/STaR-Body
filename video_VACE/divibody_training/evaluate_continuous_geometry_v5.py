"""Evaluate a geometry-v5 checkpoint on a target-bearing continuous cache.

Predictions are stitched with the unique frame-owner mask before all metrics
are computed.  Acceleration therefore includes valid triplets that cross v5
window boundaries and never crosses a sequence, person, or frame gap.  This is
an additive evaluator; earlier evaluators and model files remain unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import cache_continuous_3dpw_vace_features_v5 as cache_planner
from cache_continuous_3dpw_vace_features import frame_identities_from_samples
from cache_continuous_3dpw_vace_features_v5 import (
    build_v5_window_plan,
    validate_v5_payload,
)
from evaluate_standard_3dpw import (
    Standard3DPWDataset,
    _normalise_gender,
    create_smpl_models,
)
from geometry_metrics_3dpw import (
    hip_centered_j14,
    mean_euclidean_error,
    pa_mpjpe_per_sample,
    pelvis_aligned_j14,
    sequence_acceleration_error,
)
from models.human_decoder_geometry import GeometryDecoderConfig
from models.human_decoder_geometry_video_v4 import (
    HumanGeometryVideoQueryTransformer,
)
from real_3dpw_dataset import _load_annotation, _to_numpy
from train_divibody_continuous_geometry_v5 import (
    FORMAT,
    TARGET_FRAMES,
    camera_targets,
    forward_features,
    gendered_vertices,
    load_payload,
    smpl_vertices,
)
from train_divibody_real_3dpw_temporal import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--data-root", type=Path,
        help="Required only for a feature-only formal test cache",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--smpl-model-root", type=Path, required=True)
    parser.add_argument("--joint-regressor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", choices=("bf16", "none"), default="bf16")
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def _mean(values: list[np.ndarray]) -> float:
    return float(np.concatenate(values).astype(np.float64).mean())


def attach_read_only_3dpw_targets(
    payload: dict[str, Any], data_root: Path
) -> dict[str, Any]:
    """Read formal-test GT at evaluation time without persisting it in cache."""

    split = str(payload["signature"].get("split"))
    dataset = Standard3DPWDataset(
        root=data_root, split=split, image_size=96, frame_stride=1,
        max_samples=None, crop_to_person=True,
    )
    target_stride = int(payload["signature"].get("target_stride", 8))
    original_stride = cache_planner.TARGET_STRIDE
    cache_planner.TARGET_STRIDE = target_stride
    try:
        plan = build_v5_window_plan(frame_identities_from_samples(dataset.samples))
    finally:
        cache_planner.TARGET_STRIDE = original_stride
    metadata = payload["metadata"]
    if len(plan.windows) != len(metadata) or any(
        window.uid != record["window_uid"]
        for window, record in zip(plan.windows, metadata)
    ):
        raise RuntimeError("Read-only GT plan does not match the frozen feature cache")
    names = ("full_pose", "betas", "gender_code", "cam_pose")
    loaded: dict[int, dict[str, torch.Tensor]] = {}
    rows: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    for window in plan.windows:
        values = []
        for frame in window.target_frames:
            if frame.dataset_index not in loaded:
                reference = dataset.samples[frame.dataset_index]
                annotation = _load_annotation(reference.annotation_path)
                pose = _to_numpy(
                    annotation["poses"][reference.person_index][reference.frame_index]
                ).reshape(-1)
                betas = _to_numpy(
                    annotation["betas"][reference.person_index]
                ).reshape(-1)
                cam_pose = _to_numpy(
                    annotation["cam_poses"][reference.frame_index]
                )
                gender = annotation["genders"][reference.person_index]
                if pose.size < 72 or betas.size < 10 or cam_pose.shape != (4, 4):
                    raise ValueError("Invalid read-only 3DPW target shape")
                loaded[frame.dataset_index] = {
                    "full_pose": torch.from_numpy(pose[:72].astype(np.float32, copy=True)),
                    "betas": torch.from_numpy(betas[:10].astype(np.float32, copy=True)),
                    "gender_code": torch.tensor(_normalise_gender(gender), dtype=torch.long),
                    "cam_pose": torch.from_numpy(cam_pose.astype(np.float32, copy=True)),
                }
            values.append(loaded[frame.dataset_index])
        for name in names:
            rows[name].append(torch.stack([value[name] for value in values]))
    attached = dict(payload)
    attached["targets"] = {name: torch.stack(values) for name, values in rows.items()}
    return attached


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.fps <= 0:
        raise ValueError("batch-size and fps must be positive")
    payload = load_payload(args.cache)
    feature_only = bool(payload.get("signature", {}).get("feature_only"))
    validate_v5_payload(payload, feature_only=feature_only)
    if feature_only:
        if args.data_root is None:
            raise ValueError("A feature-only cache requires --data-root for read-only GT")
        payload = attach_read_only_3dpw_targets(payload, args.data_root)
    elif args.data_root is not None:
        raise ValueError("--data-root is only valid with a feature-only cache")
    checkpoint = load_payload(args.checkpoint)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("Checkpoint is not continuous geometry-v5")
    device = torch.device(args.device)
    config = GeometryDecoderConfig(**checkpoint["decoder_config"])
    decoder = HumanGeometryVideoQueryTransformer(config).to(device)
    decoder.load_state_dict(checkpoint["decoder"], strict=True)
    decoder.eval()
    models = create_smpl_models(args.smpl_model_root, device)
    for model in models.values():
        model.requires_grad_(False)
    regressor_np = np.load(args.joint_regressor)
    if regressor_np.shape != (17, 6890) or not np.isfinite(regressor_np).all():
        raise ValueError("Invalid H36M joint regressor")
    regressor = torch.as_tensor(regressor_np, dtype=torch.float32, device=device)

    metric_values: dict[str, list[np.ndarray]] = {
        "mpjpe_mm": [], "pa_mpjpe_mm": [], "pve_mm": []
    }
    identities: list[tuple[str, int, int]] = []
    owner_window_ids: list[int] = []
    pred_motion: list[np.ndarray] = []
    gt_motion: list[np.ndarray] = []
    count = len(payload["metadata"])
    targets = payload["targets"]
    for start in range(0, count, args.batch_size):
        stop = min(start + args.batch_size, count)
        indices = slice(start, stop)
        batch: dict[str, Any] = {
            "features": payload["features"][indices],
            "full_pose": targets["full_pose"][indices],
            "betas": targets["betas"][indices],
            "genders": targets["gender_code"][indices],
            "cam_pose": targets["cam_pose"][indices],
        }
        output = forward_features(decoder, batch["features"], device, args.amp_dtype)
        pred_rot = output["rotmat"].float()
        target_rot = camera_targets(batch, device)
        target_betas = batch["betas"].to(device=device, dtype=torch.float32)
        genders = batch["genders"].to(device=device).reshape(-1)
        pred_betas = output["betas"].float()[:, None].expand(
            -1, TARGET_FRAMES, -1
        )
        with torch.autocast(device_type=device.type, enabled=False):
            pred_vertices = smpl_vertices(
                models[2], pred_rot.reshape(-1, 24, 3, 3),
                pred_betas.reshape(-1, 10),
            )
            gt_vertices = gendered_vertices(
                models, target_rot.reshape(-1, 24, 3, 3),
                target_betas.reshape(-1, 10), genders,
            )
            pred_joints, _ = pelvis_aligned_j14(pred_vertices, regressor)
            gt_joints, _ = pelvis_aligned_j14(gt_vertices, regressor)
            shape = (stop - start, TARGET_FRAMES)
            frame_metrics = {
                "mpjpe_mm": mean_euclidean_error(pred_joints, gt_joints).reshape(shape)
                * 1000.0,
                "pa_mpjpe_mm": pa_mpjpe_per_sample(pred_joints, gt_joints).reshape(shape)
                * 1000.0,
                "pve_mm": mean_euclidean_error(pred_vertices, gt_vertices).reshape(shape)
                * 1000.0,
            }
            pred_joints = hip_centered_j14(pred_joints).reshape(
                stop - start, TARGET_FRAMES, 14, 3
            )
            gt_joints = hip_centered_j14(gt_joints).reshape(
                stop - start, TARGET_FRAMES, 14, 3
            )
        owners = payload["frame_owner_mask"][indices]
        for name, values in frame_metrics.items():
            metric_values[name].append(values.cpu()[owners].numpy())
        pred_cpu, gt_cpu = pred_joints.cpu().numpy(), gt_joints.cpu().numpy()
        for row, record in enumerate(payload["metadata"][start:stop]):
            for local, owned in enumerate(owners[row].tolist()):
                if not owned:
                    continue
                identities.append(
                    (str(record["sequence"]), int(record["person_id"]),
                     int(record["target_frame_ids"][local]))
                )
                owner_window_ids.append(start + row)
                pred_motion.append(pred_cpu[row, local])
                gt_motion.append(gt_cpu[row, local])
        print(json.dumps({"phase": "geometry_v5_evaluation", "windows": stop,
                          "total_windows": count}), flush=True)

    if len(identities) != len(set(identities)):
        raise RuntimeError("Frame-owner masks produced duplicate identities")
    acceleration = sequence_acceleration_error(
        np.asarray(pred_motion), np.asarray(gt_motion),
        [value[0] for value in identities],
        [value[1] for value in identities],
        [value[2] for value in identities],
    )
    if not len(acceleration.errors):
        raise RuntimeError("No valid consecutive acceleration triplets")
    identity_to_owner = dict(zip(identities, owner_window_ids))
    within_window = []
    for center_index in acceleration.center_sample_indices.tolist():
        sequence, person_id, frame_id = identities[center_index]
        owners = {
            identity_to_owner[(sequence, person_id, frame_id + offset)]
            for offset in (-1, 0, 1)
        }
        within_window.append(len(owners) == 1)
    within_window_array = np.asarray(within_window, dtype=bool)
    boundary_array = ~within_window_array
    result = {
        "status": "DIVIBODY_CONTINUOUS_GEOMETRY_V5_EVALUATION_OK",
        "metrics": {
            **{name: _mean(values) for name, values in metric_values.items()},
            "acceleration_error_mm_per_frame2": float(
                acceleration.errors.astype(np.float64).mean() * 1000.0
            ),
            "acceleration_error_physical_m_per_s2": float(
                acceleration.errors.astype(np.float64).mean() * args.fps**2
            ),
            "acceleration_within_window_mm_per_frame2": float(
                acceleration.errors[within_window_array].mean() * 1000.0
            ),
            "acceleration_cross_window_mm_per_frame2": float(
                acceleration.errors[boundary_array].mean() * 1000.0
            ),
        },
        "coverage": {
            "owned_person_frames": len(identities),
            "unique_owned_person_frames": len(set(identities)),
            "acceleration_valid_triplets": int(len(acceleration.errors)),
            "acceleration_within_window_triplets": int(within_window_array.sum()),
            "acceleration_cross_window_triplets": int(boundary_array.sum()),
            "development_only": bool(payload["signature"].get("development_only")),
        },
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256_file(args.checkpoint),
            "epoch": int(checkpoint["epoch"]),
        },
        "cache": {"path": str(args.cache.resolve()), "sha256": sha256_file(args.cache)},
        "test_used_for_model_selection": False,
        "ground_truth_storage": (
            "read_only_3dpw_runtime_attachment_not_persisted"
            if feature_only else "target_bearing_non_test_cache"
        ),
        "acceleration_protocol": (
            "camera-coordinate hip-midpoint-rooted J14 second-difference error; "
            "same-person consecutive full-track triplets only"
        ),
    }
    if payload["signature"].get("split") == "test" and (
        len(identities) != 35_515 or len(acceleration.errors) != 34_752
    ):
        raise RuntimeError("Formal 3DPW test coverage does not match audited constants")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
