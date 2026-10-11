"""Evaluate stride-4 caches by fusing every overlapping prediction per frame.

Rotation matrices are averaged and projected back to SO(3) before SMPL. Shape
vectors are averaged arithmetically.  Fusion uses only model predictions and
frame identities; ground truth is never used for weighting or model selection.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from cache_continuous_3dpw_vace_features_v5 import (
    STRIDE4_FORMAT,
    validate_v5_payload,
)
from evaluate_continuous_geometry_v5 import attach_read_only_3dpw_targets
from evaluate_standard_3dpw import create_smpl_models
from geometry_metrics_3dpw import (
    hip_centered_j14,
    mean_euclidean_error,
    pa_mpjpe_per_sample,
    pelvis_aligned_j14,
    sequence_acceleration_error,
)
from models.human_decoder_geometry import GeometryDecoderConfig
from models.human_decoder_geometry_video_v4 import HumanGeometryVideoQueryTransformer
from train_divibody_continuous_geometry_v5 import (
    FORMAT as CHECKPOINT_FORMAT,
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
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--smpl-model-root", type=Path, required=True)
    parser.add_argument("--joint-regressor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", choices=("bf16", "none"), default="bf16")
    parser.add_argument(
        "--fusion-weighting", choices=("equal", "triangular"), default="triangular"
    )
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def project_so3(matrices: torch.Tensor) -> torch.Tensor:
    u_matrix, _, vh_matrix = torch.linalg.svd(matrices)
    product = u_matrix @ vh_matrix
    correction = torch.eye(3, device=matrices.device, dtype=matrices.dtype)
    correction = correction.expand(*matrices.shape[:-2], 3, 3).clone()
    correction[..., -1, -1] = torch.where(
        torch.linalg.det(product) < 0,
        -torch.ones((), device=matrices.device, dtype=matrices.dtype),
        torch.ones((), device=matrices.device, dtype=matrices.dtype),
    )
    return u_matrix @ correction @ vh_matrix


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.fps <= 0:
        raise ValueError("batch-size and fps must be positive")
    payload = load_payload(args.cache)
    if payload.get("signature", {}).get("format") != STRIDE4_FORMAT:
        raise ValueError("Overlap fusion requires a stride-4 v6 cache")
    feature_only = bool(payload["signature"].get("feature_only"))
    validate_v5_payload(payload, feature_only=feature_only)
    if feature_only:
        if args.data_root is None:
            raise ValueError("Feature-only cache requires --data-root")
        payload = attach_read_only_3dpw_targets(payload, args.data_root)

    checkpoint = load_payload(args.checkpoint)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Checkpoint is not continuous geometry-v5")
    device = torch.device(args.device)
    decoder = HumanGeometryVideoQueryTransformer(
        GeometryDecoderConfig(**checkpoint["decoder_config"])
    ).to(device)
    decoder.load_state_dict(checkpoint["decoder"], strict=True)
    decoder.eval()

    rotation_sum: dict[tuple[str, int, int], torch.Tensor] = {}
    beta_sum: dict[tuple[str, int, int], torch.Tensor] = {}
    weight_sum: Counter[tuple[str, int, int]] = Counter()
    counts: Counter[tuple[str, int, int]] = Counter()
    gt_rotation: dict[tuple[str, int, int], torch.Tensor] = {}
    gt_beta: dict[tuple[str, int, int], torch.Tensor] = {}
    gt_gender: dict[tuple[str, int, int], int] = {}
    targets = payload["targets"]
    window_count = len(payload["metadata"])
    for start in range(0, window_count, args.batch_size):
        stop = min(start + args.batch_size, window_count)
        selection = slice(start, stop)
        batch = {
            "features": payload["features"][selection],
            "full_pose": targets["full_pose"][selection],
            "betas": targets["betas"][selection],
            "genders": targets["gender_code"][selection],
            "cam_pose": targets["cam_pose"][selection],
        }
        output = forward_features(decoder, batch["features"], device, args.amp_dtype)
        predicted_rotation = output["rotmat"].float().cpu()
        predicted_beta = output["betas"].float().cpu()
        target_rotation = camera_targets(batch, device).cpu()
        target_beta = batch["betas"].float()
        target_gender = batch["genders"].long()
        valid_masks = payload["target_valid_mask"][selection]
        position_weights = (
            (1.0,) * TARGET_FRAMES
            if args.fusion_weighting == "equal"
            else (1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0)
        )
        for row, record in enumerate(payload["metadata"][start:stop]):
            for local, valid in enumerate(valid_masks[row].tolist()):
                if not valid:
                    continue
                key = (
                    str(record["sequence"]), int(record["person_id"]),
                    int(record["target_frame_ids"][local]),
                )
                if key not in rotation_sum:
                    rotation_sum[key] = torch.zeros_like(predicted_rotation[row, local])
                    beta_sum[key] = torch.zeros_like(predicted_beta[row])
                    gt_rotation[key] = target_rotation[row, local].clone()
                    gt_beta[key] = target_beta[row, local].clone()
                    gt_gender[key] = int(target_gender[row, local])
                elif not torch.allclose(
                    gt_rotation[key], target_rotation[row, local], atol=1e-5, rtol=1e-5
                ):
                    raise RuntimeError("Overlapping windows disagree on target rotation")
                weight = position_weights[local]
                rotation_sum[key] += predicted_rotation[row, local] * weight
                beta_sum[key] += predicted_beta[row] * weight
                weight_sum[key] += weight
                counts[key] += 1

    keys = sorted(rotation_sum)
    if len(keys) != int(payload["signature"]["window_plan"]["owned_person_frames"]):
        raise RuntimeError("Overlap fusion did not recover every audited person-frame")
    models = create_smpl_models(args.smpl_model_root, device)
    for model in models.values():
        model.requires_grad_(False)
    regressor_np = np.load(args.joint_regressor)
    if regressor_np.shape != (17, 6890) or not np.isfinite(regressor_np).all():
        raise ValueError("Invalid H36M joint regressor")
    regressor = torch.as_tensor(regressor_np, dtype=torch.float32, device=device)

    metric_chunks = {name: [] for name in ("mpjpe_mm", "pa_mpjpe_mm", "pve_mm")}
    pred_motion, target_motion = [], []
    for start in range(0, len(keys), args.batch_size):
        selected = keys[start : start + args.batch_size]
        average_rotation = torch.stack(
            [rotation_sum[key] / weight_sum[key] for key in selected]
        ).to(device)
        average_rotation = project_so3(average_rotation)
        average_beta = torch.stack(
            [beta_sum[key] / weight_sum[key] for key in selected]
        ).to(device)
        target_rot = torch.stack([gt_rotation[key] for key in selected]).to(device)
        target_betas = torch.stack([gt_beta[key] for key in selected]).to(device)
        genders = torch.tensor([gt_gender[key] for key in selected], device=device)
        pred_vertices = smpl_vertices(models[2], average_rotation, average_beta)
        target_vertices = gendered_vertices(
            models, target_rot, target_betas, genders
        )
        pred_joints, _ = pelvis_aligned_j14(pred_vertices, regressor)
        target_joints, _ = pelvis_aligned_j14(target_vertices, regressor)
        values = {
            "mpjpe_mm": mean_euclidean_error(pred_joints, target_joints) * 1000,
            "pa_mpjpe_mm": pa_mpjpe_per_sample(pred_joints, target_joints) * 1000,
            "pve_mm": mean_euclidean_error(pred_vertices, target_vertices) * 1000,
        }
        for name, value in values.items():
            metric_chunks[name].append(value.cpu().numpy())
        pred_motion.extend(hip_centered_j14(pred_joints).cpu().numpy())
        target_motion.extend(hip_centered_j14(target_joints).cpu().numpy())

    acceleration = sequence_acceleration_error(
        np.asarray(pred_motion), np.asarray(target_motion),
        [key[0] for key in keys], [key[1] for key in keys], [key[2] for key in keys],
    )
    if payload["signature"].get("split") == "test" and (
        len(keys) != 35_515 or len(acceleration.errors) != 34_752
    ):
        raise RuntimeError("Formal test fusion coverage differs from audit constants")
    result = {
        "status": "DIVIBODY_CONTINUOUS_GEOMETRY_V6_OVERLAP_EVALUATION_OK",
        "metrics": {
            **{
                name: float(np.concatenate(values).astype(np.float64).mean())
                for name, values in metric_chunks.items()
            },
            "acceleration_error_mm_per_frame2": float(acceleration.errors.mean() * 1000),
            "acceleration_error_physical_m_per_s2": float(
                acceleration.errors.mean() * args.fps**2
            ),
        },
        "coverage": {
            "unique_person_frames": len(keys),
            "acceleration_valid_triplets": int(len(acceleration.errors)),
            "prediction_occurrences": int(sum(counts.values())),
            "predictions_per_frame_min": int(min(counts.values())),
            "predictions_per_frame_mean": float(np.mean(list(counts.values()))),
            "predictions_per_frame_max": int(max(counts.values())),
        },
        "fusion": (
            f"{args.fusion_weighting}-weight rotation-matrix mean projected to "
            "SO(3); beta weighted mean"
        ),
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256_file(args.checkpoint),
            "epoch": int(checkpoint["epoch"]),
        },
        "cache": {"path": str(args.cache.resolve()), "sha256": sha256_file(args.cache)},
        "test_used_for_model_selection": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
