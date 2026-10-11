"""Train the additive T17->center8 continuous-VACE geometry-v5 decoder.

This file is derived from ``train_divibody_real_3dpw_geometry.py``; that v3
trainer and every earlier model remain unchanged.  Frozen v5 caches contain
five genuine Wan temporal slices.  Eight explicit RGB queries predict context
positions 4..11, with owner masks preventing duplicated tail supervision.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from cache_continuous_3dpw_vace_features_v5 import (
    FORMAT as CACHE_FORMAT,
    STRIDE4_FORMAT,
    LATENT_FRAMES,
    TARGET_FRAMES,
    TOKEN_GRID,
    validate_v5_payload,
)
from evaluate_standard_3dpw import create_smpl_models
from geometry_metrics_3dpw import (
    hip_centered_j14,
    mean_euclidean_error,
    pa_mpjpe_per_sample,
    pelvis_aligned_j14,
)
from models.human_decoder_geometry import GeometryDecoderConfig
from models.human_decoder_geometry_video_v4 import (
    CENTER_RGB_POSITIONS,
    HumanGeometryVideoQueryTransformer,
)
from train_divibody_real_3dpw_geometry import (
    gendered_vertices,
    smpl_vertices,
)
from train_divibody_real_3dpw_temporal import (
    axis_angle_to_matrix,
    json_ready,
    seed_everything,
    sha256_file,
)


FORMAT = "divibody-continuous-vace-geometry-v5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--smpl-model-root", type=Path, required=True)
    parser.add_argument("--joint-regressor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--joint-weight", type=float, default=1.0)
    parser.add_argument("--vertex-weight", type=float, default=0.5)
    parser.add_argument("--rotation-weight", type=float, default=0.05)
    parser.add_argument("--shape-weight", type=float, default=0.001)
    parser.add_argument("--velocity-weight", type=float, default=0.5)
    parser.add_argument("--acceleration-weight", type=float, default=0.25)
    parser.add_argument("--boundary-acceleration-weight", type=float, default=0.0)
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--amp-dtype", choices=("bf16", "none"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-interval", type=int, default=1)
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def masked_mean(values: torch.Tensor, mask: torch.Tensor, label: str) -> torch.Tensor:
    """Average an already reduced per-primitive tensor over its unique owners."""

    if values.shape != mask.shape:
        raise ValueError(
            f"{label} values/mask shapes differ: {tuple(values.shape)} vs "
            f"{tuple(mask.shape)}"
        )
    if mask.dtype != torch.bool:
        raise TypeError(f"{label} owner mask must be boolean")
    count = mask.sum()
    if int(count) < 1:
        raise ValueError(f"{label} has no owned primitives")
    return values.masked_select(mask).mean()


def identity_rows(payload: dict[str, Any]) -> list[tuple[Any, ...]]:
    return [
        (
            str(record["sequence"]),
            int(record["person_id"]),
            tuple(int(value) for value in record["target_frame_ids"]),
        )
        for record in payload["metadata"]
    ]


def boundary_acceleration_locations(
    payload: dict[str, Any],
) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    """Resolve metric-only triplets to their three unique owner locations."""

    owners: dict[tuple[str, int, int], tuple[int, int]] = {}
    for window_index, (record, mask) in enumerate(
        zip(payload["metadata"], payload["frame_owner_mask"])
    ):
        for local, owned in enumerate(mask.tolist()):
            if not owned:
                continue
            key = (
                str(record["sequence"]), int(record["person_id"]),
                int(record["target_frame_ids"][local]),
            )
            if key in owners:
                raise RuntimeError("Duplicate v5 frame owner while resolving boundaries")
            owners[key] = (window_index, local)
    output = []
    records = payload["metric_only_cross_window_motion"]["acceleration_triplets"]
    for record in records:
        prefix = (str(record["sequence"]), int(record["person_id"]))
        locations = tuple(owners[(*prefix, int(frame_id))] for frame_id in record["frame_ids"])
        if len(locations) != 3:
            raise RuntimeError("Boundary acceleration record is not a triplet")
        output.append(locations)
    return output


def validate_cache_pair(
    train: dict[str, Any], val: dict[str, Any]
) -> dict[str, Any]:
    validate_v5_payload(train, feature_only=False)
    validate_v5_payload(val, feature_only=False)
    train_signature, val_signature = train["signature"], val["signature"]
    if train_signature.get("format") not in (CACHE_FORMAT, STRIDE4_FORMAT):
        raise ValueError("Training cache is not continuous VACE v5")
    if train_signature.get("split") != "train":
        raise ValueError("Training cache split must be train")
    if val_signature.get("split") != "validation":
        raise ValueError("Validation cache split must be validation")
    required_equal = (
        "context_frames_rgb",
        "latent_frames",
        "target_frames",
        "target_context_positions_inclusive",
        "target_stride",
        "image_size",
        "feature_layers",
        "token_order",
        "view_protocol",
        "encoder_protocol",
    )
    mismatches = [
        key for key in required_equal
        if train_signature.get(key) != val_signature.get(key)
    ]
    if mismatches:
        raise ValueError(f"Train/validation feature protocol mismatch: {mismatches}")
    train_sequences = {row[0] for row in identity_rows(train)}
    val_sequences = {row[0] for row in identity_rows(val)}
    overlap = sorted(train_sequences & val_sequences)
    if overlap:
        raise ValueError(f"Train/validation sequence leakage: {overlap}")
    return {
        "train_windows": len(train["metadata"]),
        "validation_windows": len(val["metadata"]),
        "train_sequences": sorted(train_sequences),
        "validation_sequences": sorted(val_sequences),
        "sequence_overlap": overlap,
        "development_only": bool(
            train_signature.get("development_only")
            or val_signature.get("development_only")
        ),
    }


class ContinuousGeometryDataset(Dataset):
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __len__(self) -> int:
        return int(self.payload["features"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        targets = self.payload["targets"]
        return {
            "features": self.payload["features"][index],
            "full_pose": targets["full_pose"][index],
            "betas": targets["betas"][index],
            "genders": targets["gender_code"][index],
            "cam_pose": targets["cam_pose"][index],
            "frame_owner": self.payload["frame_owner_mask"][index],
            "velocity_owner": self.payload["velocity_owner_mask"][index],
            "acceleration_owner": self.payload["acceleration_owner_mask"][index],
        }


def camera_targets(batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    poses = batch["full_pose"].to(device=device, dtype=torch.float32)
    cameras = batch["cam_pose"].to(device=device, dtype=torch.float32)[..., :3, :3]
    rotations = axis_angle_to_matrix(poses.reshape(-1, 24, 3)).reshape(
        *poses.shape[:2], 24, 3, 3
    )
    rotations[:, :, 0] = cameras @ rotations[:, :, 0]
    return rotations


def forward_features(
    decoder: HumanGeometryVideoQueryTransformer,
    features: torch.Tensor,
    device: torch.device,
    amp_dtype: str,
) -> dict[str, torch.Tensor]:
    features = features.to(device=device, dtype=torch.float32, non_blocking=True)
    if tuple(features.shape[1:4]) != (LATENT_FRAMES, *TOKEN_GRID):
        raise ValueError(f"Unexpected v5 feature grid {tuple(features.shape)}")
    batch_size = features.shape[0]
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and amp_dtype == "bf16",
    ):
        return decoder(
            features.reshape(batch_size, -1, features.shape[-1]),
            grid_size=(LATENT_FRAMES, *TOKEN_GRID),
            context_rgb_frames=17,
            target_positions=CENTER_RGB_POSITIONS,
        )


def geometric_losses(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    models: dict[int, torch.nn.Module],
    regressor: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    with torch.autocast(device_type=device.type, enabled=False):
        pred_rot = prediction["rotmat"].float()
        batch_size, frames = pred_rot.shape[:2]
        if frames != TARGET_FRAMES:
            raise ValueError("Geometry-v5 must predict exactly eight RGB frames")
        target_rot = camera_targets(batch, device)
        target_betas = batch["betas"].to(device=device, dtype=torch.float32)
        genders = batch["genders"].to(device=device).reshape(-1)
        pred_betas = prediction["betas"].float()[:, None].expand(-1, frames, -1)
        pred_vertices = smpl_vertices(
            models[2], pred_rot.reshape(-1, 24, 3, 3), pred_betas.reshape(-1, 10)
        )
        gt_vertices = gendered_vertices(
            models,
            target_rot.reshape(-1, 24, 3, 3),
            target_betas.reshape(-1, 10),
            genders,
        )
        pred_joints, _ = pelvis_aligned_j14(pred_vertices, regressor)
        gt_joints, _ = pelvis_aligned_j14(gt_vertices, regressor)
        joint_values = mean_euclidean_error(pred_joints, gt_joints).reshape(
            batch_size, frames
        )
        vertex_values = mean_euclidean_error(pred_vertices, gt_vertices).reshape(
            batch_size, frames
        )
        rotation_values = (pred_rot - target_rot).square().mean(
            dim=(-1, -2, -3)
        )
        frame_mask = batch["frame_owner"].to(device)
        joint = masked_mean(joint_values, frame_mask, "joint")
        vertex = masked_mean(vertex_values, frame_mask, "vertex")
        rotation = masked_mean(rotation_values, frame_mask, "rotation")
        shape = F.mse_loss(prediction["betas"].float(), target_betas[:, 0])

        pred_motion = hip_centered_j14(pred_joints).reshape(
            batch_size, frames, 14, 3
        )
        gt_motion = hip_centered_j14(gt_joints).reshape(
            batch_size, frames, 14, 3
        )
        motion_error = pred_motion - gt_motion
        velocity_values = torch.linalg.vector_norm(
            torch.diff(motion_error, dim=1), dim=-1
        ).mean(dim=-1)
        acceleration_values = torch.linalg.vector_norm(
            torch.diff(motion_error, n=2, dim=1), dim=-1
        ).mean(dim=-1)
        velocity = masked_mean(
            velocity_values, batch["velocity_owner"].to(device), "velocity"
        )
        acceleration = masked_mean(
            acceleration_values,
            batch["acceleration_owner"].to(device),
            "acceleration",
        )
        total = (
            args.joint_weight * joint
            + args.vertex_weight * vertex
            + args.rotation_weight * rotation
            + args.shape_weight * shape
            + args.velocity_weight * velocity
            + args.acceleration_weight * acceleration
        )
    return {
        "total": total,
        "joint_m": joint,
        "vertex_m": vertex,
        "rotation_chordal": rotation,
        "shape_mse": shape,
        "joint_velocity_m_per_frame": velocity,
        "joint_acceleration_m_per_frame2": acceleration,
    }


def boundary_acceleration_loss(
    decoder: HumanGeometryVideoQueryTransformer,
    payload: dict[str, Any],
    triplets: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]],
    models: dict[int, torch.nn.Module],
    regressor: torch.Tensor,
    amp_dtype: str,
    device: torch.device,
) -> torch.Tensor:
    """Backpropagate the full-track triplets omitted by within-window masks."""

    window_indices = sorted({window for triplet in triplets for window, _ in triplet})
    remap = {window: local for local, window in enumerate(window_indices)}
    index = torch.as_tensor(window_indices, dtype=torch.long)
    targets = payload["targets"]
    batch = {
        "features": payload["features"].index_select(0, index),
        "full_pose": targets["full_pose"].index_select(0, index),
        "betas": targets["betas"].index_select(0, index),
        "genders": targets["gender_code"].index_select(0, index),
        "cam_pose": targets["cam_pose"].index_select(0, index),
    }
    output = forward_features(decoder, batch["features"], device, amp_dtype)
    pred_rot = output["rotmat"].float()
    target_rot = camera_targets(batch, device)
    target_betas = batch["betas"].to(device=device, dtype=torch.float32)
    genders = batch["genders"].to(device=device).reshape(-1)
    pred_betas = output["betas"].float()[:, None].expand(-1, TARGET_FRAMES, -1)
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
        error = hip_centered_j14(pred_joints) - hip_centered_j14(gt_joints)
        error = error.reshape(len(window_indices), TARGET_FRAMES, 14, 3)
        values = []
        for triplet in triplets:
            points = torch.stack(
                [error[remap[window], local] for window, local in triplet]
            )
            values.append(
                torch.linalg.vector_norm(torch.diff(points, n=2, dim=0)[0], dim=-1).mean()
            )
        return torch.stack(values).mean()


@torch.no_grad()
def validate(
    loader: DataLoader,
    decoder: HumanGeometryVideoQueryTransformer,
    models: dict[int, torch.nn.Module],
    regressor: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    decoder.eval()
    sums = {name: 0.0 for name in ("mpjpe_mm", "pa_mpjpe_mm", "pve_mm")}
    count = 0
    for batch in loader:
        output = forward_features(decoder, batch["features"], device, args.amp_dtype)
        pred_rot = output["rotmat"].float()
        target_rot = camera_targets(batch, device)
        target_betas = batch["betas"].to(device=device, dtype=torch.float32)
        genders = batch["genders"].to(device=device).reshape(-1)
        pred_betas = output["betas"].float()[:, None].expand(-1, TARGET_FRAMES, -1)
        with torch.autocast(device_type=device.type, enabled=False):
            pred_vertices = smpl_vertices(
                models[2], pred_rot.reshape(-1, 24, 3, 3), pred_betas.reshape(-1, 10)
            )
            gt_vertices = gendered_vertices(
                models,
                target_rot.reshape(-1, 24, 3, 3),
                target_betas.reshape(-1, 10),
                genders,
            )
            pred_joints, _ = pelvis_aligned_j14(pred_vertices, regressor)
            gt_joints, _ = pelvis_aligned_j14(gt_vertices, regressor)
            arrays = {
                "mpjpe_mm": mean_euclidean_error(pred_joints, gt_joints) * 1000,
                "pa_mpjpe_mm": pa_mpjpe_per_sample(pred_joints, gt_joints) * 1000,
                "pve_mm": mean_euclidean_error(pred_vertices, gt_vertices) * 1000,
            }
            mask = batch["frame_owner"].to(device).reshape(-1)
            for name, values in arrays.items():
                values = values.reshape(-1).masked_select(mask)
                if not bool(torch.isfinite(values).all()):
                    raise FloatingPointError(f"Non-finite validation {name}")
                sums[name] += float(values.double().sum())
            count += int(mask.sum())
    if count < 1:
        raise RuntimeError("Validation has no owned target frames")
    return {**{name: value / count for name, value in sums.items()}, "frames": count}


def save_checkpoint(
    path: Path,
    decoder: HumanGeometryVideoQueryTransformer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val: float,
    validation: dict[str, float],
    args: argparse.Namespace,
    train_signature: dict[str, Any],
    val_signature: dict[str, Any],
    initialization: dict[str, Any],
) -> None:
    development_only = bool(
        train_signature.get("development_only")
        or val_signature.get("development_only")
    )
    payload = {
        "format": FORMAT,
        "epoch": epoch,
        "best_val": best_val,
        "selection_metric": (
            "development_validation_mpjpe_mm"
            if development_only
            else "validation_mpjpe_mm"
        ),
        "decoder_config": vars(decoder.config),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "validation": validation,
        "args": json_ready(vars(args)),
        "train_cache_signature": train_signature,
        "val_cache_signature": val_signature,
        "initialization": initialization,
        "development_only": development_only,
        "test_used_for_model_selection": False,
        "target_coordinate_system": "camera",
        "input_protocol": "T17 genuine continuous VACE -> five latent slices",
        "output_protocol": "explicit center RGB positions 4..11",
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    positive = (
        args.epochs,
        args.batch_size,
        args.val_batch_size,
        args.log_interval,
        args.lr,
        args.grad_clip,
    )
    if any(value <= 0 for value in positive) or args.num_workers < 0:
        raise ValueError("Invalid training configuration")
    if args.boundary_acceleration_weight < 0:
        raise ValueError("boundary-acceleration-weight cannot be negative")
    if any((args.output_dir / name).exists() for name in ("best.pt", "latest.pt")):
        raise FileExistsError("Refusing to overwrite a geometry-v5 experiment")
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    train_payload = load_payload(args.train_cache)
    val_payload = load_payload(args.val_cache)
    cache_audit = validate_cache_pair(train_payload, val_payload)

    source = load_payload(args.init_checkpoint)
    if source.get("format") not in (
        "divibody-real-3dpw-temporal-geometry-v3", FORMAT
    ):
        raise ValueError("Geometry-v5 requires a geometry-v3 or geometry-v5 initialization")
    config = GeometryDecoderConfig(**source["decoder_config"])
    decoder = HumanGeometryVideoQueryTransformer(config)
    if source["format"] == FORMAT:
        decoder.load_state_dict(source["decoder"], strict=True)
        loaded = {
            "source": "geometry-v5 strict continuation",
            "missing_keys": [],
            "unexpected_keys": [],
        }
    else:
        loaded = decoder.load_geometry_v3_weights(source["decoder"])
    decoder = decoder.to(device)
    for name in ("left_hand_head", "right_hand_head", "face_head", "object_head"):
        getattr(decoder, name).requires_grad_(False)
    initialization = {
        "path": str(args.init_checkpoint.resolve()),
        "sha256": sha256_file(args.init_checkpoint),
        "epoch": int(source["epoch"]),
        "format": source["format"],
        "weights": loaded,
        "new_rgb_query_embeddings_initialization": (
            "restored from geometry-v5"
            if source["format"] == FORMAT
            else "zeros"
        ),
    }
    del source
    models = create_smpl_models(args.smpl_model_root, device)
    for model in models.values():
        model.requires_grad_(False)
    regressor_np = np.load(args.joint_regressor)
    if regressor_np.shape != (17, 6890) or not np.isfinite(regressor_np).all():
        raise ValueError("Invalid H36M joint regressor")
    regressor = torch.as_tensor(regressor_np, dtype=torch.float32, device=device)
    options = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        ContinuousGeometryDataset(train_payload),
        batch_size=args.batch_size,
        shuffle=True,
        **options,
    )
    val_loader = DataLoader(
        ContinuousGeometryDataset(val_payload),
        batch_size=args.val_batch_size,
        shuffle=False,
        **options,
    )
    boundary_triplets = boundary_acceleration_locations(train_payload)
    if args.boundary_acceleration_weight > 0 and not boundary_triplets:
        raise ValueError("Boundary acceleration requested but cache has no triplets")
    optimizer = torch.optim.AdamW(
        (parameter for parameter in decoder.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "format": FORMAT,
        "args": json_ready(vars(args)),
        "cache_audit": cache_audit,
        "initialization": initialization,
        "train_cache_sha256": sha256_file(args.train_cache),
        "val_cache_sha256": sha256_file(args.val_cache),
        "train_encoder_protocol": train_payload["signature"]["encoder_protocol"],
        "test_data_used": False,
        "development_only": cache_audit["development_only"],
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    baseline = validate(val_loader, decoder, models, regressor, args, device)
    best_value, best_epoch, best_metrics = baseline["mpjpe_mm"], 0, baseline
    metrics_path = args.output_dir / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"epoch": 0, "validation": baseline}) + "\n", encoding="utf-8"
    )
    save_checkpoint(
        args.output_dir / "best.pt", decoder, optimizer, 0, best_value, baseline,
        args, train_payload["signature"], val_payload["signature"], initialization,
    )
    for epoch in range(1, args.epochs + 1):
        decoder.train()
        totals: dict[str, float] = {}
        seen = 0
        started = time.monotonic()
        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)
            output = forward_features(decoder, batch["features"], device, args.amp_dtype)
            losses = geometric_losses(output, batch, models, regressor, args, device)
            if not all(bool(torch.isfinite(value)) for value in losses.values()):
                raise FloatingPointError(f"Non-finite v5 loss at {epoch=}, {step=}")
            losses["total"].backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                decoder.parameters(), args.grad_clip, error_if_nonfinite=True
            )
            optimizer.step()
            count = len(batch["features"])
            seen += count
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * count
            if step % args.log_interval == 0 or step == len(train_loader):
                print(json.dumps({
                    "phase": "continuous_geometry_v5_training",
                    "epoch": epoch,
                    "step": step,
                    "steps": len(train_loader),
                    "joint_mm": float(losses["joint_m"].detach()) * 1000,
                    "gradient_norm": float(gradient),
                }), flush=True)
        boundary_sum = 0.0
        boundary_seen = 0
        if args.boundary_acceleration_weight > 0:
            order = torch.randperm(len(boundary_triplets)).tolist()
            for boundary_step, offset in enumerate(
                range(0, len(order), args.batch_size), 1
            ):
                selected = [
                    boundary_triplets[index]
                    for index in order[offset : offset + args.batch_size]
                ]
                optimizer.zero_grad(set_to_none=True)
                boundary = boundary_acceleration_loss(
                    decoder, train_payload, selected, models, regressor,
                    args.amp_dtype, device,
                )
                weighted_boundary = args.boundary_acceleration_weight * boundary
                if not bool(torch.isfinite(weighted_boundary)):
                    raise FloatingPointError(
                        f"Non-finite boundary loss at {epoch=}, {boundary_step=}"
                    )
                weighted_boundary.backward()
                gradient = torch.nn.utils.clip_grad_norm_(
                    decoder.parameters(), args.grad_clip, error_if_nonfinite=True
                )
                optimizer.step()
                boundary_sum += float(boundary.detach()) * len(selected)
                boundary_seen += len(selected)
                if boundary_step % args.log_interval == 0:
                    print(json.dumps({
                        "phase": "continuous_geometry_v5_boundary_training",
                        "epoch": epoch,
                        "step": boundary_step,
                        "triplets": boundary_seen,
                        "boundary_acceleration_mm_per_frame2": float(boundary.detach()) * 1000,
                        "gradient_norm": float(gradient),
                    }), flush=True)
        train_metrics = {
            **{name: value / seen for name, value in totals.items()},
            "windows": seen,
            "boundary_acceleration_m_per_frame2": (
                boundary_sum / boundary_seen if boundary_seen else None
            ),
            "boundary_acceleration_triplets": boundary_seen,
            "seconds": time.monotonic() - started,
        }
        validation = validate(val_loader, decoder, models, regressor, args, device)
        improved = validation["mpjpe_mm"] < best_value
        if improved:
            best_value, best_epoch, best_metrics = (
                validation["mpjpe_mm"], epoch, validation
            )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation,
            "best_epoch": best_epoch,
            "best_validation_mpjpe_mm": best_value,
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        save_checkpoint(
            args.output_dir / "latest.pt", decoder, optimizer, epoch, best_value,
            validation, args, train_payload["signature"], val_payload["signature"],
            initialization,
        )
        if improved:
            save_checkpoint(
                args.output_dir / "best.pt", decoder, optimizer, epoch, best_value,
                validation, args, train_payload["signature"], val_payload["signature"],
                initialization,
            )
        if args.save_every_epoch:
            save_checkpoint(
                args.output_dir / f"epoch_{epoch:03d}.pt", decoder, optimizer,
                epoch, best_value, validation, args,
                train_payload["signature"], val_payload["signature"],
                initialization,
            )
        print(json.dumps(record), flush=True)
    result = {
        "status": "DIVIBODY_CONTINUOUS_GEOMETRY_V5_TRAINING_OK",
        "best_epoch": best_epoch,
        "best_validation": best_metrics,
        "initialization_validation": baseline,
        "development_only": cache_audit["development_only"],
        "best_checkpoint": str((args.output_dir / "best.pt").resolve()),
    }
    (args.output_dir / "final_validation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
