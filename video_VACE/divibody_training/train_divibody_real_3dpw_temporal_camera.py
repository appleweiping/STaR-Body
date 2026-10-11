"""Train the temporal decoder with 3DPW global orientation in camera space.

This additive v2 experiment leaves the framewise and temporal-v1 trainers and
all original model sources unchanged.  Raw 3DPW poses store global orientation
in world coordinates; here each target is transformed with
``R_global_camera = R_camera_from_world @ R_global_world`` before supervision.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

import train_divibody_real_3dpw_temporal as base
from models.human_decoder import HumanDecoderConfig, HumanQueryTransformer


class CameraTemporalClipDataset(base.TemporalClipDataset):
    """Temporal clips augmented with the per-frame world-to-camera rotation."""

    def __init__(
        self,
        cache_path: Path,
        expected_split: str,
        clip_length: int,
        clip_stride: int,
    ) -> None:
        super().__init__(cache_path, expected_split, clip_length, clip_stride)
        camera_rotations = np.asarray(
            [record["cam_pose"] for record in self.metadata], dtype=np.float32
        )[:, :3, :3]
        if camera_rotations.shape != (len(self.metadata), 3, 3):
            raise ValueError("Every cached frame must contain a 4x4 camera pose")
        if not np.isfinite(camera_rotations).all():
            raise ValueError("Camera rotations contain non-finite values")
        identity = np.eye(3, dtype=np.float32)
        orthogonality_error = np.max(
            np.abs(camera_rotations.transpose(0, 2, 1) @ camera_rotations - identity)
        )
        determinants = np.linalg.det(camera_rotations)
        if orthogonality_error > 2.0e-3 or np.max(np.abs(determinants - 1.0)) > 2.0e-3:
            raise ValueError(
                "Camera metadata does not contain proper rotation matrices"
            )
        self.camera_rotations = torch.from_numpy(camera_rotations)
        self.stats = {
            **self.stats,
            "global_orientation_target": "camera",
            "camera_transform": "R_camera_from_world @ R_global_world",
            "max_camera_orthogonality_error": float(orthogonality_error),
            "max_camera_determinant_error": float(np.max(np.abs(determinants - 1.0))),
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        result = super().__getitem__(index)
        result["camera_rotation"] = self.camera_rotations.index_select(
            0, result["sample_indices"]
        )
        return result


def camera_temporal_losses(
    prediction: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    args: Any,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    target_global_world = batch["global_orient"].to(device=device, dtype=torch.float32)
    target_body = batch["body_pose"].to(device=device, dtype=torch.float32)
    target_betas = batch["betas"].to(device=device, dtype=torch.float32)
    target_transl = batch["transl"].to(device=device, dtype=torch.float32)
    camera_rotation = batch["camera_rotation"].to(device=device, dtype=torch.float32)
    pred_global = prediction["global_orient"].float()
    pred_body = prediction["body_pose"].float()
    pred_betas = prediction["betas"].float()
    pred_transl = prediction["transl"].float()

    pred_global_rotation = base.axis_angle_to_matrix(pred_global)
    target_global_rotation = camera_rotation @ base.axis_angle_to_matrix(
        target_global_world
    )
    pred_body_rotation = base.axis_angle_to_matrix(
        pred_body.reshape(*pred_body.shape[:-1], 21, 3)
    )
    target_body_rotation = base.axis_angle_to_matrix(
        target_body.reshape(*target_body.shape[:-1], 21, 3)
    )
    pred_rotations = torch.cat(
        (pred_global_rotation.unsqueeze(2), pred_body_rotation), dim=2
    )
    target_rotations = torch.cat(
        (target_global_rotation.unsqueeze(2), target_body_rotation), dim=2
    )

    global_rotation = F.mse_loss(pred_global_rotation, target_global_rotation)
    body_rotation = F.mse_loss(pred_body_rotation, target_body_rotation)
    # Root orientation is the coordinate-system correction this v2 experiment
    # is designed to learn.  Weight it as one semantic group rather than 1/22
    # of an undifferentiated joint mean.
    rotation = global_rotation + body_rotation
    body_axis_angle = F.smooth_l1_loss(pred_body, target_body)
    shape = F.smooth_l1_loss(pred_betas, target_betas)
    translation = F.smooth_l1_loss(pred_transl, target_transl)

    pred_velocity = pred_rotations[:, 1:] - pred_rotations[:, :-1]
    target_velocity = target_rotations[:, 1:] - target_rotations[:, :-1]
    global_velocity = F.smooth_l1_loss(pred_velocity[:, :, 0], target_velocity[:, :, 0])
    body_velocity = F.smooth_l1_loss(pred_velocity[:, :, 1:], target_velocity[:, :, 1:])
    velocity = global_velocity + body_velocity
    pred_acceleration = pred_velocity[:, 1:] - pred_velocity[:, :-1]
    target_acceleration = target_velocity[:, 1:] - target_velocity[:, :-1]
    global_acceleration = F.smooth_l1_loss(
        pred_acceleration[:, :, 0], target_acceleration[:, :, 0]
    )
    body_acceleration = F.smooth_l1_loss(
        pred_acceleration[:, :, 1:], target_acceleration[:, :, 1:]
    )
    acceleration = global_acceleration + body_acceleration

    unsupported = (
        prediction["left_hand_pose"].float().square().mean()
        + prediction["right_hand_pose"].float().square().mean()
        + prediction["jaw_pose"].float().square().mean()
        + prediction["expression"].float().square().mean()
        + prediction["object_state"].float().square().mean()
    )
    total = (
        args.rotation_weight * rotation
        + args.axis_angle_weight * body_axis_angle
        + args.shape_weight * shape
        + args.translation_weight * translation
        + args.velocity_weight * velocity
        + args.acceleration_weight * acceleration
        + args.unsupported_head_reg * unsupported
    )
    return {
        "total": total,
        "rotation_chordal": rotation,
        "global_camera_rotation_chordal": global_rotation,
        "body_rotation_chordal": body_rotation,
        "body_axis_angle": body_axis_angle,
        "shape": shape,
        "translation": translation,
        "rotation_velocity": velocity,
        "global_rotation_velocity": global_velocity,
        "body_rotation_velocity": body_velocity,
        "rotation_acceleration": acceleration,
        "global_rotation_acceleration": global_acceleration,
        "body_rotation_acceleration": body_acceleration,
        "unsupported_regularizer": unsupported,
    }


def reset_global_orientation_rows(
    decoder: HumanQueryTransformer, seed: int
) -> Dict[str, Any]:
    """Discard the frame model's world-orientation output rows only."""

    generator = torch.Generator(device="cpu").manual_seed(seed + 1701)
    weight = (
        torch.randn(
            decoder.body_head.weight[:3].shape,
            generator=generator,
            dtype=decoder.body_head.weight.dtype,
        )
        * 0.02
    )
    with torch.no_grad():
        decoder.body_head.weight[:3].copy_(weight.to(decoder.body_head.weight.device))
        decoder.body_head.bias[:3].zero_()
    return {
        "rows": "body_head[0:3]",
        "reason": "world-frame warm-start rows are incompatible with camera targets",
        "method": "deterministic normal std=0.02 and zero bias",
        "seed": seed + 1701,
    }


def validate_frame_initialization_checkpoint(
    path: Path, expected_decoder_config: Dict[str, Any]
) -> None:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("format") != "divibody-real-3dpw-v1":
        raise ValueError(
            "Camera v2 only accepts a framewise divibody-real-3dpw-v1 warm "
            "start; temporal resume is intentionally unsupported"
        )
    actual_config = checkpoint.get("decoder_config", {})
    mismatches = {
        key: (actual_config.get(key), value)
        for key, value in expected_decoder_config.items()
        if actual_config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Incompatible framewise decoder configuration: {mismatches}")


def save_checkpoint(
    path: Path,
    decoder: HumanQueryTransformer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val: float,
    validation: Dict[str, float],
    config: HumanDecoderConfig,
    args: Any,
    train_dataset: CameraTemporalClipDataset,
    val_dataset: CameraTemporalClipDataset,
    initialization: Optional[Dict[str, Any]],
) -> None:
    payload = {
        "format": "divibody-real-3dpw-temporal-camera-v2",
        "epoch": epoch,
        "best_val": best_val,
        "decoder_config": vars(config),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "validation": validation,
        "args": base.json_ready(vars(args)),
        "train_cache_signature": train_dataset.signature,
        "val_cache_signature": val_dataset.signature,
        "train_clip_stats": train_dataset.stats,
        "val_clip_stats": val_dataset.stats,
        "initialization": initialization,
        "target_coordinate_system": "camera",
        "camera_transform": "R_camera_from_world @ R_global_world",
        "pose_protocol": {
            "id": "3dpw_global_orient_camera_matrix_v1",
            "source_global_orient_frame": "world",
            "prediction_global_orient_frame": "camera",
            "cam_pose_semantics": "T_camera_from_world",
            "composition": "R_camera_from_world @ R_global_world",
            "body_pose_frame": "smpl_local_unchanged",
            "translation": "unsupervised world-cache target ignored",
            "rotation_conversion": "axis_angle_to_matrix_sinc_v1",
        },
        "loss_protocol": {
            "global_rotation": "separate equal-weight chordal matrix loss",
            "body_rotation": "21-joint mean chordal matrix loss",
            "global_axis_angle_weight": 0.0,
            "body_axis_angle_weight": args.axis_angle_weight,
            "temporal_rotation_frame": "camera",
            "root_and_body_temporal_groups": "equal weight",
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    args = base.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("epochs/batch-size must be positive and workers non-negative")
    if args.hidden_dim % args.decoder_heads:
        raise ValueError("hidden-dim must be divisible by decoder-heads")
    if args.log_interval <= 0:
        raise ValueError("log-interval must be positive")
    if args.translation_weight != 0:
        raise ValueError(
            "Camera v2 requires translation-weight=0 because cached translation "
            "remains in world coordinates"
        )
    if args.init_checkpoint is not None:
        validate_frame_initialization_checkpoint(
            args.init_checkpoint,
            {
                "input_dim": 1536,
                "hidden_dim": args.hidden_dim,
                "num_heads": args.decoder_heads,
                "num_layers": args.decoder_layers,
                "body_pose_dim": 63,
            },
        )
    base.seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    train_dataset = CameraTemporalClipDataset(
        args.train_cache, "train", args.clip_length, args.train_clip_stride
    )
    val_dataset = CameraTemporalClipDataset(
        args.val_cache, "validation", args.clip_length, args.val_clip_stride
    )
    feature_keys = ("image_size", "feature_backend", "feature_layers", "vace_timestep")
    signature_mismatches = {
        key: (train_dataset.signature.get(key), val_dataset.signature.get(key))
        for key in feature_keys
        if train_dataset.signature.get(key) != val_dataset.signature.get(key)
    }
    if signature_mismatches:
        raise ValueError(
            f"Train/validation feature-cache settings differ: {signature_mismatches}"
        )
    overlap = sorted(
        set(train_dataset.stats["sequence_names"])
        & set(val_dataset.stats["sequence_names"])
    )
    if overlap:
        raise RuntimeError(f"Train/validation sequence leakage: {overlap}")
    train_loader = base.make_loader(
        train_dataset, args.batch_size, True, args.num_workers, device
    )
    val_loader = base.make_loader(
        val_dataset, args.batch_size, False, args.num_workers, device
    )
    config = HumanDecoderConfig(
        input_dim=1536,
        hidden_dim=args.hidden_dim,
        num_heads=args.decoder_heads,
        num_layers=args.decoder_layers,
        dropout=args.dropout,
        body_pose_dim=63,
        max_latent_frames=args.clip_length,
    )
    decoder = HumanQueryTransformer(config).to(device)
    initialization = (
        base.initialise_from_frame_checkpoint(decoder, args.init_checkpoint, args.seed)
        if args.init_checkpoint is not None
        else {}
    )
    initialization = {
        **(initialization or {}),
        "global_orientation_head_reset": reset_global_orientation_rows(
            decoder, args.seed
        ),
    }
    existing_outputs = [
        args.output_dir / name
        for name in ("best.pt", "latest.pt", "metrics.jsonl", "run_config.json")
        if (args.output_dir / name).exists()
    ]
    if existing_outputs:
        raise FileExistsError(
            "Refusing to overwrite or append an existing camera-v2 run: "
            + ", ".join(map(str, existing_outputs))
        )
    optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "status": "temporal_camera_training_started",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "decoder_config": vars(config),
        "train_clip_stats": train_dataset.stats,
        "val_clip_stats": val_dataset.stats,
        "train_validation_sequence_overlap": overlap,
        "initialization": initialization,
        "args": base.json_ready(vars(args)),
        "target_coordinate_system": "camera",
        "camera_transform": "R_camera_from_world @ R_global_world",
        "pose_protocol": {
            "id": "3dpw_global_orient_camera_matrix_v1",
            "source_global_orient_frame": "world",
            "prediction_global_orient_frame": "camera",
            "cam_pose_semantics": "T_camera_from_world",
            "composition": "R_camera_from_world @ R_global_world",
            "body_pose_frame": "smpl_local_unchanged",
            "translation": "unsupervised world-cache target ignored",
            "rotation_conversion": "axis_angle_to_matrix_sinc_v1",
        },
        "loss_protocol": {
            "frame_rotation": "equal-weight camera root plus 21-joint body chordal losses",
            "frame_axis_angle": "auxiliary SmoothL1 on 21 body joints only",
            "temporal_velocity": "equal-weight camera root plus body rotation deltas matched to GT",
            "temporal_acceleration": "equal-weight camera root plus body second rotation deltas matched to GT",
            "translation_in_primary_loss": args.translation_weight > 0,
        },
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(run_config, indent=2, ensure_ascii=False), flush=True)

    base.temporal_losses = camera_temporal_losses
    metrics_path = args.output_dir / "metrics.jsonl"
    best_val = math.inf
    final_validation: Dict[str, float] = {}
    for epoch in range(1, args.epochs + 1):
        train_metrics = base.run_epoch(
            train_loader, decoder, optimizer, args, device, epoch, "train"
        )
        val_metrics = base.run_epoch(
            val_loader, decoder, None, args, device, epoch, "validation"
        )
        final_validation = val_metrics
        record = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        improved = val_metrics["total"] < best_val
        if improved:
            best_val = val_metrics["total"]
        for filename in ("latest.pt",) + (("best.pt",) if improved else ()):
            save_checkpoint(
                args.output_dir / filename,
                decoder,
                optimizer,
                epoch,
                best_val,
                val_metrics,
                config,
                args,
                train_dataset,
                val_dataset,
                initialization,
            )
        print(json.dumps(record, indent=2, ensure_ascii=False), flush=True)

    result = {
        "status": "DIVIBODY_TEMPORAL_CAMERA_3DPW_TRAINING_OK",
        "completed_epochs": args.epochs,
        "best_validation_total": best_val,
        "final_validation": final_validation,
        "best_checkpoint": str((args.output_dir / "best.pt").resolve()),
        "latest_checkpoint": str((args.output_dir / "latest.pt").resolve()),
        "metrics": str(metrics_path.resolve()),
        "target_coordinate_system": "camera",
    }
    (args.output_dir / "final_validation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
