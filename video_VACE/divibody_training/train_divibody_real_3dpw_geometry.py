"""Additive camera-space SMPL geometry training, leaving v1/v2 untouched.

The frozen VACE inputs are unchanged in this controlled experiment.  Unlike
the parameter-only trainers, the loss differentiates through neutral SMPL and
supervises actual camera-space J14, vertices and consecutive joint motion.
Checkpoint selection uses one centered prediction per validation frame, not
the training loss or any test result.  Full 72-D ground truth is re-read from
the licensed 3DPW annotations and matched to every cached image identity.
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
from torch.utils.data import DataLoader, Dataset, Subset

from evaluate_standard_3dpw import (
    Standard3DPWDataset,
    _normalise_gender,
    audit_split,
    create_smpl_models,
)
from evaluate_temporal_3dpw import (
    CenteredFrameClipDataset,
    validate_cache_and_alignment,
    validate_feature_compatibility,
)
from geometry_metrics_3dpw import (
    hip_centered_j14,
    mean_euclidean_error,
    pa_mpjpe_per_sample,
    pelvis_aligned_j14,
)
from models.human_decoder_geometry import (
    GeometryDecoderConfig,
    HumanGeometryQueryTransformer,
)
from real_3dpw_dataset import _load_annotation
from train_divibody_real_3dpw_temporal import (
    TemporalClipDataset,
    axis_angle_to_matrix,
    json_ready,
    seed_everything,
    sha256_file,
)


FORMAT = "divibody-real-3dpw-temporal-geometry-v3"
POSE_PROTOCOL = {
    "id": "3dpw_camera_smpl24_geometry_v3",
    "source_global_orient_frame": "world",
    "prediction_global_orient_frame": "camera",
    "cam_pose_semantics": "T_camera_from_world",
    "composition": "R_camera_from_world @ R_global_world",
    "body_pose_frame": "smpl_local_unchanged",
    "smpl_rotation_count": 24,
    "rotation_output": "6D residual matrix @ camera-v2 axis-angle anchor matrix",
    "translation": "not used; both SMPL meshes have translation disabled",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--smpl-model-root", type=Path, required=True)
    parser.add_argument("--joint-regressor", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--train-clip-stride", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--joint-weight", type=float, default=1.0)
    parser.add_argument("--vertex-weight", type=float, default=0.5)
    parser.add_argument("--rotation-weight", type=float, default=0.05)
    parser.add_argument("--shape-weight", type=float, default=0.001)
    parser.add_argument("--velocity-weight", type=float, default=0.5)
    parser.add_argument("--acceleration-weight", type=float, default=0.25)
    parser.add_argument("--amp-dtype", choices=("bf16", "none"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--smoke-max-steps", type=int, default=0,
                        help="Development only: truncate training and validation")
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


class FullGeometryTargets:
    """Strictly match full SMPL labels to an existing per-image feature cache."""

    def __init__(self, payload: dict[str, Any], data_root: Path, split: str) -> None:
        signature = payload["signature"]
        if split not in ("train", "validation") or signature.get("split") != split:
            raise ValueError("Training targets may only read train/validation")
        metadata = payload["metadata"]
        audit = audit_split(data_root, split)
        if (
            audit["missing_unique_rgb_frames"]
            or audit["available_valid_person_frames"] != audit["expected_valid_person_frames"]
            or audit["invalid_gender_tracks"]
            or audit["invalid_camera_frame_count"]
        ):
            raise RuntimeError(f"Incomplete/invalid official {split} data")
        dataset = Standard3DPWDataset(
            root=data_root, split=split, image_size=int(signature["image_size"]),
            frame_stride=1, max_samples=None, crop_to_person=True,
        )
        keys = [(str(m["sequence"]), int(m["person_id"]), int(m["frame_id"]))
                for m in metadata]
        expected = [(s.sequence, s.person_index, s.frame_index) for s in dataset.samples]
        if keys != expected or len(keys) != len(set(keys)):
            raise RuntimeError(f"{split} cache is not the complete strict RGB index")
        if len(keys) != audit["available_valid_person_frames"]:
            raise RuntimeError(f"Incomplete {split} cache")

        pose_list, beta_list, gender_list, camera_list = [], [], [], []
        for record, sample in zip(metadata, dataset.samples):
            annotation = _load_annotation(sample.annotation_path)
            person, frame = sample.person_index, sample.frame_index
            pose_list.append(np.asarray(annotation["poses"][person][frame],
                                        dtype=np.float32)[:72])
            beta_list.append(np.asarray(annotation["betas"][person],
                                        dtype=np.float32)[:10])
            gender_list.append(_normalise_gender(annotation["genders"][person]))
            camera = np.asarray(annotation["cam_poses"][frame], dtype=np.float32)
            if not np.allclose(camera, record["cam_pose"], atol=1e-6, rtol=0):
                raise RuntimeError(f"Cached camera disagrees with annotation: {sample}")
            if Path(record["image_path"]).name != f"image_{frame:05d}.jpg":
                if Path(record["image_path"]).stem != f"image_{frame:05d}":
                    raise RuntimeError("Cache uses a source/IMU frame instead of RGB frame")
            camera_list.append(camera[:3, :3])
        poses = torch.from_numpy(np.stack(pose_list))
        self.betas = torch.from_numpy(np.stack(beta_list))
        cameras = torch.from_numpy(np.stack(camera_list))
        self.genders = torch.tensor(gender_list, dtype=torch.long)
        if poses.shape != (len(keys), 72) or self.betas.shape != (len(keys), 10):
            raise ValueError("Invalid full SMPL annotation shape")
        if not all(torch.isfinite(t).all() for t in (poses, self.betas, cameras)):
            raise ValueError("Non-finite geometry targets")
        cached = payload["targets"]
        comparisons = {"betas": self.betas}
        if "full_pose" in cached:
            comparisons["full_pose"] = poses
            if not torch.equal(torch.as_tensor(cached["gender_code"]), self.genders):
                raise RuntimeError("Cached gender does not match original annotation")
        else:
            comparisons.update(global_orient=poses[:, :3], body_pose=poses[:, 3:66])
        for name, expected_tensor in comparisons.items():
            actual = torch.as_tensor(cached[name]).float()
            if actual.shape != expected_tensor.shape or not torch.allclose(
                actual, expected_tensor, atol=1e-6, rtol=0
            ):
                raise RuntimeError(f"Cached {name} does not match annotation")
        self.rotations = axis_angle_to_matrix(poses.reshape(-1, 24, 3))
        self.rotations[:, 0] = cameras @ self.rotations[:, 0]
        self.keys = keys
        self.audit = audit


class GeometryClipDataset(Dataset):
    def __init__(self, base: TemporalClipDataset, targets: FullGeometryTargets) -> None:
        self.base, self.targets = base, targets

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.base[index]
        indices = item["sample_indices"]
        return {
            "features": item["features"],
            "rotmat": self.targets.rotations.index_select(0, indices),
            "betas": self.targets.betas.index_select(0, indices),
            "genders": self.targets.genders.index_select(0, indices),
        }


def smpl_vertices(model: torch.nn.Module, rotations: torch.Tensor,
                  betas: torch.Tensor) -> torch.Tensor:
    return model(
        global_orient=rotations[:, :1], body_pose=rotations[:, 1:],
        betas=betas, pose2rot=False,
    ).vertices


@torch.no_grad()
def gendered_vertices(models: dict[int, torch.nn.Module], rotations: torch.Tensor,
                      betas: torch.Tensor, genders: torch.Tensor) -> torch.Tensor:
    if not bool(((genders == 0) | (genders == 1)).all()):
        raise ValueError("Official 3DPW GT must contain male/female gender labels")
    output = rotations.new_empty((len(rotations), 6890, 3))
    for code in (0, 1):
        indices = torch.nonzero(genders == code, as_tuple=False).flatten()
        if indices.numel():
            vertices = smpl_vertices(models[code], rotations.index_select(0, indices),
                                     betas.index_select(0, indices))
            output.index_copy_(0, indices, vertices)
    return output


def geometric_losses(prediction: dict[str, torch.Tensor], batch: dict[str, torch.Tensor],
                     models: dict[int, torch.nn.Module], regressor: torch.Tensor,
                     args: argparse.Namespace, device: torch.device
                     ) -> dict[str, torch.Tensor]:
    # Float32 SMPL/LBS and geometry are essential, even when the decoder uses BF16.
    with torch.autocast(device_type=device.type, enabled=False):
        pred_rot = prediction["rotmat"].float()
        batch_size, frames = pred_rot.shape[:2]
        target_rot = batch["rotmat"].to(device=device, dtype=torch.float32)
        target_betas = batch["betas"].to(device=device, dtype=torch.float32)
        genders = batch["genders"].to(device=device).flatten()
        pred_betas = prediction["betas"].float()[:, None].expand(-1, frames, -1)
        pred_vertices = smpl_vertices(models[2], pred_rot.reshape(-1, 24, 3, 3),
                                      pred_betas.reshape(-1, 10))
        gt_vertices = gendered_vertices(models, target_rot.reshape(-1, 24, 3, 3),
                                        target_betas.reshape(-1, 10), genders)
        pred_joints, _ = pelvis_aligned_j14(pred_vertices, regressor)
        target_joints, _ = pelvis_aligned_j14(gt_vertices, regressor)
        joint = mean_euclidean_error(pred_joints, target_joints).mean()
        vertex = mean_euclidean_error(pred_vertices, gt_vertices).mean()
        global_rotation = F.mse_loss(pred_rot[:, :, 0], target_rot[:, :, 0])
        body_rotation = F.mse_loss(pred_rot[:, :, 1:], target_rot[:, :, 1:])
        shape = F.mse_loss(pred_betas, target_betas)
        # Hip-centered J14 is the same temporal coordinate convention as VIBE.
        pred_motion = hip_centered_j14(pred_joints).reshape(batch_size, frames, 14, 3)
        gt_motion = hip_centered_j14(target_joints).reshape(batch_size, frames, 14, 3)
        velocity_error = torch.diff(pred_motion - gt_motion, dim=1)
        velocity = torch.linalg.vector_norm(velocity_error, dim=-1).mean()
        acceleration = torch.linalg.vector_norm(
            torch.diff(velocity_error, dim=1), dim=-1
        ).mean()
        total = (
            args.joint_weight * joint + args.vertex_weight * vertex
            + args.rotation_weight * (global_rotation + body_rotation)
            + args.shape_weight * shape + args.velocity_weight * velocity
            + args.acceleration_weight * acceleration
        )
    return {
        "total": total, "joint_m": joint, "vertex_m": vertex,
        "global_rotation_chordal": global_rotation,
        "body_rotation_chordal": body_rotation, "shape_mse": shape,
        "joint_velocity_m_per_frame": velocity,
        "joint_acceleration_m_per_frame2": acceleration,
    }


def forward_features(decoder: HumanGeometryQueryTransformer, features: torch.Tensor,
                     args: argparse.Namespace, device: torch.device
                     ) -> dict[str, torch.Tensor]:
    features = features.to(device=device, dtype=torch.float32, non_blocking=True)
    batch_size, frames, tokens, channels = features.shape
    side = math.isqrt(tokens)
    if side * side != tokens:
        raise ValueError("Expected a square spatial feature grid")
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=device.type == "cuda" and args.amp_dtype == "bf16"):
        return decoder(features.reshape(batch_size, frames * tokens, channels),
                       grid_size=(frames, side, side), target_frames=frames)


def train_epoch(loader: DataLoader, decoder: HumanGeometryQueryTransformer,
                optimizer: torch.optim.Optimizer, models: dict[int, torch.nn.Module],
                regressor: torch.Tensor, args: argparse.Namespace,
                device: torch.device, epoch: int) -> dict[str, float]:
    decoder.train()
    totals: dict[str, float] = {}
    seen = 0
    started = time.monotonic()
    for step, batch in enumerate(loader, 1):
        optimizer.zero_grad(set_to_none=True)
        output = forward_features(decoder, batch["features"], args, device)
        losses = geometric_losses(output, batch, models, regressor, args, device)
        if not all(bool(torch.isfinite(value)) for value in losses.values()):
            raise FloatingPointError(f"Non-finite geometry loss at {epoch=}, {step=}")
        losses["total"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            decoder.parameters(), args.grad_clip, error_if_nonfinite=True
        )
        optimizer.step()
        count = len(batch["features"])
        seen += count
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * count
        if step % args.log_interval == 0 or step == len(loader) or step == 1:
            print(json.dumps({"phase": "geometry_training", "epoch": epoch,
                              "step": step, "steps": len(loader), "clips": seen,
                              "loss": float(losses["total"].detach()),
                              "joint_mm": float(losses["joint_m"].detach()) * 1000,
                              "gradient_norm": float(grad_norm),
                              "seconds": time.monotonic() - started}), flush=True)
        if args.smoke_max_steps and step >= args.smoke_max_steps:
            break
    return {**{name: value / seen for name, value in totals.items()},
            "clips": seen, "seconds": time.monotonic() - started}


@torch.no_grad()
def validate(loader: DataLoader, decoder: HumanGeometryQueryTransformer,
             targets: FullGeometryTargets, models: dict[int, torch.nn.Module],
             regressor: torch.Tensor, args: argparse.Namespace,
             device: torch.device) -> dict[str, float]:
    decoder.eval()
    totals = {name: 0.0 for name in ("mpjpe_mm", "pa_mpjpe_mm", "pve_mm")}
    visited: set[int] = set()
    started = time.monotonic()
    for batch in loader:
        output = forward_features(decoder, batch["features"], args, device)
        indices = batch["target_index"].long()
        local = batch["local_position"].to(device)
        if any(int(i) in visited for i in indices) or len(set(indices.tolist())) != len(indices):
            raise RuntimeError("Validation contains duplicate frames")
        visited.update(indices.tolist())
        rows = torch.arange(len(indices), device=device)
        rotations = output["rotmat"][rows, local].float()
        with torch.autocast(device_type=device.type, enabled=False):
            pred_vertices = smpl_vertices(models[2], rotations, output["betas"].float())
            gt_vertices = gendered_vertices(
                models, targets.rotations.index_select(0, indices).to(device),
                targets.betas.index_select(0, indices).to(device),
                targets.genders.index_select(0, indices).to(device),
            )
            pred_joints, _ = pelvis_aligned_j14(pred_vertices, regressor)
            target_joints, _ = pelvis_aligned_j14(gt_vertices, regressor)
            arrays = {
                "mpjpe_mm": mean_euclidean_error(pred_joints, target_joints) * 1000,
                "pa_mpjpe_mm": pa_mpjpe_per_sample(pred_joints, target_joints) * 1000,
                "pve_mm": mean_euclidean_error(pred_vertices, gt_vertices) * 1000,
            }
            for name, values in arrays.items():
                if not torch.isfinite(values).all():
                    raise FloatingPointError(f"Non-finite validation {name}")
                totals[name] += float(values.double().sum())
    if len(visited) != len(loader.dataset):
        raise RuntimeError("Incomplete centered validation coverage")
    return {**{name: value / len(visited) for name, value in totals.items()},
            "frames": len(visited), "seconds": time.monotonic() - started}


def save_checkpoint(path: Path, decoder: HumanGeometryQueryTransformer,
                    optimizer: torch.optim.Optimizer, scheduler: Any, epoch: int,
                    best_val: float, validation: dict[str, float],
                    args: argparse.Namespace, train: TemporalClipDataset,
                    val_signature: dict[str, Any], initialization: dict[str, Any]) -> None:
    payload = {
        "format": FORMAT, "epoch": epoch, "best_val": best_val,
        "checkpoint_selection": (
            "development subset centered-frame MPJPE" if args.smoke_max_steps
            else "minimum full-validation centered-frame MPJPE"
        ),
        "selection_metric": "mpjpe_mm",
        "decoder_config": vars(decoder.config), "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "validation": validation, "args": json_ready(vars(args)),
        "train_cache_signature": train.signature, "val_cache_signature": val_signature,
        "train_clip_stats": train.stats, "initialization": initialization,
        "target_coordinate_system": "camera", "pose_protocol": POSE_PROTOCOL,
        "development_only": bool(args.smoke_max_steps),
        "test_used_for_model_selection": False,
        "test_set_previously_observed": True,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if min(args.epochs, args.batch_size, args.val_batch_size, args.log_interval) <= 0:
        raise ValueError("Epochs, batch sizes and log interval must be positive")
    if args.num_workers < 0 or args.smoke_max_steps < 0 or args.lr <= 0 or args.grad_clip <= 0:
        raise ValueError("Invalid worker/step/optimizer configuration")
    weights = [value for key, value in vars(args).items() if key.endswith("_weight")]
    if not all(math.isfinite(value) and value >= 0 for value in weights) or args.joint_weight == 0:
        raise ValueError("Loss weights must be non-negative; joint weight must be positive")
    protected = ("best.pt", "latest.pt", "metrics.jsonl", "run_config.json")
    if any((args.output_dir / name).exists() for name in protected):
        raise FileExistsError("Refusing to overwrite an existing geometry-v3 experiment")
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    source = load_payload(args.init_checkpoint)
    if (source.get("format") != "divibody-real-3dpw-temporal-camera-v2"
            or source.get("target_coordinate_system") != "camera"):
        raise ValueError("Geometry v3 requires a camera-v2 initialization, not a world model")
    if int(source["args"]["clip_length"]) != args.clip_length:
        raise ValueError("Initialization clip length differs")
    train = TemporalClipDataset(args.train_cache, "train", args.clip_length,
                                args.train_clip_stride)
    train_payload = {"features": train.features, "metadata": train.metadata,
                     "signature": train.signature, "targets": train.targets}
    val_payload = load_payload(args.val_cache)
    validate_feature_compatibility(source, train_payload)
    validate_feature_compatibility(source, val_payload)
    validate_cache_and_alignment(val_payload, args.data_root, "validation")
    train_targets = FullGeometryTargets(train_payload, args.data_root, "train")
    val_targets = FullGeometryTargets(val_payload, args.data_root, "validation")
    overlap = sorted({key[0] for key in train_targets.keys} & {key[0] for key in val_targets.keys})
    if overlap:
        raise ValueError(f"Train/validation sequence leakage: {overlap}")
    train_dataset = GeometryClipDataset(train, train_targets)
    val_dataset: Dataset = CenteredFrameClipDataset(val_payload, args.clip_length)
    if args.smoke_max_steps:
        count = min(len(val_dataset), args.val_batch_size * args.smoke_max_steps)
        selection = np.linspace(0, len(val_dataset) - 1, count, dtype=int).tolist()
        val_dataset = Subset(val_dataset, selection)
    loader_options = {"num_workers": args.num_workers, "pin_memory": device.type == "cuda",
                      "persistent_workers": args.num_workers > 0}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              **loader_options)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False,
                            **loader_options)

    config = GeometryDecoderConfig(**source["decoder_config"])
    decoder = HumanGeometryQueryTransformer(config)
    loaded = decoder.load_camera_v2_weights(source["decoder"])
    # Exact warm-start geometry: allow the spatial scale to learn away from zero.
    with torch.no_grad():
        decoder.spatial_position_scale.zero_()
    decoder = decoder.to(device)
    for name in ("left_hand_head", "right_hand_head", "face_head", "object_head"):
        getattr(decoder, name).requires_grad_(False)
    initialization = {"path": str(args.init_checkpoint.resolve()),
                      "sha256": sha256_file(args.init_checkpoint),
                      "epoch": int(source["epoch"]), "format": source["format"],
                      "weights": loaded, "spatial_position_scale_initial": 0.0,
                      "new_rotation_head_initialization": "exact identity residual"}
    del source
    models = create_smpl_models(args.smpl_model_root, device)
    for model in models.values():
        model.requires_grad_(False)
    regressor_np = np.load(args.joint_regressor)
    if regressor_np.shape != (17, 6890) or not np.isfinite(regressor_np).all():
        raise ValueError("Invalid H36M joint regressor")
    regressor = torch.as_tensor(regressor_np, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        (p for p in decoder.parameters() if p.requires_grad),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "format": FORMAT, "status": "training_started", "args": json_ready(vars(args)),
        "decoder_config": vars(config), "initialization": initialization,
        "train_clip_stats": train.stats, "train_data_audit": train_targets.audit,
        "validation_data_audit": val_targets.audit, "sequence_overlap": overlap,
        "pose_protocol": POSE_PROTOCOL,
        "selection_metric": "mpjpe_mm",
        "checkpoint_selection": (
            "development subset centered-frame MPJPE" if args.smoke_max_steps
            else "minimum full validation MPJPE; one centered prediction per frame"
        ),
        "test_data_used_in_training_or_selection": False,
        "test_set_previously_observed": True,
        "development_only": bool(args.smoke_max_steps),
        "loss_units": "joints/vertices in metres, velocity m/frame, acceleration m/frame^2",
        "supervision": "neutral predicted vs gendered full-24-joint camera SMPL",
        "parameter_count": sum(p.numel() for p in decoder.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in decoder.parameters() if p.requires_grad),
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(run_config, ensure_ascii=False), flush=True)
    baseline = validate(val_loader, decoder, val_targets, models, regressor, args, device)
    best_value, best_epoch, best_metrics = baseline["mpjpe_mm"], 0, baseline
    record = {"epoch": 0, "phase": "exact_camera_v2_warm_start", "validation": baseline}
    print(json.dumps(record), flush=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    with metrics_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
    save_checkpoint(args.output_dir / "best.pt", decoder, optimizer, scheduler, 0,
                    best_value, baseline, args, train, val_payload["signature"], initialization)
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(train_loader, decoder, optimizer, models, regressor,
                                     args, device, epoch)
        validation = validate(val_loader, decoder, val_targets, models, regressor, args, device)
        improved = validation["mpjpe_mm"] < best_value
        if improved:
            best_value, best_epoch, best_metrics = validation["mpjpe_mm"], epoch, validation
        scheduler.step()
        record = {"epoch": epoch, "train": train_metrics, "validation": validation,
                  "best_epoch": best_epoch, "best_validation_mpjpe_mm": best_value,
                  "next_lr": optimizer.param_groups[0]["lr"],
                  "spatial_position_scale": float(decoder.spatial_position_scale.detach())}
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        for name in ("latest.pt",) + (("best.pt",) if improved else ()):
            save_checkpoint(args.output_dir / name, decoder, optimizer, scheduler, epoch,
                            best_value, validation, args, train, val_payload["signature"],
                            initialization)
        print(json.dumps(record), flush=True)
    result = {"status": "DIVIBODY_GEOMETRY_3DPW_TRAINING_OK", "completed_epochs": args.epochs,
              "best_epoch": best_epoch, "best_validation": best_metrics,
              "initialization_validation": baseline, "development_only": bool(args.smoke_max_steps),
              "best_checkpoint": str((args.output_dir / "best.pt").resolve())}
    (args.output_dir / "final_validation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
