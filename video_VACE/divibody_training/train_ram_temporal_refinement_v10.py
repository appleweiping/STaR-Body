"""RGB/SMPL temporal refinement copied from the isolated v9 training path.

Select geometry epochs by validation MPJPE under PVE and calibrated PCK guards.
Train a new temporal branch and sample PoseTrack without epoch-prefix repeats.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from divibody_training.evaluate_standard_3dpw import create_smpl_models
from divibody_training.train_divibody_rgb_vace_hmr2a_v8 import load_pair
from divibody_training.train_divibody_rgb_vace_hmr2a_v9_small import spread_subset
from divibody_training.train_ram_mixed_3dpw_posetrack_v1 import load
from divibody_training.ram_temporal_runtime_v10 import (
    V10Dataset, forward_batch, validate, evaluate_pose, pose_losses, model_from_payload,
    geometric_losses,
)
from divibody_training.ram_pose_stream_v10 import ShuffledPoseDatasetV10, CyclingPoseStream
from models.human_decoder_rgb_vace_v10_temporal import TemporalRefinerConfig


FORMAT = "ram-rgb-smpl-temporal-v10"
CAMERA_PREFIXES = ("camera_residual_head.",)
GEOMETRY_PREFIXES = (
    "rotation_residual_head.",
    "shape_residual_head.",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--unfreeze-geometry', action='store_true')
    p.add_argument('--temporal-lr', type=float, default=1e-4)
    p.add_argument('--temporal-hidden-dim', type=int, default=256)
    p.add_argument('--temporal-heads', type=int, default=8)
    p.add_argument('--temporal-layers', type=int, default=2)
    p.add_argument('--temporal-dropout', type=float, default=0.1)
    p.add_argument('--temporal-anchor-weight', type=float, default=0.02)
    p.add_argument('--warmup-epochs', type=int, default=1)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--allow-partial-validation', action='store_true',
                   help='Development only: allow fewer than 12 validation sequences')
    p.add_argument('--pose-pck-tolerance', type=float, default=0.005)
    p.add_argument("--train-vace-cache", type=Path, required=True)
    p.add_argument("--train-hmr-sidecar", type=Path, required=True)
    p.add_argument("--val-vace-cache", type=Path, required=True)
    p.add_argument("--val-hmr-sidecar", type=Path, required=True)
    p.add_argument("--pose-train-vace-dir", type=Path, required=True)
    p.add_argument("--pose-train-hmr-dir", type=Path, required=True)
    p.add_argument("--pose-val-vace-dir", type=Path, required=True)
    p.add_argument("--pose-val-hmr-dir", type=Path, required=True)
    p.add_argument("--init-checkpoint", type=Path, required=True)
    p.add_argument("--smpl-model-root", type=Path, required=True)
    p.add_argument("--joint-regressor", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--camera-epochs", type=int, default=0)
    p.add_argument("--geometry-epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--val-batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-3dpw-windows", type=int, default=0)
    p.add_argument("--max-val-windows", type=int, default=0)
    p.add_argument("--max-pose-train-windows", type=int, default=0)
    p.add_argument("--max-pose-val-windows", type=int, default=0)
    p.add_argument("--camera-lr", type=float, default=2e-6)
    p.add_argument("--geometry-lr", type=float, default=2e-7)
    p.add_argument("--rejected-lr-decay", type=float, default=0.25)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--pose-every-steps", type=int, default=8)
    p.add_argument("--joint-weight", type=float, default=1.0)
    p.add_argument("--vertex-weight", type=float, default=0.5)
    p.add_argument("--rotation-weight", type=float, default=0.05)
    p.add_argument("--shape-weight", type=float, default=0.001)
    p.add_argument("--velocity-weight", type=float, default=0.5)
    p.add_argument("--acceleration-weight", type=float, default=0.25)
    p.add_argument("--anchor-weight", type=float, default=0.0)
    p.add_argument("--pose-2d-weight", type=float, default=1.0)
    p.add_argument("--pose-acceleration-weight", type=float, default=0.05)
    p.add_argument("--pose-anchor-weight", type=float, default=0.25)
    p.add_argument("--geometry-regression-tolerance-mm", type=float, default=0.5)
    p.add_argument("--amp-dtype", choices=("bf16", "none"), default="bf16")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    if args.pose_every_steps < 1 or args.camera_epochs < 0 or args.geometry_epochs < 0:
        p.error('Invalid epoch count or PoseTrack interval')
    if min(args.camera_lr, args.geometry_lr, args.temporal_lr) <= 0:
        p.error('Learning rates must be positive')
    if args.pose_pck_tolerance < 0 or args.geometry_regression_tolerance_mm < 0:
        p.error('Tolerances must be nonnegative')
    if args.patience < 1 or args.warmup_epochs < 0 or args.temporal_anchor_weight < 0:
        p.error('Invalid patience, warmup or temporal anchor weight')
    if args.batch_size < 1 or args.val_batch_size < 1 or args.num_workers < 0:
        p.error('Invalid batch size or worker count')
    if not 0 < args.rejected_lr_decay <= 1:
        p.error('Camera rejected-lr-decay must be in (0, 1]')
    if min(args.max_3dpw_windows, args.max_val_windows, args.max_pose_train_windows,
           args.max_pose_val_windows) < 0:
        p.error('Window limits must be nonnegative')
    return args


def selection_score(pose, geometry, baseline_pose, baseline_geometry, tolerance,
                    pose_tolerance=0.005):
    values = [*geometry.values(), *pose.values()]
    if not all(np.isfinite(value) for value in values):
        raise FloatingPointError('Nonfinite validation metrics')
    if any(geometry[k] > baseline_geometry[k] + tolerance for k in ('mpjpe_mm', 'pve_mm')):
        return float('-inf')
    if any(pose[k] < baseline_pose[k] - pose_tolerance for k in ('PCKn@0.05', 'PCKn@0.10')):
        return float('-inf')
    return -geometry['mpjpe_mm']


def camera_prediction(model, batch, args, device):
    # Deterministic frozen feature extraction avoids dropout-induced 3D drift.
    model.eval()
    with torch.no_grad():
        prediction = forward_batch(model, batch, device, args.amp_dtype)
    return model.camera_only_prediction(prediction, batch['base_camera'].to(device))


def stable_train_mode(model):
    model.eval()
    if any(p.requires_grad for p in model.temporal_refiner.parameters()):
        model.temporal_refiner.train()


def backward_checked(loss):
    if not torch.isfinite(loss).all():
        raise FloatingPointError('Nonfinite training loss')
    loss.backward()


def select_parameters(model, prefixes):
    selected = []
    for name, parameter in model.named_parameters():
        enabled = any(name.startswith(prefix) for prefix in prefixes)
        parameter.requires_grad_(enabled)
        if enabled:
            selected.append((name, parameter))
    if not selected:
        raise RuntimeError(f"No parameters matched prefixes: {prefixes}")
    return selected


def make_optimizer(model, args, include_geometry):
    prefixes = CAMERA_PREFIXES
    if include_geometry:
        prefixes += ('temporal_refiner.',)
        if args.unfreeze_geometry:
            prefixes += GEOMETRY_PREFIXES
    camera = select_parameters(model, prefixes)
    camera_values = [value for name, value in camera if name.startswith(CAMERA_PREFIXES)]
    groups = [{"params": camera_values, "lr": args.camera_lr, "group_name": "camera"}]
    if include_geometry and args.unfreeze_geometry:
        geometry_values = [value for name, value in camera if name.startswith(GEOMETRY_PREFIXES)]
        groups.append({"params": geometry_values, "lr": args.geometry_lr,
                       "group_name": "geometry"})
    if include_geometry:
        groups.append({'params': [p for n, p in camera if n.startswith('temporal_refiner.')],
                       'lr': args.temporal_lr, 'group_name': 'temporal'})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay), [name for name, _ in camera]


def save(path, model, optimizer, epoch, stage, validation, pose_validation, args):
    payload = {
        "format": FORMAT,
        "epoch": epoch,
        "stage": stage,
        "validation": validation,
        "pose_validation": pose_validation,
        "selection_metric": "minimum validation MPJPE with PVE and both PCK guards",
        "model": model.state_dict(),
        "model_config": vars(model.config),
        "temporal_config": asdict(model.temporal_config),
        "optimizer": optimizer.state_dict(),
        "trainable_parameters": {n: p.numel() for n, p in model.named_parameters()
                                 if p.requires_grad},
        "test_used_for_model_selection": False,
        "development_only": getattr(args, 'development_only', True),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def append_record(path, record):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def evaluate(model, val_loader, pose_val_loader, smpl_models, official_smpl,
             regressor, args, device):
    geometry = validate(val_loader, model, smpl_models, regressor, args, device)
    pose = evaluate_pose(pose_val_loader, model, official_smpl, args, device)
    return geometry, pose


def train_camera_epoch(model, loader, stream, official_smpl, optimizer, args, device):
    model.eval()
    totals = {}
    for _ in range(len(loader)):
        batch = stream.next_batch()
        optimizer.zero_grad(set_to_none=True)
        prediction = camera_prediction(model, batch, args, device)
        losses = pose_losses(prediction, batch, official_smpl, args, device)
        backward_checked(losses["pose_total"])
        torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group["params"]], args.grad_clip,
            error_if_nonfinite=True,
        )
        optimizer.step()
        stream.record_consumed(batch)
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach())
    return {name: value / len(loader) for name, value in totals.items()}


def train_geometry_epoch(model, train_loader, pose_stream, smpl_models, official_smpl,
                         regressor, optimizer, args, device, geometry_epoch=0):
    stable_train_mode(model)
    totals = {}
    pose_batches = 0
    for step, batch3d in enumerate(train_loader):
        progress = geometry_epoch + (step + 1) / len(train_loader)
        warmup = min(1.0, progress / max(args.warmup_epochs, 1e-9))
        for group in optimizer.param_groups:
            if group['group_name'] == 'temporal':
                group['lr'] = args.temporal_lr * warmup
        stable_train_mode(model)
        optimizer.zero_grad(set_to_none=True)
        prediction = forward_batch(model, batch3d, device, args.amp_dtype)
        prediction['camera'] = prediction['camera'].detach()
        losses3d = geometric_losses(prediction, batch3d, smpl_models, regressor, args, device)
        valid = batch3d['target_valid'].to(device)
        rotation_delta = prediction['temporal_rotation_delta_6d'].float()
        beta_delta = prediction['temporal_beta_delta'].float()
        temporal_anchor = (rotation_delta.square().mean(dim=(-1, -2))[valid].mean()
                           + beta_delta.square().mean(dim=-1)[valid].mean())
        total = losses3d["total"] + args.temporal_anchor_weight * temporal_anchor
        values = {f"3dpw_{name}": value for name, value in losses3d.items()}
        values['temporal_anchor'] = temporal_anchor
        batch_pose = None
        if step % args.pose_every_steps == 0:
            batch_pose = pose_stream.next_batch()
            prediction_pose = camera_prediction(model, batch_pose, args, device)
            losses_pose = pose_losses(prediction_pose, batch_pose, official_smpl, args, device)
            total = total + losses_pose["pose_total"]
            values.update(losses_pose)
            pose_batches += 1
        backward_checked(total)
        torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group["params"]], args.grad_clip,
            error_if_nonfinite=True,
        )
        optimizer.step()
        if batch_pose is not None:
            pose_stream.record_consumed(batch_pose)
        values["mixed_total"] = total
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach())
    return {
        name: value / (pose_batches if name.startswith("pose_") else len(train_loader))
        for name, value in totals.items()
    }, pose_batches


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite a nonempty v10 run")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_vace, train_hmr = load_pair(args.train_vace_cache, args.train_hmr_sidecar, "train")
    val_vace, val_hmr = load_pair(args.val_vace_cache, args.val_hmr_sidecar, "validation")
    train_sequences = {m['sequence'] for m in train_vace['metadata']}
    val_sequences = {m['sequence'] for m in val_vace['metadata']}
    if train_sequences & val_sequences:
        raise ValueError('Training and validation sequence identities overlap')
    if len(val_sequences) != 12 and not args.allow_partial_validation:
        raise ValueError(f'Expected 12 validation sequences, found {len(val_sequences)}. '
                         'Rebuild the complete validation cache; use '
                         '--allow-partial-validation only for development checks.')
    args.development_only = len(val_sequences) != 12 or any((args.max_3dpw_windows,
        args.max_val_windows, args.max_pose_train_windows, args.max_pose_val_windows))
    train3d = spread_subset(V10Dataset(train_vace, train_hmr), args.max_3dpw_windows)
    val3d = spread_subset(V10Dataset(val_vace, val_hmr), args.max_val_windows)
    pose = ShuffledPoseDatasetV10(args.pose_train_vace_dir, args.pose_train_hmr_dir,
                                 seed=args.seed, max_windows=args.max_pose_train_windows)
    pose_val = ShuffledPoseDatasetV10(args.pose_val_vace_dir, args.pose_val_hmr_dir,
                                     seed=args.seed, max_windows=args.max_pose_val_windows)
    if pose.collection_id == pose_val.collection_id or (
            args.pose_train_vace_dir.resolve() == args.pose_val_vace_dir.resolve()):
        raise ValueError('PoseTrack training and validation collections must be distinct')
    if pose.split != 'train' or pose_val.split not in ('val', 'validation'):
        raise ValueError('PoseTrack collection split labels are missing or incorrect')

    temporal_config = TemporalRefinerConfig(
        hidden_dim=args.temporal_hidden_dim, num_heads=args.temporal_heads,
        num_layers=args.temporal_layers, dropout=args.temporal_dropout)
    model = model_from_payload(load(args.init_checkpoint), temporal_config).to(device)
    smpl_models = create_smpl_models(args.smpl_model_root, device)
    for value in smpl_models.values():
        value.requires_grad_(False)
    from hmr2.models.smpl_wrapper import SMPL as OfficialSMPL
    official_smpl = OfficialSMPL(
        model_path=str(args.smpl_model_root), gender="neutral", num_betas=10,
        joint_regressor_extra=str(args.smpl_model_root.parent / "SMPL_to_J19.pkl"),
    ).to(device)
    official_smpl.requires_grad_(False)
    regressor = torch.as_tensor(np.load(args.joint_regressor), dtype=torch.float32, device=device)

    options = {"num_workers": args.num_workers, "pin_memory": True,
               "persistent_workers": args.num_workers > 0}
    train_loader = DataLoader(train3d, batch_size=args.batch_size, shuffle=True, **options)
    val_loader = DataLoader(val3d, batch_size=args.val_batch_size, shuffle=False, **options)
    pose_loader = DataLoader(pose, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)
    pose_val_loader = DataLoader(pose_val, batch_size=args.val_batch_size, shuffle=False,
                                 num_workers=0, pin_memory=True)
    pose_stream = CyclingPoseStream(pose_loader)

    args.output_dir.mkdir(parents=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    baseline_geometry, baseline_pose = evaluate(
        model, val_loader, pose_val_loader, smpl_models, official_smpl,
        regressor, args, device,
    )
    optimizer, trainable = make_optimizer(model, args, include_geometry=False)
    best_score = selection_score(
        baseline_pose, baseline_geometry, baseline_pose, baseline_geometry,
        args.geometry_regression_tolerance_mm, args.pose_pck_tolerance,
    )
    save(args.output_dir / "best.pt", model, optimizer, 0, "anchor",
         baseline_geometry, baseline_pose, args)
    manifest = {
        "format": FORMAT,
        "initial_checkpoint": str(args.init_checkpoint.resolve()),
        "stage_1": "camera residual head only",
        "stage_2": "new RGB/SMPL temporal refinement with validation selection and early stopping",
        "3dpw_train_windows": len(train3d),
        "3dpw_validation_windows": len(val3d),
        "posetrack_train_windows": len(pose),
        "posetrack_validation_windows": len(pose_val),
        "existing_model_sources_modified": False,
        "selection": "minimum validation MPJPE with calibrated PCK guards",
        "pose_pck_tolerance": args.pose_pck_tolerance,
        "unfreeze_geometry": args.unfreeze_geometry,
        "temporal_config": asdict(temporal_config),
        "validation_sequence_names": sorted(val_sequences),
        "validation_sequence_count": len(val_sequences),
        "development_only": len(val_sequences) != 12 or any((args.max_3dpw_windows,
            args.max_val_windows, args.max_pose_train_windows, args.max_pose_val_windows)),
        "sampling": "shuffled stream preserved across epochs; actual consumed UID coverage",
        "posetrack_gradient_route": "camera head only; shared features detached",
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    append_record(metrics_path, {
        "epoch": 0, "stage": "anchor", "validation": baseline_geometry,
        "pose_validation": baseline_pose, "accepted": True,
    })

    epoch = 0
    for _ in range(args.camera_epochs):
        epoch += 1
        train_metrics = train_camera_epoch(
            model, pose_loader, pose_stream, official_smpl, optimizer, args, device,
        )
        geometry, pose_metrics = evaluate(
            model, val_loader, pose_val_loader, smpl_models, official_smpl,
            regressor, args, device,
        )
        score = selection_score(
            pose_metrics, geometry, baseline_pose, baseline_geometry,
            args.geometry_regression_tolerance_mm, args.pose_pck_tolerance,
        )
        # A camera-only epoch cannot improve pelvis-aligned 3D metrics.
        old_pose = load(args.output_dir / 'best.pt')['pose_validation']
        improves_pose = (pose_metrics['PCKn@0.05'] + pose_metrics['PCKn@0.10'] >
                         old_pose['PCKn@0.05'] + old_pose['PCKn@0.10'])
        accepted = np.isfinite(score) and score >= best_score and improves_pose
        save(args.output_dir / "latest.pt", model, optimizer, epoch, "camera_only",
             geometry, pose_metrics, args)
        if accepted:
            best_score = score
            save(args.output_dir / "best.pt", model, optimizer, epoch, "camera_only",
                 geometry, pose_metrics, args)
        else:
            model.load_state_dict(load(args.output_dir / "best.pt")["model"], strict=True)
            args.camera_lr *= args.rejected_lr_decay
            optimizer, trainable = make_optimizer(model, args, include_geometry=False)
        append_record(metrics_path, {
            "epoch": epoch, "stage": "camera_only", "train": train_metrics,
            "validation": geometry, "pose_validation": pose_metrics,
            "accepted": accepted, "accepted_score": score if np.isfinite(score) else None,
            "next_camera_lr": args.camera_lr,
            "pose_coverage": pose_stream.coverage(),
        })

    model.load_state_dict(load(args.output_dir / "best.pt")["model"], strict=True)
    # Guard geometry fine-tuning against the accepted calibrated 2D baseline.
    calibrated = load(args.output_dir / 'best.pt')
    baseline_pose = calibrated['pose_validation']
    baseline_geometry = calibrated['validation']
    best_score = -baseline_geometry['mpjpe_mm']
    optimizer, trainable = make_optimizer(model, args, include_geometry=True)
    stale_epochs = 0
    for geometry_epoch in range(args.geometry_epochs):
        epoch += 1
        train_metrics, pose_batches = train_geometry_epoch(
            model, train_loader, pose_stream, smpl_models, official_smpl,
            regressor, optimizer, args, device, geometry_epoch,
        )
        geometry, pose_metrics = evaluate(
            model, val_loader, pose_val_loader, smpl_models, official_smpl,
            regressor, args, device,
        )
        score = selection_score(
            pose_metrics, geometry, baseline_pose, baseline_geometry,
            args.geometry_regression_tolerance_mm, args.pose_pck_tolerance,
        )
        accepted = score > best_score
        save(args.output_dir / "latest.pt", model, optimizer, epoch, "guarded_geometry",
             geometry, pose_metrics, args)
        if accepted:
            stale_epochs = 0
            best_score = score
            save(args.output_dir / "best.pt", model, optimizer, epoch, "guarded_geometry",
                 geometry, pose_metrics, args)
        else:
            stale_epochs += 1
        append_record(metrics_path, {
            "epoch": epoch, "stage": "guarded_geometry", "train": train_metrics,
            "pose_batches": pose_batches, "validation": geometry,
            "pose_validation": pose_metrics, "accepted": accepted,
            "accepted_score": score if np.isfinite(score) else None,
            "next_geometry_lr": args.geometry_lr,
            "temporal_lr": next(g['lr'] for g in optimizer.param_groups
                                if g['group_name'] == 'temporal'),
            "pose_coverage": pose_stream.coverage(),
            "stale_epochs": stale_epochs,
        })
        if stale_epochs >= args.patience:
            print(json.dumps({'early_stop': True, 'epoch': epoch}), flush=True)
            break


if __name__ == "__main__":
    main()
