"""Geometry-priority v9 with isolated PoseTrack camera gradients.

Select geometry epochs by validation MPJPE under PVE and calibrated PCK guards.
Optionally adapt the last VACE fusion block. Existing sources stay unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from divibody_training.evaluate_standard_3dpw import create_smpl_models
from divibody_training.train_divibody_rgb_vace_hmr2a_v8 import V8Dataset, load_pair, validate
from divibody_training.train_divibody_rgb_vace_hmr2a_v9_small import spread_subset
from divibody_training.train_divibody_rgb_vace_v7 import forward_batch, geometric_losses
from divibody_training.train_ram_joint_sharded_pose_geometry_v7 import pose_losses
from divibody_training.train_ram_mixed_3dpw_posetrack_v1 import load
from divibody_training.train_ram_posetrack_camera_v4 import evaluate_pose
from divibody_training.train_ram_posetrack_sharded_camera_v6 import ShardedPoseDataset
from models.human_decoder_rgb_vace_v9_isolated import HumanRGBVaceGeometryTransformerV9Isolated


FORMAT = "ram-geometry-priority-isolated-v9"
CAMERA_PREFIXES = ("camera_residual_head.",)
GEOMETRY_PREFIXES = (
    "rotation_residual_head.",
    "shape_residual_head.",
    "temporal_gates",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--unfreeze-last-vace', action='store_true')
    p.add_argument('--fusion-lr', type=float, default=1e-7)
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
    p.add_argument("--camera-epochs", type=int, default=2)
    p.add_argument("--geometry-epochs", type=int, default=2)
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
    p.add_argument("--anchor-weight", type=float, default=0.2)
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
    if min(args.camera_lr, args.geometry_lr, args.fusion_lr) <= 0:
        p.error('Learning rates must be positive')
    if args.pose_pck_tolerance < 0 or args.geometry_regression_tolerance_mm < 0:
        p.error('Tolerances must be nonnegative')
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
    # Only explicitly trainable fusion blocks use training-mode dropout.
    for block in model.vace_blocks:
        if any(p.requires_grad for p in block.parameters()):
            block.train()


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


class TakeIterableDataset(IterableDataset):
    """Limit a streaming dataset without converting it to an invalid Subset."""

    def __init__(self, dataset, maximum):
        self.dataset = dataset
        self.maximum = min(len(dataset), maximum) if maximum > 0 else len(dataset)

    def __len__(self):
        return self.maximum

    def __iter__(self):
        for index, item in enumerate(self.dataset):
            if index >= self.maximum:
                break
            yield item


def make_optimizer(model, args, include_geometry):
    fusion_prefix = (f'vace_blocks.{len(model.vace_blocks) - 1}.',) if (
        include_geometry and args.unfreeze_last_vace) else ()
    camera = select_parameters(
        model, CAMERA_PREFIXES + (GEOMETRY_PREFIXES if include_geometry else ()) + fusion_prefix,
    )
    camera_values = [value for name, value in camera if name.startswith(CAMERA_PREFIXES)]
    groups = [{"params": camera_values, "lr": args.camera_lr, "group_name": "camera"}]
    if include_geometry:
        geometry_values = [value for name, value in camera if name.startswith(GEOMETRY_PREFIXES)]
        groups.append({"params": geometry_values, "lr": args.geometry_lr,
                       "group_name": "geometry"})
    if fusion_prefix:
        groups.append({'params': [p for n, p in camera if n.startswith(fusion_prefix)],
                       'lr': args.fusion_lr, 'group_name': 'fusion'})
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
        "optimizer": optimizer.state_dict(),
        "test_used_for_model_selection": False,
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


def train_camera_epoch(model, loader, official_smpl, optimizer, args, device):
    model.eval()
    totals = {}
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        prediction = camera_prediction(model, batch, args, device)
        losses = pose_losses(prediction, batch, official_smpl, args, device)
        backward_checked(losses["pose_total"])
        torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group["params"]], args.grad_clip,
            error_if_nonfinite=True,
        )
        optimizer.step()
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach())
    return {name: value / len(loader) for name, value in totals.items()}


def train_geometry_epoch(model, train_loader, pose_loader, smpl_models, official_smpl,
                         regressor, optimizer, args, device):
    stable_train_mode(model)
    totals = {}
    pose_batches = 0
    pose_iterator = iter(pose_loader)
    for step, batch3d in enumerate(train_loader):
        stable_train_mode(model)
        optimizer.zero_grad(set_to_none=True)
        prediction = forward_batch(model, batch3d, device, args.amp_dtype)
        prediction['camera'] = prediction['camera'].detach()
        losses3d = geometric_losses(prediction, batch3d, smpl_models, regressor, args, device)
        total = losses3d["total"]
        values = {f"3dpw_{name}": value for name, value in losses3d.items()}
        if step % args.pose_every_steps == 0:
            try:
                batch_pose = next(pose_iterator)
            except StopIteration:
                pose_iterator = iter(pose_loader)
                batch_pose = next(pose_iterator)
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
        raise FileExistsError("Refusing to overwrite a nonempty v9 run")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_vace, train_hmr = load_pair(args.train_vace_cache, args.train_hmr_sidecar, "train")
    val_vace, val_hmr = load_pair(args.val_vace_cache, args.val_hmr_sidecar, "validation")
    train3d = spread_subset(V8Dataset(train_vace, train_hmr), args.max_3dpw_windows)
    val3d = spread_subset(V8Dataset(val_vace, val_hmr), args.max_val_windows)
    pose = TakeIterableDataset(
        ShardedPoseDataset(args.pose_train_vace_dir, args.pose_train_hmr_dir),
        args.max_pose_train_windows,
    )
    pose_val = TakeIterableDataset(
        ShardedPoseDataset(args.pose_val_vace_dir, args.pose_val_hmr_dir),
        args.max_pose_val_windows,
    )

    model = HumanRGBVaceGeometryTransformerV9Isolated().to(device)
    model.load_state_dict(load(args.init_checkpoint)["model"], strict=True)
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
        "stage_2": "camera plus low-LR pose/shape/temporal parameters with rollback",
        "3dpw_train_windows": len(train3d),
        "3dpw_validation_windows": len(val3d),
        "posetrack_train_windows": len(pose),
        "posetrack_validation_windows": len(pose_val),
        "existing_model_sources_modified": False,
        "selection": "minimum validation MPJPE with calibrated PCK guards",
        "pose_pck_tolerance": args.pose_pck_tolerance,
        "unfreeze_last_vace": args.unfreeze_last_vace,
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
            model, pose_loader, official_smpl, optimizer, args, device,
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
            "accepted": accepted, "accepted_score": score,
            "next_camera_lr": args.camera_lr,
        })

    model.load_state_dict(load(args.output_dir / "best.pt")["model"], strict=True)
    # Guard geometry fine-tuning against the accepted calibrated 2D baseline.
    calibrated = load(args.output_dir / 'best.pt')
    baseline_pose = calibrated['pose_validation']
    baseline_geometry = calibrated['validation']
    best_score = -baseline_geometry['mpjpe_mm']
    optimizer, trainable = make_optimizer(model, args, include_geometry=True)
    for _ in range(args.geometry_epochs):
        epoch += 1
        train_metrics, pose_batches = train_geometry_epoch(
            model, train_loader, pose_loader, smpl_models, official_smpl,
            regressor, optimizer, args, device,
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
            best_score = score
            save(args.output_dir / "best.pt", model, optimizer, epoch, "guarded_geometry",
                 geometry, pose_metrics, args)
        else:
            model.load_state_dict(load(args.output_dir / "best.pt")["model"], strict=True)
            args.geometry_lr *= args.rejected_lr_decay
            args.fusion_lr *= args.rejected_lr_decay
            optimizer, trainable = make_optimizer(model, args, include_geometry=True)
        append_record(metrics_path, {
            "epoch": epoch, "stage": "guarded_geometry", "train": train_metrics,
            "pose_batches": pose_batches, "validation": geometry,
            "pose_validation": pose_metrics, "accepted": accepted,
            "accepted_score": score, "next_geometry_lr": args.geometry_lr,
        })


if __name__ == "__main__":
    main()
