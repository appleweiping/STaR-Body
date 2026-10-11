"""Train spatial RGB part queries from v9 with optional visible 2D supervision."""

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
from divibody_training.ram_parts_runtime_v16 import (
    V10Dataset, forward_batch, validate, evaluate_pose, pose_losses, model_from_payload,
    geometric_losses, geometry_distance,
)
from divibody_training.ram_pose_stream_v10 import ShuffledPoseDatasetV10, CyclingPoseStream
from models.human_decoder_parts_v16 import AdaptationConfig, FORMAT


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('train-vace-cache', 'train-hmr-sidecar', 'val-vace-cache',
                 'val-hmr-sidecar', 'pose-val-vace-dir', 'pose-val-hmr-dir',
                 'init-checkpoint', 'smpl-model-root', 'joint-regressor', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    for name, default in dict(geometry_epochs=15, batch_size=4, val_batch_size=16,
            num_workers=2, max_3dpw_windows=0, max_val_windows=0,
            max_pose_val_windows=0, patience=5, seed=2026).items():
        p.add_argument('--' + name.replace('_', '-'), type=int, default=default)
    for name, default in dict(geometry_lr=1e-5, weight_decay=1e-4, grad_clip=1.,
            teacher_weight=0.05, position_consistency_weight=0.,
            position_warmup_epochs=2., geometry_regression_tolerance_mm=0.5,
            pose_pck_tolerance=0.005).items():
        p.add_argument('--' + name.replace('_', '-'), type=float, default=default)
    p.add_argument('--position-mode', choices=('original', 'shared'), default='original')
    p.add_argument('--motion-loss', choices=('on', 'off'), default='on')
    p.add_argument('--amp-dtype', choices=('bf16', 'none'), default='bf16')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--allow-partial-validation', action='store_true')
    p.add_argument('--part-hidden-dim',type=int,default=128)
    p.add_argument('--max-angle-degrees',type=float,default=10.)
    p.add_argument('--pose-train-vace-dir',type=Path)
    p.add_argument('--pose-train-hmr-dir',type=Path)
    p.add_argument('--pose-2d-weight',type=float,default=0.)
    p.add_argument('--pose-deadzone',type=float,default=.01)
    p.add_argument('--pose-every-steps',type=int,default=1)
    p.add_argument('--pose-warmup-epochs',type=float,default=1.)
    p.add_argument('--max-pose-train-windows',type=int,default=0)
    args = p.parse_args()
    if args.position_mode != 'original' or args.position_consistency_weight:
        p.error('v16 isolates spatial parts; slot mixing/consistency must remain disabled')
    if args.pose_2d_weight and (args.pose_train_vace_dir is None or args.pose_train_hmr_dir is None):
        p.error('Positive 2D weight requires training PoseTrack paths')
    if args.pose_every_steps < 1 or args.pose_warmup_epochs <= 0:
        p.error('Pose interval and warmup must be positive')
    for name, value in vars(args).items():
        if isinstance(value, (int, float)) and (not np.isfinite(value) or value < 0):
            p.error(f'{name} must be finite and nonnegative')
    if min(args.geometry_lr, args.batch_size, args.val_batch_size,
           args.patience, args.grad_clip, args.position_warmup_epochs) <= 0:
        p.error('Learning rate, batches, patience, clipping and warmup must be positive')
    args.joint_weight, args.vertex_weight = 1., 0.5
    args.rotation_weight, args.shape_weight = 0.05, 0.001
    args.velocity_weight = 0.5 if args.motion_loss == 'on' else 0.
    args.acceleration_weight = 0.25 if args.motion_loss == 'on' else 0.
    args.anchor_weight = 0.  # Replaced by the trained-v9 teacher, not raw HMR2a.
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


def make_optimizer(model, args):
    selected = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not selected or any(not n.startswith('part_branch.') for n, _ in selected):
        raise RuntimeError('Only spatial part parameters may be optimized')
    return torch.optim.AdamW([p for _, p in selected], lr=args.geometry_lr,
                             weight_decay=args.weight_decay), [n for n, _ in selected]


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
        "adaptation_config": asdict(model.adaptation_config),
        "model_revision": "joint-aligned-v16",
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


def train_geometry_epoch(model, train_loader, smpl_models, regressor,
                         optimizer, args, device, geometry_epoch, pose_stream, official_smpl):
    model.train()
    totals = {}
    pose_batches=0
    for step,batch in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)
        prediction=forward_batch(model,batch,device,args.amp_dtype)
        losses=geometric_losses(prediction,batch,smpl_models,regressor,args,device)
        teacher=dict(rotmat=prediction['teacher_rotmat'],betas=prediction['teacher_betas'])
        keep=geometry_distance(prediction,teacher,batch,smpl_models,regressor,device)
        total=losses['total']+args.teacher_weight*keep
        values=dict(losses,teacher_geometry_m=keep)
        pose_batch=None
        if pose_stream is not None and step % args.pose_every_steps == 0:
            pose_batch=pose_stream.next_batch()
            pp=forward_batch(model,pose_batch,device,args.amp_dtype)
            pl=pose_losses(pp,pose_batch,official_smpl,args,device)
            pt=dict(rotmat=pp['teacher_rotmat'],betas=pp['teacher_betas'])
            pk=geometry_distance(pp,pt,pose_batch,smpl_models,regressor,device)
            progress=geometry_epoch+(step+1)/len(train_loader)
            weight=args.pose_2d_weight*min(1.,progress/args.pose_warmup_epochs)
            total=total+weight*pl['pose_2d']+args.teacher_weight*pk
            values.update(pl,pose_teacher_m=pk)
            pose_batches+=1
        if not torch.isfinite(total):
            raise FloatingPointError('Nonfinite training loss')
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.part_branch.parameters(),args.grad_clip,error_if_nonfinite=True)
        optimizer.step()
        if pose_batch is not None:
            pose_stream.record_consumed(pose_batch)
        values['optimized_total']=total
        for name,value in values.items():
            totals[name]=totals.get(name,0.)+float(value.detach())
    metrics={name:value/(max(1,pose_batches) if name.startswith('pose_') else len(train_loader))
             for name,value in totals.items()}
    return metrics,pose_batches


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite a nonempty v16 run")
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
        args.max_val_windows, args.max_pose_val_windows, args.max_pose_train_windows))
    train3d = spread_subset(V10Dataset(train_vace, train_hmr), args.max_3dpw_windows)
    val3d = spread_subset(V10Dataset(val_vace, val_hmr), args.max_val_windows)
    pose_val = ShuffledPoseDatasetV10(args.pose_val_vace_dir, args.pose_val_hmr_dir,
                                     seed=args.seed, max_windows=args.max_pose_val_windows)
    if pose_val.split not in ('val', 'validation'):
        raise ValueError('PoseTrack validation split label is incorrect')
    pose_stream=None
    if args.pose_2d_weight:
        pose_train=ShuffledPoseDatasetV10(args.pose_train_vace_dir,args.pose_train_hmr_dir,
            seed=args.seed,max_windows=args.max_pose_train_windows)
        if pose_train.split != 'train' or pose_train.collection_id==pose_val.collection_id:
            raise ValueError('PoseTrack training must be distinct from validation')
        pose_loader=DataLoader(pose_train,batch_size=args.batch_size,shuffle=False,
                              num_workers=0,pin_memory=True,
                              generator=torch.Generator().manual_seed(args.seed+1))
        pose_stream=CyclingPoseStream(pose_loader)
    payload = load(args.init_checkpoint)
    if payload.get('format') != 'ram-geometry-priority-isolated-v9':
        raise ValueError('This controlled experiment must initialize from the original v9 checkpoint')
    model = model_from_payload(payload, AdaptationConfig(hidden_dim=args.part_hidden_dim,max_angle_degrees=args.max_angle_degrees)).to(device)
    smpl_models = create_smpl_models(args.smpl_model_root, device)
    for value in smpl_models.values():
        value.requires_grad_(False)
    model.bind_geometry(smpl_models[2])
    from hmr2.models.smpl_wrapper import SMPL as OfficialSMPL
    official_smpl = OfficialSMPL(
        model_path=str(args.smpl_model_root), gender="neutral", num_betas=10,
        joint_regressor_extra=str(args.smpl_model_root.parent / "SMPL_to_J19.pkl"),
    ).to(device)
    official_smpl.requires_grad_(False)
    regressor = torch.as_tensor(np.load(args.joint_regressor), dtype=torch.float32, device=device)

    options = {"num_workers": args.num_workers, "pin_memory": True,
               "persistent_workers": args.num_workers > 0}
    train_loader = DataLoader(train3d, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed), **options)
    val_loader = DataLoader(val3d, batch_size=args.val_batch_size, shuffle=False,
                            generator=torch.Generator().manual_seed(args.seed+2), **options)
    pose_val_loader = DataLoader(pose_val, batch_size=args.val_batch_size, shuffle=False,
                                 num_workers=0, pin_memory=True,
                                 generator=torch.Generator().manual_seed(args.seed+3))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    baseline_geometry, baseline_pose = evaluate(
        model, val_loader, pose_val_loader, smpl_models, official_smpl,
        regressor, args, device,
    )
    optimizer, trainable = make_optimizer(model, args)
    best_score = selection_score(
        baseline_pose, baseline_geometry, baseline_pose, baseline_geometry,
        args.geometry_regression_tolerance_mm, args.pose_pck_tolerance,
    )
    save(args.output_dir / "best.pt", model, optimizer, 0, "anchor",
         baseline_geometry, baseline_pose, args)
    manifest = {
        "format": FORMAT, "initial_checkpoint": str(args.init_checkpoint.resolve()),
        "3dpw_train_windows": len(train3d), "3dpw_validation_windows": len(val3d),
        "posetrack_train_windows": len(pose_stream.dataset) if pose_stream else 0, "posetrack_validation_windows": len(pose_val),
        "existing_model_sources_modified": False,
        "adaptation_config": asdict(model.adaptation_config),
        "validation_sequence_names": sorted(val_sequences),
        "validation_sequence_count": len(val_sequences),
        "development_only": args.development_only,
        "trainable_parameters": trainable,
        "test_used_for_model_selection": False,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    append_record(metrics_path, {
        "epoch": 0, "stage": "anchor", "validation": baseline_geometry,
        "pose_validation": baseline_pose, "accepted": True,
    })

    epoch = 0
    stale_epochs = 0
    for geometry_epoch in range(args.geometry_epochs):
        epoch += 1
        train_metrics, pose_batches = train_geometry_epoch(
            model, train_loader, smpl_models, regressor, optimizer, args, device, geometry_epoch, pose_stream, official_smpl,
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
            "pose_coverage": pose_stream.coverage() if pose_stream else None,
            "3dpw_windows_consumed": len(train3d),
            "stale_epochs": stale_epochs,
        })
        if stale_epochs >= args.patience:
            print(json.dumps({'early_stop': True, 'epoch': epoch}), flush=True)
            break

    append_record(metrics_path, {'completed': True, 'epochs_run': epoch,
        'best_validation_mpjpe_mm': -best_score, 'development_only': args.development_only})

if __name__ == "__main__":
    main()
