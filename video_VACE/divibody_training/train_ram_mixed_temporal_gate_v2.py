"""Conservative v2 mixed training: update temporal gates only.

This imports the additive v1 data/loss implementation read-only, freezes every
geometry parameter, applies PoseTrack supervision once per four 3DPW steps, and
keeps 3DPW validation as the sole checkpoint-selection source.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from divibody_training.evaluate_standard_3dpw import create_smpl_models
from divibody_training.train_divibody_rgb_vace_hmr2a_v8 import V8Dataset, load_pair, validate
from divibody_training.train_divibody_rgb_vace_hmr2a_v9_small import spread_subset
from divibody_training.train_divibody_rgb_vace_v7 import forward_batch, geometric_losses
from divibody_training.train_ram_mixed_3dpw_posetrack_v1 import (
    PoseDataset,
    load,
    parse_args,
    pose_losses,
)
from models.human_decoder_rgb_vace_v8_hmr2a import HumanRGBVaceGeometryTransformerV8HMR2A


FORMAT = "ram-mixed-temporal-gate-v2"
POSE_EVERY_STEPS = 4
TRAINABLE_NAMES = {"temporal_gates", "vace_spatial_scale"}


def save(path, model, optimizer, epoch, validation, args):
    payload = {
        "format": FORMAT,
        "epoch": epoch,
        "validation": validation,
        "selection_metric": "3DPW_validation_mpjpe_mm",
        "model": model.state_dict(),
        "model_config": vars(model.config),
        "optimizer": optimizer.state_dict(),
        "trainable_parameter_names": sorted(TRAINABLE_NAMES),
        "pose_every_steps": POSE_EVERY_STEPS,
        "test_used_for_model_selection": False,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite a nonempty temporal-gate run")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_vace, train_hmr = load_pair(args.train_vace_cache, args.train_hmr_sidecar, "train")
    val_vace, val_hmr = load_pair(args.val_vace_cache, args.val_hmr_sidecar, "validation")
    train3d = spread_subset(V8Dataset(train_vace, train_hmr), args.max_3dpw_windows)
    val3d = spread_subset(V8Dataset(val_vace, val_hmr), args.max_val_windows)
    pose = spread_subset(PoseDataset(load(args.pose_vace_cache), load(args.pose_hmr_sidecar)),
                         args.max_pose_windows)
    model = HumanRGBVaceGeometryTransformerV8HMR2A().to(device)
    checkpoint = load(args.init_checkpoint)
    model.load_state_dict(checkpoint["model"], strict=True)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in TRAINABLE_NAMES)
    trainable = [(name, parameter) for name, parameter in model.named_parameters()
                 if parameter.requires_grad]
    if {name for name, _ in trainable} != TRAINABLE_NAMES:
        raise RuntimeError(f"Unexpected trainable parameter set: {[name for name, _ in trainable]}")
    smpl_models = create_smpl_models(args.smpl_model_root, device)
    for value in smpl_models.values():
        value.requires_grad_(False)
    regressor = torch.as_tensor(np.load(args.joint_regressor), dtype=torch.float32, device=device)
    options = {"num_workers": args.num_workers, "pin_memory": True,
               "persistent_workers": args.num_workers > 0}
    train_loader = DataLoader(train3d, batch_size=args.batch_size, shuffle=True, **options)
    pose_loader = DataLoader(pose, batch_size=args.batch_size, shuffle=True, **options)
    val_loader = DataLoader(val3d, batch_size=args.val_batch_size, shuffle=False, **options)
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable],
                                  lr=args.lr, weight_decay=0.0)
    args.output_dir.mkdir(parents=True)
    manifest = {
        "format": FORMAT,
        "initial_checkpoint": str(args.init_checkpoint.resolve()),
        "3dpw_train_windows": len(train3d),
        "posetrack_train_windows": len(pose),
        "3dpw_validation_windows": len(val3d),
        "trainable_parameters": {name: parameter.numel() for name, parameter in trainable},
        "trainable_parameter_total": sum(parameter.numel() for _, parameter in trainable),
        "pose_every_steps": POSE_EVERY_STEPS,
        "existing_model_sources_modified": False,
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    baseline = validate(val_loader, model, smpl_models, regressor, args, device)
    best = baseline["mpjpe_mm"]
    save(args.output_dir / "best.pt", model, optimizer, 0, baseline, args)
    metrics = args.output_dir / "metrics.jsonl"
    metrics.write_text(json.dumps({"epoch": 0, "validation": baseline,
                                   "loaded_initial_checkpoint": True}) + "\n", encoding="utf-8")
    print(json.dumps({"epoch": 0, "validation": baseline,
                      "loaded_initial_checkpoint": True}), flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train(); totals = {}; batches = pose_batches = 0
        pose_iterator = iter(pose_loader)
        for step, batch3d in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            prediction3d = forward_batch(model, batch3d, device, args.amp_dtype)
            loss3d = geometric_losses(prediction3d, batch3d, smpl_models, regressor, args, device)
            total = loss3d["total"]
            values = {f"3dpw_{name}": value for name, value in loss3d.items()}
            if step % POSE_EVERY_STEPS == 0:
                try:
                    batch_pose = next(pose_iterator)
                except StopIteration:
                    pose_iterator = iter(pose_loader); batch_pose = next(pose_iterator)
                prediction_pose = forward_batch(model, batch_pose, device, args.amp_dtype)
                loss_pose = pose_losses(prediction_pose, batch_pose, smpl_models[2], regressor, args, device)
                total = total + loss_pose["pose_total"]
                values.update(loss_pose); pose_batches += 1
            total.backward()
            torch.nn.utils.clip_grad_norm_([parameter for _, parameter in trainable], args.grad_clip)
            optimizer.step()
            values["mixed_total"] = total
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
            batches += 1
        validation = validate(val_loader, model, smpl_models, regressor, args, device)
        record = {"epoch": epoch, "train": {k: v / (pose_batches if k.startswith("pose_") else batches)
                                               for k, v in totals.items()},
                  "pose_batches": pose_batches, "validation": validation,
                  "temporal_gate": model.temporal_gates.detach().tanh().cpu().tolist(),
                  "vace_spatial_scale": float(model.vace_spatial_scale.detach().cpu())}
        with metrics.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        save(args.output_dir / "latest.pt", model, optimizer, epoch, validation, args)
        if validation["mpjpe_mm"] < best:
            best = validation["mpjpe_mm"]
            save(args.output_dir / "best.pt", model, optimizer, epoch, validation, args)
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
