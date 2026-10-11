"""Corrected mixed trainer using the official HMR2 SMPL/OpenPose protocol.

The cached PoseTrack coordinates are 96-pixel VACE-crop coordinates.  This v3
normalises them by 96 (not 256), projects official OpenPose-25 joints through
the HMR2 perspective camera, and applies PoseTrack updates once per four 3DPW
steps.  Earlier additive trainers remain unchanged for auditability.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from divibody_training.evaluate_standard_3dpw import create_smpl_models
from divibody_training.train_divibody_rgb_vace_hmr2a_v8 import V8Dataset, load_pair, validate
from divibody_training.train_divibody_rgb_vace_hmr2a_v9_small import spread_subset
from divibody_training.train_divibody_rgb_vace_v7 import forward_batch, geometric_losses
from divibody_training.train_ram_mixed_3dpw_posetrack_v1 import PoseDataset, load, parse_args
from models.human_decoder_rgb_vace_v8_hmr2a import HumanRGBVaceGeometryTransformerV8HMR2A


FORMAT = "ram-mixed-official-posetrack-v3"
POSE_EVERY_STEPS = 4
POSETRACK_CROP_SIZE = 96.0
PT_FROM_OP = {0: 0, 1: 1, 3: 18, 4: 17, 5: 5, 6: 2, 7: 6, 8: 3,
              9: 7, 10: 4, 11: 12, 12: 9, 13: 13, 14: 10, 15: 14, 16: 11}


def projected_posetrack(prediction, official_smpl):
    batch, frames = prediction["rotmat"].shape[:2]
    rot = prediction["rotmat"].float().reshape(-1, 24, 3, 3)
    betas = prediction["betas"].float().reshape(-1, 10)
    output = official_smpl(
        global_orient=rot[:, :1], body_pose=rot[:, 1:], betas=betas, pose2rot=False
    )
    openpose = output.joints[:, :25]
    camera = prediction["camera"].float().reshape(-1, 3)
    translation = torch.stack(
        (camera[:, 1], camera[:, 2], 10000.0 / (256.0 * camera[:, 0].clamp_min(1e-6))),
        dim=-1,
    )
    points = openpose + translation[:, None]
    openpose_2d = points[:, :, :2] / points[:, :, 2:].clamp_min(1e-6) * (5000.0 / 256.0)
    result = openpose_2d.new_zeros((len(openpose_2d), 17, 2))
    for target, source in PT_FROM_OP.items():
        result[:, target] = openpose_2d[:, source]
    result[:, 2] = 2.0 * openpose_2d[:, 0] - openpose_2d[:, 1]
    return result.reshape(batch, frames, 17, 2)


def selected_mean(values, mask, name):
    selected = values.masked_select(mask)
    if selected.numel() == 0:
        raise RuntimeError(f"{name} mask selected no values")
    return selected.mean()


def pose_losses(prediction, batch, official_smpl, args, device):
    with torch.autocast(device_type=device.type, enabled=False):
        projected = projected_posetrack(prediction, official_smpl)
        gt = batch["keypoints"].to(device, dtype=torch.float32)
        gt_xy = gt[..., :2] / POSETRACK_CROP_SIZE - 0.5
        visible = gt[..., 2] > 0
        visible &= batch["target_valid"].to(device).unsqueeze(-1)
        visible &= batch["frame_owner"].to(device).unsqueeze(-1)
        error = F.smooth_l1_loss(projected, gt_xy, reduction="none").sum(-1)
        reprojection = selected_mean(error, visible, "official_pose_2d")
        pred_acc = torch.diff(projected, n=2, dim=1)
        gt_acc = torch.diff(gt_xy, n=2, dim=1)
        accel_visible = visible[:, :-2] & visible[:, 1:-1] & visible[:, 2:]
        accel_visible &= batch["acceleration_owner"].to(device).unsqueeze(-1)
        acceleration = selected_mean(
            F.smooth_l1_loss(pred_acc, gt_acc, reduction="none").sum(-1),
            accel_visible,
            "official_pose_acceleration",
        )
        anchor = (
            F.mse_loss(prediction["rotmat"].float(), batch["base_rotmat"].to(device).float())
            + F.mse_loss(prediction["betas"].float(), batch["base_betas"].to(device).float())
            + 0.1 * F.mse_loss(prediction["camera"].float(), batch["base_camera"].to(device).float())
        )
        total = (args.pose_2d_weight * reprojection
                 + args.pose_acceleration_weight * acceleration
                 + args.pose_anchor_weight * anchor)
        normalised = torch.linalg.vector_norm(projected - gt_xy, dim=-1) / (1.0 / 1.25)
        values = normalised.masked_select(visible)
        return {"pose_total": total, "pose_2d": reprojection,
                "pose_acceleration": acceleration, "pose_anchor": anchor,
                "pose_pck05": (values <= 0.05).float().mean(),
                "pose_pck10": (values <= 0.10).float().mean()}


def save(path, model, optimizer, epoch, validation, args):
    payload = {
        "format": FORMAT, "epoch": epoch, "validation": validation,
        "selection_metric": "3DPW_validation_mpjpe_mm", "model": model.state_dict(),
        "model_config": vars(model.config), "optimizer": optimizer.state_dict(),
        "pose_protocol": "official HMR2 SMPL OpenPose25; crop coordinate size 96",
        "pose_every_steps": POSE_EVERY_STEPS, "test_used_for_model_selection": False,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    temporary = path.with_suffix(".pt.tmp"); torch.save(payload, temporary); os.replace(temporary, path)


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite a nonempty official-pose run")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_vace, train_hmr = load_pair(args.train_vace_cache, args.train_hmr_sidecar, "train")
    val_vace, val_hmr = load_pair(args.val_vace_cache, args.val_hmr_sidecar, "validation")
    pose_vace, pose_hmr = load(args.pose_vace_cache), load(args.pose_hmr_sidecar)
    train3d = spread_subset(V8Dataset(train_vace, train_hmr), args.max_3dpw_windows)
    val3d = spread_subset(V8Dataset(val_vace, val_hmr), args.max_val_windows)
    pose = spread_subset(PoseDataset(pose_vace, pose_hmr), args.max_pose_windows)
    model = HumanRGBVaceGeometryTransformerV8HMR2A().to(device)
    model.load_state_dict(load(args.init_checkpoint)["model"], strict=True)
    smpl_models = create_smpl_models(args.smpl_model_root, device)
    for value in smpl_models.values(): value.requires_grad_(False)
    from hmr2.models.smpl_wrapper import SMPL as OfficialSMPL
    official_smpl = OfficialSMPL(
        model_path=str(args.smpl_model_root), gender="neutral", num_betas=10,
        joint_regressor_extra=str(args.smpl_model_root.parent / "SMPL_to_J19.pkl"),
    ).to(device)
    official_smpl.requires_grad_(False)
    regressor = torch.as_tensor(np.load(args.joint_regressor), dtype=torch.float32, device=device)
    opts = {"num_workers": args.num_workers, "pin_memory": True,
            "persistent_workers": args.num_workers > 0}
    train_loader = DataLoader(train3d, batch_size=args.batch_size, shuffle=True, **opts)
    pose_loader = DataLoader(pose, batch_size=args.batch_size, shuffle=True, **opts)
    val_loader = DataLoader(val3d, batch_size=args.val_batch_size, shuffle=False, **opts)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True)
    manifest = {"format": FORMAT, "initial_checkpoint": str(args.init_checkpoint.resolve()),
                "3dpw_train_windows": len(train3d), "posetrack_train_windows": len(pose),
                "3dpw_validation_windows": len(val3d), "pose_every_steps": POSE_EVERY_STEPS,
                "pose_protocol": "official HMR2 OpenPose25, 96-pixel GT coordinates",
                "existing_model_sources_modified": False}
    (args.output_dir / "run_config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    baseline = validate(val_loader, model, smpl_models, regressor, args, device)
    best = baseline["mpjpe_mm"]; save(args.output_dir / "best.pt", model, optimizer, 0, baseline, args)
    metrics = args.output_dir / "metrics.jsonl"
    metrics.write_text(json.dumps({"epoch": 0, "validation": baseline,
                                   "loaded_initial_checkpoint": True}) + "\n", encoding="utf-8")
    print(json.dumps({"epoch": 0, "validation": baseline,
                      "loaded_initial_checkpoint": True}), flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train(); totals = {}; batches = pose_batches = 0; pose_iterator = iter(pose_loader)
        for step, batch3d in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            pred3d = forward_batch(model, batch3d, device, args.amp_dtype)
            loss3d = geometric_losses(pred3d, batch3d, smpl_models, regressor, args, device)
            total = loss3d["total"]; values = {f"3dpw_{k}": v for k, v in loss3d.items()}
            if step % POSE_EVERY_STEPS == 0:
                try: batch_pose = next(pose_iterator)
                except StopIteration: pose_iterator = iter(pose_loader); batch_pose = next(pose_iterator)
                pred_pose = forward_batch(model, batch_pose, device, args.amp_dtype)
                loss_pose = pose_losses(pred_pose, batch_pose, official_smpl, args, device)
                total = total + loss_pose["pose_total"]; values.update(loss_pose); pose_batches += 1
            total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); optimizer.step()
            values["mixed_total"] = total
            for name, value in values.items(): totals[name] = totals.get(name, 0.0) + float(value.detach())
            batches += 1
        validation = validate(val_loader, model, smpl_models, regressor, args, device)
        record = {"epoch": epoch,
                  "train": {k: v / (pose_batches if k.startswith("pose_") else batches)
                            for k, v in totals.items()}, "pose_batches": pose_batches,
                  "validation": validation,
                  "temporal_gate": model.temporal_gates.detach().tanh().cpu().tolist()}
        with metrics.open("a", encoding="utf-8") as handle: handle.write(json.dumps(record) + "\n")
        save(args.output_dir / "latest.pt", model, optimizer, epoch, validation, args)
        if validation["mpjpe_mm"] < best:
            best = validation["mpjpe_mm"]; save(args.output_dir / "best.pt", model, optimizer, epoch, validation, args)
        print(json.dumps(record), flush=True)


if __name__ == "__main__": main()
