"""Additive mixed 3DPW + PoseTrack18 fine-tuning for RAM/DiviBody V11.

3DPW keeps metric 3D/mesh supervision.  PoseTrack contributes visible-joint
2D reprojection, projected acceleration, and a conservative HMR-anchor loss.
No existing model or trainer file is modified.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from divibody_training.evaluate_standard_3dpw import create_smpl_models
from divibody_training.train_divibody_continuous_geometry_v5 import smpl_vertices
from divibody_training.train_divibody_rgb_vace_hmr2a_v8 import (
    V8Dataset,
    load_pair,
    validate,
)
from divibody_training.train_divibody_rgb_vace_hmr2a_v9_small import spread_subset
from divibody_training.train_divibody_rgb_vace_v7 import forward_batch, geometric_losses
from models.human_decoder_rgb_vace_v8_hmr2a import HumanRGBVaceGeometryTransformerV8HMR2A


FORMAT = "ram-mixed-3dpw-posetrack-v1"
POSE_VACE_FORMAT = "ram-posetrack18-contiguous-vace-v1"
POSE_HMR_FORMAT = "ram-posetrack18-hmr2a-anchor-v1"

# PoseTrack target index -> H36M17 regressor index.  Ear targets 3/4 have no
# stable H36M counterpart and are deliberately excluded from supervision.
PT_FROM_H36M = {0: 9, 1: 8, 2: 10, 5: 11, 6: 14, 7: 12, 8: 15,
                 9: 13, 10: 16, 11: 4, 12: 1, 13: 5, 14: 2, 15: 6, 16: 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-vace-cache", type=Path, required=True)
    parser.add_argument("--train-hmr-sidecar", type=Path, required=True)
    parser.add_argument("--val-vace-cache", type=Path, required=True)
    parser.add_argument("--val-hmr-sidecar", type=Path, required=True)
    parser.add_argument("--pose-vace-cache", type=Path, required=True)
    parser.add_argument("--pose-hmr-sidecar", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--smpl-model-root", type=Path, required=True)
    parser.add_argument("--joint-regressor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-3dpw-windows", type=int, default=0)
    parser.add_argument("--max-pose-windows", type=int, default=0)
    parser.add_argument("--max-val-windows", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--joint-weight", type=float, default=1.0)
    parser.add_argument("--vertex-weight", type=float, default=0.5)
    parser.add_argument("--rotation-weight", type=float, default=0.05)
    parser.add_argument("--shape-weight", type=float, default=0.001)
    parser.add_argument("--velocity-weight", type=float, default=0.5)
    parser.add_argument("--acceleration-weight", type=float, default=0.25)
    parser.add_argument("--anchor-weight", type=float, default=0.02)
    parser.add_argument("--pose-2d-weight", type=float, default=2.0)
    parser.add_argument("--pose-acceleration-weight", type=float, default=0.1)
    parser.add_argument("--pose-anchor-weight", type=float, default=0.05)
    parser.add_argument("--amp-dtype", choices=("bf16", "none"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class PoseDataset(Dataset):
    def __init__(self, vace: dict[str, Any], hmr: dict[str, Any]) -> None:
        if vace.get("signature", {}).get("format") != POSE_VACE_FORMAT:
            raise ValueError("Unexpected PoseTrack VACE cache")
        if hmr.get("signature", {}).get("format") != POSE_HMR_FORMAT:
            raise ValueError("Unexpected PoseTrack HMR sidecar")
        if vace["signature"].get("split") != "train" or hmr["signature"].get("split") != "train":
            raise ValueError("Mixed training accepts PoseTrack train only")
        if len(vace["metadata"]) != len(hmr["window_uids"]):
            raise ValueError("PoseTrack VACE/HMR window counts disagree")
        if [x["window_uid"] for x in vace["metadata"]] != hmr["window_uids"]:
            raise ValueError("PoseTrack VACE/HMR window identities disagree")
        self.vace, self.hmr = vace, hmr

    def __len__(self) -> int:
        return len(self.vace["metadata"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        frames = self.hmr["window_frame_indices"][index]
        return {
            "rgb_tokens": self.hmr["rgb_tokens"][frames],
            "vace_features": self.vace["features"][index],
            "base_rotmat": self.hmr["base_rotmat"][frames],
            "base_betas": self.hmr["base_betas"][frames],
            "base_camera": self.hmr["base_camera"][frames],
            "keypoints": self.vace["target_keypoints"][index],
            "frame_owner": self.vace["frame_owner_mask"][index],
            "acceleration_owner": self.vace["acceleration_owner_mask"][index],
            "target_valid": self.vace["target_valid_mask"][index],
        }


def projected_posetrack(
    prediction: dict[str, torch.Tensor],
    smpl: torch.nn.Module,
    regressor: torch.Tensor,
) -> torch.Tensor:
    batch, frames = prediction["rotmat"].shape[:2]
    vertices = smpl_vertices(
        smpl,
        prediction["rotmat"].float().reshape(-1, 24, 3, 3),
        prediction["betas"].float().reshape(-1, 10),
    )
    joints = torch.einsum("jv,bvc->bjc", regressor, vertices)
    pose = joints.new_zeros((joints.shape[0], 17, 3))
    supported = torch.zeros(17, dtype=torch.bool, device=joints.device)
    for target, source in PT_FROM_H36M.items():
        pose[:, target] = joints[:, source]
        supported[target] = True
    camera = prediction["camera"].float().reshape(-1, 3)
    focal = 5000.0
    image_size = 256.0
    translation = torch.stack(
        (camera[:, 1], camera[:, 2], 2.0 * focal / (image_size * camera[:, 0].clamp_min(1e-6))),
        dim=-1,
    )
    translated = pose + translation[:, None]
    xy = translated[:, :, :2] / translated[:, :, 2:].clamp_min(1e-6)
    xy = xy * (focal / image_size)
    return xy.reshape(batch, frames, 17, 2), supported


def masked_smooth_l1(values: torch.Tensor, mask: torch.Tensor, name: str) -> torch.Tensor:
    selected = values.masked_select(mask)
    if selected.numel() == 0:
        raise RuntimeError(f"{name} mask selected no values")
    return selected.mean()


def pose_losses(prediction, batch, smpl, regressor, args, device):
    with torch.autocast(device_type=device.type, enabled=False):
        projected, supported = projected_posetrack(prediction, smpl, regressor)
        gt = batch["keypoints"].to(device, dtype=torch.float32)
        gt_xy = gt[..., :2] / 256.0 - 0.5
        visible = gt[..., 2] > 0
        visible &= supported.reshape(1, 1, -1)
        visible &= batch["target_valid"].to(device).unsqueeze(-1)
        visible &= batch["frame_owner"].to(device).unsqueeze(-1)
        per_joint = F.smooth_l1_loss(projected, gt_xy, reduction="none").sum(-1)
        reprojection = masked_smooth_l1(per_joint, visible, "pose_2d")

        pred_acc = torch.diff(projected, n=2, dim=1)
        gt_acc = torch.diff(gt_xy, n=2, dim=1)
        accel_visible = visible[:, :-2] & visible[:, 1:-1] & visible[:, 2:]
        accel_visible &= batch["acceleration_owner"].to(device).unsqueeze(-1)
        accel_error = F.smooth_l1_loss(pred_acc, gt_acc, reduction="none").sum(-1)
        acceleration = masked_smooth_l1(accel_error, accel_visible, "pose_acceleration")

        anchor = (
            F.mse_loss(prediction["rotmat"].float(), batch["base_rotmat"].to(device).float())
            + F.mse_loss(prediction["betas"].float(), batch["base_betas"].to(device).float())
            + 0.1 * F.mse_loss(prediction["camera"].float(), batch["base_camera"].to(device).float())
        )
        total = (
            args.pose_2d_weight * reprojection
            + args.pose_acceleration_weight * acceleration
            + args.pose_anchor_weight * anchor
        )
        return {"pose_total": total, "pose_2d": reprojection,
                "pose_acceleration": acceleration, "pose_anchor": anchor}


def save(path, model, optimizer, epoch, validation, args):
    payload = {
        "format": FORMAT,
        "epoch": epoch,
        "validation": validation,
        "selection_metric": "3DPW_validation_mpjpe_mm",
        "model": model.state_dict(),
        "model_config": vars(model.config),
        "optimizer": optimizer.state_dict(),
        "test_used_for_model_selection": False,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite a nonempty mixed-training run")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_vace, train_hmr = load_pair(args.train_vace_cache, args.train_hmr_sidecar, "train")
    val_vace, val_hmr = load_pair(args.val_vace_cache, args.val_hmr_sidecar, "validation")
    train3d = spread_subset(V8Dataset(train_vace, train_hmr), args.max_3dpw_windows)
    val3d = spread_subset(V8Dataset(val_vace, val_hmr), args.max_val_windows)
    pose = spread_subset(PoseDataset(load(args.pose_vace_cache), load(args.pose_hmr_sidecar)), args.max_pose_windows)

    model = HumanRGBVaceGeometryTransformerV8HMR2A().to(device)
    checkpoint = load(args.init_checkpoint)
    model.load_state_dict(checkpoint["model"], strict=True)
    smpl_models = create_smpl_models(args.smpl_model_root, device)
    for value in smpl_models.values():
        value.requires_grad_(False)
    regressor = torch.as_tensor(np.load(args.joint_regressor), dtype=torch.float32, device=device)
    options = {"num_workers": args.num_workers, "pin_memory": True,
               "persistent_workers": args.num_workers > 0}
    train_loader = DataLoader(train3d, batch_size=args.batch_size, shuffle=True, **options)
    pose_loader = DataLoader(pose, batch_size=args.batch_size, shuffle=True, **options)
    val_loader = DataLoader(val3d, batch_size=args.val_batch_size, shuffle=False, **options)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True)
    manifest = {
        "format": FORMAT,
        "initial_checkpoint": str(args.init_checkpoint.resolve()),
        "3dpw_train_windows": len(train3d),
        "posetrack_train_windows": len(pose),
        "3dpw_validation_windows": len(val3d),
        "instavariety_usage": "not used: ready copy contains 2D motion arrays but no RGB frames",
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
        model.train(); totals: dict[str, float] = {}; batches = 0
        pose_iterator = iter(pose_loader)
        steps = max(len(train_loader), len(pose_loader))
        train_iterator = iter(train_loader)
        for _ in range(steps):
            try:
                batch3d = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader); batch3d = next(train_iterator)
            try:
                batch_pose = next(pose_iterator)
            except StopIteration:
                pose_iterator = iter(pose_loader); batch_pose = next(pose_iterator)
            optimizer.zero_grad(set_to_none=True)
            prediction3d = forward_batch(model, batch3d, device, args.amp_dtype)
            loss3d = geometric_losses(prediction3d, batch3d, smpl_models, regressor, args, device)
            prediction_pose = forward_batch(model, batch_pose, device, args.amp_dtype)
            loss_pose = pose_losses(prediction_pose, batch_pose, smpl_models[2], regressor, args, device)
            total = loss3d["total"] + loss_pose["pose_total"]
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            values = {f"3dpw_{k}": v for k, v in loss3d.items()} | loss_pose | {"mixed_total": total}
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
            batches += 1
        validation = validate(val_loader, model, smpl_models, regressor, args, device)
        record = {"epoch": epoch, "train": {k: v / batches for k, v in totals.items()},
                  "validation": validation,
                  "temporal_gate": model.temporal_gates.detach().tanh().cpu().tolist()}
        with metrics.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        save(args.output_dir / "latest.pt", model, optimizer, epoch, validation, args)
        if validation["mpjpe_mm"] < best:
            best = validation["mpjpe_mm"]
            save(args.output_dir / "best.pt", model, optimizer, epoch, validation, args)
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
