"""Camera-only PoseTrack calibration with a disjoint validation cache."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from divibody_training.evaluate_standard_3dpw import create_smpl_models
from divibody_training.train_divibody_rgb_vace_hmr2a_v8 import V8Dataset, load_pair, validate
from divibody_training.train_divibody_rgb_vace_hmr2a_v9_small import spread_subset
from divibody_training.train_divibody_rgb_vace_v7 import forward_batch
from divibody_training.train_ram_mixed_3dpw_posetrack_v1 import load
from divibody_training.train_ram_mixed_official_pose_v3 import pose_losses
from divibody_training.train_ram_posetrack_camera_v4 import evaluate_pose
from models.human_decoder_rgb_vace_v8_hmr2a import HumanRGBVaceGeometryTransformerV8HMR2A


FORMAT = "ram-posetrack-camera-heldout-v5"
TRAINABLE_PREFIX = "camera_residual_head."


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pose-train-vace", type=Path, required=True)
    p.add_argument("--pose-train-hmr", type=Path, required=True)
    p.add_argument("--pose-val-vace", type=Path, required=True)
    p.add_argument("--pose-val-hmr", type=Path, required=True)
    p.add_argument("--geometry-val-vace", type=Path, required=True)
    p.add_argument("--geometry-val-hmr", type=Path, required=True)
    p.add_argument("--init-checkpoint", type=Path, required=True)
    p.add_argument("--smpl-model-root", type=Path, required=True)
    p.add_argument("--joint-regressor", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--val-batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-geometry-val-windows", type=int, default=192)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--pose-2d-weight", type=float, default=1.0)
    p.add_argument("--pose-acceleration-weight", type=float, default=0.05)
    p.add_argument("--pose-anchor-weight", type=float, default=0.1)
    p.add_argument("--amp-dtype", choices=("bf16", "none"), default="bf16")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


class AnyPoseDataset(Dataset):
    def __init__(self, vace, hmr):
        if len(vace["metadata"]) != len(hmr["window_uids"]):
            raise ValueError("Pose VACE/HMR window counts disagree")
        if [x["window_uid"] for x in vace["metadata"]] != hmr["window_uids"]:
            raise ValueError("Pose VACE/HMR identities disagree")
        self.vace, self.hmr = vace, hmr

    def __len__(self): return len(self.vace["metadata"])

    def __getitem__(self, index):
        frames = self.hmr["window_frame_indices"][index]
        return {"rgb_tokens": self.hmr["rgb_tokens"][frames],
                "vace_features": self.vace["features"][index],
                "base_rotmat": self.hmr["base_rotmat"][frames],
                "base_betas": self.hmr["base_betas"][frames],
                "base_camera": self.hmr["base_camera"][frames],
                "keypoints": self.vace["target_keypoints"][index],
                "frame_owner": self.vace["frame_owner_mask"][index],
                "acceleration_owner": self.vace["acceleration_owner_mask"][index],
                "target_valid": self.vace["target_valid_mask"][index]}


def score(metric, baseline):
    # Reject checkpoints that trade a loose-threshold collapse for PCK@0.05.
    if metric["PCKn@0.10"] < baseline["PCKn@0.10"] or metric["mean_normalized_joint_error"] > baseline["mean_normalized_joint_error"]:
        return float("-inf")
    return metric["PCKn@0.05"] + metric["PCKn@0.10"]


def save(path, model, optimizer, epoch, pose_train, pose_val, geometry, args):
    payload = {"format": FORMAT, "epoch": epoch, "pose_train": pose_train,
               "pose_validation": pose_val, "geometry_guard": geometry,
               "selection_rule": "max(PCK05+PCK10), with PCK10>=epoch0 and mean_error<=epoch0",
               "model": model.state_dict(), "model_config": vars(model.config),
               "optimizer": optimizer.state_dict(), "test_used_for_model_selection": False,
               "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    temporary = path.with_suffix(".pt.tmp"); torch.save(payload, temporary); os.replace(temporary, path)


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite a nonempty v5 run")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_pose = AnyPoseDataset(load(args.pose_train_vace), load(args.pose_train_hmr))
    val_pose = AnyPoseDataset(load(args.pose_val_vace), load(args.pose_val_hmr))
    val_vace, val_hmr = load_pair(args.geometry_val_vace, args.geometry_val_hmr, "validation")
    geometry_data = spread_subset(V8Dataset(val_vace, val_hmr), args.max_geometry_val_windows)
    model = HumanRGBVaceGeometryTransformerV8HMR2A().to(device)
    model.load_state_dict(load(args.init_checkpoint)["model"], strict=True)
    for name, parameter in model.named_parameters(): parameter.requires_grad_(name.startswith(TRAINABLE_PREFIX))
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    smpl_models = create_smpl_models(args.smpl_model_root, device)
    for value in smpl_models.values(): value.requires_grad_(False)
    from hmr2.models.smpl_wrapper import SMPL as OfficialSMPL
    official_smpl = OfficialSMPL(model_path=str(args.smpl_model_root), gender="neutral", num_betas=10,
        joint_regressor_extra=str(args.smpl_model_root.parent / "SMPL_to_J19.pkl")).to(device)
    official_smpl.requires_grad_(False)
    regressor = torch.as_tensor(np.load(args.joint_regressor), dtype=torch.float32, device=device)
    opts = {"num_workers": args.num_workers, "pin_memory": True,
            "persistent_workers": args.num_workers > 0}
    train_loader = DataLoader(train_pose, batch_size=args.batch_size, shuffle=True, **opts)
    train_eval = DataLoader(train_pose, batch_size=args.val_batch_size, shuffle=False, **opts)
    val_eval = DataLoader(val_pose, batch_size=args.val_batch_size, shuffle=False, **opts)
    geometry_loader = DataLoader(geometry_data, batch_size=args.val_batch_size, shuffle=False, **opts)
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=args.lr, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True)
    geometry0 = validate(geometry_loader, model, smpl_models, regressor, args, device)
    train0 = evaluate_pose(train_eval, model, official_smpl, args, device)
    val0 = evaluate_pose(val_eval, model, official_smpl, args, device)
    best_score = score(val0, val0)
    save(args.output_dir / "best.pt", model, optimizer, 0, train0, val0, geometry0, args)
    metrics = args.output_dir / "metrics.jsonl"
    metrics.write_text(json.dumps({"epoch": 0, "pose_train": train0,
                                   "pose_validation": val0, "geometry_guard": geometry0}) + "\n", encoding="utf-8")
    print(json.dumps({"epoch": 0, "pose_train": train0,
                      "pose_validation": val0, "geometry_guard": geometry0}), flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train(); totals = {}; batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            losses = pose_losses(forward_batch(model, batch, device, args.amp_dtype),
                                 batch, official_smpl, args, device)
            losses["pose_total"].backward(); torch.nn.utils.clip_grad_norm_([p for _, p in trainable], args.grad_clip); optimizer.step()
            for name, value in losses.items(): totals[name] = totals.get(name, 0.0) + float(value.detach())
            batches += 1
        train_metric = evaluate_pose(train_eval, model, official_smpl, args, device)
        val_metric = evaluate_pose(val_eval, model, official_smpl, args, device)
        current_score = score(val_metric, val0)
        record = {"epoch": epoch, "train_loss": {k: v / batches for k, v in totals.items()},
                  "pose_train": train_metric, "pose_validation": val_metric,
                  "accepted_score": current_score}
        with metrics.open("a", encoding="utf-8") as handle: handle.write(json.dumps(record) + "\n")
        save(args.output_dir / "latest.pt", model, optimizer, epoch, train_metric, val_metric, geometry0, args)
        if current_score > best_score:
            best_score = current_score; save(args.output_dir / "best.pt", model, optimizer, epoch, train_metric, val_metric, geometry0, args)
        print(json.dumps(record), flush=True)
    geometry_final = validate(geometry_loader, model, smpl_models, regressor, args, device)
    for name in ("mpjpe_mm", "pa_mpjpe_mm", "pve_mm", "acceleration_error_mm_per_frame2"):
        if abs(geometry_final[name] - geometry0[name]) > 1e-9:
            raise RuntimeError(f"Geometry invariant failed for {name}")


if __name__ == "__main__": main()
