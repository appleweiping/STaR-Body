"""PoseTrack camera-only calibration with geometry frozen by construction."""

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
from divibody_training.train_divibody_rgb_vace_v7 import forward_batch
from divibody_training.train_ram_mixed_3dpw_posetrack_v1 import PoseDataset, load, parse_args
from divibody_training.train_ram_mixed_official_pose_v3 import pose_losses, projected_posetrack
from models.human_decoder_rgb_vace_v8_hmr2a import HumanRGBVaceGeometryTransformerV8HMR2A


FORMAT = "ram-posetrack-camera-only-v4"
TRAINABLE_PREFIX = "camera_residual_head."


@torch.no_grad()
def evaluate_pose(loader, model, official_smpl, args, device):
    model.eval(); correct05 = correct10 = visible_count = 0; error_sum = 0.0
    for batch in loader:
        prediction = forward_batch(model, batch, device, args.amp_dtype)
        projected = projected_posetrack(prediction, official_smpl)
        gt = batch["keypoints"].to(device, dtype=torch.float32)
        visible = gt[..., 2] > 0
        visible &= batch["target_valid"].to(device).unsqueeze(-1)
        visible &= batch["frame_owner"].to(device).unsqueeze(-1)
        normalised = torch.linalg.vector_norm(projected - (gt[..., :2] / 96.0 - 0.5), dim=-1) / (1.0 / 1.25)
        values = normalised.masked_select(visible)
        correct05 += int((values <= 0.05).sum()); correct10 += int((values <= 0.10).sum())
        visible_count += int(values.numel()); error_sum += float(values.double().sum())
    return {"PCKn@0.05": correct05 / visible_count, "PCKn@0.10": correct10 / visible_count,
            "mean_normalized_joint_error": error_sum / visible_count,
            "visible_keypoints": visible_count}


def save(path, model, optimizer, epoch, pose_metric, geometry_metric, args):
    payload = {"format": FORMAT, "epoch": epoch, "pose_validation": pose_metric,
               "geometry_guard": geometry_metric, "selection_metric": "PoseTrack_train_subset_PCKn@0.05",
               "model": model.state_dict(), "model_config": vars(model.config),
               "optimizer": optimizer.state_dict(), "trainable_prefix": TRAINABLE_PREFIX,
               "test_used_for_model_selection": False,
               "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    temporary = path.with_suffix(".pt.tmp"); torch.save(payload, temporary); os.replace(temporary, path)


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite a nonempty camera-calibration run")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    val_vace, val_hmr = load_pair(args.val_vace_cache, args.val_hmr_sidecar, "validation")
    val3d = spread_subset(V8Dataset(val_vace, val_hmr), args.max_val_windows)
    pose = spread_subset(PoseDataset(load(args.pose_vace_cache), load(args.pose_hmr_sidecar)),
                         args.max_pose_windows)
    model = HumanRGBVaceGeometryTransformerV8HMR2A().to(device)
    model.load_state_dict(load(args.init_checkpoint)["model"], strict=True)
    for name, parameter in model.named_parameters(): parameter.requires_grad_(name.startswith(TRAINABLE_PREFIX))
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any(not name.startswith(TRAINABLE_PREFIX) for name, _ in trainable):
        raise RuntimeError(f"Unexpected trainable parameters: {[name for name, _ in trainable]}")
    smpl_models = create_smpl_models(args.smpl_model_root, device)
    for value in smpl_models.values(): value.requires_grad_(False)
    from hmr2.models.smpl_wrapper import SMPL as OfficialSMPL
    official_smpl = OfficialSMPL(model_path=str(args.smpl_model_root), gender="neutral", num_betas=10,
        joint_regressor_extra=str(args.smpl_model_root.parent / "SMPL_to_J19.pkl")).to(device)
    official_smpl.requires_grad_(False)
    regressor = torch.as_tensor(np.load(args.joint_regressor), dtype=torch.float32, device=device)
    opts = {"num_workers": args.num_workers, "pin_memory": True,
            "persistent_workers": args.num_workers > 0}
    pose_loader = DataLoader(pose, batch_size=args.batch_size, shuffle=True, **opts)
    pose_eval_loader = DataLoader(pose, batch_size=args.val_batch_size, shuffle=False, **opts)
    val_loader = DataLoader(val3d, batch_size=args.val_batch_size, shuffle=False, **opts)
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=args.lr, weight_decay=0.0)
    args.output_dir.mkdir(parents=True)
    geometry0 = validate(val_loader, model, smpl_models, regressor, args, device)
    pose0 = evaluate_pose(pose_eval_loader, model, official_smpl, args, device)
    manifest = {"format": FORMAT, "initial_checkpoint": str(args.init_checkpoint.resolve()),
                "trainable_parameters": {name: p.numel() for name, p in trainable},
                "posetrack_windows": len(pose), "geometry_guard_windows": len(val3d),
                "guarantee": "rotation/betas branches frozen; 3DPW geometry metrics invariant",
                "existing_model_sources_modified": False}
    (args.output_dir / "run_config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    best = pose0["PCKn@0.05"]; save(args.output_dir / "best.pt", model, optimizer, 0, pose0, geometry0, args)
    metrics = args.output_dir / "metrics.jsonl"
    metrics.write_text(json.dumps({"epoch": 0, "pose": pose0, "geometry_guard": geometry0}) + "\n", encoding="utf-8")
    print(json.dumps({"epoch": 0, "pose": pose0, "geometry_guard": geometry0}), flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train(); totals = {}; batches = 0
        for batch in pose_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = forward_batch(model, batch, device, args.amp_dtype)
            losses = pose_losses(prediction, batch, official_smpl, args, device)
            losses["pose_total"].backward()
            torch.nn.utils.clip_grad_norm_([p for _, p in trainable], args.grad_clip); optimizer.step()
            for name, value in losses.items(): totals[name] = totals.get(name, 0.0) + float(value.detach())
            batches += 1
        pose_metric = evaluate_pose(pose_eval_loader, model, official_smpl, args, device)
        geometry_metric = validate(val_loader, model, smpl_models, regressor, args, device)
        for name in ("mpjpe_mm", "pa_mpjpe_mm", "pve_mm", "acceleration_error_mm_per_frame2"):
            if abs(geometry_metric[name] - geometry0[name]) > 1e-9:
                raise RuntimeError(f"Camera-only geometry invariant failed for {name}")
        record = {"epoch": epoch, "train": {k: v / batches for k, v in totals.items()},
                  "pose": pose_metric, "geometry_guard": geometry_metric}
        with metrics.open("a", encoding="utf-8") as handle: handle.write(json.dumps(record) + "\n")
        save(args.output_dir / "latest.pt", model, optimizer, epoch, pose_metric, geometry_metric, args)
        if pose_metric["PCKn@0.05"] > best:
            best = pose_metric["PCKn@0.05"]; save(args.output_dir / "best.pt", model, optimizer, epoch, pose_metric, geometry_metric, args)
        print(json.dumps(record), flush=True)


if __name__ == "__main__": main()
