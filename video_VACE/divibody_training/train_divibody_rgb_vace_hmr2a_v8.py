"""Train DiviBody v8 from immutable VACE caches and HMR 2.0a sidecars.

The script is additive.  It never edits v7, HMR, VACE, or their checkpoints.
Epoch zero is validated before optimization and must reproduce cached HMR2a
anchors exactly.
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
from torch.utils.data import DataLoader, Dataset

from divibody_training.cache_hmr2a_rgb_anchor_v8 import validate_sidecar, validate_vace_source
from divibody_training.evaluate_standard_3dpw import create_smpl_models
from divibody_training.geometry_metrics_3dpw import mean_euclidean_error, pa_mpjpe_per_sample, pelvis_aligned_j14
from divibody_training.train_divibody_continuous_geometry_v5 import camera_targets, gendered_vertices, smpl_vertices
from divibody_training.train_divibody_rgb_vace_v7 import forward_batch, geometric_losses
from models.human_decoder_rgb_vace_v8_hmr2a import HumanRGBVaceGeometryTransformerV8HMR2A, RGBVaceHMR2AConfig


FORMAT = "divibody-rgb-vace-v8-hmr2a-training"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "val"):
        parser.add_argument(f"--{split}-vace-cache", type=Path, required=True)
        parser.add_argument(f"--{split}-hmr-sidecar", type=Path, required=True)
    parser.add_argument("--smpl-model-root", type=Path, required=True)
    parser.add_argument("--joint-regressor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--joint-weight", type=float, default=1.0)
    parser.add_argument("--vertex-weight", type=float, default=0.5)
    parser.add_argument("--rotation-weight", type=float, default=0.05)
    parser.add_argument("--shape-weight", type=float, default=0.001)
    parser.add_argument("--velocity-weight", type=float, default=0.5)
    parser.add_argument("--acceleration-weight", type=float, default=0.25)
    parser.add_argument("--anchor-weight", type=float, default=0.02)
    parser.add_argument("--amp-dtype", choices=("bf16", "none"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_pair(vace_path: Path, sidecar_path: Path, split: str):
    vace, sidecar = load(vace_path), load(sidecar_path)
    validate_vace_source(vace)
    validate_sidecar(sidecar, vace)
    if vace["signature"]["split"] != split or sidecar["signature"]["split"] != split:
        raise ValueError(f"Expected an exact {split} cache pair")
    if sidecar["signature"].get("development_only"):
        raise ValueError("Full v8 training refuses development-only sidecars")
    return vace, sidecar


class V8Dataset(Dataset):
    def __init__(self, vace: dict[str, Any], sidecar: dict[str, Any]) -> None:
        self.vace, self.sidecar = vace, sidecar
        self.lookup = sidecar["window_frame_indices"]

    def __len__(self) -> int:
        return len(self.lookup)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        frames = self.lookup[index]
        targets = self.vace["targets"]
        return {
            "rgb_tokens": self.sidecar["rgb_tokens"][frames],
            "vace_features": self.vace["features"][index],
            "base_rotmat": self.sidecar["base_rotmat"][frames],
            "base_betas": self.sidecar["base_betas"][frames],
            "base_camera": self.sidecar["base_camera"][frames],
            "full_pose": targets["full_pose"][index],
            "betas": targets["betas"][index],
            "gender_code": targets["gender_code"][index],
            "cam_pose": targets["cam_pose"][index],
            "frame_owner": self.vace["frame_owner_mask"][index],
            "velocity_owner": self.vace["velocity_owner_mask"][index],
            "acceleration_owner": self.vace["acceleration_owner_mask"][index],
        }


def assert_epoch0_anchor(model: torch.nn.Module, batch: dict[str, torch.Tensor], device: torch.device, amp: str) -> None:
    model.eval()
    with torch.inference_mode():
        prediction = forward_batch(model, batch, device, amp)
    for output, source in (("rotmat", "base_rotmat"), ("betas", "base_betas"), ("camera", "base_camera")):
        torch.testing.assert_close(
            prediction[output].float().cpu(), batch[source].float(), rtol=0.0, atol=1e-6
        )


@torch.no_grad()
def validate(loader, model, smpl_models, regressor, args, device):
    model.eval()
    sums = {name: 0.0 for name in ("mpjpe_mm", "pa_mpjpe_mm", "pve_mm", "acceleration_error_mm_per_frame2")}
    counts = {name: 0 for name in sums}
    for batch in loader:
        prediction = forward_batch(model, batch, device, args.amp_dtype)
        pred_rot = prediction["rotmat"].float()
        target_rot = camera_targets(batch, device)
        target_betas = batch["betas"].to(device, dtype=torch.float32)
        genders = batch["gender_code"].to(device).reshape(-1)
        pred_vertices = smpl_vertices(smpl_models[2], pred_rot.reshape(-1,24,3,3), prediction["betas"].float().reshape(-1,10))
        gt_vertices = gendered_vertices(smpl_models, target_rot.reshape(-1,24,3,3), target_betas.reshape(-1,10), genders)
        pred_joints, _ = pelvis_aligned_j14(pred_vertices, regressor)
        gt_joints, _ = pelvis_aligned_j14(gt_vertices, regressor)
        frame_mask = batch["frame_owner"].to(device).reshape(-1)
        arrays = {
            "mpjpe_mm": mean_euclidean_error(pred_joints, gt_joints) * 1000,
            "pa_mpjpe_mm": pa_mpjpe_per_sample(pred_joints, gt_joints) * 1000,
            "pve_mm": mean_euclidean_error(pred_vertices, gt_vertices) * 1000,
        }
        for name, values in arrays.items():
            selected = values.masked_select(frame_mask)
            sums[name] += float(selected.double().sum()); counts[name] += int(selected.numel())
        batch_size = pred_rot.shape[0]
        error = (pred_joints - gt_joints).reshape(batch_size, 8, 14, 3)
        accel = torch.linalg.vector_norm(torch.diff(error, n=2, dim=1), dim=-1).mean(-1) * 1000
        selected = accel.masked_select(batch["acceleration_owner"].to(device))
        sums["acceleration_error_mm_per_frame2"] += float(selected.double().sum())
        counts["acceleration_error_mm_per_frame2"] += int(selected.numel())
    if any(value == 0 for value in counts.values()):
        raise RuntimeError("Validation masks contain no owned samples")
    return {name: sums[name] / counts[name] for name in sums} | {
        "frames": counts["mpjpe_mm"], "acceleration_triplets": counts["acceleration_error_mm_per_frame2"]
    }


def save(path, model, optimizer, epoch, validation, args):
    payload = {
        "format": FORMAT, "epoch": epoch, "selection_metric": "validation_mpjpe_mm",
        "validation": validation, "model_config": vars(model.config),
        "anchor_metadata": model.anchor_metadata(), "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "test_used_for_model_selection": False,
        "args": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary); os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite a nonempty v8 experiment")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_vace, train_hmr = load_pair(args.train_vace_cache, args.train_hmr_sidecar, "train")
    val_vace, val_hmr = load_pair(args.val_vace_cache, args.val_hmr_sidecar, "validation")
    model = HumanRGBVaceGeometryTransformerV8HMR2A(RGBVaceHMR2AConfig()).to(device)
    smpl_models = create_smpl_models(args.smpl_model_root, device)
    for value in smpl_models.values(): value.requires_grad_(False)
    regressor_np = np.load(args.joint_regressor)
    regressor = torch.as_tensor(regressor_np, dtype=torch.float32, device=device)
    options = {"num_workers": args.num_workers, "pin_memory": device.type == "cuda", "persistent_workers": args.num_workers > 0}
    train_loader = DataLoader(V8Dataset(train_vace, train_hmr), batch_size=args.batch_size, shuffle=True, **options)
    val_loader = DataLoader(V8Dataset(val_vace, val_hmr), batch_size=args.val_batch_size, shuffle=False, **options)
    first = next(iter(val_loader)); assert_epoch0_anchor(model, first, device, args.amp_dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {"format": FORMAT, "model": model.anchor_metadata(), "test_data_used": False,
              "train_windows": len(train_loader.dataset), "validation_windows": len(val_loader.dataset)}
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    baseline = validate(val_loader, model, smpl_models, regressor, args, device)
    best = baseline["mpjpe_mm"]; save(args.output_dir/"best.pt", model, optimizer, 0, baseline, args)
    metrics = args.output_dir/"metrics.jsonl"; metrics.write_text(json.dumps({"epoch":0,"validation":baseline,"epoch0_exact_anchor":True})+"\n", encoding="utf-8")
    print(json.dumps({"epoch":0,"validation":baseline,"epoch0_exact_anchor":True}), flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train(); totals={}; batches=0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True); prediction=forward_batch(model,batch,device,args.amp_dtype)
            losses=geometric_losses(prediction,batch,smpl_models,regressor,args,device)
            losses["total"].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip); optimizer.step()
            for name,value in losses.items(): totals[name]=totals.get(name,0.0)+float(value.detach())
            batches += 1
        validation=validate(val_loader,model,smpl_models,regressor,args,device)
        record={"epoch":epoch,"train":{k:v/batches for k,v in totals.items()},"validation":validation,"temporal_gate":model.temporal_gates.detach().tanh().cpu().tolist()}
        with metrics.open("a",encoding="utf-8") as handle: handle.write(json.dumps(record)+"\n")
        save(args.output_dir/"latest.pt",model,optimizer,epoch,validation,args)
        if validation["mpjpe_mm"] < best: best=validation["mpjpe_mm"]; save(args.output_dir/"best.pt",model,optimizer,epoch,validation,args)
        print(json.dumps(record),flush=True)


if __name__ == "__main__":
    main()
