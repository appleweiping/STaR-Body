"""Small-sample v9 experiment before authorization of a full 3DPW train run."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from divibody_training.evaluate_standard_3dpw import create_smpl_models
from divibody_training.train_divibody_rgb_vace_hmr2a_v8 import (
    V8Dataset,
    load_pair,
    validate,
)
from divibody_training.train_divibody_rgb_vace_v7 import (
    forward_batch,
    geometric_losses,
)
from divibody_training.train_divibody_continuous_geometry_v5 import camera_targets
from models.human_decoder_rgb_vace_v9_active_temporal import (
    HumanRGBVaceGeometryTransformerV9ActiveTemporal,
    RGBVaceActiveTemporalConfig,
)


FORMAT = "divibody-rgb-vace-v9-active-temporal-small-v1"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "val"):
        p.add_argument(f"--{split}-vace-cache", type=Path, required=True)
        p.add_argument(f"--{split}-hmr-sidecar", type=Path, required=True)
    p.add_argument("--init-checkpoint", type=Path, required=True)
    p.add_argument("--smpl-model-root", type=Path, required=True)
    p.add_argument("--joint-regressor", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--max-train-windows", type=int, default=384)
    p.add_argument("--max-val-windows", type=int, default=192)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--val-batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--joint-weight", type=float, default=1.0)
    p.add_argument("--vertex-weight", type=float, default=0.5)
    p.add_argument("--rotation-weight", type=float, default=0.05)
    p.add_argument("--shape-weight", type=float, default=0.001)
    p.add_argument("--velocity-weight", type=float, default=0.75)
    p.add_argument("--acceleration-weight", type=float, default=1.5)
    p.add_argument("--anchor-weight", type=float, default=0.05)
    p.add_argument("--root-orientation-weight", type=float, default=0.35)
    p.add_argument("--shape-consistency-weight", type=float, default=0.02)
    p.add_argument("--temporal-gate", type=float, default=0.15)
    p.add_argument("--amp-dtype", choices=("bf16", "none"), default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def spread_subset(dataset, maximum: int):
    if maximum <= 0 or maximum >= len(dataset):
        return dataset
    indices = np.linspace(0, len(dataset) - 1, maximum, dtype=np.int64).tolist()
    if len(set(indices)) != maximum:
        raise RuntimeError("Non-unique spread subset")
    return Subset(dataset, indices)


def v9_losses(prediction, batch, smpl_models, regressor, args, device):
    losses = geometric_losses(prediction, batch, smpl_models, regressor, args, device)
    with torch.autocast(device_type=device.type, enabled=False):
        target_rot = camera_targets(batch, device)
        frame_mask = batch["frame_owner"].to(device)
        root_chordal = (prediction["rotmat"].float()[:, :, 0] - target_rot[:, :, 0]).square().mean((-1, -2))
        root = root_chordal.masked_select(frame_mask).mean()
        betas = prediction["betas"].float()
        valid = frame_mask.unsqueeze(-1)
        mean_shape = (betas * valid).sum(1, keepdim=True) / valid.sum(1, keepdim=True).clamp_min(1)
        consistency = ((betas - mean_shape.detach()).square() * valid).sum() / valid.sum().clamp_min(1)
        total = losses["total"] + args.root_orientation_weight * root + args.shape_consistency_weight * consistency
    return {**losses, "total": total, "root_orientation_chordal": root, "shape_consistency": consistency}


def save(path, model, optimizer, epoch, validation, args):
    payload = {
        "format": FORMAT, "epoch": epoch, "validation": validation,
        "selection_metric": "validation_mpjpe_mm", "model": model.state_dict(),
        "model_config": vars(model.config), "anchor_metadata": model.anchor_metadata(),
        "optimizer": optimizer.state_dict(), "test_used_for_model_selection": False,
        "small_sample": True,
        "args": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
    }
    tmp = path.with_suffix(path.suffix + ".tmp"); torch.save(payload, tmp); os.replace(tmp, path)


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite a v9 experiment")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_vace, train_hmr = load_pair(args.train_vace_cache, args.train_hmr_sidecar, "train")
    val_vace, val_hmr = load_pair(args.val_vace_cache, args.val_hmr_sidecar, "validation")
    train_data = spread_subset(V8Dataset(train_vace, train_hmr), args.max_train_windows)
    val_data = spread_subset(V8Dataset(val_vace, val_hmr), args.max_val_windows)
    cfg = RGBVaceActiveTemporalConfig(temporal_gate_init=args.temporal_gate)
    model = HumanRGBVaceGeometryTransformerV9ActiveTemporal(cfg)
    checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.activate_temporal_path(); model.to(device)
    smpl_models = create_smpl_models(args.smpl_model_root, device)
    for value in smpl_models.values(): value.requires_grad_(False)
    regressor = torch.as_tensor(np.load(args.joint_regressor), dtype=torch.float32, device=device)
    options = {"num_workers":args.num_workers,"pin_memory":device.type=="cuda","persistent_workers":args.num_workers>0}
    train_loader = DataLoader(train_data,batch_size=args.batch_size,shuffle=True,**options)
    val_loader = DataLoader(val_data,batch_size=args.val_batch_size,shuffle=False,**options)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.lr,weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics=args.output_dir/"metrics.jsonl"
    baseline=validate(val_loader,model,smpl_models,regressor,args,device)
    save(args.output_dir/"best.pt",model,optimizer,0,baseline,args)
    metrics.write_text(json.dumps({"epoch":0,"validation":baseline})+"\n",encoding="utf-8")
    print(json.dumps({"epoch":0,"validation":baseline}),flush=True); best=baseline["mpjpe_mm"]
    for epoch in range(1,args.epochs+1):
        model.train(); totals={}; batches=0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True); pred=forward_batch(model,batch,device,args.amp_dtype)
            losses=v9_losses(pred,batch,smpl_models,regressor,args,device)
            if not torch.isfinite(losses["total"]): raise FloatingPointError("Non-finite v9 loss")
            losses["total"].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip); optimizer.step()
            for k,v in losses.items(): totals[k]=totals.get(k,0.0)+float(v.detach())
            batches+=1
        result=validate(val_loader,model,smpl_models,regressor,args,device)
        record={"epoch":epoch,"train":{k:v/batches for k,v in totals.items()},"validation":result,"temporal_gate":model.temporal_gates.detach().tanh().cpu().tolist()}
        with metrics.open("a",encoding="utf-8") as f: f.write(json.dumps(record)+"\n")
        save(args.output_dir/"latest.pt",model,optimizer,epoch,result,args)
        if result["mpjpe_mm"]<best: best=result["mpjpe_mm"]; save(args.output_dir/"best.pt",model,optimizer,epoch,result,args)
        print(json.dumps(record),flush=True)


if __name__ == "__main__": main()
