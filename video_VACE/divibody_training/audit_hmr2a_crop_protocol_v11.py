"""Validation-only HMR2a crop sweep before rebuilding DiviBody anchors.

The legacy path squared a keypoint box before HMR2's own aspect expansion.
This audit compares it with the official-style raw rectangular box.  It never
accepts the 3DPW test split and makes no model-selection claim on test data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from divibody_training.evaluate_unified_hmr2_divibody_3dpw import (
    UnifiedHMR3DPWDataset,
    axis_angle_to_matrix,
    create_smpl_models,
    gendered_vertices,
    smpl_vertices,
)
from divibody_training.geometry_metrics_3dpw import (
    mean_euclidean_error,
    pa_mpjpe_per_sample,
    pelvis_aligned_j14,
)
from divibody_training.hmr2a_anchor import HMR2AAnchorExtractor
from divibody_training.real_3dpw_dataset import _to_numpy


FORMAT = "hmr2a-3dpw-validation-crop-audit-v11"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--hmr-source-root", type=Path, required=True)
    p.add_argument("--hmr2a-checkpoint", type=Path, required=True)
    p.add_argument("--hmr-model-config", type=Path, required=True)
    p.add_argument("--hmr-data-root", type=Path, required=True)
    p.add_argument("--smpl-model-root", type=Path, required=True)
    p.add_argument("--joint-regressor", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


class RawBBoxHMR3DPWDataset(UnifiedHMR3DPWDataset):
    """Return a rectangular keypoint box; HMR performs the only squaring."""

    def _person_bbox(self, annotation, person_index, frame_index, width, height):
        poses2d = annotation.get("poses2d")
        if poses2d is None or person_index >= len(poses2d):
            return None
        points = np.asarray(poses2d[person_index])
        if frame_index >= len(points):
            return None
        points = np.asarray(points[frame_index], dtype=np.float32)
        if points.ndim != 2:
            return None
        if points.shape[0] == 3:
            xy, confidence = points[:2].T, points[2]
        elif points.shape[1] >= 3:
            xy, confidence = points[:, :2], points[:, 2]
        else:
            return None
        visible = np.isfinite(xy).all(1) & np.isfinite(confidence) & (confidence > 0)
        xy = xy[visible]
        if len(xy) < 2:
            return None
        low, high = xy.min(0), xy.max(0)
        size = high - low
        if np.any(size < 2):
            return None
        low = low - size * self.crop_margin
        high = high + size * self.crop_margin
        return float(low[0]), float(low[1]), float(high[0]), float(high[1])


class FullImageHMR3DPWDataset(RawBBoxHMR3DPWDataset):
    def _person_bbox(self, annotation, person_index, frame_index, width, height):
        return 0.0, 0.0, float(width), float(height)


def build_dataset(root: Path, protocol: str):
    if protocol == "legacy_square_margin25":
        cls, margin = UnifiedHMR3DPWDataset, 0.25
    elif protocol == "full_image":
        cls, margin = FullImageHMR3DPWDataset, 0.0
    else:
        cls = RawBBoxHMR3DPWDataset
        margin = float(protocol.removeprefix("raw_margin")) / 100.0
    return cls(root=root, split="validation", image_size=256, frame_stride=1,
               max_samples=None, crop_to_person=True, crop_margin=margin)


@torch.inference_mode()
def evaluate_protocol(dataset, indices, extractor, models, regressor, args):
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers,
                        pin_memory=True, persistent_workers=args.num_workers > 0)
    sums = {k: 0.0 for k in ("mpjpe_mm", "pa_mpjpe_mm", "pve_mm")}
    count = 0
    device = torch.device(args.device)
    for batch in loader:
        out = extractor.encode(batch["image"])
        pred_rot = out["base_rotmat"].float()
        pred_betas = out["base_betas"].float()
        gt_rot = axis_angle_to_matrix(batch["full_pose"].to(device).reshape(-1,24,3))
        gt_rot[:,0] = batch["cam_pose"].to(device)[:,:3,:3] @ gt_rot[:,0]
        gt_betas = batch["betas"].to(device)
        genders = batch["gender_code"].to(device)
        pv = smpl_vertices(models[2], pred_rot, pred_betas)
        gv = gendered_vertices(models, gt_rot, gt_betas, genders)
        pj,_ = pelvis_aligned_j14(pv,regressor); gj,_ = pelvis_aligned_j14(gv,regressor)
        values = {
            "mpjpe_mm": mean_euclidean_error(pj,gj)*1000,
            "pa_mpjpe_mm": pa_mpjpe_per_sample(pj,gj)*1000,
            "pve_mm": mean_euclidean_error(pv,gv)*1000,
        }
        for key,value in values.items(): sums[key] += float(value.double().sum())
        count += len(pred_rot)
    return {key:value/count for key,value in sums.items()} | {"frames":count}


def main():
    args=parse_args()
    if args.samples < 1: raise ValueError("samples must be positive")
    protocols=("legacy_square_margin25","raw_margin0","raw_margin10",
               "raw_margin25","raw_margin40","full_image")
    reference=build_dataset(args.data_root,protocols[0])
    count=min(args.samples,len(reference))
    indices=np.linspace(0,len(reference)-1,count,dtype=np.int64).tolist()
    device=torch.device(args.device)
    models=create_smpl_models(args.smpl_model_root,device)
    regressor=torch.as_tensor(np.load(args.joint_regressor),dtype=torch.float32,device=device)
    results={}
    with HMR2AAnchorExtractor(source_root=args.hmr_source_root,
            checkpoint=args.hmr2a_checkpoint,model_config=args.hmr_model_config,
            hmr2_data_root=args.hmr_data_root,device=device) as extractor:
        for protocol in protocols:
            dataset=build_dataset(args.data_root,protocol)
            reference_ids=[(reference.samples[i].sequence,reference.samples[i].person_index,reference.samples[i].frame_index) for i in indices]
            actual_ids=[(dataset.samples[i].sequence,dataset.samples[i].person_index,dataset.samples[i].frame_index) for i in indices]
            if actual_ids != reference_ids: raise RuntimeError("Protocol sample identities differ")
            results[protocol]=evaluate_protocol(dataset,indices,extractor,models,regressor,args)
            print(json.dumps({"protocol":protocol,**results[protocol]}),flush=True)
    best=min(results,key=lambda x:results[x]["mpjpe_mm"])
    payload={"format":FORMAT,"split":"validation","selection_metric":"mpjpe_mm",
             "uniform_samples":count,"protocols":results,"best_protocol":best,
             "test_data_used":False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2))


if __name__=="__main__": main()
