"""Evaluate a V11 checkpoint on the complete 3DPW test window protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from divibody_training.evaluate_standard_3dpw import create_smpl_models
from divibody_training.train_divibody_rgb_vace_hmr2a_v8 import validate
from divibody_training.train_divibody_rgb_vace_hmr2a_v9_small import parse_args as _unused
from models.human_decoder_rgb_vace_v8_hmr2a import HumanRGBVaceGeometryTransformerV8HMR2A


class FullTestWindows(Dataset):
    def __init__(self, vace, sidecar, gt_cache):
        self.vace, self.sidecar, self.gt = vace, sidecar, gt_cache
        if vace["signature"]["split"] != "test" or sidecar["signature"]["split"] != "test":
            raise ValueError("Both caches must be test split")
        if len(vace["metadata"]) != len(sidecar["window_uids"]):
            raise ValueError("Window count mismatch")
        gt_lookup = {
            f"{seq}/person_{int(pid)}/frame_{int(fid):05d}": index
            for index, (seq, pid, fid) in enumerate(
                zip(gt_cache["sequences"], gt_cache["person_id"], gt_cache["frame_id"])
            )
        }
        missing = [uid for uid in sidecar["frame_uids"] if uid not in gt_lookup]
        if missing:
            raise ValueError(f"GT identity mismatch: {missing[:3]}")
        self.gt_indices = torch.tensor([gt_lookup[x] for x in sidecar["frame_uids"]], dtype=torch.long)

    def __len__(self):
        return len(self.sidecar["window_uids"])

    def __getitem__(self, index):
        frames = self.sidecar["window_frame_indices"][index]
        gt_frames = self.gt_indices[frames]
        return {
            "rgb_tokens": self.sidecar["rgb_tokens"][frames],
            "vace_features": self.vace["features"][index],
            "base_rotmat": self.sidecar["base_rotmat"][frames],
            "base_betas": self.sidecar["base_betas"][frames],
            "base_camera": self.sidecar["base_camera"][frames],
            "full_pose": self.gt["gt_full_pose"][gt_frames],
            "betas": self.gt["gt_betas"][gt_frames],
            "gender_code": self.gt["gt_gender"][gt_frames],
            "cam_pose": self.gt["gt_cam_pose"][gt_frames],
            "frame_owner": self.vace["frame_owner_mask"][index],
            "velocity_owner": self.vace["velocity_owner_mask"][index],
            "acceleration_owner": self.vace["acceleration_owner_mask"][index],
        }


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--vace-cache", type=Path, required=True)
    p.add_argument("--hmr-sidecar", type=Path, required=True)
    p.add_argument("--gt-cache", type=Path, required=True)
    p.add_argument("--smpl-model-root", type=Path, required=True)
    p.add_argument("--joint-regressor", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--amp-dtype", default="bf16")
    args = p.parse_args()
    device = torch.device(args.device)
    vace, sidecar, gt = load(args.vace_cache), load(args.hmr_sidecar), load(args.gt_cache)
    data = FullTestWindows(vace, sidecar, gt)
    model_payload = load(args.checkpoint)
    model = HumanRGBVaceGeometryTransformerV8HMR2A().to(device)
    model.load_state_dict(model_payload["model"], strict=True)
    smpl = create_smpl_models(args.smpl_model_root, device)
    for value in smpl.values():
        value.requires_grad_(False)
    regressor = torch.as_tensor(np.load(args.joint_regressor), dtype=torch.float32, device=device)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=args.num_workers > 0)
    metrics = validate(loader, model, smpl, regressor, args, device)
    expected = {"frames": 35515, "acceleration_triplets": 34752}
    for key, value in expected.items():
        if int(metrics[key]) != value:
            raise RuntimeError(f"Incomplete metric coverage {key}={metrics[key]} expected={value}")
    result = {
        "status": "RAM_V11_FULL_3DPW_TEST_OK",
        "model": "DiviBody RGB-VACE HMR2a V11 raw-bbox",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(model_payload["epoch"]),
        "metrics": metrics,
        "coverage": {"person_frames": 35515, "sequences": 24, "tracks": 37,
                     "acceleration_triplets": 34752},
        "protocol": "complete strict 3DPW test; H36M J14; pelvis aligned MPJPE/PVE; Procrustes PA-MPJPE",
        "test_used_for_model_selection": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
