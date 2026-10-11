"""Full public-GT PoseTrack18 val pose evaluation for the V11 HMR2a anchor.

V11's temporal VACE branch requires a separate PoseTrack VACE cache.  This
evaluator measures the unchanged RGB/HMR2a anchor on every valid public val
person-frame and does not claim HOTA/IDF1 or hidden-test-server results.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from divibody_training.evaluate_unified_hmr2_divibody_3dpw import load_hmr_model
from divibody_training.ram_ready_adapter_v1 import PoseTrack18PersonDataset


# HMR2 SMPL wrapper returns OpenPose-25 first.  PoseTrack-specific head points
# are formed deterministically: head_bottom=neck, head_top=2*nose-neck.
PT_FROM_OP = {0: 0, 1: 1, 3: 18, 4: 17, 5: 5, 6: 2, 7: 6, 8: 3,
              9: 7, 10: 4, 11: 12, 12: 9, 13: 13, 14: 10, 15: 14, 16: 11}


def pose_track_from_openpose(openpose):
    result = openpose.new_zeros((openpose.shape[0], 17, 2))
    for target, source in PT_FROM_OP.items():
        result[:, target] = openpose[:, source]
    result[:, 2] = 2.0 * openpose[:, 0] - openpose[:, 1]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ready-root", type=Path, required=True)
    p.add_argument("--hmr-source-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model-config", type=Path, required=True)
    p.add_argument("--hmr-data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--amp-dtype", default="bf16")
    args = p.parse_args()
    # Match the namespace consumed by the shared audited HMR loader.
    args.hmr_model_config = args.model_config
    device = torch.device(args.device)
    dataset = PoseTrack18PersonDataset(args.ready_root, split="val", image_size=256,
                                       bbox_scale=1.25, min_keypoints=1)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=args.num_workers > 0)
    model = load_hmr_model(args, args.checkpoint).to(device).eval()
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    correct05 = correct10 = visible_total = people = 0
    error_sum = 0.0
    started = time.monotonic()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda" and args.amp_dtype == "bf16"):
                output = model({"img": (images - mean) / std})
            pred = pose_track_from_openpose(output["pred_keypoints_2d"][:, :25].float())
            pred = (pred + 0.5) * 256.0
            gt = batch["keypoints"].to(device).float()
            visible = gt[:, :, 2] > 0
            error = torch.linalg.vector_norm(pred - gt[:, :, :2], dim=-1)
            # Crop side=max(bbox width,height)*1.25, hence bbox-max in crop pixels is 256/1.25.
            normalised = error / (256.0 / 1.25)
            values = normalised.masked_select(visible)
            correct05 += int((values <= 0.05).sum())
            correct10 += int((values <= 0.10).sum())
            visible_total += int(values.numel())
            error_sum += float(values.double().sum())
            people += int(images.shape[0])
            if (batch_index + 1) % 50 == 0 or people == len(dataset):
                print(json.dumps({"processed": people, "total": len(dataset),
                                  "seconds": time.monotonic() - started}), flush=True)
    if people != len(dataset) or visible_total != dataset.audit()["visible_keypoints"]:
        raise RuntimeError("Incomplete PoseTrack18 validation coverage")
    result = {
        "status": "HMR2A_FULL_POSETRACK18_PUBLIC_VAL_OK",
        "model": "HMR 2.0a RGB anchor used by RAM/DiviBody V11",
        "metrics": {
            "PCKn@0.05": correct05 / visible_total,
            "PCKn@0.10": correct10 / visible_total,
            "mean_normalized_joint_error": error_sum / visible_total,
        },
        "coverage": dataset.audit(),
        "head_mapping": "OpenPose25; head_bottom=neck; head_top=2*nose-neck",
        "normalizer": "max GT bbox side; GT person crop scale=1.25",
        "limitations": [
            "Public PoseTrack18 validation GT, not hidden test-server GT",
            "V11 temporal VACE branch is not measured until a PoseTrack VACE cache is generated",
            "No detector/tracker is run, so HOTA/MOTA/IDF1 are not claimed",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
