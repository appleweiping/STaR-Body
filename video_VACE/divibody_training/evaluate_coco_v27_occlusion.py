"""Evaluate v27 under deterministic synthetic occlusion on COCO2017 val crops."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from divibody_training.evaluate_standard_3dpw import create_smpl_models
from divibody_training.hmr2a_anchor import HMR2AAnchorExtractor
from divibody_training.ram_parts_runtime_v27 import model_from_payload
from divibody_training.real_vace_temporal_feature_encoder_v2 import (
    ProtocolAwareTemporalVaceFeatureEncoder,
)
from divibody_training.train_ram_joint_sharded_pose_geometry_v7 import (
    projected_posetrack,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vace-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--hmr-source-root", type=Path, required=True)
    parser.add_argument("--hmr2a-checkpoint", type=Path, required=True)
    parser.add_argument("--hmr-model-config", type=Path, required=True)
    parser.add_argument("--hmr-data-root", type=Path, required=True)
    parser.add_argument("--smpl-model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp-dtype", choices=("bf16", "none"), default="bf16")
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--max-people", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--occlusion",
        choices=("none", "upper", "lower"),
        required=True,
        help="Fixed crop-space occlusion applied before both HMR and VACE encoding.",
    )
    return parser.parse_args()


class CocoPersons:
    def __init__(self, root: Path, image_size: int = 256, scale: float = 1.25):
        self.root = Path(root)
        self.image_size = int(image_size)
        self.scale = float(scale)
        annotations = json.loads(
            (self.root / "annotations/person_keypoints_val2017.json").read_text()
        )
        images = {item["id"]: item for item in annotations["images"]}
        self.samples = []
        for annotation in annotations["annotations"]:
            keypoints = np.asarray(annotation.get("keypoints", []), np.float32)
            bbox = np.asarray(annotation.get("bbox", []), np.float32)
            if (
                annotation.get("iscrowd", 0)
                or keypoints.size != 51
                or bbox.size != 4
                or bbox[2] <= 1
                or bbox[3] <= 1
            ):
                continue
            keypoints = keypoints.reshape(17, 3)
            if not (keypoints[:, 2] > 0).any():
                continue
            image = images[annotation["image_id"]]
            path = self.root / "val2017" / image["file_name"]
            if path.is_file():
                self.samples.append((path, bbox, keypoints, int(annotation["id"])))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        path, bbox, keypoints, annotation_id = self.samples[index]
        x, y, width, height = map(float, bbox)
        side = max(width, height) * self.scale
        center_x, center_y = x + width / 2, y + height / 2
        box = (
            center_x - side / 2,
            center_y - side / 2,
            center_x + side / 2,
            center_y + side / 2,
        )
        with Image.open(path) as source:
            crop = source.convert("RGB").crop(box).resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            image = np.asarray(crop, dtype=np.float32) / 255.0
        target = keypoints.copy()
        target[:, :2] = (
            (target[:, :2] - np.asarray(box[:2], np.float32))
            * self.image_size
            / side
        )
        return {
            "image": torch.from_numpy(image.copy()).permute(2, 0, 1),
            "keypoints": torch.from_numpy(target),
            "annotation_id": annotation_id,
        }


def empty_totals() -> dict[str, dict[str, object]]:
    return {
        name: {
            "correct05": [0] * 8,
            "correct10": [0] * 8,
            "error_sum": [0.0] * 8,
            "count": [0] * 8,
        }
        for name in ("anchor", "direct", "assisted")
    }


def add_metrics(totals, name, prediction, target, visible) -> None:
    error = torch.linalg.vector_norm(prediction - target, dim=-1) / (1.0 / 1.25)
    for frame in range(8):
        selected = error[0, frame].masked_select(visible)
        totals[name]["correct05"][frame] += int((selected <= 0.05).sum())
        totals[name]["correct10"][frame] += int((selected <= 0.10).sum())
        totals[name]["error_sum"][frame] += float(selected.double().sum())
        totals[name]["count"][frame] += selected.numel()


def summarize(values: dict[str, object]) -> dict[str, object]:
    per_position = []
    for index in range(8):
        count = int(values["count"][index])
        if count <= 0:
            raise RuntimeError("COCO evaluation position has no visible keypoints")
        per_position.append(
            {
                "position": index,
                "PCKn@0.05": values["correct05"][index] / count,
                "PCKn@0.10": values["correct10"][index] / count,
                "mean_normalized_joint_error": values["error_sum"][index] / count,
                "visible_keypoints": count,
            }
        )
    total = sum(values["count"])
    return {
        "PCKn@0.05": sum(values["correct05"]) / total,
        "PCKn@0.10": sum(values["correct10"]) / total,
        "mean_normalized_joint_error": sum(values["error_sum"]) / total,
        "visible_keypoint_predictions": total,
        "per_temporal_position": per_position,
    }


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def apply_occlusion(image: torch.Tensor, mode: str) -> torch.Tensor:
    """Cover a fixed body region with the ImageNet-neutral mid-gray value."""
    if mode == "none":
        return image
    occluded = image.clone()
    height, width = occluded.shape[-2:]
    if mode == "upper":
        y0, y1, x0, x1 = 0.08, 0.55, 0.12, 0.88
    elif mode == "lower":
        y0, y1, x0, x1 = 0.45, 0.96, 0.18, 0.82
    else:
        raise ValueError(f"Unsupported occlusion mode: {mode}")
    occluded[
        :,
        round(y0 * height) : round(y1 * height),
        round(x0 * width) : round(x1 * width),
    ] = 0.5
    return occluded


def main() -> None:
    args = parse_args()
    if args.max_people < 0 or args.start_index < 0 or args.log_interval <= 0:
        raise ValueError("Invalid count/index/log arguments")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    dataset = CocoPersons(args.coco_root)
    stop = min(len(dataset), args.start_index + args.max_people) if args.max_people else len(dataset)
    if not 0 <= args.start_index < stop:
        raise ValueError("Requested COCO evaluation range is empty")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") != "ram-v9-geometry-coupled-v27":
        raise ValueError("This evaluator requires a native v27 checkpoint")
    device = torch.device(args.device)
    model = model_from_payload(payload).to(device).eval()
    geometry_models = create_smpl_models(args.smpl_model_root, device)
    for value in geometry_models.values():
        value.requires_grad_(False)
    model.bind_geometry(geometry_models[2])
    from hmr2.models.smpl_wrapper import SMPL as OfficialSMPL

    official_smpl = OfficialSMPL(
        model_path=str(args.smpl_model_root),
        gender="neutral",
        num_betas=10,
        joint_regressor_extra=str(args.smpl_model_root.parent / "SMPL_to_J19.pkl"),
    ).to(device).requires_grad_(False)
    encoder = ProtocolAwareTemporalVaceFeatureEncoder(
        checkpoint_dir=args.vace_checkpoint_dir,
        device=args.device,
        image_size=256,
        feature_layers=(7, 14, 22),
        timestep=0,
        diffusion_input_protocol="clean",
        text_embedding_path=None,
    )

    totals = empty_totals()
    next_index = args.start_index
    if partial.exists():
        state = json.loads(partial.read_text())
        if state["checkpoint"] != str(args.checkpoint.resolve()) or state["stop"] != stop:
            raise RuntimeError("Partial COCO result belongs to another configuration")
        totals = state["totals"]
        next_index = int(state["next_index"])

    started = time.monotonic()
    with HMR2AAnchorExtractor(
        source_root=args.hmr_source_root,
        checkpoint=args.hmr2a_checkpoint,
        model_config=args.hmr_model_config,
        hmr2_data_root=args.hmr_data_root,
        device=device,
    ) as anchor_extractor, torch.inference_mode():
        for index in range(next_index, stop):
            sample = dataset[index]
            image = apply_occlusion(sample["image"], args.occlusion).unsqueeze(0)
            anchor = anchor_extractor.encode(image)
            clip = image.unsqueeze(1).repeat(1, 17, 1, 1, 1)
            vace = encoder.encode(clip)
            frames = 8
            rgb = anchor["rgb_tokens"].unsqueeze(1).repeat(1, frames, 1, 1)
            rotations = anchor["base_rotmat"].unsqueeze(1).repeat(1, frames, 1, 1, 1)
            betas = anchor["base_betas"].unsqueeze(1).repeat(1, frames, 1)
            camera = anchor["base_camera"].unsqueeze(1).repeat(1, frames, 1)
            use_amp = device.type == "cuda" and args.amp_dtype == "bf16"
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                prediction = model(
                    rgb.to(device),
                    vace.to(device),
                    rotations.to(device),
                    betas.to(device),
                    camera.to(device),
                    frame_valid=torch.ones(1, frames, dtype=torch.bool, device=device),
                )
            anchor_prediction = {
                "rotmat": rotations.to(device),
                "betas": betas.to(device),
                "camera": camera.to(device),
            }
            anchor_2d = projected_posetrack(anchor_prediction, official_smpl)
            direct_2d = projected_posetrack(prediction, official_smpl)
            assisted_2d = direct_2d + prediction["posetrack_2d_residual"]
            target = sample["keypoints"].to(device).float()
            visible = target[:, 2] > 0
            target = (target[:, :2] / 256.0 - 0.5)[None, None].repeat(1, frames, 1, 1)
            add_metrics(totals, "anchor", anchor_2d, target, visible)
            add_metrics(totals, "direct", direct_2d, target, visible)
            add_metrics(totals, "assisted", assisted_2d, target, visible)
            next_index = index + 1
            if next_index % args.log_interval == 0 or next_index == stop:
                state = {
                    "checkpoint": str(args.checkpoint.resolve()),
                    "next_index": next_index,
                    "stop": stop,
                    "totals": totals,
                }
                atomic_json(partial, state)
                print(
                    json.dumps(
                        {
                            "people": next_index - args.start_index,
                            "total": stop - args.start_index,
                            "seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )

    result = {
        "status": "RAM_V27_COCO2017_CONTROLLED_OCCLUSION_OK",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_variant": payload.get("variant"),
        "metrics": {name: summarize(values) for name, values in totals.items()},
        "coverage": {
            "people": stop - args.start_index,
            "dataset_people": len(dataset),
            "development_only": bool(args.max_people or args.start_index),
        },
        "occlusion": {
            "mode": args.occlusion,
            "fill_rgb_01": 0.5,
            "upper_box_xyxy_fraction": [0.12, 0.08, 0.88, 0.55],
            "lower_box_xyxy_fraction": [0.18, 0.45, 0.82, 0.96],
        },
        "protocol": (
            "Controlled robustness proxy on the first fixed COCO2017-val person subset: "
            "GT square crop scaled by 1.25, deterministic crop-space occlusion before "
            "both encoders, one crop repeated for all 17 diffusion-input frames; metrics "
            "average eight temporal positions; direct SMPL projection is primary"
        ),
        "test_used_for_model_selection": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    partial.unlink(missing_ok=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
