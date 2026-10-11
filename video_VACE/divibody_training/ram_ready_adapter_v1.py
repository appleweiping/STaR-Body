"""Additive dataset adapters for the unified RAM ``datasets_ram/ready`` tree.

This module intentionally does not edit or replace any existing DiviBody, VACE,
HMR2, or dataset implementation.  It provides strict path validation, a COCO
PoseTrack person-frame adapter, and reproducible audit summaries.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


POSETRACK_JOINTS = 17


@dataclass(frozen=True)
class RAMReadyPaths:
    root: Path
    threedpw: Path
    coco2017: Path
    insta_variety: Path
    posetrack18: Path
    posetrack21: Path
    trackid3x3: Path

    @classmethod
    def discover(cls, root: str | Path) -> "RAMReadyPaths":
        root = Path(root).expanduser().resolve()
        paths = cls(
            root=root,
            threedpw=root / "3DPW",
            coco2017=root / "COCO2017",
            insta_variety=root / "InstaVariety",
            posetrack18=root / "PoseTrack18",
            posetrack21=root / "PoseTrack21",
            trackid3x3=root / "TrackID3x3",
        )
        missing = [name for name, value in vars(paths).items() if name != "root" and not value.exists()]
        if missing:
            raise FileNotFoundError(f"RAM ready tree is incomplete at {root}: {missing}")
        return paths

    def posetrack18_annotation(self, split: str) -> Path:
        split = normalise_posetrack_split(split)
        path = self.posetrack18 / "annotations" / f"posetrack18_{split}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path


def normalise_posetrack_split(split: str) -> str:
    split = split.strip().lower()
    aliases = {"validation": "val", "valid": "val", "val": "val", "train": "train"}
    if split not in aliases:
        raise ValueError("PoseTrack18 local GT supports only train/val")
    return aliases[split]


class PoseTrack18PersonDataset(Dataset):
    """Public-GT PoseTrack18 person frames with deterministic HMR-style crops.

    The hidden test annotations are not public, so this adapter deliberately
    supports only train/val.  Each item retains the original COCO keypoints,
    track identity, video identity, frame identity, bbox and crop transform.
    """

    def __init__(
        self,
        ready_root: str | Path,
        split: str = "val",
        image_size: int = 256,
        bbox_scale: float = 1.25,
        min_keypoints: int = 1,
        max_samples: int | None = None,
    ) -> None:
        super().__init__()
        self.paths = RAMReadyPaths.discover(ready_root)
        self.split = normalise_posetrack_split(split)
        self.image_size = int(image_size)
        self.bbox_scale = float(bbox_scale)
        self.min_keypoints = int(min_keypoints)
        if self.image_size <= 0 or self.bbox_scale <= 0 or self.min_keypoints < 0:
            raise ValueError("Invalid PoseTrack adapter arguments")

        annotation_path = self.paths.posetrack18_annotation(self.split)
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        self.images = {int(item["id"]): item for item in payload["images"]}
        samples: list[dict[str, Any]] = []
        for ann in payload["annotations"]:
            image = self.images.get(int(ann["image_id"]))
            if image is None or int(ann.get("iscrowd", 0)) != 0:
                continue
            keypoints = np.asarray(ann.get("keypoints", ()), dtype=np.float32)
            if keypoints.size != POSETRACK_JOINTS * 3:
                continue
            keypoints = keypoints.reshape(POSETRACK_JOINTS, 3)
            visible = int((keypoints[:, 2] > 0).sum())
            bbox = np.asarray(ann.get("bbox", ()), dtype=np.float32).reshape(-1)
            if visible < self.min_keypoints or bbox.size != 4 or bbox[2] <= 1 or bbox[3] <= 1:
                continue
            image_width, image_height = int(image["width"]), int(image["height"])
            if (
                not np.isfinite(bbox).all()
                or bbox[2] > 4 * image_width
                or bbox[3] > 4 * image_height
                or abs(float(bbox[0])) > 4 * image_width
                or abs(float(bbox[1])) > 4 * image_height
            ):
                continue
            image_path = self.paths.posetrack18 / str(image["file_name"])
            if not image_path.is_file():
                continue
            samples.append(
                {
                    "annotation_id": int(ann["id"]),
                    "image_id": int(image["id"]),
                    "image_path": image_path,
                    "video_id": str(image.get("vid_id", "")),
                    "frame_id": int(image.get("frame_id", image["id"])),
                    "track_id": int(ann.get("track_id", -1)),
                    "bbox": bbox,
                    "keypoints": keypoints,
                    "width": image_width,
                    "height": image_height,
                }
            )
        samples.sort(key=lambda x: (x["video_id"], x["frame_id"], x["track_id"], x["annotation_id"]))
        self.samples = samples if not max_samples or max_samples <= 0 else samples[: int(max_samples)]
        self.missing_images = len(payload["annotations"]) - len(samples)

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _square_crop(bbox: np.ndarray, scale: float) -> tuple[float, float, float]:
        x, y, width, height = (float(value) for value in bbox)
        side = max(width, height) * scale
        return x + width / 2.0, y + height / 2.0, side

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        center_x, center_y, side = self._square_crop(sample["bbox"], self.bbox_scale)
        crop_box = (
            center_x - side / 2.0,
            center_y - side / 2.0,
            center_x + side / 2.0,
            center_y + side / 2.0,
        )
        with Image.open(sample["image_path"]) as source:
            image = source.convert("RGB").crop(crop_box).resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            rgb = np.asarray(image, dtype=np.float32) / 255.0
        keypoints = sample["keypoints"].copy()
        keypoints[:, 0] = (keypoints[:, 0] - crop_box[0]) * self.image_size / side
        keypoints[:, 1] = (keypoints[:, 1] - crop_box[1]) * self.image_size / side
        return {
            "image": torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1),
            "keypoints": torch.from_numpy(keypoints),
            "bbox_xywh": torch.from_numpy(sample["bbox"].copy()),
            "crop_center": torch.tensor((center_x, center_y), dtype=torch.float32),
            "crop_side": torch.tensor(side, dtype=torch.float32),
            "image_path": str(sample["image_path"]),
            "image_id": sample["image_id"],
            "video_id": sample["video_id"],
            "frame_id": sample["frame_id"],
            "track_id": sample["track_id"],
            "annotation_id": sample["annotation_id"],
        }

    def audit(self) -> dict[str, Any]:
        identities = {(s["video_id"], s["track_id"]) for s in self.samples}
        visible = np.asarray([(s["keypoints"][:, 2] > 0).sum() for s in self.samples])
        return {
            "dataset": "PoseTrack18",
            "split": self.split,
            "person_frames": len(self.samples),
            "videos": len({s["video_id"] for s in self.samples}),
            "tracks": len(identities),
            "images": len({s["image_id"] for s in self.samples}),
            "visible_keypoints": int(visible.sum()),
            "mean_visible_keypoints_per_person": float(visible.mean()) if len(visible) else 0.0,
            "missing_or_filtered_annotations": self.missing_images,
            "protocol": "public PoseTrack18 COCO GT; GT person boxes; hidden test GT not used",
        }


def count_files(root: Path, suffixes: Iterable[str]) -> int:
    suffixes = {value.lower() for value in suffixes}
    total = 0
    for directory, _, files in os.walk(root.resolve(), followlinks=True):
        total += sum(1 for name in files if Path(name).suffix.lower() in suffixes)
    return total


def full_ready_audit(paths: RAMReadyPaths, pose_split: str = "val") -> dict[str, Any]:
    pose = PoseTrack18PersonDataset(paths.root, split=pose_split, image_size=64)
    return {
        "status": "RAM_READY_ADAPTER_V1_OK",
        "ready_root": str(paths.root),
        "datasets": {
            "3DPW": {"path": str(paths.threedpw), "pkl": count_files(paths.threedpw, {".pkl"})},
            "COCO2017": {"path": str(paths.coco2017)},
            "InstaVariety": {"path": str(paths.insta_variety), "npy": count_files(paths.insta_variety, {".npy"})},
            "PoseTrack18": pose.audit(),
            "PoseTrack21": {"path": str(paths.posetrack21)},
            "TrackID3x3": {"path": str(paths.trackid3x3), "videos": count_files(paths.trackid3x3 / "videos", {".mov", ".mp4", ".avi", ".mkv"})},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready-root", type=Path, required=True)
    parser.add_argument("--pose-split", default="val")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke-samples", type=int, default=8)
    args = parser.parse_args()
    paths = RAMReadyPaths.discover(args.ready_root)
    result = full_ready_audit(paths, args.pose_split)
    smoke = PoseTrack18PersonDataset(paths.root, args.pose_split, max_samples=args.smoke_samples)
    if smoke:
        first = smoke[0]
        result["smoke"] = {
            "samples": len(smoke),
            "image_shape": list(first["image"].shape),
            "keypoints_shape": list(first["keypoints"].shape),
            "image_finite": bool(torch.isfinite(first["image"]).all()),
            "keypoints_finite": bool(torch.isfinite(first["keypoints"]).all()),
            "first_image": first["image_path"],
        }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
