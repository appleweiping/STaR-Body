"""3DPW person-frame dataset for real-image DiviBody training.

This adapter is intentionally standalone: it does not modify video_VACE or any
of its model files. 3DPW's official demo addresses extracted images by the
annotation *index* (``image_{frame_index:05d}.jpg``), not by ``img_frame_ids``;
the latter is retained as ``source_frame_id`` in the returned sample.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


ImageSize = Union[int, Tuple[int, int], Sequence[int]]


@dataclass(frozen=True)
class _SampleRef:
    annotation_path: str
    image_path: str
    sequence: str
    person_index: int
    frame_index: int
    source_frame_id: int


@lru_cache(maxsize=8)
def _load_annotation(path: str) -> Dict[str, Any]:
    """Load one Python-2-compatible 3DPW pickle, with a small worker-local LRU."""

    with open(path, "rb") as handle:
        value = pickle.load(handle, encoding="latin1")
    if not isinstance(value, dict):
        raise TypeError(f"Expected a dict in 3DPW annotation: {path}")
    return value


def _normalise_image_size(image_size: ImageSize) -> Tuple[int, int]:
    if isinstance(image_size, int):
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        return image_size, image_size
    if len(image_size) != 2:
        raise ValueError("image_size must be an int or (height, width)")
    height, width = int(image_size[0]), int(image_size[1])
    if height <= 0 or width <= 0:
        raise ValueError("image_size dimensions must be positive")
    return height, width


def _find_split_directory(root: Path, split: str) -> Path:
    candidates = (
        root / "sequenceFiles" / "sequenceFiles" / split,
        root / "sequenceFiles" / split,
        root / split,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    attempted = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find 3DPW split {split!r}. Tried:\n  {attempted}"
    )


def _image_candidates(
    image_dir: Path, frame_index: int, source_frame_id: int
) -> Iterable[Path]:
    # This first form is the convention used by the official 3DPW demo and by
    # the dataset installed on the target server.
    indexes = (frame_index, source_frame_id)
    seen = set()
    for index in indexes:
        for suffix in (".jpg", ".jpeg", ".png"):
            path = image_dir / f"image_{index:05d}{suffix}"
            if path not in seen:
                seen.add(path)
                yield path


def _to_numpy(value: Any, dtype: np.dtype = np.float32) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


class Real3DPWDataset(Dataset):
    """Enumerate valid 3DPW person/frame examples backed by real RGB images.

    Each item contains these CPU values:

    - ``image``: float32 ``[3, H, W]`` RGB in ``[0, 1]``;
    - ``global_orient``: float32 ``[3]``;
    - ``body_pose``: float32 ``[63]`` (SMPL joints 1..21, i.e. pose[3:66]);
    - ``betas``: float32 ``[10]``;
    - ``transl``: float32 ``[3]``;
    - scalar long ``person_id``/``frame_id`` and string ``sequence``;
    - ``image_path``, ``source_frame_id``, camera tensors and crop metadata.

    ``split='val'`` is an alias for the official ``validation`` split. Splits
    are never silently substituted: a missing split directory raises
    ``FileNotFoundError``; an existing split with no usable images has length 0.
    """

    SPLIT_ALIASES = {"train": "train", "validation": "validation", "val": "validation", "test": "test"}

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        image_size: ImageSize = 128,
        frame_stride: int = 1,
        max_samples: Optional[int] = None,
        crop_to_person: bool = True,
        crop_margin: float = 0.25,
    ) -> None:
        super().__init__()
        self.root = Path(root).expanduser().resolve()
        split_key = str(split).strip().lower()
        if split_key not in self.SPLIT_ALIASES:
            choices = ", ".join(sorted(self.SPLIT_ALIASES))
            raise ValueError(f"Unknown split {split!r}; expected one of: {choices}")
        self.split = self.SPLIT_ALIASES[split_key]
        self.image_height, self.image_width = _normalise_image_size(image_size)
        self.frame_stride = int(frame_stride)
        if self.frame_stride <= 0:
            raise ValueError("frame_stride must be a positive integer")
        if max_samples is not None and int(max_samples) < 0:
            raise ValueError("max_samples must be non-negative or None")
        self.max_samples = None if max_samples is None else int(max_samples)
        self.crop_to_person = bool(crop_to_person)
        self.crop_margin = float(crop_margin)
        if self.crop_margin < 0:
            raise ValueError("crop_margin must be non-negative")

        self.split_directory = _find_split_directory(self.root, self.split)
        self.samples = self._build_index()
        self.num_sequences = len({sample.sequence for sample in self.samples})
        self.num_people = len(
            {(sample.sequence, sample.person_index) for sample in self.samples}
        )

    def _build_index(self) -> list[_SampleRef]:
        samples: list[_SampleRef] = []
        if self.max_samples == 0:
            return samples

        for annotation_path in sorted(self.split_directory.glob("*.pkl")):
            annotation = _load_annotation(str(annotation_path))
            sequence = str(annotation.get("sequence", annotation_path.stem))
            image_dir = self.root / "imageFiles" / sequence
            if not image_dir.is_dir():
                continue

            poses = annotation.get("poses", ())
            translations = annotation.get("trans", ())
            betas = annotation.get("betas", ())
            validity = annotation.get("campose_valid", ())
            source_ids = np.asarray(annotation.get("img_frame_ids", ())).reshape(-1)
            camera_poses = np.asarray(annotation.get("cam_poses", ()))
            num_people = min(len(poses), len(translations), len(betas), len(validity))
            if num_people == 0:
                continue

            for frame_index in range(0, len(source_ids), self.frame_stride):
                if frame_index >= len(camera_poses):
                    continue
                camera_pose = np.asarray(camera_poses[frame_index])
                if camera_pose.shape != (4, 4) or not np.isfinite(camera_pose).all():
                    continue
                source_frame_id = int(source_ids[frame_index])
                image_path = next(
                    (
                        candidate
                        for candidate in _image_candidates(
                            image_dir, frame_index, source_frame_id
                        )
                        if candidate.is_file()
                    ),
                    None,
                )
                if image_path is None:
                    continue

                for person_index in range(num_people):
                    valid = np.asarray(validity[person_index]).reshape(-1)
                    person_poses = np.asarray(poses[person_index])
                    person_trans = np.asarray(translations[person_index])
                    person_betas = np.asarray(betas[person_index]).reshape(-1)
                    if (
                        frame_index >= len(valid)
                        or not bool(valid[frame_index])
                        or frame_index >= len(person_poses)
                        or frame_index >= len(person_trans)
                        or person_poses.shape[-1] < 66
                        or person_trans.shape[-1] < 3
                        or person_betas.size < 10
                    ):
                        continue
                    supervision = (
                        person_poses[frame_index, :66],
                        person_trans[frame_index, :3],
                        person_betas[:10],
                    )
                    if not all(np.isfinite(value).all() for value in supervision):
                        continue
                    samples.append(
                        _SampleRef(
                            annotation_path=str(annotation_path),
                            image_path=str(image_path),
                            sequence=sequence,
                            person_index=person_index,
                            frame_index=frame_index,
                            source_frame_id=source_frame_id,
                        )
                    )
                    if self.max_samples is not None and len(samples) >= self.max_samples:
                        return samples
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _person_bbox(
        self,
        annotation: Dict[str, Any],
        person_index: int,
        frame_index: int,
        width: int,
        height: int,
    ) -> Optional[Tuple[float, float, float, float]]:
        poses2d = annotation.get("poses2d")
        if poses2d is None or person_index >= len(poses2d):
            return None
        person_keypoints = np.asarray(poses2d[person_index])
        if frame_index >= len(person_keypoints):
            return None
        keypoints = np.asarray(person_keypoints[frame_index], dtype=np.float32)
        if keypoints.ndim != 2:
            return None
        if keypoints.shape[0] == 3:
            xy, confidence = keypoints[:2].T, keypoints[2]
        elif keypoints.shape[1] >= 3:
            xy, confidence = keypoints[:, :2], keypoints[:, 2]
        else:
            return None
        visible = (
            np.isfinite(xy).all(axis=1)
            & np.isfinite(confidence)
            & (confidence > 0.0)
        )
        xy = xy[visible]
        if len(xy) < 2:
            return None

        x_min, y_min = xy.min(axis=0)
        x_max, y_max = xy.max(axis=0)
        box_width, box_height = float(x_max - x_min), float(y_max - y_min)
        if box_width < 2.0 or box_height < 2.0:
            return None
        x_min -= box_width * self.crop_margin
        x_max += box_width * self.crop_margin
        y_min -= box_height * self.crop_margin
        y_max += box_height * self.crop_margin

        # Use a square crop to avoid stretching a person differently in x/y.
        centre_x, centre_y = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
        side = max(x_max - x_min, y_max - y_min)
        x_min, x_max = centre_x - side / 2.0, centre_x + side / 2.0
        y_min, y_max = centre_y - side / 2.0, centre_y + side / 2.0
        x_min, y_min = max(0.0, x_min), max(0.0, y_min)
        x_max, y_max = min(float(width), x_max), min(float(height), y_max)
        if x_max - x_min < 2.0 or y_max - y_min < 2.0:
            return None
        return x_min, y_min, x_max, y_max

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        annotation = _load_annotation(sample.annotation_path)
        pose = _to_numpy(
            annotation["poses"][sample.person_index][sample.frame_index]
        ).reshape(-1)
        translation = _to_numpy(
            annotation["trans"][sample.person_index][sample.frame_index]
        ).reshape(-1)
        shape = _to_numpy(annotation["betas"][sample.person_index]).reshape(-1)

        with Image.open(sample.image_path) as image_file:
            image = image_file.convert("RGB")
            full_width, full_height = image.size
            bbox = None
            if self.crop_to_person:
                bbox = self._person_bbox(
                    annotation,
                    sample.person_index,
                    sample.frame_index,
                    full_width,
                    full_height,
                )
            if bbox is not None:
                image = image.crop(bbox)
            else:
                bbox = (0.0, 0.0, float(full_width), float(full_height))
            try:
                resampling = Image.Resampling.BILINEAR
            except AttributeError:  # Pillow < 9.1
                resampling = Image.BILINEAR
            image = image.resize((self.image_width, self.image_height), resampling)
            image_array = np.asarray(image, dtype=np.float32) / 255.0

        image_tensor = torch.from_numpy(np.ascontiguousarray(image_array)).permute(2, 0, 1)
        camera_pose = _to_numpy(annotation["cam_poses"][sample.frame_index])
        camera_intrinsics = _to_numpy(annotation["cam_intrinsics"])
        return {
            "image": image_tensor.contiguous(),
            "global_orient": torch.from_numpy(pose[:3].copy()),
            "body_pose": torch.from_numpy(pose[3:66].copy()),
            "betas": torch.from_numpy(shape[:10].copy()),
            "transl": torch.from_numpy(translation[:3].copy()),
            "person_id": torch.tensor(sample.person_index, dtype=torch.long),
            "frame_id": torch.tensor(sample.frame_index, dtype=torch.long),
            "source_frame_id": torch.tensor(sample.source_frame_id, dtype=torch.long),
            "sequence": sample.sequence,
            "image_path": sample.image_path,
            "bbox_xyxy": torch.tensor(bbox, dtype=torch.float32),
            "crop_applied": torch.tensor(
                bbox != (0.0, 0.0, float(full_width), float(full_height)),
                dtype=torch.bool,
            ),
            "cam_pose": torch.from_numpy(camera_pose.copy()),
            "cam_intrinsics": torch.from_numpy(camera_intrinsics.copy()),
        }


# Shorter alias for training scripts.
ThreeDPWPersonFrameDataset = Real3DPWDataset


def _main() -> None:
    parser = argparse.ArgumentParser(description="Self-check the real 3DPW adapter")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--no-person-crop", action="store_true")
    args = parser.parse_args()
    dataset = Real3DPWDataset(
        root=args.root,
        split=args.split,
        image_size=args.image_size,
        frame_stride=args.frame_stride,
        max_samples=args.max_samples,
        crop_to_person=not args.no_person_crop,
    )
    report: Dict[str, Any] = {
        "split": dataset.split,
        "samples": len(dataset),
        "sequences": dataset.num_sequences,
        "people": dataset.num_people,
    }
    if len(dataset):
        item = dataset[0]
        tensor_names = (
            "image",
            "global_orient",
            "body_pose",
            "betas",
            "transl",
            "bbox_xyxy",
            "cam_pose",
            "cam_intrinsics",
        )
        report["first"] = {
            name: {"shape": list(item[name].shape), "dtype": str(item[name].dtype)}
            for name in tensor_names
        }
        report["first"].update(
            {
                "sequence": item["sequence"],
                "person_id": int(item["person_id"]),
                "frame_id": int(item["frame_id"]),
                "image_min": float(item["image"].min()),
                "image_max": float(item["image"].max()),
                "crop_applied": bool(item["crop_applied"]),
                "image_path": item["image_path"],
            }
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _main()
