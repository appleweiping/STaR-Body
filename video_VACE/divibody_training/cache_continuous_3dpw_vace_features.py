"""Cache genuine continuous-video VACE features for strict 3DPW person tubes.

This additive v4 builder leaves every earlier cache, dataset, encoder, trainer,
and evaluator untouched.  A cache row is one continuous nine-frame crop tube
from one ``(sequence, person)`` track.  Wan compresses those RGB frames to
three temporal latent slices; the slices are retained as a ``[3, 6, 6, 1536]``
grid instead of being averaged or presented as nine independent images.

Overlapping input windows are intentional.  Separate owner masks assign each
real frame, consecutive velocity pair, and consecutive acceleration triplet to
exactly one row, so a future trainer can use every primitive once without
double counting.  Person crops are computed independently before VACE.  Thus
two people in the same source RGB frame have different encoder inputs and this
person-tube protocol cannot share their VACE computation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


FORMAT = "divibody-contiguous-vace-video-features-v4"
RGB_FRAMES = 9
LATENT_FRAMES = 3
IMAGE_SIZE = 96
TOKEN_GRID = (6, 6)
FEATURE_WIDTH = 1536
LATENT_SOURCE_RANGES = ((0, 0), (1, 4), (5, 8))
EVALUATION_STRIDE = RGB_FRAMES - 2
TARGET_NAMES = ("full_pose", "betas", "gender_code", "cam_pose")


@dataclass(frozen=True)
class FrameIdentity:
    """One strict RGB-backed 3DPW person-frame in the source dataset."""

    dataset_index: int
    sequence: str
    person_id: int
    frame_id: int
    source_frame_id: int
    image_path: str

    @property
    def key(self) -> tuple[str, int, int]:
        return self.sequence, self.person_id, self.frame_id

    @property
    def uid(self) -> str:
        return f"{self.sequence}/person_{self.person_id}/frame_{self.frame_id:05d}"


@dataclass(frozen=True)
class WindowSpec:
    """One input window and the supervision primitives it uniquely owns."""

    sequence: str
    person_id: int
    segment_id: int
    segment_start: int
    frames: tuple[FrameIdentity, ...]
    frame_owner: tuple[bool, ...]
    velocity_owner: tuple[bool, ...]
    acceleration_owner: tuple[bool, ...]

    @property
    def uid(self) -> str:
        identity = {
            "sequence": self.sequence,
            "person_id": self.person_id,
            "frame_ids": [frame.frame_id for frame in self.frames],
        }
        encoded = json.dumps(
            identity, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class WindowPlan:
    windows: tuple[WindowSpec, ...]
    stats: dict[str, Any]
    expected_frame_keys: frozenset[tuple[str, int, int]]
    expected_velocity_keys: frozenset[tuple[str, int, int, int]]
    expected_acceleration_keys: frozenset[tuple[str, int, int, int, int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), required=True
    )
    parser.add_argument("--vace-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip-length", type=int, default=RGB_FRAMES)
    parser.add_argument("--train-stride", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--feature-layers", type=int, nargs="+", default=(7, 14, 22))
    parser.add_argument("--vace-timestep", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument(
        "--feature-only",
        action="store_true",
        help="Omit all SMPL/camera targets for an inference-only cache",
    )
    return parser.parse_args()


def validate_first_version_configuration(
    clip_length: int, image_size: int, train_stride: int
) -> None:
    """Keep the first experiment deliberately narrow and auditable."""

    if clip_length != RGB_FRAMES:
        raise ValueError(
            "The first continuous-cache format requires exactly 9 RGB frames "
            "(Wan 9->3); lengths such as 8 silently lose tail frames upstream"
        )
    if image_size != IMAGE_SIZE:
        raise ValueError("The first v4 cache requires 96x96 person crops")
    if not 1 <= train_stride <= RGB_FRAMES - 2:
        raise ValueError(
            "train-stride must be in [1,7] so every consecutive triplet is "
            "contained in at least one nine-frame input window"
        )


def frame_identities_from_samples(samples: Sequence[Any]) -> list[FrameIdentity]:
    frames = [
        FrameIdentity(
            dataset_index=index,
            sequence=str(sample.sequence),
            person_id=int(sample.person_index),
            frame_id=int(sample.frame_index),
            source_frame_id=int(sample.source_frame_id),
            image_path=str(sample.image_path),
        )
        for index, sample in enumerate(samples)
    ]
    keys = [frame.key for frame in frames]
    if len(keys) != len(set(keys)):
        raise ValueError("Strict 3DPW index contains duplicate person-frame identities")
    return frames


def _continuous_segments(
    frames: Iterable[FrameIdentity],
) -> list[tuple[tuple[str, int, int], tuple[FrameIdentity, ...]]]:
    tracks: dict[tuple[str, int], list[FrameIdentity]] = defaultdict(list)
    for frame in frames:
        tracks[(frame.sequence, frame.person_id)].append(frame)

    segments: list[tuple[tuple[str, int, int], tuple[FrameIdentity, ...]]] = []
    for (sequence, person_id), track_frames in sorted(tracks.items()):
        ordered = sorted(track_frames, key=lambda item: item.frame_id)
        if len({frame.frame_id for frame in ordered}) != len(ordered):
            raise ValueError(f"Duplicate frame in track {(sequence, person_id)}")
        start = 0
        segment_id = 0
        for position in range(1, len(ordered) + 1):
            at_end = position == len(ordered)
            has_gap = not at_end and (
                ordered[position].frame_id != ordered[position - 1].frame_id + 1
            )
            if at_end or has_gap:
                segment = tuple(ordered[start:position])
                if segment:
                    segments.append(((sequence, person_id, segment_id), segment))
                    segment_id += 1
                start = position
    return segments


def _window_starts(segment_length: int, stride: int) -> list[int]:
    if segment_length < RGB_FRAMES:
        return []
    last = segment_length - RGB_FRAMES
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def _choose_owner(
    primitive_start: int,
    primitive_width: int,
    starts: Sequence[int],
) -> int:
    candidates = [
        index
        for index, start in enumerate(starts)
        if start <= primitive_start
        and primitive_start + primitive_width <= start + RGB_FRAMES
    ]
    if not candidates:
        raise RuntimeError(
            f"No input window contains primitive start={primitive_start}, "
            f"width={primitive_width}"
        )
    primitive_center_twice = 2 * primitive_start + primitive_width - 1
    return min(
        candidates,
        key=lambda index: (
            abs(primitive_center_twice - (2 * starts[index] + RGB_FRAMES - 1)),
            starts[index],
        ),
    )


def build_window_plan(
    frames: Iterable[FrameIdentity],
    split: str,
    train_stride: int = 4,
) -> WindowPlan:
    """Plan strict windows and unique frame/pair/triplet ownership."""

    if split not in ("train", "validation", "test"):
        raise ValueError(f"Unsupported split: {split!r}")
    validate_first_version_configuration(RGB_FRAMES, IMAGE_SIZE, train_stride)
    frames = list(frames)
    keys = [frame.key for frame in frames]
    if len(keys) != len(set(keys)):
        raise ValueError("Input person-frame identities are not unique")
    stride = train_stride if split == "train" else EVALUATION_STRIDE

    planned: list[WindowSpec] = []
    expected_frames: set[tuple[str, int, int]] = set()
    expected_velocity: set[tuple[str, int, int, int]] = set()
    expected_acceleration: set[tuple[str, int, int, int, int]] = set()
    short_examples: list[dict[str, Any]] = []
    total_segments = eligible_segments = short_segments = 0
    skipped_short_frames = skipped_short_pairs = skipped_short_triplets = 0
    input_overlap_windows = 0

    for (sequence, person_id, segment_id), segment in _continuous_segments(frames):
        total_segments += 1
        if len(segment) < RGB_FRAMES:
            short_segments += 1
            skipped_short_frames += len(segment)
            skipped_short_pairs += max(len(segment) - 1, 0)
            skipped_short_triplets += max(len(segment) - 2, 0)
            if len(short_examples) < 20:
                short_examples.append(
                    {
                        "sequence": sequence,
                        "person_id": person_id,
                        "segment_id": segment_id,
                        "first_frame": segment[0].frame_id,
                        "last_frame": segment[-1].frame_id,
                        "frames": len(segment),
                    }
                )
            continue

        eligible_segments += 1
        starts = _window_starts(len(segment), stride)
        if any(
            starts[index] < starts[index - 1] + RGB_FRAMES
            for index in range(1, len(starts))
        ):
            input_overlap_windows += sum(
                starts[index] < starts[index - 1] + RGB_FRAMES
                for index in range(1, len(starts))
            )

        frame_masks = [[False] * RGB_FRAMES for _ in starts]
        velocity_masks = [[False] * (RGB_FRAMES - 1) for _ in starts]
        acceleration_masks = [[False] * (RGB_FRAMES - 2) for _ in starts]

        for position, frame in enumerate(segment):
            owner = _choose_owner(position, 1, starts)
            frame_masks[owner][position - starts[owner]] = True
            expected_frames.add(frame.key)
        for position in range(len(segment) - 1):
            owner = _choose_owner(position, 2, starts)
            velocity_masks[owner][position - starts[owner]] = True
            expected_velocity.add(
                (sequence, person_id, segment[position].frame_id,
                 segment[position + 1].frame_id)
            )
        for position in range(len(segment) - 2):
            owner = _choose_owner(position, 3, starts)
            acceleration_masks[owner][position - starts[owner]] = True
            expected_acceleration.add(
                (
                    sequence,
                    person_id,
                    segment[position].frame_id,
                    segment[position + 1].frame_id,
                    segment[position + 2].frame_id,
                )
            )

        for window_index, start in enumerate(starts):
            window_frames = tuple(segment[start : start + RGB_FRAMES])
            if len(window_frames) != RGB_FRAMES or any(
                window_frames[index].frame_id
                != window_frames[index - 1].frame_id + 1
                for index in range(1, RGB_FRAMES)
            ):
                raise RuntimeError("A planned VACE window crosses an RGB frame gap")
            planned.append(
                WindowSpec(
                    sequence=sequence,
                    person_id=person_id,
                    segment_id=segment_id,
                    segment_start=start,
                    frames=window_frames,
                    frame_owner=tuple(frame_masks[window_index]),
                    velocity_owner=tuple(velocity_masks[window_index]),
                    acceleration_owner=tuple(acceleration_masks[window_index]),
                )
            )

    total_frames = len(frames)
    covered_frames = len(expected_frames)
    plan = WindowPlan(
        windows=tuple(planned),
        stats={
            "split": split,
            "input_stride": stride,
            "input_window_overlap_frames_nominal": RGB_FRAMES - stride,
            "input_windows": len(planned),
            "overlapping_adjacent_window_pairs": input_overlap_windows,
            "tracks": len({(frame.sequence, frame.person_id) for frame in frames}),
            "segments": total_segments,
            "eligible_segments": eligible_segments,
            "short_segments_skipped": short_segments,
            "short_segment_examples": short_examples,
            "total_valid_person_frames": total_frames,
            "owned_person_frames": covered_frames,
            "uncovered_short_segment_person_frames": skipped_short_frames,
            "owned_velocity_pairs": len(expected_velocity),
            "uncovered_short_segment_velocity_pairs": skipped_short_pairs,
            "owned_acceleration_triplets": len(expected_acceleration),
            "uncovered_short_segment_acceleration_triplets": skipped_short_triplets,
            "complete_person_frame_coverage": covered_frames == total_frames,
            "development_only": covered_frames != total_frames,
            "ownership": (
                "input windows overlap; every eligible real frame, consecutive "
                "pair, and consecutive triplet has exactly one owner"
            ),
        },
        expected_frame_keys=frozenset(expected_frames),
        expected_velocity_keys=frozenset(expected_velocity),
        expected_acceleration_keys=frozenset(expected_acceleration),
    )
    validate_window_plan(plan)
    return plan


def _owned_keys(
    plan: WindowPlan,
) -> tuple[
    list[tuple[str, int, int]],
    list[tuple[str, int, int, int]],
    list[tuple[str, int, int, int, int]],
]:
    frame_keys: list[tuple[str, int, int]] = []
    velocity_keys: list[tuple[str, int, int, int]] = []
    acceleration_keys: list[tuple[str, int, int, int, int]] = []
    for window in plan.windows:
        frame_ids = [frame.frame_id for frame in window.frames]
        for local, owned in enumerate(window.frame_owner):
            if owned:
                frame_keys.append((window.sequence, window.person_id, frame_ids[local]))
        for local, owned in enumerate(window.velocity_owner):
            if owned:
                velocity_keys.append(
                    (window.sequence, window.person_id,
                     frame_ids[local], frame_ids[local + 1])
                )
        for local, owned in enumerate(window.acceleration_owner):
            if owned:
                acceleration_keys.append(
                    (window.sequence, window.person_id,
                     frame_ids[local], frame_ids[local + 1], frame_ids[local + 2])
                )
    return frame_keys, velocity_keys, acceleration_keys


def validate_window_plan(plan: WindowPlan) -> None:
    for window in plan.windows:
        if len(window.frames) != RGB_FRAMES:
            raise ValueError("Every v4 input window must contain exactly nine frames")
        if not (
            len(window.frame_owner) == RGB_FRAMES
            and len(window.velocity_owner) == RGB_FRAMES - 1
            and len(window.acceleration_owner) == RGB_FRAMES - 2
        ):
            raise ValueError("Owner-mask lengths do not match the nine-frame protocol")
        if any(
            frame.sequence != window.sequence or frame.person_id != window.person_id
            for frame in window.frames
        ):
            raise ValueError("A person-tube window mixes tracks")
        if any(
            window.frames[index].frame_id
            != window.frames[index - 1].frame_id + 1
            for index in range(1, RGB_FRAMES)
        ):
            raise ValueError("A person-tube window crosses a frame gap")

    frames, velocities, accelerations = _owned_keys(plan)
    checks = (
        ("frame", frames, plan.expected_frame_keys),
        ("velocity", velocities, plan.expected_velocity_keys),
        ("acceleration", accelerations, plan.expected_acceleration_keys),
    )
    for label, actual, expected in checks:
        if len(actual) != len(set(actual)):
            raise RuntimeError(f"{label} owner masks assign a primitive more than once")
        if set(actual) != set(expected):
            missing = sorted(set(expected) - set(actual))[:5]
            extra = sorted(set(actual) - set(expected))[:5]
            raise RuntimeError(
                f"{label} ownership mismatch: missing={missing}, extra={extra}"
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_tree_fingerprint(root: Path) -> dict[str, Any]:
    """Content-hash every checkpoint asset and then hash that manifest."""

    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"VACE checkpoint directory not found: {root}")
    files = []
    for path in sorted((path for path in root.rglob("*") if path.is_file())):
        files.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise FileNotFoundError(f"VACE checkpoint directory is empty: {root}")
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "directory": str(root),
        "files": files,
        "tree_sha256": hashlib.sha256(canonical).hexdigest(),
    }


class PersonTubeWindowDataset(Dataset):
    """Load independent oracle person crops for each frame of a planned tube."""

    def __init__(self, base: Any, plan: WindowPlan, feature_only: bool) -> None:
        self.base = base
        self.plan = plan
        self.feature_only = bool(feature_only)

    def __len__(self) -> int:
        return len(self.plan.windows)

    def _feature_only_crop(self, index: int) -> dict[str, torch.Tensor]:
        # Import lazily so the pure ownership planner remains CPU-testable even
        # in environments that do not have the full VACE dependency tree.
        from real_3dpw_dataset import _load_annotation

        sample = self.base.samples[index]
        annotation = _load_annotation(sample.annotation_path)
        with Image.open(sample.image_path) as image_file:
            image = image_file.convert("RGB")
            width, height = image.size
            bbox = self.base._person_bbox(
                annotation, sample.person_index, sample.frame_index, width, height
            )
            if bbox is not None:
                image = image.crop(bbox)
            else:
                bbox = (0.0, 0.0, float(width), float(height))
            try:
                resampling = Image.Resampling.BILINEAR
            except AttributeError:  # Pillow < 9.1
                resampling = Image.BILINEAR
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE), resampling)
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
        return {
            "image": tensor.contiguous(),
            "bbox_xyxy": torch.tensor(bbox, dtype=torch.float32),
            "crop_applied": torch.tensor(
                bbox != (0.0, 0.0, float(width), float(height)), dtype=torch.bool
            ),
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        window = self.plan.windows[index]
        items = [
            self._feature_only_crop(frame.dataset_index)
            if self.feature_only
            else self.base[frame.dataset_index]
            for frame in window.frames
        ]
        result: dict[str, torch.Tensor] = {
            "window_index": torch.tensor(index, dtype=torch.long),
            "images": torch.stack([item["image"] for item in items]),
            "bbox_xyxy": torch.stack([item["bbox_xyxy"] for item in items]),
            "crop_applied": torch.stack([item["crop_applied"] for item in items]),
        }
        if not self.feature_only:
            for name in TARGET_NAMES:
                result[name] = torch.stack([item[name] for item in items])
        return result


def identity_metadata(plan: WindowPlan) -> list[dict[str, Any]]:
    records = []
    for window in plan.windows:
        records.append(
            {
                "window_uid": window.uid,
                "sequence": window.sequence,
                "person_id": window.person_id,
                "segment_id": window.segment_id,
                "segment_start": window.segment_start,
                "frame_ids": [frame.frame_id for frame in window.frames],
                "source_frame_ids": [frame.source_frame_id for frame in window.frames],
                "frame_uids": [frame.uid for frame in window.frames],
                "image_paths": [frame.image_path for frame in window.frames],
            }
        )
    return records


def validate_output_payload(payload: dict[str, Any], feature_only: bool) -> None:
    signature = payload.get("signature", {})
    if signature.get("format") != FORMAT:
        raise ValueError("Output payload does not have the v4 format")
    features = payload.get("features")
    metadata = payload.get("metadata", [])
    expected_features = (len(metadata), LATENT_FRAMES, *TOKEN_GRID, FEATURE_WIDTH)
    if not isinstance(features, torch.Tensor) or tuple(features.shape) != expected_features:
        raise ValueError(f"Expected feature shape {expected_features}")
    if features.dtype != torch.bfloat16 or not bool(torch.isfinite(features.float()).all()):
        raise ValueError("v4 features must be finite BF16")
    mask_shapes = {
        "valid_mask": (len(metadata), RGB_FRAMES),
        "frame_owner_mask": (len(metadata), RGB_FRAMES),
        "velocity_owner_mask": (len(metadata), RGB_FRAMES - 1),
        "acceleration_owner_mask": (len(metadata), RGB_FRAMES - 2),
    }
    for name, shape in mask_shapes.items():
        value = payload.get(name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if value.dtype != torch.bool:
            raise ValueError(f"{name} must be boolean")
    if not bool(payload["valid_mask"].all()):
        raise ValueError("First-version windows may not contain padding")
    if feature_only:
        if "targets" in payload:
            raise ValueError("A feature-only cache must not contain targets")
    else:
        targets = payload.get("targets", {})
        expected = {
            "full_pose": (len(metadata), RGB_FRAMES, 72),
            "betas": (len(metadata), RGB_FRAMES, 10),
            "gender_code": (len(metadata), RGB_FRAMES),
            "cam_pose": (len(metadata), RGB_FRAMES, 4, 4),
        }
        for name, shape in expected.items():
            value = targets.get(name)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"Target {name} must have shape {shape}")
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"Target {name} contains non-finite values")
    identities = [
        (record["sequence"], int(record["person_id"]), tuple(record["frame_ids"]))
        for record in metadata
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("Output metadata contains duplicate person-tube windows")
    if any(
        any(frames[index] != frames[index - 1] + 1 for index in range(1, len(frames)))
        for _, _, frames in identities
    ):
        raise ValueError("Output metadata contains a cross-gap window")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    validate_first_version_configuration(
        args.clip_length, args.image_size, args.train_stride
    )
    if args.output.suffix.lower() != ".pt":
        raise ValueError("output must be a .pt file")
    summary_path = args.output.with_suffix(".json")
    if args.output.exists() or summary_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite cache or summary: {args.output}, {summary_path}"
        )
    if args.batch_size <= 0 or args.num_workers < 0 or args.log_interval <= 0:
        raise ValueError("Invalid batch/worker/log interval")
    if args.split == "test" and not args.feature_only:
        raise ValueError("A test cache must use --feature-only and contain no GT")

    # Heavy project modules are imported only for a real cache run.  Planner
    # and ownership tests remain runnable on CPU without loading VACE.
    from evaluate_standard_3dpw import (
        Standard3DPWDataset,
        audit_split,
        enforce_complete_test,
    )
    from real_vace_temporal_feature_encoder import (
        FrozenTemporalVaceFeatureEncoder,
        wan_latent_frames,
    )

    started = time.monotonic()
    audit = audit_split(args.data_root, args.split)
    if args.split == "test":
        enforce_complete_test(audit, allow_incomplete=False)
    dataset = Standard3DPWDataset(
        root=args.data_root,
        split=args.split,
        image_size=args.image_size,
        frame_stride=1,
        max_samples=None,
        crop_to_person=True,
    )
    if len(dataset) != int(audit["available_valid_person_frames"]):
        raise RuntimeError("Strict dataset length differs from the official split audit")
    plan = build_window_plan(
        frame_identities_from_samples(dataset.samples),
        split=args.split,
        train_stride=args.train_stride,
    )
    if not plan.windows:
        raise RuntimeError("No complete nine-frame person-tube windows were found")
    if args.split == "test" and not plan.stats["complete_person_frame_coverage"]:
        raise RuntimeError(
            "Formal test caching refuses incomplete coverage caused by short segments"
        )

    checkpoint = checkpoint_tree_fingerprint(args.vace_checkpoint_dir)
    encoder_path = Path(__file__).with_name("real_vace_temporal_feature_encoder.py")
    if not encoder_path.is_file():
        raise FileNotFoundError(f"Temporal encoder source is missing: {encoder_path}")
    signature = {
        "format": FORMAT,
        "split": args.split,
        "feature_only": bool(args.feature_only),
        "contains_targets": not args.feature_only,
        "development_only": bool(plan.stats["development_only"]),
        "dataset_root": str(args.data_root.expanduser().resolve()),
        "dataset_preflight": audit,
        "window_plan": plan.stats,
        "strict_rgb_resolver": "image_{annotation_frame_index:05d}",
        "clip_length_rgb": RGB_FRAMES,
        "latent_frames": wan_latent_frames(RGB_FRAMES),
        "latent_source_ranges_inclusive": [list(value) for value in LATENT_SOURCE_RANGES],
        "temporal_compression": "causal Wan VAE: first RGB alone, then groups of four",
        "input_stride": plan.stats["input_stride"],
        "input_overlap_frames_nominal": plan.stats[
            "input_window_overlap_frames_nominal"
        ],
        "input_overlap_note": (
            "input windows overlap; owner masks are disjoint and uniquely assign "
            "frames, pairs, and triplets"
        ),
        "short_segments": "segments shorter than nine are skipped, never padded",
        "valid_mask": "all nine positions are real consecutive RGB frames",
        "frame_owner_mask": "unique static-loss owner for each eligible person-frame",
        "velocity_owner_mask": "unique owner for each eligible consecutive frame pair",
        "acceleration_owner_mask": (
            "unique owner for each eligible consecutive frame triplet"
        ),
        "view_protocol": "oracle_person_tube_crop_before_vace_v1",
        "crop": (
            "independent per-frame oracle 3DPW poses2d square crop with 25 percent "
            "margin; deterministic whole-image fallback when invalid"
        ),
        "multi_person_rgb_encode_reuse": False,
        "multi_person_reuse_reason": (
            "person-specific pre-VACE crops are different pixel inputs and cannot "
            "share a correct VACE encoding"
        ),
        "image_size": args.image_size,
        "feature_backend": "official frozen Wan2.1-VACE continuous video",
        "feature_layers": list(args.feature_layers),
        "feature_layer_reduction": "arithmetic mean",
        "vace_timestep": args.vace_timestep,
        "diffusion_input_protocol": (
            "clean RGB VAE latent used as diffusion x and visible VACE condition; "
            "zero text embedding; no scheduler noise added"
        ),
        "feature_shape_per_window": [LATENT_FRAMES, *TOKEN_GRID, FEATURE_WIDTH],
        "feature_dtype": "bfloat16",
        "token_order": "time-major, row-major, column-major",
        "checkpoint": checkpoint,
        "temporal_encoder_source": str(encoder_path.resolve()),
        "temporal_encoder_source_sha256": sha256_file(encoder_path),
        "cache_builder_source_sha256": sha256_file(Path(__file__)),
    }

    window_dataset = PersonTubeWindowDataset(dataset, plan, args.feature_only)
    loader = DataLoader(
        window_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    device = torch.device(args.device)
    encoder = FrozenTemporalVaceFeatureEncoder(
        checkpoint_dir=args.vace_checkpoint_dir,
        device=device,
        image_size=args.image_size,
        feature_layers=args.feature_layers,
        timestep=args.vace_timestep,
    )
    metadata = identity_metadata(plan)
    feature_chunks: list[torch.Tensor] = []
    target_chunks: dict[str, list[torch.Tensor]] = {
        name: [] for name in TARGET_NAMES
    }
    expected_window_index = 0
    crop_applied_rows = 0
    crop_fallback_rows = 0
    try:
        for step, batch in enumerate(loader, start=1):
            indices = batch["window_index"].tolist()
            expected = list(range(expected_window_index, expected_window_index + len(indices)))
            if indices != expected:
                raise RuntimeError("Cache DataLoader changed deterministic window order")
            expected_window_index += len(indices)
            clips = batch["images"].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            features = encoder.encode(clips)
            shape = (len(indices), LATENT_FRAMES, *TOKEN_GRID, FEATURE_WIDTH)
            if tuple(features.shape) != shape or not bool(torch.isfinite(features).all()):
                raise RuntimeError(
                    f"Unexpected/non-finite temporal VACE features: {tuple(features.shape)}"
                )
            feature_chunks.append(features.to(device="cpu", dtype=torch.bfloat16))
            boxes = batch["bbox_xyxy"].cpu()
            crop_flags = batch["crop_applied"].cpu()
            for row, window_index in enumerate(indices):
                metadata[window_index]["bbox_xyxy"] = boxes[row].tolist()
                metadata[window_index]["crop_applied"] = crop_flags[row].tolist()
            crop_applied_rows += int(crop_flags.sum())
            crop_fallback_rows += int((~crop_flags).sum())
            if not args.feature_only:
                for name in TARGET_NAMES:
                    value = batch[name].detach().cpu()
                    target_chunks[name].append(
                        value.float() if value.is_floating_point() else value
                    )
            if step % args.log_interval == 0 or step == len(loader):
                print(
                    json.dumps(
                        {
                            "phase": "cache_continuous_3dpw_vace_features",
                            "split": args.split,
                            "step": step,
                            "steps": len(loader),
                            "windows": expected_window_index,
                            "seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
    finally:
        encoder.close()

    signature["crop_applied_person_frame_occurrences"] = crop_applied_rows
    signature["whole_image_fallback_person_frame_occurrences"] = crop_fallback_rows
    owner_tensors = {
        "valid_mask": torch.ones((len(plan.windows), RGB_FRAMES), dtype=torch.bool),
        "frame_owner_mask": torch.tensor(
            [window.frame_owner for window in plan.windows], dtype=torch.bool
        ),
        "velocity_owner_mask": torch.tensor(
            [window.velocity_owner for window in plan.windows], dtype=torch.bool
        ),
        "acceleration_owner_mask": torch.tensor(
            [window.acceleration_owner for window in plan.windows], dtype=torch.bool
        ),
    }
    payload: dict[str, Any] = {
        "signature": signature,
        "features": torch.cat(feature_chunks),
        **owner_tensors,
        "metadata": metadata,
    }
    if not args.feature_only:
        payload["targets"] = {
            name: torch.cat(chunks) for name, chunks in target_chunks.items()
        }
    validate_output_payload(payload, args.feature_only)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, args.output)
    result = {
        "status": "DIVIBODY_CONTINUOUS_3DPW_VACE_CACHE_OK",
        "format": FORMAT,
        "split": args.split,
        "feature_only": bool(args.feature_only),
        "development_only": bool(plan.stats["development_only"]),
        "windows": len(plan.windows),
        "feature_shape": list(payload["features"].shape),
        "owned_person_frames": int(payload["frame_owner_mask"].sum()),
        "owned_velocity_pairs": int(payload["velocity_owner_mask"].sum()),
        "owned_acceleration_triplets": int(
            payload["acceleration_owner_mask"].sum()
        ),
        "output": str(args.output.expanduser().resolve()),
        "output_sha256": sha256_file(args.output),
        "seconds": time.monotonic() - started,
    }
    summary_path.write_text(
        json.dumps(_json_ready(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
