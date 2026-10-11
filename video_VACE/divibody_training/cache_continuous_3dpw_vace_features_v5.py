"""Build the v5 continuous 3DPW person-tube VACE cache.

Each row encodes a 17-RGB-frame context into five Wan/VACE latent slices and
owns predictions for eight real consecutive RGB frames at context positions
4..11.  Target blocks advance by eight.  A final block may overlap to cover a
track tail, but owner masks ensure every eligible person-frame is supervised
exactly once.  Context outside a continuous track boundary is edge-repeated;
it never crosses a frame gap or enters another track.

Velocity and acceleration losses cover only primitives contained in one
eight-target block.  Primitives crossing adjacent target-block boundaries are
recorded separately for full-track metrics after predictions are stitched.
Earlier cache builders and encoders are not modified.
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

from cache_continuous_3dpw_vace_features import (
    FrameIdentity,
    checkpoint_tree_fingerprint,
    frame_identities_from_samples,
    sha256_file,
)


FORMAT = "divibody-contiguous-vace-video-features-v5"
STRIDE4_FORMAT = "divibody-contiguous-vace-video-features-v6-stride4"
CONTEXT_FRAMES = 17
LATENT_FRAMES = 5
TARGET_FRAMES = 8
TARGET_CONTEXT_START = 4
TARGET_STRIDE = 8
IMAGE_SIZE = 96
TOKEN_GRID = (6, 6)
FEATURE_WIDTH = 1536
LATENT_SOURCE_RANGES = ((0, 0), (1, 4), (5, 8), (9, 12), (13, 16))
TARGET_NAMES = ("full_pose", "betas", "gender_code", "cam_pose")


@dataclass(frozen=True)
class V5WindowSpec:
    sequence: str
    person_id: int
    segment_id: int
    target_start: int
    context_frames: tuple[FrameIdentity, ...]
    context_edge_repeat: tuple[bool, ...]
    target_frames: tuple[FrameIdentity, ...]
    target_valid: tuple[bool, ...]
    frame_owner: tuple[bool, ...]
    velocity_owner: tuple[bool, ...]
    acceleration_owner: tuple[bool, ...]

    @property
    def uid(self) -> str:
        identity = {
            "sequence": self.sequence,
            "person_id": self.person_id,
            "context_frame_ids": [frame.frame_id for frame in self.context_frames],
            "context_edge_repeat": list(self.context_edge_repeat),
            "target_frame_ids": [frame.frame_id for frame in self.target_frames],
        }
        encoded = json.dumps(
            identity, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class V5WindowPlan:
    windows: tuple[V5WindowSpec, ...]
    stats: dict[str, Any]
    expected_frame_keys: frozenset[tuple[str, int, int]]
    expected_training_velocity_keys: frozenset[tuple[str, int, int, int]]
    expected_training_acceleration_keys: frozenset[
        tuple[str, int, int, int, int]
    ]
    metric_only_velocity_keys: frozenset[tuple[str, int, int, int]]
    metric_only_acceleration_keys: frozenset[tuple[str, int, int, int, int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), required=True
    )
    parser.add_argument("--vace-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=CONTEXT_FRAMES)
    parser.add_argument("--target-length", type=int, default=TARGET_FRAMES)
    parser.add_argument("--target-stride", type=int, default=TARGET_STRIDE)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--feature-layers", type=int, nargs="+", default=(7, 14, 22))
    parser.add_argument("--vace-timestep", type=int, default=0)
    parser.add_argument(
        "--diffusion-input-protocol",
        choices=("clean", "flow_noised"),
        default="clean",
    )
    parser.add_argument("--text-embedding-path", type=Path)
    parser.add_argument("--noise-seed-base", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--feature-only", action="store_true")
    parser.add_argument(
        "--development-max-windows",
        type=int,
        default=0,
        help=(
            "Development only: keep at most this many non-overlapping target "
            "windows. Zero builds the full split."
        ),
    )
    return parser.parse_args()


def validate_v5_configuration(
    context_length: int,
    target_length: int,
    target_stride: int,
    image_size: int,
) -> None:
    if context_length != CONTEXT_FRAMES:
        raise ValueError(
            "v5 requires exactly 17 context frames (Wan 17->5); lengths such "
            "as 16 silently lose tail frames upstream"
        )
    if target_length != TARGET_FRAMES or target_stride != TARGET_STRIDE:
        raise ValueError("v5 requires eight target frames with stride eight")
    if image_size != IMAGE_SIZE:
        raise ValueError("v5 requires 96x96 independent person crops")


def _segments(
    frames: Iterable[FrameIdentity],
) -> list[tuple[tuple[str, int, int], tuple[FrameIdentity, ...]]]:
    tracks: dict[tuple[str, int], list[FrameIdentity]] = defaultdict(list)
    for frame in frames:
        tracks[(frame.sequence, frame.person_id)].append(frame)
    output = []
    for (sequence, person_id), values in sorted(tracks.items()):
        values = sorted(values, key=lambda frame: frame.frame_id)
        if len({frame.frame_id for frame in values}) != len(values):
            raise ValueError(f"Duplicate frame in track {(sequence, person_id)}")
        start = 0
        segment_id = 0
        for position in range(1, len(values) + 1):
            at_end = position == len(values)
            gap = not at_end and values[position].frame_id != values[position - 1].frame_id + 1
            if at_end or gap:
                segment = tuple(values[start:position])
                if segment:
                    output.append(((sequence, person_id, segment_id), segment))
                    segment_id += 1
                start = position
    return output


def _target_starts(length: int) -> list[int]:
    if length < TARGET_FRAMES:
        return []
    last = length - TARGET_FRAMES
    starts = list(range(0, last + 1, TARGET_STRIDE))
    if starts[-1] != last:
        starts.append(last)
    return starts


def _context_for_target(
    segment: Sequence[FrameIdentity], target_start: int
) -> tuple[tuple[FrameIdentity, ...], tuple[bool, ...]]:
    desired_start = target_start - TARGET_CONTEXT_START
    context = []
    repeated = []
    for offset in range(CONTEXT_FRAMES):
        desired = desired_start + offset
        clamped = min(max(desired, 0), len(segment) - 1)
        context.append(segment[clamped])
        repeated.append(clamped != desired)
    return tuple(context), tuple(repeated)


def _owner_for_primitive(
    primitive_start: int, primitive_width: int, starts: Sequence[int]
) -> int | None:
    candidates = [
        index
        for index, start in enumerate(starts)
        if start <= primitive_start
        and primitive_start + primitive_width <= start + TARGET_FRAMES
    ]
    if not candidates:
        return None
    primitive_center_twice = 2 * primitive_start + primitive_width - 1
    return min(
        candidates,
        key=lambda index: (
            abs(
                primitive_center_twice
                - (2 * starts[index] + TARGET_FRAMES - 1)
            ),
            starts[index],
        ),
    )


def build_v5_window_plan(frames: Iterable[FrameIdentity]) -> V5WindowPlan:
    frames = list(frames)
    identities = [frame.key for frame in frames]
    if len(identities) != len(set(identities)):
        raise ValueError("Input person-frame identities are not unique")

    windows: list[V5WindowSpec] = []
    expected_frames: set[tuple[str, int, int]] = set()
    training_velocity: set[tuple[str, int, int, int]] = set()
    training_acceleration: set[tuple[str, int, int, int, int]] = set()
    metric_velocity: set[tuple[str, int, int, int]] = set()
    metric_acceleration: set[tuple[str, int, int, int, int]] = set()
    short_examples = []
    short_segments = padded_target_positions = 0
    segment_count = eligible_segments = 0
    tail_overlap_windows = edge_repeat_positions = 0

    for (sequence, person_id, segment_id), segment in _segments(frames):
        segment_count += 1
        if len(segment) < TARGET_FRAMES:
            short_segments += 1
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
            valid_count = len(segment)
            padded_target_positions += TARGET_FRAMES - valid_count
            targets = tuple(segment) + (segment[-1],) * (
                TARGET_FRAMES - valid_count
            )
            target_valid = (True,) * valid_count + (False,) * (
                TARGET_FRAMES - valid_count
            )
            context, repeated = _context_for_target(segment, 0)
            edge_repeat_positions += sum(repeated)
            expected_frames.update(frame.key for frame in segment)
            for position in range(valid_count - 1):
                training_velocity.add(
                    (
                        sequence,
                        person_id,
                        segment[position].frame_id,
                        segment[position + 1].frame_id,
                    )
                )
            for position in range(valid_count - 2):
                training_acceleration.add(
                    (
                        sequence,
                        person_id,
                        segment[position].frame_id,
                        segment[position + 1].frame_id,
                        segment[position + 2].frame_id,
                    )
                )
            windows.append(
                V5WindowSpec(
                    sequence=sequence,
                    person_id=person_id,
                    segment_id=segment_id,
                    target_start=0,
                    context_frames=context,
                    context_edge_repeat=repeated,
                    target_frames=targets,
                    target_valid=target_valid,
                    frame_owner=target_valid,
                    velocity_owner=(True,) * max(valid_count - 1, 0)
                    + (False,) * (TARGET_FRAMES - valid_count),
                    acceleration_owner=(True,) * max(valid_count - 2, 0)
                    + (False,)
                    * (
                        TARGET_FRAMES
                        - 2
                        - max(valid_count - 2, 0)
                    ),
                )
            )
            continue
        eligible_segments += 1
        starts = _target_starts(len(segment))
        tail_overlap_windows += sum(
            starts[index] < starts[index - 1] + TARGET_FRAMES
            for index in range(1, len(starts))
        )
        frame_masks = [[False] * TARGET_FRAMES for _ in starts]
        velocity_masks = [[False] * (TARGET_FRAMES - 1) for _ in starts]
        acceleration_masks = [[False] * (TARGET_FRAMES - 2) for _ in starts]

        for position, frame in enumerate(segment):
            owner = _owner_for_primitive(position, 1, starts)
            if owner is None:
                raise RuntimeError("A real target frame has no v5 owner window")
            frame_masks[owner][position - starts[owner]] = True
            expected_frames.add(frame.key)
        for position in range(len(segment) - 1):
            key = (
                sequence,
                person_id,
                segment[position].frame_id,
                segment[position + 1].frame_id,
            )
            owner = _owner_for_primitive(position, 2, starts)
            if owner is None:
                metric_velocity.add(key)
            else:
                velocity_masks[owner][position - starts[owner]] = True
                training_velocity.add(key)
        for position in range(len(segment) - 2):
            key = (
                sequence,
                person_id,
                segment[position].frame_id,
                segment[position + 1].frame_id,
                segment[position + 2].frame_id,
            )
            owner = _owner_for_primitive(position, 3, starts)
            if owner is None:
                metric_acceleration.add(key)
            else:
                acceleration_masks[owner][position - starts[owner]] = True
                training_acceleration.add(key)

        for local_window, start in enumerate(starts):
            targets = tuple(segment[start : start + TARGET_FRAMES])
            if len(targets) != TARGET_FRAMES or any(
                targets[index].frame_id != targets[index - 1].frame_id + 1
                for index in range(1, TARGET_FRAMES)
            ):
                raise RuntimeError("A v5 target window crosses a track gap")
            context, repeated = _context_for_target(segment, start)
            edge_repeat_positions += sum(repeated)
            windows.append(
                V5WindowSpec(
                    sequence=sequence,
                    person_id=person_id,
                    segment_id=segment_id,
                    target_start=start,
                    context_frames=context,
                    context_edge_repeat=repeated,
                    target_frames=targets,
                    target_valid=(True,) * TARGET_FRAMES,
                    frame_owner=tuple(frame_masks[local_window]),
                    velocity_owner=tuple(velocity_masks[local_window]),
                    acceleration_owner=tuple(acceleration_masks[local_window]),
                )
            )

    covered = len(expected_frames)
    plan = V5WindowPlan(
        windows=tuple(windows),
        stats={
            "tracks": len({(frame.sequence, frame.person_id) for frame in frames}),
            "segments": segment_count,
            "eligible_segments": eligible_segments,
            "short_segments_edge_padded": short_segments,
            "short_segment_examples": short_examples,
            "padded_target_positions": padded_target_positions,
            "total_valid_person_frames": len(frames),
            "owned_person_frames": covered,
            "uncovered_short_segment_person_frames": 0,
            "complete_person_frame_coverage": covered == len(frames),
            "target_windows": len(windows),
            "target_stride": TARGET_STRIDE,
            "tail_overlap_window_pairs": tail_overlap_windows,
            "context_edge_repeat_positions": edge_repeat_positions,
            "training_velocity_pairs": len(training_velocity),
            "metric_only_cross_window_velocity_pairs": len(metric_velocity),
            "training_acceleration_triplets": len(training_acceleration),
            "metric_only_cross_window_acceleration_triplets": len(
                metric_acceleration
            ),
            "development_only": covered != len(frames),
        },
        expected_frame_keys=frozenset(expected_frames),
        expected_training_velocity_keys=frozenset(training_velocity),
        expected_training_acceleration_keys=frozenset(training_acceleration),
        metric_only_velocity_keys=frozenset(metric_velocity),
        metric_only_acceleration_keys=frozenset(metric_acceleration),
    )
    validate_v5_window_plan(plan)
    return plan


def development_window_subset(
    plan: V5WindowPlan, maximum_windows: int
) -> V5WindowPlan:
    """Return a deterministic target-disjoint smoke subset with fresh owners.

    This never masquerades as full coverage: the returned plan is permanently
    marked development-only.  Recomputing owners is necessary because a tail
    window in the full plan may overlap an earlier target block.
    """

    if isinstance(maximum_windows, bool) or not isinstance(maximum_windows, int):
        raise TypeError("development-max-windows must be an integer")
    if maximum_windows <= 0:
        raise ValueError("development-max-windows must be positive")
    selected: list[V5WindowSpec] = []
    claimed: set[tuple[str, int, int]] = set()
    for window in plan.windows:
        target_keys = {
            (window.sequence, window.person_id, frame.frame_id)
            for frame, valid in zip(window.target_frames, window.target_valid)
            if valid
        }
        if claimed & target_keys:
            continue
        claimed.update(target_keys)
        selected.append(
            replace(
                window,
                frame_owner=window.target_valid,
                velocity_owner=tuple(
                    window.target_valid[index]
                    and window.target_valid[index + 1]
                    for index in range(TARGET_FRAMES - 1)
                ),
                acceleration_owner=tuple(
                    window.target_valid[index]
                    and window.target_valid[index + 1]
                    and window.target_valid[index + 2]
                    for index in range(TARGET_FRAMES - 2)
                ),
            )
        )
        if len(selected) == maximum_windows:
            break
    if not selected:
        raise RuntimeError("No target-disjoint v5 development windows exist")

    velocity = set()
    acceleration = set()
    for window in selected:
        ids = [frame.frame_id for frame in window.target_frames]
        velocity.update(
            (window.sequence, window.person_id, ids[i], ids[i + 1])
            for i in range(TARGET_FRAMES - 1)
            if window.velocity_owner[i]
        )
        acceleration.update(
            (window.sequence, window.person_id, ids[i], ids[i + 1], ids[i + 2])
            for i in range(TARGET_FRAMES - 2)
            if window.acceleration_owner[i]
        )
    stats = dict(plan.stats)
    stats.update(
        {
            "development_only": True,
            "development_subset_requested_windows": maximum_windows,
            "development_subset_windows": len(selected),
            "development_subset_target_disjoint": True,
            "owned_person_frames": len(claimed),
            "complete_person_frame_coverage": False,
            "target_windows": len(selected),
            "training_velocity_pairs": len(velocity),
            "training_acceleration_triplets": len(acceleration),
            "metric_only_cross_window_velocity_pairs": 0,
            "metric_only_cross_window_acceleration_triplets": 0,
        }
    )
    subset = V5WindowPlan(
        windows=tuple(selected),
        stats=stats,
        expected_frame_keys=frozenset(claimed),
        expected_training_velocity_keys=frozenset(velocity),
        expected_training_acceleration_keys=frozenset(acceleration),
        metric_only_velocity_keys=frozenset(),
        metric_only_acceleration_keys=frozenset(),
    )
    validate_v5_window_plan(subset)
    return subset


def _owned_keys(plan: V5WindowPlan) -> tuple[set[Any], set[Any], set[Any]]:
    frames: list[Any] = []
    velocities: list[Any] = []
    accelerations: list[Any] = []
    for window in plan.windows:
        ids = [frame.frame_id for frame in window.target_frames]
        for local, owned in enumerate(window.frame_owner):
            if owned:
                frames.append((window.sequence, window.person_id, ids[local]))
        for local, owned in enumerate(window.velocity_owner):
            if owned:
                velocities.append(
                    (window.sequence, window.person_id, ids[local], ids[local + 1])
                )
        for local, owned in enumerate(window.acceleration_owner):
            if owned:
                accelerations.append(
                    (
                        window.sequence,
                        window.person_id,
                        ids[local],
                        ids[local + 1],
                        ids[local + 2],
                    )
                )
    for label, values in (
        ("frame", frames),
        ("velocity", velocities),
        ("acceleration", accelerations),
    ):
        if len(values) != len(set(values)):
            raise RuntimeError(f"v5 {label} owner masks contain duplicates")
    return set(frames), set(velocities), set(accelerations)


def validate_v5_window_plan(plan: V5WindowPlan) -> None:
    for window in plan.windows:
        if not (
            len(window.context_frames) == CONTEXT_FRAMES
            and len(window.context_edge_repeat) == CONTEXT_FRAMES
            and len(window.target_frames) == TARGET_FRAMES
            and len(window.target_valid) == TARGET_FRAMES
            and len(window.frame_owner) == TARGET_FRAMES
            and len(window.velocity_owner) == TARGET_FRAMES - 1
            and len(window.acceleration_owner) == TARGET_FRAMES - 2
        ):
            raise ValueError("v5 window or mask length is invalid")
        if any(
            frame.sequence != window.sequence or frame.person_id != window.person_id
            for frame in (*window.context_frames, *window.target_frames)
        ):
            raise ValueError("A v5 window crosses a person track")
        target_ids = [frame.frame_id for frame in window.target_frames]
        valid_count = sum(window.target_valid)
        if valid_count < 1 or window.target_valid != (
            (True,) * valid_count + (False,) * (TARGET_FRAMES - valid_count)
        ):
            raise ValueError("v5 target-valid mask must be a nonempty true prefix")
        if any(
            target_ids[index] != target_ids[index - 1] + 1
            for index in range(1, valid_count)
        ):
            raise ValueError("A v5 target block crosses a frame gap")
        if any(
            target_ids[index] != target_ids[valid_count - 1]
            for index in range(valid_count, TARGET_FRAMES)
        ):
            raise ValueError("Padded v5 targets must repeat the last valid frame")
        if any(
            owned and not valid
            for owned, valid in zip(window.frame_owner, window.target_valid)
        ):
            raise ValueError("A padded target position cannot own a frame")
        non_repeated = [
            frame.frame_id
            for frame, repeated in zip(
                window.context_frames, window.context_edge_repeat
            )
            if not repeated
        ]
        if any(
            non_repeated[index] != non_repeated[index - 1] + 1
            for index in range(1, len(non_repeated))
        ):
            raise ValueError("A v5 context crosses a frame gap")
        expected_target_ids = [
            window.context_frames[TARGET_CONTEXT_START + index].frame_id
            for index in range(TARGET_FRAMES)
        ]
        if expected_target_ids != target_ids:
            raise ValueError("Targets are not context RGB positions 4..11")

    frames, velocities, accelerations = _owned_keys(plan)
    if frames != set(plan.expected_frame_keys):
        raise RuntimeError("v5 frame ownership has missing or extra identities")
    if velocities != set(plan.expected_training_velocity_keys):
        raise RuntimeError("v5 velocity ownership has missing or extra identities")
    if accelerations != set(plan.expected_training_acceleration_keys):
        raise RuntimeError("v5 acceleration ownership has missing or extra identities")
    if velocities & set(plan.metric_only_velocity_keys):
        raise RuntimeError("A velocity pair is both trained and metric-only")
    if accelerations & set(plan.metric_only_acceleration_keys):
        raise RuntimeError("An acceleration triplet is both trained and metric-only")


def stable_noise_seed(window_uid: str, base_seed: int) -> int:
    if base_seed < 0 or base_seed >= 2**63:
        raise ValueError("noise-seed-base must lie in [0,2**63)")
    digest = hashlib.sha256(f"{base_seed}:{window_uid}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


class V5PersonTubeDataset(Dataset):
    def __init__(self, base: Any, plan: V5WindowPlan, feature_only: bool) -> None:
        self.base = base
        self.plan = plan
        self.feature_only = bool(feature_only)

    def __len__(self) -> int:
        return len(self.plan.windows)

    def _feature_crop(self, index: int) -> dict[str, torch.Tensor]:
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
            except AttributeError:
                resampling = Image.BILINEAR
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE), resampling)
            array = np.asarray(image, dtype=np.float32) / 255.0
        return {
            "image": torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1),
            "bbox_xyxy": torch.tensor(bbox, dtype=torch.float32),
            "crop_applied": torch.tensor(
                bbox != (0.0, 0.0, float(width), float(height)), dtype=torch.bool
            ),
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        window = self.plan.windows[index]
        unique_indices = {
            frame.dataset_index
            for frame in (*window.context_frames, *window.target_frames)
        }
        loaded = {
            source_index: (
                self._feature_crop(source_index)
                if self.feature_only
                else self.base[source_index]
            )
            for source_index in unique_indices
        }
        context = [loaded[frame.dataset_index] for frame in window.context_frames]
        targets = [loaded[frame.dataset_index] for frame in window.target_frames]
        result: dict[str, torch.Tensor] = {
            "window_index": torch.tensor(index, dtype=torch.long),
            "images": torch.stack([item["image"] for item in context]),
            "bbox_xyxy": torch.stack([item["bbox_xyxy"] for item in context]),
            "crop_applied": torch.stack([item["crop_applied"] for item in context]),
        }
        if not self.feature_only:
            for name in TARGET_NAMES:
                result[name] = torch.stack([item[name] for item in targets])
        return result


def v5_metadata(plan: V5WindowPlan) -> list[dict[str, Any]]:
    return [
        {
            "window_uid": window.uid,
            "sequence": window.sequence,
            "person_id": window.person_id,
            "segment_id": window.segment_id,
            "target_start": window.target_start,
            "context_frame_ids": [frame.frame_id for frame in window.context_frames],
            "context_source_frame_ids": [
                frame.source_frame_id for frame in window.context_frames
            ],
            "context_frame_uids": [frame.uid for frame in window.context_frames],
            "context_image_paths": [frame.image_path for frame in window.context_frames],
            "target_frame_ids": [frame.frame_id for frame in window.target_frames],
            "target_source_frame_ids": [
                frame.source_frame_id for frame in window.target_frames
            ],
            "target_frame_uids": [frame.uid for frame in window.target_frames],
            "target_valid": list(window.target_valid),
        }
        for window in plan.windows
    ]


def _metric_only_records(keys: Iterable[tuple[Any, ...]]) -> list[dict[str, Any]]:
    records = []
    for key in sorted(keys):
        records.append(
            {
                "sequence": key[0],
                "person_id": key[1],
                "frame_ids": list(key[2:]),
            }
        )
    return records


def validate_v5_payload(payload: dict[str, Any], feature_only: bool) -> None:
    signature = payload.get("signature", {})
    if signature.get("format") not in (FORMAT, STRIDE4_FORMAT):
        raise ValueError("Not a v5 cache payload")
    metadata = payload.get("metadata", [])
    count = len(metadata)
    features = payload.get("features")
    if not isinstance(features, torch.Tensor) or tuple(features.shape) != (
        count,
        LATENT_FRAMES,
        *TOKEN_GRID,
        FEATURE_WIDTH,
    ):
        raise ValueError("Invalid v5 feature tensor shape")
    if features.dtype != torch.bfloat16 or not bool(torch.isfinite(features.float()).all()):
        raise ValueError("v5 features must be finite BF16")
    shapes = {
        "context_edge_repeat_mask": (count, CONTEXT_FRAMES),
        "target_valid_mask": (count, TARGET_FRAMES),
        "frame_owner_mask": (count, TARGET_FRAMES),
        "velocity_owner_mask": (count, TARGET_FRAMES - 1),
        "acceleration_owner_mask": (count, TARGET_FRAMES - 2),
    }
    for name, shape in shapes.items():
        value = payload.get(name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if value.dtype != torch.bool:
            raise ValueError(f"{name} must be boolean")
    for row, record in enumerate(metadata):
        if len(record["context_frame_ids"]) != CONTEXT_FRAMES:
            raise ValueError("Metadata context identity count is not 17")
        if len(record["target_frame_ids"]) != TARGET_FRAMES:
            raise ValueError("Metadata target identity count is not 8")
        valid = payload["target_valid_mask"][row].tolist()
        valid_count = sum(valid)
        if valid_count < 1 or valid != (
            [True] * valid_count + [False] * (TARGET_FRAMES - valid_count)
        ):
            raise ValueError("Target-valid mask must be a nonempty true prefix")
        if record.get("target_valid") != valid:
            raise ValueError("Metadata and tensor target-valid masks disagree")
        target_ids = record["target_frame_ids"]
        if any(
            target_ids[index] != target_ids[index - 1] + 1
            for index in range(1, valid_count)
        ):
            raise ValueError("Valid target metadata crosses a frame gap")
        if any(
            target_ids[index] != target_ids[valid_count - 1]
            for index in range(valid_count, TARGET_FRAMES)
        ):
            raise ValueError("Padded metadata targets must repeat the tail frame")
        if record["context_frame_ids"][4:12] != record["target_frame_ids"]:
            raise ValueError("Metadata targets are not context positions 4..11")
    if feature_only:
        if "targets" in payload:
            raise ValueError("Feature-only v5 cache contains GT targets")
    else:
        expected = {
            "full_pose": (count, TARGET_FRAMES, 72),
            "betas": (count, TARGET_FRAMES, 10),
            "gender_code": (count, TARGET_FRAMES),
            "cam_pose": (count, TARGET_FRAMES, 4, 4),
        }
        targets = payload.get("targets", {})
        for name, shape in expected.items():
            value = targets.get(name)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"Target {name} must have shape {shape}")
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"Target {name} is non-finite")


def main() -> None:
    args = parse_args()
    validate_v5_configuration(
        args.context_length, args.target_length, args.target_stride, args.image_size
    )
    if args.output.suffix.lower() != ".pt":
        raise ValueError("output must end in .pt")
    summary_path = args.output.with_suffix(".json")
    if args.output.exists() or summary_path.exists():
        raise FileExistsError("Refusing to overwrite a v5 cache or summary")
    if (
        args.batch_size <= 0
        or args.num_workers < 0
        or args.log_interval <= 0
        or args.development_max_windows < 0
    ):
        raise ValueError("Invalid batch, worker, or log interval")
    if args.split == "test" and not args.feature_only:
        raise ValueError("Test cache construction requires --feature-only")

    from evaluate_standard_3dpw import (
        Standard3DPWDataset,
        audit_split,
        enforce_complete_test,
    )
    from real_vace_temporal_feature_encoder import wan_latent_frames
    from real_vace_temporal_feature_encoder_v2 import (
        ProtocolAwareTemporalVaceFeatureEncoder,
    )

    if wan_latent_frames(CONTEXT_FRAMES) != LATENT_FRAMES:
        raise RuntimeError("Wan temporal geometry changed: v5 requires 17->5")
    started = time.monotonic()
    audit = audit_split(args.data_root, args.split)
    if args.split == "test":
        enforce_complete_test(audit, allow_incomplete=False)
    dataset = Standard3DPWDataset(
        root=args.data_root,
        split=args.split,
        image_size=IMAGE_SIZE,
        frame_stride=1,
        max_samples=None,
        crop_to_person=True,
    )
    if len(dataset) != int(audit["available_valid_person_frames"]):
        raise RuntimeError("Strict dataset length differs from the split audit")
    plan = build_v5_window_plan(frame_identities_from_samples(dataset.samples))
    if args.development_max_windows:
        if args.split == "test":
            raise ValueError("Formal test forbids development-max-windows")
        plan = development_window_subset(plan, args.development_max_windows)
    if not plan.windows:
        raise RuntimeError("No eligible eight-target track window exists")
    if args.split == "test" and plan.stats["development_only"]:
        raise RuntimeError("Formal test cache refuses skipped short track segments")

    checkpoint = checkpoint_tree_fingerprint(args.vace_checkpoint_dir)
    encoder_path = Path(__file__).with_name(
        "real_vace_temporal_feature_encoder_v2.py"
    )
    base_encoder_path = Path(__file__).with_name(
        "real_vace_temporal_feature_encoder.py"
    )
    encoder = ProtocolAwareTemporalVaceFeatureEncoder(
        checkpoint_dir=args.vace_checkpoint_dir,
        device=args.device,
        image_size=IMAGE_SIZE,
        feature_layers=args.feature_layers,
        timestep=args.vace_timestep,
        diffusion_input_protocol=args.diffusion_input_protocol,
        text_embedding_path=args.text_embedding_path,
    )
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
        "context_frames_rgb": CONTEXT_FRAMES,
        "context_edge_policy": (
            "repeat first/last frame only within the same continuous person track"
        ),
        "context_edge_repeat_mask": "true where desired context was edge-clamped",
        "latent_frames": LATENT_FRAMES,
        "latent_source_ranges_inclusive": [list(value) for value in LATENT_SOURCE_RANGES],
        "target_frames": TARGET_FRAMES,
        "target_context_positions_inclusive": [4, 11],
        "target_stride": TARGET_STRIDE,
        "target_valid_mask": (
            "true for real consecutive targets; a short track tail is padded by "
            "repeating its final RGB frame and marked false"
        ),
        "frame_owner_mask": "each eligible target person-frame has one owner",
        "velocity_owner_mask": "unique training owner within an eight-target block",
        "acceleration_owner_mask": (
            "unique training owner within an eight-target block"
        ),
        "cross_window_motion": (
            "boundary pairs/triplets absent from a single target block are listed "
            "for metrics after full-track prediction stitching"
        ),
        "view_protocol": "oracle_person_tube_crop_before_vace_v1",
        "crop": (
            "independent per-frame oracle poses2d square crop with 25 percent "
            "margin; whole-image fallback when invalid"
        ),
        "multi_person_rgb_encode_reuse": False,
        "multi_person_reuse_reason": (
            "person-specific pre-VACE crops are different pixel inputs"
        ),
        "image_size": IMAGE_SIZE,
        "feature_shape_per_window": [LATENT_FRAMES, *TOKEN_GRID, FEATURE_WIDTH],
        "feature_dtype": "bfloat16",
        "token_order": "time-major, row-major, column-major",
        "feature_backend": "official frozen Wan2.1-VACE continuous video",
        "feature_layers": list(args.feature_layers),
        "encoder_protocol": encoder.protocol_signature(),
        "noise_seed_base": args.noise_seed_base,
        "noise_seed_derivation": "sha256(base_seed:window_uid) first uint64 mod 2^63",
        "checkpoint": checkpoint,
        "temporal_encoder_v2_source": str(encoder_path.resolve()),
        "temporal_encoder_v2_source_sha256": sha256_file(encoder_path),
        "temporal_encoder_base_source_sha256": sha256_file(base_encoder_path),
        "cache_builder_source_sha256": sha256_file(Path(__file__)),
    }

    window_dataset = V5PersonTubeDataset(dataset, plan, args.feature_only)
    loader = DataLoader(
        window_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    metadata = v5_metadata(plan)
    feature_chunks: list[torch.Tensor] = []
    target_chunks = {name: [] for name in TARGET_NAMES}
    expected_window = 0
    crop_applied = crop_fallback = 0
    try:
        for step, batch in enumerate(loader, start=1):
            indices = batch["window_index"].tolist()
            expected = list(range(expected_window, expected_window + len(indices)))
            if indices != expected:
                raise RuntimeError("DataLoader changed deterministic v5 window order")
            expected_window += len(indices)
            seeds = [
                stable_noise_seed(plan.windows[index].uid, args.noise_seed_base)
                for index in indices
            ]
            noise_seeds = (
                seeds if args.diffusion_input_protocol == "flow_noised" else None
            )
            clips = batch["images"].to(
                device=args.device, dtype=torch.float32, non_blocking=True
            )
            features = encoder.encode(clips, noise_seeds=noise_seeds)
            expected_shape = (len(indices), LATENT_FRAMES, *TOKEN_GRID, FEATURE_WIDTH)
            if tuple(features.shape) != expected_shape or not bool(torch.isfinite(features).all()):
                raise RuntimeError(f"Invalid v5 VACE output {tuple(features.shape)}")
            feature_chunks.append(features.to(device="cpu", dtype=torch.bfloat16))
            boxes = batch["bbox_xyxy"].cpu()
            crop_flags = batch["crop_applied"].cpu()
            for row, index in enumerate(indices):
                metadata[index]["context_bbox_xyxy"] = boxes[row].tolist()
                metadata[index]["context_crop_applied"] = crop_flags[row].tolist()
                metadata[index]["noise_seed"] = seeds[row]
            crop_applied += int(crop_flags.sum())
            crop_fallback += int((~crop_flags).sum())
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
                            "phase": "cache_continuous_3dpw_vace_v5",
                            "split": args.split,
                            "step": step,
                            "steps": len(loader),
                            "windows": expected_window,
                            "seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
    finally:
        encoder.close()

    signature["crop_applied_context_occurrences"] = crop_applied
    signature["whole_image_fallback_context_occurrences"] = crop_fallback
    payload: dict[str, Any] = {
        "signature": signature,
        "features": torch.cat(feature_chunks),
        "context_edge_repeat_mask": torch.tensor(
            [window.context_edge_repeat for window in plan.windows], dtype=torch.bool
        ),
        "target_valid_mask": torch.tensor(
            [window.target_valid for window in plan.windows], dtype=torch.bool
        ),
        "frame_owner_mask": torch.tensor(
            [window.frame_owner for window in plan.windows], dtype=torch.bool
        ),
        "velocity_owner_mask": torch.tensor(
            [window.velocity_owner for window in plan.windows], dtype=torch.bool
        ),
        "acceleration_owner_mask": torch.tensor(
            [window.acceleration_owner for window in plan.windows], dtype=torch.bool
        ),
        "metric_only_cross_window_motion": {
            "velocity_pairs": _metric_only_records(plan.metric_only_velocity_keys),
            "acceleration_triplets": _metric_only_records(
                plan.metric_only_acceleration_keys
            ),
        },
        "metadata": metadata,
    }
    if not args.feature_only:
        payload["targets"] = {
            name: torch.cat(chunks) for name, chunks in target_chunks.items()
        }
    validate_v5_payload(payload, args.feature_only)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, args.output)
    result = {
        "status": "DIVIBODY_CONTINUOUS_3DPW_VACE_V5_CACHE_OK",
        "format": FORMAT,
        "split": args.split,
        "feature_only": bool(args.feature_only),
        "development_only": bool(plan.stats["development_only"]),
        "windows": len(plan.windows),
        "feature_shape": list(payload["features"].shape),
        "owned_person_frames": int(payload["frame_owner_mask"].sum()),
        "training_velocity_pairs": int(payload["velocity_owner_mask"].sum()),
        "training_acceleration_triplets": int(
            payload["acceleration_owner_mask"].sum()
        ),
        "metric_only_velocity_pairs": len(plan.metric_only_velocity_keys),
        "metric_only_acceleration_triplets": len(
            plan.metric_only_acceleration_keys
        ),
        "output": str(args.output.expanduser().resolve()),
        "output_sha256": sha256_file(args.output),
        "seconds": time.monotonic() - started,
    }
    summary_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
