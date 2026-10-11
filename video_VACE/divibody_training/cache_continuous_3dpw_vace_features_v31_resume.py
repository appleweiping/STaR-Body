"""Resumable V31 builder preserving the original v5/v6 cache protocol.

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
import uuid
import re
import math
import platform
from contextlib import contextmanager
import PIL
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
    parser.add_argument("--target-stride", type=int, choices=(4, 8), default=None)
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
    parser.add_argument("--num-workers", type=int, default=0)
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
    parser.add_argument("--shard-dir", type=Path, help="Default: OUTPUT.pt.shards")
    parser.add_argument("--shard-windows", type=int, choices=(32, 64), default=32)
    parser.add_argument("--max-new-windows", type=int, default=0,
                        help="Commit at most N new windows, then stop; zero finishes the full plan")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true",
                        help="CPU dataset/window audit only; no encoder, assets hashing, or files written")
    parser.add_argument("--gpu-memory-limit-gib", type=float, default=12.0)
    parser.add_argument("--threads", type=int, default=4)
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



def make_signature(args, audit, plan, checkpoint, protocol):
    encoder_path = Path(__file__).with_name("real_vace_temporal_feature_encoder_v2.py")
    base_encoder_path = Path(__file__).with_name("real_vace_temporal_feature_encoder.py")
    return {
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
        "encoder_protocol": protocol,
        "noise_seed_base": args.noise_seed_base,
        "noise_seed_derivation": "sha256(base_seed:window_uid) first uint64 mod 2^63",
        "checkpoint": checkpoint,
        "temporal_encoder_v2_source": str(encoder_path.resolve()),
        "temporal_encoder_v2_source_sha256": sha256_file(encoder_path),
        "temporal_encoder_base_source_sha256": sha256_file(base_encoder_path),
        "cache_builder_source_sha256": sha256_file(Path(__file__)),
    }

# Appended to a byte-preserving copy of the original v5 helper definitions.
SHARD_FORMAT = "star-body-v31-vace-shard-v1"
RUN_FORMAT = "star-body-v31-vace-resume-v1"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def sync_directory(path: Path) -> None:
    if os.name == "posix":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def atomic_json(path: Path, value: Any) -> None:
    """Publish a new JSON file, refusing to replace any existing destination."""
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    # Linking is an atomic no-clobber publication on the same filesystem.
    os.link(temporary, path)
    temporary.unlink()
    sync_directory(path.parent)


@contextmanager
def run_lock(directory: Path):
    """An OS lock is released even after process death; the marker is reusable."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+b") as stream:
        if os.name == "posix":
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another cache writer holds the shard directory") from error
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        else:
            import msvcrt
            stream.write(b"0")
            stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def bind_run(directory: Path, manifest: dict[str, Any], resume: bool) -> None:
    path = directory / "run.json"
    if path.exists():
        if not resume:
            raise FileExistsError("Shard run already exists; use --resume with identical assets/config")
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored != manifest:
            keys = sorted(key for key in set(stored) | set(manifest)
                          if stored.get(key) != manifest.get(key))
            raise ValueError("Resume invariant mismatch: " + ", ".join(keys))
    else:
        if resume:
            raise FileNotFoundError("--resume requires an existing run.json")
        if any(directory.glob("shard_*")):
            raise RuntimeError("Committed shards exist without a run manifest")
        atomic_json(path, manifest)


def scan_shards(directory: Path, run_sha256: str, window_uids: list[str]) -> list[dict[str, Any]]:
    """Verify every committed byte before loading tensors or starting the GPU."""
    records = []
    cursor = 0
    for path in sorted(directory.glob("shard_*")):
        match = re.fullmatch(r"shard_(\d{8})_(\d{8})", path.name)
        if match is None or not path.is_dir() or path.is_symlink():
            raise ValueError(f"Unexpected committed shard path: {path}")
        start, end = map(int, match.groups())
        if start != cursor or not start < end <= len(window_uids):
            raise ValueError(f"Non-contiguous, overlapping, or out-of-range shard: {path}")
        if {item.name for item in path.iterdir()} != {"record.json", "data.pt"}:
            raise ValueError(f"Incomplete or unexpected shard contents: {path}")
        record_path, data_path = path / "record.json", path / "data.pt"
        if record_path.is_symlink() or data_path.is_symlink():
            raise ValueError("Shard symlinks are not allowed")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        expected = {"format": SHARD_FORMAT, "run_sha256": run_sha256,
                    "start": start, "end": end,
                    "window_uids": window_uids[start:end]}
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Shard UID or invariant mismatch: {path}")
        if record.get("data_bytes") != data_path.stat().st_size:
            raise ValueError(f"Shard byte length mismatch: {path}")
        if record.get("data_sha256") != sha256_file(data_path):
            raise ValueError(f"Shard SHA256 mismatch: {path}")
        records.append(record)
        cursor = end
    return records


def commit_shard(directory: Path, run_sha256: str, start: int, end: int,
                 window_uids: list[str], payload: dict[str, Any]) -> dict[str, Any]:
    destination = directory / f"shard_{start:08d}_{end:08d}"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite committed shard: {destination}")
    staging = directory / (".staging-" + uuid.uuid4().hex)
    staging.mkdir()
    data_path = staging / "data.pt"
    with data_path.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    record = {"format": SHARD_FORMAT, "run_sha256": run_sha256,
              "start": start, "end": end, "window_uids": window_uids[start:end],
              "data_bytes": data_path.stat().st_size, "data_sha256": sha256_file(data_path)}
    atomic_json(staging / "record.json", record)
    sync_directory(staging)
    os.rename(staging, destination)
    sync_directory(directory)
    return record


def source_fingerprints() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = [Path(__file__).name, "cache_continuous_3dpw_vace_features_v5.py",
             "cache_continuous_3dpw_vace_features.py", "evaluate_standard_3dpw.py",
             "real_3dpw_dataset.py", "real_vace_feature_encoder.py",
             "real_vace_temporal_feature_encoder.py", "real_vace_temporal_feature_encoder_v2.py"]
    result = {name: sha256_file(root / name) for name in names}
    import importlib.util
    spec = importlib.util.find_spec("wan")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("Official Wan package is unavailable; configure WAN_SOURCE_ROOT")
    wan_root = Path(next(iter(spec.submodule_search_locations))).resolve()
    for path in sorted(wan_root.rglob("*.py")):
        result["wan/" + path.relative_to(wan_root).as_posix()] = sha256_file(path)
    # The VACE DiT wrapper and SDPA compatibility shim are part of the encoder.
    result["run_vace_sdpa.py"] = sha256_file(root / "run_vace_sdpa.py")
    vace_spec = importlib.util.find_spec("vace")
    if vace_spec is None or not vace_spec.submodule_search_locations:
        raise ImportError("VACE wrapper package is unavailable")
    vace_root = Path(next(iter(vace_spec.submodule_search_locations))).resolve()
    for path in sorted(vace_root.rglob("*.py")):
        result["vace/" + path.relative_to(vace_root).as_posix()] = sha256_file(path)
    return result


def expected_encoder_protocol(args: argparse.Namespace) -> dict[str, Any]:
    """Read the original protocol method and optional text tensor on CPU."""
    from types import SimpleNamespace
    from real_vace_temporal_feature_encoder_v2 import ProtocolAwareTemporalVaceFeatureEncoder as Encoder
    path = None if args.text_embedding_path is None else args.text_embedding_path.expanduser().resolve()
    embedding, text_protocol, prompt = Encoder._load_text_embedding(path)
    del embedding
    properties = SimpleNamespace(diffusion_input_protocol=args.diffusion_input_protocol,
                                 timestep=args.vace_timestep, text_protocol=text_protocol,
                                 text_prompt=prompt, text_embedding_path=path)
    return Encoder.protocol_signature(properties)


def dataset_fingerprint(dataset: Any, plan: V5WindowPlan) -> dict[str, Any]:
    """Content hash annotations and every unique RGB actually used by the plan."""
    paths = sorted({str(path.resolve()) for path in dataset.split_directory.glob("*.pkl")}
                   | {str(Path(frame.image_path).resolve()) for window in plan.windows
                      for frame in window.context_frames})
    records = []
    for index, name in enumerate(paths, start=1):
        path = Path(name)
        before = path.stat()
        digest = sha256_file(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Asset changed while hashing: {path}")
        records.append({"path": name, "bytes": after.st_size, "sha256": digest})
        if index % 1000 == 0 or index == len(paths):
            print(json.dumps({"phase": "fingerprint_raw_assets", "files": index,
                              "total_files": len(paths)}), flush=True)
    return {"files": records, "tree_sha256": canonical_sha256(records)}


def plan_identity(plan: V5WindowPlan) -> dict[str, Any]:
    return {"metadata": v5_metadata(plan),
            "context_edge_repeat": [list(w.context_edge_repeat) for w in plan.windows],
            "target_valid": [list(w.target_valid) for w in plan.windows],
            "frame_owner": [list(w.frame_owner) for w in plan.windows],
            "velocity_owner": [list(w.velocity_owner) for w in plan.windows],
            "acceleration_owner": [list(w.acceleration_owner) for w in plan.windows],
            "metric_only_velocity": _metric_only_records(plan.metric_only_velocity_keys),
            "metric_only_acceleration": _metric_only_records(plan.metric_only_acceleration_keys)}


def load_and_validate_shard(directory: Path, record: dict[str, Any], plan: V5WindowPlan,
                            args: argparse.Namespace) -> dict[str, Any]:
    start, end = record["start"], record["end"]
    path = directory / f"shard_{start:08d}_{end:08d}" / "data.pt"
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("run_sha256") != record["run_sha256"]:
        raise ValueError(f"Embedded shard invariant mismatch: {path}")
    if value.get("start") != start or value.get("end") != end:
        raise ValueError(f"Embedded shard range mismatch: {path}")
    features = value.get("features")
    if not isinstance(features, torch.Tensor) or features.dtype != torch.bfloat16:
        raise ValueError("Shard features must be BF16 tensors")
    if tuple(features.shape) != (end-start, LATENT_FRAMES, *TOKEN_GRID, FEATURE_WIDTH):
        raise ValueError("Shard feature shape mismatch")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("Non-finite shard features")
    expected_metadata = v5_metadata(replace(plan, windows=plan.windows[start:end]))
    if len(value.get("metadata", [])) != end-start:
        raise ValueError("Shard metadata length mismatch")
    for offset, expected in enumerate(expected_metadata):
        actual = value["metadata"][offset]
        if any(actual.get(key) != item for key, item in expected.items()):
            raise ValueError("Shard metadata differs from the complete raw window plan")
        if actual.get("noise_seed") != stable_noise_seed(expected["window_uid"], args.noise_seed_base):
            raise ValueError("Shard noise seed mismatch")
        if len(actual.get("context_crop_applied", [])) != CONTEXT_FRAMES:
            raise ValueError("Shard crop flag length mismatch")
        boxes = np.asarray(actual.get("context_bbox_xyxy", []))
        if boxes.shape != (CONTEXT_FRAMES, 4) or not np.isfinite(boxes).all():
            raise ValueError("Shard crop boxes are invalid")
    if args.feature_only:
        if "targets" in value:
            raise ValueError("Feature-only shard contains labels")
    else:
        shapes = {"full_pose": (end-start, 8, 72), "betas": (end-start, 8, 10),
                  "gender_code": (end-start, 8), "cam_pose": (end-start, 8, 4, 4)}
        for name, shape in shapes.items():
            tensor = value.get("targets", {}).get(name)
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
                raise ValueError(f"Shard target {name} shape mismatch")
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"Non-finite shard target {name}")
    return value


def merge_shards(args: argparse.Namespace, plan: V5WindowPlan, directory: Path,
                 records: list[dict[str, Any]], signature: dict[str, Any], started: float) -> None:
    """One final tensor plus one small shard; never collect all feature chunks."""
    count = len(plan.windows)
    if not records or records[-1]["end"] != count:
        raise ValueError("Final publication requires every planned window")
    payload = {"signature": signature,
               "features": torch.empty((count, LATENT_FRAMES, *TOKEN_GRID, FEATURE_WIDTH),
                                       dtype=torch.bfloat16),
               "metadata": [],
               "metric_only_cross_window_motion": {
                   "velocity_pairs": _metric_only_records(plan.metric_only_velocity_keys),
                   "acceleration_triplets": _metric_only_records(plan.metric_only_acceleration_keys)}}
    for key, field in (("context_edge_repeat_mask", "context_edge_repeat"),
                       ("target_valid_mask", "target_valid"), ("frame_owner_mask", "frame_owner"),
                       ("velocity_owner_mask", "velocity_owner"),
                       ("acceleration_owner_mask", "acceleration_owner")):
        payload[key] = torch.tensor([getattr(w, field) for w in plan.windows], dtype=torch.bool)
    for record in records:
        shard = load_and_validate_shard(directory, record, plan, args)
        start, end = record["start"], record["end"]
        payload["features"][start:end].copy_(shard["features"])
        payload["metadata"].extend(shard["metadata"])
        if not args.feature_only:
            if "targets" not in payload:
                payload["targets"] = {name: torch.empty((count, *value.shape[1:]), dtype=value.dtype)
                                      for name, value in shard["targets"].items()}
            for name, value in shard["targets"].items():
                payload["targets"][name][start:end].copy_(value)
        del shard
    applied = sum(sum(record["context_crop_applied"]) for record in payload["metadata"])
    signature["crop_applied_context_occurrences"] = applied
    signature["whole_image_fallback_context_occurrences"] = count*CONTEXT_FRAMES - applied
    # Original validator casts the entire feature tensor to FP32. Validate in
    # small slices instead, keeping identical metadata/mask/target checks.
    for start in range(0, count, 32):
        end = min(start+32, count)
        part = {key: (value[start:end] if isinstance(value, torch.Tensor) else value)
                for key, value in payload.items() if key not in ("metadata", "targets")}
        part["metadata"] = payload["metadata"][start:end]
        if "targets" in payload:
            part["targets"] = {key: value[start:end] for key, value in payload["targets"].items()}
        validate_v5_payload(part, args.feature_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    digest = sha256_file(temporary)
    os.link(temporary, args.output)  # Never overwrite a concurrently published output.
    temporary.unlink()
    sync_directory(args.output.parent)
    result = {"status": "DIVIBODY_CONTINUOUS_3DPW_VACE_V31_RESUME_CACHE_OK",
              "format": FORMAT, "split": args.split, "feature_only": bool(args.feature_only),
              "development_only": bool(plan.stats["development_only"]), "windows": count,
              "feature_shape": list(payload["features"].shape),
              "owned_person_frames": int(payload["frame_owner_mask"].sum()),
              "training_velocity_pairs": int(payload["velocity_owner_mask"].sum()),
              "training_acceleration_triplets": int(payload["acceleration_owner_mask"].sum()),
              "metric_only_velocity_pairs": len(plan.metric_only_velocity_keys),
              "metric_only_acceleration_triplets": len(plan.metric_only_acceleration_keys),
              "output": str(args.output.resolve()), "output_sha256": digest,
              "run_sha256": signature["resumable_builder"]["run_sha256"],
              "shards": len(records), "seconds_this_invocation": time.monotonic()-started}
    atomic_json(args.output.with_suffix(".json"), result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    global FORMAT, TARGET_STRIDE
    args = parse_args()
    args.target_stride = args.target_stride or (8 if args.split == "train" else 4)
    TARGET_STRIDE = args.target_stride
    FORMAT = STRIDE4_FORMAT if TARGET_STRIDE == 4 else "divibody-contiguous-vace-video-features-v5"
    validate_v5_configuration(args.context_length, args.target_length, args.target_stride, args.image_size)
    if args.output.suffix.lower() != ".pt":
        raise ValueError("output must end in .pt")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError("Refusing to overwrite a completed cache or summary")
    if args.batch_size != 1 or args.num_workers < 0 or args.threads <= 0:
        raise ValueError("This bounded-memory builder requires batch-size=1 and valid threads/workers")
    if args.shard_windows not in (32, 64) or args.max_new_windows < 0 or args.log_interval <= 0:
        raise ValueError("shard-windows must be 32 or 64; invalid maximum/log interval")
    if args.development_max_windows < 0 or not math.isfinite(args.gpu_memory_limit_gib) or args.gpu_memory_limit_gib <= 0:
        raise ValueError("Invalid development limit or GPU memory ceiling")
    if args.split == "test" and (not args.feature_only or args.development_max_windows):
        raise ValueError("Test requires --feature-only and the complete plan")
    if not args.vace_checkpoint_dir.is_dir():
        raise FileNotFoundError(args.vace_checkpoint_dir)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    from evaluate_standard_3dpw import Standard3DPWDataset, audit_split, enforce_complete_test
    started = time.monotonic()
    audit = audit_split(args.data_root, args.split)
    if args.split == "test":
        enforce_complete_test(audit, allow_incomplete=False)
    if (int(audit["missing_unique_rgb_frames"]) or audit["invalid_gender_tracks"]
            or int(audit["expected_valid_person_frames"]) != int(audit["available_valid_person_frames"])):
        raise RuntimeError("Full cache requires all annotation-backed RGB frames")
    dataset = Standard3DPWDataset(root=args.data_root, split=args.split, image_size=IMAGE_SIZE,
                                 frame_stride=1, max_samples=None, crop_to_person=True)
    if len(dataset) != int(audit["available_valid_person_frames"]):
        raise RuntimeError("Strict dataset length differs from the split audit")
    plan = build_v5_window_plan(frame_identities_from_samples(dataset.samples))
    if args.development_max_windows:
        plan = development_window_subset(plan, args.development_max_windows)
    if not plan.windows:
        raise RuntimeError("No eligible target windows")
    from real_vace_temporal_feature_encoder import wan_latent_frames
    if wan_latent_frames(CONTEXT_FRAMES) != LATENT_FRAMES:
        raise RuntimeError("Wan temporal geometry changed: expected 17 RGB -> 5 latent frames")
    identity = plan_identity(plan)
    count = len(plan.windows)
    window_uids = [window.uid for window in plan.windows]
    directory = (args.shard_dir or Path(str(args.output) + ".shards")).expanduser().resolve()
    if args.plan_only:
        print(json.dumps({"status": "V31_VACE_CPU_PLAN_ONLY", "format": FORMAT,
                          "split": args.split, "target_stride": TARGET_STRIDE,
                          "windows": count, "window_plan": plan.stats,
                          "window_plan_sha256": canonical_sha256(identity),
                          "feature_tensor_bytes": count*LATENT_FRAMES*36*FEATURE_WIDTH*2,
                          "shard_dir": str(directory), "output": str(args.output.resolve()),
                          "asset_content_hashes_checked": False, "cuda_encoder_loaded": False,
                          "dataset_preflight": audit}, indent=2), flush=True)
        return
    with run_lock(directory):
        source = source_fingerprints()
        print(json.dumps({"phase": "fingerprint_checkpoint"}), flush=True)
        checkpoint = checkpoint_tree_fingerprint(args.vace_checkpoint_dir)
        raw_assets = dataset_fingerprint(dataset, plan)
        protocol = expected_encoder_protocol(args)
        config = {key: value for key, value in vars(args).items()
                  if key not in {"output", "shard_dir", "resume", "plan_only", "max_new_windows",
                                 "shard_windows", "log_interval", "device", "gpu_memory_limit_gib"}}
        config = {key: str(value.expanduser().resolve()) if isinstance(value, Path) else value
                  for key, value in config.items()}
        # JSON round-trip makes argparse tuples and their restored lists identical.
        config = json.loads(json.dumps(config))
        assets = {"checkpoint": checkpoint, "raw_dataset": raw_assets,
                  "text_embedding": None if args.text_embedding_path is None else {
                      "path": str(args.text_embedding_path.resolve()),
                      "sha256": sha256_file(args.text_embedding_path)}}
        manifest = {"format": RUN_FORMAT, "config": config, "sources": source,
                    "assets": assets, "encoder_protocol": protocol,
                    "window_plan_sha256": canonical_sha256(identity),
                    "window_uids": window_uids, "window_plan": plan.stats,
                    "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                    "numpy": np.__version__, "pillow": PIL.__version__,
                                    "cuda_build": torch.version.cuda}}
        bind_run(directory, manifest, args.resume)
        run_sha = canonical_sha256(manifest)
        records = scan_shards(directory, run_sha, window_uids)
        for record in records:
            del_payload = load_and_validate_shard(directory, record, plan, args)
            del del_payload
        cursor = records[-1]["end"] if records else 0
        stop = count if args.max_new_windows == 0 else min(count, cursor+args.max_new_windows)
        if cursor < stop:
            if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
                raise RuntimeError("Encoding requires CUDA; --plan-only is CPU-only")
            device = torch.device(args.device)
            if device.index is None:
                device = torch.device('cuda', torch.cuda.current_device())
            total = torch.cuda.get_device_properties(device).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, args.gpu_memory_limit_gib*2**30/total), device)
            torch.manual_seed(args.noise_seed_base)
            np.random.seed(args.noise_seed_base % 2**32)
            from real_vace_temporal_feature_encoder_v2 import ProtocolAwareTemporalVaceFeatureEncoder
            encoder = ProtocolAwareTemporalVaceFeatureEncoder(
                checkpoint_dir=args.vace_checkpoint_dir, device=args.device, image_size=IMAGE_SIZE,
                feature_layers=args.feature_layers, timestep=args.vace_timestep,
                diffusion_input_protocol=args.diffusion_input_protocol, text_embedding_path=args.text_embedding_path)
            try:
                observed_protocol = encoder.protocol_signature()
                if protocol != observed_protocol:
                    raise ValueError("Resumed encoder protocol changed")
                window_dataset = V5PersonTubeDataset(dataset, plan, args.feature_only)
                loader = DataLoader(torch.utils.data.Subset(window_dataset, range(cursor, stop)),
                                    batch_size=1, shuffle=False, num_workers=args.num_workers,
                                    pin_memory=True, persistent_workers=args.num_workers > 0)
                shard_start = cursor
                feature_buffer = torch.empty((min(args.shard_windows, stop-cursor), LATENT_FRAMES,
                                               *TOKEN_GRID, FEATURE_WIDTH), dtype=torch.bfloat16)
                target_buffer: dict[str, torch.Tensor] = {}
                metadata_buffer = []
                for batch in loader:
                    indices = batch["window_index"].tolist()
                    if indices != [cursor]:
                        raise RuntimeError("DataLoader changed deterministic window order")
                    seed = stable_noise_seed(plan.windows[cursor].uid, args.noise_seed_base)
                    # Only observed RGB goes to CUDA; annotation targets remain on CPU.
                    clips = batch["images"].to(device=args.device, dtype=torch.float32, non_blocking=True)
                    with torch.inference_mode():
                        features = encoder.encode(clips, noise_seeds=[seed] if args.diffusion_input_protocol == "flow_noised" else None)
                    if tuple(features.shape) != (1, LATENT_FRAMES, *TOKEN_GRID, FEATURE_WIDTH) or not bool(torch.isfinite(features).all()):
                        raise RuntimeError("Invalid VACE output shape or non-finite values")
                    offset = cursor-shard_start
                    feature_buffer[offset].copy_(features[0].to(device="cpu", dtype=torch.bfloat16))
                    record = dict(identity["metadata"][cursor])
                    record.update(context_bbox_xyxy=batch["bbox_xyxy"][0].cpu().tolist(),
                                  context_crop_applied=batch["crop_applied"][0].cpu().tolist(), noise_seed=seed)
                    metadata_buffer.append(record)
                    if not args.feature_only:
                        for name in TARGET_NAMES:
                            value = batch[name][0].detach().cpu()
                            value = value.float() if value.is_floating_point() else value
                            if name not in target_buffer:
                                target_buffer[name] = torch.empty((len(feature_buffer), *value.shape), dtype=value.dtype)
                            target_buffer[name][offset].copy_(value)
                    del clips, features
                    cursor += 1
                    if cursor-shard_start == len(feature_buffer) or cursor == stop:
                        shard = {"run_sha256": run_sha, "start": shard_start, "end": cursor,
                                 "features": feature_buffer[:cursor-shard_start], "metadata": metadata_buffer}
                        if not args.feature_only:
                            shard["targets"] = {name: value[:cursor-shard_start] for name, value in target_buffer.items()}
                        records.append(commit_shard(directory, run_sha, shard_start, cursor, window_uids, shard))
                        print(json.dumps({"phase": "shard_committed", "start": shard_start, "end": cursor,
                                          "windows": count, "run_sha256": run_sha,
                                          "seconds_this_invocation": time.monotonic()-started}), flush=True)
                        del shard
                        shard_start = cursor
                        if cursor < stop:
                            feature_buffer = torch.empty((min(args.shard_windows, stop-cursor), LATENT_FRAMES,
                                                           *TOKEN_GRID, FEATURE_WIDTH), dtype=torch.bfloat16)
                            target_buffer, metadata_buffer = {}, []
                    elif cursor % args.log_interval == 0:
                        print(json.dumps({"phase": "encode", "encoded_windows": cursor,
                                          "committed_windows": shard_start, "windows": count}), flush=True)
            finally:
                encoder.close()
        if cursor < count:
            print(json.dumps({"status": "V31_VACE_PARTIAL_CHECKPOINTED", "completed_windows": cursor,
                              "total_windows": count, "run_sha256": run_sha,
                              "resume_with": "same command plus --resume; max-new-windows may change",
                              "output_published": False}), flush=True)
            return
        if protocol is None:
            raise RuntimeError("No encoder protocol exists for final merge")
        signature = make_signature(args, audit, plan, checkpoint, protocol)
        signature["resumable_builder"] = {"format": RUN_FORMAT, "run_sha256": run_sha,
                                          "source_sha256": canonical_sha256(source),
                                          "asset_sha256": canonical_sha256(assets),
                                          "config_sha256": canonical_sha256(config)}
        # Recheck immutable source files before publication; per-file asset hashes
        # were recomputed on every invocation before any shards were trusted.
        if source_fingerprints() != source:
            raise RuntimeError("Builder/encoder source changed during encoding")
        merge_shards(args, plan, directory, records, signature, started)


if __name__ == "__main__":
    main()
