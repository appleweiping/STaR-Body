"""CPU tests for v5 continuous 3DPW cache planning and payload contracts."""

from __future__ import annotations

import unittest

import torch

import cache_continuous_3dpw_vace_features_v5 as cache_v5
from cache_continuous_3dpw_vace_features import FrameIdentity
from cache_continuous_3dpw_vace_features_v5 import (
    CONTEXT_FRAMES,
    FEATURE_WIDTH,
    FORMAT,
    IMAGE_SIZE,
    LATENT_FRAMES,
    TARGET_FRAMES,
    TOKEN_GRID,
    build_v5_window_plan,
    development_window_subset,
    stable_noise_seed,
    v5_metadata,
    validate_v5_configuration,
    validate_v5_payload,
)


def make_frames(
    frame_ids: list[int], sequence: str = "sequence_a", person_id: int = 0
) -> list[FrameIdentity]:
    return [
        FrameIdentity(
            dataset_index=index,
            sequence=sequence,
            person_id=person_id,
            frame_id=frame_id,
            source_frame_id=frame_id * 2,
            image_path=f"/{sequence}/image_{frame_id:05d}.jpg",
        )
        for index, frame_id in enumerate(frame_ids)
    ]


class V5WindowPlanTests(unittest.TestCase):
    def test_configuration_rejects_silent_tail_lengths(self) -> None:
        validate_v5_configuration(17, 8, 8, 96)
        for context in (8, 9, 16, 18):
            with self.subTest(context=context):
                with self.assertRaisesRegex(ValueError, "17 context"):
                    validate_v5_configuration(context, 8, 8, 96)
        with self.assertRaisesRegex(ValueError, "eight target"):
            validate_v5_configuration(17, 7, 8, 96)

    def test_exact_blocks_have_unique_frames_and_metric_boundaries(self) -> None:
        plan = build_v5_window_plan(make_frames(list(range(24))))
        self.assertEqual([window.target_start for window in plan.windows], [0, 8, 16])
        self.assertEqual(len(plan.expected_frame_keys), 24)
        self.assertEqual(sum(sum(window.frame_owner) for window in plan.windows), 24)
        self.assertEqual(len(plan.expected_training_velocity_keys), 21)
        self.assertEqual(len(plan.metric_only_velocity_keys), 2)
        self.assertEqual(len(plan.expected_training_acceleration_keys), 18)
        self.assertEqual(len(plan.metric_only_acceleration_keys), 4)
        self.assertEqual(
            plan.metric_only_velocity_keys,
            frozenset(
                {
                    ("sequence_a", 0, 7, 8),
                    ("sequence_a", 0, 15, 16),
                }
            ),
        )
        self.assertEqual(sum(plan.windows[0].context_edge_repeat), 4)
        self.assertEqual(sum(plan.windows[-1].context_edge_repeat), 5)
        for window in plan.windows:
            self.assertEqual(
                [frame.frame_id for frame in window.context_frames[4:12]],
                [frame.frame_id for frame in window.target_frames],
            )

    def test_tail_target_overlap_still_owns_each_frame_once(self) -> None:
        plan = build_v5_window_plan(make_frames(list(range(18))))
        self.assertEqual([window.target_start for window in plan.windows], [0, 8, 10])
        self.assertEqual(plan.stats["tail_overlap_window_pairs"], 1)
        self.assertEqual(sum(sum(window.frame_owner) for window in plan.windows), 18)
        self.assertEqual(len(plan.expected_frame_keys), 18)
        self.assertEqual(
            plan.metric_only_velocity_keys,
            frozenset({("sequence_a", 0, 7, 8)}),
        )
        self.assertEqual(
            plan.metric_only_acceleration_keys,
            frozenset(
                {
                    ("sequence_a", 0, 6, 7, 8),
                    ("sequence_a", 0, 7, 8, 9),
                }
            ),
        )

    def test_stride4_overlap_contains_every_motion_primitive(self) -> None:
        original = cache_v5.TARGET_STRIDE
        cache_v5.TARGET_STRIDE = 4
        try:
            plan = build_v5_window_plan(make_frames(list(range(18))))
        finally:
            cache_v5.TARGET_STRIDE = original
        self.assertEqual([window.target_start for window in plan.windows], [0, 4, 8, 10])
        self.assertEqual(len(plan.expected_frame_keys), 18)
        self.assertEqual(len(plan.expected_training_velocity_keys), 17)
        self.assertEqual(len(plan.expected_training_acceleration_keys), 16)
        self.assertFalse(plan.metric_only_velocity_keys)
        self.assertFalse(plan.metric_only_acceleration_keys)

    def test_gap_is_never_crossed_and_short_segment_is_edge_padded(self) -> None:
        frames = make_frames(list(range(10)) + list(range(12, 19)))
        plan = build_v5_window_plan(frames)
        self.assertEqual(plan.stats["segments"], 2)
        self.assertEqual(plan.stats["short_segments_edge_padded"], 1)
        self.assertEqual(plan.stats["padded_target_positions"], 1)
        self.assertEqual(plan.stats["uncovered_short_segment_person_frames"], 0)
        self.assertFalse(plan.stats["development_only"])
        self.assertEqual(len(plan.expected_frame_keys), 17)
        short = next(window for window in plan.windows if window.segment_id == 1)
        self.assertEqual(short.target_valid, (True,) * 7 + (False,))
        self.assertEqual(short.frame_owner, short.target_valid)
        self.assertEqual(short.velocity_owner, (True,) * 6 + (False,))
        self.assertEqual(short.acceleration_owner, (True,) * 5 + (False,))
        self.assertEqual(
            [frame.frame_id for frame in short.target_frames],
            [12, 13, 14, 15, 16, 17, 18, 18],
        )
        for window in plan.windows:
            context_ids = {frame.frame_id for frame in window.context_frames}
            self.assertFalse(context_ids & set(range(10)) and context_ids & set(range(12, 19)))

    def test_tracks_are_independent(self) -> None:
        first = make_frames(list(range(8)), "sequence_a", 0)
        second = [
            FrameIdentity(
                dataset_index=100 + index,
                sequence="sequence_a",
                person_id=1,
                frame_id=frame_id,
                source_frame_id=frame_id * 2,
                image_path=f"/sequence_a/p1_{frame_id}.jpg",
            )
            for index, frame_id in enumerate(range(8))
        ]
        plan = build_v5_window_plan(first + second)
        self.assertEqual(len(plan.windows), 2)
        self.assertEqual(len(plan.expected_frame_keys), 16)
        self.assertEqual({window.person_id for window in plan.windows}, {0, 1})

    def test_noise_seed_is_stable_and_window_specific(self) -> None:
        first = stable_noise_seed("window-a", 2026)
        self.assertEqual(first, stable_noise_seed("window-a", 2026))
        self.assertNotEqual(first, stable_noise_seed("window-b", 2026))
        self.assertNotEqual(first, stable_noise_seed("window-a", 2027))
        with self.assertRaisesRegex(ValueError, "noise-seed-base"):
            stable_noise_seed("window-a", -1)

    def test_development_subset_rebuilds_unique_owners(self) -> None:
        full = build_v5_window_plan(make_frames(list(range(18))))
        subset = development_window_subset(full, 2)
        self.assertEqual(len(subset.windows), 2)
        self.assertTrue(subset.stats["development_only"])
        self.assertTrue(subset.stats["development_subset_target_disjoint"])
        self.assertEqual(sum(sum(w.frame_owner) for w in subset.windows), 16)
        self.assertEqual(len(subset.expected_frame_keys), 16)
        self.assertEqual(len(subset.expected_training_velocity_keys), 14)
        self.assertEqual(len(subset.expected_training_acceleration_keys), 12)
        self.assertFalse(subset.metric_only_velocity_keys)
        self.assertFalse(subset.metric_only_acceleration_keys)
        with self.assertRaisesRegex(ValueError, "positive"):
            development_window_subset(full, 0)


class V5PayloadContractTests(unittest.TestCase):
    def make_payload(self, feature_only: bool) -> dict:
        plan = build_v5_window_plan(make_frames(list(range(8))))
        count = len(plan.windows)
        payload = {
            "signature": {"format": FORMAT},
            "features": torch.zeros(
                count, LATENT_FRAMES, *TOKEN_GRID, FEATURE_WIDTH,
                dtype=torch.bfloat16,
            ),
            "context_edge_repeat_mask": torch.tensor(
                [window.context_edge_repeat for window in plan.windows],
                dtype=torch.bool,
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
                "velocity_pairs": [],
                "acceleration_triplets": [],
            },
            "metadata": v5_metadata(plan),
        }
        if not feature_only:
            payload["targets"] = {
                "full_pose": torch.zeros(count, TARGET_FRAMES, 72),
                "betas": torch.zeros(count, TARGET_FRAMES, 10),
                "gender_code": torch.zeros(count, TARGET_FRAMES, dtype=torch.long),
                "cam_pose": torch.eye(4).reshape(1, 1, 4, 4).expand(
                    count, TARGET_FRAMES, 4, 4
                ).clone(),
            }
        return payload

    def test_feature_only_payload_has_no_targets(self) -> None:
        payload = self.make_payload(feature_only=True)
        validate_v5_payload(payload, feature_only=True)
        self.assertNotIn("targets", payload)
        payload["targets"] = {}
        with self.assertRaisesRegex(ValueError, "contains GT"):
            validate_v5_payload(payload, feature_only=True)

    def test_training_payload_has_exact_target_shapes(self) -> None:
        payload = self.make_payload(feature_only=False)
        validate_v5_payload(payload, feature_only=False)
        self.assertEqual(payload["targets"]["full_pose"].shape, (1, 8, 72))
        payload["targets"]["betas"] = torch.zeros(1, 7, 10)
        with self.assertRaisesRegex(ValueError, "Target betas"):
            validate_v5_payload(payload, feature_only=False)

    def test_target_context_alignment_is_enforced(self) -> None:
        payload = self.make_payload(feature_only=True)
        payload["metadata"][0]["target_frame_ids"][0] = 999
        with self.assertRaisesRegex(ValueError, "frame gap|positions 4..11"):
            validate_v5_payload(payload, feature_only=True)


if __name__ == "__main__":
    unittest.main()
