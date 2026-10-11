from __future__ import annotations

import argparse
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from evaluate_temporal_3dpw import (
    CenteredFrameClipDataset,
    validate_feature_compatibility,
    validate_geometry_inputs,
)
from train_divibody_real_3dpw_temporal import (
    TemporalClipDataset,
    axis_angle_to_matrix,
    temporal_losses,
)


class TemporalTrainingTests(unittest.TestCase):
    def test_axis_angle_known_rotation(self) -> None:
        vectors = torch.tensor(((0.0, 0.0, 0.0), (0.0, 0.0, math.pi / 2)))
        matrices = axis_angle_to_matrix(vectors)
        expected = torch.tensor(
            (
                ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            )
        )
        self.assertTrue(torch.allclose(matrices, expected, atol=1.0e-6))

    def test_clips_never_cross_tracks_or_gaps(self) -> None:
        metadata = [
            {"sequence": "a", "person_id": 0, "frame_id": frame}
            for frame in (0, 1, 2, 4, 5, 6)
        ] + [
            {"sequence": "a", "person_id": 1, "frame_id": frame}
            for frame in (0, 1, 2, 3)
        ]
        count = len(metadata)
        payload = {
            "signature": {
                "format": "divibody-frozen-vace-features-v1",
                "split": "train",
            },
            "features": torch.zeros(count, 4, 1536, dtype=torch.bfloat16),
            "targets": {
                "global_orient": torch.zeros(count, 3),
                "body_pose": torch.zeros(count, 63),
                "betas": torch.zeros(count, 10),
                "transl": torch.zeros(count, 3),
            },
            "metadata": metadata,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.pt"
            torch.save(payload, path)
            dataset = TemporalClipDataset(path, "train", 3, 2)
        self.assertEqual(dataset.stats["tracks"], 2)
        self.assertEqual(dataset.stats["frame_gaps"], 1)
        for indices in dataset.clips:
            records = [metadata[int(index)] for index in indices]
            identities = {(item["sequence"], item["person_id"]) for item in records}
            frames = [item["frame_id"] for item in records]
            self.assertEqual(len(identities), 1)
            self.assertEqual(frames, list(range(frames[0], frames[0] + 3)))

    def test_identical_prediction_has_zero_primary_losses(self) -> None:
        batch_size, frames = 2, 4
        global_orient = torch.zeros(batch_size, frames, 3)
        body_pose = torch.zeros(batch_size, frames, 63)
        betas = torch.zeros(batch_size, 10)
        transl = torch.zeros(batch_size, frames, 3)
        prediction = {
            "global_orient": global_orient.clone(),
            "body_pose": body_pose.clone(),
            "betas": betas.clone(),
            "transl": transl.clone(),
            "left_hand_pose": torch.zeros(batch_size, frames, 45),
            "right_hand_pose": torch.zeros(batch_size, frames, 45),
            "jaw_pose": torch.zeros(batch_size, frames, 3),
            "expression": torch.zeros(batch_size, frames, 10),
            "object_state": torch.zeros(batch_size, frames, 9),
        }
        batch = {
            "global_orient": global_orient,
            "body_pose": body_pose,
            "betas": betas,
            "transl": transl,
        }
        args = argparse.Namespace(
            rotation_weight=1.0,
            axis_angle_weight=0.25,
            shape_weight=0.1,
            translation_weight=0.0,
            velocity_weight=0.5,
            acceleration_weight=0.25,
            unsupported_head_reg=1.0e-4,
        )
        losses = temporal_losses(prediction, batch, args, torch.device("cpu"))
        for name, value in losses.items():
            self.assertEqual(float(value), 0.0, name)


class TemporalEvaluationStitchTests(unittest.TestCase):
    @staticmethod
    def _payload() -> dict:
        metadata = (
            [
                {"sequence": "seq_a", "person_id": 0, "frame_id": frame}
                for frame in (0, 1, 2, 4, 5)
            ]
            + [
                {"sequence": "seq_a", "person_id": 1, "frame_id": frame}
                for frame in (0, 1, 2, 3, 4, 5)
            ]
            + [{"sequence": "seq_b", "person_id": 0, "frame_id": 10}]
        )
        return {
            "features": torch.zeros(len(metadata), 4, 1536, dtype=torch.bfloat16),
            "metadata": metadata,
        }

    def test_centered_stitch_covers_each_frame_exactly_once(self) -> None:
        payload = self._payload()
        dataset = CenteredFrameClipDataset(payload, clip_length=4)

        target_indices = [target for _, target, _ in dataset.requests]
        self.assertEqual(len(dataset), len(payload["metadata"]))
        self.assertEqual(sorted(target_indices), list(range(len(dataset))))
        self.assertEqual(len(target_indices), len(set(target_indices)))
        for request_index, (indices, target_index, local_position) in enumerate(
            dataset.requests
        ):
            sample = dataset[request_index]
            self.assertEqual(int(indices[local_position]), target_index)
            self.assertEqual(int(sample["target_index"]), target_index)
            self.assertEqual(int(sample["local_position"]), local_position)

    def test_centered_context_never_crosses_track_or_frame_gap(self) -> None:
        payload = self._payload()
        metadata = payload["metadata"]
        dataset = CenteredFrameClipDataset(payload, clip_length=4)

        for indices, target_index, _ in dataset.requests:
            records = [metadata[int(index)] for index in indices]
            target = metadata[target_index]
            identities = {
                (record["sequence"], record["person_id"]) for record in records
            }
            frames = [record["frame_id"] for record in records]
            self.assertEqual(identities, {(target["sequence"], target["person_id"])})
            self.assertTrue(
                all(right - left in (0, 1) for left, right in zip(frames, frames[1:]))
            )

    def test_short_run_edge_repeat_stays_inside_target_run(self) -> None:
        payload = self._payload()
        metadata = payload["metadata"]
        dataset = CenteredFrameClipDataset(payload, clip_length=4)

        expected_run_frames = {
            ("seq_a", 0, 0): {0, 1, 2},
            ("seq_a", 0, 1): {4, 5},
            ("seq_b", 0, 0): {10},
        }
        for indices, target_index, _ in dataset.requests:
            target = metadata[target_index]
            if target["sequence"] == "seq_a" and target["person_id"] == 1:
                continue
            run_id = (
                1
                if target["sequence"] == "seq_a"
                and target["person_id"] == 0
                and target["frame_id"] >= 4
                else 0
            )
            allowed_frames = expected_run_frames[
                (target["sequence"], target["person_id"], run_id)
            ]
            clip_frames = [metadata[int(index)]["frame_id"] for index in indices]
            self.assertEqual(len(clip_frames), 4)
            self.assertTrue(set(clip_frames).issubset(allowed_frames))
            self.assertEqual(clip_frames, sorted(clip_frames))
            for frame in set(clip_frames):
                if clip_frames.count(frame) > 1:
                    self.assertIn(frame, (min(allowed_frames), max(allowed_frames)))


class TemporalEvaluationValidationTests(unittest.TestCase):
    @staticmethod
    def _compatible_feature_inputs() -> tuple[dict, dict]:
        signature = {
            "image_size": 512,
            "feature_layers": [4, 9, 14, 19],
            "vace_timestep": 0,
            "feature_backend": "official frozen Wan2.1-VACE",
        }
        checkpoint = {
            "train_cache_signature": {
                **signature,
                "feature_layers": list(signature["feature_layers"]),
            }
        }
        payload = {
            "signature": {
                **signature,
                "feature_layers": list(signature["feature_layers"]),
            }
        }
        return checkpoint, payload

    def test_matching_feature_configuration_is_accepted(self) -> None:
        checkpoint, payload = self._compatible_feature_inputs()

        validate_feature_compatibility(checkpoint, payload)

    def test_each_feature_configuration_mismatch_is_rejected(self) -> None:
        mismatches = {
            "image_size": 256,
            "feature_layers": [4, 9, 14],
            "vace_timestep": 1,
            "feature_backend": "dinov2",
        }
        for key, incompatible_value in mismatches.items():
            with self.subTest(key=key):
                checkpoint, payload = self._compatible_feature_inputs()
                payload["signature"][key] = incompatible_value

                with self.assertRaisesRegex(ValueError, key):
                    validate_feature_compatibility(checkpoint, payload)

    def test_invalid_gender_code_is_rejected(self) -> None:
        sample_count = 2
        pred_pose = np.zeros((sample_count, 72), dtype=np.float32)
        pred_betas = np.zeros((sample_count, 10), dtype=np.float32)
        gt_pose = np.zeros((sample_count, 72), dtype=np.float32)
        gt_betas = np.zeros((sample_count, 10), dtype=np.float32)
        camera_rotations = np.repeat(
            np.eye(3, dtype=np.float32)[None], sample_count, axis=0
        )

        for invalid_code in (-1, 3):
            with self.subTest(gender_code=invalid_code):
                gender_codes = np.array([0, invalid_code], dtype=np.int64)
                with self.assertRaisesRegex(ValueError, "Invalid gender codes"):
                    validate_geometry_inputs(
                        pred_pose,
                        pred_betas,
                        gt_pose,
                        gt_betas,
                        gender_codes,
                        camera_rotations,
                    )


if __name__ == "__main__":
    unittest.main()
