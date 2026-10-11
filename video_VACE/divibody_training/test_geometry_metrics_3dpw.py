"""Small CPU tests for the standalone 3DPW geometry metric implementation."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from geometry_metrics_3dpw import (
    batch_similarity_transform,
    hip_centered_j14,
    sequence_acceleration_error,
)


class GeometryMetricTests(unittest.TestCase):
    def test_vibe_hip_midpoint_root(self) -> None:
        joints = torch.zeros(2, 14, 3, dtype=torch.float64)
        joints[:, 2, 0] = 2.0
        joints[:, 3, 0] = 4.0
        joints[1] += torch.tensor([7.0, -2.0, 5.0], dtype=torch.float64)
        centered = hip_centered_j14(joints)
        hip_midpoint = centered[:, [2, 3], :].mean(dim=1)
        self.assertTrue(torch.allclose(hip_midpoint, torch.zeros_like(hip_midpoint)))
        self.assertTrue(torch.allclose(centered[0], centered[1]))

    def test_similarity_transform_recovers_scale_rotation_translation(self) -> None:
        torch.manual_seed(7)
        prediction = torch.randn(4, 14, 3, dtype=torch.float64)
        angle = torch.tensor(0.73, dtype=torch.float64)
        cosine, sine = torch.cos(angle), torch.sin(angle)
        rotation = torch.tensor(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )
        target = 1.7 * (prediction @ rotation.T) + torch.tensor(
            [2.0, -1.0, 0.25], dtype=torch.float64
        )
        aligned = batch_similarity_transform(prediction, target)
        self.assertTrue(torch.allclose(aligned, target, atol=1e-9, rtol=1e-9))

    def test_acceleration_never_crosses_track_or_frame_gap(self) -> None:
        target = np.zeros((8, 14, 3), dtype=np.float64)
        prediction = target.copy()
        prediction[1, :, 0] = 1.0
        result = sequence_acceleration_error(
            prediction,
            target,
            sequences=("a", "a", "a", "a", "b", "b", "b", "b"),
            person_ids=(0, 0, 0, 0, 0, 0, 0, 0),
            frame_ids=(0, 1, 2, 4, 0, 1, 3, 4),
        )
        # Track a contributes only frames (0,1,2); track b has no consecutive
        # triplet.  In particular there is no window over a's 2->4 gap or the
        # a->b sequence boundary.
        self.assertEqual(result.errors.shape, (1,))
        self.assertEqual(result.center_sample_indices.tolist(), [1])
        self.assertAlmostEqual(float(result.errors[0]), 2.0)

    def test_zero_error_for_identical_tracks(self) -> None:
        rng = np.random.default_rng(11)
        joints = rng.normal(size=(6, 14, 3))
        result = sequence_acceleration_error(
            joints,
            joints.copy(),
            sequences=("seq",) * 6,
            person_ids=(2,) * 6,
            frame_ids=range(6),
        )
        self.assertEqual(len(result.errors), 4)
        np.testing.assert_allclose(result.errors, 0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
