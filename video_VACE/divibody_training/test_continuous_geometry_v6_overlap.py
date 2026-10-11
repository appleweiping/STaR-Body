"""CPU tests for stride-4 overlap fusion geometry."""

from __future__ import annotations

import unittest

import torch

from evaluate_continuous_geometry_v6_overlap import project_so3


class OverlapFusionTests(unittest.TestCase):
    def test_identity_is_unchanged(self) -> None:
        identity = torch.eye(3).reshape(1, 1, 3, 3)
        self.assertTrue(torch.allclose(project_so3(identity), identity))

    def test_projection_is_a_proper_rotation(self) -> None:
        matrices = torch.tensor(
            [
                [[1.0, 0.2, 0.0], [0.0, 0.8, -0.1], [0.0, 0.0, 1.1]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
            ]
        )
        rotations = project_so3(matrices)
        expected = torch.eye(3).expand(2, 3, 3)
        self.assertTrue(
            torch.allclose(rotations.transpose(-1, -2) @ rotations, expected, atol=1e-5)
        )
        self.assertTrue(torch.allclose(torch.linalg.det(rotations), torch.ones(2), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
