"""CPU unit tests for geometry-v5 training contracts."""

from __future__ import annotations

import unittest

import torch

from train_divibody_continuous_geometry_v5 import (
    boundary_acceleration_locations,
    masked_mean,
)


class ContinuousGeometryV5Tests(unittest.TestCase):
    def test_masked_mean_uses_only_unique_owners(self) -> None:
        values = torch.tensor(((1.0, 100.0), (3.0, 5.0)))
        mask = torch.tensor(((True, False), (True, False)))
        result = masked_mean(values, mask, "unit")
        self.assertEqual(float(result), 2.0)

    def test_masked_mean_rejects_invalid_contracts(self) -> None:
        values = torch.zeros(2, 3)
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            masked_mean(values, torch.zeros(2, 2, dtype=torch.bool), "unit")
        with self.assertRaisesRegex(TypeError, "boolean"):
            masked_mean(values, torch.zeros(2, 3), "unit")
        with self.assertRaisesRegex(ValueError, "no owned"):
            masked_mean(values, torch.zeros(2, 3, dtype=torch.bool), "unit")

    def test_boundary_triplet_resolves_unique_cross_window_owners(self) -> None:
        payload = {
            "metadata": [
                {"sequence": "s", "person_id": 0, "target_frame_ids": list(range(8))},
                {"sequence": "s", "person_id": 0, "target_frame_ids": list(range(8, 16))},
            ],
            "frame_owner_mask": torch.ones(2, 8, dtype=torch.bool),
            "metric_only_cross_window_motion": {
                "acceleration_triplets": [
                    {"sequence": "s", "person_id": 0, "frame_ids": [6, 7, 8]},
                    {"sequence": "s", "person_id": 0, "frame_ids": [7, 8, 9]},
                ]
            },
        }
        self.assertEqual(
            boundary_acceleration_locations(payload),
            [((0, 6), (0, 7), (1, 0)), ((0, 7), (1, 0), (1, 1))],
        )


if __name__ == "__main__":
    unittest.main()
