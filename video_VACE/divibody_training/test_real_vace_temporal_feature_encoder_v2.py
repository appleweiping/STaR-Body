"""CPU contract tests for protocol-aware continuous VACE features."""

from __future__ import annotations

import unittest

import torch

from real_vace_temporal_feature_encoder_v2 import (
    ProtocolAwareTemporalVaceFeatureEncoder,
    validate_noise_seeds,
)


class ProtocolAwareEncoderTests(unittest.TestCase):
    def test_noise_seeds_are_per_clip_and_strict(self) -> None:
        self.assertEqual(validate_noise_seeds([7, 9], 2, True), (7, 9))
        self.assertIsNone(validate_noise_seeds(None, 3, False))
        with self.assertRaisesRegex(ValueError, "one deterministic"):
            validate_noise_seeds(None, 1, True)
        with self.assertRaisesRegex(ValueError, "Expected 2"):
            validate_noise_seeds([1], 2, True)
        with self.assertRaisesRegex(ValueError, "seeds must lie"):
            validate_noise_seeds([-1], 1, True)

    def test_deterministic_noise_is_batch_composition_independent(self) -> None:
        reference = torch.zeros(2, 3, 2, 2)
        together = ProtocolAwareTemporalVaceFeatureEncoder._deterministic_noise(
            reference, (100, 200)
        )
        first = ProtocolAwareTemporalVaceFeatureEncoder._deterministic_noise(
            reference[:1], (100,)
        )
        second = ProtocolAwareTemporalVaceFeatureEncoder._deterministic_noise(
            reference[:1], (200,)
        )
        self.assertTrue(torch.equal(together[0], first[0]))
        self.assertTrue(torch.equal(together[1], second[0]))
        self.assertFalse(torch.equal(together[0], together[1]))


if __name__ == "__main__":
    unittest.main()
