"""CPU contract tests for the additive continuous-video VACE encoder."""

from __future__ import annotations

import unittest

import torch

from real_vace_temporal_feature_encoder import (
    reshape_temporal_tokens,
    wan_latent_frames,
)


class TemporalEncoderContractTests(unittest.TestCase):
    def test_wan_latent_frames_accepts_complete_groups(self) -> None:
        for frames, latent in ((1, 1), (5, 2), (9, 3), (17, 5)):
            with self.subTest(frames=frames):
                self.assertEqual(wan_latent_frames(frames), latent)

    def test_wan_latent_frames_rejects_silent_tail_loss(self) -> None:
        for frames in (0, 2, 4, 8, 10, 16):
            with self.subTest(frames=frames):
                with self.assertRaisesRegex(ValueError, r"4n\+1"):
                    wan_latent_frames(frames)

    def test_temporal_token_reshape_preserves_fhw_order(self) -> None:
        tokens = torch.arange(2 * 12 * 5).reshape(2, 12, 5)
        grid = reshape_temporal_tokens(tokens, (3, 2, 2))
        self.assertEqual(grid.shape, (2, 3, 2, 2, 5))
        self.assertTrue(torch.equal(grid.reshape(2, 12, 5), tokens))
        self.assertTrue(torch.equal(grid[0, 1, 0, 0], tokens[0, 4]))

    def test_temporal_token_reshape_rejects_misalignment(self) -> None:
        with self.assertRaisesRegex(ValueError, r"F\*H\*W"):
            reshape_temporal_tokens(torch.zeros(1, 11, 4), (3, 2, 2))


if __name__ == "__main__":
    unittest.main()
