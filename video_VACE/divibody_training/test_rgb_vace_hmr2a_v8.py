"""Unit checks for the additive HMR 2.0a v8 model contract."""

from __future__ import annotations

import torch

from models.human_decoder_rgb_vace_v8_hmr2a import (
    HMR2A_SHA256,
    HumanRGBVaceGeometryTransformerV8HMR2A,
)


def test_v8_starts_as_exact_hmr2a_anchor() -> None:
    model = HumanRGBVaceGeometryTransformerV8HMR2A().eval()
    batch, frames = 1, 8
    inputs = {
        "rgb_tokens": torch.randn(batch, frames, 192, 1280),
        "vace_features": torch.randn(batch, 5, 6, 6, 1536),
        "base_rotmat": torch.eye(3).reshape(1, 1, 1, 3, 3).repeat(
            batch, frames, 24, 1, 1
        ),
        "base_betas": torch.randn(batch, frames, 10),
        "base_camera": torch.randn(batch, frames, 3),
    }
    with torch.inference_mode():
        output = model(**inputs)
    torch.testing.assert_close(output["rotmat"], inputs["base_rotmat"])
    torch.testing.assert_close(output["betas"], inputs["base_betas"])
    torch.testing.assert_close(output["camera"], inputs["base_camera"])
    torch.testing.assert_close(output["temporal_gate"], torch.zeros(2))


def test_v8_records_exact_anchor_provenance() -> None:
    metadata = HumanRGBVaceGeometryTransformerV8HMR2A().anchor_metadata()
    assert metadata["model_format"] == "divibody-rgb-vace-v8-hmr2a"
    assert metadata["anchor_checkpoint_sha256"] == HMR2A_SHA256
    assert metadata["anchor_preserving_initialization"] is True
