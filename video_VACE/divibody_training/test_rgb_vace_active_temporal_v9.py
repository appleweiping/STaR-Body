import torch

from models.human_decoder_rgb_vace_v9_active_temporal import (
    HumanRGBVaceGeometryTransformerV9ActiveTemporal,
    RGBVaceActiveTemporalConfig,
)


def test_v9_activates_vace_without_breaking_anchor_initialization():
    model = HumanRGBVaceGeometryTransformerV9ActiveTemporal().eval()
    assert torch.allclose(torch.tanh(model.temporal_gates), torch.full((2,), 0.15))
    assert not model.temporal_gates.requires_grad
    inputs = {
        "rgb_tokens": torch.randn(1, 8, 192, 1280),
        "vace_features": torch.randn(1, 5, 6, 6, 1536),
        "base_rotmat": torch.eye(3).reshape(1,1,1,3,3).repeat(1,8,24,1,1),
        "base_betas": torch.randn(1,8,10),
        "base_camera": torch.randn(1,8,3),
    }
    with torch.inference_mode(): output = model(**inputs)
    torch.testing.assert_close(output["rotmat"], inputs["base_rotmat"])
    torch.testing.assert_close(output["betas"], inputs["base_betas"])


def test_v9_rejects_unsafe_gate():
    try:
        HumanRGBVaceGeometryTransformerV9ActiveTemporal(
            RGBVaceActiveTemporalConfig(temporal_gate_init=0.8)
        )
    except ValueError as error:
        assert "temporal gate" in str(error)
    else:
        raise AssertionError("unsafe temporal gate accepted")
