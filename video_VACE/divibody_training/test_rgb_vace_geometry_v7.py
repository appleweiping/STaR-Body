"""CPU tests for the additive high-resolution RGB + VACE v7 model."""

from __future__ import annotations

import unittest

import torch

from models.human_decoder_rgb_vace_v7 import (
    HumanRGBVaceGeometryTransformerV7,
    RGBVaceGeometryConfig,
)


def small_config() -> RGBVaceGeometryConfig:
    return RGBVaceGeometryConfig(
        rgb_input_dim=12,
        vace_input_dim=16,
        hidden_dim=32,
        num_heads=4,
        rgb_layers=1,
        vace_layers=2,
        dropout=0.0,
        output_frames=8,
        max_vace_frames=8,
        max_rgb_frames=8,
    )


def sample_inputs(config: RGBVaceGeometryConfig, batch: int = 2):
    torch.manual_seed(7101)
    rgb = torch.randn(batch, 8, 6, config.rgb_input_dim)
    vace = torch.randn(batch, 5, 2, 3, config.vace_input_dim)
    base_rotmat = torch.eye(3).reshape(1, 1, 1, 3, 3).repeat(batch, 8, 24, 1, 1)
    base_betas = torch.randn(batch, 8, 10)
    base_camera = torch.randn(batch, 8, 3)
    return rgb, vace, base_rotmat, base_betas, base_camera


class RGBVaceGeometryV7Tests(unittest.TestCase):
    def test_initial_model_exactly_preserves_hmr_anchor(self) -> None:
        config = small_config()
        model = HumanRGBVaceGeometryTransformerV7(config).eval()
        inputs = sample_inputs(config)
        with torch.no_grad():
            output = model(*inputs)
        self.assertTrue(torch.equal(output["rotmat"], inputs[2]))
        self.assertTrue(torch.equal(output["betas"], inputs[3]))
        self.assertTrue(torch.equal(output["camera"], inputs[4]))
        self.assertEqual(int(torch.count_nonzero(output["temporal_gate"])), 0)
        self.assertEqual(tuple(output["rgb_queries"].shape), (2, 8, 32))
        self.assertEqual(tuple(output["fused_queries"].shape), (2, 8, 32))

    def test_shared_base_shape_is_expanded_without_changing_values(self) -> None:
        config = small_config()
        model = HumanRGBVaceGeometryTransformerV7(config).eval()
        inputs = list(sample_inputs(config))
        inputs[3] = inputs[3][:, 0]
        with torch.no_grad():
            output = model(*inputs)
        expected = inputs[3][:, None].expand(-1, 8, -1)
        self.assertTrue(torch.equal(output["betas"], expected))

    def test_open_temporal_gate_makes_prediction_depend_on_vace(self) -> None:
        config = small_config()
        model = HumanRGBVaceGeometryTransformerV7(config).eval()
        inputs = list(sample_inputs(config))
        with torch.no_grad():
            model.temporal_gates.fill_(1.0)
            model.camera_residual_head.weight.normal_(std=0.05)
            output_a = model(*inputs)["camera"]
            # LayerNorm deliberately removes a uniform offset, so perturb the
            # token pattern rather than shifting every channel equally.
            inputs[1] = inputs[1] + 2.0 * torch.randn_like(inputs[1])
            output_b = model(*inputs)["camera"]
        self.assertGreater(float((output_a - output_b).abs().max()), 1.0e-6)

    def test_anchor_heads_and_temporal_gate_receive_gradients(self) -> None:
        config = small_config()
        model = HumanRGBVaceGeometryTransformerV7(config).train()
        inputs = list(sample_inputs(config))
        inputs[0].requires_grad_()
        inputs[1].requires_grad_()
        target_camera = torch.randn_like(inputs[4])
        output = model(*inputs)
        loss = (
            (output["camera"] - target_camera).square().mean()
            + (output["betas"] - 0.5).square().mean()
            + (output["rotmat"] - 0.25).square().mean()
        )
        loss.backward()
        self.assertIsNotNone(model.camera_residual_head.weight.grad)
        self.assertGreater(
            float(model.camera_residual_head.weight.grad.abs().sum()), 0.0
        )
        self.assertIsNotNone(model.temporal_gates.grad)
        self.assertTrue(torch.isfinite(model.temporal_gates.grad).all())

    def test_invalid_shapes_are_rejected(self) -> None:
        config = small_config()
        model = HumanRGBVaceGeometryTransformerV7(config)
        inputs = list(sample_inputs(config, batch=1))
        bad_calls = []
        value = list(inputs)
        value[0] = value[0][:, :7]
        bad_calls.append(("8 nonempty frames", value))
        value = list(inputs)
        value[1] = value[1][..., :15]
        bad_calls.append(("VACE token width", value))
        value = list(inputs)
        value[2] = value[2][:, :, :23]
        bad_calls.append(("base_rotmat", value))
        for message, values in bad_calls:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    model(*values)


if __name__ == "__main__":
    unittest.main()
