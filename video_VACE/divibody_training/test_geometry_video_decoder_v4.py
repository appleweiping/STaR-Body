"""CPU regressions for the additive continuous-video geometry-v4 decoder.

Run from the repository root with:
    python -m unittest divibody_training.test_geometry_video_decoder_v4 -v
No VACE weights, SMPL assets, images, or GPU are required.
"""

from __future__ import annotations

import unittest

import torch

from models.human_decoder_geometry import (
    GeometryDecoderConfig,
    HumanGeometryQueryTransformer,
)
from models.human_decoder_geometry_video_v4 import (
    CENTER_RGB_POSITIONS,
    HumanGeometryVideoQueryTransformer,
    rgb_frame_group_offset,
)


def small_config() -> GeometryDecoderConfig:
    return GeometryDecoderConfig(
        input_dim=16,
        hidden_dim=32,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
        max_latent_frames=8,
    )


class GeometryVideoDecoderV4Tests(unittest.TestCase):
    def test_wan_rgb_group_and_offset_mapping(self) -> None:
        positions = torch.arange(17)
        groups, offsets = rgb_frame_group_offset(positions)
        self.assertTrue(torch.equal(
            groups,
            torch.tensor((0, 1, 1, 1, 1, 2, 2, 2, 2,
                          3, 3, 3, 3, 4, 4, 4, 4)),
        ))
        self.assertTrue(torch.equal(
            offsets,
            torch.tensor((0, 0, 1, 2, 3, 0, 1, 2, 3,
                          0, 1, 2, 3, 0, 1, 2, 3)),
        ))
        center_groups, center_offsets = rgb_frame_group_offset(
            torch.tensor(CENTER_RGB_POSITIONS)
        )
        self.assertTrue(torch.equal(
            center_groups, torch.tensor((1, 2, 2, 2, 2, 3, 3, 3))
        ))
        self.assertTrue(torch.equal(
            center_offsets, torch.tensor((3, 0, 1, 2, 3, 0, 1, 2))
        ))

    def test_t17_forward_directly_emits_center_eight_shapes(self) -> None:
        torch.manual_seed(4101)
        config = small_config()
        model = HumanGeometryVideoQueryTransformer(config).eval()
        features = torch.randn(2, 5 * 2 * 3, config.input_dim)
        with torch.no_grad():
            output = model(
                features,
                (5, 2, 3),
                context_rgb_frames=17,
                target_positions=CENTER_RGB_POSITIONS,
            )
        expected = {
            "rotmat": (2, 8, 24, 3, 3),
            "global_orient": (2, 8, 3),
            "body_pose": (2, 8, 63),
            "transl": (2, 8, 3),
            "left_hand_pose": (2, 8, 45),
            "right_hand_pose": (2, 8, 45),
            "jaw_pose": (2, 8, 3),
            "expression": (2, 8, 10),
            "object_state": (2, 8, 9),
            "betas": (2, 10),
            "latent_queries": (2, 8, 5, 32),
        }
        for key, shape in expected.items():
            self.assertEqual(tuple(output[key].shape), shape, key)
            self.assertTrue(torch.isfinite(output[key]).all(), key)
        self.assertEqual(output["rotmat"].dtype, torch.float32)
        identity = torch.eye(3).expand_as(output["rotmat"])
        self.assertTrue(torch.allclose(
            output["rotmat"].transpose(-1, -2) @ output["rotmat"],
            identity,
            atol=1.0e-5,
            rtol=1.0e-5,
        ))

    def test_forward_rejects_invalid_context_and_target_positions(self) -> None:
        model = HumanGeometryVideoQueryTransformer(small_config())
        features = torch.randn(1, 5 * 2 * 2, 16)
        invalid_calls = (
            (
                "4\\*latent_frames-3",
                dict(context_rgb_frames=16),
            ),
            (
                "exactly 8",
                dict(context_rgb_frames=17, target_positions=range(4, 11)),
            ),
            (
                "strictly consecutive",
                dict(
                    context_rgb_frames=17,
                    target_positions=(4, 5, 6, 8, 9, 10, 11, 12),
                ),
            ),
            (
                "inside the RGB context",
                dict(context_rgb_frames=17, target_positions=range(10, 18)),
            ),
        )
        for message, kwargs in invalid_calls:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    model(features, (5, 2, 2), **kwargs)
        with self.assertRaisesRegex(TypeError, "integer dtype"):
            model(
                features,
                (5, 2, 2),
                context_rgb_frames=17,
                target_positions=torch.arange(8, dtype=torch.float32),
            )
    def test_geometry_v3_warm_start_is_strict_and_source_is_unchanged(self) -> None:
        torch.manual_seed(4102)
        config = small_config()
        geometry_v3 = HumanGeometryQueryTransformer(config)
        source = geometry_v3.state_dict()
        snapshot = {key: value.clone() for key, value in source.items()}
        model = HumanGeometryVideoQueryTransformer(config)
        report = model.load_geometry_v3_weights(source)

        new_keys = {
            "rgb_group_embedding.weight",
            "rgb_offset_embedding.weight",
            "rgb_context_position_embedding.weight",
        }
        self.assertEqual(set(report["loaded"]), set(source))
        self.assertEqual(set(report["missing"]), new_keys)
        for key, value in source.items():
            self.assertTrue(torch.equal(model.state_dict()[key], value), key)
        for key in new_keys:
            self.assertEqual(int(torch.count_nonzero(model.state_dict()[key])), 0)

        with torch.no_grad():
            model.body_head.weight.add_(1.0)
            model.rgb_group_embedding.weight.add_(1.0)
        for key, original_value in snapshot.items():
            self.assertTrue(torch.equal(source[key], original_value), key)
            self.assertTrue(
                torch.equal(geometry_v3.state_dict()[key], original_value), key
            )

    def test_geometry_v3_loader_rejects_mismatches_atomically(self) -> None:
        config = small_config()
        source = HumanGeometryQueryTransformer(config).state_dict()
        invalid_states = []
        missing = dict(source)
        del missing["body_head.weight"]
        invalid_states.append(("missing", missing))
        unexpected = dict(source)
        unexpected["unexpected_parameter"] = torch.zeros(1)
        invalid_states.append(("unexpected", unexpected))
        mismatched = dict(source)
        mismatched["body_head.weight"] = source["body_head.weight"][:1]
        invalid_states.append(("mismatched", mismatched))

        for error, invalid_state in invalid_states:
            with self.subTest(error=error):
                model = HumanGeometryVideoQueryTransformer(config)
                before = {
                    key: value.clone()
                    for key, value in model.state_dict().items()
                }
                with self.assertRaisesRegex(RuntimeError, error):
                    model.load_geometry_v3_weights(invalid_state)
                for key, value in before.items():
                    self.assertTrue(
                        torch.equal(model.state_dict()[key], value), key
                    )

    def test_new_queries_and_inputs_receive_finite_gradients(self) -> None:
        torch.manual_seed(4103)
        config = small_config()
        model = HumanGeometryVideoQueryTransformer(config).train()
        features = torch.randn(
            2, 5 * 2 * 2, config.input_dim, requires_grad=True
        )
        output = model(
            features,
            (5, 2, 2),
            context_rgb_frames=17,
            target_positions=CENTER_RGB_POSITIONS,
        )
        rotation_target = torch.randn_like(output["rotmat"])
        loss = (
            (output["rotmat"] - rotation_target).square().mean()
            + output["body_pose"].square().mean()
            + output["transl"].square().mean()
            + output["object_state"].square().mean()
            + output["betas"].square().mean()
        )
        loss.backward()

        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(float(features.grad.abs().sum()), 0.0)
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())
        for embedding in (
            model.rgb_group_embedding,
            model.rgb_offset_embedding,
            model.rgb_context_position_embedding,
        ):
            self.assertIsNotNone(embedding.weight.grad)
            self.assertTrue(torch.isfinite(embedding.weight.grad).all())
            self.assertGreater(float(embedding.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
