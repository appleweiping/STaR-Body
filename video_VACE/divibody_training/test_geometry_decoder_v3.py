"""Model-only regressions for the additive geometry decoder.

Run from the repository root with:
    python -m unittest divibody_training.test_geometry_decoder_v3 -v
No VACE weights, SMPL assets, images, or GPU are needed for these tests.
"""

from __future__ import annotations

import dataclasses
import json
import math
import unittest

import torch

from models.human_decoder import HumanDecoderConfig, HumanQueryTransformer
from models.human_decoder_geometry import (
    GeometryDecoderConfig,
    HumanGeometryQueryTransformer,
    axis_angle_to_matrix,
    rotation_6d_to_matrix,
    spatial_sincos_position_encoding,
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


class RotationGeometryTests(unittest.TestCase):
    def test_six_dimensional_rotations_are_proper_with_finite_gradients(self) -> None:
        torch.manual_seed(2401)
        rotation_6d = torch.cat(
            (
                torch.randn(64, 6),
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 2.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 1.0, 1.0e-8, 0.0],
                    ]
                ),
            ),
            dim=0,
        ).requires_grad_()
        matrices = rotation_6d_to_matrix(rotation_6d)
        identity = torch.eye(3).expand_as(matrices)
        self.assertEqual(matrices.dtype, torch.float32)
        self.assertTrue(torch.isfinite(matrices).all())
        self.assertTrue(torch.allclose(
            matrices.transpose(-1, -2) @ matrices,
            identity,
            atol=1.0e-5,
            rtol=1.0e-5,
        ))
        self.assertTrue(torch.allclose(
            torch.linalg.det(matrices),
            torch.ones(matrices.shape[0]),
            atol=1.0e-5,
            rtol=1.0e-5,
        ))
        self.assertTrue(torch.equal(matrices[-4], torch.eye(3)))
        (matrices * torch.randn_like(matrices)).sum().backward()
        self.assertIsNotNone(rotation_6d.grad)
        self.assertTrue(torch.isfinite(rotation_6d.grad).all())
        self.assertGreater(float(rotation_6d.grad[:64].abs().sum()), 0.0)

    def test_axis_angle_identity_known_rotation_and_finite_gradients(self) -> None:
        vectors = torch.tensor(
            ((0.0, 0.0, 0.0), (0.0, 0.0, math.pi / 2.0),
             (1.0e-9, -2.0e-9, 3.0e-9)),
            requires_grad=True,
        )
        matrices = axis_angle_to_matrix(vectors)
        expected_z = torch.tensor(
            ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        )
        self.assertTrue(torch.equal(matrices[0], torch.eye(3)))
        self.assertTrue(torch.allclose(matrices[1], expected_z, atol=1.0e-6))
        matrices[..., 1, 0].sum().backward()
        self.assertIsNotNone(vectors.grad)
        self.assertTrue(torch.isfinite(vectors.grad).all())
        self.assertAlmostEqual(float(vectors.grad[0, 2]), 1.0, places=6)

    def test_rotation_helpers_disable_autocast_and_return_fp32(self) -> None:
        vectors = torch.tensor(
            ((0.1, 0.2, -0.3), (0.0, 0.0, 0.0)), dtype=torch.bfloat16
        ).requires_grad_()
        rotation_6d = torch.tensor(
            ((1.0, 0.2, -0.1, 0.1, 1.0, 0.2),), dtype=torch.bfloat16
        ).requires_grad_()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            aa_matrices = axis_angle_to_matrix(vectors)
            residual_matrices = rotation_6d_to_matrix(rotation_6d)
        for matrices in (aa_matrices, residual_matrices):
            self.assertEqual(matrices.dtype, torch.float32)
            self.assertTrue(torch.allclose(
                matrices.transpose(-1, -2) @ matrices,
                torch.eye(3).expand_as(matrices),
                atol=1.0e-5,
                rtol=1.0e-5,
            ))
        loss = aa_matrices[..., 1, 0].sum() + residual_matrices[..., 1, 0].sum()
        loss.backward()
        self.assertTrue(torch.isfinite(vectors.grad).all())
        self.assertTrue(torch.isfinite(rotation_6d.grad).all())


class GeometryDecoderTests(unittest.TestCase):
    def test_identity_residual_matches_original_and_does_not_mutate_source(self) -> None:
        torch.manual_seed(2402)
        config = small_config()
        original = HumanQueryTransformer(
            HumanDecoderConfig(**dataclasses.asdict(config))
        ).eval()
        source_state = original.state_dict()
        source_snapshot = {key: value.clone() for key, value in source_state.items()}
        model = HumanGeometryQueryTransformer(config).eval()
        report = model.load_camera_v2_weights(source_state)
        self.assertEqual(set(report["loaded"]), set(source_state))
        self.assertEqual(set(report["missing"]), {
            "spatial_position_scale",
            "rotation_residual_head.weight",
            "rotation_residual_head.bias",
        })
        json.dumps(report)
        self.assertEqual(int(torch.count_nonzero(model.rotation_residual_head.weight)), 0)
        self.assertTrue(torch.equal(
            model.rotation_residual_head.bias,
            torch.tensor((1.0, 0.0, 0.0, 0.0, 1.0, 0.0)).repeat(24),
        ))
        with torch.no_grad():
            model.spatial_position_scale.zero_()
            features = torch.randn(2, 3 * 2 * 2, config.input_dim)
            expected = original(features, (3, 2, 2), target_frames=5)
            actual = model(features, (3, 2, 2), target_frames=5)
        for key, expected_tensor in expected.items():
            self.assertTrue(torch.allclose(
                actual[key], expected_tensor, atol=1.0e-6, rtol=1.0e-5
            ), key)
        base_pose = torch.cat(
            (expected["global_orient"], expected["body_pose"],
             torch.zeros(2, 5, 6)), dim=-1
        ).reshape(2, 5, 24, 3)
        self.assertTrue(torch.allclose(
            actual["rotmat"], axis_angle_to_matrix(base_pose),
            atol=1.0e-6, rtol=1.0e-5,
        ))
        self.assertTrue(torch.equal(
            actual["rotmat"][..., 22:, :, :],
            torch.eye(3).expand(2, 5, 2, 3, 3),
        ))
        # A subsequent change to the new model must not alias the source.
        with torch.no_grad():
            model.body_head.weight.add_(1.0)
        for key, original_value in source_snapshot.items():
            self.assertTrue(torch.equal(source_state[key], original_value), key)
            self.assertTrue(torch.equal(original.state_dict()[key], original_value), key)

    def test_spatial_positions_break_within_frame_token_permutation_invariance(self) -> None:
        torch.manual_seed(2403)
        config = small_config()
        model = HumanGeometryQueryTransformer(config).eval()
        features = torch.randn(1, 2, 9, config.input_dim)
        permutation = torch.tensor((8, 2, 5, 0, 7, 1, 3, 6, 4))
        original = features.reshape(1, 18, config.input_dim)
        shuffled = features[:, :, permutation].reshape(1, 18, config.input_dim)
        with torch.no_grad():
            model.spatial_position_scale.zero_()
            without_position = model(original, (2, 3, 3))["rotmat"]
            without_position_shuffled = model(shuffled, (2, 3, 3))["rotmat"]
            self.assertTrue(torch.allclose(
                without_position, without_position_shuffled,
                atol=1.0e-6, rtol=1.0e-5,
            ))
            model.spatial_position_scale.fill_(0.1)
            with_position = model(original, (2, 3, 3))["rotmat"]
            with_position_shuffled = model(shuffled, (2, 3, 3))["rotmat"]
        self.assertGreater(
            float((with_position - with_position_shuffled).abs().max()), 1.0e-5
        )

    def test_shapes_and_both_previously_missing_joints_have_learnable_gradients(self) -> None:
        torch.manual_seed(2404)
        config = small_config()
        model = HumanGeometryQueryTransformer(config)
        features = torch.randn(2, 2 * 2 * 2, config.input_dim)
        output = model(features, (2, 2, 2), target_frames=5)
        self.assertEqual(output["rotmat"].shape, (2, 5, 24, 3, 3))
        self.assertEqual(output["rotmat"].dtype, torch.float32)
        self.assertEqual(output["betas"].shape, (2, 10))
        self.assertEqual(output["body_pose"].shape, (2, 5, 63))
        self.assertEqual(output["global_orient"].shape, (2, 5, 3))
        self.assertEqual(output["latent_queries"].shape, (2, 2, 5, 32))
        target_rotation = axis_angle_to_matrix(torch.tensor((0.0, 0.0, 0.25)))
        loss = (output["rotmat"][..., 22:, :, :] - target_rotation).square().mean()
        loss.backward()
        weight_gradient = model.rotation_residual_head.weight.grad.reshape(24, 6, -1)
        bias_gradient = model.rotation_residual_head.bias.grad.reshape(24, 6)
        for joint_index in (22, 23):
            self.assertGreater(float(weight_gradient[joint_index].abs().sum()), 0.0)
            self.assertGreater(float(bias_gradient[joint_index].abs().sum()), 0.0)
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_forward_rotations_remain_fp32_under_autocast(self) -> None:
        torch.manual_seed(2405)
        model = HumanGeometryQueryTransformer(small_config()).train()
        features = torch.randn(1, 8, 16)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            rotations = model(features, (2, 2, 2))["rotmat"]
        self.assertEqual(rotations.dtype, torch.float32)
        self.assertTrue(torch.allclose(
            rotations.transpose(-1, -2) @ rotations,
            torch.eye(3).expand_as(rotations),
            atol=1.0e-5, rtol=1.0e-5,
        ))
        rotations[..., 1, 0].sum().backward()
        self.assertTrue(torch.isfinite(model.rotation_residual_head.weight.grad).all())

    def test_legacy_loader_rejects_other_key_and_shape_mismatches_atomically(self) -> None:
        torch.manual_seed(2406)
        config = small_config()
        original = HumanQueryTransformer(HumanDecoderConfig(**dataclasses.asdict(config)))
        source = original.state_dict()
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
                model = HumanGeometryQueryTransformer(config)
                before = {key: value.clone() for key, value in model.state_dict().items()}
                with self.assertRaisesRegex(RuntimeError, error):
                    model.load_camera_v2_weights(invalid_state)
                for key, value in before.items():
                    self.assertTrue(torch.equal(model.state_dict()[key], value), key)

    def test_spatial_encoding_supports_small_and_odd_hidden_dimensions(self) -> None:
        for hidden_dim in (1, 2, 3, 4, 5, 7, 8, 31, 32):
            with self.subTest(hidden_dim=hidden_dim):
                positions = spatial_sincos_position_encoding(
                    2, 3, hidden_dim, torch.device("cpu")
                )
                self.assertEqual(positions.shape, (6, hidden_dim))
                self.assertTrue(torch.isfinite(positions).all())
                self.assertFalse(torch.equal(positions[0], positions[1]))
                self.assertFalse(torch.equal(positions[0], positions[3]))


if __name__ == "__main__":
    unittest.main()
