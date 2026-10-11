"""License-free gradient/unit tests for the additive geometry-v3 trainer."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from train_divibody_real_3dpw_geometry import (
    FullGeometryTargets,
    gendered_vertices,
    geometric_losses,
)
from train_divibody_real_3dpw_temporal import axis_angle_to_matrix


class FakeSMPL(torch.nn.Module):
    """Small deterministic differentiable stand-in; not a benchmark body model."""

    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(19)
        self.register_buffer("template", torch.randn(6890, 3, generator=generator) * 0.2)
        self.register_buffer("joint_ids", torch.arange(6890) % 24)
        self.register_buffer("shape_basis", torch.randn(10, 6890, 3, generator=generator) * 0.01)

    def forward(self, global_orient, body_pose, betas, pose2rot=False):
        if pose2rot:
            raise AssertionError("Geometry trainer must use rotation matrices")
        matrices = torch.cat((global_orient, body_pose), dim=1)[:, self.joint_ids]
        shaped = self.template[None] + torch.einsum("bk,kvc->bvc", betas, self.shape_basis)
        vertices = torch.einsum("bvij,bvj->bvi", matrices, shaped)
        return SimpleNamespace(vertices=vertices)


class GeometryTrainingV3Test(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.device = torch.device("cpu")
        model = FakeSMPL()
        self.models = {0: model, 1: model, 2: model}
        self.regressor = torch.zeros(17, 6890)
        self.regressor[torch.arange(17), torch.arange(17) * 17] = 1.0
        self.args = SimpleNamespace(joint_weight=1.0, vertex_weight=0.5,
                                    rotation_weight=0.05, shape_weight=0.001,
                                    velocity_weight=0.5, acceleration_weight=0.25)
        self.angles = torch.randn(2, 4, 24, 3) * 0.15
        self.betas = torch.randn(2, 10) * 0.1
        self.batch = {
            "rotmat": axis_angle_to_matrix(self.angles),
            "betas": self.betas[:, None].expand(-1, 4, -1),
            "genders": torch.tensor([[0] * 4, [1] * 4]),
        }

    def test_perfect_geometry_has_zero_loss(self) -> None:
        pred = {"rotmat": self.batch["rotmat"].clone(), "betas": self.betas.clone()}
        losses = geometric_losses(pred, self.batch, self.models, self.regressor,
                                  self.args, self.device)
        for name, value in losses.items():
            self.assertLess(abs(float(value)), 1e-7, name)

    def test_pose_and_shape_receive_finite_gradients(self) -> None:
        angles = (self.angles + torch.randn_like(self.angles) * 0.04).requires_grad_()
        betas = (self.betas + 0.1).requires_grad_()
        pred = {"rotmat": axis_angle_to_matrix(angles), "betas": betas}
        losses = geometric_losses(pred, self.batch, self.models, self.regressor,
                                  self.args, self.device)
        losses["total"].backward()
        self.assertGreater(float(losses["joint_m"]), 0)
        self.assertGreater(float(losses["joint_acceleration_m_per_frame2"]), 0)
        for value in (angles, betas):
            self.assertTrue(bool(torch.isfinite(value.grad).all()))
            self.assertGreater(float(value.grad.abs().sum()), 0)
        self.assertGreater(float(angles.grad[:, :, 22:].abs().sum()), 0)

    def test_test_split_is_rejected_before_loading_annotations(self) -> None:
        with self.assertRaisesRegex(ValueError, "train/validation"):
            FullGeometryTargets({"signature": {"split": "test"}}, None, "test")

    def test_missing_gender_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "male/female"):
            gendered_vertices(self.models, self.batch["rotmat"][0],
                               self.batch["betas"][0], torch.full((4,), 2))


if __name__ == "__main__":
    unittest.main()
