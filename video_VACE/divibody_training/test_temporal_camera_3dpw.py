"""Focused tests for camera-aligned temporal 3DPW supervision."""

from __future__ import annotations

import math
import types
import unittest

import torch

from train_divibody_real_3dpw_temporal import axis_angle_to_matrix
from train_divibody_real_3dpw_temporal_camera import camera_temporal_losses


def zero_prediction(global_orient: torch.Tensor) -> dict[str, torch.Tensor]:
    batch, frames, _ = global_orient.shape
    return {
        "global_orient": global_orient,
        "body_pose": torch.zeros(batch, frames, 63),
        "betas": torch.zeros(batch, 10),
        "transl": torch.zeros(batch, frames, 3),
        "left_hand_pose": torch.zeros(batch, frames, 45),
        "right_hand_pose": torch.zeros(batch, frames, 45),
        "jaw_pose": torch.zeros(batch, frames, 3),
        "expression": torch.zeros(batch, frames, 10),
        "object_state": torch.zeros(batch, frames, 9),
    }


def loss_args() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        rotation_weight=1.0,
        axis_angle_weight=0.25,
        shape_weight=0.1,
        translation_weight=0.0,
        velocity_weight=0.5,
        acceleration_weight=0.25,
        unsupported_head_reg=1.0e-4,
    )


class CameraTemporalLossTests(unittest.TestCase):
    def test_noncommuting_camera_left_multiply_has_zero_loss(self) -> None:
        frames = 3
        world_rx = torch.tensor([math.pi / 2.0, 0.0, 0.0]).repeat(1, frames, 1)
        camera_rz = axis_angle_to_matrix(
            torch.tensor([0.0, 0.0, math.pi / 2.0])
        ).repeat(1, frames, 1, 1)
        # qz(90) * qx(90) = (w,x,y,z)=(.5,.5,.5,.5), hence a
        # 120-degree rotation around the (1,1,1) axis.
        composite_axis_angle = torch.full(
            (1, frames, 3), 2.0 * math.pi / (3.0 * math.sqrt(3.0))
        )
        batch = {
            "global_orient": world_rx,
            "body_pose": torch.zeros(1, frames, 63),
            "betas": torch.zeros(1, 10),
            "transl": torch.zeros(1, frames, 3),
            "camera_rotation": camera_rz,
        }
        losses = camera_temporal_losses(
            zero_prediction(composite_axis_angle),
            batch,
            loss_args(),
            torch.device("cpu"),
        )
        self.assertLess(float(losses["total"]), 1.0e-10)
        expected = camera_rz @ axis_angle_to_matrix(world_rx)
        actual = axis_angle_to_matrix(composite_axis_angle)
        self.assertTrue(torch.allclose(actual, expected, atol=1.0e-6, rtol=1.0e-6))
        reverse = axis_angle_to_matrix(world_rx) @ camera_rz
        self.assertFalse(torch.allclose(actual, reverse, atol=1.0e-4, rtol=1.0e-4))

    def test_world_prediction_is_rejected_by_nonidentity_camera_target(self) -> None:
        frames = 3
        batch = {
            "global_orient": torch.zeros(1, frames, 3),
            "body_pose": torch.zeros(1, frames, 63),
            "betas": torch.zeros(1, 10),
            "transl": torch.zeros(1, frames, 3),
            "camera_rotation": axis_angle_to_matrix(
                torch.tensor([0.0, 0.0, math.pi / 2.0])
            ).repeat(1, frames, 1, 1),
        }
        wrong = camera_temporal_losses(
            zero_prediction(torch.zeros(1, frames, 3)),
            batch,
            loss_args(),
            torch.device("cpu"),
        )
        correct = camera_temporal_losses(
            zero_prediction(
                torch.tensor([0.0, 0.0, math.pi / 2.0]).repeat(1, frames, 1)
            ),
            batch,
            loss_args(),
            torch.device("cpu"),
        )
        self.assertGreater(float(wrong["global_camera_rotation_chordal"]), 0.1)
        self.assertLess(float(correct["total"]), 1.0e-10)


if __name__ == "__main__":
    unittest.main()
