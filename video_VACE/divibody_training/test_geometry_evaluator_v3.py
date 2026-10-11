"""Focused, asset-free safety tests for the new v3 geometry evaluator."""

from __future__ import annotations

import contextlib
import copy
import io
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import evaluate_temporal_geometry_3dpw as evaluator
from train_divibody_real_3dpw_temporal import axis_angle_to_matrix


def checkpoint_contract() -> dict:
    return {
        "format": "divibody-real-3dpw-temporal-geometry-v3",
        "target_coordinate_system": "camera",
        "pose_protocol": {"id": "3dpw_camera_smpl24_geometry_v3"},
        "decoder_config": {"input_dim": 1536, "max_latent_frames": 128},
        "args": {"clip_length": 4},
    }


def geometry_inputs(count: int = 2) -> tuple[np.ndarray, ...]:
    return (
        np.tile(np.eye(3, dtype=np.float32), (count, 24, 1, 1)),
        np.zeros((count, 10), np.float32),
        np.zeros((count, 72), np.float32),
        np.zeros((count, 10), np.float32),
        np.zeros(count, np.int64),
        np.tile(np.eye(3, dtype=np.float32), (count, 1, 1)),
    )


class GeometryCheckpointContractTests(unittest.TestCase):
    def test_v3_needs_no_v2_axis_angle_metadata(self) -> None:
        self.assertEqual(
            evaluator.validate_checkpoint_protocol(checkpoint_contract()), 4
        )

    def test_trained_context_not_embedding_capacity_is_returned(self) -> None:
        checkpoint = checkpoint_contract()
        checkpoint["args"]["clip_length"] = 8
        self.assertEqual(evaluator.validate_checkpoint_protocol(checkpoint), 8)

    def test_development_checkpoint_is_not_allowed_on_formal_test(self) -> None:
        checkpoint = checkpoint_contract()
        checkpoint["development_only"] = True
        self.assertEqual(
            evaluator.validate_checkpoint_protocol(checkpoint, "validation"), 4
        )
        with self.assertRaisesRegex(ValueError, "Development-only"):
            evaluator.validate_checkpoint_protocol(checkpoint, "test")
        checkpoint["development_only"] = False
        checkpoint["args"]["smoke_max_steps"] = 1
        with self.assertRaisesRegex(ValueError, "Development-only"):
            evaluator.validate_checkpoint_protocol(checkpoint, "test")

    def test_wrong_format_coordinate_and_protocol_are_rejected(self) -> None:
        for key, value in (
            ("format", "divibody-real-3dpw-temporal-camera-v2"),
            ("target_coordinate_system", "world"),
            ("pose_protocol", {"id": "3dpw_global_orient_camera_matrix_v1"}),
            ("pose_protocol", None),
        ):
            with self.subTest(key=key, value=value):
                checkpoint = checkpoint_contract()
                checkpoint[key] = value
                with self.assertRaises(ValueError):
                    evaluator.validate_checkpoint_protocol(checkpoint)

    def test_missing_or_invalid_trained_context_is_rejected(self) -> None:
        for length in (None, True, 2, 129, 4.0):
            with self.subTest(length=length):
                checkpoint = checkpoint_contract()
                checkpoint["args"]["clip_length"] = length
                with self.assertRaises(ValueError):
                    evaluator.validate_checkpoint_protocol(checkpoint)

    def test_feature_signature_missing_values_cannot_silently_match(self) -> None:
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            evaluator.validate_feature_compatibility(
                {"train_cache_signature": {}}, {"signature": {}}
            )

    def test_train_and_evaluation_vace_signatures_match(self) -> None:
        settings = {
            "image_size": 96,
            "feature_layers": [7, 14, 22],
            "vace_timestep": 500,
        }
        checkpoint = {
            "train_cache_signature": {
                **settings,
                "split": "train",
                "feature_backend": "vace",
            }
        }
        payload = {
            "signature": {
                **settings,
                "split": "validation",
                "feature_backend": "official frozen Wan2.1-VACE",
            }
        }
        evaluator.validate_feature_compatibility(checkpoint, payload)
        checkpoint["train_cache_signature"]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "train split"):
            evaluator.validate_feature_compatibility(checkpoint, payload)


class _FeatureOnlyDecoder(torch.nn.Module):
    """No camera/targets/old AA output exists in this test model's interface."""

    def forward(self, features, grid_size, target_frames):
        frames, height, width = grid_size
        assert target_frames == frames
        batch = features.shape[0]
        markers = features.reshape(batch, frames, height * width, -1)[:, :, 0, 0]
        root_aa = torch.stack(
            (torch.zeros_like(markers), torch.zeros_like(markers), markers * 0.1),
            dim=-1,
        )
        rotations = torch.eye(3, device=features.device).repeat(batch, frames, 24, 1, 1)
        rotations[:, :, 0] = axis_angle_to_matrix(root_aa)
        end_aa = torch.stack(
            (
                0.3 + markers * 0.02,
                torch.zeros_like(markers),
                torch.zeros_like(markers),
            ),
            dim=-1,
        )
        rotations[:, :, 23] = axis_angle_to_matrix(end_aa)
        return {"rotmat": rotations, "betas": torch.zeros(batch, 10)}


class GeometryCenteredPredictionTests(unittest.TestCase):
    @staticmethod
    def payload() -> dict:
        metadata = [
            {"sequence": "a", "person_id": 0, "frame_id": frame}
            for frame in (0, 1, 2, 4, 5)
        ] + [
            {"sequence": "a", "person_id": 1, "frame_id": frame}
            for frame in (0, 1, 2, 3, 4)
        ]
        count = len(metadata)
        return {
            "features": torch.arange(count).float()[:, None, None].repeat(1, 4, 1536),
            "metadata": metadata,
        }

    def run_predictions(self, payload: dict) -> tuple[np.ndarray, np.ndarray]:
        dataset = evaluator.CenteredFrameClipDataset(payload, 4)
        with contextlib.redirect_stdout(io.StringIO()):
            return evaluator.run_temporal_predictions(
                _FeatureOnlyDecoder(), dataset, 3, 0, torch.device("cpu")
            )

    def test_each_frame_is_selected_once_and_all_24_rotations_survive(self) -> None:
        payload = self.payload()
        predicted, betas = self.run_predictions(payload)
        count = len(payload["metadata"])
        self.assertEqual(predicted.shape, (count, 24, 3, 3))
        self.assertEqual(betas.shape, (count, 10))
        markers = torch.arange(count).float()
        expected_root = axis_angle_to_matrix(
            torch.stack((markers * 0, markers * 0, markers * 0.1), dim=-1)
        ).numpy()
        expected_end = axis_angle_to_matrix(
            torch.stack((0.3 + markers * 0.02, markers * 0, markers * 0), dim=-1)
        ).numpy()
        np.testing.assert_allclose(predicted[:, 0], expected_root, atol=1e-6)
        np.testing.assert_allclose(predicted[:, 23], expected_end, atol=1e-6)
        self.assertFalse(np.allclose(predicted[:, 23], np.eye(3)))

    def test_inference_does_not_read_gt_gender_or_camera_metadata(self) -> None:
        payload = self.payload()
        first = self.run_predictions(payload)
        payload["targets"] = {"full_pose": "must not be read", "gender_code": None}
        for record in payload["metadata"]:
            record["cam_pose"] = "must not be read"
            record["gender_code"] = "must not be read"
        second = self.run_predictions(payload)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_context_never_crosses_identity_or_missing_frame(self) -> None:
        payload = self.payload()
        dataset = evaluator.CenteredFrameClipDataset(payload, 4)
        targets = []
        for indices, target, local in dataset.requests:
            records = [payload["metadata"][int(index)] for index in indices]
            identities = {(item["sequence"], item["person_id"]) for item in records}
            frames = [item["frame_id"] for item in records]
            self.assertEqual(len(identities), 1)
            self.assertTrue(all(b - a in (0, 1) for a, b in zip(frames, frames[1:])))
            self.assertEqual(int(indices[local]), target)
            targets.append(target)
        self.assertEqual(sorted(targets), list(range(len(payload["metadata"]))))


class GeometryInputValidationTests(unittest.TestCase):
    def test_valid_rotations_are_accepted(self) -> None:
        evaluator.validate_geometry_inputs(*geometry_inputs())

    def test_reflection_and_degenerate_rotations_are_rejected(self) -> None:
        for value in (-1.0, 0.0):
            with self.subTest(value=value):
                values = geometry_inputs()
                values[0][0, 23, 0, 0] = value
                with self.assertRaisesRegex(ValueError, "proper SO"):
                    evaluator.validate_geometry_inputs(*values)

    def test_axis_angle_only_output_cannot_be_misinterpreted_as_rotations(self) -> None:
        values = list(geometry_inputs())
        values[0] = np.zeros((2, 72), np.float32)
        with self.assertRaisesRegex(ValueError, "shapes"):
            evaluator.validate_geometry_inputs(*values)

    def test_nonfinite_predictions_are_rejected(self) -> None:
        values = geometry_inputs()
        values[0][0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            evaluator.validate_geometry_inputs(*values)


class GeometrySourceAlignmentTests(unittest.TestCase):
    @staticmethod
    def fixture() -> tuple[dict, list, dict]:
        payload = {
            "targets": {
                "full_pose": torch.zeros(1, 72),
                "betas": torch.zeros(1, 10),
                "gender_code": torch.zeros(1, dtype=torch.long),
            },
            "metadata": [
                {
                    "cam_pose": np.eye(4).tolist(),
                    "image_path": "/images/a/image_00000.jpg",
                }
            ],
        }
        samples = [
            SimpleNamespace(
                annotation_path="unused.pkl",
                sequence="a",
                person_index=0,
                frame_index=0,
            )
        ]
        annotation = {
            "poses": [np.zeros((1, 72))],
            "betas": [np.zeros(10)],
            "genders": ["m"],
            "cam_poses": [np.eye(4)],
        }
        return payload, samples, annotation

    def test_matching_source_targets_are_accepted(self) -> None:
        payload, samples, annotation = self.fixture()
        with patch.object(evaluator, "_load_annotation", return_value=annotation):
            evaluator.validate_cached_annotation_values(payload, samples)

    def test_target_gender_and_camera_mismatch_are_rejected(self) -> None:
        payload, samples, annotation = self.fixture()
        for mismatch in ("full_pose", "betas", "gender_code", "camera"):
            with self.subTest(mismatch=mismatch):
                changed = copy.deepcopy(payload)
                if mismatch == "camera":
                    changed["metadata"][0]["cam_pose"][0][3] = 1.0
                else:
                    changed["targets"][mismatch].reshape(-1)[0] = 1
                with patch.object(
                    evaluator, "_load_annotation", return_value=annotation
                ):
                    with self.assertRaisesRegex(ValueError, "disagree"):
                        evaluator.validate_cached_annotation_values(changed, samples)

    def test_source_frame_fallback_is_rejected(self) -> None:
        payload, samples, annotation = self.fixture()
        payload["metadata"][0]["image_path"] = "/images/a/image_00001.jpg"
        with patch.object(evaluator, "_load_annotation", return_value=annotation):
            with self.assertRaisesRegex(ValueError, "strict annotation"):
                evaluator.validate_cached_annotation_values(payload, samples)


class GeometryCameraCompositionTests(unittest.TestCase):
    def test_pred_is_not_rotated_twice_and_gt_is_left_composed_once(self) -> None:
        pred_rotmat, betas, gt_pose, gt_betas, genders, cameras = geometry_inputs()
        gt_pose[:, 0] = math.pi / 2
        gt_pose[:, 71] = 0.4
        camera_rotation = axis_angle_to_matrix(torch.tensor([0.0, 0.0, math.pi / 2]))
        cameras[:] = camera_rotation.numpy()
        expected = axis_angle_to_matrix(torch.from_numpy(gt_pose).reshape(2, 24, 3))
        expected[:, 0] = camera_rotation @ expected[:, 0]
        pred_rotmat[:] = expected.numpy()
        genders[:] = (0, 1)
        cam_pose = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
        cam_pose[:, :3, :3] = cameras
        payload = {
            "targets": {
                "full_pose": gt_pose,
                "betas": gt_betas,
                "gender_code": genders,
            },
            "metadata": [{"cam_pose": camera.tolist()} for camera in cam_pose],
        }
        calls = []

        class SpySMPL:
            def __init__(self, gender):
                self.gender = gender

            def __call__(self, **kwargs):
                calls.append((self.gender, kwargs))
                root = kwargs["global_orient"][:, 0]
                positions = torch.arange(6890, dtype=root.dtype, device=root.device)
                template = torch.stack(
                    (
                        positions * 0.001,
                        torch.sin(positions * 0.13),
                        torch.cos(positions * 0.17),
                    ),
                    dim=-1,
                )
                vertices = torch.einsum("bij,vj->bvi", root, template)
                return SimpleNamespace(vertices=vertices)

        regressor = np.zeros((17, 6890), np.float32)
        regressor[np.arange(17), np.arange(17)] = 1.0
        with (
            patch.object(
                evaluator,
                "create_smpl_models",
                return_value={i: SpySMPL(i) for i in (0, 1, 2)},
            ),
            patch.object(evaluator.np, "load", return_value=regressor),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            metrics, prediction, target = evaluator.compute_geometry_metrics(
                pred_rotmat,
                betas,
                payload,
                Path("unused"),
                Path("unused.npy"),
                8,
                torch.device("cpu"),
                "camera",
            )
        self.assertEqual([gender for gender, _ in calls], [2, 0, 1])
        for _, kwargs in calls:
            self.assertIs(kwargs["pose2rot"], False)
            self.assertEqual(kwargs["global_orient"].shape[1:], (1, 3, 3))
            self.assertEqual(kwargs["body_pose"].shape[1:], (23, 3, 3))
        torch.testing.assert_close(calls[0][1]["global_orient"], expected[:, :1])
        torch.testing.assert_close(calls[0][1]["body_pose"], expected[:, 1:])
        for row, (_, kwargs) in enumerate(calls[1:]):
            torch.testing.assert_close(
                kwargs["global_orient"], expected[row : row + 1, :1]
            )
        wrong_twice = camera_rotation @ expected[:, 0]
        self.assertFalse(
            torch.allclose(calls[0][1]["global_orient"][:, 0], wrong_twice)
        )
        np.testing.assert_allclose(prediction, target, atol=1e-6)
        for name, values in metrics.items():
            self.assertLess(float(np.max(values)), 0.01, name)


if __name__ == "__main__":
    unittest.main()
