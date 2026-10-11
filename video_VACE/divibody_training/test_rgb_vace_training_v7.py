"""CPU cache-contract tests for the new RGB+VACE v7 trainer."""

from __future__ import annotations

import copy
import unittest

import torch

from divibody_training.train_divibody_rgb_vace_v7 import (
    CACHE_FORMAT,
    RGBVaceGeometryDataset,
    model_config,
    validate_cache,
)


def valid_payload(split: str = "train", windows: int = 3) -> dict:
    return {
        "format": CACHE_FORMAT,
        "signature": {
            "split": split,
            "contains_test_ground_truth": False,
            "hmr_protocol": "official-hmr2-vit-h-256x192",
            "vace_protocol": "t17-clean-t500-zero-text",
        },
        "inputs": {
            "rgb_tokens": torch.randn(windows, 8, 12, 32),
            "vace_features": torch.randn(windows, 5, 2, 3, 48),
            "base_rotmat": torch.eye(3)
            .reshape(1, 1, 1, 3, 3)
            .repeat(windows, 8, 24, 1, 1),
            "base_betas": torch.randn(windows, 8, 10),
            "base_camera": torch.randn(windows, 8, 3),
        },
        "targets": {
            "full_pose": torch.randn(windows, 8, 72),
            "betas": torch.randn(windows, 8, 10),
            "gender_code": torch.zeros(windows, 8, dtype=torch.long),
            "cam_pose": torch.eye(4).reshape(1, 1, 4, 4).repeat(windows, 8, 1, 1),
        },
        "masks": {
            "frame_owner": torch.ones(windows, 8, dtype=torch.bool),
            "velocity_owner": torch.ones(windows, 7, dtype=torch.bool),
            "acceleration_owner": torch.ones(windows, 6, dtype=torch.bool),
        },
    }


class RGBVaceTrainingV7Tests(unittest.TestCase):
    def test_valid_cache_audit_dataset_and_config(self) -> None:
        payload = valid_payload()
        audit = validate_cache(payload, "train")
        self.assertEqual(audit["windows"], 3)
        self.assertEqual(audit["rgb_token_shape"], [8, 12, 32])
        dataset = RGBVaceGeometryDataset(payload)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(tuple(dataset[1]["rgb_tokens"].shape), (8, 12, 32))
        config = model_config(payload)
        self.assertEqual(config.rgb_input_dim, 32)
        self.assertEqual(config.vace_input_dim, 48)

    def test_training_rejects_test_data_even_if_named_train(self) -> None:
        payload = valid_payload()
        payload["signature"]["contains_test_ground_truth"] = True
        with self.assertRaisesRegex(ValueError, "test ground truth"):
            validate_cache(payload, "train")
        payload = valid_payload("test")
        with self.assertRaisesRegex(ValueError, "Expected a 'train'"):
            validate_cache(payload, "train")

    def test_invalid_input_and_mask_shapes_are_rejected(self) -> None:
        payload = valid_payload()
        payload["inputs"]["base_camera"] = payload["inputs"]["base_camera"][:, :7]
        with self.assertRaisesRegex(ValueError, "camera shape"):
            validate_cache(payload, "train")
        payload = valid_payload()
        payload["masks"]["velocity_owner"] = torch.ones(3, 8, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "velocity_owner"):
            validate_cache(payload, "train")

    def test_validation_does_not_mutate_cache(self) -> None:
        payload = valid_payload()
        snapshot = copy.deepcopy(payload)
        validate_cache(payload, "train")
        for section in ("inputs", "targets", "masks"):
            for name, value in snapshot[section].items():
                self.assertTrue(torch.equal(payload[section][name], value), name)


if __name__ == "__main__":
    unittest.main()
