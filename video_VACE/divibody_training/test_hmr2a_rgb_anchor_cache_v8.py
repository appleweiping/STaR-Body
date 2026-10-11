import copy

import pytest
import torch

from divibody_training.cache_hmr2a_rgb_anchor_v8 import (
    FORMAT,
    build_anchor_plan,
    semantic_vace_digest,
    validate_sidecar,
)
from divibody_training.hmr2a_anchor import HMR2A_SHA256


def _metadata():
    return [
        {
            "window_uid": "seq/person_0/window_0",
            "target_frame_uids": [f"seq/person_0/frame_{i:05d}" for i in range(8)],
        },
        {
            "window_uid": "seq/person_0/window_4",
            "target_frame_uids": [f"seq/person_0/frame_{i:05d}" for i in range(4, 12)],
        },
    ]


def _payload():
    metadata = _metadata()
    vace = {"signature": {"split": "validation"}, "metadata": metadata}
    plan = build_anchor_plan(metadata, "validation")
    count = len(plan.frame_uids)
    sidecar = {
        "signature": {
            "format": FORMAT,
            "split": "validation",
            "hmr2a_checkpoint_sha256": HMR2A_SHA256,
            "vace_semantic_sha256": semantic_vace_digest(vace, 2),
        },
        "frame_uids": plan.frame_uids,
        "window_uids": plan.window_uids,
        "window_frame_indices": plan.window_frame_indices,
        "rgb_tokens": torch.zeros(count, 192, 1280, dtype=torch.bfloat16),
        "base_rotmat": torch.zeros(count, 24, 3, 3),
        "base_betas": torch.zeros(count, 10),
        "base_camera": torch.zeros(count, 3),
    }
    return sidecar, vace


def test_plan_deduplicates_overlapping_frames():
    plan = build_anchor_plan(_metadata(), "validation")
    assert len(plan.frame_uids) == 12
    assert plan.window_frame_indices[1, :4].tolist() == [4, 5, 6, 7]
    assert not plan.development_only


def test_development_plan_and_test_leakage_guard():
    plan = build_anchor_plan(_metadata(), "train", max_windows=1)
    assert plan.development_only and len(plan.window_uids) == 1
    with pytest.raises(ValueError, match="test split"):
        build_anchor_plan(_metadata(), "test")


def test_sidecar_validates_exact_vace_alignment():
    sidecar, vace = _payload()
    validate_sidecar(sidecar, vace)
    broken = copy.deepcopy(sidecar)
    broken["window_frame_indices"][1, 0] = 0
    with pytest.raises(ValueError, match="frame identities"):
        validate_sidecar(broken, vace)


def test_sidecar_rejects_wrong_checkpoint():
    sidecar, vace = _payload()
    sidecar["signature"]["hmr2a_checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint provenance"):
        validate_sidecar(sidecar, vace)
