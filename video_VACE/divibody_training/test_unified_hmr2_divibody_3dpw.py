"""Protocol-level tests for the additive unified 3DPW evaluator."""

from __future__ import annotations

import pytest

from divibody_training.evaluate_unified_hmr2_divibody_3dpw import (
    HMR_TARGET_ASPECT,
    hmr_bbox_size,
    metric_protocol,
)


def test_hmr_bbox_expands_wide_box_before_square_precrop() -> None:
    center_x, center_y, side = hmr_bbox_size((10.0, 20.0, 110.0, 70.0))
    assert center_x == 60.0
    assert center_y == 45.0
    assert side == pytest.approx(100.0 / HMR_TARGET_ASPECT)


def test_hmr_bbox_keeps_tall_box_height() -> None:
    _, _, side = hmr_bbox_size((10.0, 20.0, 60.0, 120.0))
    assert side == pytest.approx(100.0)


def test_unified_protocol_names_all_four_metrics() -> None:
    protocol = metric_protocol()
    assert set(("mpjpe", "pa_mpjpe", "pve", "acceleration")) <= set(protocol)


def test_degenerate_hmr_bbox_is_rejected() -> None:
    with pytest.raises(ValueError, match="Degenerate"):
        hmr_bbox_size((1.0, 1.0, 1.0, 4.0))
