"""HMR 2.0a anchored RGB+VACE geometry refinement (additive v8 path).

The v7 implementation remains unchanged.  This module gives the HMR 2.0a
experiment its own model type and records the exact anchor identity in every
configuration/checkpoint while retaining v7's anchor-preserving initialization.
At initialization, v8 therefore reproduces the supplied HMR 2.0a rotations,
shape and camera exactly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import torch

from models.human_decoder_rgb_vace_v7 import (
    HumanRGBVaceGeometryTransformerV7,
    RGBVaceGeometryConfig,
)


HMR2A_SHA256 = "2dcf79638109781d1ae5f5c44fee5f55bc83291c210653feead9b7f04fa6f20e"

__all__ = [
    "HMR2A_SHA256",
    "RGBVaceHMR2AConfig",
    "HumanRGBVaceGeometryTransformerV8HMR2A",
]


@dataclass(frozen=True)
class RGBVaceHMR2AConfig(RGBVaceGeometryConfig):
    """v8 dimensions plus immutable HMR 2.0a provenance."""

    anchor_name: str = "HMR 2.0a"
    anchor_checkpoint_sha256: str = HMR2A_SHA256

    def validate(self) -> None:
        super().validate()
        if self.anchor_name != "HMR 2.0a":
            raise ValueError("v8 requires an HMR 2.0a anchor")
        if self.anchor_checkpoint_sha256.lower() != HMR2A_SHA256:
            raise ValueError("Unexpected HMR 2.0a checkpoint SHA-256")


class HumanRGBVaceGeometryTransformerV8HMR2A(
    HumanRGBVaceGeometryTransformerV7
):
    """Additive v8 refinement whose base prediction must come from HMR 2.0a."""

    def __init__(self, config: Optional[RGBVaceHMR2AConfig] = None) -> None:
        resolved = config or RGBVaceHMR2AConfig()
        resolved.validate()
        super().__init__(resolved)

    def anchor_metadata(self) -> dict[str, object]:
        """Return serializable provenance for caches and trained checkpoints."""

        return {
            "model_format": "divibody-rgb-vace-v8-hmr2a",
            "anchor_name": self.config.anchor_name,
            "anchor_checkpoint_sha256": self.config.anchor_checkpoint_sha256,
            "anchor_preserving_initialization": True,
            "config": asdict(self.config),
        }


class HumanRGBVaceGeometryTransformerV9Isolated(HumanRGBVaceGeometryTransformerV8HMR2A):
    """Checkpoint-compatible refinement with a camera-only gradient route."""

    def camera_only_prediction(self, prediction, base_camera):
        # Also detach shared features, otherwise camera supervision changes 3D.
        result = {key: value.detach() if torch.is_tensor(value) else value
                  for key, value in prediction.items()}
        with torch.autocast(device_type=base_camera.device.type, enabled=False):
            result['camera'] = base_camera.detach().float() + self.camera_residual_head(
                prediction['fused_queries'].detach().float())
        return result
