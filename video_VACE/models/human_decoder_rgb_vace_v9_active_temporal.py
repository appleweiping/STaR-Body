"""Active-temporal DiviBody v9, additive to the unchanged v8 model.

v8's zero temporal gate converged near zero in the full 3DPW run.  v9 keeps
the exact same tensor/checkpoint contract but starts the VACE path at a
controlled nonzero strength, so cached video features actually reach the
residual heads.  A v8 checkpoint can therefore initialize v9 strictly.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

import torch

from models.human_decoder_rgb_vace_v8_hmr2a import (
    HumanRGBVaceGeometryTransformerV8HMR2A,
    RGBVaceHMR2AConfig,
)


@dataclass(frozen=True)
class RGBVaceActiveTemporalConfig(RGBVaceHMR2AConfig):
    temporal_gate_init: float = 0.15
    freeze_temporal_gate: bool = True

    def validate(self) -> None:
        super().validate()
        if not 0.0 < self.temporal_gate_init < 0.5:
            raise ValueError("v9 temporal gate must be in (0, 0.5)")


class HumanRGBVaceGeometryTransformerV9ActiveTemporal(
    HumanRGBVaceGeometryTransformerV8HMR2A
):
    def __init__(
        self, config: Optional[RGBVaceActiveTemporalConfig] = None
    ) -> None:
        resolved = config or RGBVaceActiveTemporalConfig()
        resolved.validate()
        super().__init__(resolved)
        self.activate_temporal_path()

    def activate_temporal_path(self) -> None:
        value = math.atanh(float(self.config.temporal_gate_init))
        with torch.no_grad():
            self.temporal_gates.fill_(value)
        self.temporal_gates.requires_grad_(not self.config.freeze_temporal_gate)

    def anchor_metadata(self) -> dict[str, object]:
        return {
            "model_format": "divibody-rgb-vace-v9-active-temporal",
            "anchor_name": self.config.anchor_name,
            "anchor_checkpoint_sha256": self.config.anchor_checkpoint_sha256,
            "v8_checkpoint_compatible": True,
            "active_temporal_gate": float(self.config.temporal_gate_init),
            "frozen_temporal_gate": bool(self.config.freeze_temporal_gate),
            "config": asdict(self.config),
        }
