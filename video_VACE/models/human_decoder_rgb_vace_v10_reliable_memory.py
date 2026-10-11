"""Reliable anchor-memory correction added on top of the unchanged v8 model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from models.human_decoder_geometry import rotation_6d_to_matrix
from models.human_decoder_rgb_vace_v8_hmr2a import (
    HumanRGBVaceGeometryTransformerV8HMR2A,
    RGBVaceHMR2AConfig,
)


@dataclass(frozen=True)
class ReliableMemoryConfig(RGBVaceHMR2AConfig):
    memory_layers: int = 2
    shape_consensus: float = 1.0

    def validate(self) -> None:
        super().validate()
        if self.memory_layers < 1 or not 0.0 <= self.shape_consensus <= 1.0:
            raise ValueError("Invalid reliable-memory configuration")


class HumanRGBVaceGeometryTransformerV10ReliableMemory(
    HumanRGBVaceGeometryTransformerV8HMR2A
):
    """Jointly correct all SMPL joints from an eight-frame anchor history."""

    NEW_PREFIXES = ("anchor_projection", "memory_encoder", "memory_norm", "memory_")

    def __init__(self, config: Optional[ReliableMemoryConfig] = None) -> None:
        cfg = config or ReliableMemoryConfig()
        cfg.validate()
        super().__init__(cfg)
        anchor_width = 24 * 6 + cfg.num_betas + 3
        self.anchor_projection = nn.Sequential(
            nn.LayerNorm(anchor_width), nn.Linear(anchor_width, cfg.hidden_dim), nn.GELU()
        )
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim, nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim * 2, dropout=cfg.dropout,
            batch_first=True, norm_first=True,
        )
        self.memory_encoder = nn.TransformerEncoder(layer, cfg.memory_layers)
        self.memory_norm = nn.LayerNorm(cfg.hidden_dim)
        self.memory_rotation_head = nn.Linear(cfg.hidden_dim, 24 * 6)
        self.memory_shape_head = nn.Linear(cfg.hidden_dim, cfg.num_betas)
        self.memory_camera_head = nn.Linear(cfg.hidden_dim, 3)
        self._initialize_memory_heads()

    def _initialize_memory_heads(self) -> None:
        for layer in (self.memory_rotation_head, self.memory_shape_head, self.memory_camera_head):
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)
        with torch.no_grad():
            self.memory_rotation_head.bias.copy_(
                torch.tensor([1., 0., 0., 0., 1., 0.]).repeat(24)
            )

    def forward(self, rgb_tokens, vace_features, base_rotmat, base_betas, base_camera):
        base = super().forward(rgb_tokens, vace_features, base_rotmat, base_betas, base_camera)
        anchor = torch.cat((
            base["rotmat"][..., :2, :].reshape(*base["rotmat"].shape[:2], -1),
            base["betas"], base["camera"],
        ), dim=-1)
        memory = self.memory_encoder(base["fused_queries"] + self.anchor_projection(anchor))
        memory = self.memory_norm(memory)
        residual_6d = self.memory_rotation_head(memory).reshape(*memory.shape[:2], 24, 6)
        with torch.autocast(device_type=memory.device.type, enabled=False):
            rotmat = rotation_6d_to_matrix(residual_6d.float()) @ base["rotmat"].float()
        raw_betas = base["betas"].float() + self.memory_shape_head(memory).float()
        consensus = raw_betas.mean(1, keepdim=True).expand_as(raw_betas)
        betas = torch.lerp(raw_betas, consensus, float(self.config.shape_consensus))
        camera = base["camera"].float() + self.memory_camera_head(memory).float()
        return {**base, "rotmat":rotmat, "betas":betas, "camera":camera,
                "memory_rotation_residual_6d":residual_6d, "memory_tokens":memory}

    def memory_parameters(self):
        for name, parameter in self.named_parameters():
            if name.startswith(self.NEW_PREFIXES):
                yield parameter
