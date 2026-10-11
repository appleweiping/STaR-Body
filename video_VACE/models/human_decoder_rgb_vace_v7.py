"""High-resolution RGB geometry anchor with gated VACE temporal residuals.

This additive v7 model deliberately leaves every earlier decoder unchanged.
It consumes per-frame human-pretrained RGB tokens and the corresponding HMR
SMPL prediction as a strong geometry anchor.  Continuous VACE tokens can only
modify that anchor through zero-initialized gates and zero/identity-initialized
residual heads.  Consequently, a newly constructed model exactly reproduces
the supplied HMR prediction before any v7 optimization.

Expected inputs for the formal T17 -> center-eight protocol are:

* ``rgb_tokens``: ``[B, 8, N_rgb, C_rgb]`` high-resolution HMR/ViTPose tokens;
* ``vace_features``: ``[B, 5, H_vace, W_vace, C_vace]`` continuous VACE tokens;
* ``base_rotmat``: ``[B, 8, 24, 3, 3]`` camera-frame HMR rotations;
* ``base_betas``: ``[B, 8, 10]`` (or ``[B, 10]``) HMR body shape;
* ``base_camera``: ``[B, 8, 3]`` HMR weak-perspective camera parameters.

The implementation is independent of the official HMR 2.0 Python package so
that official features/predictions can be cached once and training remains
deterministic and lightweight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from models.human_decoder_geometry import (
    rotation_6d_to_matrix,
    spatial_sincos_position_encoding,
)


__all__ = [
    "RGBVaceGeometryConfig",
    "HumanRGBVaceGeometryTransformerV7",
]


@dataclass(frozen=True)
class RGBVaceGeometryConfig:
    """Dimensions for the additive RGB+VACE v7 geometry model."""

    rgb_input_dim: int = 1280
    vace_input_dim: int = 1536
    hidden_dim: int = 512
    num_heads: int = 8
    rgb_layers: int = 2
    vace_layers: int = 2
    dropout: float = 0.1
    output_frames: int = 8
    num_betas: int = 10
    max_vace_frames: int = 32
    max_rgb_frames: int = 32

    def validate(self) -> None:
        integer_fields = (
            self.rgb_input_dim,
            self.vace_input_dim,
            self.hidden_dim,
            self.num_heads,
            self.rgb_layers,
            self.vace_layers,
            self.output_frames,
            self.num_betas,
            self.max_vace_frames,
            self.max_rgb_frames,
        )
        if any(value <= 0 for value in integer_fields):
            raise ValueError("All v7 dimension/layer fields must be positive")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.output_frames > self.max_rgb_frames:
            raise ValueError("output_frames exceeds max_rgb_frames")


class _CrossAttentionBlock(nn.Module):
    """Pre-normalized cross attention followed by a residual MLP."""

    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        attended, _ = self.cross_attention(
            self.query_norm(queries),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )
        queries = queries + self.attention_dropout(attended)
        return queries + self.mlp(self.mlp_norm(queries))


def _zero_linear(layer: nn.Linear) -> None:
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class HumanRGBVaceGeometryTransformerV7(nn.Module):
    """Refine an HMR geometry anchor using RGB detail and VACE dynamics.

    RGB attention builds a geometry-aware query independently for each output
    frame.  VACE attention sees all output queries and all continuous video
    tokens, but its update is multiplied by ``tanh(temporal_gates[layer])``.
    Gates start at zero.  The final SMPL/camera heads are also initialized to
    preserve the supplied base prediction exactly.
    """

    def __init__(self, config: Optional[RGBVaceGeometryConfig] = None) -> None:
        super().__init__()
        self.config = config or RGBVaceGeometryConfig()
        self.config.validate()
        cfg = self.config

        self.rgb_projection = nn.Sequential(
            nn.LayerNorm(cfg.rgb_input_dim),
            nn.Linear(cfg.rgb_input_dim, cfg.hidden_dim),
        )
        self.vace_projection = nn.Sequential(
            nn.LayerNorm(cfg.vace_input_dim),
            nn.Linear(cfg.vace_input_dim, cfg.hidden_dim),
        )
        self.smpl_query = nn.Parameter(torch.zeros(1, 1, cfg.hidden_dim))
        self.rgb_frame_embedding = nn.Embedding(cfg.max_rgb_frames, cfg.hidden_dim)
        self.vace_time_embedding = nn.Embedding(cfg.max_vace_frames, cfg.hidden_dim)
        self.vace_spatial_scale = nn.Parameter(torch.zeros(()))

        self.rgb_blocks = nn.ModuleList(
            _CrossAttentionBlock(cfg.hidden_dim, cfg.num_heads, cfg.dropout)
            for _ in range(cfg.rgb_layers)
        )
        self.vace_blocks = nn.ModuleList(
            _CrossAttentionBlock(cfg.hidden_dim, cfg.num_heads, cfg.dropout)
            for _ in range(cfg.vace_layers)
        )
        self.temporal_gates = nn.Parameter(torch.zeros(cfg.vace_layers))
        self.output_norm = nn.LayerNorm(cfg.hidden_dim)

        self.rotation_residual_head = nn.Linear(cfg.hidden_dim, 24 * 6)
        self.shape_residual_head = nn.Linear(cfg.hidden_dim, cfg.num_betas)
        self.camera_residual_head = nn.Linear(cfg.hidden_dim, 3)
        self._initialize_anchor_preserving_heads()

    def _initialize_anchor_preserving_heads(self) -> None:
        _zero_linear(self.rotation_residual_head)
        identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).repeat(24)
        with torch.no_grad():
            self.rotation_residual_head.bias.copy_(identity_6d)
        _zero_linear(self.shape_residual_head)
        _zero_linear(self.camera_residual_head)

    def _validate_inputs(
        self,
        rgb_tokens: torch.Tensor,
        vace_features: torch.Tensor,
        base_rotmat: torch.Tensor,
        base_betas: torch.Tensor,
        base_camera: torch.Tensor,
    ) -> tuple[int, int, int, int]:
        cfg = self.config
        if rgb_tokens.ndim != 4:
            raise ValueError("rgb_tokens must be [B,T,N,C]")
        batch, frames, rgb_count, rgb_width = rgb_tokens.shape
        if frames != cfg.output_frames or rgb_count < 1:
            raise ValueError(
                f"rgb_tokens must contain {cfg.output_frames} nonempty frames"
            )
        if rgb_width != cfg.rgb_input_dim:
            raise ValueError("Unexpected rgb token width")
        if vace_features.ndim != 5:
            raise ValueError("vace_features must be [B,F,H,W,C]")
        if vace_features.shape[0] != batch:
            raise ValueError("RGB and VACE batch sizes differ")
        latent_frames, grid_h, grid_w, vace_width = vace_features.shape[1:]
        if latent_frames < 1 or grid_h < 1 or grid_w < 1:
            raise ValueError("VACE token grid cannot be empty")
        if latent_frames > cfg.max_vace_frames:
            raise ValueError("VACE latent length exceeds max_vace_frames")
        if vace_width != cfg.vace_input_dim:
            raise ValueError("Unexpected VACE token width")
        if base_rotmat.shape != (batch, frames, 24, 3, 3):
            raise ValueError("base_rotmat must be [B,T,24,3,3]")
        if base_betas.shape not in (
            (batch, frames, cfg.num_betas),
            (batch, cfg.num_betas),
        ):
            raise ValueError("base_betas must be [B,T,10] or [B,10]")
        if base_camera.shape != (batch, frames, 3):
            raise ValueError("base_camera must be [B,T,3]")
        floating_inputs = (
            rgb_tokens,
            vace_features,
            base_rotmat,
            base_betas,
            base_camera,
        )
        if any(not tensor.is_floating_point() for tensor in floating_inputs):
            raise TypeError("All v7 inputs must use floating point tensors")
        return batch, frames, grid_h, grid_w

    def forward(
        self,
        rgb_tokens: torch.Tensor,
        vace_features: torch.Tensor,
        base_rotmat: torch.Tensor,
        base_betas: torch.Tensor,
        base_camera: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, frames, grid_h, grid_w = self._validate_inputs(
            rgb_tokens,
            vace_features,
            base_rotmat,
            base_betas,
            base_camera,
        )
        cfg = self.config

        # The HMR image encoder already includes its learned 2D position
        # embeddings.  RGB attention is frame-local to protect fine geometry.
        rgb_memory = self.rgb_projection(rgb_tokens)
        frame_ids = torch.arange(frames, device=rgb_tokens.device)
        queries = self.smpl_query + self.rgb_frame_embedding(frame_ids)[None]
        queries = queries.expand(batch, -1, -1)
        local_queries = queries.reshape(batch * frames, 1, cfg.hidden_dim)
        local_memory = rgb_memory.reshape(
            batch * frames, rgb_memory.shape[2], cfg.hidden_dim
        )
        for block in self.rgb_blocks:
            local_queries = block(local_queries, local_memory)
        rgb_queries = local_queries.reshape(batch, frames, cfg.hidden_dim)

        latent_frames = vace_features.shape[1]
        vace_memory = self.vace_projection(vace_features)
        time_ids = torch.arange(latent_frames, device=vace_features.device)
        time_position = self.vace_time_embedding(time_ids)
        spatial_position = (
            spatial_sincos_position_encoding(
                grid_h, grid_w, cfg.hidden_dim, vace_features.device
            )
            .to(dtype=vace_memory.dtype)
            .reshape(grid_h, grid_w, cfg.hidden_dim)
        )
        vace_memory = vace_memory + time_position[:, None, None, :]
        vace_memory = vace_memory + (
            torch.tanh(self.vace_spatial_scale) * spatial_position[None, :, :, :]
        )
        vace_memory = vace_memory.reshape(batch, -1, cfg.hidden_dim)

        fused_queries = rgb_queries
        for gate, block in zip(self.temporal_gates, self.vace_blocks):
            proposal = block(fused_queries, vace_memory)
            fused_queries = fused_queries + torch.tanh(gate) * (
                proposal - fused_queries
            )
        fused_queries = self.output_norm(fused_queries)

        residual_6d = self.rotation_residual_head(fused_queries).reshape(
            batch, frames, 24, 6
        )
        with torch.autocast(device_type=rgb_tokens.device.type, enabled=False):
            residual_rotmat = rotation_6d_to_matrix(residual_6d.float())
            rotmat = residual_rotmat @ base_rotmat.float()
        if base_betas.ndim == 2:
            base_betas = base_betas[:, None, :].expand(-1, frames, -1)
        betas = base_betas.float() + self.shape_residual_head(fused_queries).float()
        camera = base_camera.float() + self.camera_residual_head(fused_queries).float()

        return {
            "rotmat": rotmat,
            "betas": betas,
            "camera": camera,
            "rotation_residual_6d": residual_6d,
            "rgb_queries": rgb_queries,
            "fused_queries": fused_queries,
            "temporal_gate": torch.tanh(self.temporal_gates),
        }
