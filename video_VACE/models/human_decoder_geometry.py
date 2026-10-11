"""Geometry-supervised extension copied from ``models/human_decoder.py``.

The original decoder is deliberately unchanged. Its parameter names, legacy
axis-angle head, part queries, and shape head are retained for camera-v2 warm
starts. New spatial encoding and full-SMPL residual rotations are additive.
Use ``rotmat``, not the legacy axis-angle outputs, for SMPL geometry and losses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle vectors to FP32 matrices with stable gradients at zero.

    The sinc form avoids division by the rotation angle and is differentiable
    at the identity. Geometry is deliberately outside mixed-precision regions.
    """
    if axis_angle.shape[-1] != 3:
        raise ValueError("axis_angle must have final dimension 3")
    with torch.autocast(device_type=axis_angle.device.type, enabled=False):
        vectors = axis_angle.float()
        angle = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
        x, y, z = vectors.unbind(dim=-1)
        zero = torch.zeros_like(x)
        skew = torch.stack(
            (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
        ).reshape(*vectors.shape[:-1], 3, 3)
        identity = torch.eye(3, dtype=torch.float32, device=vectors.device)
        sin_over_angle = torch.sinc(angle / math.pi)[..., None]
        one_minus_cos_over_angle_squared = (
            0.5 * torch.sinc(angle / (2.0 * math.pi)).square()
        )[..., None]
        return (
            identity
            + sin_over_angle * skew
            + one_minus_cos_over_angle_squared * (skew @ skew)
        )


def rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Gram-Schmidt two column vectors into a proper FP32 rotation matrix.

    The convention is ``[column_1, column_2]``: identity is
    ``[1, 0, 0, 0, 1, 0]``. Degenerate vectors use deterministic orthogonal
    fallbacks, preventing zero/collinear head outputs from producing NaNs or
    singular matrices. The ordinary nondegenerate path is differentiable.
    """
    if rotation_6d.shape[-1] != 6:
        raise ValueError("rotation_6d must have final dimension 6")
    with torch.autocast(device_type=rotation_6d.device.type, enabled=False):
        vectors = rotation_6d.float()
        first, second = vectors[..., :3], vectors[..., 3:]
        epsilon = 1.0e-6
        first_norm = torch.linalg.vector_norm(first, dim=-1, keepdim=True)
        first_unit = first / first_norm.clamp_min(epsilon)
        canonical_first = torch.zeros_like(first)
        canonical_first[..., 0] = 1.0
        first_unit = torch.where(
            first_norm > epsilon, first_unit, canonical_first
        )
        second_orthogonal = second - (
            first_unit * second
        ).sum(dim=-1, keepdim=True) * first_unit
        second_norm = torch.linalg.vector_norm(
            second_orthogonal, dim=-1, keepdim=True
        )
        second_unit = second_orthogonal / second_norm.clamp_min(epsilon)

        # Pick the least parallel Cartesian axis for an unambiguous fallback.
        fallback_axis = F.one_hot(
            first_unit.abs().argmin(dim=-1), num_classes=3
        ).to(dtype=torch.float32)
        fallback = fallback_axis - (
            first_unit * fallback_axis
        ).sum(dim=-1, keepdim=True) * first_unit
        fallback = F.normalize(fallback, dim=-1, eps=epsilon)
        second_unit = torch.where(
            second_norm > epsilon, second_unit, fallback
        )
        third_unit = torch.cross(first_unit, second_unit, dim=-1)
        # Re-orthogonalize to limit cancellation for nearly parallel columns.
        third_unit = F.normalize(third_unit, dim=-1, eps=epsilon)
        second_unit = torch.cross(third_unit, first_unit, dim=-1)
        return torch.stack((first_unit, second_unit, third_unit), dim=-1)


def spatial_sincos_position_encoding(
    grid_h: int,
    grid_w: int,
    hidden_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Return row-major 2D sin/cos positions of shape ``[H*W, D]``.

    Interleaving coordinates and rounding the frequency count upward supports
    hidden dimensions not divisible by four, including very small test models.
    """
    if grid_h < 1 or grid_w < 1 or hidden_dim < 1:
        raise ValueError("Spatial grid sizes and hidden_dim must be positive")
    frequency_count = max(1, (hidden_dim + 3) // 4)
    frequency_ids = torch.arange(
        frequency_count, dtype=torch.float32, device=device
    )
    frequencies = torch.exp(
        -math.log(10000.0) * frequency_ids / max(frequency_count - 1, 1)
    )
    y = torch.arange(grid_h, dtype=torch.float32, device=device)
    x = torch.arange(grid_w, dtype=torch.float32, device=device)
    y_phase = y[:, None, None] * frequencies[None, None, :]
    x_phase = x[None, :, None] * frequencies[None, None, :]
    y_phase = y_phase.expand(grid_h, grid_w, -1)
    x_phase = x_phase.expand(grid_h, grid_w, -1)
    if hidden_dim == 1:
        return (x_phase.sin() + y_phase.cos()).reshape(grid_h * grid_w, 1)
    encoding = torch.stack(
        (x_phase.sin(), y_phase.sin(), x_phase.cos(), y_phase.cos()), dim=-1
    ).flatten(start_dim=-2)
    return encoding[..., :hidden_dim].reshape(grid_h * grid_w, hidden_dim)


@dataclass
class GeometryDecoderConfig:
    input_dim: int = 1536
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    num_betas: int = 10
    num_expression: int = 10
    body_pose_dim: int = 63
    hand_pose_dim: int = 45
    max_latent_frames: int = 128


class QueryDecoderLayer(nn.Module):
    """Pre-normalized query self-attention and feature cross-attention."""

    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor,
                memory: torch.Tensor) -> torch.Tensor:
        q = self.query_norm(queries)
        queries = queries + self.self_attn(q, q, q, need_weights=False)[0]
        queries = queries + self.cross_attn(
            self.cross_norm(queries), self.memory_norm(memory),
            self.memory_norm(memory), need_weights=False)[0]
        return queries + self.ffn(self.ffn_norm(queries))


class HumanGeometryQueryTransformer(nn.Module):
    """Add full-SMPL residual geometry to the camera-v2 temporal decoder.

    Input tokens use shape ``[batch, tokens, input_dim]``. ``grid_size`` must
    describe the unpadded DiT token grid as ``(latent_time, height, width)``.
    Predictions are returned at ``target_frames`` through temporal interpolation.
    ``rotmat`` contains the final camera-root/local-body rotations for all 24
    SMPL joints, whereas axis-angle fields expose the legacy warm-start base.
    """

    part_names = ("body", "left_hand", "right_hand", "face", "object")

    def __init__(self, config: Optional[GeometryDecoderConfig] = None) -> None:
        super().__init__()
        self.config = config or GeometryDecoderConfig()
        cfg = self.config
        if cfg.body_pose_dim != 63:
            raise ValueError(
                "Geometry decoder preserves the camera-v2 body_pose_dim=63 "
                "head; its residual head supplies all 23 SMPL body joints"
            )

        self.input_projection = nn.Sequential(
            nn.LayerNorm(cfg.input_dim),
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.hidden_dim),
        )
        self.source_time_embedding = nn.Embedding(
            cfg.max_latent_frames, cfg.hidden_dim)
        self.part_embedding = nn.Parameter(
            torch.randn(len(self.part_names), cfg.hidden_dim) * 0.02)
        self.query_time_embedding = nn.Embedding(
            cfg.max_latent_frames, cfg.hidden_dim)
        self.layers = nn.ModuleList([
            QueryDecoderLayer(cfg.hidden_dim, cfg.num_heads, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])
        self.output_norm = nn.LayerNorm(cfg.hidden_dim)

        self.body_head = nn.Linear(
            cfg.hidden_dim, 3 + cfg.body_pose_dim + 3)
        self.left_hand_head = nn.Linear(cfg.hidden_dim, cfg.hand_pose_dim)
        self.right_hand_head = nn.Linear(cfg.hidden_dim, cfg.hand_pose_dim)
        self.face_head = nn.Linear(
            cfg.hidden_dim, 3 + cfg.num_expression)
        self.object_head = nn.Linear(cfg.hidden_dim, 9)
        self.shape_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.num_betas),
        )
        self.spatial_position_scale = nn.Parameter(torch.tensor(0.1))
        self.rotation_residual_head = nn.Linear(cfg.hidden_dim, 24 * 6)
        nn.init.zeros_(self.rotation_residual_head.weight)
        with torch.no_grad():
            identity_6d = torch.tensor(
                (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                dtype=self.rotation_residual_head.bias.dtype,
            )
            self.rotation_residual_head.bias.copy_(identity_6d.repeat(24))

    def load_camera_v2_weights(
        self, state_dict: Mapping[str, torch.Tensor]
    ) -> Dict[str, list[str]]:
        """Strictly load legacy weights while allowing only new parameters.

        All keys and shapes are checked before changing this model. Loading
        copies the values; the supplied mapping/tensors are never modified.
        The returned key lists are suitable for JSON checkpoint provenance.
        """
        current = self.state_dict()
        incoming_keys = set(state_dict)
        current_keys = set(current)
        allowed_missing = {
            "spatial_position_scale",
            "rotation_residual_head.weight",
            "rotation_residual_head.bias",
        }
        missing = current_keys - incoming_keys
        unexpected = incoming_keys - current_keys
        mismatched = []
        for key in sorted(current_keys & incoming_keys):
            value = state_dict[key]
            if not isinstance(value, torch.Tensor):
                mismatched.append(f"{key}: expected a Tensor")
            elif current[key].shape != value.shape:
                mismatched.append(
                    f"{key}: expected {tuple(current[key].shape)}, "
                    f"got {tuple(value.shape)}"
                )
        if missing - allowed_missing or unexpected or mismatched:
            raise RuntimeError(
                "Incompatible camera-v2 state_dict: "
                f"missing={sorted(missing - allowed_missing)}, "
                f"unexpected={sorted(unexpected)}, mismatched={mismatched}"
            )
        loaded = self.load_state_dict(state_dict, strict=False)
        return {
            "loaded": sorted(incoming_keys),
            "missing": sorted(loaded.missing_keys),
        }

    @staticmethod
    def _resize_time(x: torch.Tensor, target_frames: int) -> torch.Tensor:
        if x.shape[1] == target_frames:
            return x
        return F.interpolate(
            x.transpose(1, 2), size=target_frames, mode="linear",
            align_corners=False).transpose(1, 2)

    def _build_queries(self, batch_size: int, latent_frames: int,
                       device: torch.device) -> torch.Tensor:
        time_ids = torch.arange(latent_frames, device=device)
        time_embed = self.query_time_embedding(time_ids)[:, None, :]
        queries = time_embed + self.part_embedding[None, :, :]
        return queries.reshape(1, latent_frames * len(self.part_names), -1).expand(
            batch_size, -1, -1)

    def forward(
        self,
        features: torch.Tensor,
        grid_size: Tuple[int, int, int],
        target_frames: Optional[int] = None,
        valid_tokens: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        latent_frames, grid_h, grid_w = grid_size
        if min(latent_frames, grid_h, grid_w) < 1:
            raise ValueError("All grid_size entries must be positive")
        if latent_frames > self.config.max_latent_frames:
            raise ValueError(
                f"latent_frames={latent_frames} exceeds max_latent_frames="
                f"{self.config.max_latent_frames}")
        expected_tokens = latent_frames * grid_h * grid_w
        valid_tokens = expected_tokens if valid_tokens is None else valid_tokens
        if valid_tokens != expected_tokens:
            raise ValueError(
                f"valid_tokens={valid_tokens} does not match grid product "
                f"{expected_tokens}")
        if features.ndim != 3 or features.shape[1] < valid_tokens:
            raise ValueError(
                "features must be [B, L, C] with L >= valid_tokens; got "
                f"{tuple(features.shape)}")

        memory = self.input_projection(features[:, :valid_tokens])
        time_ids = torch.arange(latent_frames, device=features.device)
        source_time = self.source_time_embedding(time_ids)
        source_time = source_time[:, None, :].expand(
            latent_frames, grid_h * grid_w, -1).reshape(1, valid_tokens, -1)
        memory = memory + source_time
        spatial_positions = spatial_sincos_position_encoding(
            grid_h, grid_w, self.config.hidden_dim, features.device
        ).to(dtype=memory.dtype)
        spatial_positions = spatial_positions[None, :, :].expand(
            latent_frames, -1, -1
        ).reshape(1, valid_tokens, self.config.hidden_dim)
        memory = memory + self.spatial_position_scale * spatial_positions

        queries = self._build_queries(features.shape[0], latent_frames,
                                      features.device)
        for layer in self.layers:
            queries = layer(queries, memory)
        queries = self.output_norm(queries).reshape(
            features.shape[0], latent_frames, len(self.part_names), -1)

        target_frames = latent_frames if target_frames is None else target_frames
        if target_frames < 1:
            raise ValueError("target_frames must be positive")
        body_queries = self._resize_time(queries[:, :, 0], target_frames)
        body = self.body_head(body_queries)
        residual_6d = self.rotation_residual_head(body_queries).reshape(
            features.shape[0], target_frames, 24, 6
        )
        left_hand = self.left_hand_head(
            self._resize_time(queries[:, :, 1], target_frames))
        right_hand = self.right_hand_head(
            self._resize_time(queries[:, :, 2], target_frames))
        face = self.face_head(self._resize_time(queries[:, :, 3], target_frames))
        object_state = self.object_head(
            self._resize_time(queries[:, :, 4], target_frames))

        cfg = self.config
        body_pose_end = 3 + cfg.body_pose_dim
        with torch.autocast(device_type=features.device.type, enabled=False):
            # The legacy head models root + 21 body joints. Only its base has
            # two padded identities; the learned residual covers all 24 joints.
            base_axis_angle = torch.cat(
                (body[..., :body_pose_end].float(), body.new_zeros(
                    (*body.shape[:-1], 6), dtype=torch.float32
                )), dim=-1
            ).reshape(features.shape[0], target_frames, 24, 3)
            base_rotmat = axis_angle_to_matrix(base_axis_angle)
            residual_rotmat = rotation_6d_to_matrix(residual_6d)
            rotmat = residual_rotmat @ base_rotmat
        return {
            "global_orient": body[..., :3],
            "body_pose": body[..., 3:body_pose_end],
            "transl": body[..., body_pose_end:body_pose_end + 3],
            "left_hand_pose": left_hand,
            "right_hand_pose": right_hand,
            "jaw_pose": face[..., :3],
            "expression": face[..., 3:],
            "betas": self.shape_head(memory.mean(dim=1)),
            "object_state": object_state,
            "latent_queries": queries,
            "rotmat": rotmat,
        }


def temporal_consistency_loss(prediction: torch.Tensor) -> torch.Tensor:
    """First- and second-order temporal smoothness for any sequence tensor."""
    if prediction.shape[1] < 2:
        return prediction.new_zeros(())
    velocity = prediction[:, 1:] - prediction[:, :-1]
    loss = velocity.abs().mean()
    if prediction.shape[1] >= 3:
        acceleration = velocity[:, 1:] - velocity[:, :-1]
        loss = loss + acceleration.abs().mean()
    return loss
