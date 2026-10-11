"""Add a mask-aware RGB/SMPL temporal residual to the unchanged v9 decoder.

This file was copied from v10 for bounded limb-only refinement.  Old RGB, VACE,
geometry and camera paths remain inherited from v9; the new parameters all
live under ``temporal_refiner``.  Zero output heads reproduce any complete v9
checkpoint exactly at initialization, including checkpoints with trained
geometry and camera heads.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn

from models.human_decoder_rgb_vace_v9_isolated import (
    HMR2A_SHA256,
    RGBVaceHMR2AConfig,
    HumanRGBVaceGeometryTransformerV9Isolated,
)


__all__ = [
    "HMR2A_SHA256",
    "RGBVaceHMR2AConfig",
    "TemporalRefinerConfig",
    "HumanRGBVaceGeometryTransformerV12Bounded",
]


@dataclass(frozen=True)
class TemporalRefinerConfig:
    """Architecture of the independent v12 temporal residual branch."""

    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.1
    max_angle_degrees: float = 5.0

    def validate(self) -> None:
        if not 0 < self.max_angle_degrees <= 45:
            raise ValueError('max_angle_degrees must be in (0, 45]')
        for name in ("hidden_dim", "num_heads", "num_layers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Temporal {name} must be a positive integer")
        if self.hidden_dim % self.num_heads:
            raise ValueError("Temporal hidden_dim must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("Temporal dropout must lie in [0, 1)")


class _RGBSMPLTemporalRefiner(nn.Module):
    """Attend over time within each person/window, never across the batch."""

    def __init__(
        self, geometry_config: RGBVaceHMR2AConfig, config: TemporalRefinerConfig
    ) -> None:
        super().__init__()
        self.num_betas = geometry_config.num_betas
        self.max_frames = geometry_config.max_rgb_frames
        geometry_dim = 24 * 6 + self.num_betas
        self.rgb_projection = nn.Sequential(
            nn.LayerNorm(geometry_config.hidden_dim),
            nn.Linear(geometry_config.hidden_dim, config.hidden_dim),
        )
        self.geometry_projection = nn.Sequential(
            nn.LayerNorm(geometry_dim),
            nn.Linear(geometry_dim, config.hidden_dim),
        )
        # Keep the magnitude of inter-frame motion; do not normalize a nearly
        # stationary frame difference into a unit-variance motion feature.
        self.motion_projection = nn.Linear(geometry_dim, config.hidden_dim)
        self.position_embedding = nn.Embedding(self.max_frames, config.hidden_dim)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        # Construct layers independently instead of cloning identical weights.
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=4 * config.hidden_dim,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(config.num_layers)
        )
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.max_angle = config.max_angle_degrees * torch.pi / 180
        self.rotation_delta_head = nn.Linear(config.hidden_dim, 23 * 3)
        for head in (self.rotation_delta_head,):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        rgb_queries: torch.Tensor,
        rotmat: torch.Tensor,
        betas: torch.Tensor,
        frame_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, frames, _ = rgb_queries.shape
        # The new branch, small residual heads, and rotation composition all
        # stay FP32, even when the inherited v9 feature path uses BF16 AMP.
        with torch.autocast(device_type=rgb_queries.device.type, enabled=False):
            rgb = rgb_queries.float().masked_fill(~frame_valid[..., None], 0.0)
            pose_6d = torch.cat(
                (rotmat.float()[..., :, 0], rotmat.float()[..., :, 1]), dim=-1
            ).reshape(batch, frames, 24 * 6)
            geometry = torch.cat((pose_6d, betas.float()), dim=-1)
            geometry = geometry.masked_fill(~frame_valid[..., None], 0.0)
            motion = torch.zeros_like(geometry)
            consecutive_valid = frame_valid[:, 1:] & frame_valid[:, :-1]
            motion[:, 1:] = (geometry[:, 1:] - geometry[:, :-1]).masked_fill(
                ~consecutive_valid[..., None], 0.0
            )
            positions = torch.arange(frames, device=rgb_queries.device)
            queries = (
                self.rgb_projection(rgb)
                + self.geometry_projection(geometry)
                + self.motion_projection(motion)
                + self.position_embedding(positions)[None]
            )
            # Mask values as well as attention keys.  In particular, a padded
            # frame cannot affect the difference feature of its next frame.
            queries = queries.masked_fill(~frame_valid[..., None], 0.0)
            for block in self.blocks:
                queries = block(queries, src_key_padding_mask=~frame_valid)
                queries = queries.masked_fill(~frame_valid[..., None], 0.0)
            queries = self.output_norm(queries)
            queries = queries.masked_fill(~frame_valid[..., None], 0.0)
            raw = self.rotation_delta_head(queries).reshape(batch, frames, 23, 3)
            # Smooth radial bound, with a finite nonzero derivative at zero.
            limb_delta = self.max_angle * raw / torch.sqrt(1 + raw.square().sum(-1, keepdim=True))
            rotation_delta = torch.cat((torch.zeros_like(limb_delta[:, :, :1]), limb_delta), dim=2)
            beta_delta = torch.zeros_like(betas.float())
            rotation_delta = rotation_delta.masked_fill(
                ~frame_valid[..., None, None], 0.0
            )
            beta_delta = beta_delta.masked_fill(~frame_valid[..., None], 0.0)
        return rotation_delta, beta_delta, queries


class HumanRGBVaceGeometryTransformerV12Bounded(
    HumanRGBVaceGeometryTransformerV9Isolated
):
    """Refine only v9 limb rotations with bounded RGB/SMPL context.

    ``frame_valid`` is an optional Boolean ``[B,T]`` tensor.  Every window
    must contain at least one valid frame.  Padding cannot be an attention
    key, and receives zero temporal residuals.  The v9 camera and its
    ``camera_only_prediction`` detached-feature route are inherited intact.
    """

    def __init__(
        self,
        config: Optional[RGBVaceHMR2AConfig] = None,
        temporal_config: Optional[TemporalRefinerConfig] = None,
    ) -> None:
        super().__init__(config)
        self.temporal_config = temporal_config or TemporalRefinerConfig()
        self.temporal_config.validate()
        self.temporal_refiner = _RGBSMPLTemporalRefiner(
            self.config, self.temporal_config
        )

    def _validated_frame_mask(
        self, rgb_tokens: torch.Tensor, frame_valid: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if rgb_tokens.ndim != 4:
            raise ValueError("rgb_tokens must be [B,T,N,C]")
        batch, frames = rgb_tokens.shape[:2]
        if frame_valid is None:
            frame_valid = torch.ones(
                batch, frames, dtype=torch.bool, device=rgb_tokens.device
            )
        if not isinstance(frame_valid, torch.Tensor):
            raise TypeError("frame_valid must be a Boolean tensor")
        if frame_valid.shape != (batch, frames):
            raise ValueError("frame_valid must be [B,T]")
        if frame_valid.dtype != torch.bool:
            raise TypeError("frame_valid must have dtype torch.bool")
        frame_valid = frame_valid.to(device=rgb_tokens.device)
        if not bool(frame_valid.any(dim=1).all()):
            raise ValueError("Every temporal window needs at least one valid frame")
        return frame_valid

    def forward(
        self,
        rgb_tokens: torch.Tensor,
        vace_features: torch.Tensor,
        base_rotmat: torch.Tensor,
        base_betas: torch.Tensor,
        base_camera: torch.Tensor,
        frame_valid: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        frame_valid = self._validated_frame_mask(rgb_tokens, frame_valid)
        prediction = super().forward(
            rgb_tokens, vace_features, base_rotmat, base_betas, base_camera
        )
        rotation_delta, beta_delta, temporal_queries = self.temporal_refiner(
            prediction["rgb_queries"],
            prediction["rotmat"],
            prediction["betas"],
            frame_valid,
        )
        with torch.autocast(device_type=rgb_tokens.device.type, enabled=False):
            x, y, z = rotation_delta.unbind(-1)
            zero = torch.zeros_like(x)
            skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*x.shape, 3, 3)
            rotation_residual = torch.matrix_exp(skew)
            old_rotmat = prediction["rotmat"].float()
            identity_matrix = torch.eye(3, device=old_rotmat.device)
            # This is R_delta @ R_v9, written as an additive residual so zero
            # heads preserve v9 elementwise without re-multiplying it by I.
            prediction["rotmat"] = old_rotmat + (
                rotation_residual - identity_matrix
            ) @ old_rotmat
            prediction["betas"] = prediction["betas"].float() + beta_delta.float()
        prediction.update(
            temporal_rotation_delta_axis_angle=rotation_delta,
            temporal_beta_delta=beta_delta,
            temporal_queries=temporal_queries,
        )
        return prediction

    def load_compatible_state_dict(self, state: Mapping[str, torch.Tensor]):
        """Load a complete v12 or complete v9 state, rejecting partial models.

        A v9 warm start may omit *all and only* ``temporal_refiner.*`` keys.
        Once any temporal parameter is present, the checkpoint must contain
        the complete v12 state.  Validate names and shapes before writing
        parameters, so a malformed checkpoint does not partially load.
        """
        if not isinstance(state, Mapping):
            raise TypeError("Model state must be a mapping of parameter tensors")
        expected = self.state_dict()
        temporal_keys = {key for key in expected if key.startswith("temporal_refiner.")}
        has_temporal = any(key.startswith("temporal_refiner.") for key in state)
        required = set(expected) if has_temporal else set(expected) - temporal_keys
        supplied = set(state)
        missing, extra = sorted(required - supplied), sorted(supplied - required)
        if missing or extra:
            version = "v12" if has_temporal else "v9"
            raise RuntimeError(
                f"Expected a complete {version} model state; "
                f"missing keys: {missing}; unexpected keys: {extra}"
            )
        malformed = [
            key for key in sorted(required)
            if not isinstance(state[key], torch.Tensor)
            or state[key].shape != expected[key].shape
        ]
        if malformed:
            raise RuntimeError(f"Checkpoint tensors have incompatible shapes: {malformed}")
        return self.load_state_dict(state, strict=has_temporal)

    def anchor_metadata(self) -> dict[str, object]:
        metadata = super().anchor_metadata()
        metadata.update(
            model_format="divibody-rgb-vace-v12-bounded",
            temporal_config=asdict(self.temporal_config),
            v9_preserving_initialization=True,
        )
        return metadata
