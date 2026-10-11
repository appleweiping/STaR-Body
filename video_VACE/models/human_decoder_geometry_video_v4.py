"""Continuous-video geometry decoder derived from geometry-v3.

The geometry-v3 source remains unchanged.  This additive model preserves every
v3 parameter name for strict warm starts, but replaces latent-time interpolation
with eight explicit RGB-frame queries.  Each query carries output-order, Wan
temporal-group, within-group offset, and absolute context-position identities
before attending all spatiotemporal VACE tokens.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from models.human_decoder_geometry import (
    GeometryDecoderConfig,
    HumanGeometryQueryTransformer,
    axis_angle_to_matrix,
    rotation_6d_to_matrix,
    spatial_sincos_position_encoding,
    temporal_consistency_loss,
)


__all__ = [
    "CENTER_RGB_POSITIONS",
    "GeometryDecoderConfig",
    "HumanGeometryQueryTransformerVideoV4",
    "HumanGeometryVideoQueryTransformer",
    "axis_angle_to_matrix",
    "rgb_frame_group_offset",
    "rotation_6d_to_matrix",
    "spatial_sincos_position_encoding",
    "temporal_consistency_loss",
]


CENTER_RGB_POSITIONS = tuple(range(4, 12))
RGB_OFFSETS_PER_LATENT = 4


def rgb_frame_group_offset(
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map RGB context positions to Wan latent groups and offsets.

    Wan's first latent represents RGB frame zero.  Every later latent represents
    the next group of four RGB frames, so positions ``1..4`` map to group one,
    positions ``5..8`` to group two, and so on.  Offsets are zero-based within
    each four-frame group; the special first frame uses offset zero.
    """

    if positions.ndim != 1:
        raise ValueError("positions must be a one-dimensional tensor")
    if positions.dtype == torch.bool or positions.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError("positions must use an integer dtype")
    if positions.numel() < 1:
        raise ValueError("positions cannot be empty")
    if bool(torch.any(positions < 0)):
        raise ValueError("positions cannot be negative")

    groups = torch.where(
        positions == 0,
        torch.zeros_like(positions),
        torch.div(
            positions + RGB_OFFSETS_PER_LATENT - 1,
            RGB_OFFSETS_PER_LATENT,
            rounding_mode="floor",
        ),
    )
    offsets = torch.where(
        positions == 0,
        torch.zeros_like(positions),
        torch.remainder(positions - 1, RGB_OFFSETS_PER_LATENT),
    )
    return groups, offsets


class HumanGeometryVideoQueryTransformer(HumanGeometryQueryTransformer):
    """Geometry-v4 decoder for the center eight frames of a Wan RGB context.

    ``features`` uses the flattened VACE layout ``[B,F*H*W,C]`` and
    ``grid_size=(F,H,W)``.  For the formal T17 path, ``F=5``,
    ``context_rgb_frames=17``, and the model directly emits context positions
    ``4..11``.  No latent-time interpolation is used.
    """

    output_rgb_frames = len(CENTER_RGB_POSITIONS)

    def __init__(self, config: Optional[GeometryDecoderConfig] = None) -> None:
        super().__init__(config=config)
        cfg = self.config
        if cfg.max_latent_frames < self.output_rgb_frames:
            raise ValueError(
                "max_latent_frames must provide at least eight existing "
                "query_time_embedding rows for geometry-v4 output order"
            )

        # Existing query_time_embedding is deliberately reused for output order.
        # All additional positional identities start at zero so a v3 warm start
        # changes only the query schedule, not its additive feature magnitude.
        self.rgb_group_embedding = nn.Embedding(
            cfg.max_latent_frames, cfg.hidden_dim
        )
        self.rgb_offset_embedding = nn.Embedding(
            RGB_OFFSETS_PER_LATENT, cfg.hidden_dim
        )
        self.rgb_context_position_embedding = nn.Embedding(
            RGB_OFFSETS_PER_LATENT * cfg.max_latent_frames - 3,
            cfg.hidden_dim,
        )
        nn.init.zeros_(self.rgb_group_embedding.weight)
        nn.init.zeros_(self.rgb_offset_embedding.weight)
        nn.init.zeros_(self.rgb_context_position_embedding.weight)

    @property
    def _geometry_v4_parameter_names(self) -> set[str]:
        return {
            "rgb_group_embedding.weight",
            "rgb_offset_embedding.weight",
            "rgb_context_position_embedding.weight",
        }

    def load_geometry_v3_weights(
        self, state_dict: Mapping[str, torch.Tensor]
    ) -> Dict[str, list[str]]:
        """Atomically load geometry-v3 weights, allowing only v4 keys missing.

        Validation completes before this model is changed.  ``load_state_dict``
        copies tensor values, so neither the supplied mapping nor its tensors are
        mutated or aliased by a successful load.
        """

        current = self.state_dict()
        incoming_keys = set(state_dict)
        current_keys = set(current)
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
        disallowed_missing = missing - self._geometry_v4_parameter_names
        if disallowed_missing or unexpected or mismatched:
            raise RuntimeError(
                "Incompatible geometry-v3 state_dict: "
                f"missing={sorted(disallowed_missing)}, "
                f"unexpected={sorted(unexpected)}, mismatched={mismatched}"
            )

        loaded = self.load_state_dict(state_dict, strict=False)
        loaded_missing = set(loaded.missing_keys)
        if loaded_missing - self._geometry_v4_parameter_names:
            raise RuntimeError(
                "Internal geometry-v4 loader error: unexpected missing keys "
                f"{sorted(loaded_missing)}"
            )
        return {
            "loaded": sorted(incoming_keys),
            "missing": sorted(loaded.missing_keys),
        }

    def _validate_target_positions(
        self,
        target_positions: Optional[Sequence[int] | torch.Tensor],
        context_rgb_frames: int,
        device: torch.device,
    ) -> torch.Tensor:
        if target_positions is None:
            positions = torch.tensor(
                CENTER_RGB_POSITIONS, dtype=torch.long, device=device
            )
        elif isinstance(target_positions, torch.Tensor):
            if target_positions.ndim != 1:
                raise ValueError("target_positions must be one-dimensional")
            if target_positions.dtype == torch.bool or target_positions.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            ):
                raise TypeError("target_positions must use an integer dtype")
            positions = target_positions.to(device=device, dtype=torch.long)
        else:
            raw_positions = tuple(target_positions)
            if any(isinstance(value, bool) or not isinstance(value, int)
                   for value in raw_positions):
                raise TypeError("target_positions must contain only integers")
            positions = torch.tensor(
                raw_positions, dtype=torch.long, device=device
            )

        if positions.numel() != self.output_rgb_frames:
            raise ValueError(
                f"target_positions must contain exactly {self.output_rgb_frames} "
                "RGB frames"
            )
        if bool(torch.any(positions[1:] != positions[:-1] + 1)):
            raise ValueError("target_positions must be strictly consecutive")
        if int(positions[0]) < 0 or int(positions[-1]) >= context_rgb_frames:
            raise ValueError(
                "target_positions must lie inside the RGB context range "
                f"[0, {context_rgb_frames})"
            )
        return positions

    def _build_rgb_queries(
        self,
        batch_size: int,
        positions: torch.Tensor,
        latent_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        group_ids, offset_ids = rgb_frame_group_offset(positions)
        if int(group_ids.max()) >= latent_frames:
            raise ValueError(
                "target_positions map beyond the available Wan latent groups"
            )

        output_order = torch.arange(
            self.output_rgb_frames, device=positions.device
        )
        frame_queries = (
            self.query_time_embedding(output_order)
            + self.rgb_group_embedding(group_ids)
            + self.rgb_offset_embedding(offset_ids)
            + self.rgb_context_position_embedding(positions)
        )
        queries = frame_queries[:, None, :] + self.part_embedding[None, :, :]
        queries = queries.reshape(
            1, self.output_rgb_frames * len(self.part_names), -1
        ).expand(batch_size, -1, -1)
        return queries, group_ids, offset_ids

    def forward(
        self,
        features: torch.Tensor,
        grid_size: Tuple[int, int, int],
        context_rgb_frames: int,
        target_positions: Optional[Sequence[int] | torch.Tensor] = None,
        valid_tokens: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        latent_frames, grid_h, grid_w = grid_size
        if min(latent_frames, grid_h, grid_w) < 1:
            raise ValueError("All grid_size entries must be positive")
        if latent_frames > self.config.max_latent_frames:
            raise ValueError(
                f"latent_frames={latent_frames} exceeds max_latent_frames="
                f"{self.config.max_latent_frames}"
            )
        if (
            isinstance(context_rgb_frames, bool)
            or not isinstance(context_rgb_frames, int)
        ):
            raise TypeError("context_rgb_frames must be an integer")
        expected_context_frames = (
            RGB_OFFSETS_PER_LATENT * latent_frames - 3
        )
        if context_rgb_frames != expected_context_frames:
            raise ValueError(
                f"context_rgb_frames={context_rgb_frames} does not match "
                f"4*latent_frames-3={expected_context_frames}"
            )

        expected_tokens = latent_frames * grid_h * grid_w
        valid_tokens = expected_tokens if valid_tokens is None else valid_tokens
        if valid_tokens != expected_tokens:
            raise ValueError(
                f"valid_tokens={valid_tokens} does not match grid product "
                f"{expected_tokens}"
            )
        if features.ndim != 3 or features.shape[1] < valid_tokens:
            raise ValueError(
                "features must be [B, L, C] with L >= valid_tokens; got "
                f"{tuple(features.shape)}"
            )
        positions = self._validate_target_positions(
            target_positions, context_rgb_frames, features.device
        )

        memory = self.input_projection(features[:, :valid_tokens])
        time_ids = torch.arange(latent_frames, device=features.device)
        source_time = self.source_time_embedding(time_ids)
        source_time = source_time[:, None, :].expand(
            latent_frames, grid_h * grid_w, -1
        ).reshape(1, valid_tokens, -1)
        memory = memory + source_time
        spatial_positions = spatial_sincos_position_encoding(
            grid_h, grid_w, self.config.hidden_dim, features.device
        ).to(dtype=memory.dtype)
        spatial_positions = spatial_positions[None, :, :].expand(
            latent_frames, -1, -1
        ).reshape(1, valid_tokens, self.config.hidden_dim)
        memory = memory + self.spatial_position_scale * spatial_positions

        queries, _, _ = self._build_rgb_queries(
            features.shape[0], positions, latent_frames
        )
        for layer in self.layers:
            queries = layer(queries, memory)
        queries = self.output_norm(queries).reshape(
            features.shape[0], self.output_rgb_frames, len(self.part_names), -1
        )

        body_queries = queries[:, :, 0]
        body = self.body_head(body_queries)
        residual_6d = self.rotation_residual_head(body_queries).reshape(
            features.shape[0], self.output_rgb_frames, 24, 6
        )
        left_hand = self.left_hand_head(queries[:, :, 1])
        right_hand = self.right_hand_head(queries[:, :, 2])
        face = self.face_head(queries[:, :, 3])
        object_state = self.object_head(queries[:, :, 4])

        cfg = self.config
        body_pose_end = 3 + cfg.body_pose_dim
        with torch.autocast(device_type=features.device.type, enabled=False):
            base_axis_angle = torch.cat(
                (
                    body[..., :body_pose_end].float(),
                    body.new_zeros(
                        (*body.shape[:-1], 6), dtype=torch.float32
                    ),
                ),
                dim=-1,
            ).reshape(features.shape[0], self.output_rgb_frames, 24, 3)
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


# Descriptive alias for callers that encode the version in the class name.
HumanGeometryQueryTransformerVideoV4 = HumanGeometryVideoQueryTransformer
