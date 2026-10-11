"""Hierarchical whole-body decoder for frozen VACE DiT features.

This research module is intentionally separate from the original VACE model.
It consumes full intermediate token sequences and predicts temporally dense
SMPL-X parameters with part-specific learned queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HumanDecoderConfig:
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


class HumanQueryTransformer(nn.Module):
    """Decode VACE DiT tokens into structured, time-varying SMPL-X states.

    Input tokens use shape ``[batch, tokens, input_dim]``. ``grid_size`` must
    describe the unpadded DiT token grid as ``(latent_time, height, width)``.
    Predictions are returned at ``target_frames`` through temporal interpolation.
    """

    part_names = ("body", "left_hand", "right_hand", "face", "object")

    def __init__(self, config: Optional[HumanDecoderConfig] = None) -> None:
        super().__init__()
        self.config = config or HumanDecoderConfig()
        cfg = self.config

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

        queries = self._build_queries(features.shape[0], latent_frames,
                                      features.device)
        for layer in self.layers:
            queries = layer(queries, memory)
        queries = self.output_norm(queries).reshape(
            features.shape[0], latent_frames, len(self.part_names), -1)

        target_frames = target_frames or latent_frames
        body = self.body_head(self._resize_time(queries[:, :, 0], target_frames))
        left_hand = self.left_hand_head(
            self._resize_time(queries[:, :, 1], target_frames))
        right_hand = self.right_hand_head(
            self._resize_time(queries[:, :, 2], target_frames))
        face = self.face_head(self._resize_time(queries[:, :, 3], target_frames))
        object_state = self.object_head(
            self._resize_time(queries[:, :, 4], target_frames))

        cfg = self.config
        body_pose_end = 3 + cfg.body_pose_dim
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
