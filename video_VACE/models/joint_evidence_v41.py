"""Projected-joint-conditioned evidence on the real HMR2 16 x 12 token grid.

Coordinates are relative to the square 256 crop: x/y in [-.5, .5]. HMR2
tokens cover its central 192-pixel-wide crop, not the entire square. Multiple
sampling radii add context, not image resolution. This module predicts image
evidence only; a separate SMPL/camera refiner must consume it for evaluation.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class JointEvidenceConfig:
    hidden_dim: int = 128
    num_heads: int = 4
    grid_height: int = 16
    grid_width: int = 12
    search_radii: tuple = (1., 2., 4.)
    temporal_enabled: bool = True

    def validate(self):
        if self.hidden_dim < 4 or self.num_heads < 1 or self.hidden_dim % self.num_heads:
            raise ValueError('hidden_dim must be positive and divisible by num_heads')
        if min(self.grid_height, self.grid_width) < 1:
            raise ValueError('Spatial grid dimensions must be positive')
        if not self.search_radii or any(not math.isfinite(r) or r <= 0 for r in self.search_radii):
            raise ValueError('search_radii must contain positive finite values')


class JointEvidenceRefiner(nn.Module):
    def __init__(self, rgb_width, cfg=None):
        super().__init__()
        self.cfg = cfg or JointEvidenceConfig()
        self.cfg.validate()
        if rgb_width < 1:
            raise ValueError('rgb_width must be positive')
        self.rgb_width = rgb_width
        d, h, w = self.cfg.hidden_dim, self.cfg.grid_height, self.cfg.grid_width
        self.rgb_projection = nn.Sequential(nn.LayerNorm(rgb_width), nn.Linear(rgb_width, d))
        self.position = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.joint_queries = nn.Parameter(torch.randn(17, d) * .02)
        self.query_position = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        count = 9 * len(self.cfg.search_radii)
        self.local_projection = nn.Linear(count * (d + 1), d)
        self.query = nn.Linear(d, d)
        self.key = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.feature_mlp = nn.Sequential(nn.Linear(d, 2*d), nn.GELU(), nn.Linear(2*d, d))
        self.fine_offset = nn.Linear(d, 2)
        self.reliability = nn.Linear(d, 1)
        self.prior_strength = nn.Parameter(torch.tensor(0.))
        self.temporal_qkv = nn.Linear(d, 3*d)
        self.temporal_output = nn.Linear(d, d)
        self.time_bias = nn.Sequential(nn.Linear(2, 16), nn.GELU(), nn.Linear(16, self.cfg.num_heads))
        self.temporal_gate = nn.Linear(2*d + 1, 1)
        self.temporal_offset = nn.Linear(d, 2)
        nn.init.zeros_(self.fine_offset.weight)
        nn.init.zeros_(self.fine_offset.bias)
        nn.init.zeros_(self.temporal_offset.weight)
        nn.init.zeros_(self.temporal_offset.bias)
        # These are centers of actual HMR2 patch cells in square-crop coordinates.
        y = (torch.arange(h, dtype=torch.float32) + .5) / h - .5
        x = .75 * ((torch.arange(w, dtype=torch.float32) + .5) / w - .5)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        self.register_buffer('square_coordinates', torch.stack((xx, yy), -1).reshape(h*w, 2))
        offsets = [(radius*x*2/w, radius*y*2/h)
                   for radius in self.cfg.search_radii for y in (-1, 0, 1) for x in (-1, 0, 1)]
        self.register_buffer('sampling_offsets', torch.tensor(offsets, dtype=torch.float32))

    def _check_inputs(self, tokens, uv, valid, independent, source):
        cfg = self.cfg
        if tokens.ndim != 4 or tokens.shape[2:] != (cfg.grid_height*cfg.grid_width, self.rgb_width):
            raise ValueError('rgb_tokens must be [B,T,grid_height*grid_width,rgb_width]')
        b, t = tokens.shape[:2]
        if min(b, t) < 1 or uv.shape != (b, t, 17, 2):
            raise ValueError('projected_uv must be [B,T,17,2], with nonempty B/T')
        if valid.shape != (b, t) or independent.shape != (b, t):
            raise ValueError('frame_valid and independent must be [B,T]')
        if valid.dtype != torch.bool or independent.dtype != torch.bool:
            raise TypeError('frame_valid and independent must be Boolean masks')
        values = (uv, valid, independent) if source is None else (uv, valid, independent, source)
        if any(value.device != tokens.device for value in values):
            raise ValueError('All inputs must use the same device')
        if not tokens.is_floating_point() or not uv.is_floating_point():
            raise TypeError('RGB features and projected coordinates must be floating point')
        if not torch.isfinite(tokens[valid]).all() or not torch.isfinite(uv[valid]).all():
            raise ValueError('Valid frames must contain finite features and coordinates')
        if source is not None and (source.shape != (b, t) or source.dtype not in (torch.int32, torch.int64)):
            raise ValueError('source_frame_ids must be integer [B,T]')

    @staticmethod
    def _independent_mask(valid, independent, source):
        # A repeated source never contributes another key, even if mislabeled independent.
        keys = valid & independent
        if source is not None:
            keys = keys.clone()
            for i in range(1, keys.shape[1]):
                repeated = ((source[:, :i] == source[:, i, None]) & keys[:, :i]).any(-1)
                keys[:, i] &= ~repeated
        return keys

    def _local_features(self, memory, uv, valid):
        b, t, _, d = memory.shape
        cfg = self.cfg
        maps = memory.transpose(-1, -2).reshape(b*t, d, cfg.grid_height, cfg.grid_width)
        grid = uv.float() * uv.new_tensor([8./3., 2.], dtype=torch.float32)
        locations = grid[..., None, :] + self.sampling_offsets
        inside = (locations.abs() <= 1).all(-1) & valid[..., None, None]
        samples = F.grid_sample(maps.float(), locations.reshape(b*t, 17, -1, 2),
                                mode='bilinear', padding_mode='zeros', align_corners=False)
        samples = samples.transpose(1, 2).transpose(2, 3).reshape(b, t, 17, -1, d)
        samples = samples * inside[..., None]
        sampled = torch.cat((samples, inside[..., None].float()), -1).flatten(-2)
        return self.local_projection(sampled.to(memory.dtype))

    def _temporal(self, features, raw_uv, projected, reliability, valid, keys, source):
        b, t, j, d = features.shape
        heads, width = self.cfg.num_heads, d // self.cfg.num_heads
        qkv = self.temporal_qkv(features).reshape(b, t, j, 3, heads, width)
        q, k, v = [qkv[:, :, :, i].permute(0, 2, 3, 1, 4).float() for i in range(3)]
        logits = q @ k.transpose(-1, -2) / math.sqrt(width)
        if source is not None:
            delta = source[:, :, None].double() - source[:, None, :].double()
            lag = torch.log1p(delta.abs()).float() / math.log(33.)
            timing = torch.stack((lag.clamp_max(8), delta.sign().float()), -1)
            bias = self.time_bias(timing.to(features.dtype)).permute(0, 3, 1, 2).float()
            logits = logits + bias[:, None]
        logits = logits + reliability.permute(0, 2, 3, 1).float().clamp_min(1e-6).log()[..., None, :]
        mask = keys[:, None, None, None, :]
        # Finite logits plus an explicitly renormalized mask handle all-empty rows.
        attention = logits.masked_fill(~mask, -1e4).softmax(-1) * mask.float()
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
        context = (attention @ v).permute(0, 3, 1, 2, 4).reshape(b, t, j, d)
        context = self.temporal_output(context.to(features.dtype))
        weights = attention.mean(2)
        # Transport observation residuals rather than smoothing moving absolute joints.
        residual = (raw_uv - projected).permute(0, 2, 1, 3)
        transported = (weights @ residual.float()).permute(0, 2, 1, 3)
        temporal_uv = projected.float() + transported
        temporal_uv = temporal_uv + .02 * self.temporal_offset(context).float().tanh()
        support = ((keys.sum(1) > 1)[:, None] & valid)[..., None, None].expand(b, t, j, 1)
        gate_input = torch.cat((features, context, reliability.to(features.dtype)), -1)
        gate = self.temporal_gate(gate_input).float().sigmoid() * (1 - reliability.float())
        blended = raw_uv + gate * (temporal_uv - raw_uv)
        observed = torch.where(support, blended, raw_uv)  # Exact single-frame fallback.
        contextual = torch.where(support, self.norm(features + context), features)
        return observed, contextual, support

    def forward(self, rgb_tokens, projected_uv, frame_valid, independent, source_frame_ids=None):
        self._check_inputs(rgb_tokens, projected_uv, frame_valid, independent, source_frame_ids)
        cfg = self.cfg
        b, t = frame_valid.shape
        # Invalid padding may contain arbitrary values or NaNs; sanitize before any layer.
        safe_tokens = torch.where(frame_valid[..., None, None], rgb_tokens, torch.zeros_like(rgb_tokens))
        projected = torch.where(frame_valid[..., None, None], projected_uv, torch.zeros_like(projected_uv))
        memory = self.rgb_projection(safe_tokens)
        coordinates = self.square_coordinates.to(device=memory.device)
        keys = self._independent_mask(frame_valid, independent, source_frame_ids)
        local = self._local_features(memory, projected, frame_valid)
        query = self.joint_queries + self.query_position(projected.to(memory.dtype)) + local
        positioned = memory + self.position(coordinates.to(memory.dtype))[None, None]
        logits = torch.einsum('btjd,btnd->btjn', self.query(query).float(), self.key(positioned).float())
        logits = logits / math.sqrt(cfg.hidden_dim)
        distance = (coordinates[None, None, None] - projected.float()[..., None, :]).square().sum(-1)
        logits = logits - F.softplus(self.prior_strength.float()) * distance / (.15 ** 2)
        probability = logits.softmax(-1)
        attended = torch.einsum('btjn,btnd->btjd', probability.to(memory.dtype), memory)
        features = self.norm(query + attended)
        features = features + self.feature_mlp(features)
        raw_uv = torch.einsum('btjn,nc->btjc', probability, coordinates)
        cell_bound = projected.new_tensor([.375/cfg.grid_width, .5/cfg.grid_height])
        raw_uv = raw_uv + self.fine_offset(features).float().tanh() * cell_bound
        entropy = -(probability * probability.clamp_min(1e-8).log()).sum(-1, keepdim=True)
        entropy = entropy / max(math.log(cfg.grid_height*cfg.grid_width), 1.)
        reliability = self.reliability(features).float().sigmoid() * (1 - .5*entropy)
        valid_joint = frame_valid[..., None, None]
        raw_uv = torch.where(valid_joint, raw_uv, torch.zeros_like(raw_uv))
        features = torch.where(valid_joint, features, torch.zeros_like(features))
        reliability = torch.where(valid_joint, reliability, torch.zeros_like(reliability))
        if cfg.temporal_enabled:
            observed, features, support = self._temporal(features, raw_uv, projected, reliability,
                                                         frame_valid, keys, source_frame_ids)
        else:
            observed, support = raw_uv, torch.zeros_like(raw_uv[..., :1], dtype=torch.bool)
        observed = torch.where(valid_joint, observed, torch.zeros_like(observed))
        features = torch.where(valid_joint, features, torch.zeros_like(features))
        return dict(observed_uv=observed, raw_observed_uv=raw_uv,
                    confidence=reliability * keys[..., None, None], features=features,
                    spatial_logits=torch.where(valid_joint, logits, torch.zeros_like(logits)),
                    temporal_support=support)
