"""C11 identity-preserving evidence router; generalises joint_feedback_v37 (unchanged).

Each evidence token is one 2D landmark observation from a declared source:
  head      the inherited observation head (17 PoseTrack/COCO landmarks, as in C10);
  external  an external detector cached by scripts/cache_external_keypoints_v43.py
            (COCO-17, e.g. ViTPose). Detector outputs only; never ground truth.

Default configuration (sources=('head',), routing='anatomy', identity=True,
pooling='joint', init_seed=370026) creates exactly the parameters, buffers, names and
initial values of v37 ``JointFeedback`` and computes the same function, so native C10
checkpoints load strictly and reproduce C10 (see scripts/test_evidence_router_v43.py).

Ablation switches (each changes one factor and keeps the token features fixed):
  routing   'anatomy'  soft kinematic-tree bias -D(j, a(l))/2 (C10)
            'none'     no anatomical bias
            'shuffled' bias columns permuted by a fixed derangement (wrong anatomy,
                       identical bias statistics)
  identity  True: learned landmark (+schema) embeddings; False: content-only tokens, i.e.
            a permutation-invariant evidence set.
  pooling   'joint'    24 learned pose queries read the evidence set (C10)
            'mean'     capacity-matched mean control: masked mean of the same encoded
                       tokens, a 128-256-128 MLP (~= attention parameter count), the same
                       zero-initialised output, broadcast identically to all 24 joints.
            'concat'   ordered-vector control (WHAM/TRAM-style per-frame keypoint input):
                       the N encoded tokens (invalid ones zeroed) are concatenated in slot
                       order, mapped by Linear(N*W, W) + MLP, and broadcast to all joints.
                       Identity is kept by position, but nothing routes evidence to joints.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from models.joint_feedback_v37 import ANCHORS, anatomy_bias

SOURCES = ('head', 'external')
ROUTINGS = ('anatomy', 'none', 'shuffled')
POOLINGS = ('joint', 'mean', 'concat')
# Fixed derangement of the 17 landmark slots used by routing='shuffled' (no fixed point).
SHUFFLE = (9, 13, 5, 16, 11, 2, 15, 0, 12, 4, 14, 1, 8, 6, 3, 10, 7)
DEFAULT_INIT_SEED = 370026


@dataclass(frozen=True)
class RouterConfig:
    sources: tuple = ('head',)
    routing: str = 'anatomy'
    identity: bool = True
    pooling: str = 'joint'
    init_seed: int = DEFAULT_INIT_SEED
    external_min_confidence: float = 0.

    def __post_init__(self):
        sources = tuple(self.sources)
        object.__setattr__(self, 'sources', sources)
        if not sources or any(s not in SOURCES for s in sources) or len(set(sources)) != len(sources):
            raise ValueError(f'Invalid evidence sources {sources}')
        if self.routing not in ROUTINGS or self.pooling not in POOLINGS:
            raise ValueError('Invalid routing/pooling')
        if self.pooling in ('mean', 'concat') and (self.identity or self.routing != 'none'):
            raise ValueError(f"pooling='{self.pooling}' is a global control: use identity=False, routing='none'")
        if not 0. <= self.external_min_confidence < 1.:
            raise ValueError('external_min_confidence must be in [0,1)')

    @property
    def is_c10_default(self):
        return (self.sources == ('head',) and self.routing == 'anatomy' and self.identity
                and self.pooling == 'joint' and self.init_seed == DEFAULT_INIT_SEED)


def _bias_columns(schema, routing):
    if routing == 'none':
        return torch.zeros(24, 17)
    bias = anatomy_bias(schema)
    return bias[:, list(SHUFFLE)] if routing == 'shuffled' else bias


class EvidenceRouter(nn.Module):
    def __init__(self, width, cfg: RouterConfig):
        super().__init__()
        self.width, self.cfg = width, cfg
        # --- v37 parameters, created in v37 order (identical RNG consumption) ---
        self.token = nn.Sequential(nn.Linear(7, width), nn.SiLU(), nn.Linear(width, width))
        self.landmark = nn.Embedding(17, width)
        self.pose_joint = nn.Embedding(24, width)
        self.schema = nn.Embedding(2, width)
        self.attention = nn.MultiheadAttention(width, 4, dropout=0., batch_first=True)
        self.output = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.output.weight)
        self.register_buffer('bias_posetrack', anatomy_bias('posetrack'))
        self.register_buffer('bias_coco', anatomy_bias('coco'))
        # --- C11 additions; only created when used, after all v37 tensors ---
        if 'external' in cfg.sources:
            self.external_landmark = nn.Embedding(17, width)
        if cfg.routing != 'anatomy':
            self.register_buffer('route_posetrack', _bias_columns('posetrack', cfg.routing))
            self.register_buffer('route_coco', _bias_columns('coco', cfg.routing))
        if cfg.pooling == 'mean':
            self.pooled = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * width), nn.SiLU(),
                                        nn.Linear(2 * width, width))
        if cfg.pooling == 'concat':
            slots = 17 * len(cfg.sources)
            self.pooled = nn.Sequential(nn.Linear(slots * width, width), nn.SiLU(), nn.LayerNorm(width),
                                        nn.Linear(width, width))

    def _route(self, schema):
        if schema not in ANCHORS:
            raise ValueError('Unknown feedback landmark schema: ' + str(schema))
        name = ('bias_' if self.cfg.routing == 'anatomy' else 'route_') + schema
        return getattr(self, name)

    def _encode(self, source):
        """v37 feature construction for one 17-slot source; returns tokens [B*T,17,W], valid [B,T,17]."""
        residual, projected = source['residual'], source['projected']
        logvar, visibility, frame_valid = source['logvar'], source['visibility'], source['frame_valid']
        b, t, j, _ = residual.shape
        if j != 17 or projected.shape != residual.shape:
            raise ValueError('Expected 17 declared landmark coordinates')
        if logvar.shape != (b, t, 17, 1) or visibility.shape != (b, t, 17, 1) or frame_valid.shape != (b, t):
            raise ValueError('Feedback shape mismatch')
        finite = (torch.isfinite(residual).all(-1) & torch.isfinite(projected).all(-1)
                  & torch.isfinite(logvar).all(-1) & torch.isfinite(visibility).all(-1))
        valid = finite & frame_valid.bool()[..., None]
        if source.get('slot_valid') is not None:
            valid = valid & source['slot_valid'].bool()
        residual = torch.where(valid[..., None], residual, torch.zeros_like(residual)).clamp(-2, 2)
        projected = torch.where(valid[..., None], projected, torch.zeros_like(projected)).clamp(-4, 4)
        logvar = torch.where(valid[..., None], logvar, torch.zeros_like(logvar)).clamp(-6, 6)
        visibility = torch.where(valid[..., None], visibility, torch.zeros_like(visibility)).clamp(0, 1)
        features = torch.cat((residual, projected, logvar, visibility, valid[..., None].to(residual.dtype)), -1)
        tokens = self.token(features).reshape(b * t, 17, self.width)
        if self.cfg.identity:
            if source['name'] == 'head':
                identity = self.landmark.weight + self.schema.weight[0 if source['schema'] == 'posetrack' else 1]
            else:
                identity = self.external_landmark.weight + self.schema.weight[1]
            tokens = tokens + identity[None]
        return tokens, valid

    def forward(self, sources, *, batch_shape):
        """sources: list of dicts (name, schema, residual, projected, logvar, visibility,
        frame_valid, optional slot_valid), ordered as cfg.sources."""
        if [s['name'] for s in sources] != list(self.cfg.sources):
            raise ValueError('Evidence sources differ from the configured router')
        b, t = batch_shape
        encoded = [self._encode(s) for s in sources]
        tokens = torch.cat([e[0] for e in encoded], 1)
        valid = torch.cat([e[1] for e in encoded], -1)
        n = tokens.shape[1]
        flat_valid = valid.reshape(b * t, n)
        any_valid = flat_valid.any(-1)
        if self.cfg.pooling in ('mean', 'concat'):
            weight = flat_valid.to(tokens.dtype)[..., None]
            if self.cfg.pooling == 'mean':
                pooled = (tokens * weight).sum(1) / weight.sum(1).clamp_min(1.)
            else:
                pooled = (tokens * weight).reshape(b * t, n * self.width)
            result = self.output(self.pooled(pooled))[:, None].expand(-1, 24, -1).reshape(b, t, 24, self.width)
            return result * any_valid.reshape(b, t, 1, 1).to(result.dtype)
        queries = self.pose_joint.weight[None].expand(b * t, -1, -1)
        # Explicit finite fallback for an all-invalid frame, then exactly zero it (v37).
        safe_valid = flat_valid.clone()
        safe_valid[:, 0] |= ~any_valid
        bias = torch.cat([self._route(s['schema']) for s in sources], 1).to(tokens.dtype)
        mask = bias[None].expand(b * t, -1, -1).clone()
        mask = mask.masked_fill(~safe_valid[:, None, :], float('-inf'))
        mask = mask.repeat_interleave(4, 0)
        mixed = self.attention(queries, tokens, tokens, attn_mask=mask, need_weights=False)[0]
        result = self.output(mixed).reshape(b, t, 24, self.width)
        return result * any_valid.reshape(b, t, 1, 1).to(result.dtype)


def make_evidence_router(width, cfg: RouterConfig):
    """Isolate CPU initialization without changing any CUDA/global generator (as v37)."""
    with torch.random.fork_rng(devices=[]):
        generator = torch.Generator(device='cpu').manual_seed(int(cfg.init_seed))
        torch.random.set_rng_state(generator.get_state())
        return EvidenceRouter(width, cfg)


def external_evidence_features(keypoints, projected, frame_valid, min_confidence=0.):
    """Detector keypoints [B,T,17,3] (u, v, conf) -> router source tensors.

    Confidence c is mapped to a log-variance-like feature -2 log c (c=1 -> 0) and a
    visibility feature c; a slot is valid only if finite and c > min_confidence.
    """
    if keypoints.shape[-2:] != (17, 3):
        raise ValueError('External evidence must be [B,T,17,3]')
    uv = keypoints[..., :2].float()
    conf = keypoints[..., 2:3].float()
    slot_valid = torch.isfinite(keypoints).all(-1) & (keypoints[..., 2] > min_confidence)
    logvar = (-2. * conf.clamp(2.5e-3, 1.).log()).clamp(-6, 6)
    return dict(name='external', schema='coco', residual=uv - projected, projected=projected,
                logvar=logvar, visibility=conf.clamp(0, 1), frame_valid=frame_valid, slot_valid=slot_valid)
