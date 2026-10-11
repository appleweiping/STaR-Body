"""V29 extension using real, global SAM/DINO RGB evidence.

The pooled evidence gives every spatial RGB token the same per-frame bias.
It is a global evidence hypothesis, not a spatial or fine-grained feature map.
The original V29 implementation and its frozen teacher are unchanged.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from models.human_decoder_parts_v29 import (
    HumanV9AdaptationV29, AdaptationConfig, FORMAT as V29_FORMAT,
    model_from_payload as load_v29_payload,
)


FORMAT = 'ram-geometry-global-strong-rgb-v29x'
SUPPORTED_MIGRATIONS = {'ram-iterative-anchor-routed-v28', V29_FORMAT}


@dataclass(frozen=True)
class StrongRGBConfig:
    input_dim: int = 1280
    hidden_dim: int = 256
    delta_bound: float = .1
    feature_policy: str = 'real-sam-dinov3-global-spatial-mean'

    def validate(self):
        if isinstance(self.input_dim, bool) or not isinstance(self.input_dim, int) or self.input_dim < 1:
            raise ValueError('Strong RGB input dimension must be a positive integer')
        if self.hidden_dim != 256:
            raise ValueError('This architecture declares an exact 256-wide strong RGB adapter')
        if not math.isfinite(self.delta_bound) or self.delta_bound <= 0:
            raise ValueError('Strong RGB delta bound must be finite and positive')
        if self.feature_policy != 'real-sam-dinov3-global-spatial-mean':
            raise ValueError('Only declared real SAM/DINO globally pooled evidence is supported')


class HumanV9AdaptationStrongRGBV29X(HumanV9AdaptationV29):
    """Retain original parameter namespaces through direct inheritance."""
    def __init__(self, config=None, adaptation_config=None, teacher_config=None, strong_config=None):
        super().__init__(config, adaptation_config, teacher_config)
        self.strong_config = strong_config or StrongRGBConfig()
        self.strong_config.validate()
        self.strong_rgb_adapter = nn.Sequential(
            nn.LayerNorm(self.strong_config.input_dim),
            nn.Linear(self.strong_config.input_dim, 256),
            nn.GELU(),
            nn.Linear(256, self.config.rgb_input_dim),
        )
        nn.init.zeros_(self.strong_rgb_adapter[-1].weight)
        nn.init.zeros_(self.strong_rgb_adapter[-1].bias)

    def forward(self, rgb_tokens, vace_features, base_rotmat, base_betas, base_camera,
                frame_valid=None, position_offset=0, landmark_schema='posetrack',
                source_frame_ids=None, force_anchor=False, *, strong_rgb_pooled=None):
        if strong_rgb_pooled is None:
            raise ValueError('Real strong_rgb_pooled evidence is mandatory; missing-feature fallbacks are forbidden')
        if not torch.is_tensor(strong_rgb_pooled) or not torch.is_floating_point(strong_rgb_pooled):
            raise TypeError('Strong RGB pooled evidence must be a floating Tensor')
        if rgb_tokens.ndim != 4 or rgb_tokens.shape[-1] != self.config.rgb_input_dim:
            raise ValueError('RGB tokens must be [B,T,N,rgb_width]')
        expected = (*rgb_tokens.shape[:2], self.strong_config.input_dim)
        if tuple(strong_rgb_pooled.shape) != expected:
            raise ValueError(f'Strong RGB evidence must be [B,T,D]={expected}')
        if strong_rgb_pooled.device != rgb_tokens.device:
            raise ValueError('Strong evidence and RGB tokens must share a device')
        if not torch.isfinite(strong_rgb_pooled).all():
            raise FloatingPointError('Strong RGB evidence contains nonfinite values')
        evidence = strong_rgb_pooled.to(dtype=self.strong_rgb_adapter[0].weight.dtype)
        learned = self.strong_rgb_adapter(evidence)
        delta = (self.strong_config.delta_bound * learned.float().tanh()).to(dtype=rgb_tokens.dtype)
        # Spatially constant bias: this does not recreate DINO spatial details.
        conditioned_rgb = rgb_tokens + delta.unsqueeze(2)
        result = super().forward(
            conditioned_rgb, vace_features, base_rotmat, base_betas, base_camera,
            frame_valid=frame_valid, position_offset=position_offset,
            landmark_schema=landmark_schema, source_frame_ids=source_frame_ids,
            force_anchor=force_anchor,
        )
        result['strong_rgb_delta'] = delta
        return result

    def checkpoint_config(self):
        """Merge this schema into checkpoint metadata before model state saving."""
        return dict(
            format=FORMAT, model_config=asdict(self.config),
            adaptation_config=asdict(self.adaptation_config),
            teacher_config=asdict(self.teacher_config),
            strong_config=asdict(self.strong_config),
        )

    def checkpoint_payload(self, metadata=None):
        """Build a native payload; callers can add training/selection metadata."""
        metadata = dict(metadata or {})
        reserved = set(self.checkpoint_config()) | {'model'}
        if reserved.intersection(metadata):
            raise ValueError('Checkpoint metadata cannot override architecture or state fields')
        return metadata | self.checkpoint_config() | dict(model=self.state_dict())


def _config_equal(requested, saved, name):
    if requested is not None and asdict(requested) != asdict(saved):
        raise ValueError(f'Requested {name} differs from native strong RGB checkpoint')


def model_from_payload(payload, adaptation_config=None, strong_config=None):
    """Native strict restore, or verified V28/V29 zero-output migration only."""
    checkpoint_format = payload.get('format')
    if checkpoint_format == FORMAT:
        from models.human_decoder_parts_v29 import RGBVaceHMR2AConfig, V27Config
        saved_adaptation = AdaptationConfig(**payload['adaptation_config'])
        saved_strong = StrongRGBConfig(**payload['strong_config'])
        _config_equal(adaptation_config, saved_adaptation, 'adaptation config')
        _config_equal(strong_config, saved_strong, 'strong RGB config')
        model = HumanV9AdaptationStrongRGBV29X(
            RGBVaceHMR2AConfig(**payload['model_config']), saved_adaptation,
            V27Config(**payload['teacher_config']), saved_strong,
        )
        model.load_state_dict(payload['model'], strict=True)
    elif checkpoint_format in SUPPORTED_MIGRATIONS:
        # The V29 loader verifies every old key and all architecture/trust bounds,
        # including the exact permitted shape_head-only migration from V28.
        base = load_v29_payload(payload, adaptation_config)
        model = HumanV9AdaptationStrongRGBV29X(
            base.config, base.adaptation_config, base.teacher_config, strong_config,
        )
        missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
        expected_missing = {name for name in model.state_dict() if name.startswith('strong_rgb_adapter.')}
        if set(missing) != expected_missing or unexpected:
            raise ValueError('Migration may introduce only declared strong_rgb_adapter parameters')
        if torch.count_nonzero(model.strong_rgb_adapter[-1].weight).item() or torch.count_nonzero(model.strong_rgb_adapter[-1].bias).item():
            raise ValueError('Migrated strong adapter must begin with exactly zero output')
    else:
        raise ValueError('Strong RGB V29X requires an exact V28/V29 source or native strong RGB checkpoint')
    model.teacher.requires_grad_(False).eval()
    return model
