"""Copied from v9; adapt limb readout while retaining an immutable v9 reference."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from models.human_decoder_rgb_vace_v9_isolated import HumanRGBVaceGeometryTransformerV9Isolated, RGBVaceHMR2AConfig
from models.human_decoder_geometry import rotation_6d_to_matrix, spatial_sincos_position_encoding

FORMAT = 'ram-v9-geometry-adaptation-v13'

@dataclass(frozen=True)
class AdaptationConfig:
    position_mode: str = 'original'
    def validate(self):
        if self.position_mode not in ('original', 'shared'):
            raise ValueError('position_mode must be original or shared')

class HumanV9AdaptationV13(nn.Module):
    def __init__(self, config=None, adaptation_config=None):
        super().__init__()
        self.config = config or RGBVaceHMR2AConfig()
        self.adaptation_config = adaptation_config or AdaptationConfig()
        self.adaptation_config.validate()
        self.reference = HumanRGBVaceGeometryTransformerV9Isolated(self.config)
        self.reference.requires_grad_(False)
        self.reference.eval()
        # Delta to trained v9 limb readout: root/shape/camera have no optimizer rows.
        self.limb_head_delta = nn.Linear(self.config.hidden_dim, 23 * 6)
        nn.init.zeros_(self.limb_head_delta.weight)
        nn.init.zeros_(self.limb_head_delta.bias)
        self.register_buffer('position_mix', torch.tensor(0., dtype=torch.float32))

    def train(self, mode=True):
        super().train(mode)
        self.reference.eval()
        return self

    def set_position_mix(self, value):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('position_mix must be finite and in [0,1]')
        if self.adaptation_config.position_mode == 'original' and value != 0:
            raise ValueError('original position mode requires zero mixing')
        self.position_mix.fill_(value)

    @torch.no_grad()
    def _read_features(self, rgb_tokens, vace_features, offset=0):
        # Original readout with only RGB slot embeddings interpolated.
        ref = self.reference
        batch, frames, _, _ = rgb_tokens.shape
        memory = ref.rgb_projection(rgb_tokens)
        positions = ref.rgb_frame_embedding.weight[:frames].roll(int(offset), dims=0)
        positions = positions + self.position_mix * (positions.mean(0, keepdim=True) - positions)
        query = (ref.smpl_query + positions[None]).expand(batch, -1, -1)
        query = query.reshape(batch * frames, 1, self.config.hidden_dim)
        memory = memory.reshape(batch * frames, memory.shape[2], self.config.hidden_dim)
        for block in ref.rgb_blocks:
            query = block(query, memory)
        query = query.reshape(batch, frames, self.config.hidden_dim)
        latent_frames, grid_h, grid_w = vace_features.shape[1:4]
        video = ref.vace_projection(vace_features)
        times = ref.vace_time_embedding(torch.arange(latent_frames, device=video.device))
        spatial = spatial_sincos_position_encoding(grid_h, grid_w, self.config.hidden_dim, video.device)
        spatial = spatial.to(video.dtype).reshape(grid_h, grid_w, self.config.hidden_dim)
        video = video + times[:, None, None, :]
        video = video + torch.tanh(ref.vace_spatial_scale) * spatial[None]
        video = video.reshape(batch, -1, self.config.hidden_dim)
        for gate, block in zip(ref.temporal_gates, ref.vace_blocks):
            proposal = block(query, video)
            query = query + torch.tanh(gate) * (proposal - query)
        return ref.output_norm(query)

    def forward(self, rgb_tokens, vace_features, base_rotmat, base_betas,
                base_camera, frame_valid=None, position_offset=0):
        with torch.no_grad():
            original = self.reference(rgb_tokens, vace_features, base_rotmat, base_betas, base_camera)
        batch, frames = base_rotmat.shape[:2]
        if frame_valid is None:
            frame_valid = torch.ones(batch, frames, dtype=torch.bool, device=base_rotmat.device)
        if frame_valid.shape != (batch, frames) or frame_valid.dtype != torch.bool:
            raise ValueError('frame_valid must be Boolean [B,T]')
        if not bool(frame_valid.any(1).all()):
            raise ValueError('Each window needs a valid target frame')
        frame_valid = frame_valid.to(base_rotmat.device)
        shifted = position_offset != 0 or float(self.position_mix) != 0.
        if shifted:
            features = self._read_features(rgb_tokens, vace_features, position_offset)
            with torch.no_grad():
                logits = self.reference.rotation_residual_head(features).reshape(batch, frames, 24, 6)
        else:
            features, logits = original['fused_queries'], original['rotation_residual_6d']
        # FP32 delta retains tiny updates despite BF16 reference inference.
        with torch.autocast(device_type=rgb_tokens.device.type, enabled=False):
            delta = self.limb_head_delta(features.float()).reshape(batch, frames, 23, 6)
            old = original['rotation_residual_6d'].float()
            limb = logits[:, :, 1:].float() + delta
            limb = torch.where(frame_valid[..., None, None], limb, old[:, :, 1:])
            updated = torch.cat((old[:, :, :1], limb), dim=2)
            old_matrix = rotation_6d_to_matrix(old)
            new_matrix = rotation_6d_to_matrix(updated)
            change = (new_matrix[:, :, 1:] - old_matrix[:, :, 1:]) @ base_rotmat[:, :, 1:].float()
            rotmat = torch.cat((original['rotmat'][:, :, :1],
                               original['rotmat'][:, :, 1:] + change), dim=2)
        result = dict(original)
        result.update(rotmat=rotmat, rotation_residual_6d=updated,
                      teacher_rotmat=original['rotmat'], teacher_betas=original['betas'])
        return result

    def anchor_metadata(self):
        return dict(model_format=FORMAT, adaptation_config=asdict(self.adaptation_config),
                    reference='immutable trained v9', trainable='limb readout delta only')
