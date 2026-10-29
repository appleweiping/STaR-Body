"""V33: offline track shape, reliable visual memory and pose completion.
Copied from human_decoder_spatiotemporal_v31.py before editing. Full shape mode
intentionally changes zero-init betas; force_anchor returns frozen V28.
Past-memory ablation is not causal: VACE and shape can use future RGB.
No trained performance is implied.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.human_decoder_parts_v29 import (
    model_from_payload as load_reference, ObservationHead,
    FORMAT as REFERENCE_FORMAT, SCHEMAS, project_landmarks, camera_project,
    sample_joint_neighborhood, spatial_sincos_position_encoding,
)

FORMAT = 'star-body-reliable-memory-v33'
from models.star_memory_modules_v33 import ReliableJointMemory, PoseCompletionPrior


@dataclass(frozen=True)
class AdaptationConfig:
    hidden_dim: int = 256
    num_heads: int = 8
    depth: int = 4
    iterations: int = 3
    grid_height: int = 16
    grid_width: int = 12
    shape_bound: float = 3.
    camera_log_scale_bound: float = .5
    camera_translation_bound: float = .5
    local_radius: float = 1.
    use_local_feedback: bool = True
    use_temporal: bool = True
    use_vace: bool = True
    frame_shape_delta_scale: float = 1.
    local_radius_schedule: tuple[float, ...] = ()
    shared_track_shape: bool = False
    use_memory: bool = False
    use_motion: bool = False
    memory_horizon: int = 32
    memory_topk: int = 8
    shape_prototype_frames: int = 8
    memory_mode: str = 'offline'
    feedback_dropout: float = 0.
    motion_depth: int = 4

    def __post_init__(self):
        # Canonicalize lists restored from JSON without changing the payload API.
        object.__setattr__(self, 'local_radius_schedule', tuple(self.local_radius_schedule))

    def radius_at(self, iteration):
        return self.local_radius_schedule[iteration] if self.local_radius_schedule else self.local_radius

    def validate(self):
        if not 1 <= self.memory_horizon <= 64 or not 1 <= self.memory_topk <= self.memory_horizon:
            raise ValueError('Memory top-k/horizon must satisfy 1 <= k <= horizon <= 64')
        if not 1 <= self.shape_prototype_frames <= 16 or not 1 <= self.motion_depth <= 8:
            raise ValueError('Invalid prototype/motion depth')
        if self.memory_mode not in ('offline','past_memory'):
            raise ValueError('Only offline and past-memory ablations are supported; not a causal full pipeline')
        if not math.isfinite(self.feedback_dropout) or not 0 <= self.feedback_dropout < 1:
            raise ValueError('Invalid feedback dropout')
        if self.num_heads < 1 or self.hidden_dim < 32 or self.hidden_dim % 4 or self.hidden_dim % self.num_heads:
            raise ValueError('Invalid V33 width or attention heads')
        if self.num_heads < 1 or not 1 <= self.depth <= 8 or self.iterations not in (1, 2, 3):
            raise ValueError('Invalid V33 attention/iteration budget')
        if (self.grid_height, self.grid_width) != (16, 12):
            raise ValueError('V33 cache contract is actual HMR2 16x12 spatial tokens')
        for value in (self.shape_bound, self.camera_log_scale_bound, self.camera_translation_bound, self.local_radius):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Finite positive V33 shape/camera/sampling bounds required')
        if not math.isfinite(self.frame_shape_delta_scale) or not 0 <= self.frame_shape_delta_scale <= 1:
            raise ValueError('V33 frame shape delta scale must be in [0, 1]')
        if self.local_radius_schedule and (len(self.local_radius_schedule) != self.iterations
                or any(not math.isfinite(value) or value <= 0 for value in self.local_radius_schedule)):
            raise ValueError('V33 radius schedule must contain one finite positive radius per iteration')


def rotation_6d(values):
    """Rows convention with scale-aware collinearity fallback and re-orthogonalization.

    Orthogonal subtraction alone can retain a large parallel FP32 residual.
    Normalize each input by its largest component before forming the basis, then
    reconstruct the second row from a normalized cross product. Input values
    must be finite; callers should diagnose nonfinite network states separately.
    """
    values = values.float()
    a, b = values[..., :3], values[..., 3:]
    a_scale = a.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    b_scale = b.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    a_unit = F.normalize(a / a_scale, dim=-1, eps=1e-6)
    b_unit = F.normalize(b / b_scale, dim=-1, eps=1e-6)
    first = torch.where((a.abs().amax(dim=-1, keepdim=True) < 1e-6),
                        values.new_tensor([1., 0., 0.]), a_unit)
    third = torch.cross(first, b_unit, dim=-1)
    # Inputs are independently normalized, so this is an angular threshold.
    axis = F.one_hot(first.abs().argmin(-1), 3).to(values.dtype)
    fallback = torch.cross(first, axis, dim=-1)
    third = torch.where(third.norm(dim=-1, keepdim=True) < 1e-6, fallback, third)
    third = F.normalize(third, dim=-1, eps=1e-6)
    second = F.normalize(torch.cross(third, first, dim=-1), dim=-1, eps=1e-6)
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), -2)


def shape_delta_from_values(shape_values, independent, cfg):
    """A shared correction plus an ablatable frame correction, not shared betas."""
    weights = independent.float()[..., None]
    shared = (shape_values[..., :10]*weights).sum(1, keepdim=True)/weights.sum(1, keepdim=True)
    return cfg.shape_bound*torch.tanh(shared + cfg.frame_shape_delta_scale*shape_values[..., 10:])


class AxisAttention(nn.Module):
    def __init__(self, cfg, temporal):
        super().__init__()
        self.temporal = temporal
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.attention = nn.MultiheadAttention(cfg.hidden_dim, cfg.num_heads, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(cfg.hidden_dim), nn.Linear(cfg.hidden_dim, 2*cfg.hidden_dim),
                                 nn.GELU(), nn.Linear(2*cfg.hidden_dim, cfg.hidden_dim))

    def forward(self, tokens, independent):
        b, t, j, d = tokens.shape
        sequence = tokens.permute(0, 2, 1, 3).reshape(b*j, t, d) if self.temporal else tokens.reshape(b*t, j, d)
        mask = (~independent)[:, None].expand(b, j, t).reshape(b*j, t) if self.temporal else None
        normed = self.norm(sequence)
        update, _ = self.attention(normed, normed, normed, key_padding_mask=mask, need_weights=False)
        sequence = sequence + update
        sequence = sequence + self.mlp(sequence)
        return sequence.reshape(b, j, t, d).permute(0, 2, 1, 3) if self.temporal else sequence.reshape(b, t, j, d)


class DualOrderBlock(nn.Module):
    """Learn a mixture of spatial鈫抰emporal and temporal鈫抯patial reasoning."""
    def __init__(self, cfg):
        super().__init__()
        self.use_temporal = cfg.use_temporal
        self.st_space = AxisAttention(cfg, False)
        self.st_time = AxisAttention(cfg, True)
        self.ts_time = AxisAttention(cfg, True)
        self.ts_space = AxisAttention(cfg, False)
        self.mix = nn.Linear(2*cfg.hidden_dim, 2)

    def forward(self, x, independent):
        if not self.use_temporal:
            return self.st_space(x, independent)
        st = self.st_time(self.st_space(x, independent), independent)
        ts = self.ts_space(self.ts_time(x, independent), independent)
        weight = self.mix(torch.cat((st, ts), -1)).softmax(-1)
        return weight[..., :1]*st + weight[..., 1:]*ts


class HumanStarMemoryV33(nn.Module):
    def __init__(self, reference_payload, adaptation_config=None):
        super().__init__()
        self.adaptation_config = adaptation_config or AdaptationConfig()
        self.adaptation_config.validate()
        self.reference = load_reference(reference_payload).requires_grad_(False).eval()
        self.config = self.reference.config
        self.reference_spec = dict(format=REFERENCE_FORMAT, model_config=asdict(self.reference.config),
            adaptation_config=asdict(self.reference.adaptation_config), teacher_config=asdict(self.reference.teacher_config))
        cfg = self.adaptation_config; d = cfg.hidden_dim
        self.rgb_projection = nn.Sequential(nn.LayerNorm(self.config.rgb_input_dim), nn.Linear(self.config.rgb_input_dim, d))
        self.vace_projection = nn.Sequential(nn.LayerNorm(self.config.vace_input_dim), nn.Linear(self.config.vace_input_dim, d))
        self.state_projection = nn.Linear(9+10+3, d)
        self.feedback_projection = nn.Linear(9*d+9+2, d)
        self.joint_embedding = nn.Parameter(torch.randn(26, d)*.02)
        self.image_attention = nn.MultiheadAttention(d, cfg.num_heads, batch_first=True)
        self.vace_attention = nn.MultiheadAttention(d, cfg.num_heads, batch_first=True)
        self.observation = ObservationHead(self.config.rgb_input_dim, cfg)
        self.observation_attention = nn.MultiheadAttention(d, cfg.num_heads, batch_first=True)
        self.blocks = nn.ModuleList([DualOrderBlock(cfg) for _ in range(cfg.depth)])
        self.body_head = nn.Linear(d, 6)
        self.root_head = nn.Linear(d, 6)
        self.camera_head = nn.Linear(d, 3)
        self.shape_head = nn.Linear(d, 20)  # shared + frame correction, both supervised
        self.output_norm = nn.LayerNorm(d)
        self.joint_memory = ReliableJointMemory(d,cfg.memory_topk)
        self.motion_prior = PoseCompletionPrior(d,cfg.num_heads,cfg.motion_depth)
        self.motion_fusion = nn.Sequential(nn.LayerNorm(2*d+2),nn.Linear(2*d+2,d),nn.GELU(),nn.Linear(d,1))
        nn.init.zeros_(self.motion_fusion[-1].weight);nn.init.constant_(self.motion_fusion[-1].bias,2.)
        self.visual_quality = nn.Linear(d,1)
        self.track_shape_projection = nn.Sequential(nn.LayerNorm(self.config.rgb_input_dim+10),
            nn.Linear(self.config.rgb_input_dim+10,d),nn.GELU(),nn.Linear(d,d),nn.GELU())
        self.track_shape_head = nn.Linear(d,10)
        nn.init.zeros_(self.track_shape_head.weight);nn.init.zeros_(self.track_shape_head.bias)
        for head in (self.body_head, self.root_head, self.camera_head, self.shape_head):
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
        self.collect_diagnostics = False
        self.fusion_diagnostics = {}

    def train(self, mode=True):
        super().train(mode); self.reference.eval(); return self

    def bind_geometry(self, neutral_smpl):
        neutral_smpl.eval().requires_grad_(False)
        self.reference.bind_geometry(neutral_smpl)
        object.__setattr__(self, '_neutral_smpl', neutral_smpl)
        return self

    def bind_landmarks(self, official_smpl):
        official_smpl.eval().requires_grad_(False)
        self.reference.bind_landmarks(official_smpl)
        object.__setattr__(self, '_official_smpl', official_smpl)
        return self

    def time_embedding(self, source_ids, valid):
        first = source_ids.masked_fill(~valid, torch.iinfo(torch.long).max).min(1).values
        times = (source_ids-first[:, None]).float()
        freq = torch.exp(torch.arange(0, self.adaptation_config.hidden_dim, 2, device=times.device).float()
                         * (-math.log(10000.)/self.adaptation_config.hidden_dim))
        phase = times[..., None]*freq
        return torch.stack((phase.sin(), phase.cos()), -1).flatten(-2)

    def forward(self, rgb_tokens, vace_features, base_rotmat, base_betas, base_camera,
                frame_valid=None, position_offset=0, landmark_schema='posetrack', source_frame_ids=None, force_anchor=False, memory_rgb_tokens=None,
                memory_rotmat=None, memory_betas=None, memory_camera=None,
                memory_source_frame_ids=None, memory_annotation_ids=None, memory_valid=None,
                target_annotation_ids=None, track_shape_base=None, track_shape_features=None):
        cfg = self.adaptation_config
        b, t = base_rotmat.shape[:2]
        if landmark_schema not in SCHEMAS or position_offset:
            raise ValueError('Unsupported landmark schema/position offset')
        if source_frame_ids is None or source_frame_ids.shape != (b, t) or source_frame_ids.dtype != torch.long:
            raise ValueError('V33 requires explicit integer source/annotation-time IDs [B,T]')
        valid = frame_valid if frame_valid is not None else torch.ones((b, t), dtype=torch.bool, device=base_rotmat.device)
        if valid.shape != (b, t) or valid.dtype != torch.bool or not valid.any(1).all():
            raise ValueError('Every track window needs a valid frame')
        if rgb_tokens.shape[:3] != (b, t, cfg.grid_height*cfg.grid_width):
            raise ValueError('Actual spatial HMR2 RGB tokens required')
        if rgb_tokens.shape[-1] != self.config.rgb_input_dim or vace_features.ndim != 5 or vace_features.shape[0] != b:
            raise ValueError('Feature widths/layout differ from the declared cache contract')
        if vace_features.shape[-1] != self.config.vace_input_dim or base_rotmat.shape != (b, t, 24, 3, 3):
            raise ValueError('VACE channels/SMPL rotation shape differ')
        if not hasattr(self, '_neutral_smpl') or not hasattr(self, '_official_smpl'):
            raise RuntimeError('Bind native and official-landmark SMPL before forwarding')
        if not torch.isfinite(base_camera).all() or not (base_camera[..., 0] > 0).all():
            raise ValueError('Invalid input camera')
        independent = valid.clone()
        for i in range(1, t):
            independent[:, i] &= ~((source_frame_ids[:, :i] == source_frame_ids[:, i, None]) & valid[:, :i]).any(1)
        with torch.no_grad():
            reference = self.reference(rgb_tokens, vace_features, base_rotmat, base_betas, base_camera,
                frame_valid=valid, source_frame_ids=source_frame_ids, landmark_schema=landmark_schema)
            initial = {key: reference[key].detach().float() for key in ('rotmat', 'betas', 'camera')}
            anchor_uv = project_landmarks(initial, self._official_smpl, landmark_schema)
        if force_anchor:
            return reference
        needs_context=cfg.use_memory or cfg.use_motion or cfg.shared_track_shape
        if needs_context:
            required=(memory_rgb_tokens,memory_rotmat,memory_betas,memory_camera,
                memory_source_frame_ids,memory_annotation_ids,memory_valid,target_annotation_ids,
                track_shape_base,track_shape_features)
            if any(x is None for x in required):raise ValueError('Context-enabled V33 requires TrackContextDataset inputs')
            m=memory_rotmat.shape[1]
            expected={'memory_rgb_tokens':(b,m,192,self.config.rgb_input_dim),
                'memory_rotmat':(b,m,24,3,3),'memory_betas':(b,m,10),'memory_camera':(b,m,3),
                'memory_source_frame_ids':(b,m),'memory_annotation_ids':(b,m),
                'memory_valid':(b,m),'target_annotation_ids':(b,t),
                'track_shape_base':(b,10),'track_shape_features':(b,self.config.rgb_input_dim)}
            for name,value in zip(expected,required):
                if tuple(value.shape)!=expected[name]:raise ValueError('Context shape mismatch: '+name)
                if value.is_floating_point() and not torch.isfinite(value).all():raise ValueError('Nonfinite context: '+name)
            if m!=cfg.memory_horizon or memory_valid.dtype!=torch.bool:
                raise ValueError('Context horizon/validity differs from saved model')
            for value in (memory_source_frame_ids,memory_annotation_ids,target_annotation_ids):
                if value.dtype!=torch.long:raise ValueError('Integer context times required')
            if not (memory_camera[...,0]>0).all():raise ValueError('Invalid context camera scale')
            allowed=memory_valid[:,None,:].expand(b,t,m).clone()
            allowed &= memory_source_frame_ids[:,None,:]!=source_frame_ids[:,:,None]
            allowed &= memory_annotation_ids[:,None,:]!=target_annotation_ids[:,:,None]
            allowed &= valid[:,:,None]
            if cfg.memory_mode=='past_memory':
                allowed &= memory_annotation_ids[:,None,:]<target_annotation_ids[:,:,None]
            if cfg.shared_track_shape:
                features=self.track_shape_projection(torch.cat((track_shape_features,track_shape_base),-1))
                absolute_shape=track_shape_base.float()+cfg.shape_bound*torch.tanh(self.track_shape_head(features).float())
            if cfg.use_motion:
                motion_features,motion_raw,motion_quality,motion_available=self.motion_prior(
                    memory_rotmat,memory_source_frame_ids,source_frame_ids,allowed)
                motion_rotmat=rotation_6d(motion_raw.new_tensor([1.,0.,0.,0.,1.,0.])+motion_raw)
            if cfg.use_memory:
                memory_image=self.rgb_projection(memory_rgb_tokens).reshape(b*m,192,cfg.hidden_dim)
                memory_pos=spatial_sincos_position_encoding(16,12,cfg.hidden_dim,memory_image.device)
                memory_maps=(memory_image+memory_pos[None].to(memory_image.dtype)).transpose(1,2).reshape(b*m,cfg.hidden_dim,16,12)
                with torch.autocast(device_type=base_rotmat.device.type,enabled=False):
                    memory_body=self._neutral_smpl(global_orient=memory_rotmat.float().reshape(-1,24,3,3)[:,:1],
                        body_pose=memory_rotmat.float().reshape(-1,24,3,3)[:,1:],betas=memory_betas.float().reshape(-1,10),pose2rot=False)
                    memory_uv,memory_uv_valid=camera_project(memory_body.joints[:,:24],memory_camera.float().reshape(-1,3))
                local,sampled=sample_joint_neighborhood(memory_maps,memory_uv*memory_uv.new_tensor([8./3,2.]),
                    memory_uv_valid & memory_valid.reshape(-1,1),cfg.local_radius)
                native=torch.cat((memory_rotmat.flatten(-2),memory_betas[:,:,None].expand(-1,-1,24,-1),
                    memory_camera[:,:,None].expand(-1,-1,24,-1)),-1)
                memory_tokens=self.state_projection(native)+self.feedback_projection(
                    torch.cat((local.flatten(-2),sampled.float(),memory_uv),-1)).reshape(b,m,24,-1)
                memory_joint_valid=(memory_uv_valid & sampled.any(-1)).reshape(b,m,24).permute(0,2,1)

        image = self.rgb_projection(rgb_tokens).reshape(b*t, -1, cfg.hidden_dim)
        image_pos = spatial_sincos_position_encoding(cfg.grid_height, cfg.grid_width, cfg.hidden_dim, image.device)
        image = image + image_pos.reshape(1, -1, cfg.hidden_dim).to(image.dtype)
        maps = image.transpose(1, 2).reshape(b*t, cfg.hidden_dim, cfg.grid_height, cfg.grid_width)
        # Compress VACE spatially, preserving each latent time and six spatial cells.
        f, h, w = vace_features.shape[1:4]
        if cfg.use_vace:
            prior = self.vace_projection(vace_features).permute(0, 1, 4, 2, 3).reshape(b*f, cfg.hidden_dim, h, w)
            prior = F.adaptive_avg_pool2d(prior, (2, 3)).reshape(b, f, cfg.hidden_dim, 6).permute(0, 1, 3, 2)
            pos = spatial_sincos_position_encoding(2, 3, cfg.hidden_dim, image.device).reshape(1, 1, 6, -1)
            ids = torch.arange(f, device=image.device)[None].expand(b, -1)
            prior = prior + pos + self.time_embedding(ids, torch.ones_like(ids, dtype=torch.bool))[:, :, None]
            prior = prior.reshape(b, f*6, cfg.hidden_dim)
        obs_uv, logvar, visibility, obs_features, _ = self.observation(rgb_tokens, anchor_uv, landmark_schema)
        time = self.time_embedding(source_frame_ids, valid)[:, :, None]
        state = initial
        geometry_stages, projections = [], []
        identity = image.new_tensor([1., 0., 0., 0., 1., 0.]).float()
        for iteration in range(cfg.iterations):
            with torch.autocast(device_type=image.device.type, enabled=False):
                body = self._neutral_smpl(global_orient=state['rotmat'].reshape(-1, 24, 3, 3)[:, :1],
                    body_pose=state['rotmat'].reshape(-1, 24, 3, 3)[:, 1:], betas=state['betas'].reshape(-1, 10), pose2rot=False)
                uv, uv_valid = camera_project(body.joints[:, :24], state['camera'].reshape(-1, 3))
            native = torch.cat((state['rotmat'].flatten(-2), state['betas'][:, :, None].expand(-1, -1, 24, -1),
                                state['camera'][:, :, None].expand(-1, -1, 24, -1)), -1)
            joints = self.state_projection(native)
            if cfg.use_local_feedback:
                grid = uv*uv.new_tensor([8./3, 2.])
                local, sampled = sample_joint_neighborhood(maps, grid, uv_valid & valid.reshape(-1, 1), cfg.radius_at(iteration))
                local = self.feedback_projection(torch.cat((local.flatten(-2), sampled.float(), uv), -1)).reshape(b, t, 24, -1)
                local = F.dropout(local,p=cfg.feedback_dropout,training=self.training)
                joints = joints + local
            # Joint 0 is root; 24/25 are explicit shape/camera queries.
            tokens = torch.cat((joints, joints.mean(2, keepdim=True).expand(-1, -1, 2, -1)), 2)
            tokens = tokens + self.joint_embedding + time
            attended, _ = self.image_attention(tokens.reshape(b*t, 26, -1), image, image, need_weights=False)
            tokens = tokens + attended.reshape(b, t, 26, -1)
            if cfg.use_vace:
                motion, _ = self.vace_attention(tokens.reshape(b, t*26, -1), prior, prior, need_weights=False)
                tokens = tokens + motion.reshape(b, t, 26, -1)
            observed, _ = self.observation_attention(tokens.reshape(b*t, 26, -1), obs_features.reshape(b*t, 17, -1),
                obs_features.reshape(b*t, 17, -1), need_weights=False)
            tokens = tokens + observed.reshape(b, t, 26, -1)
            if cfg.use_memory:
                update,memory_report=self.joint_memory(tokens[:,:,:24],memory_tokens,
                    allowed[:,None] & memory_joint_valid[:,:,None,:],source_frame_ids,memory_source_frame_ids)
                tokens=torch.cat((tokens[:,:,:24]+update,tokens[:,:,24:]),2)
            if cfg.use_motion:
                visual=tokens[:,:,:24]
                visual_quality=self.visual_quality(visual).squeeze(-1)
                fusion_input=torch.cat((visual,motion_features,visual_quality[...,None],motion_quality[...,None]),-1)
                gate_logits=self.motion_fusion(fusion_input).squeeze(-1)
                gate=torch.where(motion_available[...,None],gate_logits.sigmoid(),torch.ones_like(gate_logits))
                fused=gate[...,None]*visual+(1-gate[...,None])*motion_features
                tokens=torch.cat((fused,tokens[:,:,24:]),2)
            for block in self.blocks:
                tokens = block(tokens, independent)
            tokens = self.output_norm(tokens)
            raw = torch.cat((self.root_head(tokens[:, :, :1]), self.body_head(tokens[:, :, 1:24])), 2).float()
            camera_delta = torch.tanh(self.camera_head(tokens[:, :, 25]).float())
            camera_delta = camera_delta*camera_delta.new_tensor([cfg.camera_log_scale_bound, cfg.camera_translation_bound, cfg.camera_translation_bound])
            shape_values = self.shape_head(tokens[:, :, 24]).float()
            shape_delta = (absolute_shape[:,None]-initial['betas'] if cfg.shared_track_shape
                           else shape_delta_from_values(shape_values, independent, cfg))
            if force_anchor:
                raw = raw*0.; camera_delta = camera_delta*0.; shape_delta = shape_delta*0.
            with torch.autocast(device_type=image.device.type, enabled=False):
                rotations = rotation_6d(identity+raw) @ initial['rotmat']
                camera = torch.cat((initial['camera'][..., :1]*camera_delta[..., :1].exp(),
                                    initial['camera'][..., 1:]+camera_delta[..., 1:]), -1)
                state = dict(rotmat=torch.where(valid[..., None, None, None], rotations, initial['rotmat']),
                    betas=(torch.where(valid[...,None],absolute_shape[:,None].expand(-1,t,-1),initial['betas'])
                           if cfg.shared_track_shape else initial['betas']+shape_delta*valid[..., None]),
                    camera=torch.where(valid[..., None], camera, initial['camera']))
                geometry_stages.append(state)
                projection = project_landmarks(state, self._official_smpl, landmark_schema)
                projections.append(projection)
        result = state | dict(geometry_stages=geometry_stages, stage_projections=torch.stack(projections),
            direct_2d=projections[-1], observed_2d=obs_uv, observation_log_variance=logvar, visibility_logits=visibility,
            anchor_2d=anchor_uv, anchor_prediction=initial, teacher_prediction=initial,
            rotation_update=raw, camera_update=camera_delta, shape_correction=shape_delta,
            frame_valid=valid, independent_frames=independent, fused_queries=tokens,
            posetrack_2d_residual=obs_uv-projections[-1], posetrack_confidence=(-logvar.exp()).exp(),
            rotation_residual_6d=(state['rotmat'] @ initial['rotmat'].transpose(-1, -2))[..., :2, :].flatten(-2))
        if cfg.use_motion:
            result.update(motion_rotmat=motion_rotmat,motion_available=motion_available,
                motion_quality_logits=motion_quality,visual_quality_logits=visual_quality,
                motion_gate_logits=gate_logits,visual_gate=gate)
        if cfg.use_memory:
            result.update(memory_selected_count=memory_report['selected_count'],memory_available=memory_report['available'])
        if cfg.shared_track_shape:result['track_betas']=absolute_shape
        return result


def model_from_payload(payload, adaptation_config=None):
    if payload.get('format') == FORMAT:
        saved = AdaptationConfig(**payload['adaptation_config'])
        if adaptation_config is not None and asdict(saved) != asdict(adaptation_config):
            raise ValueError('Native V33 configuration differs from saved architecture')
        state = payload['model']
        reference_state = {key[len('reference.'):]: value for key, value in state.items() if key.startswith('reference.')}
        if not reference_state:
            raise ValueError('Native V33 checkpoint lacks its frozen reference')
        reference = payload['reference_spec'] | dict(model=reference_state)
        model = HumanStarMemoryV33(reference, saved)
        if asdict(model.config) != payload['model_config']:
            raise ValueError('Native V33 feature configuration differs from reference')
        model.load_state_dict(state, strict=True)
    elif payload.get('format') in ('ram-iterative-anchor-routed-v28', REFERENCE_FORMAT):
        model = HumanStarMemoryV33(payload, adaptation_config)
    else:
        raise ValueError('V33 accepts only native V33 or audited V28/V29 initialization')
    model.reference.requires_grad_(False).eval()
    return model
