"""v26: frozen v25 geometry plus a dedicated temporal 2D keypoint branch.

Copied from v20; previous experiment files remain unchanged. All RGB/VACE
fusion layers, temporal gates, prediction heads, and the local correction branch
are trainable. Cached feature extractors and the fixed SMPL asset are external.
"""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.human_decoder_rgb_vace_v9_isolated import HumanRGBVaceGeometryTransformerV9Isolated, RGBVaceHMR2AConfig
from models.human_decoder_geometry import spatial_sincos_position_encoding

FORMAT = 'ram-v9-pose2d-query-v26'
V25_FORMAT = 'ram-v9-local-residual-v25'
V22_FORMAT = 'ram-v9-full-decoder-v22'
# SMPL indices: six disjoint groups cover all 23 non-root joints.
PARTS = ((1,4,7,10),(2,5,8,11),(3,6,9),(12,15),(13,16,18,20,22),(14,17,19,21,23))
# With the default 20-degree bound these logits yield effective limits
# [7.5, 7.5, 12.5, 12.5, 7.5, 7.5] degrees.
DEFAULT_PART_LOGITS = (-0.5108256238,-0.5108256238,0.5108256238,0.5108256238,
                       -0.5108256238,-0.5108256238)

@dataclass(frozen=True)
class AdaptationConfig:
    position_mode: str = 'original'
    hidden_dim: int = 128
    num_heads: int = 4
    grid_height: int = 16
    grid_width: int = 12
    max_angle_degrees: float = 20.
    local_radius: float = 1.
    def validate(self):
        if self.position_mode != 'original':
            raise ValueError('v25 preserves original reference slot embeddings')
        if min(self.hidden_dim,self.num_heads,self.grid_height,self.grid_width) < 1:
            raise ValueError('Invalid part dimensions')
        if self.hidden_dim % self.num_heads or self.hidden_dim % 4:
            raise ValueError('Part width must be divisible by heads and four')
        if not math.isfinite(self.max_angle_degrees) or not 0 < self.max_angle_degrees <= 30:
            raise ValueError('Angle bound must be in (0,30] degrees')
        if not math.isfinite(self.local_radius) or self.local_radius <= 0:
            raise ValueError('Local radius must be positive')

def axis_angle_matrix(vector):
    """Stable Rodrigues, finite derivative at exactly zero."""
    theta = torch.linalg.vector_norm(vector,dim=-1)
    x,y,z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),dim=-1).reshape(*vector.shape[:-1],3,3)
    a = torch.sinc(theta / math.pi)[...,None,None]
    b = .5 * torch.sinc(theta / (2*math.pi)).square()[...,None,None]
    return torch.eye(3,device=vector.device,dtype=vector.dtype)+a*skew+b*(skew@skew)

def project_joint_grid(joints, camera):
    """HMR2 square-crop projection -> central 192/256 crop, align_corners=False."""
    camera = camera.float()
    translation = torch.stack((camera[...,1], camera[...,2],
                              10000. / (256. * camera[...,0].clamp_min(1e-6))), -1)
    points = joints.float() + translation[...,None,:]
    square = points[...,:2] / points[...,2:].clamp_min(1e-6) * (5000./256.)
    grid = square * square.new_tensor([8./3., 2.])
    valid = (torch.isfinite(points).all(-1) & torch.isfinite(grid).all(-1)
             & (points[...,2] > 1e-6) & (camera[...,0,None] > 0))
    return torch.where(valid[...,None], grid, torch.zeros_like(grid)), valid


def sample_joint_neighborhood(memory, grid, valid, radius=1.):
    """Sample 3x3 feature neighborhoods; each invalid location contributes zero."""
    height, width = memory.shape[-2:]
    offsets = grid.new_tensor([(x*2./width, y*2./height)
                              for y in (-1,0,1) for x in (-1,0,1)]) * radius
    locations = grid[:,:,None,:] + offsets
    mask = valid[:,:,None] & (locations.abs() <= 1).all(-1)
    locations = torch.where(mask[...,None], locations, torch.zeros_like(locations))
    # FP32 grid_sample supports both mixed precision and CPU regression checks.
    samples = F.grid_sample(memory.float(), locations.float(), mode='bilinear',
                            padding_mode='border', align_corners=False)
    samples = samples.permute(0,2,3,1) * mask[...,None]
    return samples, mask


class SpatialPartBranch(nn.Module):
    def __init__(self, rgb_width, context_width, config):
        super().__init__()
        d=config.hidden_dim
        self.config=config
        self.rgb_projection=nn.Sequential(nn.LayerNorm(rgb_width),nn.Linear(rgb_width,d))
        self.context_projection=nn.Linear(context_width,d)
        self.queries=nn.Parameter(torch.randn(23,d)*.02)
        self.part_logits=nn.Parameter(torch.tensor(DEFAULT_PART_LOGITS,dtype=torch.float32))
        joint_parts=torch.empty(23,dtype=torch.long)
        for group,joints in enumerate(PARTS):
            for joint in joints:
                joint_parts[joint-1]=group
        self.register_buffer('joint_parts',joint_parts,persistent=False)
        self.local_projection=nn.Linear(9*d,d)
        self.pose_projection=nn.Linear(18,d)
        self.valid_projection=nn.Linear(9,d)
        self.register_buffer('parents',torch.tensor([0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21]))
        self.query_norm=nn.LayerNorm(d)
        self.memory_norm=nn.LayerNorm(d)
        self.attention=nn.MultiheadAttention(d,config.num_heads,batch_first=True,dropout=0.)
        self.mlp=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d))
        self.heads=nn.ModuleList([nn.Linear(d,3)])
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self,tokens,context,rotations,grid,projection_valid):
        batch,frames,count,_=tokens.shape
        cfg=self.config
        if count != cfg.grid_height*cfg.grid_width:
            raise ValueError('RGB token count differs from the declared spatial grid')
        memory=self.rgb_projection(tokens).reshape(batch*frames,count,cfg.hidden_dim)
        feature_map=memory.transpose(1,2).reshape(batch*frames,cfg.hidden_dim,cfg.grid_height,cfg.grid_width)
        local,mask=sample_joint_neighborhood(feature_map,grid.reshape(-1,23,2),
                                            projection_valid.reshape(-1,23),cfg.local_radius)
        position=spatial_sincos_position_encoding(cfg.grid_height,cfg.grid_width,cfg.hidden_dim,tokens.device)
        memory=memory+position.reshape(1,count,cfg.hidden_dim).to(memory.dtype)
        queries=self.queries[None]+self.context_projection(context).reshape(batch*frames,1,-1)
        pose=rotations.reshape(-1,24,3,3)
        pose=torch.cat((pose[:,1:].flatten(-2),pose[:,self.parents].flatten(-2)),dim=-1)
        queries=queries+self.local_projection(local.flatten(-2).to(memory.dtype))
        queries=queries+self.pose_projection(pose.to(memory.dtype))+self.valid_projection(mask.to(memory.dtype))
        attended,_=self.attention(self.query_norm(queries),self.memory_norm(memory),
                                  self.memory_norm(memory),need_weights=False)
        queries=queries+attended
        queries=queries+self.mlp(queries)
        with torch.autocast(device_type=tokens.device.type,enabled=False):
            raw=self.heads[0](queries.float()).reshape(batch,frames,23,3)
            trust=torch.sigmoid(self.part_logits.float())[self.joint_parts].reshape(1,1,23,1)
            return trust*math.radians(cfg.max_angle_degrees)/math.sqrt(3)*torch.tanh(raw)

class PoseTrackQueryBranch(nn.Module):
    """Predict bounded crop-coordinate residuals for the 17 PoseTrack joints."""
    def __init__(self, rgb_width, context_width, config):
        super().__init__()
        d = config.hidden_dim
        self.config = config
        self.rgb_projection = nn.Sequential(nn.LayerNorm(rgb_width), nn.Linear(rgb_width, d))
        self.context_projection = nn.Linear(context_width, d)
        self.queries = nn.Parameter(torch.randn(17, d) * .02)
        self.query_norm = nn.LayerNorm(d)
        self.memory_norm = nn.LayerNorm(d)
        self.spatial_attention = nn.MultiheadAttention(d, config.num_heads, batch_first=True, dropout=0.)
        self.spatial_mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2*d), nn.GELU(), nn.Linear(2*d, d))
        layer = nn.TransformerEncoderLayer(d, config.num_heads, 2*d, dropout=0., activation='gelu', batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.temporal_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.max_residual = .25

    def forward(self, tokens, context, frame_valid):
        batch, frames, count, _ = tokens.shape
        d = self.config.hidden_dim
        memory = self.rgb_projection(tokens).reshape(batch*frames, count, d)
        position = spatial_sincos_position_encoding(self.config.grid_height, self.config.grid_width, d, tokens.device)
        memory = memory + position.reshape(1, count, d).to(memory.dtype)
        query = self.queries[None] + self.context_projection(context).reshape(batch*frames, 1, d)
        attended, _ = self.spatial_attention(self.query_norm(query), self.memory_norm(memory), self.memory_norm(memory), need_weights=False)
        query = query + attended
        query = query + self.spatial_mlp(query)
        sequence = query.reshape(batch, frames, 17, d).permute(0, 2, 1, 3).reshape(batch*17, frames, d)
        padding = (~frame_valid)[:, None, :].expand(batch, 17, frames).reshape(batch*17, frames)
        sequence = self.temporal(sequence, src_key_padding_mask=padding)
        sequence = self.temporal_norm(sequence).reshape(batch, 17, frames, d).permute(0, 2, 1, 3)
        residual = self.max_residual * torch.tanh(self.head(sequence.float()))
        return residual * frame_valid[..., None, None]


class HumanV9AdaptationV26(nn.Module):
    def __init__(self,config=None,adaptation_config=None):
        super().__init__()
        self.config=config or RGBVaceHMR2AConfig()
        self.adaptation_config=adaptation_config or AdaptationConfig()
        self.adaptation_config.validate()
        # Preserve the validated v22/v19 prediction and train only the bounded
        # residual branch. State keys remain unchanged for strict migration.
        self.reference=HumanRGBVaceGeometryTransformerV9Isolated(self.config)
        self.part_branch=SpatialPartBranch(self.config.rgb_input_dim,self.config.hidden_dim,self.adaptation_config)
        self.pose2d_branch=PoseTrackQueryBranch(self.config.rgb_input_dim,self.config.hidden_dim,self.adaptation_config)
        self.reference.requires_grad_(False).eval()
        self.part_branch.requires_grad_(False).eval()
        self.pose2d_branch.requires_grad_(True)
        self.collect_diagnostics = False
        self.fusion_diagnostics = {}
        for block in self.reference.vace_blocks:
            block.register_forward_hook(self._record_fusion)

    def _record_fusion(self, module, inputs, proposal):
        if self.collect_diagnostics:
            with torch.no_grad():
                query = inputs[0].detach().float()
                index = next(i for i, block in enumerate(self.reference.vace_blocks) if module is block)
                gate = torch.tanh(self.reference.temporal_gates[index].detach().float())
                update = gate * (proposal.detach().float()-query)
                self.fusion_diagnostics[f'block_{index}'] = {
                    'gate_tanh': float(gate.item()),
                    'fusion_update_ratio': float((update.norm()/query.norm().clamp_min(1e-12)).item()),
                }

    def train(self, mode=True):
        super().train(mode)
        self.reference.eval()
        self.part_branch.eval()
        self.pose2d_branch.train(mode)
        return self

    def bind_geometry(self, neutral_smpl):
        # External frozen asset, deliberately excluded from the learned checkpoint.
        neutral_smpl.requires_grad_(False).eval()
        object.__setattr__(self, '_neutral_smpl', neutral_smpl)
        return self

    def forward(self,rgb_tokens,vace_features,base_rotmat,base_betas,base_camera,
                frame_valid=None,position_offset=0):
        if position_offset:
            raise ValueError('Position perturbation is not part of the v25 experiment')
        batch,frames=base_rotmat.shape[:2]
        if frame_valid is None:
            frame_valid=torch.ones(batch,frames,dtype=torch.bool,device=base_rotmat.device)
        if frame_valid.shape != (batch,frames) or frame_valid.dtype != torch.bool or not frame_valid.any(1).all():
            raise ValueError('Expected a Boolean [B,T] mask with valid targets in each window')
        if not hasattr(self,'_neutral_smpl'):
            raise RuntimeError('Bind a neutral native-SMPL model before v25 inference')
        self.fusion_diagnostics = {}
        # The reference is a fixed anchor; only the bounded local correction
        # receives gradients.
        with torch.no_grad():
            original=self.reference(rgb_tokens,vace_features,base_rotmat,base_betas,base_camera)
            with torch.autocast(device_type=rgb_tokens.device.type,enabled=False):
                r=original['rotmat'].float().reshape(-1,24,3,3)
                body=self._neutral_smpl(global_orient=r[:,:1],body_pose=r[:,1:],
                     betas=original['betas'].float().reshape(-1,10),pose2rot=False)
                # Native SMPL joints 0:24, not the OpenPose-reordered HMR2 wrapper.
                grid,projection_valid=project_joint_grid(body.joints[:,1:24],original['camera'].reshape(-1,3))
        with torch.no_grad():
            delta=self.part_branch(rgb_tokens,original['fused_queries'],original['rotmat'],grid,projection_valid)
        pose2d_residual=self.pose2d_branch(rgb_tokens,original['fused_queries'].detach(),frame_valid)
        delta=delta*frame_valid.to(delta.device)[...,None,None]
        with torch.autocast(device_type=rgb_tokens.device.type,enabled=False):
            correction=axis_angle_matrix(delta.float())
            old=original['rotmat'].float()
            # Identity-centred expression exactly preserves trained v9 at initialization.
            change=(correction-torch.eye(3,device=old.device))@old[:,:,1:]
            rotations=torch.cat((old[:,:,:1],old[:,:,1:]+change),dim=2)
        result=dict(original)
        result.update(rotmat=rotations,part_delta_axis_angle=delta,
                      joint_grid=grid.reshape(batch,frames,23,2),
                      joint_projection_valid=projection_valid.reshape(batch,frames,23),
                      posetrack_2d_residual=pose2d_residual)
        return result

    def anchor_metadata(self):
        return dict(model_format=FORMAT,adaptation_config=asdict(self.adaptation_config),
                    reference='frozen validated v22, v19 or v9 decoder',
                    trainable='17-query spatial-temporal PoseTrack residual branch',
                    fixed_external='cached image/video features and native SMPL geometry asset')


def model_from_payload(payload, adaptation_config=None):
    """Strict v9/v19 initialization or native v25 model-weight restoration.

    v19 teacher weights are intentionally discarded; no teacher is constructed.
    Optimizer and scheduler resumption are the caller's responsibility.
    """
    config = RGBVaceHMR2AConfig(**payload['model_config'])
    source_format = payload.get('format')
    if source_format == FORMAT:
        resolved = AdaptationConfig(**payload['adaptation_config'])
        if adaptation_config is not None and asdict(adaptation_config) != asdict(resolved):
            raise ValueError('Requested adaptation configuration differs from checkpoint')
        model = HumanV9AdaptationV26(config, resolved)
        model.load_state_dict(payload['model'], strict=True)
    elif source_format == V25_FORMAT:
        resolved = adaptation_config or AdaptationConfig(**payload['adaptation_config'])
        model = HumanV9AdaptationV26(config, resolved)
        missing, unexpected = model.load_state_dict(payload['model'], strict=False)
        expected = [name for name in model.state_dict() if name.startswith('pose2d_branch.')]
        if missing != expected or unexpected:
            raise RuntimeError(f'Invalid v25 migration: missing={missing}, unexpected={unexpected}')
    elif source_format == V22_FORMAT:
        values = dict(payload['adaptation_config'])
        values['max_angle_degrees'] = 5.
        resolved = adaptation_config or AdaptationConfig(**values)
        model = HumanV9AdaptationV26(config, resolved)
        missing,unexpected=model.load_state_dict(payload['model'], strict=False)
        if missing != ['part_branch.part_logits'] or unexpected:
            raise RuntimeError(f'Invalid v22 migration: missing={missing}, unexpected={unexpected}')
    elif source_format in ('ram-geometry-priority-isolated-v9', 'ram-v9-joint-aligned-v19-coupled'):
        if source_format == 'ram-geometry-priority-isolated-v9':
            model = HumanV9AdaptationV25(config, adaptation_config)
            model.reference.load_state_dict(payload['model'], strict=True)
        else:
            values = dict(payload['adaptation_config'])
            values['max_angle_degrees'] = 5.
            resolved = AdaptationConfig(**values)
            if adaptation_config is not None and asdict(adaptation_config) != asdict(resolved):
                raise ValueError('Requested adaptation configuration differs from v19 checkpoint')
            model = HumanV9AdaptationV26(config, resolved)
            allowed = ('reference.', 'part_branch.', 'teacher_reference.')
            unexpected = [name for name in payload['model'] if not name.startswith(allowed)]
            if unexpected:
                raise ValueError(f'Unexpected v19 state keys: {unexpected[:5]}')
            state = {name: value for name, value in payload['model'].items()
                     if not name.startswith('teacher_reference.')}
            missing,unexpected=model.load_state_dict(state, strict=False)
            if missing != ['part_branch.part_logits'] or unexpected:
                raise RuntimeError(f'Invalid v19 migration: missing={missing}, unexpected={unexpected}')
    else:
        raise ValueError('Only complete v9, v19, v22 or v25 checkpoints are supported')
    if any(parameter.requires_grad for parameter in list(model.reference.parameters()) + list(model.part_branch.parameters())):
        raise RuntimeError('The v26 geometry path must remain frozen')
    if not all(parameter.requires_grad for parameter in model.pose2d_branch.parameters()):
        raise RuntimeError('The v26 PoseTrack branch must remain trainable')
    return model
