"""v20: independent last VACE gate and controlled local-context gradients."""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize
from models.human_decoder_rgb_vace_v9_isolated import HumanRGBVaceGeometryTransformerV9Isolated, RGBVaceHMR2AConfig
from models.human_decoder_geometry import spatial_sincos_position_encoding

FORMAT = 'ram-v9-joint-aligned-v20-gated'
# SMPL indices: six disjoint groups cover all 23 non-root joints.
PARTS = ((1,4,7,10),(2,5,8,11),(3,6,9),(12,15),(13,16,18,20,22),(14,17,19,21,23))

@dataclass(frozen=True)
class AdaptationConfig:
    position_mode: str = 'original'
    hidden_dim: int = 128
    num_heads: int = 4
    grid_height: int = 16
    grid_width: int = 12
    max_angle_degrees: float = 10.
    local_radius: float = 1.
    fusion_mode: str = 'block_gate'
    detach_context: bool = False
    local_mode: str = 'image'
    def validate(self):
        if self.fusion_mode not in ('frozen', 'block', 'block_gate'):
            raise ValueError('Unknown fusion mode')
        if self.local_mode not in ('image', 'bias') or type(self.detach_context) is not bool:
            raise ValueError('Invalid local mode or context-gradient switch')
        if self.position_mode != 'original':
            raise ValueError('v20 preserves original reference slot embeddings')
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
        memory=self.rgb_projection(tokens.detach()).reshape(batch*frames,count,cfg.hidden_dim)
        feature_map=memory.transpose(1,2).reshape(batch*frames,cfg.hidden_dim,cfg.grid_height,cfg.grid_width)
        local,mask=sample_joint_neighborhood(feature_map,grid.reshape(-1,23,2),
                                            projection_valid.reshape(-1,23),cfg.local_radius)
        position=spatial_sincos_position_encoding(cfg.grid_height,cfg.grid_width,cfg.hidden_dim,tokens.device)
        memory=memory+position.reshape(1,count,cfg.hidden_dim).to(memory.dtype)
        query_context = context.detach() if cfg.detach_context else context
        queries=self.queries[None]+self.context_projection(query_context).reshape(batch*frames,1,-1)
        pose=rotations.detach().reshape(-1,24,3,3)
        pose=torch.cat((pose[:,1:].flatten(-2),pose[:,self.parents].flatten(-2)),dim=-1)
        queries=queries+self.local_projection(local.flatten(-2).to(memory.dtype))
        queries=queries+self.pose_projection(pose.to(memory.dtype))+self.valid_projection(mask.to(memory.dtype))
        attended,_=self.attention(self.query_norm(queries),self.memory_norm(memory),
                                  self.memory_norm(memory),need_weights=False)
        queries=queries+attended
        queries=queries+self.mlp(queries)
        with torch.autocast(device_type=tokens.device.type,enabled=False):
            raw=self.heads[0](queries.float()).reshape(batch,frames,23,3)
            return math.radians(cfg.max_angle_degrees)/math.sqrt(3)*torch.tanh(raw)

class LastTemporalGate(nn.Module):
    """Only the last scalar is trainable; earlier gates remain frozen storage."""
    def __init__(self, gates):
        super().__init__()
        self.last_gate = nn.Parameter(gates[-1:].detach().clone())

    def forward(self, original):
        return torch.cat((original[:-1], self.last_gate))


class JointBiasBranch(nn.Module):
    """Image-independent control with 69 learned rotation-bias parameters."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bias = nn.Parameter(torch.zeros(23, 3))

    def forward(self, tokens, context, rotations, grid, projection_valid):
        bound = math.radians(self.config.max_angle_degrees) / math.sqrt(3)
        return (bound * torch.tanh(self.bias.float()))[None, None].expand(
            tokens.shape[0], tokens.shape[1], -1, -1)


class HumanV9AdaptationV20(nn.Module):
    def __init__(self,config=None,adaptation_config=None):
        super().__init__()
        self.config=config or RGBVaceHMR2AConfig()
        self.adaptation_config=adaptation_config or AdaptationConfig()
        self.adaptation_config.validate()
        # ``reference`` is initialized from v9 and may expose only its final
        # temporal-fusion block.  ``teacher_reference`` remains an immutable
        # copy of v9 so the geometry-distance term still constrains fusion.
        self.reference=HumanRGBVaceGeometryTransformerV9Isolated(self.config)
        self.teacher_reference=HumanRGBVaceGeometryTransformerV9Isolated(self.config)
        parametrize.register_parametrization(self.reference, 'temporal_gates',
                                             LastTemporalGate(self.reference.temporal_gates))
        self.reference.requires_grad_(False)
        self.teacher_reference.requires_grad_(False)
        self.reference.eval()
        self.teacher_reference.eval()
        self.part_branch=(JointBiasBranch(self.adaptation_config) if self.adaptation_config.local_mode == 'bias'
            else SpatialPartBranch(self.config.rgb_input_dim,self.config.hidden_dim,self.adaptation_config))
        self.collect_diagnostics = False
        self.fusion_diagnostics = {}
        self.reference.vace_blocks[-1].register_forward_hook(self._record_fusion)

    @property
    def last_temporal_gate(self):
        return self.reference.parametrizations.temporal_gates[0].last_gate

    def _record_fusion(self, module, inputs, proposal):
        if self.collect_diagnostics:
            with torch.no_grad():
                query = inputs[0].detach().float()
                update = torch.tanh(self.last_temporal_gate.detach().float()) * (proposal.detach().float()-query)
                self.fusion_diagnostics = {
                    'gate_tanh': float(torch.tanh(self.last_temporal_gate.detach()).item()),
                    'fusion_update_ratio': float(update.norm()/query.norm().clamp_min(1e-12)),
                }

    def set_fusion_trainable(self, enabled):
        """Warm-up freezes both block and gate, including stale gradients."""
        mode = self.adaptation_config.fusion_mode
        for parameter in self.reference.vace_blocks[-1].parameters():
            parameter.requires_grad_(enabled and mode != 'frozen')
            if not parameter.requires_grad:
                parameter.grad = None
        self.last_temporal_gate.requires_grad_(enabled and mode == 'block_gate')
        if not self.last_temporal_gate.requires_grad:
            self.last_temporal_gate.grad = None
        return self

    def bind_geometry(self, neutral_smpl):
        # External frozen asset, deliberately excluded from the learned checkpoint.
        neutral_smpl.requires_grad_(False).eval()
        object.__setattr__(self, '_neutral_smpl', neutral_smpl)
        return self

    def train(self,mode=True):
        super().train(mode)
        self.reference.eval()
        self.teacher_reference.eval()
        # Training behavior is enabled only for the intentionally exposed block.
        if any(parameter.requires_grad for parameter in self.reference.vace_blocks[-1].parameters()):
            self.reference.vace_blocks[-1].train(mode)
        return self

    def forward(self,rgb_tokens,vace_features,base_rotmat,base_betas,base_camera,
                frame_valid=None,position_offset=0):
        if position_offset:
            raise ValueError('Position perturbation is not part of the v20 experiment')
        original=self.reference(rgb_tokens,vace_features,base_rotmat,base_betas,base_camera)
        with torch.no_grad():
            teacher=self.teacher_reference(rgb_tokens,vace_features,base_rotmat,base_betas,base_camera)
            if not hasattr(self,'_neutral_smpl'):
                raise RuntimeError('Bind a neutral native-SMPL model before v20 inference')
            with torch.autocast(device_type=rgb_tokens.device.type,enabled=False):
                r=original['rotmat'].float().reshape(-1,24,3,3)
                body=self._neutral_smpl(global_orient=r[:,:1],body_pose=r[:,1:],
                     betas=original['betas'].float().reshape(-1,10),pose2rot=False)
                # Native SMPL joints 0:24, not the OpenPose-reordered HMR2 wrapper.
                grid,projection_valid=project_joint_grid(body.joints[:,1:24],original['camera'].reshape(-1,3))
        batch,frames=base_rotmat.shape[:2]
        if frame_valid is None:
            frame_valid=torch.ones(batch,frames,dtype=torch.bool,device=base_rotmat.device)
        if frame_valid.shape != (batch,frames) or frame_valid.dtype != torch.bool or not frame_valid.any(1).all():
            raise ValueError('Expected a Boolean [B,T] mask with valid targets in each window')
        delta=self.part_branch(rgb_tokens,original['fused_queries'],original['rotmat'],grid,projection_valid)
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
                      teacher_rotmat=teacher['rotmat'],teacher_betas=teacher['betas'],
                      teacher_camera=teacher['camera'])
        return result

    def anchor_metadata(self):
        return dict(model_format=FORMAT,adaptation_config=asdict(self.adaptation_config),
                    reference='trained v9 with final fusion block optionally exposed',
                    teacher='immutable trained v9',
                    trainable='local branch, optional final VACE block and independent last gate')


def model_from_payload(payload, adaptation_config=None):
    """Strict migration: base v9, warm-start v19, or native v20 (not optimizer resume)."""
    config = RGBVaceHMR2AConfig(**payload['model_config'])
    source_format = payload.get('format')
    if source_format == FORMAT:
        resolved = AdaptationConfig(**payload['adaptation_config'])
        if adaptation_config is not None and asdict(adaptation_config) != asdict(resolved):
            raise ValueError('Requested adaptation configuration differs from checkpoint')
        model = HumanV9AdaptationV20(config, resolved)
        model.load_state_dict(payload['model'], strict=True)
    elif source_format in ('ram-geometry-priority-isolated-v9', 'ram-v9-joint-aligned-v19-coupled'):
        resolved = adaptation_config
        if resolved is None and source_format.endswith('v19-coupled'):
            resolved = AdaptationConfig(**payload['adaptation_config'])
        model = HumanV9AdaptationV20(config, resolved)
        if source_format == 'ram-geometry-priority-isolated-v9':
            state = dict(payload['model'])
            gates = state.pop('temporal_gates')
            state['parametrizations.temporal_gates.original'] = gates
            state['parametrizations.temporal_gates.0.last_gate'] = gates[-1:].clone()
            model.reference.load_state_dict(state, strict=True)
            model.teacher_reference.load_state_dict(payload['model'], strict=True)
        else:
            if model.adaptation_config.local_mode != 'image':
                raise ValueError('Bias control must start from v9, not a trained image branch')
            state = dict(payload['model'])
            gates = state.pop('reference.temporal_gates')
            state['reference.parametrizations.temporal_gates.original'] = gates
            state['reference.parametrizations.temporal_gates.0.last_gate'] = gates[-1:].clone()
            model.load_state_dict(state, strict=True)
    else:
        raise ValueError('Only complete v9, v19 or v20 checkpoints are supported')
    model.set_fusion_trainable(False)
    return model
