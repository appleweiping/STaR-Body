"""Copied v9-reference wrapper with six spatial RGB body-part queries."""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from models.human_decoder_rgb_vace_v9_isolated import HumanRGBVaceGeometryTransformerV9Isolated, RGBVaceHMR2AConfig
from models.human_decoder_geometry import spatial_sincos_position_encoding

FORMAT = 'ram-v9-spatial-parts-v15'
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
    def validate(self):
        if self.position_mode != 'original':
            raise ValueError('v15 preserves original reference slot embeddings')
        if min(self.hidden_dim,self.num_heads,self.grid_height,self.grid_width) < 1:
            raise ValueError('Invalid part dimensions')
        if self.hidden_dim % self.num_heads or self.hidden_dim % 4:
            raise ValueError('Part width must be divisible by heads and four')
        if not math.isfinite(self.max_angle_degrees) or not 0 < self.max_angle_degrees <= 30:
            raise ValueError('Angle bound must be in (0,30] degrees')

def axis_angle_matrix(vector):
    """Stable Rodrigues, finite derivative at exactly zero."""
    theta = torch.linalg.vector_norm(vector,dim=-1)
    x,y,z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),dim=-1).reshape(*vector.shape[:-1],3,3)
    a = torch.sinc(theta / math.pi)[...,None,None]
    b = .5 * torch.sinc(theta / (2*math.pi)).square()[...,None,None]
    return torch.eye(3,device=vector.device,dtype=vector.dtype)+a*skew+b*(skew@skew)

class SpatialPartBranch(nn.Module):
    def __init__(self, rgb_width, context_width, config):
        super().__init__()
        d=config.hidden_dim
        self.config=config
        self.rgb_projection=nn.Sequential(nn.LayerNorm(rgb_width),nn.Linear(rgb_width,d))
        self.context_projection=nn.Linear(context_width,d)
        self.queries=nn.Parameter(torch.randn(6,d)*.02)
        self.query_norm=nn.LayerNorm(d)
        self.memory_norm=nn.LayerNorm(d)
        self.attention=nn.MultiheadAttention(d,config.num_heads,batch_first=True,dropout=0.)
        self.mlp=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d))
        self.heads=nn.ModuleList(nn.Linear(d,len(part)*3) for part in PARTS)
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self,tokens,context):
        batch,frames,count,_=tokens.shape
        cfg=self.config
        if count != cfg.grid_height*cfg.grid_width:
            raise ValueError('RGB token count differs from the declared spatial grid')
        memory=self.rgb_projection(tokens.detach()).reshape(batch*frames,count,cfg.hidden_dim)
        position=spatial_sincos_position_encoding(cfg.grid_height,cfg.grid_width,cfg.hidden_dim,tokens.device)
        memory=memory+position.reshape(1,count,cfg.hidden_dim).to(memory.dtype)
        queries=self.queries[None]+self.context_projection(context.detach()).reshape(batch*frames,1,-1)
        attended,_=self.attention(self.query_norm(queries),self.memory_norm(memory),
                                  self.memory_norm(memory),need_weights=False)
        queries=queries+attended
        queries=queries+self.mlp(queries)
        with torch.autocast(device_type=tokens.device.type,enabled=False):
            chunks=[head(queries[:,i].float()).reshape(batch,frames,len(part),3)
                    for i,(head,part) in enumerate(zip(self.heads,PARTS))]
            by_joint={joint:chunk[:,:,i] for part,chunk in zip(PARTS,chunks) for i,joint in enumerate(part)}
            raw=torch.stack([by_joint[j] for j in range(1,24)],dim=2)
            return math.radians(cfg.max_angle_degrees)/math.sqrt(3)*torch.tanh(raw)

class HumanV9AdaptationV15(nn.Module):
    def __init__(self,config=None,adaptation_config=None):
        super().__init__()
        self.config=config or RGBVaceHMR2AConfig()
        self.adaptation_config=adaptation_config or AdaptationConfig()
        self.adaptation_config.validate()
        self.reference=HumanRGBVaceGeometryTransformerV9Isolated(self.config)
        self.reference.requires_grad_(False)
        self.reference.eval()
        self.part_branch=SpatialPartBranch(self.config.rgb_input_dim,self.config.hidden_dim,self.adaptation_config)

    def train(self,mode=True):
        super().train(mode)
        self.reference.eval()
        return self

    def forward(self,rgb_tokens,vace_features,base_rotmat,base_betas,base_camera,
                frame_valid=None,position_offset=0):
        if position_offset:
            raise ValueError('Position perturbation is not part of the v15 spatial ablation')
        with torch.no_grad():
            original=self.reference(rgb_tokens,vace_features,base_rotmat,base_betas,base_camera)
        batch,frames=base_rotmat.shape[:2]
        if frame_valid is None:
            frame_valid=torch.ones(batch,frames,dtype=torch.bool,device=base_rotmat.device)
        if frame_valid.shape != (batch,frames) or frame_valid.dtype != torch.bool or not frame_valid.any(1).all():
            raise ValueError('Expected a Boolean [B,T] mask with valid targets in each window')
        delta=self.part_branch(rgb_tokens,original['fused_queries'])
        delta=delta*frame_valid.to(delta.device)[...,None,None]
        with torch.autocast(device_type=rgb_tokens.device.type,enabled=False):
            correction=axis_angle_matrix(delta.float())
            old=original['rotmat'].float()
            # Identity-centred expression exactly preserves trained v9 at initialization.
            change=(correction-torch.eye(3,device=old.device))@old[:,:,1:]
            rotations=torch.cat((old[:,:,:1],old[:,:,1:]+change),dim=2)
        result=dict(original)
        result.update(rotmat=rotations,part_delta_axis_angle=delta,
                      teacher_rotmat=original['rotmat'],teacher_betas=original['betas'],
                      teacher_camera=original['camera'])
        return result

    def anchor_metadata(self):
        return dict(model_format=FORMAT,adaptation_config=asdict(self.adaptation_config),
                    reference='immutable trained v9',trainable='six RGB part queries and readout')
