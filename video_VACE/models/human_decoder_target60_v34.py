"""Target60 V34: strict native V30 warm start and optional zero-init adapters."""
from dataclasses import dataclass,asdict
from contextlib import nullcontext
import math
import torch
from torch import nn
from models.human_decoder_spatiotemporal_v30 import model_from_payload as load_v30,FORMAT as V30_FORMAT
from models.human_decoder_parts_v29 import camera_project,sample_joint_neighborhood,project_landmarks
FORMAT='star-body-target60-v34'
CONTEXT_KEYS=('track_shape_base','track_shape_features')
GROUPS=((0,),(3,6,9,12,13,14,15),(16,18,20,22),(17,19,21,23),(1,4,7,10),(2,5,8,11))
@dataclass(frozen=True)
class AdaptationConfig:
    use_track_shape:bool=False
    use_gru:bool=False
    shape_bound:float=.3
    gru_width:int=128
    refinement_steps:int=3
    root_step_degrees:float=2.
    body_step_degrees:float=5.
    freeze_base:bool=False
    freeze_camera:bool=True
    def __post_init__(self):
        if not 32<=self.gru_width<=512 or not 1<=self.refinement_steps<=4:
            raise ValueError('Invalid GRU budget')
        if any(not math.isfinite(x) or x<=0 for x in (self.shape_bound,self.root_step_degrees,self.body_step_degrees)):
            raise ValueError('Finite positive residual bounds required')
        if self.freeze_base and not (self.use_track_shape or self.use_gru):
            raise ValueError('Frozen V30 requires an enabled adapter')

def bounded_rotation(raw,limit):
    # Smooth bounded axis-angle, no angle/norm singularity at zero.
    value=torch.tanh(raw.float());length=value.square().sum(-1,keepdim=True)
    omega=math.radians(limit)*value/torch.sqrt(1+length)
    x,y,z=omega.unbind(-1);zero=torch.zeros_like(x)
    skew=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(*x.shape,3,3)
    angle=omega.norm(dim=-1)
    a=torch.sinc(angle/math.pi)[...,None,None]
    b=(.5*torch.sinc(angle/(2*math.pi)).square())[...,None,None]
    return torch.eye(3,device=raw.device).expand_as(skew)+a*skew+b*(skew@skew)

class GroupedRefiner(nn.Module):
    def __init__(self,rgb_dim,query_dim,cfg):
        super().__init__();self.cfg=cfg
        self.rgb=nn.Sequential(nn.LayerNorm(rgb_dim),nn.Linear(rgb_dim,64))
        self.input=nn.Sequential(nn.LayerNorm(query_dim+64+22),nn.Linear(query_dim+64+22,cfg.gru_width),nn.GELU())
        self.cells=nn.ModuleList([nn.GRUCell(cfg.gru_width,cfg.gru_width) for _ in GROUPS])
        self.heads=nn.ModuleList([nn.Linear(cfg.gru_width,3) for _ in GROUPS])
        for head in self.heads:nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
    def forward(self,base,rgb_tokens,smpl,valid):
        b,t=valid.shape;cfg=self.cfg;state={k:base[k] for k in ('rotmat','betas','camera')}
        maps=self.rgb(rgb_tokens).reshape(b*t,16,12,64).permute(0,3,1,2)
        hidden=[None]*len(GROUPS);stages=[];raw_updates=[]
        for iteration in range(cfg.refinement_steps):
            with torch.autocast(device_type=valid.device.type,enabled=False):
                r=state['rotmat'].float().reshape(-1,24,3,3)
                body=smpl(global_orient=r[:,:1],body_pose=r[:,1:],betas=state['betas'].float().reshape(-1,10),pose2rot=False)
                uv,uv_valid=camera_project(body.joints[:,:24],state['camera'].reshape(-1,3))
            local,sampled=sample_joint_neighborhood(maps,uv*uv.new_tensor([8./3,2.]),
                uv_valid&valid.reshape(-1,1),[2.,1.,.5,.25][iteration])
            weights=sampled.float()[...,None]
            evidence=(local*weights).sum(-2)/weights.sum(-2).clamp_min(1)
            native=torch.cat((state['rotmat'].flatten(-2),state['betas'][:,:,None].expand(-1,-1,24,-1),
                state['camera'][:,:,None].expand(-1,-1,24,-1)),-1)
            encoded=self.input(torch.cat((base['fused_queries'][:,:,:24],evidence.reshape(b,t,24,64),native),-1))
            updates=torch.zeros_like(state['rotmat'][...,:3,0])
            for group,joints in enumerate(GROUPS):
                idx=torch.tensor(joints,device=valid.device);n=len(joints)
                values=encoded.index_select(2,idx).reshape(b*t*n,-1)
                hidden[group]=self.cells[group](values,hidden[group])
                delta=self.heads[group](hidden[group]).reshape(b,t,n,3)
                updates=updates.index_copy(2,idx,delta.float())
            root=bounded_rotation(updates[:,:,:1],cfg.root_step_degrees)
            limbs=bounded_rotation(updates[:,:,1:],cfg.body_step_degrees)
            with torch.autocast(device_type=valid.device.type,enabled=False):
                r=torch.cat((root,limbs),2)@state['rotmat'].float()
                state=state|dict(rotmat=torch.where(valid[...,None,None,None],r,state['rotmat']))
            stages.append(state);raw_updates.append(updates)
        return state,stages,torch.stack(raw_updates)

class HumanTarget60V34(nn.Module):
    def __init__(self,backbone_payload,adaptation_config=None):
        super().__init__();self.adaptation_config=adaptation_config or AdaptationConfig()
        self.backbone=load_v30(backbone_payload)
        self.backbone_spec={k:backbone_payload[k] for k in ('format','model_config','adaptation_config','reference_spec')}
        self.reference_spec=self.backbone.reference_spec;self.config=self.backbone.config
        cfg=self.adaptation_config
        if cfg.use_track_shape:
            self.track_shape=nn.Sequential(nn.LayerNorm(self.config.rgb_input_dim+10),
                nn.Linear(self.config.rgb_input_dim+10,128),nn.GELU(),nn.Linear(128,10))
            nn.init.zeros_(self.track_shape[-1].weight);nn.init.zeros_(self.track_shape[-1].bias)
        if cfg.use_gru:self.refiner=GroupedRefiner(self.config.rgb_input_dim,self.backbone.adaptation_config.hidden_dim,cfg)
        if cfg.freeze_base:self.backbone.requires_grad_(False)
        self.backbone.reference.requires_grad_(False).eval()
        if cfg.freeze_camera:self.backbone.camera_head.requires_grad_(False)
    def train(self,mode=True):
        super().train(mode);self.backbone.reference.eval()
        if self.adaptation_config.freeze_base:self.backbone.eval()
        return self
    def bind_geometry(self,smpl):
        self.backbone.bind_geometry(smpl);object.__setattr__(self,'_neutral_smpl',smpl);return self
    def bind_landmarks(self,smpl):
        self.backbone.bind_landmarks(smpl);object.__setattr__(self,'_official_smpl',smpl);return self
    def forward(self,*values,force_anchor=False,track_shape_base=None,track_shape_features=None,**kw):
        cfg=self.adaptation_config
        with torch.no_grad() if cfg.freeze_base else nullcontext():
            base=self.backbone(*values,**kw)
        if force_anchor:return base
        result=base.copy();valid=base['frame_valid'];b,t=valid.shape
        if cfg.use_track_shape:
            if track_shape_base is None or track_shape_features is None:raise ValueError('Audited V30 track prototype inputs required')
            if track_shape_base.shape!=(b,10) or track_shape_features.shape!=(b,self.config.rgb_input_dim):
                raise ValueError('Invalid track prototype shape')
            if not all(torch.isfinite(x).all() for x in (track_shape_base,track_shape_features)):
                raise ValueError('Nonfinite track prototypes')
            delta=cfg.shape_bound*torch.tanh(self.track_shape(torch.cat((track_shape_features,track_shape_base),-1)).float())
            result['betas']=base['betas'].float()+delta[:,None]*valid[...,None]
            result['track_shape_update']=delta
        if cfg.use_gru:
            state,extra,raw=self.refiner(result,values[0],self._neutral_smpl,valid)
            result.update(state);result['refinement_updates']=raw
            result['geometry_stages']=[*base['geometry_stages'][:-1],state]
            result['refinement_stages']=extra
        elif cfg.use_track_shape:
            result['geometry_stages']=[*base['geometry_stages'][:-1],{k:result[k] for k in ('rotmat','betas','camera')}]
        if cfg.use_track_shape or cfg.use_gru:
            result['direct_2d']=project_landmarks(result,self._official_smpl,kw.get('landmark_schema','posetrack'))
            result['stage_projections']=torch.cat((base['stage_projections'][:-1],result['direct_2d'][None]),0)
            result['posetrack_2d_residual']=base['observed_2d']-result['direct_2d']
            result['rotation_residual_6d']=(result['rotmat']@base['anchor_prediction']['rotmat'].transpose(-1,-2))[...,:2,:].flatten(-2)
        return result

def model_from_payload(payload,adaptation_config=None):
    if payload.get('format')==V30_FORMAT:return HumanTarget60V34(payload,adaptation_config)
    if payload.get('format')!=FORMAT:raise ValueError('V34 requires trained native V30 or native V34; no format relabeling')
    cfg=AdaptationConfig(**payload['adaptation_config'])
    if adaptation_config is not None and asdict(cfg)!=asdict(adaptation_config):raise ValueError('Native V34 architecture mismatch')
    state=payload['model'];prefix='backbone.'
    base=payload['backbone_spec']|dict(model={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)})
    model=HumanTarget60V34(base,cfg);model.load_state_dict(state,strict=True)
    return model
