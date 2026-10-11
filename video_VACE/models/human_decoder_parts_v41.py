"""v41 copied from v39; appended projection-conditioned evidence correction.

Frozen v27 provides a candidate. The exact fallback is original HMR2a. The
trainable observation, visibility routing, root/camera and body update paths
share weights across iterations. Encoders run once. No legacy files are edited.
"""
from dataclasses import dataclass,asdict
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.human_decoder_parts_v27 import (
    HumanV9AdaptationV27,AdaptationConfig as V27Config,RGBVaceHMR2AConfig,
    PARTS,axis_angle_matrix,sample_joint_neighborhood)
from models.human_decoder_geometry import spatial_sincos_position_encoding
from models.body_projection_v39 import camera_project,project_landmarks,SCHEMAS

FORMAT='ram-iterative-anchor-routed-v39-lsp-head'

@dataclass(frozen=True)
class AdaptationConfig:
    hidden_dim:int=128
    num_heads:int=4
    iterations:int=2
    grid_height:int=16
    grid_width:int=12
    max_angle_degrees:float=8.
    camera_log_scale_bound:float=.15
    camera_translation_bound:float=.10
    local_radius:float=1.
    def validate(self):
        if self.num_heads<1 or self.hidden_dim<4 or self.hidden_dim%4 or self.hidden_dim%self.num_heads:
            raise ValueError('Invalid hidden width/heads')
        if self.iterations not in (1,2,3) or min(self.grid_height,self.grid_width)<1:
            raise ValueError('Invalid iteration count/grid')
        for x in (self.max_angle_degrees,self.camera_log_scale_bound,self.camera_translation_bound,self.local_radius):
            if not math.isfinite(x) or x<=0:
                raise ValueError('Bounds must be positive and finite')

def frozen_rotation_log(matrix):
    """Stable SO(3) log, including identity/pi; detached teacher states only."""
    if matrix.requires_grad:
        raise ValueError('Teacher rotation log must be detached')
    m=matrix.float()
    a,b,c=m[...,0,0],m[...,1,1],m[...,2,2]
    qabs=torch.stack((1+a+b+c,1+a-b-c,1-a+b-c,1-a-b+c),-1).clamp_min(0).sqrt()
    candidates=torch.stack((
        torch.stack((qabs[...,0]**2,m[...,2,1]-m[...,1,2],m[...,0,2]-m[...,2,0],m[...,1,0]-m[...,0,1]),-1),
        torch.stack((m[...,2,1]-m[...,1,2],qabs[...,1]**2,m[...,1,0]+m[...,0,1],m[...,0,2]+m[...,2,0]),-1),
        torch.stack((m[...,0,2]-m[...,2,0],m[...,1,0]+m[...,0,1],qabs[...,2]**2,m[...,2,1]+m[...,1,2]),-1),
        torch.stack((m[...,1,0]-m[...,0,1],m[...,2,0]+m[...,0,2],m[...,2,1]+m[...,1,2],qabs[...,3]**2),-1),
    ),-2)/(2*qabs.clamp_min(.1)[...,:,None])
    idx=qabs.argmax(-1)
    q=candidates.gather(-2,idx[...,None,None].expand(*idx.shape,1,4)).squeeze(-2)
    q=F.normalize(q,dim=-1)
    q=torch.where(q[...,:1]<0,-q,q)
    half=torch.atan2(torch.linalg.vector_norm(q[...,1:],dim=-1),q[...,0])
    return q[...,1:]*(2./torch.sinc(half/math.pi))[...,None]

class ObservationHead(nn.Module):
    def __init__(self,rgb_width,cfg):
        super().__init__()
        self.cfg=cfg
        d=cfg.hidden_dim
        self.projection=nn.Sequential(nn.LayerNorm(rgb_width),nn.Linear(rgb_width,d))
        self.queries=nn.Parameter(torch.randn(2,17,d)*.02)
        self.attention=nn.MultiheadAttention(d,cfg.num_heads,batch_first=True)
        self.norm=nn.LayerNorm(d)
        self.mlp=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d))
        self.coordinate=nn.ModuleList([nn.Linear(d,2) for _ in SCHEMAS])
        self.log_variance=nn.Linear(d,1)
        self.visibility=nn.Linear(d,1)
        for head in self.coordinate:
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
        nn.init.zeros_(self.log_variance.weight); nn.init.constant_(self.log_variance.bias,-4.)
        nn.init.zeros_(self.visibility.weight); nn.init.zeros_(self.visibility.bias)

    def forward(self,tokens,anchor_uv,schema):
        b,t,n,_=tokens.shape
        cfg=self.cfg
        if n!=cfg.grid_height*cfg.grid_width:
            raise ValueError('RGB grid does not match declared spatial layout')
        memory=self.projection(tokens).reshape(b*t,n,cfg.hidden_dim)
        pos=spatial_sincos_position_encoding(cfg.grid_height,cfg.grid_width,cfg.hidden_dim,tokens.device)
        positioned=memory+pos.reshape(1,n,cfg.hidden_dim).to(memory.dtype)
        idx=SCHEMAS.index(schema)
        query=self.queries[idx][None].expand(b*t,-1,-1)
        attended,_=self.attention(query,positioned,positioned,need_weights=False)
        feature=self.norm(query+attended)
        feature=(feature+self.mlp(feature)).reshape(b,t,17,cfg.hidden_dim)
        uv=anchor_uv.detach()+.3*torch.tanh(self.coordinate[idx](feature).float())
        logvar=self.log_variance(feature).float().clamp(-8.,2.)
        visibility=self.visibility(feature).float()
        maps=memory.transpose(1,2).reshape(b*t,cfg.hidden_dim,cfg.grid_height,cfg.grid_width)
        return uv,logvar,visibility,feature,maps

class IterativeGeometryCell(nn.Module):
    def __init__(self,vace_width,cfg):
        super().__init__()
        self.cfg=cfg
        d=cfg.hidden_dim
        self.state=nn.Linear(24,d) # R9 + camera3 + beta10 + native uv2
        self.evidence=nn.Linear(8,d) # projected2 + observed2 + error2 + logvar1 + visibility1
        self.local=nn.Linear(9*d+9,d)
        self.offset=nn.Linear(d,2)
        nn.init.zeros_(self.offset.weight); nn.init.zeros_(self.offset.bias)
        self.vace_projection=nn.Sequential(nn.LayerNorm(vace_width),nn.Linear(vace_width,d))
        self.parts=nn.Parameter(torch.randn(6,d)*.02)
        self.part_attention=nn.MultiheadAttention(d,cfg.num_heads,batch_first=True)
        self.temporal_attention=nn.MultiheadAttention(d,cfg.num_heads,batch_first=True)
        self.observation_attention=nn.MultiheadAttention(d,cfg.num_heads,batch_first=True)
        self.token_attention=nn.MultiheadAttention(d,cfg.num_heads,batch_first=True)
        self.route=nn.Linear(3*d+2,3)
        self.norm=nn.LayerNorm(d)
        self.mlp=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d))
        self.joint_queries=nn.Parameter(torch.randn(25,d)*.02)
        self.rotation=nn.Linear(d,3)
        self.camera=nn.Linear(d,3)
        self.gate=nn.Linear(d,1)
        self.static_gate_bias=nn.Parameter(torch.tensor(-6.))
        for head in (self.rotation,self.camera):
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
        nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias,3.)
        joint_parts=torch.full((24,),2,dtype=torch.long)
        for part,joints in enumerate(PARTS):
            joint_parts[list(joints)]=part
        self.register_buffer('joint_parts',joint_parts)

    def prepare_prior(self,features):
        b,f,h,w,_=features.shape
        memory=self.vace_projection(features)
        pos=spatial_sincos_position_encoding(h,w,self.cfg.hidden_dim,features.device)
        memory=memory+pos.reshape(1,1,h,w,-1).to(memory.dtype)
        ids=torch.arange(f,device=features.device).float()[:,None]
        freq=torch.exp(torch.arange(0,self.cfg.hidden_dim,2,device=features.device).float()*(-math.log(10000.)/self.cfg.hidden_dim))
        phase=ids*freq
        time=torch.stack((phase.sin(),phase.cos()),-1).flatten(-2)
        return (memory+time[None,:,None,None].to(memory.dtype)).reshape(b,f*h*w,-1)

    def forward(self,rotations,camera,betas,native_uv,native_valid,maps,obs_uv,projected,logvar,visibility,observation_features,prior,valid,independent):
        b,t=valid.shape
        d=self.cfg.hidden_dim
        state=torch.cat((rotations.flatten(-2),camera[...,None,:].expand(-1,-1,24,-1),
            betas[...,None,:].expand(-1,-1,24,-1),native_uv),-1)
        state=self.state(state)
        offset=torch.tanh(self.offset(state)).float()*native_uv.new_tensor([2./self.cfg.grid_width,2./self.cfg.grid_height])
        grid=native_uv*native_uv.new_tensor([8./3,2.])+offset
        local,sampled=sample_joint_neighborhood(maps,grid.reshape(b*t,24,2),
            (valid[...,None]&native_valid).reshape(b*t,24),self.cfg.local_radius)
        local=self.local(torch.cat((local.flatten(-2),sampled.float()),-1)).reshape(b,t,24,d)
        part_query=torch.stack([local[:,:,list(js)].mean(2) for js in PARTS],2)+self.parts
        pv,_=self.part_attention(part_query.reshape(b,t*6,d),prior,prior,need_weights=False)
        pv=pv.reshape(b,t,6,d)[:,:,self.joint_parts]
        seq=local.permute(0,2,1,3).reshape(b*24,t,d)
        tv,_=self.temporal_attention(seq,seq,seq,
            key_padding_mask=(~independent)[:,None].expand(b,24,t).reshape(b*24,t),need_weights=False)
        tv=tv.reshape(b,24,t,d).permute(0,2,1,3)
        has_temporal=(independent.sum(1)>1)[:,None,None,None]
        # Limb/ear semantics are shared by these two schemas; avoid neck/eye slots.
        part_landmarks=((11,13,15),(12,14,16),(5,6,11,12),(0,3,4),(5,7,9),(6,8,10))
        stats=torch.cat((logvar,visibility.sigmoid()),-1)
        stats=torch.stack([stats[:,:,list(js)].mean(2) for js in part_landmarks],2)[:,:,self.joint_parts]
        logits=self.route(torch.cat((local,pv,tv,stats),-1))
        logits=torch.cat((logits[...,:2],logits[...,2:].masked_fill(~has_temporal,-1e4)),-1)
        route=logits.softmax(-1)
        mixed=route[...,:1]*local+route[...,1:2]*pv+route[...,2:]*tv
        evidence=torch.cat((projected,obs_uv,obs_uv-projected,logvar,visibility.sigmoid()),-1)
        obs=observation_features+self.evidence(evidence)
        tokens=state+mixed
        tokens=torch.cat((tokens,tokens.mean(2,keepdim=True)),2)+self.joint_queries
        aligned,_=self.observation_attention(tokens.reshape(b*t,25,d),obs.reshape(b*t,17,d),obs.reshape(b*t,17,d),need_weights=False)
        tokens=tokens.reshape(b*t,25,d)+aligned
        cross,_=self.token_attention(self.norm(tokens),self.norm(tokens),self.norm(tokens),need_weights=False)
        tokens=tokens+cross
        tokens=(tokens+self.mlp(tokens)).reshape(b,t,25,d)
        rotation=torch.tanh(self.rotation(tokens[:,:,:24]).float())*math.radians(self.cfg.max_angle_degrees)/math.sqrt(3.)
        camera_delta=torch.tanh(self.camera(tokens[:,:,24]).float())
        camera_delta=camera_delta*camera_delta.new_tensor([self.cfg.camera_log_scale_bound,self.cfg.camera_translation_bound,self.cfg.camera_translation_bound])
        gate=torch.sigmoid(self.gate(tokens).float()+(~has_temporal).float()*self.static_gate_bias)
        return rotation,camera_delta,gate*valid[...,None,None],route

class HumanV9AdaptationV28(nn.Module):
    def __init__(self,config=None,adaptation_config=None,teacher_config=None):
        super().__init__()
        self.config=config or RGBVaceHMR2AConfig()
        self.adaptation_config=adaptation_config or AdaptationConfig()
        self.adaptation_config.validate()
        self.teacher_config=teacher_config or V27Config()
        self.teacher=HumanV9AdaptationV27(self.config,self.teacher_config).requires_grad_(False).eval()
        self.observation=ObservationHead(self.config.rgb_input_dim,self.adaptation_config)
        self.refiner=IterativeGeometryCell(self.config.vace_input_dim,self.adaptation_config)
        self.collect_diagnostics=False
        self.fusion_diagnostics={}

    def train(self,mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def bind_geometry(self,neutral_smpl):
        neutral_smpl.requires_grad_(False).eval()
        self.teacher.bind_geometry(neutral_smpl)
        object.__setattr__(self,'_neutral_smpl',neutral_smpl)
        return self

    def bind_landmarks(self,official_smpl):
        official_smpl.requires_grad_(False).eval()
        object.__setattr__(self,'_official_smpl',official_smpl)
        return self

    def forward(self,rgb_tokens,vace_features,base_rotmat,base_betas,base_camera,
                frame_valid=None,position_offset=0,landmark_schema='posetrack',source_frame_ids=None,force_anchor=False):
        if landmark_schema not in SCHEMAS or position_offset:
            raise ValueError('Unsupported schema/position perturbation')
        if not hasattr(self,'_official_smpl') or not hasattr(self,'_neutral_smpl'):
            raise RuntimeError('Bind native and official landmark SMPL before inference')
        b,t=base_rotmat.shape[:2]
        valid=frame_valid if frame_valid is not None else torch.ones(b,t,device=rgb_tokens.device,dtype=torch.bool)
        if valid.dtype!=torch.bool or valid.shape!=(b,t) or not valid.any(1).all():
            raise ValueError('Each sequence must contain a valid frame')
        independent=valid.clone()
        if source_frame_ids is not None:
            if source_frame_ids.shape!=(b,t):
                raise ValueError('source_frame_ids must be [B,T]')
            for i in range(1,t):
                independent[:,i]&=~((source_frame_ids[:,:i]==source_frame_ids[:,i,None])&valid[:,:i]).any(1)
        if base_betas.ndim==2:
            base_betas=base_betas[:,None].expand(-1,t,-1)
        if not torch.isfinite(base_camera).all() or not (base_camera[...,0]>0).all():
            raise ValueError('HMR camera must be finite with positive scale')
        anchor=dict(rotmat=base_rotmat.float(),betas=base_betas.float(),camera=base_camera.float())
        with torch.no_grad():
            candidate=self.teacher(rgb_tokens,vace_features,base_rotmat,base_betas,base_camera,frame_valid=valid)
            candidate_omega=frozen_rotation_log(candidate['rotmat']@anchor['rotmat'].transpose(-1,-2))
            candidate_camera=torch.cat(((candidate['camera'][...,:1].clamp_min(1e-6)/anchor['camera'][...,:1]).log(),
                candidate['camera'][...,1:]-anchor['camera'][...,1:]),-1)
            anchor_uv=project_landmarks(anchor,self._official_smpl,landmark_schema)
        obs_uv,logvar,visibility,obs_feature,maps=self.observation(rgb_tokens,anchor_uv,landmark_schema)
        prior=self.refiner.prepare_prior(vace_features)
        rotation_delta=torch.zeros_like(candidate_omega)
        camera_delta=torch.zeros_like(candidate_camera)
        state=anchor
        stages=[]
        for _ in range(self.adaptation_config.iterations):
            with torch.autocast(device_type=rgb_tokens.device.type,enabled=False):
                r=state['rotmat'].float().reshape(-1,24,3,3)
                body=self._neutral_smpl(global_orient=r[:,:1],body_pose=r[:,1:],betas=state['betas'].float().reshape(-1,10),pose2rot=False)
                native_uv,native_valid=camera_project(body.joints[:,:24],state['camera'].reshape(-1,3))
                projected=project_landmarks(state,self._official_smpl,landmark_schema)
            dr,dc,gates,routes=self.refiner(state['rotmat'],state['camera'],state['betas'],native_uv.reshape(b,t,24,2),native_valid.reshape(b,t,24),
                maps,obs_uv,projected,logvar,visibility,obs_feature,prior,valid,independent)
            rotation_delta=rotation_delta+dr/self.adaptation_config.iterations
            camera_delta=camera_delta+dc/self.adaptation_config.iterations
            if force_anchor:
                gates=gates*0.
            with torch.autocast(device_type=rgb_tokens.device.type,enabled=False):
                omega=gates[:,:,:24]*(candidate_omega+rotation_delta)
                rotations=axis_angle_matrix(omega)@anchor['rotmat']
                camres=gates[:,:,24]*(candidate_camera+camera_delta)
                camera=torch.cat((anchor['camera'][...,:1]*camres[...,:1].exp(),anchor['camera'][...,1:]+camres[...,1:]),-1)
                betas=anchor['betas']+gates[:,:,:24].mean(2)*(candidate['betas']-anchor['betas'])
                state=dict(rotmat=rotations,betas=betas,camera=camera)
                uv=project_landmarks(state,self._official_smpl,landmark_schema)
            stages.append(uv)
        result=dict(state)
        result.update(observed_2d=obs_uv,direct_2d=uv,stage_projections=torch.stack(stages),
            observation_log_variance=logvar,visibility_logits=visibility,anchor_2d=anchor_uv,
            anchor_prediction=anchor,teacher_prediction={k:candidate[k] for k in ('rotmat','betas','camera')},
            update_gates=gates,modality_routes=routes,frame_valid=valid,independent_frames=independent,
            rotation_update=rotation_delta,camera_update=camera_delta,
            posetrack_2d_residual=obs_uv-uv,posetrack_confidence=(-logvar.exp()).exp(),
            rotation_residual_6d=(rotations@anchor['rotmat'].transpose(-1,-2))[...,:2,:].flatten(-2),
            fused_queries=candidate['fused_queries'])
        return result

def model_from_payload(payload,adaptation_config=None):
    config=RGBVaceHMR2AConfig(**payload['model_config'])
    if payload['format'] in (FORMAT,'ram-iterative-anchor-routed-v28'):
        saved=AdaptationConfig(**payload['adaptation_config'])
        if adaptation_config is not None and asdict(adaptation_config)!=asdict(saved):
            raise ValueError('Requested v28 config differs from checkpoint')
        model=HumanV9AdaptationV28(config,saved,V27Config(**payload['teacher_config']))
        model.load_state_dict(payload['model'],strict=True)
    elif payload['format']=='ram-v9-geometry-coupled-v27':
        model=HumanV9AdaptationV28(config,adaptation_config,V27Config(**payload['adaptation_config']))
        model.teacher.load_state_dict(payload['model'],strict=True)
    else:
        raise ValueError('v28 requires a native v27 or v28 checkpoint')
    model.teacher.requires_grad_(False).eval()
    return model

# v39's copied source remains above for provenance; the frozen base below uses
# its unchanged original module. Only evidence/correction parameters train.
from models.human_decoder_parts_v39 import model_from_payload as load_frozen_v39
from models.joint_evidence_v41 import JointEvidenceConfig, JointEvidenceRefiner
from models.body_projection_v39 import POSETRACK_FROM_OPENPOSE, COCO_FROM_OPENPOSE
FORMAT='ram-v41-projection-conditioned-joint-evidence'

@dataclass(frozen=True)
class V41Config:
    hidden_dim:int=128
    num_heads:int=4
    temporal_enabled:bool=False
    enable_pose:bool=False
    pose_bound_degrees:float=2.
    camera_log_bound:float=.10
    camera_translation_bound:float=.04
    solver_damping:float=.01
    def validate(self):
        if self.num_heads<1 or self.hidden_dim<4 or self.hidden_dim%self.num_heads:
            raise ValueError('Invalid v41 hidden width/heads')
        for v in (self.pose_bound_degrees,self.camera_log_bound,self.camera_translation_bound,self.solver_damping):
            if not math.isfinite(v) or v<=0: raise ValueError('Invalid v41 bounds/damping')

def damped_camera_step(joints,camera,residual,confidence,damping):
    """One physical camera: Gauss-Newton increments [log scale, tx, ty]."""
    joints,camera,residual=joints.float(),camera.float(),residual.float()
    tz=10000./(256.*camera[...,:1].clamp_min(1e-6))
    depth=joints[...,2]+tz
    valid=torch.isfinite(joints).all(-1)&(depth>1e-4)
    d=depth.clamp_min(1e-4); k=5000./256.
    uv=k*(joints[...,:2]+camera[...,None,1:])/d[...,None]
    jac=joints.new_zeros(*joints.shape[:-1],2,3)
    jac[...,0,0]=uv[...,0]*tz/d; jac[...,1,0]=uv[...,1]*tz/d
    jac[...,0,1]=k/d; jac[...,1,2]=k/d
    w=confidence.detach().squeeze(-1).clamp(.05,1.)*valid
    w=w[...,None].expand(*w.shape,2).reshape(*camera.shape[:-1],-1,1)
    jac=jac.reshape(*camera.shape[:-1],-1,3)
    weighted=jac*w
    normal=jac.transpose(-1,-2)@weighted+damping*torch.eye(3,device=joints.device)
    rhs=weighted.transpose(-1,-2)@residual.reshape(*camera.shape[:-1],-1,1)
    return torch.linalg.solve(normal,rhs).squeeze(-1)

class EvidenceCorrection(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg; d=cfg.hidden_dim
        self.embed=nn.Sequential(nn.Linear(d+5,d),nn.GELU(),nn.LayerNorm(d))
        self.camera=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,3))
        self.camera_solver_gain=nn.Parameter(torch.zeros(()))
        self.joint_queries=nn.Parameter(torch.randn(24,d)*.02)
        self.joint_attention=nn.MultiheadAttention(d,cfg.num_heads,batch_first=True)
        self.rotation=nn.Linear(d,3)
        for head in (self.camera[-1],self.rotation):
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
        self.rotation.requires_grad_(cfg.enable_pose)
    def forward(self,evidence,projected,joints,camera,valid):
        confidence=evidence['confidence']; residual=evidence['observed_uv']-projected
        tokens=self.embed(torch.cat((evidence['features'],projected,residual,confidence.detach()),-1))
        weights=confidence.detach().clamp(.05,1.)
        pooled=(tokens*weights).sum(-2)/weights.sum(-2).clamp_min(1e-6)
        with torch.autocast(device_type=tokens.device.type,enabled=False):
            analytic=damped_camera_step(joints,camera,residual,confidence,self.cfg.solver_damping)
        limits=analytic.new_tensor([self.cfg.camera_log_bound,self.cfg.camera_translation_bound,self.cfg.camera_translation_bound])
        dc=torch.tanh(self.camera(pooled).float()+
            self.camera_solver_gain.clamp(0.,1.)*torch.tanh(analytic/limits))*limits
        b,t=valid.shape; query=self.joint_queries[None].expand(b*t,-1,-1)
        attended,_=self.joint_attention(query,tokens.reshape(b*t,17,-1),tokens.reshape(b*t,17,-1),need_weights=False)
        dr=torch.tanh(self.rotation(query+attended).float()).reshape(b,t,24,3)
        dr=dr*math.radians(self.cfg.pose_bound_degrees)/math.sqrt(3.)
        if not self.cfg.enable_pose: dr=dr*0.
        return dr*valid[...,None,None],dc*valid[...,None]

class HumanV41(nn.Module):
    def __init__(self,base,v41_config=None):
        super().__init__(); self.v41_config=v41_config or V41Config(); self.v41_config.validate()
        self.base=base.requires_grad_(False).eval()
        self.config,self.adaptation_config,self.teacher_config=base.config,base.adaptation_config,base.teacher_config
        cfg=self.v41_config
        self.evidence=JointEvidenceRefiner(self.config.rgb_input_dim,JointEvidenceConfig(
            hidden_dim=cfg.hidden_dim,num_heads=cfg.num_heads,temporal_enabled=cfg.temporal_enabled))
        self.correction=EvidenceCorrection(cfg)
        self.collect_diagnostics=False; self.fusion_diagnostics={}
    def train(self,mode=True):
        super().train(mode); self.base.eval(); return self
    def bind_geometry(self,smpl):
        self.base.bind_geometry(smpl); object.__setattr__(self,'_neutral_smpl',smpl); return self
    def bind_landmarks(self,smpl):
        self.base.bind_landmarks(smpl); object.__setattr__(self,'_official_smpl',smpl); return self
    def forward(self,rgb_tokens,vace_features,base_rotmat,base_betas,base_camera,
                frame_valid=None,position_offset=0,landmark_schema='posetrack',source_frame_ids=None,force_anchor=False):
        # New source identities affect only new evidence: coarse baseline is exact.
        with torch.no_grad():
            coarse=self.base(rgb_tokens,vace_features,base_rotmat,base_betas,base_camera,
                frame_valid=frame_valid,position_offset=position_offset,landmark_schema=landmark_schema,
                source_frame_ids=None,force_anchor=force_anchor)
        valid=coarse['frame_valid']; independent=valid.clone()
        if source_frame_ids is not None:
            for i in range(1,valid.shape[1]):
                independent[:,i]&=~((source_frame_ids[:,:i]==source_frame_ids[:,i,None])&valid[:,:i]).any(1)
        evidence=self.evidence(rgb_tokens,coarse['direct_2d'],valid,independent,source_frame_ids)
        b,t=valid.shape
        with torch.autocast(device_type=rgb_tokens.device.type,enabled=False):
            r=coarse['rotmat'].float().reshape(-1,24,3,3)
            body=self._official_smpl(global_orient=r[:,:1],body_pose=r[:,1:],betas=coarse['betas'].float().reshape(-1,10),pose2rot=False)
            indices=POSETRACK_FROM_OPENPOSE if landmark_schema=='posetrack' else COCO_FROM_OPENPOSE
            joints=body.joints[:,list(indices)].reshape(b,t,17,3)
        dr,dc=self.correction(evidence,coarse['direct_2d'],joints,coarse['camera'],valid)
        if force_anchor: dr,dc=dr*0.,dc*0.
        with torch.autocast(device_type=rgb_tokens.device.type,enabled=False):
            # camera stage guarantees exact geometry, even after its head trains.
            rotations=axis_angle_matrix(dr)@coarse['rotmat'] if self.v41_config.enable_pose else coarse['rotmat']
            camera=torch.cat((coarse['camera'][...,:1]*dc[...,:1].exp(),coarse['camera'][...,1:]+dc[...,1:]),-1)
            state=dict(rotmat=rotations,betas=coarse['betas'],camera=camera)
            uv=project_landmarks(state,self._official_smpl,landmark_schema)
        result=dict(coarse); result.update(state)
        result.update(observed_2d=evidence['observed_uv'],direct_2d=uv,stage_projections=uv[None],
            geometry_anchor={k:coarse[k] for k in state},rotation_update=dr,camera_update=dc,
            posetrack_2d_residual=evidence['observed_uv']-uv,posetrack_confidence=evidence['confidence'],
            evidence_logits=evidence['spatial_logits'],temporal_support=evidence['temporal_support'],independent_frames=independent)
        result['rotation_residual_6d']=(rotations@coarse['anchor_prediction']['rotmat'].transpose(-1,-2))[...,:2,:].flatten(-2)
        return result

def model_from_payload(payload,adaptation_config=None,v41_config=None,allow_pose_migration=False):
    if payload['format']==FORMAT:
        saved=V41Config(**payload['v41_config'])
        target=v41_config or saved
        if asdict(saved)!=asdict(target):
            expected=asdict(saved)|{'enable_pose':True}
            if not (allow_pose_migration and not saved.enable_pose and target.enable_pose and asdict(target)==expected):
                raise ValueError('v41 config differs from checkpoint; camera-to-pose requires explicit migration')
        legacy=dict(format='ram-iterative-anchor-routed-v39-lsp-head',model_config=payload['model_config'],
            adaptation_config=payload['adaptation_config'],teacher_config=payload['teacher_config'],
            model={k.removeprefix('base.'):v for k,v in payload['model'].items() if k.startswith('base.')})
        model=HumanV41(load_frozen_v39(legacy,adaptation_config),target)
        model.load_state_dict(payload['model'],strict=True); return model
    return HumanV41(load_frozen_v39(payload,adaptation_config),v41_config)
