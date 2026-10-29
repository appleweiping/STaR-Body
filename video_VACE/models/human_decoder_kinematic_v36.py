"""Native V36 new-phase model, strictly initialized from trained A5/V34."""

from dataclasses import dataclass,asdict,replace

import torch

from torch import nn

from models.human_decoder_target60_v34 import HumanTarget60V34,AdaptationConfig,FORMAT as PARENT_FORMAT

from models.part_time_refiner_v36 import PartTimeRefiner

from models.kinematic_ik_v36 import residual_ik,forward_kinematics

from models.human_decoder_spatiotemporal_v30 import rotation_6d

from models.human_decoder_parts_v29 import project_landmarks

FORMAT='star-body-joint-feedback-v37-r1'

@dataclass(frozen=True)

class V36Config:

    arm:str='R0'

    heads_unfreeze_step:int=300

    late_unfreeze_step:int=800

    temporal:bool=False

    skeleton:bool=False

    ik:bool=False

    shape_consensus:bool=False

    ik_start:int=300

    ik_ramp:int=500
    source_stride:int=2
    source_jitter:int=1
    feedback_mode:str='mean'

    def __post_init__(self):

        if self.arm not in ('R0','R1','R2','R3a','R3b','R4'):raise ValueError('Unknown V36 arm')

        if min(self.heads_unfreeze_step,self.late_unfreeze_step,self.ik_start)<0 or self.ik_ramp<1:raise ValueError('Bad phase schedule')

        if self.source_stride<1 or self.source_jitter<0:raise ValueError('Invalid source cadence')
        if self.feedback_mode not in ('mean','joint'):raise ValueError('Unknown feedback mode')
        if self.feedback_mode=='joint' and not self.temporal:raise ValueError('Joint feedback requires the R2-compatible refiner')
        if self.ik and not self.skeleton:raise ValueError('IK requires supervised skeleton branch')



class HumanKinematicV36(HumanTarget60V34):

    def __init__(self,parent,cfg):

        if parent.get('format')!=PARENT_FORMAT:raise ValueError('Strict trained native A5/V34 initialization required')

        ac=AdaptationConfig(**parent['adaptation_config'])

        if not ac.use_gru or not ac.use_track_shape:raise ValueError('V36 campaign requires A5 parent architecture')

        state=parent['model'];prefix='backbone.'

        base=parent['backbone_spec']|dict(model={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)})

        super().__init__(base,replace(ac,freeze_base=False))

        self.load_state_dict(state,strict=True)

        self.parent_spec={k:parent[k] for k in ('format','adaptation_config','backbone_spec','model_config','reference_spec')}

        self.v36_config=cfg;self.phase_step=0;self.parent_only=False

        if cfg.temporal:

            new=PartTimeRefiner(self.config.rgb_input_dim,self.backbone.adaptation_config.hidden_dim,self.adaptation_config,source_stride=cfg.source_stride,source_jitter=cfg.source_jitter,feedback_mode=cfg.feedback_mode)

            self.gru_migrated_keys=new.migrate(self.refiner);self.refiner=new

        if cfg.skeleton:

            self.joint_head=nn.Linear(self.backbone.adaptation_config.hidden_dim,3)

            self.twist_head=nn.Linear(self.backbone.adaptation_config.hidden_dim,1)

            for head in (self.joint_head,self.twist_head):nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)

        if cfg.shape_consensus:

            self.consensus=nn.Sequential(nn.LayerNorm(self.config.rgb_input_dim+10),nn.Linear(self.config.rgb_input_dim+10,11))

            nn.init.zeros_(self.consensus[-1].weight);nn.init.zeros_(self.consensus[-1].bias)

        self.configure_phase(0)

    def parameter_category(self,name):

        if not name.startswith('backbone.'):return 'adapter'

        key=name[len('backbone.'):]

        if key.startswith(('body_head.','root_head.','shape_head.','output_norm.')):return 'heads'

        if key.startswith(('blocks.2.','blocks.3.','feedback_projection.','image_attention.')):return 'late'

        return 'frozen'

    def configure_phase(self,step):

        self.phase_step=int(step);cfg=self.v36_config

        active=['adapter']

        if cfg.arm!='R0':

            if step>=cfg.heads_unfreeze_step:active.append('heads')

            if step>=cfg.late_unfreeze_step:active.append('late')

        for name,p in self.named_parameters():p.requires_grad_(self.parameter_category(name) in active)

        self.backbone.reference.requires_grad_(False).eval()

        return dict(new_phase_step=step,active_categories=active,

                    trainable_count=sum(p.numel() for p in self.parameters() if p.requires_grad))

    def train(self,mode=True):

        super().train(mode)

        # Parent used dropout=0, but keep all frozen backbone modules in eval.

        if mode:

            self.configure_phase(self.phase_step)

            for name,module in self.backbone.named_children():

                if not any(p.requires_grad for p in module.parameters()):module.eval()

        return self

    def forward(self,*values,force_anchor=False,**kw):

        if self.v36_config.temporal:

            object.__setattr__(self.refiner,'source_ids',kw.get('source_frame_ids'))

            object.__setattr__(self.refiner,'official_smpl',self._official_smpl)

            self.refiner.landmark_schema=kw.get('landmark_schema','posetrack')

        # force_anchor is the V36 parent-reproduction mode, NOT an untrained V28 fallback.

        if force_anchor or self.parent_only:

            if self.v36_config.temporal:

                from models.human_decoder_target60_v34 import GroupedRefiner

                old_forward=self.refiner.forward

                try:

                    self.refiner.forward=lambda *a,**k:GroupedRefiner.forward(self.refiner,*a,**k)

                    return super().forward(*values,force_anchor=False,**kw)

                finally:self.refiner.forward=old_forward

            return super().forward(*values,force_anchor=False,**kw)

        result=super().forward(*values,force_anchor=False,**kw)

        valid=result['frame_valid'];b,t=valid.shape;cfg=self.v36_config;changed=False

        if cfg.shape_consensus:

            x=torch.cat((kw['track_shape_features'],kw['track_shape_base']),-1)

            raw=self.consensus(x).float();route=min(1.,self.phase_step/500.)

            alpha=route*raw[...,:1].sigmoid();delta=.3*raw[...,1:].tanh()

            result['betas']=result['betas']+valid[...,None]*(alpha[:,None]*(kw['track_shape_base'][:,None]-result['betas'])+delta[:,None])

            result['shape_consensus_alpha']=alpha;changed=True

        if cfg.skeleton:

            with torch.autocast(device_type=valid.device.type,enabled=False):

                r=result['rotmat'].float().reshape(-1,24,3,3);betas=result['betas'].float().reshape(-1,10)

                body=self._neutral_smpl(global_orient=r[:,:1],body_pose=r[:,1:],betas=betas,pose2rot=False)

                identity=torch.eye(3,device=r.device).expand(len(r),24,3,3)

                rest=self._neutral_smpl(global_orient=identity[:,:1],body_pose=identity[:,1:],betas=betas,pose2rot=False).joints[:,:24]

                q=result['fused_queries'][:,:,:24]

                delta=.1*torch.tanh(self.joint_head(q).float())

                target=body.joints[:,:24].reshape(b,t,24,3)+delta

                result['skeleton_joints']=target-target[:,:,:1]

                result['skeleton_delta']=delta

                result['twist_radians']=.15*torch.tanh(self.twist_head(q).float().squeeze(-1))

                if cfg.ik:

                    ik=residual_ik(r,rest,target.reshape(-1,24,3),result['twist_radians'].reshape(-1,24)).reshape(b,t,24,3,3)

                    route=min(1.,max(0.,(self.phase_step-cfg.ik_start)/cfg.ik_ramp))

                    mixed=(1-route)*result['rotmat'].float()+route*ik

                    result['rotmat']=torch.where(valid[...,None,None,None],rotation_6d(mixed[...,:2,:].flatten(-2)),result['rotmat'])

                    result['ik_route']=route;changed=True

        if changed:

            state={k:result[k] for k in ('rotmat','betas','camera')}

            result['geometry_stages']=[*result['geometry_stages'][:-1],state]

            result['direct_2d']=project_landmarks(state,self._official_smpl,kw.get('landmark_schema','posetrack'))

            result['stage_projections']=torch.cat((result['stage_projections'][:-1],result['direct_2d'][None]),0)

            result['rotation_residual_6d']=(result['rotmat']@result['anchor_prediction']['rotmat'].transpose(-1,-2))[...,:2,:].flatten(-2)

        return result



def model_from_payload(payload,config=None):

    if payload.get('format')==PARENT_FORMAT:

        if payload.get('development_only',True) or payload.get('initial_fallback_only',True) or payload.get('step',0)<=0:

            raise ValueError('Parent must be a trained full-data checkpoint, not initial/development')

        return HumanKinematicV36(payload,config or V36Config())

    if payload.get('format')!=FORMAT:raise ValueError('Unsupported native V36 checkpoint')

    cfg=V36Config(**payload['v36_config'])

    if config is not None and asdict(config)!=asdict(cfg):raise ValueError('V36 architecture mismatch')

    parent=payload['parent_spec'];allowed=('backbone.','track_shape.','refiner.')

    # Rebuild the exact A5 parent using only its known tensors, no strict=False.

    ac=AdaptationConfig(**parent['adaptation_config'])

    base=parent['backbone_spec']|dict(model={k[len('backbone.'):]:v for k,v in payload['model'].items() if k.startswith('backbone.')})

    skeleton=HumanTarget60V34(base,ac)

    parent_state={k:payload['model'][k] for k in skeleton.state_dict()}

    skeleton.load_state_dict(parent_state,strict=True)

    model=HumanKinematicV36(parent|dict(model=parent_state),cfg)

    model.load_state_dict(payload['model'],strict=True);model.configure_phase(payload['step'])

    return model

