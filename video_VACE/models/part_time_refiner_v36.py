"""A5-compatible iterative GRU with masked part/time residual and explicit feedback.



Old parameter keys are preserved. New pathways start as exact zero residuals.

Temporal attention never crosses duplicate IDs, invalid frames, or track gaps.

"""

import torch

from torch import nn

from models.human_decoder_target60_v34 import GroupedRefiner,GROUPS,bounded_rotation
from models.joint_feedback_v37 import make_joint_feedback

from models.human_decoder_parts_v29 import camera_project,sample_joint_neighborhood,project_landmarks



def temporal_allowed(ids,valid,source_stride=1,source_jitter=0):
    if ids.shape!=valid.shape or valid.dtype!=torch.bool:raise ValueError('Bad temporal metadata')
    if source_stride<1 or source_jitter<0:raise ValueError('Bad declared source cadence')
    b,t=ids.shape;independent=valid.clone()
    for i in range(1,t):
        independent[:,i]&=~((ids[:,:i]==ids[:,i,None])&valid[:,:i]).any(1)
    # Source IDs are mapped video IDs, NOT annotation IDs. Current audited cache
    # has nominal delta2 and rounded delta3; duplicates are skipped, not gaps.
    previous=torch.zeros(b,device=ids.device,dtype=ids.dtype)
    has_previous=torch.zeros(b,device=ids.device,dtype=torch.bool)
    segment=torch.zeros(b,device=ids.device,dtype=torch.long);segments=[]
    for i in range(t):
        fresh=independent[:,i];delta=ids[:,i]-previous
        connected=has_previous&(delta>=source_stride)&(delta<=source_stride+source_jitter)
        segment=segment+(fresh&~connected).long();segments.append(segment)
        previous=torch.where(fresh,ids[:,i],previous)
        has_previous=(has_previous|fresh)&valid[:,i]
    segments=torch.stack(segments,1)
    allowed=(segments[:,:,None]==segments[:,None,:])&independent[:,:,None]&independent[:,None,:]
    allowed|=torch.eye(t,device=ids.device,dtype=torch.bool)[None]
    return allowed,independent

class PartTimeBlock(nn.Module):

    def __init__(self,width,source_stride=2,source_jitter=1):

        super().__init__();self.norm=nn.LayerNorm(width)
        self.source_stride=source_stride;self.source_jitter=source_jitter

        self.time=nn.MultiheadAttention(width,4,dropout=0.,batch_first=True)

        self.part=nn.MultiheadAttention(width,4,dropout=0.,batch_first=True)

        self.project=nn.Linear(width,width)

        nn.init.zeros_(self.project.weight);nn.init.zeros_(self.project.bias)

    def forward(self,x,ids,valid):

        b,t,g,d=x.shape;allowed,independent=temporal_allowed(ids,valid,self.source_stride,self.source_jitter)

        z=self.norm(x);mask=(~allowed)[:,None].expand(b,g,t,t).reshape(b*g,t,t)

        mask=mask.repeat_interleave(4,0)

        temporal=z.permute(0,2,1,3).reshape(b*g,t,d)

        temporal=self.time(temporal,temporal,temporal,attn_mask=mask,need_weights=False)[0]

        temporal=temporal.reshape(b,g,t,d).permute(0,2,1,3)

        # Part attention stays within one frame. No temporal leakage via part mixing.

        part=z.reshape(b*t,g,d);part=self.part(part,part,part,need_weights=False)[0].reshape(b,t,g,d)

        return x+self.project(temporal+part)*independent[:,:,None,None]



class PartTimeRefiner(GroupedRefiner):

    def __init__(self,rgb_dim,query_dim,cfg,source_stride=2,source_jitter=1,feedback_mode="mean"):

        super().__init__(rgb_dim,query_dim,cfg)
        self.source_stride=source_stride;self.source_jitter=source_jitter

        self.feedback=nn.Linear(4,cfg.gru_width,bias=False)
        self.feedback_mode=feedback_mode
        if feedback_mode not in ("mean","joint"):raise ValueError("Invalid feedback mode")


        self.time_input=nn.Linear(2,cfg.gru_width,bias=False)

        self.temporal=nn.ModuleList([PartTimeBlock(cfg.gru_width,source_stride,source_jitter) for _ in range(2)])

        nn.init.zeros_(self.feedback.weight);nn.init.zeros_(self.time_input.weight)

        self.source_ids=None;self.official_smpl=None;self.landmark_schema='posetrack'
        if feedback_mode=='joint':
            # Preserve all inherited/new R2 RNG draws outside the added module.
            self.joint_feedback=make_joint_feedback(cfg.gru_width)

    def migrate(self,old):

        inherited=old.state_dict();state=self.state_dict()

        if not inherited.keys()<=state.keys():raise ValueError('Old GRU tensors lost during migration')

        for key,value in inherited.items():

            if state[key].shape!=value.shape:raise ValueError('GRU migration shape mismatch: '+key)

            state[key]=value

        self.load_state_dict(state,strict=True)

        return sorted(inherited)

    def forward(self,base,rgb_tokens,smpl,valid):

        b,t=valid.shape;cfg=self.cfg

        if self.source_ids is None or self.official_smpl is None:raise ValueError('Missing explicit temporal context')

        ids=self.source_ids;allowed,independent=temporal_allowed(ids,valid,self.source_stride,self.source_jitter)

        state={k:base[k] for k in ('rotmat','betas','camera')}

        maps=self.rgb(rgb_tokens).reshape(b*t,16,12,64).permute(0,3,1,2)

        hidden=[None]*len(GROUPS);stages=[];raw_updates=[]

        dt=torch.zeros(b,t,device=ids.device);dt[:,1:]=(ids[:,1:]-ids[:,:-1]).float().clamp(-32,32)/32.

        time=self.time_input(torch.stack((dt,independent.float()),-1))

        for iteration in range(cfg.refinement_steps):

            with torch.autocast(device_type=valid.device.type,enabled=False):

                r=state['rotmat'].float().reshape(-1,24,3,3)

                body=smpl(global_orient=r[:,:1],body_pose=r[:,1:],betas=state['betas'].float().reshape(-1,10),pose2rot=False)

                uv,uv_valid=camera_project(body.joints[:,:24],state['camera'].reshape(-1,3))

                projected=project_landmarks(state,self.official_smpl,self.landmark_schema)

                # Existing observation heads are NOT calibrated 2D labels.

                residual=(base['observed_2d'].detach()-projected).clamp(-2,2)

                logvar=base['observation_log_variance'].detach().clamp(-6,6)

                visibility=base['visibility_logits'].detach().sigmoid()

                feedback=torch.cat((residual.mean(-2),logvar.mean(-2),visibility.mean(-2)),-1)
                if self.feedback_mode=='joint':
                    feedback_valid=valid&torch.isfinite(state['camera']).all(-1)&(state['camera'][...,0]>0)
                    joint_feedback=self.joint_feedback(base['observed_2d'].detach()-projected,projected,base['observation_log_variance'].detach(),visibility,feedback_valid,schema=self.landmark_schema)

            local,sampled=sample_joint_neighborhood(maps,uv*uv.new_tensor([8./3,2.]),

                uv_valid&valid.reshape(-1,1),[2.,1.,.5,.25][iteration])

            weights=sampled.float()[...,None]

            evidence=(local*weights).sum(-2)/weights.sum(-2).clamp_min(1)

            native=torch.cat((state['rotmat'].flatten(-2),state['betas'][:,:,None].expand(-1,-1,24,-1),

                state['camera'][:,:,None].expand(-1,-1,24,-1)),-1)

            encoded=self.input(torch.cat((base['fused_queries'][:,:,:24],evidence.reshape(b,t,24,64),native),-1))

            if self.feedback_mode=='joint':
                encoded=encoded+joint_feedback+time[:,:,None]
            else:
                encoded=encoded+(self.feedback(feedback)+time)[:,:,None]

            parts=torch.stack([encoded[:,:,list(joints)].mean(2) for joints in GROUPS],2)

            mixed=parts

            for block in self.temporal:mixed=block(mixed,ids,valid)

            updates=torch.zeros_like(state['rotmat'][...,:3,0])

            for group,joints in enumerate(GROUPS):

                idx=torch.tensor(joints,device=valid.device);n=len(joints)

                values=(encoded.index_select(2,idx)+(mixed[:,:,group]-parts[:,:,group])[:,:,None]).reshape(b*t*n,-1)

                hidden[group]=self.cells[group](values,hidden[group])

                delta=self.heads[group](hidden[group]).reshape(b,t,n,3)

                updates=updates.index_copy(2,idx,delta.float())

            with torch.autocast(device_type=valid.device.type,enabled=False):

                delta=torch.cat((bounded_rotation(updates[:,:,:1],cfg.root_step_degrees),

                                 bounded_rotation(updates[:,:,1:],cfg.body_step_degrees)),2)

                state=state|dict(rotmat=torch.where(valid[...,None,None,None],delta@state['rotmat'].float(),state['rotmat']))

            stages.append(state);raw_updates.append(updates)

        return state,stages,torch.stack(raw_updates)

