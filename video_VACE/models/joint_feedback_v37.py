"""Per-frame 17-landmark to SMPL24 feedback. No GT or cross-frame inputs.

Old visibility/log variance are feature inputs, not calibrated confidence or GT.
The last projection alone is zero initialized; there is no second zero gate.
"""
import torch
from torch import nn

PARENTS=(-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21)
ANCHORS={
    'posetrack':(15,12,15,15,15,16,17,18,19,20,21,1,2,4,5,7,8),
    'coco':(15,15,15,15,15,16,17,18,19,20,21,1,2,4,5,7,8),
}

def anatomy_bias(schema):
    if schema not in ANCHORS:raise ValueError('Unknown feedback landmark schema: '+str(schema))
    distance=torch.full((24,24),100.,dtype=torch.float32)
    distance.fill_diagonal_(0.)
    for j,p in enumerate(PARENTS):
        if p>=0:distance[j,p]=distance[p,j]=1.
    for k in range(24):distance=torch.minimum(distance,distance[:,k,None]+distance[None,k,:])
    return -.5*distance[:,list(ANCHORS[schema])]

class JointFeedback(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.width=width
        self.token=nn.Sequential(nn.Linear(7,width),nn.SiLU(),nn.Linear(width,width))
        self.landmark=nn.Embedding(17,width)
        self.pose_joint=nn.Embedding(24,width)
        self.schema=nn.Embedding(2,width)
        self.attention=nn.MultiheadAttention(width,4,dropout=0.,batch_first=True)
        self.output=nn.Linear(width,width,bias=False)
        nn.init.zeros_(self.output.weight)
        self.register_buffer('bias_posetrack',anatomy_bias('posetrack'))
        self.register_buffer('bias_coco',anatomy_bias('coco'))

    def forward(self,residual,projected,logvar,visibility,frame_valid,*,schema='posetrack'):
        if schema not in ANCHORS:raise ValueError('Unknown feedback landmark schema: '+str(schema))
        b,t,j,_=residual.shape
        if j!=17 or projected.shape!=residual.shape:raise ValueError('Expected 17 declared landmark coordinates')
        if logvar.shape!=(b,t,17,1) or visibility.shape!=(b,t,17,1) or frame_valid.shape!=(b,t):
            raise ValueError('Feedback shape mismatch')
        finite=(torch.isfinite(residual).all(-1)&torch.isfinite(projected).all(-1)
                &torch.isfinite(logvar).all(-1)&torch.isfinite(visibility).all(-1))
        valid=finite&frame_valid.bool()[...,None]
        residual=torch.where(valid[...,None],residual,torch.zeros_like(residual)).clamp(-2,2)
        projected=torch.where(valid[...,None],projected,torch.zeros_like(projected)).clamp(-4,4)
        logvar=torch.where(valid[...,None],logvar,torch.zeros_like(logvar)).clamp(-6,6)
        visibility=torch.where(valid[...,None],visibility,torch.zeros_like(visibility)).clamp(0,1)
        features=torch.cat((residual,projected,logvar,visibility,valid[...,None].to(residual.dtype)),-1)
        tokens=self.token(features).reshape(b*t,17,self.width)
        identity=self.landmark.weight+self.schema.weight[0 if schema=='posetrack' else 1]
        tokens=tokens+identity[None]
        queries=self.pose_joint.weight[None].expand(b*t,-1,-1)
        # Explicit finite fallback for an all-invalid frame, then exactly zero it.
        flat_valid=valid.reshape(b*t,17);any_valid=flat_valid.any(-1)
        safe_valid=flat_valid.clone();safe_valid[:,0]|=~any_valid
        bias=getattr(self,'bias_'+schema).to(tokens.dtype)
        mask=bias[None].expand(b*t,-1,-1).clone()
        mask=mask.masked_fill(~safe_valid[:,None,:],float('-inf'))
        mask=mask.repeat_interleave(4,0)
        mixed=self.attention(queries,tokens,tokens,attn_mask=mask,need_weights=False)[0]
        result=self.output(mixed).reshape(b,t,24,self.width)
        return result*any_valid.reshape(b,t,1,1).to(result.dtype)


def make_joint_feedback(width):
    """Isolate CPU initialization without changing any CUDA generator."""
    with torch.random.fork_rng(devices=[]):
        generator=torch.Generator(device='cpu').manual_seed(370026)
        torch.random.set_rng_state(generator.get_state())
        return JointFeedback(width)
