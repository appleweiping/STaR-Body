"""Joint memory and pose-only completion. No labels or persistent cross-track state."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def temporal_embedding(ids, origin, width):
    delta=(ids-origin[:,None]).float()
    freq=torch.exp(torch.arange(0,width,2,device=ids.device).float()*(-math.log(10000.)/width))
    phase=delta[...,None]*freq
    return torch.stack((phase.sin(),phase.cos()),-1).flatten(-2)


def masked_weights(scores, allowed):
    """All-masked queries produce zeros, never NaNs or artificial fallback evidence."""
    scores=scores.float().masked_fill(~allowed,-torch.inf)
    present=allowed.any(-1,keepdim=True)
    stable=torch.where(present,scores,torch.zeros_like(scores))
    weights=stable.softmax(-1)*allowed.float()
    return weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)


class ReliableJointMemory(nn.Module):
    def __init__(self,width,topk):
        super().__init__();self.topk=topk
        self.query=nn.Linear(width,width);self.key=nn.Linear(width,width)
        self.value=nn.Linear(width,width);self.quality=nn.Linear(width,1)
        self.output=nn.Linear(width,width);self.norm=nn.LayerNorm(width)
        self.log_time_penalty=nn.Parameter(torch.tensor(-3.))
        self.gain=nn.Parameter(torch.tensor(0.))

    def forward(self,queries,memory,allowed,query_ids,memory_ids):
        # queries [B,T,J,D], memory [B,M,J,D], allowed [B,T,M].
        q=self.query(self.norm(queries)).permute(0,2,1,3).float()
        k=self.key(self.norm(memory)).permute(0,2,1,3).float()
        scores=torch.einsum('bjtd,bjmd->bjtm',q,k)/math.sqrt(q.shape[-1])
        quality=self.quality(memory).squeeze(-1).permute(0,2,1).float()
        age=(query_ids[:,:,None]-memory_ids[:,None,:]).abs().float()
        scores=scores+F.logsigmoid(quality[:,:,None,:])-F.softplus(self.log_time_penalty)*age[:,None]
        mask=(allowed[:,None] if allowed.ndim==3 else allowed).expand_as(scores)
        rank=scores.masked_fill(~mask,-torch.inf)
        indices=rank.topk(min(self.topk,memory.shape[1]),dim=-1).indices
        selected=torch.zeros_like(mask).scatter_(-1,indices,True)&mask
        weights=masked_weights(scores,selected)
        v=self.value(memory).permute(0,2,1,3)
        update=torch.einsum('bjtm,bjmd->bjtd',weights.to(v.dtype),v).permute(0,2,1,3)
        return torch.tanh(self.gain)*self.output(update)*selected.any(-1).permute(0,2,1)[...,None],dict(
            selected_count=selected.sum(-1).permute(0,2,1),
            available=selected.any(-1).permute(0,2,1),quality=quality.permute(0,2,1))


class PoseCompletionPrior(nn.Module):
    """Predict joint rotations from other frames; no RGB or target-frame pose enters queries.

    No temporal encoder is applied to memory keys: otherwise a masked query's pose
    could leak indirectly through an unmasked key. Spatial attention stays within
    each memory frame; temporal cross attention masks the query frame explicitly.
    """
    def __init__(self,width=256,heads=8,depth=4):
        super().__init__();self.width=width
        self.pose_projection=nn.Sequential(nn.LayerNorm(9),nn.Linear(9,width))
        self.joint_embedding=nn.Parameter(torch.randn(24,width)*.02)
        self.spatial_norm=nn.LayerNorm(width)
        self.spatial=nn.MultiheadAttention(width,heads,batch_first=True)
        self.layers=nn.ModuleList([nn.ModuleDict(dict(
            q=nn.Linear(width,width),k=nn.Linear(width,width),v=nn.Linear(width,width),
            norm=nn.LayerNorm(width),out=nn.Linear(width,width),
            mlp=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,2*width),nn.GELU(),nn.Linear(2*width,width))))
            for _ in range(depth)])
        self.output_norm=nn.LayerNorm(width)
        self.rotation_head=nn.Linear(width,6)
        nn.init.zeros_(self.rotation_head.weight);nn.init.zeros_(self.rotation_head.bias)
        self.confidence=nn.Linear(width,1)

    def forward(self,rotations,memory_ids,query_ids,allowed):
        b,m,j,_,_=rotations.shape
        if j!=24 or allowed.shape!=(b,query_ids.shape[1],m):
            raise ValueError('Invalid motion memory shapes')
        pose=self.pose_projection(rotations.flatten(-2))+self.joint_embedding
        spatial=pose.reshape(b*m,24,self.width)
        norm=self.spatial_norm(spatial)
        spatial=spatial+self.spatial(norm,norm,norm,need_weights=False)[0]
        origin=query_ids.min(1).values
        memory=spatial.reshape(b,m,24,self.width)+temporal_embedding(memory_ids,origin,self.width)[:,:,None]
        query=self.joint_embedding[None,None]+temporal_embedding(query_ids,origin,self.width)[:,:,None]
        query=query.expand(b,-1,-1,-1)
        key=memory.permute(0,2,1,3)
        for layer in self.layers:
            q=layer['q'](layer['norm'](query)).permute(0,2,1,3)
            k=layer['k'](key)
            scores=torch.einsum('bjtd,bjmd->bjtm',q.float(),k.float())/math.sqrt(self.width)
            weights=masked_weights(scores,allowed[:,None].expand_as(scores))
            values=layer['v'](key)
            update=torch.einsum('bjtm,bjmd->bjtd',weights.to(values.dtype),values).permute(0,2,1,3)
            query=query+layer['out'](update)
            query=query+layer['mlp'](query)
        available=allowed.any(-1)
        features=self.output_norm(query)*available[...,None,None]
        raw=self.rotation_head(features).float()
        return features,raw,self.confidence(features).squeeze(-1),available
