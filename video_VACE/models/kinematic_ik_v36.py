"""Differentiable residual swing/twist IK on the SMPL24 kinematic tree.

This is a residual solver preserving the parent twist, not a HybrIK reimplementation.
Degenerate target bones keep the current orientation; antiparallel cases are finite.
"""
import torch
import torch.nn.functional as F
PARENTS=(-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21)
CHILD=(3,4,5,6,7,8,9,10,11,12,-1,-1,15,16,17,-1,18,19,20,21,22,23,-1,-1)

def skew(v):
    x,y,z=v.unbind(-1);o=torch.zeros_like(x)
    return torch.stack((o,-z,y,z,o,-x,-y,x,o),-1).reshape(*v.shape[:-1],3,3)

def align_vectors(a,b,eps=1e-7):
    if a.shape!=b.shape or a.shape[-1]!=3:raise ValueError('Matched 3D vectors required')
    good=(a.norm(dim=-1)>eps)&(b.norm(dim=-1)>eps)
    u=F.normalize(a,dim=-1,eps=eps);v=F.normalize(b,dim=-1,eps=eps)
    cross=torch.linalg.cross(u,v);cos=(u*v).sum(-1).clamp(-1,1)
    k=skew(cross);eye=torch.eye(3,device=a.device,dtype=a.dtype).expand_as(k)
    regular=eye+k+(k@k)/(1+cos).clamp_min(eps)[...,None,None]
    choice=F.one_hot(u.abs().argmin(-1),3).to(u.dtype)
    axis=F.normalize(torch.linalg.cross(u,choice),dim=-1,eps=eps)
    opposite=2*axis[...,None]*axis[...,None,:]-eye
    result=torch.where((cos < -1+1e-5)[...,None,None],opposite,regular)
    return torch.where(good[...,None,None],result,eye)

def exp_axis(omega):
    angle=omega.norm(dim=-1);k=skew(omega)
    eye=torch.eye(3,device=omega.device,dtype=omega.dtype).expand_as(k)
    return eye+torch.sinc(angle/torch.pi)[...,None,None]*k+(.5*torch.sinc(angle/(2*torch.pi)).square())[...,None,None]*(k@k)

def forward_kinematics(local,rest):
    worlds=[];positions=[]
    for j,p in enumerate(PARENTS):
        worlds.append(local[...,j,:,:] if p<0 else worlds[p]@local[...,j,:,:])
        positions.append(rest[...,j,:] if p<0 else positions[p]+(worlds[p]@(rest[...,j,:]-rest[...,p,:])[...,None]).squeeze(-1))
    return torch.stack(positions,-2),torch.stack(worlds,-3)

def residual_ik(local,rest,target,twist):
    if local.shape[-3:]!=(24,3,3) or rest.shape[-2:]!=(24,3) or target.shape!=rest.shape or twist.shape!=rest.shape[:-1]:
        raise ValueError('SMPL24 IK shape mismatch')
    worlds=[];output=[]
    eye=torch.eye(3,device=local.device,dtype=local.dtype).expand(*local.shape[:-3],3,3)
    for j,p in enumerate(PARENTS):
        parent=eye if p<0 else worlds[p];world=parent@local[...,j,:,:];child=CHILD[j]
        if child>=0:
            restbone=rest[...,child,:]-rest[...,j,:]
            current=(world@restbone[...,None]).squeeze(-1)
            wanted=target[...,child,:]-target[...,j,:]
            world=align_vectors(current,wanted)@world
            axis=F.normalize(wanted,dim=-1,eps=1e-7)
            world=exp_axis(axis*twist[...,j,None])@world
        output.append(parent.transpose(-1,-2)@world);worlds.append(world)
    return torch.stack(output,-3)
