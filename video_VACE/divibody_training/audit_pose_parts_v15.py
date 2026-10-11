"""Audit real PoseTrack projection and gradients without updating any weights."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader
from hmr2.models.smpl_wrapper import SMPL
from hmr2.utils.geometry import perspective_projection
from divibody_training.ram_pose_stream_v10 import ShuffledPoseDatasetV10
from divibody_training.ram_parts_runtime_v15 import model_from_payload,forward_batch,pose_losses
from divibody_training.train_ram_joint_sharded_pose_geometry_v7 import projected_posetrack,PT_FROM_OP

def main():
    root=Path('/mnt/32a56f7d-8270-4d4e-b9cb-fb8857bd7e93/testpro/xmj/sam-3d-body/divibody_vace')
    runs=root/'training_runs'
    collection=runs/'ram_posetrack_sharded_v2_full_20260902'
    data=ShuffledPoseDatasetV10(collection/'train',collection/'hmr_train',seed=2026,max_windows=8)
    assert data.split=='train'
    batch=next(iter(DataLoader(data,batch_size=4,num_workers=0)))
    device=torch.device('cuda:0')
    payload=torch.load(runs/'ram_v9_geometry_priority_full_20260907_retry1/run/best.pt',
                       map_location='cpu',weights_only=False,mmap=True)
    model=model_from_payload(payload).to(device).train()
    smpl=SMPL(model_path=str(root/'checkpoints/4DHumans/extracted/data/smpl'),gender='neutral',
              num_betas=10,joint_regressor_extra=str(root/'checkpoints/4DHumans/extracted/data/SMPL_to_J19.pkl')).to(device)
    smpl.requires_grad_(False)
    prediction=forward_batch(model,batch,device,'bf16')
    projected=projected_posetrack(prediction,smpl)
    r=prediction['rotmat'].reshape(-1,24,3,3)
    body=smpl(global_orient=r[:,:1],body_pose=r[:,1:],betas=prediction['betas'].reshape(-1,10),pose2rot=False)
    camera=prediction['camera'].reshape(-1,3)
    translation=torch.stack((camera[:,1],camera[:,2],10000/(256*camera[:,0].clamp_min(1e-6))),dim=-1)
    official=perspective_projection(body.joints[:,:25],translation=translation,
                                    focal_length=torch.full((len(r),2),5000.,device=device))/256.
    expected=official.new_zeros(len(r),17,2)
    for target,source in PT_FROM_OP.items():expected[:,target]=official[:,source]
    expected[:,2]=2*official[:,0]-official[:,1]
    error=float((projected.reshape_as(expected)-expected).abs().max())
    assert error<1e-5,error
    # Same crop in both RGB sizes: annotation coordinates normalize identically.
    coords=torch.tensor([[10.,20.],[50.,70.]])
    crop_origin=torch.tensor([-5.,4.]); side=100.
    a=(coords-crop_origin)*96/side/96-.5
    b=(coords-crop_origin)*256/side/256-.5
    torch.testing.assert_close(a,b)
    args=SimpleNamespace(pose_deadzone=.01)
    losses=pose_losses(prediction,batch,smpl,args,device)
    losses['pose_2d'].backward()
    grad=sum(float(p.grad.abs().sum()) for p in model.part_branch.heads.parameters())
    assert grad>0
    assert all(p.grad is None for p in model.reference.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.part_branch.parameters() if p.grad is not None)
    report=dict(official_projection_max_difference=error,
                pose_visible_points=int(losses['pose_visible_points']),
                pose_loss=float(losses['pose_2d'].detach()),part_head_pose_gradient_l1=grad,
                reference_has_no_grad=True,optimizer_steps=0)
    invisible=dict(batch);invisible['keypoints']=batch['keypoints'].clone()
    invisible['keypoints'][...,:2]=float('nan');invisible['keypoints'][...,2]=0
    empty=pose_losses(prediction,invisible,smpl,args,device)
    assert torch.isfinite(empty['pose_2d']) and float(empty['pose_2d'])==0
    print(json.dumps(report,indent=2),flush=True)
    print('V15_POSE_AUDIT_OK',flush=True)
if __name__=='__main__':main()
