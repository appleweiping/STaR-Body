"""v39 dataset landmark projection with native LSP PoseTrack head joints.

Copied from v28; v28 remains unchanged.
"""
import torch

COCO_FROM_OPENPOSE = (0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11)
POSETRACK_FROM_OPENPOSE = (0,37,38,18,17,5,2,6,3,7,4,12,9,13,10,14,11)
SCHEMAS = ('posetrack','coco')

def camera_project(joints,camera):
    camera=camera.float()
    translation=torch.stack((camera[...,1],camera[...,2],10000./(256.*camera[...,0].clamp_min(1e-6))),-1)
    points=joints.float()+translation[...,None,:]
    uv=points[...,:2]/points[...,2:].clamp_min(1e-6)*(5000./256.)
    valid=torch.isfinite(uv).all(-1)&(points[...,2]>1e-6)&(camera[...,0,None]>0)
    return torch.where(valid[...,None],uv,torch.zeros_like(uv)),valid

def select_landmarks(uv,schema):
    if schema not in SCHEMAS:
        raise ValueError(f'Unknown landmark schema: {schema}')
    indices=COCO_FROM_OPENPOSE if schema=='coco' else POSETRACK_FROM_OPENPOSE
    result=uv[...,list(indices),:].clone()
    return result

def project_landmarks(prediction,official_smpl,schema='posetrack'):
    shape=prediction['rotmat'].shape[:2]
    with torch.autocast(device_type=prediction['rotmat'].device.type,enabled=False):
        r=prediction['rotmat'].float().reshape(-1,24,3,3)
        body=official_smpl(global_orient=r[:,:1],body_pose=r[:,1:],
            betas=prediction['betas'].float().reshape(-1,10),pose2rot=False)
        uv,_=camera_project(body.joints,prediction['camera'].reshape(-1,3))
        return select_landmarks(uv,schema).reshape(*shape,17,2)
