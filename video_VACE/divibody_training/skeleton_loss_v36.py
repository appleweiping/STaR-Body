"""Train-only SMPL24 skeleton supervision; official benchmark remains J14 from mesh."""
import torch
from models.kinematic_ik_v36 import PARENTS
from divibody_training.train_divibody_continuous_geometry_v5 import camera_targets
def skeleton_loss(pred,batch,models,device):
    with torch.autocast(device_type=device.type,enabled=False):
        with torch.no_grad():
            r=camera_targets(batch,device).reshape(-1,24,3,3)
            betas=batch['betas'].to(device,dtype=torch.float32).reshape(-1,10)
            genders=batch['gender_code'].to(device).reshape(-1);target=torch.empty(len(r),24,3,device=device)
            for code,body in models.items():
                mask=genders==code
                if mask.any():target[mask]=body(global_orient=r[mask,:1],body_pose=r[mask,1:],betas=betas[mask],pose2rot=False).joints[:,:24]
            if not all(int(g) in models for g in genders.unique()):raise ValueError('Unknown training gender code')
            target=target.reshape_as(pred['skeleton_joints']);target=target-target[:,:,:1]
        owned=(batch['frame_owner']&batch['target_valid']).to(device)
        if not owned.any():raise ValueError('Skeleton loss needs owned training frames')
        p=pred['skeleton_joints'].float();position=(p-target).norm(dim=-1).mean(-1)[owned].mean()
        parents=list(PARENTS[1:]);pv=p[:,:,1:]-p[:,:,parents];tv=target[:,:,1:]-target[:,:,parents]
        length=(pv.norm(dim=-1)-tv.norm(dim=-1)).abs().mean(-1)[owned].mean()
        direction=(1-torch.nn.functional.cosine_similarity(pv,tv,dim=-1,eps=1e-6)).mean(-1)[owned].mean()
        # All coefficients preregistered; direction is dimensionless and logged separately.
        total=position+.1*length+.001*direction
    return dict(skeleton_total=total,skeleton_position_m=position,skeleton_bone_length_m=length,skeleton_direction=direction)
