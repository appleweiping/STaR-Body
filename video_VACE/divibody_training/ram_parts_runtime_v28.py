"""v28 losses, feature-cache interfaces and direct geometric evaluation."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from divibody_training.ram_parts_runtime_v27 import (
    V10Dataset,validate,evaluate_pose,geometric_losses as legacy_geometry,INPUT_KEYS)
from models.human_decoder_parts_v28 import model_from_payload,FORMAT,AdaptationConfig

def forward_batch(model,batch,device,amp_dtype,position_offset=0):
    values=[batch[k].to(device,non_blocking=True) for k in INPUT_KEYS]
    valid=batch.get('target_valid')
    valid=valid.to(device,dtype=torch.bool) if valid is not None else None
    schema=batch.get('landmark_schema','posetrack')
    if isinstance(schema,(list,tuple)):
        if len(set(schema))!=1:
            raise ValueError('Each batch must have one landmark convention')
        schema=schema[0]
    source=batch.get('source_frame_ids')
    if source is not None:
        source=source.to(device)
    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda' and amp_dtype=='bf16'):
        return model(*values,frame_valid=valid,landmark_schema=schema,source_frame_ids=source,position_offset=position_offset)

def geometric_losses(prediction,batch,models,regressor,args,device):
    legacy_args=SimpleNamespace(**vars(args))
    legacy_args.anchor_weight=0.
    result=legacy_geometry(prediction,batch,models,regressor,legacy_args,device)
    owned=batch['frame_owner'].to(device)&batch['target_valid'].to(device)
    update=prediction['rotation_update'].square().mean((-1,-2))+prediction['camera_update'].square().mean(-1)
    penalty=update[owned].mean() if owned.any() else update.sum()*0.
    result['total']=result['total']+args.update_regularization*penalty
    result['update_regularization']=penalty
    return result

def pose_losses(prediction,batch,official_smpl,args,device):
    # Never add an unconstrained image-space residual to the primary projection.
    with torch.autocast(device_type=device.type,enabled=False):
        gt=batch['keypoints'].to(device,dtype=torch.float32)
        owned=batch['target_valid'].to(device)[...,None]&batch['frame_owner'].to(device)[...,None]
        mask=(gt[...,2]>0)&torch.isfinite(gt).all(-1)&owned
        size=batch.get('keypoint_image_size',96.)
        if torch.is_tensor(size):
            size=size.to(device).reshape(-1,1,1,1)
        target=gt[...,:2]/size-.5
        zero=prediction['direct_2d'].sum()*0.
        if mask.any():
            def reprojection(uv):
                return F.smooth_l1_loss((uv[mask]-target[mask])/.8,torch.zeros_like(target[mask]),beta=.05)
            stage=torch.stack([reprojection(uv) for uv in prediction['stage_projections']]).mean()
            direct=reprojection(prediction['direct_2d'])
            error=(prediction['observed_2d'][mask]-target[mask])/.8
            logvar=prediction['observation_log_variance'].squeeze(-1)[mask]
            # Proper Gaussian NLL, with bounded variance and independently supervised coordinates.
            nll=(.5*error.square().sum(-1)*torch.exp(-logvar)+logvar).mean()
            obs=reprojection(prediction['observed_2d'])
            anchor_error=torch.linalg.vector_norm((prediction['anchor_2d'][mask]-target[mask])/.8,dim=-1)
            final_error=torch.linalg.vector_norm((prediction['direct_2d'][mask]-target[mask])/.8,dim=-1)
            reliable=anchor_error<.05
            preserve=(final_error-anchor_error.detach()).clamp_min(0)[reliable].mean() if reliable.any() else zero
        else:
            stage=direct=obs=nll=preserve=zero
        visibility_loss=zero
        labels=batch.get('visibility_targets')
        schema=batch.get('landmark_schema','posetrack')
        is_coco=schema=='coco' if isinstance(schema,str) else all(x=='coco' for x in schema)
        if labels is None and is_coco:
            labels=torch.where(gt[...,2]>0,(gt[...,2]==2).float(),-torch.ones_like(gt[...,2]))
        if labels is not None:
            labels=labels.to(device).float()
            known=(labels>=0)&owned&torch.isfinite(labels)
            if known.any():
                visibility_loss=F.binary_cross_entropy_with_logits(prediction['visibility_logits'].squeeze(-1)[known],labels[known])
        total=(direct+args.stage_2d_weight*stage+args.observation_weight*obs+
               args.uncertainty_weight*nll+args.visibility_weight*visibility_loss+args.anchor_weight*preserve)
    return dict(pose_2d=total,pose_2d_projected=direct,pose_2d_observation=obs,
        pose_2d_stages=stage,observation_nll=nll,visibility_bce=visibility_loss,
        anchor_preservation=preserve,pose_visible_points=mask.sum().float())

class CocoTrainingCache(Dataset):
    """Explicit opt-in train2017 cache; refuses val/test or undeclared provenance.

    Manifest: {split:'train2017', landmark_schema:'coco', files:[...]}; each file
    contains one B-free dictionary with INPUT_KEYS, [T,17,3] keypoints in 256
    crop pixels, target_valid/frame_owner and source_frame_ids (duplicates=0).
    Features must be encoded from the corresponding clean/occluded source.
    """
    def __init__(self,manifest):
        manifest=Path(manifest)
        data=json.loads(manifest.read_text())
        if data.get('split')!='train2017' or data.get('landmark_schema')!='coco':
            raise ValueError('COCO optimization requires explicit train2017/coco provenance')
        self.files=[manifest.parent/p for p in data['files']]
        if not self.files or not all(p.is_file() for p in self.files):
            raise ValueError('Incomplete COCO training cache')
    def __len__(self):
        return len(self.files)
    def __getitem__(self,index):
        sample=torch.load(self.files[index],map_location='cpu',weights_only=False)
        required=(*INPUT_KEYS,'keypoints','source_frame_ids','target_valid','frame_owner')
        if any(k not in sample for k in required):
            raise ValueError('Missing COCO cache fields')
        return sample|dict(landmark_schema='coco',keypoint_image_size=256.)
