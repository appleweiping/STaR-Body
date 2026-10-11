"""v39b copied from v39: numeric-only metric payload for selection compatibility."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from divibody_training.ram_parts_runtime_v27 import (
    V10Dataset as LegacyV10Dataset,geometric_losses as legacy_geometry,INPUT_KEYS,
    smpl_vertices,gendered_vertices,camera_targets,pelvis_aligned_j14,
    mean_euclidean_error,pa_mpjpe_per_sample)
from models.human_decoder_parts_v41 import model_from_payload,FORMAT,AdaptationConfig

class V10Dataset(LegacyV10Dataset):
    def __getitem__(self,index):
        sample=super().__getitem__(index)
        source=self.vace['metadata'][index].get('target_source_frame_ids')
        if source is None: raise ValueError('v41 requires actual 3DPW source frame identities')
        sample['source_frame_ids']=torch.as_tensor(source,dtype=torch.long)
        return sample

def forward_batch(model,batch,device,amp_dtype,position_offset=0):
    values=[batch[k].to(device,non_blocking=True) for k in INPUT_KEYS]
    if device.type != 'cuda' or amp_dtype == 'none':
        values=[v.float() for v in values]
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

@torch.no_grad()
def validate(loader,model,smpl_models,regressor,args,device):
    # Exact v27 metric calculations; only v41's identity-aware forward changes.
    model.eval()
    names=('mpjpe_mm','pa_mpjpe_mm','pve_mm','acceleration_error_mm_per_frame2')
    sums,counts=dict.fromkeys(names,0.),dict.fromkeys(names,0)
    for batch in loader:
        prediction=forward_batch(model,batch,device,args.amp_dtype)
        pred_rot=prediction['rotmat'].float()
        target_rot=camera_targets(batch,device)
        target_betas=batch['betas'].to(device,dtype=torch.float32)
        genders=batch['gender_code'].to(device).reshape(-1)
        pred_vertices=smpl_vertices(smpl_models[2],pred_rot.reshape(-1,24,3,3),prediction['betas'].float().reshape(-1,10))
        gt_vertices=gendered_vertices(smpl_models,target_rot.reshape(-1,24,3,3),target_betas.reshape(-1,10),genders)
        pred_joints,_=pelvis_aligned_j14(pred_vertices,regressor)
        gt_joints,_=pelvis_aligned_j14(gt_vertices,regressor)
        frame_mask=batch['frame_owner'].to(device).reshape(-1)
        arrays={'mpjpe_mm':mean_euclidean_error(pred_joints,gt_joints)*1000,
            'pa_mpjpe_mm':pa_mpjpe_per_sample(pred_joints,gt_joints)*1000,
            'pve_mm':mean_euclidean_error(pred_vertices,gt_vertices)*1000}
        for name,values in arrays.items():
            selected=values.masked_select(frame_mask)
            sums[name]+=float(selected.double().sum()); counts[name]+=selected.numel()
        b,t=pred_rot.shape[:2]
        error=(pred_joints-gt_joints).reshape(b,t,14,3)
        accel=torch.linalg.vector_norm(torch.diff(error,n=2,dim=1),dim=-1).mean(-1)*1000
        selected=accel.masked_select(batch['acceleration_owner'].to(device))
        name='acceleration_error_mm_per_frame2'
        sums[name]+=float(selected.double().sum()); counts[name]+=selected.numel()
    if any(count==0 for count in counts.values()): raise RuntimeError('Validation masks contain no owned samples')
    return {k:sums[k]/counts[k] for k in names}|{'frames':counts['mpjpe_mm'],
        'acceleration_triplets':counts['acceleration_error_mm_per_frame2']}

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


@torch.no_grad()
def evaluate_pose(loader,model,official_smpl,args,device):
    model.eval()
    totals={prefix+key:0.0 for prefix in ('','projected_') for key in ('correct05','correct10','error_sum')}
    visible_count=0; confidence_sum=0.0
    for batch in loader:
        prediction=forward_batch(model,batch,device,args.amp_dtype)
        pairs=(('',prediction['observed_2d']),('projected_',prediction['direct_2d']))
        gt=batch['keypoints'].to(device,dtype=torch.float32)
        visible=(gt[...,2]>0)&torch.isfinite(gt).all(-1)
        visible &= batch['target_valid'].to(device).unsqueeze(-1)
        visible &= batch['frame_owner'].to(device).unsqueeze(-1)
        size=batch.get('keypoint_image_size',96.)
        if torch.is_tensor(size): size=size.to(device).reshape(-1,1,1,1)
        target=gt[...,:2]/size-.5
        for prefix,projected in pairs:
            values=(torch.linalg.vector_norm(projected-target,dim=-1)/.8).masked_select(visible)
            totals[prefix+'correct05']+=int((values<=.05).sum()); totals[prefix+'correct10']+=int((values<=.10).sum())
            totals[prefix+'error_sum']+=float(values.double().sum())
        visible_count+=int(visible.sum())
        confidence_sum+=float(prediction['posetrack_confidence'].squeeze(-1).masked_select(visible).double().sum())
    if not visible_count: raise RuntimeError('PoseTrack validation has no visible owned keypoints')
    return {'PCKn@0.05':totals['correct05']/visible_count,'PCKn@0.10':totals['correct10']/visible_count,
      'mean_normalized_joint_error':totals['error_sum']/visible_count,
      'projected_PCKn@0.05':totals['projected_correct05']/visible_count,
      'projected_PCKn@0.10':totals['projected_correct10']/visible_count,
      'projected_mean_normalized_joint_error':totals['projected_error_sum']/visible_count,
      'mean_visible_confidence':confidence_sum/visible_count,'visible_keypoints':visible_count}

_legacy_pose_losses = pose_losses

def pose_losses(prediction,batch,official_smpl,args,device):
    result=_legacy_pose_losses(prediction,batch,official_smpl,args,device)
    gt=batch['keypoints'].to(device,dtype=torch.float32)
    owned=batch['target_valid'].to(device)[...,None]&batch['frame_owner'].to(device)[...,None]
    mask=(gt[...,2]>0)&torch.isfinite(gt).all(-1)&owned
    size=batch.get('keypoint_image_size',96.)
    if torch.is_tensor(size): size=size.to(device).reshape(-1,1,1,1)
    target=gt[...,:2]/size-.5
    target=torch.where(mask[...,None],target,torch.zeros_like(target))
    logits=prediction['evidence_logits'].float()
    b,t,j,n=logits.shape
    yy,xx=torch.meshgrid(torch.arange(16,device=device).float(),torch.arange(12,device=device).float(),indexing='ij')
    centers=torch.stack(((xx+.5)/16.-.375,(yy+.5)/16.-.5),-1).reshape(1,1,1,n,2)
    gaussian=torch.exp(-((centers-target[...,None,:])/.045).square().sum(-1)/2.)
    gaussian=gaussian/gaussian.sum(-1,keepdim=True).clamp_min(1e-12)
    represented=mask&(target[...,0].abs()<.375)&(target[...,1].abs()<.5)
    zero=logits.sum()*0.
    heat=-(gaussian.detach()*logits.log_softmax(-1)).sum(-1)[represented].mean() if represented.any() else zero
    probs=logits.detach().softmax(-1).reshape(b*t*j,1,16,12)
    grid=prediction['direct_2d']*target.new_tensor([8./3.,2.])
    sampled=F.grid_sample(probs,grid.reshape(b*t*j,1,1,2),align_corners=False).reshape(b,t,j)
    heat_projection=-sampled.clamp_min(1e-8).log()[represented].mean() if represented.any() else zero
    rotation_distill=(prediction['rotmat']-prediction['geometry_anchor']['rotmat']).square().mean()
    # Quality calibration from annotated joint error, not fabricated visibility.
    confidence_mask=mask&prediction['independent_frames'][...,None]
    quality_error=torch.linalg.vector_norm((prediction['observed_2d'].detach()-target)/.8,dim=-1)
    quality_target=torch.exp(-.5*(quality_error/.05).square())
    confidence=prediction['posetrack_confidence'].squeeze(-1).clamp(1e-6,1-1e-6)
    calibration=F.binary_cross_entropy(confidence[confidence_mask],quality_target[confidence_mask]) if confidence_mask.any() else zero
    result['pose_2d']=(result['pose_2d']+args.heatmap_weight*heat+
        args.heat_projection_weight*heat_projection+args.geometry_distill_weight*rotation_distill+
        args.pose_confidence_weight*calibration)
    result.update(heatmap_cross_entropy=heat,heat_projection_nll=heat_projection,
        geometry_rotation_distill=rotation_distill,quality_calibration_bce=calibration,
        heatmap_supervised_points=represented.sum().float())
    return result

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
