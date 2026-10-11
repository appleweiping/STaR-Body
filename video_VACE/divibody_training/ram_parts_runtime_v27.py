"""v27 geometry-coupled training and paired direct/assisted pose metrics."""
from dataclasses import asdict

import torch
import torch.nn.functional as F

from divibody_training.geometry_metrics_3dpw import (
    mean_euclidean_error, pa_mpjpe_per_sample, pelvis_aligned_j14, hip_centered_j14,
)
from divibody_training.train_divibody_continuous_geometry_v5 import (
    camera_targets, gendered_vertices, smpl_vertices,
)
from divibody_training.train_divibody_rgb_vace_hmr2a_v8 import V8Dataset
from models.human_decoder_parts_v27 import (
    HumanV9AdaptationV27, RGBVaceHMR2AConfig, AdaptationConfig, FORMAT,
    model_from_payload as build_model_from_payload,
)

INPUT_KEYS = ('rgb_tokens', 'vace_features', 'base_rotmat', 'base_betas', 'base_camera')


def _owned_mean(values, mask, *, optional=False):
    if values.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError('Loss values and Boolean ownership masks must have matching shapes')
    if mask.any():
        return values[mask].mean()
    if optional:
        return values.sum() * 0.0
    raise ValueError('Geometric batch has no owned frames')


def geometric_losses(prediction, batch, models, regressor, args, device):
    # Copied from v7, retaining its losses and weights. The only mask change is
    # allowing zero temporal loss when a batch consists solely of short tracks.
    with torch.autocast(device_type=device.type, enabled=False):
        pred_rot = prediction['rotmat'].float()
        target_rot = camera_targets(batch, device)
        target_betas = batch['betas'].to(device, dtype=torch.float32)
        pred_betas = prediction['betas'].float()
        genders = batch['gender_code'].to(device).reshape(-1)
        pv = smpl_vertices(models[2], pred_rot.reshape(-1, 24, 3, 3), pred_betas.reshape(-1, 10))
        gv = gendered_vertices(models, target_rot.reshape(-1, 24, 3, 3),
                               target_betas.reshape(-1, 10), genders)
        pj, _ = pelvis_aligned_j14(pv, regressor)
        gj, _ = pelvis_aligned_j14(gv, regressor)
        batch_size, frames = pred_rot.shape[:2]
        mask = batch['frame_owner'].to(device)
        joint = _owned_mean(mean_euclidean_error(pj, gj).reshape(batch_size, frames), mask)
        vertex = _owned_mean(mean_euclidean_error(pv, gv).reshape(batch_size, frames), mask)
        rotation = _owned_mean((pred_rot - target_rot).square().mean(dim=(-1, -2, -3)), mask)
        shape = _owned_mean((pred_betas - target_betas).square().mean(dim=-1), mask)
        motion_error = (hip_centered_j14(pj) - hip_centered_j14(gj)).reshape(batch_size, frames, 14, 3)
        velocity = _owned_mean(torch.linalg.vector_norm(torch.diff(motion_error, dim=1),
            dim=-1).mean(-1), batch['velocity_owner'].to(device), optional=True)
        acceleration = _owned_mean(torch.linalg.vector_norm(torch.diff(motion_error, n=2, dim=1),
            dim=-1).mean(-1), batch['acceleration_owner'].to(device), optional=True)
        base_betas = batch['base_betas'].to(device)
        if base_betas.ndim == 2:
            base_betas = base_betas[:, None]
        identity_6d = prediction['rotation_residual_6d'].new_tensor([1., 0., 0., 0., 1., 0.])
        anchor = ((prediction['rotation_residual_6d'] - identity_6d).square().mean()
                  + (pred_betas - base_betas).square().mean()
                  + (prediction['camera'] - batch['base_camera'].to(device)).square().mean())
        total = (args.joint_weight * joint + args.vertex_weight * vertex
                 + args.rotation_weight * rotation + args.shape_weight * shape
                 + args.velocity_weight * velocity + args.acceleration_weight * acceleration
                 + args.anchor_weight * anchor)
    return {'total': total, 'joint_m': joint, 'vertex_m': vertex,
            'rotation_chordal': rotation, 'shape_mse': shape,
            'velocity_m_per_frame': velocity, 'acceleration_m_per_frame2': acceleration,
            'anchor': anchor}


class V10Dataset(V8Dataset):
    def __getitem__(self, index):
        batch = super().__getitem__(index)
        batch['target_valid'] = self.vace['target_valid_mask'][index]
        return batch


def model_from_payload(payload, adaptation_config=None):
    return build_model_from_payload(payload, adaptation_config)


def geometry_distance(left, right, batch, models, regressor, device):
    """J14 + 0.5 vertex distance in metres; owned valid frames only."""
    with torch.autocast(device_type=device.type, enabled=False):
        def vertices(pred):
            return smpl_vertices(models[2], pred['rotmat'].float().reshape(-1,24,3,3),
                                 pred['betas'].float().reshape(-1,10))
        lv, rv = vertices(left), vertices(right)
        lj, _ = pelvis_aligned_j14(lv, regressor)
        rj, _ = pelvis_aligned_j14(rv, regressor)
        mask = batch['frame_owner'].to(device) & batch['target_valid'].to(device)
        distance = mean_euclidean_error(lj, rj) + .5 * mean_euclidean_error(lv, rv)
        return _owned_mean(distance.reshape(mask.shape), mask)


def forward_batch(model, batch, device, amp_dtype, position_offset=0):
    if amp_dtype not in ('bf16', 'none'):
        raise ValueError('amp_dtype must be bf16 or none')
    use_amp = device.type == 'cuda' and amp_dtype == 'bf16'
    values = [batch[k].to(device, non_blocking=True) for k in INPUT_KEYS]
    if not use_amp:
        values = [v.float() for v in values]
    valid = batch.get('target_valid')
    valid = valid.to(device, dtype=torch.bool) if valid is not None else None
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
        return model(*values, frame_valid=valid, position_offset=position_offset)


@torch.no_grad()
def validate(loader, model, smpl_models, regressor, args, device):
    # Copied metric calculations from train_divibody_rgb_vace_hmr2a_v8.validate.
    # PVE intentionally retains that evaluator's unaligned vertex convention.
    model.eval()
    names = ('mpjpe_mm', 'pa_mpjpe_mm', 'pve_mm', 'acceleration_error_mm_per_frame2')
    sums, counts = dict.fromkeys(names, 0.0), dict.fromkeys(names, 0)
    for batch in loader:
        prediction = forward_batch(model, batch, device, args.amp_dtype)
        pred_rot = prediction['rotmat'].float()
        target_rot = camera_targets(batch, device)
        target_betas = batch['betas'].to(device, dtype=torch.float32)
        genders = batch['gender_code'].to(device).reshape(-1)
        pred_vertices = smpl_vertices(smpl_models[2], pred_rot.reshape(-1, 24, 3, 3),
                                      prediction['betas'].float().reshape(-1, 10))
        gt_vertices = gendered_vertices(smpl_models, target_rot.reshape(-1, 24, 3, 3),
                                        target_betas.reshape(-1, 10), genders)
        pred_joints, _ = pelvis_aligned_j14(pred_vertices, regressor)
        gt_joints, _ = pelvis_aligned_j14(gt_vertices, regressor)
        frame_mask = batch['frame_owner'].to(device).reshape(-1)
        arrays = {
            'mpjpe_mm': mean_euclidean_error(pred_joints, gt_joints) * 1000,
            'pa_mpjpe_mm': pa_mpjpe_per_sample(pred_joints, gt_joints) * 1000,
            'pve_mm': mean_euclidean_error(pred_vertices, gt_vertices) * 1000,
        }
        for name, values in arrays.items():
            selected = values.masked_select(frame_mask)
            sums[name] += float(selected.double().sum())
            counts[name] += selected.numel()
        batch_size, frames = pred_rot.shape[:2]
        error = (pred_joints - gt_joints).reshape(batch_size, frames, 14, 3)
        accel = torch.linalg.vector_norm(torch.diff(error, n=2, dim=1), dim=-1).mean(-1) * 1000
        selected = accel.masked_select(batch['acceleration_owner'].to(device))
        key = 'acceleration_error_mm_per_frame2'
        sums[key] += float(selected.double().sum())
        counts[key] += selected.numel()
    if any(count == 0 for count in counts.values()):
        raise RuntimeError('Validation masks contain no owned samples')
    return {k: sums[k] / counts[k] for k in names} | {
        'frames': counts['mpjpe_mm'],
        'acceleration_triplets': counts['acceleration_error_mm_per_frame2'],
    }


@torch.no_grad()
def evaluate_pose(loader, model, official_smpl, args, device):
    from divibody_training.train_ram_joint_sharded_pose_geometry_v7 import projected_posetrack
    model.eval()
    totals = {prefix + key: 0.0 for prefix in ('', 'projected_')
              for key in ('correct05', 'correct10', 'error_sum')}
    visible_count = 0
    confidence_sum = 0.0
    for batch in loader:
        prediction = forward_batch(model, batch, device, args.amp_dtype)
        direct = projected_posetrack(prediction, official_smpl)
        assisted = direct + prediction['posetrack_2d_residual']
        gt = batch['keypoints'].to(device, dtype=torch.float32)
        visible = gt[..., 2] > 0
        visible &= batch['target_valid'].to(device).unsqueeze(-1)
        visible &= batch['frame_owner'].to(device).unsqueeze(-1)
        target = gt[..., :2] / 96.0 - 0.5
        for prefix, projected in (('', assisted), ('projected_', direct)):
            values = (torch.linalg.vector_norm(projected-target, dim=-1)/(1.0/1.25)).masked_select(visible)
            totals[prefix+'correct05'] += int((values <= 0.05).sum())
            totals[prefix+'correct10'] += int((values <= 0.10).sum())
            totals[prefix+'error_sum'] += float(values.double().sum())
        visible_count += int(visible.sum())
        confidence_sum += float(prediction['posetrack_confidence'].squeeze(-1).masked_select(visible).double().sum())
    if not visible_count:
        raise RuntimeError('PoseTrack validation has no visible owned keypoints')
    return {'PCKn@0.05': totals['correct05']/visible_count,
            'PCKn@0.10': totals['correct10']/visible_count,
            'mean_normalized_joint_error': totals['error_sum']/visible_count,
            'projected_PCKn@0.05': totals['projected_correct05']/visible_count,
            'projected_PCKn@0.10': totals['projected_correct10']/visible_count,
            'projected_mean_normalized_joint_error': totals['projected_error_sum']/visible_count,
            'mean_visible_confidence': confidence_sum/visible_count,
            'visible_keypoints': visible_count}


def pose_losses(prediction,batch,official_smpl,args,device):
    """Visible owned annotations only; robust 2D limb supervision in crop units."""
    from divibody_training.train_ram_joint_sharded_pose_geometry_v7 import projected_posetrack
    with torch.autocast(device_type=device.type,enabled=False):
        direct=projected_posetrack(prediction,official_smpl)
        projected=direct+prediction['posetrack_2d_residual']
        gt=batch['keypoints'].to(device,dtype=torch.float32)
        visible=(gt[...,2]>0)&torch.isfinite(gt).all(-1)
        visible &= batch['target_valid'].to(device)[...,None]
        visible &= batch['frame_owner'].to(device)[...,None]
        # Select before arithmetic: invisible NaN coordinates must not poison gradients.
        target=gt[...,:2]/96.-.5
        residual=(projected[visible]-target[visible])/.8
        distance=torch.linalg.vector_norm(residual,dim=-1)
        excess=(distance-args.pose_deadzone).clamp_min(0)
        reprojection=F.smooth_l1_loss(excess,torch.zeros_like(excess),beta=.05,reduction='sum')
        reprojection=reprojection/max(1,int(visible.sum()))
        direct_residual=(direct[visible]-target[visible])/.8
        direct_distance=torch.linalg.vector_norm(direct_residual,dim=-1)
        direct_excess=(direct_distance-args.pose_deadzone).clamp_min(0)
        direct_loss=F.smooth_l1_loss(direct_excess,torch.zeros_like(direct_excess),beta=.05,reduction='sum')
        direct_loss=direct_loss/max(1,int(visible.sum()))
        owned = batch['target_valid'].to(device)[...,None] & batch['frame_owner'].to(device)[...,None]
        owned = owned.expand_as(visible)
        confidence_target = visible.to(torch.float32)
        confidence = prediction['posetrack_confidence'].squeeze(-1).float().clamp(1e-4,1-1e-4)
        confidence_loss = F.binary_cross_entropy(confidence[owned], confidence_target[owned])
        if not visible.any():
            reprojection=projected.sum()*0.
            direct_loss=direct.sum()*0.
        if not owned.any():
            confidence_loss=confidence.sum()*0.
    total = (reprojection + args.pose_projected_weight*direct_loss
             + args.pose_confidence_weight*confidence_loss)
    return dict(pose_2d=total,pose_2d_assisted=reprojection,pose_2d_projected=direct_loss,
                pose_confidence_bce=confidence_loss,pose_visible_points=visible.sum().float())
