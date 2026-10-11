"""Copied project metrics with injected forward, coverage checks, and resource timing.

No gradients, optimization, checkpoint selection, or label-based hypothesis
selection is performed. Metric formulas are unchanged from the V29 reporter.
"""
import json
import time
import torch
from torch.utils.data import Subset, SequentialSampler
from divibody_training.geometry_metrics_3dpw import (
    mean_euclidean_error,pa_mpjpe_per_sample,pelvis_aligned_j14,hip_centered_j14)
from divibody_training.train_divibody_continuous_geometry_v5 import camera_targets,gendered_vertices,smpl_vertices


def sequence_acceleration(errors):
    total=0.
    count=0
    for uid,value in errors.items():
        prefix,frame=uid.rsplit('/frame_',1)
        index=int(frame)
        left=f'{prefix}/frame_{index-1:05d}'
        right=f'{prefix}/frame_{index+1:05d}'
        if left in errors and right in errors:
            total+=float(torch.linalg.vector_norm(errors[left]-2*value+errors[right],dim=-1).mean())*1000
            count+=1
    return total/count if count else 0.,count


@torch.no_grad()
def validate(loader,model,models,regressor,args,device,*,forward_batch,expected_uids):
    if not isinstance(loader.sampler,SequentialSampler) or loader.drop_last:
        raise ValueError('Owned-frame evaluation requires sequential complete sampling')
    dataset=loader.dataset
    indices=list(range(len(dataset)))
    if isinstance(dataset,Subset):
        indices=list(dataset.indices);dataset=dataset.dataset
    metadata=dataset.vace['metadata']
    model.eval();cursor=0;count=0
    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    synchronize()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started=time.perf_counter();forward_seconds=0.;forward_frames=0
    sums=dict(mpjpe_mm=0.,pa_mpjpe_mm=0.,pve_mm=0.)
    errors={}
    sequences={}
    for batch_index,batch in enumerate(loader):
        for key in ('rgb_tokens','vace_features','base_rotmat','base_betas','base_camera',
                    'full_pose','betas','cam_pose'):
            if not torch.isfinite(batch[key]).all():
                raise ValueError('Nonfinite evaluation input: '+key)
        synchronize();forward_started=time.perf_counter()
        pred=forward_batch(model,batch,device,args.amp_dtype)
        synchronize();forward_seconds+=time.perf_counter()-forward_started
        b,t=pred['rotmat'].shape[:2]
        forward_frames+=b*t
        for key in ('rotmat','betas','camera'):
            if not torch.isfinite(pred[key]).all():
                raise FloatingPointError('Nonfinite model prediction: '+key)
        pv=smpl_vertices(models[2],pred['rotmat'].float().reshape(-1,24,3,3),pred['betas'].float().reshape(-1,10))
        gv=gendered_vertices(models,camera_targets(batch,device).reshape(-1,24,3,3),
                              batch['betas'].to(device,dtype=torch.float32).reshape(-1,10),
                              batch['gender_code'].to(device).reshape(-1))
        pj,_=pelvis_aligned_j14(pv,regressor);gj,_=pelvis_aligned_j14(gv,regressor)
        owner=(batch['frame_owner']&batch['target_valid']).to(device)
        values=dict(mpjpe_mm=mean_euclidean_error(pj,gj)*1000,
                    pa_mpjpe_mm=pa_mpjpe_per_sample(pj,gj)*1000,
                    pve_mm=mean_euclidean_error(pv,gv)*1000)
        if any(not torch.isfinite(value).all() for value in values.values()):
            raise FloatingPointError('Nonfinite evaluation metric')
        for k,v in values.items():sums[k]+=float(v.masked_select(owner.reshape(-1)).double().sum())
        motion=(hip_centered_j14(pj)-hip_centered_j14(gj)).reshape(b,t,14,3).cpu()
        for row in range(b):
            record=metadata[indices[cursor+row]]
            for col,uid in enumerate(record['target_frame_uids']):
                if bool(owner[row,col]):
                    if uid in errors:raise ValueError('Duplicate owned frame: '+uid)
                    errors[uid]=motion[row,col]
                    sequence=uid.split('/person_',1)[0]
                    summary=sequences.setdefault(sequence,dict(frames=0,mpjpe_mm_sum=0.,pa_mpjpe_mm_sum=0.,pve_mm_sum=0.))
                    summary['frames']+=1
                    for key,value in values.items():summary[key+'_sum']+=float(value.reshape(b,t)[row,col])
        cursor+=b;count+=int(owner.sum())
        if batch_index%25==0:print(json.dumps(dict(event='geometry_validation_progress',windows=cursor,frames=count)),flush=True)
    if not count or cursor!=len(indices):raise ValueError('Incomplete or empty evaluation')
    if set(errors) != set(expected_uids):
        raise ValueError('Evaluated owned UID set differs from the audited coverage')
    acceleration,triplets=sequence_acceleration(errors)
    for sequence,summary in sequences.items():
        subset={uid:value for uid,value in errors.items() if uid.split('/person_',1)[0] == sequence}
        seq_accel,seq_triplets=sequence_acceleration(subset)
        for key in ('mpjpe_mm','pa_mpjpe_mm','pve_mm'):
            summary[key]=summary.pop(key+'_sum')/summary['frames']
        summary.update(acceleration_error_mm_per_frame2=seq_accel,acceleration_triplets=seq_triplets,
                       tracks=len({uid.rsplit('/frame_',1)[0] for uid in subset}))
    synchronize();elapsed=time.perf_counter()-started
    resources=dict(evaluation_seconds=elapsed,model_forward_seconds=forward_seconds,
                   forward_frames_including_padding=forward_frames,
                   forward_ms_per_owned_frame=forward_seconds*1000/count,
                   forward_ms_per_input_frame=forward_seconds*1000/forward_frames,
                   owned_frames_per_evaluation_second=count/elapsed,
                   timing_scope='Forward includes input device transfer and all frozen-reference/SMPL feedback; evaluation includes metric SMPL and CPU aggregation',
                   warmup_iterations=0,cuda_synchronized=device.type=='cuda',
                   peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else None,
                   peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved(device)) if device.type=='cuda' else None)
    return {k:v/count for k,v in sums.items()}|dict(acceleration_error_mm_per_frame2=acceleration,
                                                  frames=count,acceleration_triplets=triplets,per_sequence=sequences,
                                                  resources=resources)
