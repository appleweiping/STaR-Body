"""Evaluate owned frames once and include valid acceleration across windows."""
import json
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
def validate(loader,model,models,regressor,args,device):
    from divibody_training.target60_runtime_v34 import forward_batch
    if not isinstance(loader.sampler,SequentialSampler) or loader.drop_last:
        raise ValueError('Owned-frame evaluation requires sequential complete sampling')
    dataset=loader.dataset
    indices=list(range(len(dataset)))
    if isinstance(dataset,Subset):
        indices=list(dataset.indices);dataset=dataset.dataset
    metadata=dataset.vace['metadata']
    model.eval();cursor=0;count=0
    sums=dict(mpjpe_mm=0.,pa_mpjpe_mm=0.,pve_mm=0.)
    errors={}
    sequences={}
    for batch_index,batch in enumerate(loader):
        pred=forward_batch(model,batch,device,args.amp_dtype)
        b,t=pred['rotmat'].shape[:2]
        pv=smpl_vertices(models[2],pred['rotmat'].float().reshape(-1,24,3,3),pred['betas'].float().reshape(-1,10))
        gv=gendered_vertices(models,camera_targets(batch,device).reshape(-1,24,3,3),
                              batch['betas'].to(device,dtype=torch.float32).reshape(-1,10),
                              batch['gender_code'].to(device).reshape(-1))
        pj,_=pelvis_aligned_j14(pv,regressor);gj,_=pelvis_aligned_j14(gv,regressor)
        owner=(batch['frame_owner']&batch['target_valid']).to(device)
        values=dict(mpjpe_mm=mean_euclidean_error(pj,gj)*1000,
                    pa_mpjpe_mm=pa_mpjpe_per_sample(pj,gj)*1000,
                    pve_mm=mean_euclidean_error(pv,gv)*1000)
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
    acceleration,triplets=sequence_acceleration(errors)
    return {k:v/count for k,v in sums.items()}|dict(acceleration_error_mm_per_frame2=acceleration,
                                                  frames=count,acceleration_triplets=triplets,per_sequence=sequences)
