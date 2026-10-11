"""Prepare real train RGB evidence with existing SAM 3D Body weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--max-frames',type=int,default=2)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dinov3-source-root',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    if a.max_frames<1:raise ValueError('Finite positive frame budget required')
    root=a.root.resolve();source=root.parent/'divibody_vace';sam_source=root.parent
    sys.path.insert(0,str(sam_source))
    from sam_3d_body import load_sam_3d_body,SAM3DBodyEstimator
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(10*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
    torch.manual_seed(2026);np.random.seed(2026)
    cache=source/'training_runs/continuous_vace_v5_full_zero_text/features_train.pt'
    data=torch.load(cache,map_location='cpu',weights_only=False,mmap=True)
    if data['signature'].get('split')!='train' or data['signature'].get('development_only'):
        raise ValueError('Teacher preparation requires complete declared train cache')
    samples=[];seen=set()
    for row,record in enumerate(data['metadata']):
        for col,uid in enumerate(record['target_frame_uids']):
            if not bool(data['frame_owner_mask'][row,col] and data['target_valid_mask'][row,col]):continue
            if uid in seen:raise ValueError('Duplicate train owner')
            seen.add(uid)
            image=source/'datasets_full/3DPW/imageFiles'/record['sequence']/f"image_{record['target_frame_ids'][col]:05d}.jpg"
            if not image.is_file():raise FileNotFoundError(image)
            bbox=np.asarray(record['context_bbox_xyxy'][4+col],dtype=np.float32).reshape(1,4)
            if not np.isfinite(bbox).all() or np.any(bbox[0,2:]<=bbox[0,:2]):raise ValueError('Invalid known training crop')
            samples.append((uid,image,bbox))
    ckpt=sam_source/'checkpoints/sam-3d-body-dinov3/model.ckpt'
    mhr=sam_source/'checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt'
    manifest=dict(split='train',source_cache=str(cache),unique_train_frames=len(samples),
        requested_frames=a.max_frames,checkpoint_sha256=digest(ckpt),mhr_sha256=digest(mhr),
        train_frame_manifest_sha256=hashlib.sha256(('\n'.join(sorted(seen))+'\n').encode()).hexdigest(),
        input_bbox_policy='existing train-only poses2d person crop from VACE metadata',
        representation='MHR; no SMPL or J14 comparison until conversion verified',
        no_test_frames=True,teacher_not_yet_used_for_training=True)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    if not (a.dinov3_source_root/'hubconf.py').is_file():raise FileNotFoundError('Verified local DINOv3 source is required')
    original_hub_load=torch.hub.load
    def local_hub_load(repo_or_dir,model,*args,**kwargs):
        if repo_or_dir=='facebookresearch/dinov3':
            kwargs.pop('trust_repo',None);kwargs['source']='local'
            return original_hub_load(str(a.dinov3_source_root),model,*args,**kwargs)
        return original_hub_load(repo_or_dir,model,*args,**kwargs)
    torch.hub.load=local_hub_load
    manifest['dinov3_hubconf_sha256']=digest(a.dinov3_source_root/'hubconf.py')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    model,config=load_sam_3d_body(str(ckpt),device='cuda',mhr_path=str(mhr))
    estimator=SAM3DBodyEstimator(model,config)
    rows=[];started=time.monotonic()
    for uid,image,bbox in samples[:a.max_frames]:
        with torch.inference_mode():result=estimator.process_one_image(str(image),bboxes=bbox,inference_type='body')
        if len(result)!=1:raise ValueError('Teacher must return one requested person')
        arrays={key:np.asarray(value) for key,value in result[0].items() if value is not None}
        for key,value in arrays.items():
            if np.issubdtype(value.dtype,np.number) and not np.isfinite(value).all():raise FloatingPointError(key)
        path=a.output/f'{len(rows):06d}.npz';np.savez_compressed(path,**arrays)
        rows.append(dict(uid=uid,image=str(image),file=path.name,shapes={k:list(v.shape) for k,v in arrays.items()}))
        (a.output/'progress.json').write_text(json.dumps(dict(completed_frames=len(rows),finite_budget=a.max_frames),indent=2))
        print(json.dumps(dict(event='teacher_real_rgb',uid=uid,completed=len(rows))),flush=True)
    result=dict(status='passed',real_rgb=True,random_features=False,completed_frames=len(rows),
        seconds=time.monotonic()-started,manifest=manifest,samples=rows,
        full_training_cache=len(rows)==len(samples),sota_status='not_established')
    (a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
