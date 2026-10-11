"""Prepare immutable, split-specific real SAM/DINO RGB backbone evidence."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch


def uid_digest(uids):
    return hashlib.sha256(('\n'.join(uids)+'\n').encode('utf-8')).hexdigest()


def feature_tree(value, path='output'):
    """Describe tensor outputs without saving image features in the manifest."""
    if torch.is_tensor(value):
        return {path:dict(shape=list(value.shape),dtype=str(value.dtype))}
    result={}
    if isinstance(value,dict):
        for key,item in value.items():result.update(feature_tree(item,f'{path}.{key}'))
    elif isinstance(value,(tuple,list)):
        for index,item in enumerate(value):result.update(feature_tree(item,f'{path}[{index}]'))
    return result


def image_feature_candidates(value, path='output'):
    if torch.is_tensor(value):
        return [(path,value)] if value.ndim==4 else []
    result=[]
    if isinstance(value,dict):
        for key,item in value.items():result.extend(image_feature_candidates(item,f'{path}.{key}'))
    elif isinstance(value,(tuple,list)):
        for index,item in enumerate(value):result.extend(image_feature_candidates(item,f'{path}[{index}]'))
    return result


def feature_channel_hints(backbone):
    hints=set()
    for module in backbone.modules():
        for name in ('embed_dim','num_features','out_channels'):
            value=getattr(module,name,None)
            if isinstance(value,int) and not isinstance(value,bool) and value>=32:hints.add(value)
    return hints


def pool_image_feature(output, channel_hints):
    """Pool an unambiguous genuine BCHW or BHWC backbone image feature."""
    candidates=[]
    for path,tensor in image_feature_candidates(output):
        shape=tuple(tensor.shape)
        if shape[0]!=1 or not torch.is_floating_point(tensor):continue
        layouts=[]
        for layout,channel_axis,spatial_axes in (('BCHW',1,(2,3)),('BHWC',3,(1,2))):
            channel=shape[channel_axis]
            spatial=[shape[axis] for axis in spatial_axes]
            if min(spatial)<2 or channel<32:continue
            has_hint=channel in channel_hints
            obvious_channel=channel>max(spatial)
            if has_hint or obvious_channel:
                layouts.append((has_hint,layout,spatial_axes))
        # Prefer a declared feature width; otherwise require a unique layout.
        hinted=[item for item in layouts if item[0]]
        layouts=hinted or layouts
        if len(layouts)!=1:continue
        _,layout,axes=layouts[0]
        preferred=path.rsplit('.',1)[-1] in ('features','image_embeddings','last_hidden_state','feature_map')
        candidates.append((preferred,path,tensor,layout,axes))
    preferred=[item for item in candidates if item[0]]
    candidates=preferred or candidates
    if len(candidates)!=1:
        raise ValueError(f'Backbone must expose one unambiguous 4D image feature; candidates={[item[1] for item in candidates]}, outputs={feature_tree(output)}')
    _,path,tensor,layout,axes=candidates[0]
    pooled=tensor.detach().float().mean(dim=axes)[0]
    if pooled.ndim!=1 or not torch.isfinite(pooled).all():
        raise FloatingPointError('Invalid real backbone pooled feature')
    return pooled.cpu().numpy().astype(np.float32,copy=False),dict(
        output_path=path,forward_shape=list(tensor.shape),layout=layout,
        pooled_dim=int(pooled.numel()),output_tree=feature_tree(output))


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--max-frames',type=int,default=64)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dinov3-source-root',type=Path,required=True)
    p.add_argument('--split',choices=('train','validation'),required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--backbone-only',action='store_true')
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    if a.max_frames<1:raise ValueError('Finite positive frame budget required')
    root=a.root.resolve();source=root.parent/'divibody_vace';sam_source=root.parent
    sys.path.insert(0,str(sam_source))
    from sam_3d_body import load_sam_3d_body,SAM3DBodyEstimator
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(10*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
    torch.manual_seed(2026);np.random.seed(2026)
    cache=a.cache.resolve()
    data=torch.load(cache,map_location='cpu',weights_only=False,mmap=True)
    if data['signature'].get('split')!=a.split or data['signature'].get('development_only'):
        raise ValueError('Teacher preparation requires complete declared matching split cache')
    samples=[];seen=set()
    for row,record in enumerate(data['metadata']):
        for col,uid in enumerate(record['target_frame_uids']):
            if not bool(data['frame_owner_mask'][row,col] and data['target_valid_mask'][row,col]):continue
            if uid in seen:raise ValueError('Duplicate train owner')
            seen.add(uid)
            annotation_id=int(record['target_frame_ids'][col])
            mapped=record.get('target_source_frame_ids')
            if data['signature'].get('strict_rgb_resolver')!='image_{annotation_frame_index:05d}':
                raise ValueError('Cache must declare the audited annotation-index RGB filename resolver')
            source_id=int(mapped[col]) if mapped is not None else annotation_id
            image=source/'datasets_full/3DPW/imageFiles'/record['sequence']/f"image_{annotation_id:05d}.jpg"
            if Path(record['context_image_paths'][4+col]).resolve()!=image.resolve():
                raise ValueError('Strong image identity must match the existing VACE target-context RGB path')
            if not image.is_file():raise FileNotFoundError(image)
            bbox=np.asarray(record['context_bbox_xyxy'][4+col],dtype=np.float32).reshape(1,4)
            if not np.isfinite(bbox).all() or np.any(bbox[0,2:]<=bbox[0,:2]):raise ValueError('Invalid known training crop')
            samples.append((uid,image,bbox,annotation_id,source_id))
    if not samples:raise ValueError('No owned valid train RGB frames')
    samples.sort(key=lambda item:item[0])
    selection_count=min(a.max_frames,len(samples))
    selection_indices=np.linspace(0,len(samples)-1,num=selection_count,dtype=np.int64).tolist()
    if len(set(selection_indices))!=selection_count:raise ValueError('Non-unique deterministic sampling indices')
    selected=[samples[index] for index in selection_indices]
    selected_uids=[item[0] for item in selected]
    ckpt=sam_source/'checkpoints/sam-3d-body-dinov3/model.ckpt'
    mhr=sam_source/'checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt'
    manifest=dict(format='v29-strong-rgb-feature-cache-v5',split=a.split,source_cache=str(cache),source_vace_cache_sha256=digest(cache),unique_train_frames=len(samples),
        requested_frames=a.max_frames,checkpoint_sha256=digest(ckpt),mhr_sha256=digest(mhr),
        train_frame_manifest_sha256=uid_digest(sorted(seen)),
        selected_train_uids=selected_uids,selected_train_uid_sha256=uid_digest(selected_uids),
        selected_train_indices=selection_indices,selected_frames=selection_count,
        sampling_policy='UID-sorted complete train owner list; deterministic evenly spaced indices across its full range',
        selected_sequence_count=len({str(item[1].parent.name) for item in selected}),
        input_bbox_policy='existing split-specific poses2d person crop from VACE metadata',
        teacher_inputs=['real RGB image','existing annotation-derived person bbox'],
        three_d_targets_not_teacher_inputs=True,bbox_annotation_derived=True,
        label_policy='No 3D pose, betas, gender, camera pose or target features are passed to the teacher; annotation-derived crop is disclosed separately',
        representation='Real SAM/DINOv3 pooled RGB backbone features only; no MHR 3D targets or pseudo-SMPL supervision',
        no_test_frames=True,teacher_not_yet_used_for_training=True,
        source_frame_policy='target_source_frame_ids are capture/time IDs; actual RGB filename uses cache-declared strict annotation index and must equal context_image_paths[4+target_column]',
        backbone_only=a.backbone_only,source_script_sha256=digest(__file__),
        complete_owner_uid_sha256=uid_digest(sorted(seen)),unique_split_frames=len(samples),
        selected_uids=selected_uids,selected_uid_sha256=uid_digest(selected_uids),complete=False,
        images_root=str((source/'datasets_full/3DPW/imageFiles').resolve()))
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
    import yaml
    config_text=json.dumps(yaml.safe_load(config.dump()),sort_keys=True)
    manifest['encoder_config_sha256']=hashlib.sha256(config_text.encode()).hexdigest()
    (a.output/'encoder_config.json').write_text(config_text)
    estimator=SAM3DBodyEstimator(model,config)
    if not isinstance(getattr(model,'backbone',None),torch.nn.Module):raise TypeError('SAM model has no torch backbone')
    backbone_source=Path(inspect.getfile(type(model.backbone))).resolve()
    if not backbone_source.is_file():raise FileNotFoundError(backbone_source)
    manifest['model_backbone_source']=str(backbone_source)
    manifest['model_backbone_source_sha256']=digest(backbone_source)
    manifest['model_backbone_type']=f'{type(model.backbone).__module__}.{type(model.backbone).__qualname__}'
    manifest['feature_policy']='real-sam-dinov3-global-spatial-mean'
    manifest['feature_description']='Actual model.backbone body forward output, global spatial mean, float32; no random or placeholder features'
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    channel_hints=feature_channel_hints(model.backbone)
    captured=[];hook_active=False
    class BackboneComplete(Exception):pass
    def capture_body_feature(module,inputs,output):
        if hook_active:
            captured.append(pool_image_feature(output,channel_hints))
            if a.backbone_only:raise BackboneComplete()
    hook=model.backbone.register_forward_hook(capture_body_feature)
    rows=[];started=time.monotonic()
    try:
        for uid,image,bbox,annotation_id,source_id in selected:
            captured.clear();hook_active=True
            try:
                try:
                    with torch.inference_mode():result=estimator.process_one_image(str(image),bboxes=bbox,inference_type='body')
                except BackboneComplete:
                    if not a.backbone_only or len(captured)!=1:raise
                    result=None
            finally:hook_active=False
            if not a.backbone_only and len(result)!=1:raise ValueError('Teacher must return one requested person')
            if len(captured)!=1:raise ValueError(f'Expected one body backbone forward for requested person, got {len(captured)}')
            pooled,feature_info=captured[0]
            if rows and feature_info['pooled_dim']!=rows[0]['strong_rgb_feature']['pooled_dim']:
                raise ValueError('Inconsistent backbone pooled dimension')
            arrays=dict(strong_rgb_pooled=pooled,bbox=bbox.reshape(4))
            for key,value in arrays.items():
                if np.issubdtype(value.dtype,np.number) and not np.isfinite(value).all():raise FloatingPointError(key)
            path=a.output/f'{len(rows):06d}.npz';np.savez_compressed(path,**arrays)
            rows.append(dict(uid=uid,image=str(image),image_sha256=digest(image),file=path.name,npz_sha256=digest(path),
                annotation_frame_id=annotation_id,source_frame_id=source_id,bbox_xyxy=bbox.reshape(4).tolist(),
                shapes={k:list(v.shape) for k,v in arrays.items()},strong_rgb_feature=feature_info))
            manifest['strong_rgb_pooled_dim']=feature_info['pooled_dim']
            manifest['feature_dim']=feature_info['pooled_dim']
            manifest['strong_rgb_forward_shape']=feature_info['forward_shape']
            if len(rows)%100==0 or len(rows)==selection_count:
                (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
                (a.output/'progress.json').write_text(json.dumps(dict(completed_frames=len(rows),finite_budget=a.max_frames,selected_frames=selection_count),indent=2))
                print(json.dumps(dict(event='teacher_real_rgb',uid=uid,completed=len(rows),strong_rgb_pooled_dim=feature_info['pooled_dim'],seconds=time.monotonic()-started)),flush=True)
    finally:hook_active=False;hook.remove()
    manifest['complete']=len(rows)==len(samples)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    result=dict(status='passed',real_rgb=True,random_features=False,completed_frames=len(rows),
        seconds=time.monotonic()-started,manifest=manifest,samples=rows,
        full_training_cache=a.split=='train' and len(rows)==len(samples),complete_split_cache=len(rows)==len(samples),sota_status='not_established')
    (a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('manifest','samples')}))


if __name__=='__main__':main()
