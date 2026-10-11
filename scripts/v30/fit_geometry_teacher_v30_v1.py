"""Fit full train/validation pseudo-SMPL anchors without 3D labels."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from mhr_smpl_anchor_v30_v1 import mapped_surface, fit_smpl_surface


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--teacher',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--hmr-cache',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=64);p.add_argument('--steps',type=int,default=400)
    p.add_argument('--pilot-only',action='store_true')
    a=p.parse_args();root=a.root.resolve();a.output.mkdir(exist_ok=False)
    if not 1<=a.batch_size<=64 or a.steps!=400:raise ValueError('Frozen pilot-fitting budget required')
    torch.set_num_threads(4);torch.manual_seed(2026);device=torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
    original=json.loads((root/'original_code_hashes.json').read_text())
    if any(digest(root/name)!=sha for name,sha in original.items()):raise ValueError('Original code changed')
    extraction=json.loads((a.teacher/'result.json').read_text());declaration=extraction['manifest']
    if (extraction['status']!='passed' or (not extraction.get('complete_split_cache') and not a.pilot_only)
        or declaration.get('format')!='v30-real-mhr-geometry-cache-v1' or (not declaration['complete'] and not a.pilot_only)
        or declaration['split'] not in ('train','validation') or declaration['no_test_frames'] is not True
        or extraction.get('three_d_gt_used_for_inference') is not False):
        raise ValueError('Completed actual geometry extraction required; no test fitting')
    code=root/'research_dependencies/MHR_conversion_sources_v2_20261007'
    source_manifest=json.loads((code/'source_manifest.json').read_text())
    if source_manifest['status']!='complete':raise ValueError('Incomplete conversion snapshot')
    for row in source_manifest['files']:
        if digest(code/row['path'])!=row['sha256']:raise ValueError('Conversion asset changed')
    mapping=code/'tools/mhr_smpl_conversion/assets/mhr2smpl_mapping.npz'
    with np.load(mapping,allow_pickle=False) as d:
        triangles=torch.tensor(d['triangle_ids'].astype(np.int64),device=device)
        barycentric=torch.tensor(d['baryc_coords'],device=device,dtype=torch.float32)
    checkpoint=root.parent/'checkpoints/sam-3d-body-dinov3/model.ckpt'
    native=root.parent/'checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt'
    if digest(checkpoint)!=declaration['checkpoint_sha256'] or digest(native)!=declaration['mhr_sha256']:
        raise ValueError('Frozen teacher assets changed')
    state=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    faces_cpu=state['head_pose.faces'].long()
    jit=torch.jit.load(str(native),map_location='cpu')
    if not torch.equal(faces_cpu,dict(jit.named_buffers())['character_torch.mesh.faces'].long()):
        raise ValueError('MHR topology differs')
    faces=faces_cpu.to(device);del state,jit
    vace_path=Path(declaration['source_cache'])
    if digest(vace_path)!=declaration['source_vace_cache_sha256']:raise ValueError('VACE source changed')
    vace=torch.load(vace_path,map_location='cpu',mmap=True,weights_only=False)
    hmr=torch.load(a.hmr_cache,map_location='cpu',mmap=True,weights_only=False)
    split=declaration['split']
    if vace['signature']['split']!=split or hmr['signature']['split']!=split:raise ValueError('Split mismatch')
    from divibody_training.ram_parts_runtime_v29 import V10Dataset
    from divibody_training.evaluate_standard_3dpw import create_smpl_models
    dataset=V10Dataset(vace,hmr)
    lookup={}
    for index,metadata in enumerate(vace['metadata']):
        for col,uid in enumerate(metadata['target_frame_uids']):
            if bool(vace['frame_owner_mask'][index,col] and vace['target_valid_mask'][index,col]):
                if uid in lookup:raise ValueError('Duplicate owner')
                lookup[uid]=(index,col)
    rows=extraction['samples'];uids=[r['uid'] for r in rows]
    expected=sorted(lookup)
    if a.pilot_only:
        if split!='train' or len(uids)!=64:raise ValueError('Pilot must use fixed 64 train frames')
        expected=[expected[int(i)] for i in np.linspace(0,len(expected)-1,64,dtype=np.int64)]
    if uids!=expected or declaration['selected_uids']!=uids:raise ValueError('Declared split coverage differs')
    config=json.loads((root/'configs/geometry.json').read_text())
    smpl=create_smpl_models(Path(config['smpl-model-root']),device)[2]
    metadata=dict(format='v30-native-smpl-teacher-anchors-v1',split=split,complete=False,development_only=a.pilot_only,
        teacher_result_sha256=digest(a.teacher/'result.json'),hmr_cache=str(a.hmr_cache.resolve()),
        hmr_cache_sha256=digest(a.hmr_cache),vace_cache_sha256=declaration['source_vace_cache_sha256'],
        teacher_checkpoint_sha256=declaration['checkpoint_sha256'],mapping_sha256=digest(mapping),
        fitter_sha256=digest(__file__),solver_sha256=digest(root/'scripts/v30/mhr_smpl_anchor_v30_v1.py'),
        selected_uids=uids,frames=len(uids),steps=a.steps,batch_size=a.batch_size,
        gt_used_for_fitting=False,teacher_annotation_derived_bbox=True,test_frames_used=False,
        camera_translation_not_replaced=True,translation_policy='Fitting translation separate from HMR camera',
        betas_bound=5.,per_frame_independent=True,official_parameter_solver=False,
        fit_selection='Minimum teacher surface residual only; no GT filtering',sota_status='not_established')
    atomic(a.output/'manifest.json',metadata);files=[];started=time.monotonic();residuals=[]
    for start in range(0,len(rows),a.batch_size):
        group=rows[start:start+a.batch_size];surfaces=[];rotations=[];betas=[];evidence=[]
        for row in group:
            index,col=lookup[row['uid']];record=vace['metadata'][index]
            path=a.teacher/row['file']
            if path.parent.resolve()!=a.teacher.resolve() or digest(path)!=row['npz_sha256']:
                raise ValueError('Teacher file identity differs')
            image=Path(row['image'])
            if image.resolve()!=Path(record['context_image_paths'][4+col]).resolve() or digest(image)!=row['image_sha256']:
                raise ValueError('Actual RGB differs')
            with np.load(path,allow_pickle=False) as d:
                if not np.allclose(d['bbox'],record['context_bbox_xyxy'][4+col],atol=1e-4,rtol=0):
                    raise ValueError('Teacher crop differs')
                surfaces.append(torch.tensor(d['pred_vertices'].copy(),device=device,dtype=torch.float32))
            sample=dataset[index]
            # Only HMR predictions enter fitting; labels exposed by the legacy dataset are unused.
            rotations.append(sample['base_rotmat'][col]);betas.append(sample['base_betas'][col])
            evidence.append(dict(uid=row['uid'],teacher_npz_sha256=row['npz_sha256'],image_sha256=row['image_sha256']))
        target=mapped_surface(torch.stack(surfaces),faces,triangles,barycentric)
        fit=fit_smpl_surface(smpl,target,torch.stack(rotations).to(device).float(),torch.stack(betas).to(device).float(),steps=a.steps)
        path=a.output/f'{len(files):06d}.npz'
        values=dict(rotmat=fit['rotmat'].cpu().numpy(),betas=fit['betas'].cpu().numpy(),
            translation=fit['translation'].cpu().numpy(),surface_error_mm=fit['surface_error_mm'].cpu().numpy(),
            initial_surface_error_mm=fit['initial_surface_error_mm'].cpu().numpy(),uids=np.asarray([r['uid'] for r in group]))
        for key,value in values.items():
            if key!='uids' and not np.isfinite(value).all():raise FloatingPointError(key)
        with path.open('xb') as stream:np.savez_compressed(stream,**values)
        files.append(dict(file=path.name,sha256=digest(path),frames=len(group),start=start,evidence=evidence))
        residuals.extend(values['surface_error_mm'].tolist())
        atomic(a.output/'progress.json',dict(completed_frames=start+len(group),target_frames=len(rows),batches=len(files),
            seconds=time.monotonic()-started,mean_surface_error_mm=float(np.mean(residuals))))
        print(json.dumps(dict(event='gt_free_smpl_fit',completed=start+len(group),total=len(rows),seconds=time.monotonic()-started)),flush=True)
    if any(digest(root/name)!=sha for name,sha in original.items()):raise ValueError('Original code changed during fit')
    metadata.update(complete=True,files=files,mean_surface_error_mm=float(np.mean(residuals)),
        p95_surface_error_mm=float(np.percentile(residuals,95)),original_python_files_unchanged=len(original))
    atomic(a.output/'manifest.json',metadata)
    atomic(a.output/'result.json',dict(status='complete',frames=len(rows),split=split,seconds=time.monotonic()-started,
        manifest_sha256=digest(a.output/'manifest.json'),development_only=a.pilot_only,gt_used_for_fitting=False,model_training_started=False,sota_status='not_established'))


if __name__=='__main__':main()
