"""Evaluate complete frozen native teacher anchors on validation, without fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--anchors',type=Path,required=True)
    p.add_argument('--teacher',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();a.output.mkdir(exist_ok=False)
    torch.set_num_threads(4);device=torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(6*1024**3/torch.cuda.get_device_properties(0).total_memory)
    native=json.loads((a.anchors/'manifest.json').read_text());completion=json.loads((a.anchors/'result.json').read_text())
    extraction=json.loads((a.teacher/'result.json').read_text())
    if (native.get('format')!='v30-native-smpl-teacher-anchors-v1' or native.get('split')!='validation'
        or native.get('complete') is not True or native.get('development_only') is not False
        or native.get('gt_used_for_fitting') is not False or native.get('test_frames_used') is not False
        or native.get('frames')!=10413 or completion.get('status')!='complete'
        or completion.get('manifest_sha256')!=digest(a.anchors/'manifest.json')
        or native['teacher_result_sha256']!=digest(a.teacher/'result.json')
        or extraction.get('complete_split_cache') is not True or extraction['manifest']['split']!='validation'):
        raise ValueError('Only frozen, complete, GT-free native validation anchors accepted')
    frozen_manifest_sha=digest(a.anchors/'manifest.json')
    original=json.loads((root/'original_code_hashes.json').read_text())
    if any(digest(root/name)!=sha for name,sha in original.items()):raise ValueError('Original code changed')
    plan=json.loads((root/'training_runs/v29y_strong_rgb_20261006/plan.json').read_text())['strong_motion_40']
    if digest(plan['val-vace-cache'])!=native['vace_cache_sha256'] or digest(plan['val-hmr-sidecar'])!=native['hmr_cache_sha256']:
        raise ValueError('Validation source differs')
    from divibody_training.ram_parts_runtime_v29 import V10Dataset
    from divibody_training.evaluate_standard_3dpw import create_smpl_models
    from divibody_training.v29_strong_features_v1 import expected_frame_records
    from divibody_training.train_divibody_continuous_geometry_v5 import camera_targets,gendered_vertices,smpl_vertices
    from divibody_training.geometry_metrics_3dpw import pelvis_aligned_j14,mean_euclidean_error,pa_mpjpe_per_sample,hip_centered_j14
    from divibody_training.v29_sequence_evaluation_report import sequence_acceleration
    vace=torch.load(plan['val-vace-cache'],mmap=True,map_location='cpu',weights_only=False)
    hmr=torch.load(plan['val-hmr-sidecar'],mmap=True,map_location='cpu',weights_only=False)
    expected=expected_frame_records(vace,'validation');uids=list(expected)
    if uids!=native['selected_uids'] or [r['uid'] for r in extraction['samples']]!=uids:raise ValueError('UID coverage differs')
    dataset=V10Dataset(vace,hmr);lookup={}
    for index,record in enumerate(vace['metadata']):
        for col,uid in enumerate(record['target_frame_uids']):
            if bool(vace['frame_owner_mask'][index,col] and vace['target_valid_mask'][index,col]):
                if uid in lookup:raise ValueError('Duplicate owner')
                lookup[uid]=(index,col)
    if set(lookup)!=set(uids):raise ValueError('Source ownership incomplete')
    config=json.loads((root/'configs/geometry.json').read_text());models=create_smpl_models(Path(config['smpl-model-root']),device)
    for model in models.values():model.eval().requires_grad_(False)
    regressor=torch.tensor(np.load(config['joint-regressor'],allow_pickle=False),device=device,dtype=torch.float32)
    baseline_v28=json.loads((root/'training_runs/v30_architecture_v3_20261007/spatial_motion/metrics.jsonl').read_text().splitlines()[0])['validation']
    keys=('mpjpe_mm','pa_mpjpe_mm','pve_mm');sums={name:dict.fromkeys(keys,0.) for name in ('teacher','hmr')}
    sequences={name:{} for name in sums};motion={name:{} for name in sums};cursor=0;started=time.monotonic()
    for row in native['files']:
        path=a.anchors/row['file']
        if path.parent.resolve()!=a.anchors.resolve() or digest(path)!=row['sha256']:raise ValueError('Native anchor changed')
        with np.load(path,allow_pickle=False) as d:
            batch_uids=d['uids'].tolist();rot=d['rotmat'].copy();betas=d['betas'].copy()
        if batch_uids!=uids[cursor:cursor+row['frames']] or row['start']!=cursor:
            raise ValueError('Native batch identities differ')
        n=len(batch_uids)
        if rot.shape!=(n,24,3,3) or betas.shape!=(n,10) or not np.isfinite(rot).all() or not np.isfinite(betas).all():
            raise ValueError('Invalid native anchor values')
        labels=[];hmr_r=[];hmr_b=[]
        for uid in batch_uids:
            index,col=lookup[uid];sample=dataset[index]
            labels.append({key:sample[key][col] for key in ('full_pose','betas','gender_code','cam_pose')})
            hmr_r.append(sample['base_rotmat'][col]);hmr_b.append(sample['base_betas'][col])
        batch={key:torch.stack([r[key] for r in labels])[:,None] for key in labels[0]}
        with torch.no_grad():
            gt=gendered_vertices(models,camera_targets(batch,device).reshape(-1,24,3,3),batch['betas'].to(device).reshape(-1,10),batch['gender_code'].to(device).reshape(-1))
            gj,_=pelvis_aligned_j14(gt,regressor)
            geometry={'teacher':smpl_vertices(models[2],torch.tensor(rot,device=device),torch.tensor(betas,device=device)),
                'hmr':smpl_vertices(models[2],torch.stack(hmr_r).to(device).float(),torch.stack(hmr_b).to(device).float())}
            for name,pred in geometry.items():
                pj,_=pelvis_aligned_j14(pred,regressor)
                values=dict(mpjpe_mm=mean_euclidean_error(pj,gj)*1000,pa_mpjpe_mm=pa_mpjpe_per_sample(pj,gj)*1000,pve_mm=mean_euclidean_error(pred,gt)*1000)
                errors=(hip_centered_j14(pj)-hip_centered_j14(gj)).cpu()
                for key in keys:sums[name][key]+=float(values[key].double().sum())
                for i,uid in enumerate(batch_uids):
                    motion[name][uid]=errors[i];seq=uid.split('/person_',1)[0]
                    summary=sequences[name].setdefault(seq,dict(frames=0,mpjpe_mm_sum=0.,pa_mpjpe_mm_sum=0.,pve_mm_sum=0.));summary['frames']+=1
                    for key in keys:summary[key+'_sum']+=float(values[key][i])
        cursor+=n
        print(json.dumps(dict(event='frozen_teacher_validation',frames=cursor,total=len(uids))),flush=True)
    if cursor!=10413 or digest(a.anchors/'manifest.json')!=frozen_manifest_sha:raise ValueError('Frozen coverage changed')
    report={}
    for name in sums:
        accel,count=sequence_acceleration(motion[name]);report[name]={k:v/cursor for k,v in sums[name].items()}|dict(
            frames=cursor,acceleration_error_mm_per_frame2=accel,acceleration_triplets=count,per_sequence=sequences[name])
        if count!=10226:raise ValueError('Validation acceleration coverage differs')
    result=dict(status='complete',split='validation',metrics=report,baseline_v28=baseline_v28,
        teacher_minus_v28={k:report['teacher'][k]-baseline_v28[k] for k in (*keys,'acceleration_error_mm_per_frame2')},
        native_manifest_sha256=frozen_manifest_sha,evaluator_sha256=digest(__file__),original_python_files_unchanged=len(original),
        gt_used_for_fitting=False,labels_used_only_for_postfit_reporting=True,student_model_performance=False,
        test_used_for_model_selection=False,seconds=time.monotonic()-started,sota_status='not_established')
    with (a.output/'result.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('metrics','baseline_v28')}))


if __name__=='__main__':main()
