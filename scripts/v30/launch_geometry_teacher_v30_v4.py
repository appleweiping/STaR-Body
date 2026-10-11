"""Finite geometry preparation with pilot equivalence and bounded resources."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/v30_geometry_teacher_v4_20261007'
PILOT=ROOT/'runs/v30_geometry_teacher_v1_20261007/pilot_raw'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def status(phase,**values):
    atomic(OUT/'status.json',dict(status=phase,sota_status='not_established',test_frames_used=False,**values))


def assert_originals():
    original=json.loads((ROOT/'original_code_hashes.json').read_text())
    if any(digest(ROOT/name)!=sha for name,sha in original.items()):raise ValueError('Original code changed')
    return len(original)


def run(command,label,teacher=False):
    deadline=time.monotonic()+7200
    while True:
        free,util=map(int,subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.free,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True).strip().split(','))
        available=int(next(line for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')).split()[1])
        architecture=ROOT/'training_runs/v30_architecture_v3_20261007'
        active=json.loads((architecture/'status.json').read_text());owned=None
        if active.get('status')=='formal_architecture_training' and active.get('active_arm'):
            process=json.loads((architecture/(active['active_arm']+'_process.json')).read_text())
            proc=Path('/proc')/str(process['pid'])/'cmdline'
            if proc.exists() and b'train_spatiotemporal_v30_v3' in proc.read_bytes() and str(architecture).encode() in proc.read_bytes():owned=process['pid']
        usage=subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).strip()
        occupants={int(row.split(',')[0]):int(row.split(',')[1]) for row in usage.splitlines() if row.strip()}
        allowed={1690842}
        if owned is not None:allowed.add(owned)
        known=not (set(occupants)-allowed) and occupants.get(1690842,0)<=2048
        busy_ok=(owned in occupants and util<=50) if owned is not None else util<5
        if known and busy_ok and free>=14336 and available>=12*1024**2:break
        status('waiting_resources',next_phase=label,gpu=1,shared_with_owned_v30_pid=owned,free_gpu_mib=free,available_host_kib=available,known_occupants=known)
        if time.monotonic()>deadline:raise TimeoutError('Two-hour finite resource wait exceeded')
        time.sleep(20)
    status(label,gpu=1,shared_with_owned_v30_pid=owned)
    environment=os.environ.copy();environment.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    if teacher:environment['PYTHONPATH']=str(ROOT.parent/'divibody_vace/env/lib/python3.10/site-packages')+':'+str(ROOT.parent)
    with (OUT/(label+'.log')).open('x') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
        atomic(OUT/(label+'_process.json'),dict(pid=child.pid,command=command,gpu=1))
        code=child.wait()
    if code:raise RuntimeError(label+' exit '+str(code))
    assert_originals()


def extraction(cache,split,count,output,label):
    run([str(ROOT/'env_teacher/bin/python'),'-u',str(ROOT/'scripts/v30/prepare_geometry_teacher_v30_v1.py'),
        '--root',str(ROOT),'--cache',str(cache),'--split',split,'--max-frames',str(count),'--output',str(output),
        '--dinov3-source-root',str(ROOT/'dependencies/dinov3_source_20261006')],label,teacher=True)


def fit(teacher,hmr,output,label,pilot=False):
    command=[sys.executable,'-u',str(ROOT/'scripts/v30/fit_geometry_teacher_v30_v1.py'),
        '--root',str(ROOT),'--teacher',str(teacher),'--hmr-cache',str(hmr),'--output',str(output),'--steps','400','--batch-size','64']
    if pilot:command.append('--pilot-only')
    run(command,label)


def compare_pilot(directory):
    import numpy as np
    previous=ROOT/'runs/sam3d_strong_features64_v4_20261006'
    old=json.loads((previous/'result.json').read_text());new=json.loads((directory/'result.json').read_text())
    if new['completed_frames']!=64 or new.get('full_geometric_teacher') is not True:raise ValueError('Incomplete actual pilot')
    if [r['uid'] for r in old['samples']]!=[r['uid'] for r in new['samples']]:raise ValueError('Pilot UID order changed')
    frozen=json.loads((ROOT/'runs/mhr_smpl_anchor_pilot_v3_20261007/result.json').read_text())
    evidence={r['uid']:r for r in frozen['evidence']}
    if set(evidence)!={r['uid'] for r in old['samples']}:raise ValueError('Frozen source evidence incomplete')
    maximum={};keys=('pred_vertices','pred_cam_t','pred_keypoints_3d','pred_keypoints_2d','bbox','focal_length','strong_rgb_pooled')
    for before,after in zip(old['samples'],new['samples']):
        proof=evidence[before['uid']]
        if Path(before['image']).resolve()!=Path(after['image']).resolve() or proof['image_sha256']!=after['image_sha256'] or digest(before['image'])!=proof['image_sha256']:
            raise ValueError('Pilot image differs from frozen evidence')
        if digest(previous/before['file'])!=proof['npz_sha256'] or digest(directory/after['file'])!=after['npz_sha256']:
            raise ValueError('Pilot evidence differs from preserved 64-frame fitting hashes')
        with np.load(previous/before['file'],allow_pickle=False) as a,np.load(directory/after['file'],allow_pickle=False) as b:
            for key in keys:
                tolerance=1e-3 if key=='pred_keypoints_2d' else (2e-6 if key in ('pred_vertices','pred_cam_t','pred_keypoints_3d') else 0.)
                if a[key].shape!=b[key].shape or not np.allclose(a[key],b[key],atol=tolerance,rtol=0):
                    raise ValueError('Pilot prediction equivalence failed: '+key)
                maximum[key]=max(maximum.get(key,0.),float(np.abs(a[key]-b[key]).max()))
    atomic(OUT/'pilot_equivalence.json',dict(status='passed',frames=64,maximum_absolute_difference=maximum,
        old_result_sha256=digest(previous/'result.json'),new_result_sha256=digest(directory/'result.json'),
        comparisons='Actual same UID, RGB, crop and full-body predictions; no GT',
        tolerance_policy='2 micrometers for FP32 camera/geometry; 0.001 pixel for perspective projection; bbox/focal/pooled features exact',
        previous_diagnostic_sha256=digest(ROOT/'runs/v30_geometry_teacher_v2_20261007/pilot_difference_diagnostic.json'),sota_status='not_established'))


def compare_fitted_pilot(directory):
    import numpy as np
    old=ROOT/'runs/mhr_smpl_anchor_pilot_v3_20261007/fitted_train_anchors.npz'
    new=directory/'000000.npz'
    with np.load(old,allow_pickle=False) as a,np.load(new,allow_pickle=False) as b:
        if not np.array_equal(a['uids'],b['uids']):raise ValueError('Fitted pilot UID changed')
        maximum={}
        for key in ('rotmat','betas','translation'):
            if not np.allclose(a[key],b[key],atol=1e-4,rtol=1e-4):raise ValueError('Fitted pilot differs: '+key)
            maximum[key]=float(np.abs(a[key]-b[key]).max())
    atomic(OUT/'pilot_fit_equivalence.json',dict(status='passed',frames=64,maximum_absolute_difference=maximum,
        gt_used_for_fitting=False,comparison='Same frozen 400-step solver and HMR initialization'))


def main():
    OUT.mkdir(exist_ok=False);assert_originals()
    installation=json.loads((ROOT/'scripts/v30/geometry_teacher_installation_v4.json').read_text())
    for name,sha in installation['sha256'].items():
        if digest(ROOT/'scripts/v30'/name)!=sha:raise ValueError('Preparation code changed')
    plan=json.loads((ROOT/'training_runs/v29y_strong_rgb_20261006/plan.json').read_text())['strong_motion_40']
    pilot_report=json.loads((ROOT/'runs/mhr_smpl_anchor_pilot_v3_20261007/result.json').read_text())
    if pilot_report['status']!='complete' or pilot_report['frames']!=64 or pilot_report['gt_used_for_fitting'] is not False:
        raise ValueError('Prior topology/coordinate/fitting evidence missing')
    if shutil.disk_usage(ROOT).free<30*1024**3:raise ValueError('At least 30 GiB free disk required')
    atomic(OUT/'plan.json',dict(train_frames=22735,validation_frames=10413,steps_per_fitting_batch=400,
        stages=['pilot_extraction','pilot_fitting','full_train_extraction','full_train_fitting','full_validation_extraction','full_validation_fitting'],
        input_cache_sha256={key:digest(plan[key]) for key in ('train-vace-cache','train-hmr-sidecar','val-vace-cache','val-hmr-sidecar')},
        implementation_sha256=installation['sha256'],test_used_for_selection=False,gt_used_for_fitting=False,
        model_training_started=False,resource_policy='Share GPU1 with verified owned V30; no unknown occupants; preserve existing <=2 GiB reservation; min 14 GiB GPU free and 12 GiB host available; util <=50 percent with owned training or <5 percent alone; finite waits',sota_status='not_established'))
    with (ROOT/'training_runs/.v30_teacher_gpu1_shared.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        completed=json.loads((PILOT/'result.json').read_text())
        if completed['status']!='passed' or completed['completed_frames']!=64 or completed['manifest']['source_script_sha256']!=installation['sha256']['prepare_geometry_teacher_v30_v1.py']:
            raise ValueError('Cannot reuse incomplete or differently generated pilot')
        atomic(OUT/'reused_pilot.json',dict(path=str(PILOT),result_sha256=digest(PILOT/'result.json'),reason='Completed actual prediction; v1 controller failed on absent legacy SHA field'))
        compare_pilot(PILOT)
        fit(PILOT,plan['train-hmr-sidecar'],OUT/'pilot_smpl','pilot_fitting',pilot=True)
        compare_fitted_pilot(OUT/'pilot_smpl')
        for split,count,prefix in [('train',22735,'train'),('validation',10413,'val')]:
            raw=OUT/(split+'_raw');anchors=OUT/(split+'_smpl')
            extraction(plan[prefix+'-vace-cache'],split,count,raw,'full_'+split+'_extraction')
            result=json.loads((raw/'result.json').read_text())
            if not result['complete_split_cache'] or result['completed_frames']!=count:
                raise ValueError('Full extraction did not cover preregistered split')
            fit(raw,plan[prefix+'-hmr-sidecar'],anchors,'full_'+split+'_fitting')
            result=json.loads((anchors/'result.json').read_text())
            if result['status']!='complete' or result['frames']!=count or result['development_only']:
                raise ValueError('Full native-SMPL fitting incomplete')
    status('complete',original_python_files_unchanged=assert_originals(),model_training_started=False,
        train_manifest=str(OUT/'train_smpl/manifest.json'),validation_manifest=str(OUT/'validation_smpl/manifest.json'))


if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():status('failed',error=repr(error))
        raise
