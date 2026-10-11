"""Finite, preregistered architecture experiment; no automatic test selection."""
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'training_runs/v30_architecture_v1_20261007'


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):value.update(block)
    return value.hexdigest()


def atomic(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)


def status(phase,**values):
    atomic(OUT/'status.json',dict(status=phase,sota_status='not_established',**values))


def run(config,gpu,label):
    environment=os.environ.copy();environment.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4',V28D_MEMORY_LIMIT_GIB='10',PYTHONUNBUFFERED='1')
    command=[sys.executable,'-u','-m','divibody_training.v28d_gpu_entry','divibody_training.train_spatiotemporal_v30_v2']
    for name,value in config.items():
        if value is True:command.append('--'+name)
        elif value is not False and value is not None:command.extend(['--'+name,str(value)])
    with (ROOT/'training_runs'/f'.v29_gpu{gpu}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);deadline=time.monotonic()+7200
        while True:
            free,util=map(int,subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free,utilization.gpu',
                '--format=csv,noheader,nounits'],text=True).strip().split(','))
            available=int(next(line for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')).split()[1])
            if free>=12288 and util<5 and available>=40*1024**2:break
            if time.monotonic()>deadline:raise TimeoutError('Finite GPU/host-memory resource wait exceeded')
            time.sleep(20)
        with (OUT/(label+'.log')).open('x') as log:
            child=subprocess.Popen(command,env=environment,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            atomic(OUT/(label+'_process.json'),dict(pid=child.pid,gpu=gpu,command=command))
            code=child.wait()
        if code:raise RuntimeError(label+' failed with exit '+str(code))


def main():
    OUT.mkdir(exist_ok=False)
    result_path=ROOT/'runs/v30_real_forward_v2_20261007/result.json'
    proof=json.loads(result_path.read_text())
    if proof['status']!='passed' or proof['frames']!=8 or proof['optimizer_step_performed'] is not False:
        raise ValueError('Real V30 functional proof missing')
    original=json.loads((ROOT/'original_code_hashes.json').read_text())
    if any(digest(ROOT/name)!=sha for name,sha in original.items()):raise ValueError('An original source changed')
    plan=dict(json.loads((ROOT/'training_runs/v29y_strong_rgb_20261006/plan.json').read_text())['strong_motion_40'])
    for name in ('strong-train-manifest','strong-validation-manifest','strong-input-dim','max-angle-degrees'):
        plan.pop(name,None)
    plan.update(iterations=3,**{'model-width':256,'model-heads':8,'model-depth':4,'shape-bound':3.,
        'max-steps':6000,'eval-every':1000,'warmup-steps':200,'seed':2026,'geometry-lr':1e-4,
        'local-lr':1e-4,'camera-lr':1e-5,'strict-3d-selection':True,'acceleration-ceiling-mm':22.329070147964064,
        'geometry-only-validation':True,'pose-2d-weight':0.,'max-train-windows':0,'max-val-windows':0})
    plans={}
    for name,disabled in [('spatial_motion',False),('no_new_temporal',True)]:
        plans[name]=plan|{'output-dir':str(OUT/name),'disable-temporal':disabled}
    registration=dict(plans=plans,proof_sha256=digest(result_path),initialization_sha256=digest(Path(plan['init-checkpoint'])),
        compared_to='Original V28 exact initialization; full train and validation',
        single_seed_exploratory=True,test_used_for_selection=False,
        test_previously_observed_for_older_project_models=True,
        ablation_scope='Only new V30 temporal module disabled; frozen reference retains its own temporal modeling',
        sota_status='not_established')
    with (OUT/'plan.json').open('x') as f:json.dump(registration,f,indent=2)
    status('full_cache_four_step_smoke',gpu=1)
    smoke=plan|{'max-steps':4,'eval-every':4,'warmup-steps':1,'output-dir':str(OUT/'smoke4')}
    run(smoke,1,'smoke4')
    completion=json.loads((OUT/'smoke4/completion.json').read_text())
    if completion['completed_steps']!=4 or completion['complete'] is not True:raise ValueError('Actual four-step smoke incomplete')
    status('formal_architecture_training',target_steps_per_arm=6000,arms=list(plans))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        jobs=[executor.submit(run,plans[name],gpu,name) for name,gpu in [('spatial_motion',1),('no_new_temporal',2)]]
        for job in jobs:job.result()
    reports=[]
    for name in plans:
        directory=OUT/name;completion=json.loads((directory/'completion.json').read_text())
        if not completion.get('complete') or completion['completed_steps']!=6000:raise ValueError('Incomplete architecture arm '+name)
        rows=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
        candidates=[row for row in rows if row.get('step',0)>0 and row.get('accepted')]
        reports.append(dict(arm=name,qualified=bool(candidates),selected=candidates[-1] if candidates else rows[0],
            best_checkpoint_sha256=digest(directory/'best.pt')))
    with (OUT/'comparison.json').open('x') as f:json.dump(dict(runs=reports,test_used_for_selection=False,
        single_seed_exploratory=True,sota_status='not_established'),f,indent=2)
    status('complete',comparison=str(OUT/'comparison.json'))


if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():status('failed',error=repr(error))
        raise
