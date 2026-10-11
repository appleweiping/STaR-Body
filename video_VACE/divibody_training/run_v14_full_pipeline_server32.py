"""Server32 full v14 pipeline: parallel phase caches, then one full-data training run."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path('/mnt/32a56f7d-8270-4d4e-b9cb-fb8857bd7e93/testpro/xmj/sam-3d-body/divibody_vace')
PROJECT=ROOT/'video_VACE'
RUNS=ROOT/'training_runs'
CACHE_DIR=RUNS/'v14_phase_cache_full_legacy_20260908'
TRAIN_DIR=RUNS/'v14_phase_full_train_legacy_20260908'
SOURCE_VACE=RUNS/'continuous_vace_v5_full_zero_text/features_train.pt'
SOURCE_HMR=RUNS/'hmr2a_anchor_v11_raw25/features_train.pt'
VAL_VACE=RUNS/'continuous_vace_v6_stride4_zero_text/features_validation.pt'
VAL_HMR=RUNS/'hmr2a_anchor_v11_raw25/features_validation_stride4.pt'
POSE=RUNS/'ram_posetrack_sharded_v2_full_20260902'
V9=RUNS/'ram_v9_geometry_priority_full_20260907_retry1/run/best.pt'

def run_logged(cmd,log,env):
    with log.open('x') as handle:
        result=subprocess.run(cmd,cwd=PROJECT,env=env,stdout=handle,stderr=subprocess.STDOUT)
    log.with_suffix('.exit').write_text(str(result.returncode))
    return result.returncode

def cache_command(phase):
    return [sys.executable,'-u','-m','divibody_training.cache_vace_phase_v14',
        '--data-root',str(ROOT/'datasets_full/3DPW'),'--split','train',
        '--vace-checkpoint-dir',str(PROJECT/'models/Wan2.1-VACE-1.3B'),
        '--output',str(CACHE_DIR/f'phase_{phase}.pt'),'--window-phase',str(phase),
        '--crop-mode','legacy','--num-workers','2','--device','cuda:0',
        '--log-interval','100','--vace-timestep','500',
        '--diffusion-input-protocol','clean','--feature-layers','7','14','22']

def main():
    for p in (SOURCE_VACE,SOURCE_HMR,VAL_VACE,VAL_HMR,V9,POSE/'val',POSE/'hmr_val'):
        if not p.exists(): raise FileNotFoundError(p)
    CACHE_DIR.mkdir(parents=True,exist_ok=False)
    if TRAIN_DIR.exists(): raise FileExistsError(TRAIN_DIR)
    base=os.environ.copy()
    jobs=[]
    # Keep GPU3 untouched; phase 6 follows phase 0 on GPU0.
    for phase,gpu in ((0,'0'),(2,'1'),(4,'2')):
        cmd=cache_command(phase)
        (CACHE_DIR/f'phase_{phase}_command.json').write_text(json.dumps(cmd,indent=2))
        env=dict(base,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        log=(CACHE_DIR/f'phase_{phase}.log').open('x')
        process=subprocess.Popen(cmd,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((phase,gpu,process,log))
    failed=[]
    for phase,gpu,process,log in jobs:
        code=process.wait();log.close()
        (CACHE_DIR/f'phase_{phase}.exit').write_text(str(code))
        print(json.dumps({'phase':phase,'exit_code':code}),flush=True)
        if code: failed.append(phase)
        if phase==0 and code==0:
            cmd=cache_command(6)
            (CACHE_DIR/'phase_6_command.json').write_text(json.dumps(cmd,indent=2))
            env=dict(base,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
            code6=run_logged(cmd,CACHE_DIR/'phase_6.log',env)
            print(json.dumps({'phase':6,'exit_code':code6}),flush=True)
            if code6: failed.append(6)
    if failed: raise RuntimeError(f'Cache phases failed: {failed}')
    manifest={'format':'v9-training-phases-v14',
        'source_vace':str(SOURCE_VACE.resolve()),'source_hmr':str(SOURCE_HMR.resolve()),
        'phases':[{'phase':p,'vace':f'phase_{p}.pt'} for p in (0,2,4,6)]}
    (CACHE_DIR/'phases.json').write_text(json.dumps(manifest,indent=2))
    (CACHE_DIR/'phase0_control.json').write_text(json.dumps(
        dict(manifest,phases=manifest['phases'][:1]),indent=2))
    command=[sys.executable,'-u','-m','divibody_training.train_v9_phase_v14',
        '--train-vace-cache',str(SOURCE_VACE),'--train-hmr-sidecar',str(SOURCE_HMR),
        '--val-vace-cache',str(VAL_VACE),'--val-hmr-sidecar',str(VAL_HMR),
        '--pose-val-vace-dir',str(POSE/'val'),'--pose-val-hmr-dir',str(POSE/'hmr_val'),
        '--init-checkpoint',str(V9),'--smpl-model-root',str(ROOT/'checkpoints/4DHumans/extracted/data/smpl'),
        '--joint-regressor',str(ROOT/'body_models/J_regressor_h36m.npy'),
        '--output-dir',str(TRAIN_DIR),'--geometry-epochs','15','--batch-size','4',
        '--val-batch-size','16','--num-workers','2','--geometry-lr','1e-7',
        '--teacher-weight','0.05','--position-mode','original','--motion-loss','on',
        '--position-consistency-weight','0','--position-warmup-epochs','2',
        '--patience','5','--device','cuda:0','--phase-manifest',str(CACHE_DIR/'phases.json'),
        '--allow-partial-validation']
    (RUNS/'v14_full_pipeline_train_command_20260908.json').write_text(json.dumps(command,indent=2))
    env=dict(base,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    code=run_logged(command,RUNS/'v14_full_pipeline_train_20260908.log',env)
    print(json.dumps({'training_exit_code':code,'output':str(TRAIN_DIR)}),flush=True)
    if code: raise SystemExit(code)

if __name__=='__main__':main()
