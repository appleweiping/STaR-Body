"""After finite calibration, freeze its winner and run independent official test."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
SUITE=ROOT/'training_runs/v29_geometry_20261006'
OUT=ROOT/'runs/v29_frozen_test_stride8_20261006'
SOURCE=ROOT.parent/'divibody_vace'


def status(state,**values):
    tmp=OUT/'status.tmp';tmp.write_text(json.dumps(dict(status=state,sota_status='not_established',**values),indent=2))
    tmp.replace(OUT/'status.json')


def main():
    OUT.mkdir(exist_ok=False)
    while True:
        state=json.loads((SUITE/'status.json').read_text())
        if state['status']=='failed':raise RuntimeError('Training suite failed: '+json.dumps(state))
        if state['status']=='complete':break
        status('waiting_for_training_completion');time.sleep(60)
    # Reuse the winner frozen before the first test attempt; do not select again.
    freeze=json.loads((ROOT/'runs/v29_frozen_test_20261006/selection.json').read_text())
    checkpoint=Path(freeze['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=freeze['checkpoint_sha256']:
        raise ValueError('Previously frozen checkpoint changed')
    if freeze['test_used_for_model_selection'] is not False:raise ValueError('Invalid freeze provenance')
    (OUT/'selection.json').write_text(json.dumps(freeze,indent=2))
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',V28D_MEMORY_LIMIT_GIB='10')
    gpu_lock=(ROOT/'training_runs/.v29_gpu1.lock').open('a')
    with gpu_lock:
        fcntl.flock(gpu_lock,fcntl.LOCK_EX)
        while True:
            free,util=map(int,subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.free,utilization.gpu',
                                                      '--format=csv,noheader,nounits'],text=True).strip().split(','))
            if free>=12288 and util<5:break
            status('waiting_for_free_gpu',gpu=1,free_mib=free);time.sleep(30)
        config=json.loads((ROOT/'configs/geometry.json').read_text())
        for label,ckpt in [('selected',str(checkpoint)),('baseline',config['init-checkpoint'])]:
            status('running_frozen_official_test',model=label)
            command=[sys.executable,'-u','-m','divibody_training.v29_gpu_entry',
                     str(ROOT/'scripts/v29/evaluate_v29_frozen_test_stride8.py'),
                     '--checkpoint',ckpt,'--config',str(ROOT/'configs/geometry.json'),
                     '--vace-cache',str(SOURCE/'training_runs/continuous_vace_v5_formal_test_zero_text/features_test.pt'),
                     '--hmr-source-vace-cache',str(SOURCE/'training_runs/continuous_vace_v6_stride4_zero_text/features_test.pt'),
                     '--hmr-cache',str(SOURCE/'training_runs/hmr2a_anchor_v11_raw25/features_test.pt'),
                     '--annotation-archive',str(ROOT.parent/'v28d_research_20261005/data/sequenceFiles.zip'),
                     '--image-root',str(SOURCE/'datasets_full/3DPW/imageFiles'),
                     '--output',str(OUT/label)]
            with (OUT/(label+'.log')).open('x') as log:
                subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    selected=json.loads((OUT/'selected/result.json').read_text());baseline=json.loads((OUT/'baseline/result.json').read_text())
    delta={k:selected['metrics'][k]-baseline['metrics'][k] for k in ('mpjpe_mm','pa_mpjpe_mm','pve_mm')}
    (OUT/'comparison.json').write_text(json.dumps(dict(selected=selected,baseline=baseline,delta=delta,
                test_used_for_selection=False,sota_status='not_established'),indent=2))
    status('complete',baseline_deltas=delta,selection=str(OUT/'selection.json'),
           next_stage='reproduce named SOTA baselines under matched input/representation/metric protocol')


if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():status('failed',error=repr(error))
        raise
