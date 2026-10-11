"""Launch validation-selected v25 per-part residual experiments without changing v22."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path('/mnt/32a56f7d-8270-4d4e-b9cb-fb8857bd7e93/testpro/xmj/sam-3d-body/divibody_vace')
PROJECT=ROOT/'video_VACE'
RUNS=ROOT/'training_runs'
COLLECTION=RUNS/'ram_posetrack_sharded_v2_full_20260902'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('smoke','full'),required=True)
    p.add_argument('--run-name',required=True)
    p.add_argument('--epochs',type=int,default=15)
    p.add_argument('--gpu',type=int,choices=(0,1,2,3),default=0)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--local-lr',type=float,default=1e-5)
    p.add_argument('--max-angle-degrees',type=float,default=20.)
    p.add_argument('--init-checkpoint',type=Path,
                   default=RUNS/'v22_full_decoder_mixed_ema_seed2026_20260922/run/best.pt')
    p.add_argument('--validation-manifest',type=Path)
    p.add_argument('--allow-partial-validation',action='store_true')
    p.add_argument('--eval-only',action='store_true')
    p.add_argument('--eval-part-sweep',action='store_true')
    p.add_argument('--eval-part-angles',default='')
    args=p.parse_args()
    if Path(args.run_name).name!=args.run_name or args.run_name in ('.','..'):
        p.error('run-name must be one new directory name')
    if args.epochs<1:
        p.error('epochs must be positive')
    if args.stage=='full' and args.validation_manifest is None and not args.allow_partial_validation:
        p.error('Incomplete validation must be explicitly marked')
    paths={
      'train-vace-cache':RUNS/'continuous_vace_v5_full_zero_text/features_train.pt',
      'train-hmr-sidecar':RUNS/'hmr2a_anchor_v11_raw25/features_train.pt',
      'val-vace-cache':RUNS/'continuous_vace_v6_stride4_zero_text/features_validation.pt',
      'val-hmr-sidecar':RUNS/'hmr2a_anchor_v11_raw25/features_validation_stride4.pt',
      'pose-train-vace-dir':COLLECTION/'train','pose-train-hmr-dir':COLLECTION/'hmr_train',
      'pose-val-vace-dir':COLLECTION/'val','pose-val-hmr-dir':COLLECTION/'hmr_val',
      'init-checkpoint':args.init_checkpoint,
      'smpl-model-root':ROOT/'checkpoints/4DHumans/extracted/data/smpl',
      'joint-regressor':ROOT/'body_models/J_regressor_h36m.npy'}
    if args.validation_manifest is not None:
        paths['validation-manifest']=args.validation_manifest
    for name,path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f'{name}: {path}')
    out=RUNS/args.run_name
    out.mkdir(exist_ok=False)
    options=dict(paths)
    options.update({'output-dir':out/'run','epochs':args.epochs,'batch-size':4,'pose-batch-size':8,
      'val-batch-size':16,'num-workers':2,'seed':args.seed,'device':'cuda:0',
      'rgb-lr':1e-6,'vace-lr':1e-6,'heads-lr':1e-6,'camera-lr':3e-6,
      'gate-lr':1e-5,'local-lr':args.local_lr,'max-angle-degrees':args.max_angle_degrees,
      'pose-2d-weight':.02,'ema-decay':.995,
      'warmup-epochs':1.,'patience':6,'min-epochs':8,'diagnostic-every':100})
    if args.stage=='smoke':
        options.update({'epochs':2,'min-epochs':1,'patience':2,'max-train-windows':32,
          'max-val-windows':32,'max-pose-train-windows':64,'max-pose-val-windows':32,
          'num-workers':0,'diagnostic-every':1})
    command=[sys.executable,'-u','-m','divibody_training.train_parts_v25']
    for key,value in options.items():
        command.extend(['--'+key,str(value)])
    if args.stage=='smoke' or args.allow_partial_validation:
        command.append('--allow-partial-validation')
    if args.eval_only:
        command.append('--eval-only')
    if args.eval_part_sweep:
        command.append('--eval-part-sweep')
    if args.eval_part_angles:
        command.extend(['--eval-part-angles',args.eval_part_angles])
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    env['PYTHONPATH']=':'.join(['/mnt/32a56f7d-8270-4d4e-b9cb-fb8857bd7e93/testpro/xmj/Backup/4D-Humans',
      str(PROJECT),str(PROJECT/'divibody_training'),str(ROOT/'dependencies/Wan2.1'),str(PROJECT/'vace/models')])
    (out/'launch.json').write_text(json.dumps(dict(command=command,cwd=str(PROJECT),stage=args.stage,gpu=args.gpu,
      seed=args.seed,existing_sources_modified=False,test_used_for_model_selection=False),indent=2))
    with (out/'train.log').open('x') as log:
        process=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
        (out/'train.pid').write_text(str(process.pid))
        print(json.dumps(dict(run=str(out),pid=process.pid)),flush=True)
        code=process.wait()
    (out/'exit_code.txt').write_text(str(code))
    print(json.dumps(dict(exit_code=code,run=str(out))),flush=True)
    raise SystemExit(code)

if __name__=='__main__':
    main()
