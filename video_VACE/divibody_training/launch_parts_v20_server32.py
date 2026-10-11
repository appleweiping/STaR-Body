"""Launch isolated v20 experiments; incomplete validation remains development only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/mnt/32a56f7d-8270-4d4e-b9cb-fb8857bd7e93/testpro/xmj/sam-3d-body/divibody_vace')
PROJECT = ROOT / 'video_VACE'
RUNS = ROOT / 'training_runs'
COLLECTION = RUNS / 'ram_posetrack_sharded_v2_full_20260902'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['smoke', 'fit', 'full'], required=True)
    parser.add_argument('--fit-windows', type=int, default=64)
    parser.add_argument('--geometry-epochs', type=int, default=15)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--val-vace-cache', type=Path)
    parser.add_argument('--val-hmr-sidecar', type=Path)
    parser.add_argument('--position-mode', choices=['original', 'shared'], default='original')
    parser.add_argument('--motion-loss', choices=['on', 'off'], default='on')
    parser.add_argument('--geometry-lr', type=float, default=5e-7)
    parser.add_argument('--head-lr', type=float, default=5e-6)
    parser.add_argument('--gate-lr', type=float, default=1e-5)
    parser.add_argument('--fusion-mode', choices=('frozen','block','block_gate'), default='block_gate')
    parser.add_argument('--detach-context', action='store_true')
    parser.add_argument('--local-mode', choices=('image','bias'), default='image')
    parser.add_argument('--diagnostic-every', type=int, default=25)
    parser.add_argument('--validation-manifest', type=Path)
    parser.add_argument('--local-init', type=Path)
    parser.add_argument('--init-checkpoint', type=Path)
    parser.add_argument('--fusion-lr', type=float, default=1e-7)
    parser.add_argument('--fusion-warmup-epochs', type=int, default=1)
    parser.add_argument('--teacher-weight', type=float, default=0.05)
    parser.add_argument('--position-consistency-weight', type=float, default=0.)
    parser.add_argument('--pose-2d-weight',type=float,default=0.)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--gpu', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--allow-partial-validation', action='store_true')
    args = parser.parse_args()
    if args.fit_windows <= 0 or args.geometry_epochs <= 0:
        parser.error('fit-windows and geometry-epochs must be positive')
    if Path(args.run_name).name != args.run_name or args.run_name in ('.', '..'):
        parser.error('run-name must be a single new directory name')
    if args.stage == 'full' and not args.allow_partial_validation and args.validation_manifest is None:
        parser.error('Formal full runs require --validation-manifest; partial runs must be explicitly marked')
    out = RUNS / args.run_name
    paths = {
        'train-vace-cache': RUNS / 'continuous_vace_v5_full_zero_text/features_train.pt',
        'train-hmr-sidecar': RUNS / 'hmr2a_anchor_v11_raw25/features_train.pt',
        'val-vace-cache': RUNS / 'continuous_vace_v6_stride4_zero_text/features_validation.pt',
        'val-hmr-sidecar': RUNS / 'hmr2a_anchor_v11_raw25/features_validation_stride4.pt',
        'pose-train-vace-dir': COLLECTION / 'train',
        'pose-train-hmr-dir': COLLECTION / 'hmr_train',
        'pose-val-vace-dir': COLLECTION / 'val',
        'pose-val-hmr-dir': COLLECTION / 'hmr_val',
        'init-checkpoint': RUNS / 'ram_v9_geometry_priority_full_20260907_retry1/run/best.pt',
        'smpl-model-root': ROOT / 'checkpoints/4DHumans/extracted/data/smpl',
        'joint-regressor': ROOT / 'body_models/J_regressor_h36m.npy',
    }
    if args.val_vace_cache is not None:
        paths['val-vace-cache'] = args.val_vace_cache
    if args.val_hmr_sidecar is not None:
        paths['val-hmr-sidecar'] = args.val_hmr_sidecar
    if args.init_checkpoint is not None:
        paths['init-checkpoint'] = args.init_checkpoint
    if args.validation_manifest is not None:
        paths['validation-manifest'] = args.validation_manifest
    if args.local_init is not None:
        paths['local-init'] = args.local_init
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f'{name}: {path}')
    out.mkdir(exist_ok=False)
    options = dict(paths)
    options.update({'output-dir': out / 'run', 'geometry-epochs': args.geometry_epochs,
                    'batch-size': 4, 'val-batch-size': 16, 'num-workers': 2,
                    'geometry-lr': args.geometry_lr, 'teacher-weight': args.teacher_weight,
                    'head-lr': args.head_lr, 'gate-lr': args.gate_lr,
                    'fusion-mode': args.fusion_mode, 'local-mode': args.local_mode,
                    'diagnostic-every': args.diagnostic_every,
                    'fusion-lr': args.fusion_lr,
                    'fusion-warmup-epochs': args.fusion_warmup_epochs,
                    'pose-2d-weight': args.pose_2d_weight,
                    'position-mode': args.position_mode, 'motion-loss': args.motion_loss,
                    'position-consistency-weight': args.position_consistency_weight,
                    'position-warmup-epochs': 2, 'patience': 5, 'seed': args.seed,
                    'device': 'cuda:0'})
    if args.stage == 'smoke':
        options.update({'geometry-epochs': 2, 'max-3dpw-windows': 32,
                        'max-val-windows': 32, 'max-pose-val-windows': 32,
                        'max-pose-train-windows': 32, 'num-workers': 0})
    elif args.stage == 'fit':
        options.update({'max-3dpw-windows': args.fit_windows, 'max-val-windows': 32,
                        'max-pose-val-windows': 32, 'max-pose-train-windows': args.fit_windows,
                        'num-workers': 0, 'diagnostic-every': 1})
    command = [sys.executable, '-u', '-m',
               'divibody_training.train_parts_v20']
    for key, value in options.items():
        command.extend(['--' + key, str(value)])
    if args.stage in ('smoke', 'fit'):
        command.append('--allow-partial-validation')
    elif args.allow_partial_validation:
        command.append('--allow-partial-validation')
    if args.detach_context:
        command.append('--detach-context')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               PYTHONUNBUFFERED='1')
    env['PYTHONPATH'] = ':'.join([
        '/mnt/32a56f7d-8270-4d4e-b9cb-fb8857bd7e93/testpro/xmj/Backup/4D-Humans',
        str(PROJECT), str(PROJECT / 'divibody_training'),
        str(ROOT / 'dependencies/Wan2.1'), str(PROJECT / 'vace/models')])
    (out / 'launch.json').write_text(json.dumps({
        'command': command, 'cwd': str(PROJECT), 'stage': args.stage,
        'gpu': args.gpu, 'seed': args.seed, 'existing_sources_modified': False,
        'test_used_for_model_selection': False,
    }, indent=2))
    with (out / 'train.log').open('x') as log:
        process = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT)
        (out / 'train.pid').write_text(str(process.pid))
        print(json.dumps({'run': str(out), 'pid': process.pid}), flush=True)
        code = process.wait()
    (out / 'exit_code.txt').write_text(str(code))
    print(json.dumps({'exit_code': code, 'run': str(out)}), flush=True)
    raise SystemExit(code)

if __name__ == '__main__':
    main()
