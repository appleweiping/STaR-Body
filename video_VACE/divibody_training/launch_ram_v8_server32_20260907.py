"""Run v8 on server32 using existing caches and separate output directories."""
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
    parser.add_argument('--stage', choices=['smoke', 'full'], required=True)
    parser.add_argument('--run-name', required=True)
    args = parser.parse_args()
    out = RUNS / args.run_name
    out.mkdir(exist_ok=False)
    paths = {
        'train-vace-cache': RUNS / 'continuous_vace_v5_full_zero_text/features_train.pt',
        'train-hmr-sidecar': RUNS / 'hmr2a_anchor_v11_raw25/features_train.pt',
        'val-vace-cache': RUNS / 'continuous_vace_v6_stride4_zero_text/features_validation.pt',
        'val-hmr-sidecar': RUNS / 'hmr2a_anchor_v11_raw25/features_validation_stride4.pt',
        'pose-train-vace-dir': COLLECTION / 'train',
        'pose-train-hmr-dir': COLLECTION / 'hmr_train',
        'pose-val-vace-dir': COLLECTION / 'val',
        'pose-val-hmr-dir': COLLECTION / 'hmr_val',
        'init-checkpoint': RUNS / 'ram_posetrack_sharded_camera_v6_full_20260903/run/best.pt',
        'smpl-model-root': ROOT / 'checkpoints/4DHumans/extracted/data/smpl',
        'joint-regressor': ROOT / 'body_models/J_regressor_h36m.npy',
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f'{name}: {path}')
    options = dict(paths)
    options.update({'output-dir': out / 'run', 'camera-epochs': 2,
                    'geometry-epochs': 2, 'batch-size': 4, 'val-batch-size': 16,
                    'num-workers': 2, 'camera-lr': 2e-6, 'geometry-lr': 2e-7,
                    'geometry-regression-tolerance-mm': 0.5, 'device': 'cuda:0'})
    if args.stage == 'smoke':
        options.update({'camera-epochs': 1, 'geometry-epochs': 1,
                        'max-3dpw-windows': 32, 'max-val-windows': 192,
                        'max-pose-train-windows': 256, 'max-pose-val-windows': 256})
    command = [sys.executable, '-u', '-m',
               'divibody_training.train_ram_two_stage_camera_geometry_v8']
    for key, value in options.items():
        command.extend(['--' + key, str(value)])
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               PYTHONUNBUFFERED='1')
    env['PYTHONPATH'] = ':'.join([
        '/mnt/32a56f7d-8270-4d4e-b9cb-fb8857bd7e93/testpro/xmj/Backup/4D-Humans',
        str(PROJECT), str(PROJECT / 'divibody_training'),
        str(ROOT / 'dependencies/Wan2.1'), str(PROJECT / 'vace/models')])
    (out / 'launch.json').write_text(json.dumps({
        'command': command, 'cwd': str(PROJECT), 'stage': args.stage,
        'gpu': 0, 'existing_sources_modified': False,
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
