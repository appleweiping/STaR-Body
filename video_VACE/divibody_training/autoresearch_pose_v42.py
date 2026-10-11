"""v42 bounded SMPL pose-refinement autoresearch driver.

Copied from the v41 driver. It enables v41's zero-initialized, 2-degree joint
rotation branch from the best v41 spatial-evidence checkpoint. 3DPW geometry
gradients remain protected and all selection uses direct SMPL projections.
Isolated runs, two-observation idle-GPU queue, two-seed screening and
conditional full confirmation. No test feedback, process eviction, automatic
crash retries, or legacy experiment overwrites.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path('/mnt/32a56f7d-8270-4d4e-b9cb-fb8857bd7e93/testpro/xmj/sam-3d-body/divibody_vace')
MODULE = 'divibody_training.train_parts_v41_joint_evidence'
GEOMETRY = ('mpjpe_mm', 'pa_mpjpe_mm', 'pve_mm')
PCK = ('projected_PCKn@0.05', 'projected_PCKn@0.10')
SOURCES = ('models/joint_evidence_v41.py', 'models/human_decoder_parts_v41.py',
           'models/body_projection_v39.py', 'divibody_training/ram_parts_runtime_v41.py',
           'divibody_training/ram_pose_stream_v41.py', 'divibody_training/parts_training_utils_v41.py',
           'divibody_training/train_parts_v41_joint_evidence.py',
           'divibody_training/evaluate_hmr2a_official_posetrack_v41.py',
           'divibody_training/autoresearch_joint_evidence_v41.py')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def append_record(root, value):
    row = dict(timestamp=stamp(), **value)
    with (root/'results.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, allow_nan=False)+'\n')
    print(json.dumps(row, allow_nan=False), flush=True)


def status(root, phase, **fields):
    row = dict(version='v42_pose_refinement_autoresearch', phase=phase,
               driver_pid=os.getpid(), updated_at=stamp(), session=str(root),
               test_used_for_model_selection=False, ram_superiority_verified=False, **fields)
    temporary = root/('status.'+uuid.uuid4().hex+'.tmp')
    write_new(temporary, row)
    temporary.replace(root/'status.json')


def gpu_snapshot():
    def query(options):
        out = subprocess.run(['nvidia-smi', *options, '--format=csv,noheader,nounits'],
                             check=True, text=True, capture_output=True, timeout=20)
        return list(csv.reader(line for line in out.stdout.splitlines() if line.strip()))
    cards = {}
    for row in query(['--query-gpu=index,uuid,utilization.gpu,memory.used']):
        if len(row) != 4:
            raise RuntimeError('Unparseable nvidia-smi GPU observation')
        index, identity, utilization, memory = (v.strip() for v in row)
        cards[int(index)] = dict(index=int(index), uuid=identity,
                                  utilization=float(utilization), memory_mib=float(memory), pids=[])
    by_uuid = {card['uuid']: card for card in cards.values()}
    for row in query(['--query-compute-apps=gpu_uuid,pid']):
        if len(row) == 1 and row[0].strip() == 'No running processes found':
            continue
        if len(row) != 2 or row[0].strip() not in by_uuid:
            raise RuntimeError('Cannot attribute a compute process to a physical GPU')
        by_uuid[row[0].strip()]['pids'].append(int(row[1].strip()))
    if not cards:
        raise RuntimeError('nvidia-smi returned no GPU observations')
    return cards


def idle(card):
    return not card['pids'] and card['utilization'] < 10 and card['memory_mib'] < 1000


def wait_gpu(root, allowed):
    import fcntl
    while True:
        first = gpu_snapshot()
        available = [i for i in allowed if i in first and idle(first[i])]
        status(root, 'queued_for_gpu', reason='Require two idle observations 30 seconds apart',
               first_observation=first, available_indices=available)
        time.sleep(30)
        second = gpu_snapshot()
        for i in available:
            if i not in second or second[i]['uuid'] != first[i]['uuid'] or not idle(second[i]):
                continue
            lease = (root.parent/f'.autoresearch_v42_gpu_{i}.lock').open('a+')
            try:
                fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lease.close()
                continue
            latest = gpu_snapshot()
            if i in latest and latest[i]['uuid'] == second[i]['uuid'] and idle(latest[i]):
                return latest[i], lease
            lease.close()


def qualifies(row, baseline):
    try:
        g, p = row['validation'], row['pose_validation']
        bg, bp = baseline['validation'], baseline['pose_validation']
        if not all(math.isfinite(float(v)) for v in (*g.values(), *p.values())):
            return False, 'nonfinite_metrics'
        if g.get('frames') != bg.get('frames') or p.get('visible_keypoints') != bp.get('visible_keypoints'):
            return False, 'development_coverage_changed'
        if any(g[k] > bg[k]+1e-5 for k in GEOMETRY):
            return False, 'geometry_regressed'
        if any(p[k] < bp[k] for k in PCK):
            return False, 'direct_pck_regressed'
        if not any(p[k] > bp[k] for k in PCK):
            return False, 'no_strict_direct_pck_gain'
        return True, 'three_geometry_metrics_preserved_and_direct_pck_improved'
    except (KeyError, TypeError, ValueError):
        return False, 'invalid_metrics_schema'


def same_baseline(row, fixed):
    g, p = row['validation'], row['pose_validation']
    bg, bp = fixed['validation'], fixed['pose_validation']
    return (g.get('frames') == bg.get('frames') and p.get('visible_keypoints') == bp.get('visible_keypoints')
            and all(abs(g[k]-bg[k]) <= 1e-5 for k in GEOMETRY)
            and all(abs(p[k]-bp[k]) <= 1e-12 for k in PCK))

def preflight_commands(session,project,config_path,allowed):
    """Optional audited reference jobs share the same idle-GPU lease."""
    spec=json.loads(config_path.read_text())
    write_new(session/'preflight_spec.json',spec)
    for job in spec['jobs']:
        name=job['name']
        if Path(name).name!=name: raise ValueError('Invalid preflight job name')
        out=session/('preflight_'+name); out.mkdir(exist_ok=False)
        command=[str(v).replace('{session}',str(session)) for v in job['command']]
        card,lease=wait_gpu(session,allowed)
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=card['uuid'],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
        env['PYTHONPATH']=':'.join([str(project),str(project/'divibody_training'),
            str(project.parent.parent.parent/'Backup/4D-Humans'),str(project.parent/'dependencies/Wan2.1'),str(project/'vace/models')])
        child=None
        try:
            write_new(out/'launch.json',dict(command=command,gpu=card,cwd=str(project)))
            with (out/'run.log').open('x') as log:
                child=subprocess.Popen(command,cwd=project,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                (out/'pid').write_text(str(child.pid))
                while child.poll() is None:
                    status(session,'reference_audit_running',job=name,child_pid=child.pid,gpu=card)
                    time.sleep(30)
            if child.returncode: raise RuntimeError(f'Reference audit {name} exited {child.returncode}')
            result_path=Path(job['result'].replace('{session}',str(session)))
            result=json.loads(result_path.read_text())
            if result['status']!=job['expected_status']:
                raise RuntimeError(f'Reference audit {name} did not meet success/coverage guard')
            append_record(session,dict(disposition='reference_audit_completed',job=name,result=str(result_path),sha256=digest(result_path)))
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL); child.wait()
            lease.close()


def read_checkpoint_metadata(path):
    import torch  # CPU metadata read only; never initializes a CUDA device.
    payload = torch.load(path, map_location='cpu', weights_only=False)
    return {k: payload[k] for k in ('epoch', 'variant', 'validation', 'pose_validation')}


def make_paths(root, checkpoint):
    runs = root/'training_runs'
    collection = runs/'ram_posetrack_sharded_v2_full_20260902'
    return {'train-vace-cache': runs/'continuous_vace_v5_full_zero_text/features_train.pt',
            'train-hmr-sidecar': runs/'hmr2a_anchor_v11_raw25/features_train.pt',
            'val-vace-cache': runs/'continuous_vace_v6_stride4_zero_text/features_validation.pt',
            'val-hmr-sidecar': runs/'hmr2a_anchor_v11_raw25/features_validation_stride4.pt',
            'pose-train-vace-dir': collection/'train', 'pose-train-hmr-dir': collection/'hmr_train',
            'pose-val-vace-dir': collection/'val', 'pose-val-hmr-dir': collection/'hmr_val',
            'smpl-model-root': root/'checkpoints/4DHumans/extracted/data/smpl',
            'joint-regressor': root/'body_models/J_regressor_h36m.npy', 'init-checkpoint': checkpoint}


def trial(root, project, paths, hashes, fixed, seed, stage, allowed):
    name = f'{stage}_spatial_seed{seed}_{uuid.uuid4().hex[:8]}'
    out = root/name
    out.mkdir(exist_ok=False)
    full = stage == 'pose_full_confirmation'
    epochs = 4 if full else 2
    options = dict(paths)
    options.update({'output-dir': out/'run', 'epochs': epochs, 'min-epochs': epochs,
                    'patience': epochs, 'batch-size': 4, 'pose-batch-size': 8,
                    'val-batch-size': 16, 'num-workers': 2, 'seed': seed, 'device': 'cuda:0',
                    'max-train-windows': 2048 if full else 1024,
                    'max-pose-train-windows': 0 if full else 4096,
                    'max-val-windows': 0, 'max-pose-val-windows': 0,
                    'max-angle-degrees': 20, 'iterations': 3, 'v41-stage': 'pose',
                    'pose-gradient-route': 'shared', 'geometry-gradient-scale': 1,
                    'pose-2d-weight': 1., 'camera-lr': 3e-6, 'local-lr': 1e-5,
                    'geometry-lr': 2e-5, 'ema-decay': .995, 'warmup-epochs': 1.,
                    'diagnostic-every': 100, 'amp-dtype': 'bf16'})
    command = [sys.executable, '-u', '-m', MODULE]
    for key, value in options.items():
        command.extend(['--'+key, str(value)])
    command += ['--allow-partial-validation', '--protect-geometry-gradient',
                '--v41-enable-pose-migration']
    card, lease = wait_gpu(root, allowed)
    child = None
    try:
        if any(digest(project/rel) != sha for rel, sha in hashes.items()):
            raise RuntimeError('Source changed after provenance snapshot; create a new session')
        expected_init = json.loads((root/'source_manifest.json').read_text())['initialization_sha256']
        if digest(paths['init-checkpoint']) != expected_init:
            raise RuntimeError('Initialization checkpoint changed after provenance snapshot')
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=card['uuid'], OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')
        env['PYTHONPATH'] = ':'.join([str(project.parent.parent.parent/'Backup/4D-Humans'), str(project),
                                    str(project/'divibody_training'), str(project.parent/'dependencies/Wan2.1'),
                                    str(project/'vace/models')])
        write_new(out/'launch.json', dict(command=command, cwd=str(project), gpu=card,
                  stage=stage, seed=seed, source_sha256=hashes,
                  test_used_for_model_selection=False, selection_split='3DPW validation + PoseTrack val development',
                  coverage_protocol='All existing development caches, not an independent PoseTrack test'))
        with (out/'train.log').open('x', encoding='utf-8') as log:
            child = subprocess.Popen(command, cwd=project, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            (out/'train.pid').write_text(str(child.pid)+'\n', encoding='utf-8')
            while child.poll() is None:
                progress = {}
                try:
                    progress = json.loads((out/'run/progress.json').read_text())
                except (FileNotFoundError, json.JSONDecodeError):
                    pass
                status(root, 'running', trial=str(out), stage=stage,
                       seed=seed, child_pid=child.pid, gpu=card, progress=progress)
                time.sleep(30)
        code = child.returncode
        (out/'exit_code.txt').write_text(str(code)+'\n', encoding='utf-8')
        if code:
            append_record(root, dict(name=name, disposition='crash', exit_code=code,
                                    reason='training_process_failed', trial=str(out)))
            raise RuntimeError(f'Trial {name} exited {code}; no retries or further trials started')
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)  # Only our newly created process group.
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        lease.close()
    rows = [json.loads(line) for line in (out/'run/metrics.jsonl').read_text().splitlines() if line.strip()]
    baseline_rows = [row for row in rows if row.get('epoch') == 0 and row.get('variant') == 'initial']
    if len(baseline_rows) != 1:
        raise RuntimeError('Expected exactly one epoch-0 development baseline')
    baseline = baseline_rows[0]
    if fixed is None:
        fixed = baseline
        write_new(root/'fixed_epoch0_baseline.json', fixed)
    elif not same_baseline(baseline, fixed):
        raise RuntimeError('Epoch-0 baseline changed; candidate comparison is invalid')
    completion = json.loads((out/'run/completion.json').read_text())
    if completion['completed_epochs'] != epochs:
        raise RuntimeError('Trial did not complete the prescribed epoch count')
    config = json.loads((out/'run/run_config.json').read_text())
    if config['args']['max_val_windows'] or config['args']['max_pose_val_windows']:
        raise RuntimeError('A trial restricted development coverage unexpectedly')
    artifacts = []
    for filename in ('best.pt', 'latest.pt'):
        path = out/'run'/filename
        if path.is_file():
            metadata = read_checkpoint_metadata(path)
            passed, reason = qualifies(metadata, fixed)
            artifacts.append(dict(checkpoint=str(path), metadata=metadata, eligible=passed, reason=reason))
    eligible = [a for a in artifacts if a['eligible']]
    coverage = completion['pose_coverage']
    complete_pool = coverage['unique_windows'] == coverage['pool_windows']
    chosen = max(eligible, key=lambda a: sum(a['metadata']['pose_validation'][k] for k in PCK)) if eligible else None
    kept = bool(chosen) and complete_pool
    result = dict(name=name, trial=str(out), seed=seed, temporal=False, stage=stage,
                  disposition='keep' if kept else 'discard',
                  reason=('strict_improvement_on_complete_training_pool' if kept else
                          'incomplete_training_pool' if not complete_pool else 'no_persisted_strict_pareto_improvement'),
                  artifacts=artifacts, pose_coverage=coverage, completed_epochs=epochs,
                  coverage=config['coverage'], development_only=config['development_only'],
                  improved_metric_rows=sum(qualifies(row, fixed)[0] for row in rows if row['epoch'] > 0),
                  selected_checkpoint=chosen['checkpoint'] if kept else None)
    if kept:
        result['selected_checkpoint_sha256'] = digest(chosen['checkpoint'])
    write_new(out/'disposition.json', result)
    append_record(root, result)
    status(root, 'trial_completed', latest=result)
    return result, fixed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--project', type=Path)
    parser.add_argument('--run-name')
    parser.add_argument('--init-checkpoint', type=Path)
    parser.add_argument('--preflight-commands',type=Path)
    parser.add_argument('--gpus', default='0,1,2,3', help='Priority order; busy devices are never used')
    args = parser.parse_args()
    project = args.project or args.root/'video_VACE'
    checkpoint = args.init_checkpoint or args.root/'training_runs/autoresearch_v41_20260930_1054/full_confirmation_spatial_seed2026_6fa7a5af/run/best.pt'
    name = args.run_name or 'autoresearch_v42_pose_refinement_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8]
    if Path(name).name != name or name in ('.', '..'):
        parser.error('run-name must be one new directory name')
    allowed = [int(v) for v in args.gpus.split(',')]
    if not allowed or len(set(allowed)) != len(allowed) or min(allowed) < 0:
        parser.error('gpus must contain unique nonnegative physical GPU indices')
    import fcntl
    runs = args.root/'training_runs'
    lock = (runs/'.autoresearch_v42_pose_driver.lock').open('a+')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('An autoresearch v41 driver already owns the lock; nothing started')
    session = runs/name
    session.mkdir(exist_ok=False)
    (session/'driver.pid').write_text(str(os.getpid())+'\n', encoding='utf-8')
    def interrupted(signum, _frame):
        raise RuntimeError(f'Driver interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    status(session, 'preflight', allowed_gpu_indices=allowed)
    try:
        paths = make_paths(args.root, checkpoint)
        for key, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f'{key}: {path}')
        hashes = {rel: digest(project/rel) for rel in SOURCES}
        hashes['divibody_training/autoresearch_pose_v42.py'] = digest(project/'divibody_training/autoresearch_pose_v42.py')
        write_new(session/'source_manifest.json', dict(source_sha256=hashes,
                  copied_from='autoresearch_joint_evidence_v41.py; v41 pose branch enabled', legacy_driver_sha256=digest(project/'divibody_training/autoresearch_body_v29.py'),
                  initialization=str(checkpoint), initialization_sha256=digest(checkpoint),
                  full_confirmation_requires_two_seeds=True, seed_schedule=[2026, 2027],
                  geometry_tolerance_mm=1e-5, test_used_for_model_selection=False))
        if args.preflight_commands is not None:
            preflight_commands(session,project,args.preflight_commands,allowed)
        fixed, results = None, []
        screen, fixed = trial(session, project, paths, hashes, fixed, 2026, 'pose_screen', allowed)
        results.append(screen)
        verified = False
        if screen['disposition'] == 'keep':
            confirm, fixed = trial(session, project, paths, hashes, fixed, 2027, 'pose_seed_confirmation', allowed)
            results.append(confirm)
            verified = confirm['disposition'] == 'keep'
        if verified:
            confirmation, fixed = trial(session, project, paths, hashes, fixed, 2026, 'pose_full_confirmation', allowed)
            results.append(confirmation)
        write_new(session/'completion.json', dict(completed_at=stamp(), results=results,
                  full_confirmation_started=bool(verified), ram_superiority_verified=False,
                  reason='Two-seed gate completed' if verified else 'Pose-migration candidate failed a development gate'))
        status(session, 'completed', results=results, full_confirmation_started=bool(verified))
    except Exception as error:
        append_record(session, dict(disposition='crash', reason=str(error), exception_type=type(error).__name__))
        status(session, 'failed', reason=str(error), automatic_retry=False)
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
