"""Executed on the authorized server after real pilot evidence is available."""
import hashlib
import json
from pathlib import Path

root = Path('/media/lenovo/data2/mjx/STaR-Body_V31_20261008')
base = Path('/media/lenovo/data2/v28d_parallel_20261002')
external = Path('/media/lenovo/data2/shenjihe/mjx/divibody_vace')
hmr = external/'dependencies/4D-Humans'
cache = base/'data/star_body_v31_cache_20261008'
out = root/'runs/full_rebuild_pipeline_001'
configs = root/'configs/rebuilt_phase2_20261008.local'
runs = root/'training_runs/rebuilt_phase2_20261008'
baseline = out/'v28_full_validation_bf16.json'
smpl = root/'checkpoints/local_smpl/smpl'
regressor = base/'body_models/J_regressor_h36m.npy'
initial = base/'checkpoints/trained_v28_20261001.pt'
manifest = base/'data/official_reference_v2/validation_reference.json'
out.mkdir(parents=True, exist_ok=True)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            value.update(block)
    return value.hexdigest()


def command(script, *args):
    return ['bash', 'scripts/run_star_body_v31.sh', script, *map(str, args)]


stages = []


def add(name, cmd, outputs, check, *, marker=None, gpu=14336, host=16):
    stage = dict(name=name, command=cmd, outputs=list(map(str, outputs)),
                 min_gpu_free_mib=gpu, min_host_available_gib=host, min_disk_free_gib=40,
                 completion_check=dict(command=check, timeout_seconds=7200))
    if marker:
        stage['resume_marker'] = str(marker)
    stages.append(stage)


def caches(split, stride, windows, frames):
    vace = cache/f'vace_{split}_s{stride}.pt'
    side = cache/f'hmr_{split}_s{stride}_raw25.pt'
    add('vace_'+split, command('video_VACE/divibody_training/cache_continuous_3dpw_vace_features_v31_resume.py',
        '--data-root', external/'datasets/3dpw', '--split', split, '--target-stride', stride,
        '--vace-checkpoint-dir', external/'video_VACE/models/Wan2.1-VACE-1.3B', '--output', vace,
        '--device', 'cuda:0', '--gpu-memory-limit-gib', 12, '--threads', 4, '--shard-windows', 32),
        [vace], command('scripts/verify_rebuild_stage_v31.py', '--kind', 'vace', '--output', vace,
            '--split', split, '--expected-windows', windows, '--expected-frames', frames),
        marker=Path(str(vace)+'.shards/run.json'))
    add('hmr_'+split, command('video_VACE/divibody_training/cache_hmr2a_rgb_anchor_v31_resume.py',
        '--vace-cache', vace, '--data-root', external/'datasets/3dpw', '--hmr-source-root', hmr,
        '--hmr2a-checkpoint', hmr/'logs/train/multiruns/hmr2a/0/checkpoints/epoch=10-step=25000.ckpt',
        '--hmr-model-config', hmr/'logs/train/multiruns/hmr2a/0/model_config.yaml',
        '--hmr-data-root', hmr/'data', '--output', side, '--device', 'cuda:0',
        '--cuda-memory-limit-gib', 8, '--cpu-threads', 4, '--seed', 2026, '--batch-size', 2),
        [side], command('scripts/verify_rebuild_stage_v31.py', '--kind', 'hmr', '--output', side,
            '--split', split, '--expected-windows', windows, '--expected-frames', frames),
        marker=Path(str(side)+'.parts/contract.json'), gpu=10240)


def eval_command(version, checkpoint, report):
    return command('scripts/evaluate_body_v31.py', '--model-version', version, '--checkpoint', checkpoint,
        '--checkpoint-weights', 'model', '--split', 'validation',
        '--vace-cache', cache/'vace_validation_s4.pt', '--hmr-sidecar', cache/'hmr_validation_s4_raw25.pt',
        '--coverage-manifest', manifest, '--smpl-model-root', smpl, '--joint-regressor', regressor,
        '--output', report, '--device', 'cuda:0', '--batch-size', 8, '--num-workers', 0,
        '--amp-dtype', 'bf16', '--seed', 20261008, '--max-cuda-memory-gib', 10, '--hash-large-caches')


caches('validation', 4, 2578, 10413)
add('v28_full_validation', eval_command('v28', initial, baseline), [baseline],
    command('scripts/verify_rebuild_stage_v31.py', '--kind', 'evaluation', '--output', baseline,
        '--split', 'validation', '--expected-frames', 10413), gpu=12288)
caches('train', 8, 2909, 22735)
add('prepare_experiments', command('scripts/prepare_rebuilt_experiments_v31.py',
    '--cache-root', cache, '--baseline-report', baseline, '--config-dir', configs,
    '--runs-root', runs, '--init-checkpoint', initial, '--smpl-model-root', smpl,
    '--joint-regressor', regressor, '--validation-manifest', manifest, '--amp-dtype', 'bf16'),
    [configs/'experiment_plan.json'], command('scripts/verify_experiment_stage_v31.py',
        '--kind', 'prepared', '--output', configs/'experiment_plan.json'), gpu=0)
for short, version in (('v30', 'v30'), ('v31_robust', 'v31')):
    wrapper = 'scripts/train_from_config_v30_rebuild.py' if version == 'v30' else 'scripts/train_from_config_v31.py'
    for mode in ('uninterrupted', 'paused', 'resume'):
        config = configs/f'smoke_{short}_{mode}.json'
        run = runs/'smoke'/(short+('_uninterrupted' if mode == 'uninterrupted' else '_resumed'))
        snapshot = out/f'{short}_step2'
        if mode == 'paused':
            cmd = command('scripts/run_paused_training_stage_v31.py', '--config', config,
                          '--model-version', version, '--snapshot-prefix', snapshot)
            output = snapshot.with_suffix('.json')
            outputs = [output, snapshot.with_suffix('.pt')]
            check = command('scripts/verify_experiment_stage_v31.py', '--kind', 'training',
                '--config', config, '--output', output, '--steps', 2, '--paused',
                '--checkpoint', snapshot.with_suffix('.pt'))
        else:
            cmd = command(wrapper, '--config', config)
            output = run/'completion.json'
            outputs = [output] if mode == 'resume' else [output, run/'latest.pt']
            check = command('scripts/verify_experiment_stage_v31.py', '--kind', 'training',
                            '--config', config, '--output', output, '--steps', 4)
        add(f'smoke_{short}_{mode}', cmd, outputs, check, gpu=12288)
    comparison = out/f'{short}_exact_resume.json'
    add(short+'_compare_resume', command('scripts/verify_training_resume_v31.py',
        '--uninterrupted', runs/'smoke'/(short+'_uninterrupted')/'latest.pt',
        '--resumed', runs/'smoke'/(short+'_resumed')/'latest.pt', '--output', comparison),
        [comparison], command('scripts/verify_experiment_stage_v31.py', '--kind', 'resume-result',
                              '--output', comparison), gpu=0)
for name in ('v30_rebuild', 'v31_robust', 'v31_shared_shape', 'v31_local_schedule'):
    version = 'v30' if name == 'v30_rebuild' else 'v31'
    wrapper = 'scripts/train_from_config_v30_rebuild.py' if version == 'v30' else 'scripts/train_from_config_v31.py'
    config = configs/(name+'.json'); run = runs/'formal'/name
    add(name+'_train6000', command(wrapper, '--config', config), [run/'completion.json', run/'latest.pt', run/'best.pt'],
        command('scripts/verify_experiment_stage_v31.py', '--kind', 'training', '--config', config,
                '--output', run/'completion.json', '--steps', 6000), gpu=12288)
    report = out/(name+'_selected_full_validation.json')
    add(name+'_selected_full_validation', eval_command(version, run/'best.pt', report), [report],
        command('scripts/verify_experiment_stage_v31.py', '--kind', 'evaluation', '--config', config,
                '--output', report), gpu=12288)

prerequisites = []
for path, status in ((root/'runs/cache_rebuild_pilot_002/vace_parity.json', 'exact_match'),
                     (root/'runs/cache_rebuild_pilot_003/hmr_original_batch2_parity.json', 'exact_match'),
                     (root/'runs/real_precision_v30_bf16_001/result.json', 'passed'),
                     (root/'runs/real_precision_v31_bf16_001/result.json', 'passed')):
    data = json.loads(path.read_text())
    if data['status'] != status:
        raise ValueError('Real pilot prerequisite failed: '+str(path))
    prerequisites.append(dict(path=str(path), sha256=digest(path), required_status=status))
sources = {p.relative_to(root).as_posix():digest(p) for folder in ('scripts','video_VACE')
           for p in (root/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.sh') and '__pycache__' not in p.parts}
for name in ('configs/star_body_v31.compat.example.json',):
    sources[name] = digest(root/name)
plan = dict(format='star-body-finite-full-rebuild-pipeline-v31-v1', test_used_for_selection=False,
            test_evaluation_started=False, sota_claim=False, stages=stages, source_sha256=sources,
            prerequisites=prerequisites, physical_gpu=1, training_seed=2026, evaluation_seed=20261008,
            cache_seed=2026, training_budget_per_arm=6000, formal_arms=4,
            hmr_parity_reference='Unmodified original v11 raw25 builder using matched batch2; historical small numerical differences are retained separately in hmr_parity.json',
            fullcache_protocol=dict(train=dict(stride=8,windows=2909,frames=22735),
                                    validation=dict(stride=4,windows=2578,frames=10413)),
            checkpoint_selection='Existing strict full-validation-only student/EMA policy; measured matched V28 acceleration ceiling',
            public_sota_comparison='Not established: current matched project protocol uses annotation-derived person crops')
path = out/'pipeline_plan.json'
with path.open('x') as stream:
    json.dump(plan, stream, indent=2, allow_nan=False)
print(json.dumps(dict(plan=str(path), sha256=digest(path), stages=len(stages), source_files=len(sources))))
