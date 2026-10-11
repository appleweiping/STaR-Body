"""Render a non-executable evidence-pending plan for one finite V30 smoke pair."""
import argparse
import json
from pathlib import Path, PurePosixPath

PROJECT = '/media/lenovo/data2/mjx/STaR-Body_V31_20261008'
BASE = '/media/lenovo/data2/v28d_parallel_20261002'
RUN = '/home/lenovo/star_body_scratch_20261009/phase4_v30_smoke_001'
CACHE = BASE + '/data/star_body_v31_cache_20261008'
GIB = 1024 ** 3
ESTIMATED_K = 318131840
ESTIMATED_B = 113319548
MARGIN = GIB // 2
FLOOR = 2 * GIB


def command(script, *args):
    return ['bash', 'scripts/run_star_body_v31.sh', script, *map(str, args)]


def stages():
    run, configs, training = PurePosixPath(RUN), PurePosixPath(RUN) / 'configs', PurePosixPath(RUN) / 'training'
    prep = command('scripts/prepare_rebuilt_experiments_v32_storage.py', '--cache-root', CACHE,
        '--baseline-report', PROJECT + '/runs/phase3_eval_memory_001/v28_full_validation_bf16.json',
        '--config-dir', configs, '--runs-root', training,
        '--init-checkpoint', BASE + '/checkpoints/trained_v28_20261001.pt',
        '--smpl-model-root', PROJECT + '/checkpoints/local_smpl/smpl',
        '--joint-regressor', BASE + '/body_models/J_regressor_h36m.npy',
        '--validation-manifest', BASE + '/data/official_reference_v2/validation_reference.json', '--amp-dtype', 'bf16')
    result = [{'name': 'prepare', 'gpu': False, 'command': prep,
        'check': command('scripts/verify_experiment_stage_v31.py', '--kind', 'prepared',
                         '--output', configs / 'experiment_plan.json')}]
    for mode in ('uninterrupted', 'paused', 'resume'):
        config = configs / ('smoke_v30_' + mode + '.json')
        directory = training / 'smoke' / ('v30_uninterrupted' if mode == 'uninterrupted' else 'v30_resumed')
        output = directory / 'completion.json'
        cmd = command('scripts/train_from_config_v30_rebuild.py', '--config', config)
        check = command('scripts/verify_experiment_stage_v31.py', '--kind', 'training',
                        '--config', config, '--output', output, '--steps', 4)
        if mode == 'paused':
            prefix = run / 'evidence/v30_step2'
            cmd = command('scripts/run_paused_training_stage_v31.py', '--config', config,
                           '--model-version', 'v30', '--snapshot-prefix', prefix)
            check = command('scripts/verify_experiment_stage_v31.py', '--kind', 'training',
                '--config', config, '--output', str(prefix) + '.json', '--steps', 2, '--paused',
                '--checkpoint', str(prefix) + '.pt')
        result.append({'name': mode, 'gpu': True, 'command': cmd, 'check': check})
    compared = run / 'evidence/v30_exact_resume.json'
    result.append({'name': 'compare', 'gpu': False,
        'command': command('scripts/verify_training_resume_v31.py',
            '--uninterrupted', training / 'smoke/v30_uninterrupted/latest.pt',
            '--resumed', training / 'smoke/v30_resumed/latest.pt', '--output', compared),
        'check': command('scripts/verify_experiment_stage_v31.py', '--kind', 'resume-result', '--output', compared)})
    return result


def remaining_budget(stage, k=ESTIMATED_K, b=ESTIMATED_B):
    # Existing output already consumes disk; never add it to currently available space.
    extra = {'prepare': 4*k + 2*b, 'uninterrupted': 4*k + 2*b,
             'paused': max(3*k + b, 2*k + 2*b), 'resume': max(k, b), 'compare': 0}[stage]
    required = FLOOR + MARGIN + extra
    return max(4 * GIB, required) if stage in ('prepare', 'uninterrupted') else required


def template():
    return {'format': 'star-body-finite-v30-pair-plan-v1', 'ready_for_execution': False,
        'project_root': PROJECT, 'run_dir': RUN, 'numerical_protocol_change': False,
        'v31_training_authorized': False, 'formal_training_authorized': False, 'test_used': False,
        'resources': {'host_start_gib': 12, 'host_runtime_gib': 3, 'session_anon_gib': 8,
            'gpu_free_mib': 12288, 'allocator_limit_gib': 10, 'root_runtime_bytes': FLOOR,
            'physical_gpu_index': 1, 'logical_training_device': 'cuda:0',
            'initial_pair_tensor_bytes': 4*ESTIMATED_K + 2*ESTIMATED_B,
            'metadata_margin_bytes': MARGIN, 'initial_required_free_bytes': remaining_budget('uninterrupted'),
            'poll_seconds': 1, 'stage_timeout_seconds': 7200, 'resource_wait_seconds': 3600},
        'source_sha256': {}, 'parent_plan': None,
        'prerequisites': {'cache_acceptance': {}, 'baseline_acceptance': None, 'hmr_recovery_status': None},
        'stages': stages(), 'postcondition': 'Only one original V30 pair; stop after exact comparison; no 6000-step or V31 dispatch'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(template(), stream, indent=2)
        stream.write('\n')
