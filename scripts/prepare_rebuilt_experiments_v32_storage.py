"""CPU-only preregistration of matched V30/V31 experiments from complete caches.

No model forward, optimizer, teacher fit, or training run is started. Existing
templates, reports, assets and source files are never edited.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
CACHE_NAMES = {'train-vace-cache': 'vace_train_s8.pt',
               'train-hmr-sidecar': 'hmr_train_s8_raw25.pt',
               'val-vace-cache': 'vace_validation_s4.pt',
               'val-hmr-sidecar': 'hmr_validation_s4_raw25.pt'}
V31_SWITCHES = ('frame-shape-delta-scale', 'local-radius-schedule', 'deduplicate-shape-consistency')


def read_json(path):
    def reject(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=reject)


def fingerprint(path):
    path = Path(path).expanduser().resolve()
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(data)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Asset changed during hashing: ' + str(path))
    return dict(path=str(path), bytes=after.st_size, sha256=digest.hexdigest())


def config_cli(config):
    values = []
    for key, value in config.items():
        if value is None or value is False:
            continue
        values.append('--' + key)
        if value is True:
            continue
        if key == 'local-radius-schedule':
            value = ','.join(str(radius) for radius in value)
        if not isinstance(value, (str, int, float)):
            raise ValueError('Unsupported configuration value: ' + key)
        values.append(str(value))
    return values


def baseline_header(report, amp_dtype):
    if (report.get('status') != 'complete_matched_project_protocol'
            or report.get('model_version') != 'v28' or report.get('split') != 'validation'):
        raise ValueError('A completed full V28 validation report is required')
    metrics = report.get('metrics', {})
    if (metrics.get('frames'), metrics.get('acceleration_triplets')) != (10413, 10226):
        raise ValueError('Baseline must cover 10413 frames and 10226 adjacent triplets')
    for key in ('mpjpe_mm', 'pa_mpjpe_mm', 'pve_mm', 'acceleration_error_mm_per_frame2'):
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid measured baseline metric: ' + key)
    if report.get('resources', {}).get('amp_dtype') != amp_dtype:
        raise ValueError('Baseline AMP and requested experiment AMP differ')
    if report.get('optimizer_steps') != 0 or report.get('model_or_teacher_fitting_performed') is not False:
        raise ValueError('Baseline must be evaluation-only')
    ceiling = metrics.get('acceleration_error_mm_per_frame2')
    if isinstance(ceiling, bool) or not isinstance(ceiling, (float, int)) or not math.isfinite(ceiling) or ceiling < 0:
        raise ValueError('Invalid measured acceleration ceiling')
    return float(ceiling)


def build_configs(template, paths, runs_root, baseline_report, amp_dtype, weights, ceiling):
    """Pure configuration construction, separated from real cache validation."""
    base = deepcopy(template)
    if any('test' in key.lower() for key in base):
        raise ValueError('Experiment templates cannot contain test options')
    required = {'batch-size': 4, 'max-steps': 6000, 'eval-every': 1000, 'seed': 2026, 'iterations': 3}
    if any(base.get(key) != value for key, value in required.items()):
        raise ValueError('Expected the preregistered template: batch4/6000steps/eval1000/seed2026/iterations3')
    base.update(paths)
    base.update({'amp-dtype': amp_dtype, 'val-batch-size': 8, 'num-workers': 0,
                 'verify-cache-sha256': True, 'pose-2d-weight': 0., 'coco-weight': 0.,
                 'coco-warmup-steps': 0, 'coco-train-manifest': None,
                 'geometry-only-validation': True, 'geometry-only-steps': 6001,
                 'max-train-windows': 0, 'max-val-windows': 0,
                 'allow-partial-validation': False, 'strict-3d-selection': True,
                 'acceleration-weight': weights['within'], 'boundary-weight': weights['boundary'],
                 'acceleration-ceiling-mm': ceiling, 'stop-after': 0, 'resume-checkpoint': None})
    for key in ('pose-train-vace-dir', 'pose-train-hmr-dir', 'pose-val-vace-dir', 'pose-val-hmr-dir'):
        base[key] = None
    base.pop('baseline-report', None)
    base.update({'frame-shape-delta-scale': 1., 'local-radius-schedule': [],
                 'deduplicate-shape-consistency': False})
    specs = {}
    for name, version in (('v30_rebuild', 'v30'), ('v31_robust', 'v31'),
                          ('v31_shared_shape', 'v31'), ('v31_local_schedule', 'v31')):
        config = deepcopy(base)
        config['output-dir'] = str(runs_root / 'formal' / name)
        if version == 'v30':
            for key in V31_SWITCHES:
                config.pop(key)
            config['baseline-report'] = str(baseline_report)
        elif name == 'v31_shared_shape':
            config['frame-shape-delta-scale'] = 0.
        elif name == 'v31_local_schedule':
            config['local-radius-schedule'] = [2., 1., .5]
        specs[name] = dict(kind='formal', model_version=version, config=config,
                           initialization='independent original V28 initialization')
    for source, short in (('v30_rebuild', 'v30'), ('v31_robust', 'v31_robust')):
        smoke = deepcopy(specs[source]['config'])
        smoke.update({'max-steps': 4, 'eval-every': 2})
        for mode in ('uninterrupted', 'paused', 'resume'):
            config = deepcopy(smoke)
            run_name = short + ('_uninterrupted' if mode == 'uninterrupted' else '_resumed')
            config['output-dir'] = str(runs_root / 'smoke' / run_name)
            config['stop-after'] = 2 if mode == 'paused' else 0
            config['resume-checkpoint'] = str(Path(config['output-dir']) / 'latest.pt') if mode == 'resume' else None
            specs['smoke_' + short + '_' + mode] = dict(kind='engine_resume_acceptance',
                model_version=specs[source]['model_version'], config=config,
                initialization='same-run latest.pt at step2' if mode == 'resume' else 'independent original V28 initialization')
    return specs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache-root', 'baseline-report', 'config-dir', 'runs-root', 'init-checkpoint',
                 'smpl-model-root', 'joint-regressor', 'validation-manifest'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--template', type=Path, default=ROOT / 'configs/star_body_v31.compat.example.json')
    parser.add_argument('--amp-dtype', choices=('none', 'bf16'), default='bf16')
    return parser.parse_args(argv)


def main():
    args = parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    if args.config_dir.exists() and (not args.config_dir.is_dir() or any(args.config_dir.iterdir())):
        raise FileExistsError('config-dir must be new or empty')
    template = read_json(args.template)
    baseline = read_json(args.baseline_report)
    ceiling = baseline_header(baseline, args.amp_dtype)
    paths = {key: str((args.cache_root / filename).resolve(strict=True)) for key, filename in CACHE_NAMES.items()}
    paths.update({'init-checkpoint': str(args.init_checkpoint), 'smpl-model-root': str(args.smpl_model_root),
                  'joint-regressor': str(args.joint_regressor), 'validation-manifest': str(args.validation_manifest)})
    asset_paths = {key.replace('-', '_'): Path(value) for key, value in paths.items() if key != 'smpl-model-root'}
    asset_paths['smpl_to_j19'] = args.smpl_model_root.parent / 'SMPL_to_J19.pkl'
    asset_paths.update({'smpl_' + gender.lower(): args.smpl_model_root / ('SMPL_' + gender + '.pkl')
                       for gender in ('NEUTRAL', 'MALE', 'FEMALE')})
    assets = {}
    for name, path in asset_paths.items():
        print(json.dumps({'phase': 'fingerprint_assets', 'asset': name}), flush=True)
        assets[name] = fingerprint(path)
    sys.path[:0] = [str(ROOT / 'video_VACE'), str(ROOT / 'video_VACE/divibody_training'), str(ROOT / 'scripts'), str(ROOT)]
    sys.dont_write_bytecode = True
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    from evaluate_body_v31 import audit_coverage, verify_cache_provenance
    from divibody_training.ram_parts_runtime_v29 import V10Dataset
    from divibody_training.v29_boundary_training import build_boundary_pool
    from divibody_training.v31_boundary_partition import audit_boundary_partition, class_fraction_weights
    from divibody_training import v30_rebuild_training_contract as v30_contract
    from divibody_training import v31_training_contract as v31_contract
    loaded = {key: torch.load(paths[key], map_location='cpu', weights_only=False, mmap=True) for key in CACHE_NAMES}
    train, train_hmr = loaded['train-vace-cache'], loaded['train-hmr-sidecar']
    val, val_hmr = loaded['val-vace-cache'], loaded['val-hmr-sidecar']
    for payload, split, stride in ((train, 'train', 8), (val, 'validation', 4)):
        if payload['signature'].get('split') != split or payload['signature'].get('target_stride') != stride:
            raise ValueError('Rebuilt cache split/stride differs from the preregistered protocol')
    provenance = {'train': verify_cache_provenance(train, train_hmr, paths['train-vace-cache'], 'train', False),
                  'validation': verify_cache_provenance(val, val_hmr, paths['val-vace-cache'], 'validation', False)}
    # Train has no label-driven independent evaluation manifest; use this helper
    # for exact UID/owner audit, then explicitly enforce full expected counts.
    train_coverage, train_uids = audit_coverage(train, train_hmr, 'train', None, True)
    validation_coverage, validation_uids = audit_coverage(val, val_hmr, 'validation', args.validation_manifest, False)
    if (train_coverage['frames'], train_coverage['acceleration_triplets']) != (22735, 22481):
        raise ValueError('Formal training requires all 22735 train frames / 22481 triplets')
    train_sequences = {uid.split('/person_', 1)[0] for uid in train_uids}
    val_sequences = {uid.split('/person_', 1)[0] for uid in validation_uids}
    if train_sequences & val_sequences:
        raise ValueError('Train and validation contain overlapping sequences')
    dataset = V10Dataset(train, train_hmr)
    pool = build_boundary_pool(dataset, forbidden_sequences=val_sequences)
    partition = audit_boundary_partition(dataset, pool)
    weights = class_fraction_weights(4., partition)
    binding_args = SimpleNamespace(baseline_report=args.baseline_report, allow_partial_validation=False,
                                  acceleration_ceiling_mm=ceiling, amp_dtype=args.amp_dtype)
    binding = v30_contract.verify_baseline_binding(binding_args, {'assets': assets}, {'complete': True}, val)
    reference = baseline['coverage']['manifest']
    if reference['sha256'] != assets['validation_manifest']['sha256']:
        raise ValueError('Current independent validation manifest differs from the baseline report')
    configs = build_configs(template, paths, args.runs_root, args.baseline_report, args.amp_dtype, weights, ceiling)
    for name, spec in configs.items():
        parser = v30_contract.parse_training_args if spec['model_version'] == 'v30' else v31_contract.parse_training_args
        parser(config_cli(spec['config']))
        run_directory = Path(spec['config']['output-dir'])
        if run_directory.exists() and (not run_directory.is_dir() or any(run_directory.iterdir())):
            raise FileExistsError('New experiment output must be absent or empty: ' + str(run_directory))
    sources = v30_contract.source_fingerprints() | v31_contract.source_fingerprints()
    for path in (Path(__file__), ROOT / 'scripts/evaluate_body_v31.py'):
        sources[str(path.relative_to(ROOT))] = fingerprint(path)['sha256']
    template_record = fingerprint(args.template)
    baseline_record = fingerprint(args.baseline_report)
    args.config_dir.mkdir(parents=True, exist_ok=True)
    copies = []
    for name, spec in configs.items():
        destination = args.config_dir / (name + '.json')
        if destination.exists():
            raise FileExistsError(destination)
        # Copy first, then edit the new configuration; originals stay immutable.
        shutil.copyfile(args.template, destination)
        copied = fingerprint(destination)['sha256']
        if copied != template_record['sha256']:
            raise RuntimeError('Template changed while copying')
        destination.write_text(json.dumps(spec['config'], indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
        config_record = fingerprint(destination)
        copies.append(dict(source=template_record, destination=str(destination),
                           copied_sha256_before_edit=copied, destination_sha256_after_edit=config_record['sha256']))
        script = 'scripts/train_from_config_v30_rebuild.py' if spec['model_version'] == 'v30' else 'scripts/train_from_config_v31.py'
        spec.update(config_file=config_record, command=['bash', 'scripts/run_star_body_v31.sh', script, '--config', str(destination)])
        Path(spec['config']['output-dir']).mkdir(parents=True, exist_ok=True)
    plan = dict(format='star-body-rebuilt-experiments-v31-v1', status='prepared_not_run',
                amp_dtype=args.amp_dtype, training_started=False, test_used_for_selection=False,
                baseline=baseline_record, baseline_metrics=baseline['metrics'], baseline_binding=binding,
                template=template_record, assets=assets, source_sha256=sources, partition=partition,
                motion_total_weight=4., motion_weights=weights, train_coverage=train_coverage,
                validation_coverage=validation_coverage, cache_provenance=provenance,
                initialization='Each formal arm independently loads the identical original V28 model field',
                selection_rule='Full validation only; existing strict 3D policy; acceleration ceiling from same-precision V28 full validation',
                evaluation_protocol=baseline['protocol'], formal_arms=[name for name, spec in configs.items() if spec['kind']=='formal'],
                resume_acceptance='Compare final latest.pt at step4: uninterrupted vs paused at step2 then resumed to step4; model/EMA/optimizer/scheduler/RNG and stream states require exact equality',
                experiments=configs)
    for filename, value in (('configuration_copy_provenance.json', copies), ('experiment_plan.json', plan)):
        with (args.config_dir / filename).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({'status': 'prepared_not_run', 'config_dir': str(args.config_dir),
                      'formal_arms': 4, 'engine_acceptance_configs': 6, 'partition': partition,
                      'motion_weights': weights, 'amp_dtype': args.amp_dtype}, indent=2), flush=True)


if __name__ == '__main__':
    main()
