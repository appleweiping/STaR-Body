"""Read-only controller receipts for prepared configs and completed training.

Checks stored state and content identity; does not claim to recompute metrics.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8*1024**2), b''):
            value.update(data)
    return value.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def normalize(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [normalize(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('prepared', 'training', 'resume-result', 'evaluation'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--checkpoint', type=Path, help='Immutable paused snapshot; otherwise run/latest.pt')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--paused', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.kind == 'prepared':
        plan = json.loads(args.output.read_text())
        require(plan['status'] == 'prepared_not_run' and plan['test_used_for_selection'] is False,
                'Not the preregistered preparation receipt')
        for entry in (plan['template'], plan['baseline']):
            require(digest(entry['path']) == entry['sha256'], 'Input receipt changed')
        for name, expected in plan['source_sha256'].items():
            require(digest(root/name) == expected, 'Registered source changed: '+name)
        require(len(plan['experiments']) == 10, 'Expected four arms and six resume configs')
        for spec in plan['experiments'].values():
            record = spec['config_file']
            require(digest(record['path']) == record['sha256'], 'Prepared config changed')
            require(json.loads(Path(record['path']).read_text()) == spec['config'], 'Config receipt mismatch')
    elif args.kind == 'resume-result':
        result = json.loads(args.output.read_text())
        require(result.get('format') == 'star-body-exact-training-resume-verification-v31'
                and result.get('passed') is True and result.get('difference_count') == 0
                and result.get('synthetic_mechanism_test_only') is False,
                'Real resume comparison did not pass')
        for name in ('uninterrupted', 'resumed'):
            record = result['inputs'][name]
            require(digest(record['path']) == record['sha256'], 'Compared checkpoint changed')
    elif args.kind == 'evaluation':
        require(args.config is not None, 'Evaluation needs the registered training config')
        config = json.loads(args.config.read_text())
        result = json.loads(args.output.read_text())
        require(result['status'] == 'complete_matched_project_protocol' and result['split'] == 'validation',
                'Expected complete matched validation')
        require((result['metrics']['frames'], result['metrics']['acceleration_triplets']) == (10413, 10226),
                'Incomplete validation counts')
        version = 'v30' if 'baseline-report' in config else 'v31'
        require(result['model_version'] == version, 'Model version differs')
        require(result['resources']['amp_dtype'] == config['amp-dtype'], 'Evaluation AMP differs')
        require(result['optimizer_steps'] == 0 and result['model_or_teacher_fitting_performed'] is False,
                'Evaluation unexpectedly fitted a model')
        require(result['coverage']['manifest']['exact_uid_match'] is True
                and result['coverage']['manifest']['independent_archive_verified'] is True,
                'Missing independent full coverage audit')
        expected_paths = dict(checkpoint=str(Path(config['output-dir'])/'best.pt'),
                              vace_cache=config['val-vace-cache'], hmr_sidecar=config['val-hmr-sidecar'])
        for name, value in expected_paths.items():
            record = result['assets'][name]
            require(Path(record['path']).resolve() == Path(value).resolve(), 'Evaluated asset path differs')
            require(digest(value) == record['sha256'], 'Evaluated asset changed')
        for name, expected in result['source_sha256'].items():
            require(digest(root/name) == expected, 'Evaluation source changed')
    else:
        require(args.config is not None and args.steps is not None, 'Training needs config and expected steps')
        import torch
        torch.set_num_threads(4)
        config = json.loads(args.config.read_text())
        run = Path(config['output-dir'])
        state = torch.load(args.checkpoint or run/'latest.pt', map_location='cpu', weights_only=False, mmap=True)
        require(state['format'] in ('spatial-dual-order-smpl-feedback-v30', 'spatial-dual-order-smpl-feedback-v31'),
                'Expected native V30/V31 format')
        require(state['step'] == args.steps, 'Checkpoint step differs')
        require(state['test_used_for_model_selection'] is False, 'Unexpected test selection')
        require(state['cache_hash_policy'] == 'sha256', 'Full cache hashes required')
        require(state['source_sha256'] == digest(config['init-checkpoint']), 'Initialization changed')
        require(all(key in state for key in ('optimizer','ema_model','stream_states','torch_rng','invariant_config')),
                'Missing resumable state')
        for key, value in config.items():
            actual = state['args'].get(key.replace('-', '_'))
            require(normalize(actual) == normalize(value), 'Saved training argument differs: '+key)
        for name, expected in state['source_files_sha256'].items():
            require(digest(root/name) == expected, 'Training source changed: '+name)
        for name, value in state['model'].items():
            if torch.is_tensor(value) and value.is_floating_point():
                require(bool(torch.isfinite(value).all()), 'Nonfinite parameter: '+name)
        receipt = json.loads(args.output.read_text())
        if args.paused:
            require(receipt['step'] == args.steps and receipt['resumable'] is True, 'Missing pause receipt')
        else:
            require(receipt['complete'] is True and receipt['completed_steps'] == args.steps, 'Missing completion receipt')
    print(json.dumps(dict(status='passed', kind=args.kind, artifact=str(args.output),
                          sha256=digest(args.output), metrics_recomputed=False)))


if __name__ == '__main__':
    main()
