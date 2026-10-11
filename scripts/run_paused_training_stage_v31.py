"""Run the original four-step trainer to step two and retain immutable evidence.

Copied from train_from_config_v31.py before editing. Original trainers/engines
remain unchanged. A fully verified existing paused run can finish snapshot
publication after a launcher interruption; an existing snapshot is never replaced.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PATH_KEYS = {'train-vace-cache', 'train-hmr-sidecar', 'val-vace-cache', 'val-hmr-sidecar',
    'pose-train-vace-dir', 'pose-train-hmr-dir', 'pose-val-vace-dir', 'pose-val-hmr-dir',
    'init-checkpoint', 'smpl-model-root', 'joint-regressor', 'output-dir', 'validation-manifest',
    'coco-train-manifest', 'resume-checkpoint', 'baseline-report'}
FORMATS = {version: 'spatial-dual-order-smpl-feedback-' + version for version in ('v30', 'v31')}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(data)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def resolved(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def normalize(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [normalize(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}
    return value


def validate_config(config, version):
    require(isinstance(config, dict) and config, 'Expected a nonempty training configuration')
    require(config.get('max-steps') == 4 and config.get('stop-after') == 2,
            'This launcher only accepts max-steps=4 and stop-after=2')
    require(config.get('resume-checkpoint') is None and not config.get('eval-only', False),
            'Pause evidence must come from a fresh training run, not resume/eval-only')
    if version == 'v30':
        require(not {'frame-shape-delta-scale', 'local-radius-schedule', 'deduplicate-shape-consistency'} & config.keys(),
                'Original V30 cannot use V31 model/loss switches')
    require(config.get('verify-cache-sha256') is True, 'Pause acceptance requires full asset hash bindings')
    return resolved(config['output-dir'])


def verify_paused_run(config, version, directory):
    import torch
    torch.set_num_threads(4)
    receipt_path, checkpoint_path = directory / 'paused.json', directory / 'latest.pt'
    receipt = read_json(receipt_path)
    require(receipt.get('step') == 2 and receipt.get('reason') == 'stop_after' and receipt.get('resumable') is True,
            'The original engine did not publish a resumable step-two pause receipt')
    before = checkpoint_path.stat()
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
    require(state.get('format') == FORMATS[version] and state.get('step') == 2
            and state.get('variant') == 'student' and state.get('ema_updates') == 2,
            'Expected the selected native model with two completed optimizer/EMA updates')
    require(state.get('test_used_for_model_selection') is False and state.get('cache_hash_policy') == 'sha256',
            'Pause state lacks the declared data/selection contract')
    require(all(key in state for key in ('model', 'ema_model', 'optimizer', 'stream_states', 'torch_rng',
            'python_rng', 'numpy_rng', 'cuda_rng', 'invariant_config', 'args', 'asset_manifest', 'source_files_sha256')),
            'Missing resumable optimizer/RNG/stream/provenance state')
    for key, value in config.items():
        expected = str(resolved(value)) if key in PATH_KEYS and value is not None else normalize(value)
        require(normalize(state['args'].get(key.replace('-', '_'))) == expected,
                'Saved paused-run argument differs from configuration: ' + key)
    require(state['invariant_config'].get('max_steps') == 4 and state['args'].get('stop_after') == 2
            and state['args'].get('resume_checkpoint') is None, 'Pause trajectory arguments differ')
    require(state['source_sha256'] == digest(resolved(config['init-checkpoint'])), 'Original V28 initialization changed')
    for name, expected in state['source_files_sha256'].items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT) and digest(path) == expected, 'Training source changed: ' + name)
    for field in ('model', 'ema_model'):
        require(isinstance(state[field], dict) and state[field], 'Empty model state: ' + field)
        for name, tensor in state[field].items():
            if torch.is_tensor(tensor) and tensor.is_floating_point():
                flat = tensor.reshape(-1)
                for offset in range(0, flat.numel(), 1024**2):
                    require(bool(torch.isfinite(flat[offset:offset + 1024**2]).all()), 'Nonfinite paused state: ' + field + '.' + name)
    after = checkpoint_path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'Checkpoint changed during verification')
    return {'checkpoint': checkpoint_path, 'receipt': receipt_path, 'initialization_sha256': state['source_sha256'],
            'source_files_sha256': state['source_files_sha256'], 'format': state['format'], 'step': 2}


def stage_copy(source, destination):
    """Copy to a private sibling, fsync, then atomically publish without overwrite."""
    source, destination = Path(source), Path(destination)
    require(not destination.exists(), 'Refusing to overwrite snapshot: ' + str(destination))
    temporary = destination.with_name(destination.name + '.tmp-' + uuid.uuid4().hex)
    before = source.stat()
    try:
        with source.open('rb') as reader, temporary.open('xb') as writer:
            shutil.copyfileobj(reader, writer, 8 * 1024**2)
            writer.flush()
            os.fsync(writer.fileno())
        after = source.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'Source changed while copying')
        sha = digest(temporary)
        require(temporary.stat().st_size == after.st_size and sha == digest(source), 'Snapshot differs from original bytes')
        os.link(temporary, destination)
        return dict(path=str(destination), bytes=after.st_size, sha256=sha,
                    copied_from=str(source), source_sha256=sha)
    finally:
        temporary.unlink(missing_ok=True)


def publish_json(path, value):
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--model-version', choices=('v30', 'v31'), required=True)
    parser.add_argument('--snapshot-prefix', type=Path, required=True, help='New paths PREFIX.pt, PREFIX.json, PREFIX.provenance.json')
    args = parser.parse_args(argv)
    args.config = args.config.expanduser().resolve(strict=True)
    config_sha = digest(args.config)
    config = read_json(args.config)
    directory = validate_config(config, args.model_version)
    prefix = args.snapshot_prefix.expanduser().resolve()
    destinations = {key: Path(str(prefix) + suffix) for key, suffix in
                    (('checkpoint', '.pt'), ('receipt', '.json'), ('provenance', '.provenance.json'))}
    require(all(not path.exists() for path in destinations.values()), 'Snapshot prefix already exists; snapshots are immutable')
    require(all(path.resolve() not in (directory / 'latest.pt', directory / 'paused.json') for path in destinations.values()),
            'Snapshot must be separate from mutable engine outputs')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    wrapper = ROOT / ('scripts/train_from_config_v30_rebuild.py' if args.model_version == 'v30' else 'scripts/train_from_config_v31.py')
    command = [sys.executable, '-u', str(wrapper), '--config', str(args.config)]
    pause_exists, latest_exists = (directory / 'paused.json').exists(), (directory / 'latest.pt').exists()
    recovered = pause_exists and latest_exists
    if pause_exists or latest_exists:
        require(recovered, 'Incomplete preexisting pause state; refusing to restart or fabricate a receipt')
    else:
        completed = subprocess.run(command, cwd=ROOT)
        require(completed.returncode == 0, 'Original training wrapper failed; no pause snapshot was published')
    require(digest(args.config) == config_sha, 'Training configuration changed while running')
    verified = verify_paused_run(config, args.model_version, directory)
    copied = {key: stage_copy(verified[key], destinations[key]) for key in ('checkpoint', 'receipt')}
    provenance = dict(format='star-body-paused-training-snapshot-v31', status='paused_snapshot_ready',
        model_version=args.model_version, step=2, max_steps=4, recovered_existing_paused_run=recovered,
        config=dict(path=str(args.config), sha256=config_sha), wrapper=dict(path=str(wrapper), sha256=digest(wrapper)),
        launcher=dict(path=str(Path(__file__).resolve()), sha256=digest(__file__)),
        command=command, snapshot=copied, source_files_sha256=verified['source_files_sha256'],
        initialization_sha256=verified['initialization_sha256'], test_used_for_selection=False)
    publish_json(destinations['provenance'], provenance)
    print(json.dumps(dict(status='paused_snapshot_ready', model_version=args.model_version, step=2,
                         snapshots=copied, provenance=str(destinations['provenance']), recovered=recovered)), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
