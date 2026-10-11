"""Read-only exact comparison of trusted native four-step training checkpoints.

This checks final saved state, not the authenticity of the preceding run logs.
Only use checkpoints produced by this project: torch pickle loading is trusted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import uuid


NATIVE_FORMATS = ('spatial-dual-order-smpl-feedback-v30',
                  'spatial-dual-order-smpl-feedback-v31')
SOURCE_PROFILES = {
    NATIVE_FORMATS[0]: {
        'engine': 'video_VACE/divibody_training/v30_rebuild_training_engine.py',
        'required': (
            'video_VACE/models/human_decoder_spatiotemporal_v30.py',
            'video_VACE/divibody_training/ram_parts_runtime_v30.py',
            'video_VACE/divibody_training/parts_training_utils_v30.py',
            'video_VACE/divibody_training/v29_boundary_partition.py',
            'video_VACE/divibody_training/v30_rebuild_training_contract.py',
            'video_VACE/divibody_training/train_spatiotemporal_v30_rebuild.py',
            'video_VACE/divibody_training/v30_rebuild_training_engine.py',
        ),
    },
    NATIVE_FORMATS[1]: {
        'engine': 'video_VACE/divibody_training/v31_training_engine.py',
        'required': (
            'video_VACE/models/human_decoder_spatiotemporal_v31.py',
            'video_VACE/divibody_training/ram_parts_runtime_v31.py',
            'video_VACE/divibody_training/parts_training_utils_v31.py',
            'video_VACE/divibody_training/v31_boundary_partition.py',
            'video_VACE/divibody_training/v31_training_contract.py',
            'video_VACE/divibody_training/train_spatiotemporal_v31.py',
            'video_VACE/divibody_training/v31_training_engine.py',
        ),
    },
}
STATE_FIELDS = ('model', 'ema_model', 'optimizer', 'ema_updates', 'step', 'epoch',
                'stream_states', 'torch_rng', 'cuda_rng', 'python_rng', 'numpy_rng',
                'best_score', 'best_step', 'stale_evaluations', 'invariant_config',
                'baseline_geometry', 'baseline_pose', 'validation', 'pose_validation',
                'history')
METADATA_FIELDS = ('format', 'model_config', 'adaptation_config', 'reference_spec',
                   'source_sha256', 'source_files_sha256', 'asset_manifest',
                   'cache_hash_policy', 'environment', 'development_only',
                   'test_used_for_model_selection', 'selection_metric', 'variant')
OPERATIONAL_ARGS = ('output_dir', 'resume_checkpoint', 'stop_after', 'num_workers')
# Exact key allowlist, only inside history and evaluation records. Each ignored
# path is reported. Names such as "loss" or "score" can never match this list.
RESOURCE_FIELDS = frozenset(('elapsed_seconds', 'elapsed_time_seconds',
    'wall_time_seconds', 'inference_seconds', 'elapsed', 'duration_seconds',
    'peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes', 'peak_memory_bytes',
    'peak_memory_mb', 'rss_bytes', 'peak_rss_bytes', 'timestamp', 'time_unix'))
METRIC_ROOTS = frozenset(('history', 'baseline_geometry', 'baseline_pose',
                         'validation', 'pose_validation'))
MAX_DEPTH = 64
MAX_NODES = 1000000
MAX_RECORDED_DIFFERENCES = 200


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def atomic_new_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)  # Never overwrite an earlier acceptance report.
    finally:
        temporary.unlink()


class ExactComparison:
    def __init__(self, torch, numpy):
        self.torch, self.np = torch, numpy
        self.differences = []
        self.difference_count = 0
        self.nodes = self.tensors = self.arrays = 0
        self.ignored_paths = []

    def difference(self, path, reason, **details):
        self.difference_count += 1
        if len(self.differences) < MAX_RECORDED_DIFFERENCES:
            self.differences.append(dict(path=path, reason=reason, **details))

    def require(self, condition, path, reason):
        if not condition:
            self.difference(path, reason)

    def finite_tensor(self, value):
        if value.is_floating_point() or value.is_complex():
            flat = value.reshape(-1)
            for start in range(0, flat.numel(), 262144):
                if not bool(self.torch.isfinite(flat[start:start + 262144]).all()):
                    return False
        return True

    def compare(self, left, right, path, ignore_resources=False):
        # Iterative traversal has explicit limits and rejects cyclic structures.
        pending = [(left, right, path, 0, frozenset())]
        while pending:
            left, right, current, depth, ancestors = pending.pop()
            self.nodes += 1
            if self.nodes > MAX_NODES or depth > MAX_DEPTH:
                raise ValueError('Checkpoint nested state exceeds explicit comparison limits')
            if isinstance(left, self.torch.Tensor) or isinstance(right, self.torch.Tensor):
                self.tensors += 1
                if not isinstance(left, self.torch.Tensor) or not isinstance(right, self.torch.Tensor):
                    self.difference(current, 'tensor_type_mismatch')
                elif (left.dtype != right.dtype or left.shape != right.shape or left.layout != right.layout):
                    self.difference(current, 'tensor_schema_mismatch',
                        left_shape=list(left.shape), right_shape=list(right.shape),
                        left_dtype=str(left.dtype), right_dtype=str(right.dtype))
                elif left.layout != self.torch.strided or left.is_quantized or right.is_quantized:
                    self.difference(current, 'unsupported_tensor_layout')
                elif not self.finite_tensor(left) or not self.finite_tensor(right):
                    self.difference(current, 'nonfinite_tensor')
                elif not self.torch.equal(left, right):
                    self.difference(current, 'tensor_values_not_exact')
                continue
            if isinstance(left, self.np.ndarray) or isinstance(right, self.np.ndarray):
                self.arrays += 1
                if not isinstance(left, self.np.ndarray) or not isinstance(right, self.np.ndarray):
                    self.difference(current, 'numpy_type_mismatch')
                elif left.shape != right.shape or left.dtype != right.dtype:
                    self.difference(current, 'numpy_schema_mismatch')
                elif left.dtype.hasobject:
                    self.difference(current, 'object_numpy_array_not_allowed')
                elif (left.dtype.kind in 'fc' and
                      (not self.np.isfinite(left).all() or not self.np.isfinite(right).all())):
                    self.difference(current, 'nonfinite_numpy_array')
                elif not self.np.array_equal(left, right):
                    self.difference(current, 'numpy_values_not_exact')
                continue
            if isinstance(left, dict) and isinstance(right, dict):
                pair = (id(left), id(right))
                if pair in ancestors:
                    raise ValueError('Cyclic checkpoint dictionaries are not accepted')
                if any(not isinstance(key, (str, int)) for key in (*left.keys(), *right.keys())):
                    self.difference(current, 'unsupported_dictionary_key_type')
                    continue
                keys = set(left) | set(right)
                for key in sorted(keys, key=lambda value: (type(value).__name__, str(value))):
                    child = current + '[' + repr(key) + ']'
                    if ignore_resources and key in RESOURCE_FIELDS:
                        self.ignored_paths.append(child)
                        continue
                    if key not in left or key not in right:
                        self.difference(child, 'missing_key', left_present=key in left, right_present=key in right)
                    else:
                        pending.append((left[key], right[key], child, depth + 1, ancestors | {pair}))
                continue
            if type(left) is not type(right):
                self.difference(current, 'value_type_mismatch',
                                left_type=type(left).__name__, right_type=type(right).__name__)
                continue
            if isinstance(left, (tuple, list)):
                pair = (id(left), id(right))
                if pair in ancestors:
                    raise ValueError('Cyclic checkpoint sequences are not accepted')
                if len(left) != len(right):
                    self.difference(current, 'sequence_length_mismatch', left_length=len(left), right_length=len(right))
                for index in range(min(len(left), len(right))):
                    pending.append((left[index], right[index], f'{current}[{index}]', depth + 1, ancestors | {pair}))
            elif isinstance(left, self.np.generic):
                if left.dtype.kind in 'fc' and (not self.np.isfinite(left) or not self.np.isfinite(right)):
                    self.difference(current, 'nonfinite_numpy_scalar')
                elif left.dtype != right.dtype or left != right:
                    self.difference(current, 'numpy_scalar_not_exact')
            elif isinstance(left, (float, complex)):
                finite = (math.isfinite(left.real) and math.isfinite(left.imag)
                          and math.isfinite(right.real) and math.isfinite(right.imag))
                if not finite:
                    self.difference(current, 'nonfinite_scalar')
                elif left != right:
                    self.difference(current, 'scalar_not_exact')
            elif left is None or isinstance(left, (str, int, bool, bytes)):
                if left != right:
                    self.difference(current, 'scalar_not_exact')
            else:
                self.difference(current, 'unsupported_value_type', value_type=type(left).__name__)


def validate_checkpoint(payload, name, comparison):
    if not isinstance(payload, dict):
        comparison.difference(name, 'checkpoint_must_be_dictionary')
        return False
    missing = [key for key in (*STATE_FIELDS, *METADATA_FIELDS, 'args') if key not in payload]
    for key in missing:
        comparison.difference(name + '.' + key, 'required_resumable_field_missing')
    comparison.require(payload.get('format') in NATIVE_FORMATS, name + '.format', 'not_native_v30_or_v31')
    comparison.require(payload.get('step') == 4 and type(payload.get('step')) is int,
                       name + '.step', 'expected_exactly_four_completed_steps')
    comparison.require(type(payload.get('ema_updates')) is int and payload['ema_updates'] == 4,
                       name + '.ema_updates', 'expected_four_ema_updates')
    for key in ('model', 'ema_model', 'optimizer', 'stream_states', 'invariant_config', 'args'):
        comparison.require(isinstance(payload.get(key), dict) and bool(payload.get(key)),
                           name + '.' + key, 'required_nonempty_state_dictionary')
    if missing or not isinstance(payload.get('invariant_config'), dict) or not isinstance(payload.get('args'), dict):
        return False
    invariant, arguments = payload['invariant_config'], payload['args']
    comparison.require('geometry' in payload['stream_states'], name + '.stream_states', 'geometry_stream_state_missing')
    comparison.require(isinstance(payload['optimizer'].get('state'), dict)
                       and bool(payload['optimizer'].get('state'))
                       and isinstance(payload['optimizer'].get('param_groups'), list)
                       and bool(payload['optimizer'].get('param_groups')),
                       name + '.optimizer', 'optimizer_state_or_parameter_groups_missing')
    for key in ('model', 'ema_model'):
        comparison.require(any(str(field).startswith('reference.') for field in payload[key]),
                           name + '.' + key, 'native_frozen_reference_state_missing')
    rng = payload['torch_rng']
    comparison.require(isinstance(rng, comparison.torch.Tensor) and rng.dtype == comparison.torch.uint8
                       and rng.ndim == 1 and rng.numel() > 0, name + '.torch_rng', 'invalid_torch_rng_state')
    comparison.require(isinstance(payload['python_rng'], tuple) and len(payload['python_rng']) == 3,
                       name + '.python_rng', 'invalid_python_rng_state')
    comparison.require(isinstance(payload['numpy_rng'], tuple) and len(payload['numpy_rng']) == 5,
                       name + '.numpy_rng', 'invalid_numpy_rng_state')
    cuda_rng = payload['cuda_rng']
    comparison.require(cuda_rng is None or (isinstance(cuda_rng, list) and bool(cuda_rng)
                       and all(isinstance(value, comparison.torch.Tensor) and value.dtype == comparison.torch.uint8
                       and value.ndim == 1 and value.numel() > 0 for value in cuda_rng)),
                       name + '.cuda_rng', 'invalid_cuda_rng_state')
    comparison.require(invariant.get('max_steps') == 4 and arguments.get('max_steps') == 4,
                       name + '.max_steps', 'four_step_budget_not_bound')
    comparison.require(type(invariant.get('seed')) is int and 0 <= invariant['seed'] < 2**32,
                       name + '.seed', 'missing_or_invalid_seed')
    comparison.require(invariant.get('amp_dtype') in ('none', 'bf16'), name + '.amp_dtype', 'precision_not_bound')
    for key in ('seed', 'amp_dtype', 'max_steps'):
        comparison.compare(invariant.get(key), arguments.get(key), name + '.args_invariant.' + key)
    comparison.require(is_sha256(payload['source_sha256']), name + '.source_sha256', 'init_sha256_missing')
    comparison.compare(payload['source_sha256'], invariant.get('init_sha256'), name + '.init_binding')
    comparison.require(payload['cache_hash_policy'] == 'sha256', name + '.cache_hash_policy', 'full_asset_content_hashes_required')
    for key in ('source_files_sha256', 'asset_manifest', 'environment', 'cache_hash_policy'):
        comparison.compare(payload[key], invariant.get(key), name + '.invariant_binding.' + key)
    sources = payload['source_files_sha256']
    comparison.require(isinstance(sources, dict) and bool(sources), name + '.source_files_sha256', 'source_bindings_missing')
    if isinstance(sources, dict):
        for key, digest in sources.items():
            comparison.require(is_sha256(digest), name + '.source_files_sha256.' + str(key), 'invalid_source_sha256')
        profile = SOURCE_PROFILES.get(payload['format'])
        if profile is not None:
            for filename in profile['required']:
                comparison.require(filename in sources,
                                   name + '.source_files_sha256.' + filename, 'critical_source_binding_missing')
            comparison.require(is_sha256(invariant.get('engine_sha256')),
                               name + '.invariant_config.engine_sha256', 'engine_sha256_missing')
            comparison.compare(sources.get(profile['engine']), invariant.get('engine_sha256'),
                               name + '.engine_source_binding')
    assets = payload['asset_manifest']
    comparison.require(isinstance(assets, dict) and bool(assets), name + '.asset_manifest', 'asset_bindings_missing')
    if isinstance(assets, dict):
        required = ('init_checkpoint', 'train_vace_cache', 'train_hmr_sidecar', 'val_vace_cache', 'val_hmr_sidecar',
                    'smpl_neutral', 'smpl_male', 'smpl_female', 'smpl_to_j19', 'joint_regressor')
        for key in required:
            comparison.require(key in assets, name + '.asset_manifest.' + key, 'required_asset_binding_missing')
        for key, record in assets.items():
            comparison.require(isinstance(record, dict) and is_sha256(record.get('sha256'))
                               and type(record.get('bytes')) is int and record['bytes'] > 0,
                               name + '.asset_manifest.' + str(key), 'asset_full_content_binding_missing')
        if isinstance(assets.get('init_checkpoint'), dict):
            comparison.compare(assets['init_checkpoint'].get('sha256'), payload['source_sha256'], name + '.asset_init_binding')
    environment = payload['environment']
    for key in ('torch', 'cuda', 'deterministic_algorithms', 'matmul_allow_tf32', 'cudnn_allow_tf32'):
        comparison.require(isinstance(environment, dict) and key in environment,
                           name + '.environment.' + key, 'numerical_environment_not_bound')
    comparison.require(payload['test_used_for_model_selection'] is False,
                       name + '.test_used_for_model_selection', 'test_selection_flag_not_false')
    comparison.require(payload['variant'] == 'student', name + '.variant', 'resumable_checkpoint_must_save_student')
    return True


def verify(left, right, comparison):
    validate_checkpoint(left, 'uninterrupted', comparison)
    validate_checkpoint(right, 'resumed', comparison)
    if not isinstance(left, dict) or not isinstance(right, dict):
        return
    for key in (*METADATA_FIELDS, *STATE_FIELDS):
        if key in left and key in right:
            comparison.compare(left[key], right[key], key, ignore_resources=key in METRIC_ROOTS)
    for key, payload in (('uninterrupted', left), ('resumed', right)):
        arguments = payload.get('args', {})
        if not isinstance(arguments, dict):
            continue
        resume = arguments.get('resume_checkpoint')
        comparison.require(bool(resume) if key == 'resumed' else not resume,
                           key + '.args.resume_checkpoint', 'expected_fresh_vs_resumed_run_argument')
    if isinstance(left.get('args'), dict) and isinstance(right.get('args'), dict):
        a = {key: value for key, value in left['args'].items() if key not in OPERATIONAL_ARGS}
        b = {key: value for key, value in right['args'].items() if key not in OPERATIONAL_ARGS}
        comparison.compare(a, b, 'args_without_engine_operational_fields')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--uninterrupted', type=Path, required=True)
    parser.add_argument('--resumed', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New JSON acceptance report; no overwrite')
    args = parser.parse_args(argv)
    args.output = args.output.expanduser().resolve()
    if args.output.exists() or args.output.suffix.lower() != '.json':
        parser.error('--output must be a new .json path')
    started = time.perf_counter()
    result = dict(format='star-body-exact-training-resume-verification-v31', passed=False,
        expected_steps=4, expected_max_steps=4, equality='exact; no tolerance',
        checkpoint_trust='trusted project-produced pickle outputs only',
        provenance_scope='Compare embedded content SHA256 bindings; external assets are not reread',
        trajectory_limit='Final checkpoint equality does not prove stop-at-2 execution; retain run logs and intermediate checkpoint hashes',
        metrics_claim='No generalization or SOTA claim; this is a resume-state audit',
        explicitly_ignored_operational_args=list(OPERATIONAL_ARGS),
        permitted_metric_resource_keys=sorted(RESOURCE_FIELDS), inputs={}, differences=[])
    try:
        import numpy as np
        import torch
        torch.set_num_threads(4)
        paths = [args.uninterrupted.expanduser().resolve(strict=True), args.resumed.expanduser().resolve(strict=True)]
        if os.path.samefile(*paths):
            raise ValueError('Two distinct run checkpoints are required')
        snapshots = []
        for name, path in zip(('uninterrupted', 'resumed'), paths):
            stat = path.stat()
            snapshots.append((stat.st_size, stat.st_mtime_ns))
            result['inputs'][name] = dict(path=str(path), bytes=stat.st_size, sha256=sha256_file(path))
        payloads = [torch.load(path, map_location='cpu', weights_only=False, mmap=True) for path in paths]
        comparison = ExactComparison(torch, np)
        verify(*payloads, comparison)
        result.update(passed=comparison.difference_count == 0, difference_count=comparison.difference_count,
            differences=comparison.differences,
            differences_truncated=comparison.difference_count > len(comparison.differences),
            compared_nodes=comparison.nodes, compared_tensors=comparison.tensors, compared_numpy_arrays=comparison.arrays,
            ignored_metric_resource_paths=sorted(set(comparison.ignored_paths)),
            synthetic_mechanism_test_only=any(isinstance(p, dict) and p.get('synthetic_mechanism_test_only') is True for p in payloads),
            native_formats=[p.get('format') if isinstance(p, dict) else None for p in payloads])
        for path, snapshot in zip(paths, snapshots):
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != snapshot:
                raise ValueError('Checkpoint changed during the read-only audit: ' + str(path))
    except Exception as error:
        result.update(passed=False, error=dict(type=type(error).__name__, message=str(error)))
    result['elapsed_seconds'] = time.perf_counter() - started
    atomic_new_json(args.output, result)
    print(json.dumps(dict(passed=result['passed'], output=str(args.output),
                         difference_count=result.get('difference_count'), error=result.get('error'))))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
