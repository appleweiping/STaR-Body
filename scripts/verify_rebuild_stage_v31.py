"""Read-only acceptance of complete V31 rebuild artifacts and crash receipts.

No model is loaded and no metric is recomputed. All declared input/source files
are content-hashed; final cache tensors are compared to committed shards using
CPU mmap and bounded chunks. A missing final summary is recoverable only when
the immutable contract and every committed shard prove the complete output.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / 'video_VACE/divibody_training'
HMR_SHA = '2dcf79638109781d1ae5f5c44fee5f55bc83291c210653feead9b7f04fa6f20e'
HMR_BUILD = 'star-body-hmr2a-raw25-resume-v31'
VACE_BUILD = 'star-body-v31-vace-resume-v1'
FIELDS = {'rgb_tokens': ((192, 1280), 'bfloat16'), 'base_rotmat': ((24, 3, 3), 'float32'),
          'base_betas': ((10,), 'float32'), 'base_camera': ((3,), 'float32')}
MASKS = {'context_edge_repeat_mask': ('context_edge_repeat', 17),
         'target_valid_mask': ('target_valid', 8), 'frame_owner_mask': ('frame_owner', 8),
         'velocity_owner_mask': ('velocity_owner', 7), 'acceleration_owner_mask': ('acceleration_owner', 6)}
META = ('window_uid', 'sequence', 'person_id', 'segment_id', 'target_start',
        'context_frame_ids', 'context_source_frame_ids', 'context_frame_uids', 'context_image_paths',
        'target_frame_ids', 'target_source_frame_ids', 'target_frame_uids', 'target_valid')
UID = re.compile(r'^([^/]+)/person_(\d+)/frame_(\d{5,})$')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + value)))


def canonical(value, ascii=True):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=ascii, allow_nan=False).encode()).hexdigest()


class Audit:
    def __init__(self, torch):
        self.torch, self.hashes, self.bytes_hashed = torch, {}, 0
        self.page_advice = sys.platform.startswith('linux')
        self.libc = ctypes.CDLL(None, use_errno=True) if self.page_advice else None
        if self.libc:
            self.libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
            self.libc.madvise.restype = ctypes.c_int

    def digest(self, path):
        path = Path(path).expanduser().resolve(strict=True)
        require(path.is_file(), 'Expected a regular file: ' + str(path))
        stat = path.stat()
        identity = (str(path), stat.st_size, stat.st_mtime_ns)
        if identity not in self.hashes:
            value = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024**2), b''):
                    value.update(block)
            after = path.stat()
            require((stat.st_size, stat.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                    'File changed while hashing: ' + str(path))
            self.hashes[identity] = value.hexdigest()
            self.bytes_hashed += stat.st_size
        return self.hashes[identity]

    def record(self, record, path=None):
        path = Path(path if path is not None else record['path'])
        require(isinstance(record.get('sha256'), str) and len(record['sha256']) == 64,
                'Full content SHA256 required: ' + str(path))
        if 'bytes' in record:
            require(path.stat().st_size == record['bytes'], 'File size mismatch: ' + str(path))
        require(self.digest(path) == record['sha256'], 'File SHA256 mismatch: ' + str(path))

    def load(self, path):
        # These are trusted project-produced pickle containers, never downloads.
        return self.torch.load(path, map_location='cpu', weights_only=False, mmap=True)

    def release(self, tensor):
        if self.libc and tensor.numel():
            size = os.sysconf('SC_PAGE_SIZE')
            address, length = tensor.data_ptr(), tensor.numel() * tensor.element_size()
            aligned = address - address % size
            length = ((address + length - aligned + size - 1) // size) * size
            require(self.libc.madvise(aligned, length, 4) == 0,
                    'madvise failed; refusing an unbounded mmap working set')

    def tensor(self, value, shape, dtype=None, label='tensor'):
        require(isinstance(value, self.torch.Tensor) and tuple(value.shape) == tuple(shape),
                'Tensor shape mismatch: ' + label)
        require(value.is_contiguous(), 'Non-contiguous tensor is not the builder format: ' + label)
        if dtype is not None:
            require(value.dtype == getattr(self.torch, dtype), 'Tensor dtype mismatch: ' + label)
        for start in range(0, len(value), 16):
            part = value[start:start + 16]
            require(bool(self.torch.isfinite(part).all()), 'Nonfinite tensor: ' + label)
            self.release(part)

    def equal(self, left, right, label):
        require(left.shape == right.shape and left.dtype == right.dtype, 'Shard tensor schema differs: ' + label)
        for start in range(0, len(left), 16):
            a, b = left[start:start + 16], right[start:start + 16]
            require(bool(self.torch.equal(a, b)), 'Final tensor differs from committed shard: ' + label)
            self.release(a)
            self.release(b)


def compare_subset(actual, expected, label):
    if isinstance(expected, dict):
        require(isinstance(actual, dict), 'Expected object at ' + label)
        for key, value in expected.items():
            normalized = key.replace('-', '_') if label.endswith('config') else key
            require(normalized in actual, 'Missing expected key: ' + label + '.' + key)
            compare_subset(actual[normalized], value, label + '.' + normalized)
    else:
        require(actual == expected, 'Planned value differs: ' + label)


def check_expected(args, config, assets, sources):
    if args.expected_config is None:
        return
    expected = read_json(args.expected_config)
    require(isinstance(expected, dict) and bool(expected), 'Expected config must be a nonempty JSON object')
    if any(key in expected for key in ('config', 'assets', 'source_sha256')):
        require(set(expected) <= {'config', 'assets', 'source_sha256'}, 'Unknown expected-config section')
        compare_subset({'config': config, 'assets': assets, 'source_sha256': sources}, expected, 'expected')
    else:
        compare_subset(config, expected, 'config')


def counts(args, windows, frames):
    require(windows > 0 and frames > 0, 'Empty artifact')
    for expected, actual, name in ((args.expected_windows, windows, 'windows'), (args.expected_frames, frames, 'frames')):
        require(expected is None or expected == actual, 'Unexpected ' + name)


def vace_layout(audit, payload, split, development_only=False):
    sig, rows = payload['signature'], payload['metadata']
    require(sig.get('split') == split, 'VACE split mismatch')
    require(sig.get('format') in ('divibody-contiguous-vace-video-features-v5',
                                  'divibody-contiguous-vace-video-features-v6-stride4'), 'Unsupported VACE format')
    require(isinstance(sig.get('development_only'), bool), 'Missing explicit VACE development declaration')
    require(development_only or sig['development_only'] is False, 'Incomplete/development VACE is not a completed formal stage')
    count = len(rows)
    audit.tensor(payload['features'], (count, 5, 6, 6, 1536), 'bfloat16', 'features')
    for key, (_, width) in MASKS.items():
        audit.tensor(payload[key], (count, width), 'bool', key)
    owners, valid_uids, windows = [], set(), []
    for index, row in enumerate(rows):
        windows.append(row['window_uid'])
        valid, owned = payload['target_valid_mask'][index].tolist(), payload['frame_owner_mask'][index].tolist()
        uids, ids, nvalid = row['target_frame_uids'], row['target_frame_ids'], sum(valid)
        require(valid == [True] * nvalid + [False] * (8 - nvalid) and nvalid > 0, 'Invalid target padding')
        require(row['target_valid'] == valid and len(uids) == len(ids) == 8, 'Target identity shape mismatch')
        require(len(row['context_frame_ids']) == len(row['context_frame_uids']) == 17, 'Context length mismatch')
        require(row['context_frame_ids'][4:12] == ids and row['context_frame_uids'][4:12] == uids, 'Target/context identity differs')
        require(len(row['target_source_frame_ids']) == 8 and len(row['context_source_frame_ids']) == 17
                and row['context_source_frame_ids'][4:12] == row['target_source_frame_ids'], 'Source frame identity differs')
        require(ids[:nvalid] == list(range(ids[0], ids[0] + nvalid)) and ids[nvalid:] == [ids[nvalid - 1]] * (8 - nvalid), 'Targets cross a gap')
        for uid, fid, real, own in zip(uids, ids, valid, owned):
            match = UID.fullmatch(uid)
            require(match is not None and match.group(1) == row['sequence'] and int(match.group(2)) == row['person_id']
                    and int(match.group(3)) == fid and (not own or real), 'Invalid owned person/frame UID')
            if real:
                valid_uids.add(uid)
            if own:
                owners.append(uid)
    require(len(set(windows)) == count and len(set(owners)) == len(owners) and set(owners) == valid_uids,
            'Duplicated/missing windows or unique frame ownership')
    feature_only = sig.get('feature_only')
    require(isinstance(feature_only, bool), 'Missing explicit target policy')
    if feature_only:
        require('targets' not in payload, 'Feature-only cache has labels')
    else:
        for key, shape in {'full_pose': (8, 72), 'betas': (8, 10), 'gender_code': (8,), 'cam_pose': (8, 4, 4)}.items():
            audit.tensor(payload['targets'][key], (count, *shape), label='targets/' + key)
    return windows, owners


def vace_source_path(name):
    if name.startswith('wan/'):
        require(bool(os.environ.get('WAN_SOURCE_ROOT')), 'WAN_SOURCE_ROOT required for external Wan source verification')
        return Path(os.environ['WAN_SOURCE_ROOT']) / name
    if name.startswith('vace/'):
        return ROOT / 'video_VACE' / name
    require(Path(name).name == name, 'Unrecognized VACE source key: ' + name)
    return TRAINING / name


def verify_vace(args, audit):
    development = getattr(args, 'development_only', False)
    payload = audit.load(args.output)
    windows, owners = vace_layout(audit, payload, args.split, development)
    counts(args, len(windows), len(owners))
    directory = args.work_dir or Path(str(args.output) + '.shards')
    run = read_json(directory / 'run.json')
    run_sha = canonical(run, ascii=False)
    binding = payload['signature']['resumable_builder']
    require(run['format'] == binding['format'] == VACE_BUILD and binding['run_sha256'] == run_sha, 'VACE run binding differs')
    require(run['window_uids'] == windows and run['config']['split'] == args.split, 'VACE run UID/split binding differs')
    for key, value in (('source_sha256', run['sources']), ('asset_sha256', run['assets']), ('config_sha256', run['config'])):
        require(binding[key] == canonical(value, ascii=False), 'VACE binding digest differs: ' + key)
    plan = {'metadata': [{key: row[key] for key in META} for row in payload['metadata']],
            **{field: payload[key].tolist() for key, (field, _) in MASKS.items()},
            'metric_only_velocity': payload['metric_only_cross_window_motion']['velocity_pairs'],
            'metric_only_acceleration': payload['metric_only_cross_window_motion']['acceleration_triplets']}
    require(canonical(plan, ascii=False) == run['window_plan_sha256'], 'Final VACE plan differs from immutable run')
    stats = run['window_plan']
    require(stats['target_windows'] == len(windows) and stats['owned_person_frames'] == len(owners)
            and stats['total_valid_person_frames'] >= len(owners)
            and stats['development_only'] == payload['signature']['development_only'], 'VACE plan coverage declaration differs')
    if not development:
        require(stats['complete_person_frame_coverage'] is True and stats['development_only'] is False
                and stats['total_valid_person_frames'] == len(owners), 'Incomplete VACE formal plan')
    elif stats['complete_person_frame_coverage'] is False:
        require(stats['development_only'] is True, 'Partial VACE must declare development-only')
    require(payload['signature']['window_plan'] == stats, 'Signature plan differs from run')
    require(bool(run['sources']) and 'cache_continuous_3dpw_vace_features_v31_resume.py' in run['sources'], 'Missing builder source binding')
    for name, sha in run['sources'].items():
        audit.record({'sha256': sha}, vace_source_path(name))
    checkpoint = run['assets']['checkpoint']
    root = Path(checkpoint['directory'])
    require(canonical(checkpoint['files']) == checkpoint['tree_sha256'], 'Checkpoint tree digest differs')
    require({path.relative_to(root).as_posix() for path in root.rglob('*') if path.is_file()}
            == {row['relative_path'] for row in checkpoint['files']}, 'Checkpoint file set differs')
    for record in checkpoint['files']:
        audit.record(record, root / record['relative_path'])
    raw = run['assets']['raw_dataset']
    require(raw['files'] and canonical(raw['files'], ascii=False) == raw['tree_sha256'], 'Raw asset manifest differs')
    require({str(Path(name).resolve()) for row in payload['metadata'] for name in row['context_image_paths']}
            <= {str(Path(record['path']).resolve()) for record in raw['files']}, 'Unbound VACE context image asset')
    for record in raw['files']:
        audit.record(record)
    if run['assets']['text_embedding'] is not None:
        audit.record(run['assets']['text_embedding'])
    require(payload['signature']['checkpoint'] == checkpoint and payload['signature']['encoder_protocol'] == run['encoder_protocol'], 'Encoder assets/protocol differ')
    for row in payload['metadata']:
        seed = int.from_bytes(hashlib.sha256(f"{run['config']['noise_seed_base']}:{row['window_uid']}".encode()).digest()[:8], 'big') % (2**63)
        require(row['noise_seed'] == seed, 'Per-window noise seed differs from bound configuration')
    cursor, shard_count = 0, 0
    for folder in sorted(directory.glob('shard_*')):
        match = re.fullmatch(r'shard_(\d{8})_(\d{8})', folder.name)
        require(match is not None and folder.is_dir() and not folder.is_symlink(), 'Invalid committed shard path')
        start, end = map(int, match.groups())
        require(start == cursor and start < end <= len(windows), 'VACE shard gap/overlap')
        require({path.name for path in folder.iterdir()} == {'record.json', 'data.pt'}, 'Unexpected VACE shard files')
        record = read_json(folder / 'record.json')
        compare_subset(record, {'format': 'star-body-v31-vace-shard-v1', 'run_sha256': run_sha,
                       'start': start, 'end': end, 'window_uids': windows[start:end]}, 'shard')
        audit.record({'sha256': record['data_sha256'], 'bytes': record['data_bytes']}, folder / 'data.pt')
        shard = audit.load(folder / 'data.pt')
        require(shard['run_sha256'] == run_sha and shard['start'] == start and shard['end'] == end, 'Embedded VACE shard identity differs')
        require(shard['metadata'] == payload['metadata'][start:end], 'Final metadata differs from shard')
        audit.equal(payload['features'][start:end], shard['features'], 'features')
        require(set(shard.get('targets', {})) == set(payload.get('targets', {})), 'Shard label policy differs')
        for key in payload.get('targets', {}):
            audit.equal(payload['targets'][key][start:end], shard['targets'][key], key)
        cursor, shard_count = end, shard_count + 1
        del shard
    require(cursor == len(windows), 'Incomplete VACE committed shard coverage')
    digest = audit.digest(args.output)
    summary = args.output.with_suffix('.json')
    if summary.exists():
        compare_subset(read_json(summary), {'status': 'DIVIBODY_CONTINUOUS_3DPW_VACE_V31_RESUME_CACHE_OK',
            'split': args.split, 'development_only': payload['signature']['development_only'], 'windows': len(windows), 'owned_person_frames': len(owners),
            'output_sha256': digest, 'run_sha256': run_sha, 'shards': shard_count}, 'summary')
    check_expected(args, run['config'], run['assets'], run['sources'])
    return dict(windows=len(windows), frames=len(owners), shards=shard_count, binding_sha256=run_sha,
                output_sha256=digest, summary_present=summary.exists(), cache_development_only=payload['signature']['development_only'])


def verify_hmr(args, audit):
    development = getattr(args, 'development_only', False)
    payload = audit.load(args.output)
    sig = payload['signature']
    directory = args.work_dir or Path(str(args.output) + '.parts')
    contract, owner, state = (read_json(directory / name) for name in ('contract.json', 'owner.json', 'state.json'))
    digest = canonical(contract)
    require(contract['format'] == owner['format'] == state['format'] == HMR_BUILD, 'Unsupported HMR build format')
    require(owner['contract_sha256'] == state['contract_sha256'] == sig['build_contract_sha256'] == digest, 'HMR contract binding differs')
    require(owner['build_id'] == state['build_id'], 'HMR state owner differs')
    require(Path(owner['output']).resolve() == args.output.resolve()
            and Path(contract['parameters']['output']).resolve() == args.output.resolve(), 'HMR output ownership differs')
    require(contract['split'] == sig['split'] == args.split and isinstance(sig.get('development_only'), bool), 'HMR split/development mismatch')
    require(development or sig['development_only'] is False, 'Development HMR is not a completed formal stage')
    require(sig['format'] == 'divibody-hmr2a-anchor-sidecar-v8' and sig['builder_revision'] == HMR_BUILD
            and sig['hmr2a_checkpoint_sha256'] == HMR_SHA, 'HMR encoder identity differs')
    require(sig['anchor_revision'] == 'raw-bbox-v11' and sig['raw_bbox_margin'] == contract['parameters']['margin'] == .25,
            'HMR raw25 crop protocol differs')
    require(Path(sig['hmr2a_checkpoint']).resolve() == Path(contract['hmr_checkpoint']['path']).resolve()
            and Path(sig['vace_cache']).resolve() == Path(contract['vace']['path']).resolve()
            and sig['vace_cache_bytes'] == contract['vace']['bytes'], 'HMR asset paths/size differ from contract')
    require(contract['hmr_checkpoint']['sha256'] == HMR_SHA, 'HMR checkpoint is not the fixed audited weight')
    frames, windows = payload['frame_uids'], payload['window_uids']
    counts(args, len(windows), len(frames))
    require(sig['windows'] == len(windows) and sig['unique_frames'] == len(frames), 'HMR declared counts differ')
    require(frames == contract['frame_uids'] and windows == contract['window_uids'], 'HMR UID contract differs')
    require(len(set(frames)) == len(frames) and len(set(windows)) == len(windows), 'Duplicate HMR UIDs')
    require(canonical(payload['window_frame_indices'].tolist()) == contract['window_frame_indices_sha256'], 'HMR frame lookup binding differs')
    for key, (shape, dtype) in FIELDS.items():
        audit.tensor(payload[key], (len(frames), *shape), dtype, key)
    for key in ('vace', 'hmr_checkpoint', 'hmr_model_config'):
        audit.record(contract[key])
    for key in ('source_files', 'hmr_assets', 'annotations', 'images'):
        require(bool(contract[key]), 'Empty HMR asset/source manifest: ' + key)
        for record in contract[key]:
            audit.record(record)
    require(any(Path(record['path']).name == 'cache_hmr2a_rgb_anchor_v31_resume.py'
                for record in contract['source_files']), 'HMR builder source binding is missing')
    vace = audit.load(contract['vace']['path'])
    vace_windows, owners = vace_layout(audit, vace, args.split, development)
    require(windows == vace_windows and set(frames) == set(owners), 'HMR/VACE UID coverage differs')
    lookup = payload['window_frame_indices']
    audit.tensor(lookup, (len(windows), 8), 'int64', 'window_frame_indices')
    require(int(lookup.min()) >= 0 and int(lookup.max()) < len(frames), 'HMR lookup out of bounds')
    for row, indices in zip(vace['metadata'], lookup.tolist()):
        require([frames[index] for index in indices] == row['target_frame_uids'], 'HMR target lookup differs')
    semantic = {'signature': vace['signature'], 'window_uids': windows,
                'target_frame_uids': [row['target_frame_uids'] for row in vace['metadata']]}
    require(sig['vace_semantic_sha256'] == canonical(semantic) and sig['vace_cache_sha256'] == contract['vace']['sha256'], 'HMR VACE source binding differs')
    coverage = contract['coverage']
    require(coverage['development_only'] == sig['development_only']
            and coverage['source_development_only'] == vace['signature']['development_only']
            and coverage['actual_windows'] == len(windows) and coverage['full_windows'] >= len(windows)
            and coverage['owned_frames'] == coverage['valid_unique_frames'] == len(frames)
            and coverage['raw_split_frames'] >= len(frames)
            and coverage['owner_uid_sha256'] == canonical(sorted(frames)), 'HMR subset coverage binding differs')
    if not development or coverage['complete_against_full_raw_split_plan'] is True:
        require(coverage['complete_against_full_raw_split_plan'] is True
                and coverage['full_windows'] == len(windows) and coverage['raw_split_frames'] == len(frames)
                and coverage['full_plan_uid_sha256'] == canonical(windows), 'HMR full coverage contract differs')
    else:
        require(coverage['development_only'] is True, 'Partial HMR must declare development-only')
    if not development:
        require(coverage['development_only'] is False and coverage['source_development_only'] is False, 'Development HMR coverage is not formal')
    require({path.name for path in (directory / 'shards').glob('shard_*.pt')}
            == {row['file'] for row in state['shards']}, 'Missing/unrecorded HMR shard')
    cursor = 0
    for record in state['shards']:
        start, end = record['start'], record['end']
        require(start == cursor and start < end <= len(frames)
                and record['file'] == f'shard_{start:08d}_{end:08d}.pt', 'HMR shard gap/overlap/name mismatch')
        path = directory / 'shards' / record['file']
        require(not path.is_symlink(), 'HMR shard symlink rejected')
        audit.record(record, path)
        shard = audit.load(path)
        require(shard['format'] == HMR_BUILD + '-shard' and shard['contract_sha256'] == digest
                and shard['start'] == start and shard['end'] == end and shard['frame_uids'] == frames[start:end]
                and record['frame_uid_sha256'] == canonical(frames[start:end]), 'Embedded HMR shard identity differs')
        for key in FIELDS:
            audit.equal(payload[key][start:end], shard[key], key)
        cursor = end
        del shard
    require(cursor == len(frames), 'Incomplete HMR committed shard coverage')
    require(state['next_frame'] == len(frames) and state.get('inflight') is None, 'HMR state has an incomplete shard transaction')
    output_sha = audit.digest(args.output)
    summary = args.output.with_suffix('.json')
    if summary.exists():
        compare_subset(read_json(summary), {'status': 'completed', 'real_hmr_encoding': True, 'development_only': sig['development_only'],
            'coverage': coverage, 'sha256': output_sha, 'bytes': args.output.stat().st_size,
            'contract_sha256': digest, 'frames': len(frames), 'windows': len(windows)}, 'summary')
    check_expected(args, contract['parameters'], {key: contract[key] for key in
        ('vace', 'hmr_checkpoint', 'hmr_model_config', 'hmr_assets', 'annotations', 'images')},
        {record['path']: record['sha256'] for record in contract['source_files']})
    return dict(windows=len(windows), frames=len(frames), shards=len(state['shards']), binding_sha256=digest,
                output_sha256=output_sha, summary_present=summary.exists(), cache_development_only=sig['development_only'])


def verify_evaluation(args, audit):
    result = read_json(args.output)
    require(args.split == 'validation', 'Baseline acceptance only supports full validation, never test selection')
    compare_subset(result, {'status': 'complete_matched_project_protocol', 'model_version': 'v28', 'split': 'validation',
        'optimizer_steps': 0, 'model_or_teacher_fitting_performed': False, 'sota_claim': False,
        'checkpoint': {'format': 'ram-iterative-anchor-routed-v28', 'initialization_only': False, 'weights_field': 'model'},
        'config': {'model_version': 'v28', 'split': 'validation', 'development_only': False,
                   'allow_initialization': False, 'hash_large_caches': True},
        'protocol': {'name': 'matched_project_3dpw_owned_sequence_v1', 'mpjpe': 'H36M pelvis aligned',
                     'pa_mpjpe': 'per-frame Procrustes similarity aligned J14', 'pve': 'unaligned SMPL vertex Euclidean error',
                     'distance_unit': 'mm', 'acceleration_unit': 'mm/frame^2',
                     'flip_test': False, 'postprocessing': 'none', 'test_used_for_selection_in_this_run': False}}, 'evaluation')
    require(isinstance(result.get('model_config'), dict) and isinstance(result.get('adaptation_config'), dict), 'Missing actual model configuration')
    metrics = result['metrics']
    require(metrics['frames'] == 10413 and metrics['acceleration_triplets'] == 10226, 'Incomplete validation metrics')
    names = ('mpjpe_mm', 'pa_mpjpe_mm', 'pve_mm', 'acceleration_error_mm_per_frame2')
    for values in [metrics, *metrics['per_sequence'].values()]:
        for key in names:
            require(isinstance(values[key], (int, float)) and math.isfinite(values[key]) and values[key] >= 0, 'Invalid metric: ' + key)
    require(sum(row['frames'] for row in metrics['per_sequence'].values()) == 10413
            and sum(row['acceleration_triplets'] for row in metrics['per_sequence'].values()) == 10226, 'Per-sequence metric counts differ')
    for name in names:
        count_key = 'acceleration_triplets' if name == 'acceleration_error_mm_per_frame2' else 'frames'
        mean = sum(row[name] * row[count_key] for row in metrics['per_sequence'].values()) / metrics[count_key]
        require(math.isclose(mean, metrics[name], rel_tol=1e-6, abs_tol=1e-6), 'Per-sequence metric aggregation differs: ' + name)
    assets = result['assets']
    for key in ('vace_cache', 'hmr_sidecar', 'checkpoint', 'joint_regressor', 'joint_regressor_extra'):
        audit.record(assets[key])
    for gender in ('NEUTRAL', 'MALE', 'FEMALE'):
        audit.record(assets['smpl'][gender])
    checkpoint = audit.load(assets['checkpoint']['path'])
    require(checkpoint.get('format') == 'ram-iterative-anchor-routed-v28' and 'model' in checkpoint, 'Baseline checkpoint format differs')
    for key in ('model_config', 'adaptation_config'):
        require(isinstance(checkpoint.get(key), dict) and checkpoint[key], 'Checkpoint configuration missing: ' + key)
        compare_subset(result[key], checkpoint[key], key)
    del checkpoint
    require(result['source_sha256'] and 'scripts/evaluate_body_v31.py' in result['source_sha256'], 'Missing evaluator source binding')
    for name, sha in result['source_sha256'].items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT), 'Source is outside project: ' + name)
        audit.record({'sha256': sha}, path)
    config = result['config']
    for key in ('vace_cache', 'hmr_sidecar', 'checkpoint', 'joint_regressor'):
        require(Path(config[key]).resolve() == Path(assets[key]['path']).resolve(), 'Config/asset path mismatch: ' + key)
    require(result['seed'] == config['seed'] and result['resources']['amp_dtype'] == config['amp_dtype'], 'Config/resource protocol mismatch')
    require(Path(config['output']).resolve() == args.output.resolve(), 'Evaluation output path differs from config')
    spec = importlib.util.spec_from_file_location('_stage_evaluator', ROOT / 'scripts/evaluate_body_v31.py')
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    vace, sidecar = audit.load(assets['vace_cache']['path']), audit.load(assets['hmr_sidecar']['path'])
    vace_layout(audit, vace, args.split)
    for key, (shape, dtype) in FIELDS.items():
        audit.tensor(sidecar[key], (len(sidecar['frame_uids']), *shape), dtype, key)
    require(result['cache_signatures'] == {'vace': vace['signature'], 'hmr': sidecar['signature']}, 'Recorded cache signature differs')
    semantic = canonical({'signature': vace['signature'], 'window_uids': [row['window_uid'] for row in vace['metadata']],
                          'target_frame_uids': [row['target_frame_uids'] for row in vace['metadata']]})
    require(sidecar['signature']['hmr2a_checkpoint_sha256'] == HMR_SHA
            and sidecar['signature']['vace_cache_sha256'] == assets['vace_cache']['sha256']
            and sidecar['signature']['vace_semantic_sha256'] == semantic
            and result['cache_provenance']['semantic_vace_sha256'] == semantic
            and result['cache_provenance']['verified_vace_file_sha256'] == assets['vace_cache']['sha256'],
            'Evaluation HMR/VACE immutable source binding differs')
    # Reuse only the pure UID/manifest audit; no evaluator main/model/forward call.
    coverage, _ = evaluator.audit_coverage(vace, sidecar, args.split, config['coverage_manifest'], False)
    for key in ('windows', 'frames', 'acceleration_triplets', 'sequences', 'tracks', 'owned_uid_sha256', 'source_time_identity'):
        require(coverage[key] == result['coverage'][key], 'Recomputed coverage differs: ' + key)
    require(coverage['manifest']['sha256'] == result['coverage']['manifest']['sha256'], 'Independent manifest differs')
    require(coverage['manifest']['archive']['sha256'] == result['coverage']['manifest']['archive']['sha256'], 'Independent archive differs')
    counts(args, coverage['windows'], coverage['frames'])
    check_expected(args, config, assets, result['source_sha256'])
    return dict(windows=coverage['windows'], frames=10413, acceleration_triplets=10226,
                output_sha256=audit.digest(args.output), metrics_recomputed=False,
                owned_uid_sha256=coverage['owned_uid_sha256'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('vace', 'hmr', 'evaluation'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'validation'), required=True)
    parser.add_argument('--expected-windows', type=int)
    parser.add_argument('--expected-frames', type=int)
    parser.add_argument('--expected-config', type=Path, help='JSON option subset, or {config,assets,source_sha256} subsets')
    parser.add_argument('--work-dir', type=Path, help='Custom immutable shard directory; defaults to OUTPUT.pt.shards/.parts')
    parser.add_argument('--development-only', action='store_true',
                        help='Cache pilot audit only: permit declared partial plans; return diagnostic_passed, never formal passed')
    args = parser.parse_args(argv)
    started = time.perf_counter()
    try:
        require(not (args.development_only and args.kind == 'evaluation'), 'Development receipt mode is for cache pilots only')
        import torch
        torch.set_num_threads(4)
        args.output = args.output.expanduser().resolve(strict=True)
        audit = Audit(torch)
        detail = {'vace': verify_vace, 'hmr': verify_hmr, 'evaluation': verify_evaluation}[args.kind](args, audit)
        print(json.dumps(dict(status='diagnostic_passed' if args.development_only else 'passed',
            kind=args.kind, split=args.split, output=str(args.output), **detail,
            formal_protocol_satisfied=not args.development_only,
            development_only=args.development_only,
            files_hashed=len(audit.hashes), bytes_hashed=audit.bytes_hashed, mmap_page_release=audit.page_advice,
            seconds=time.perf_counter() - started, read_only=True, optimizer_steps=0), allow_nan=False), flush=True)
        return 0
    except Exception as error:
        print(json.dumps(dict(status='rejected', kind=args.kind, split=args.split, error_type=type(error).__name__,
                              error=str(error), read_only=True), allow_nan=False), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
