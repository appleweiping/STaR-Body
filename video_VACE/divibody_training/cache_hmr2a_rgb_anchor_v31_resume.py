"""Resumable raw25 HMR2a sidecars with bounded shards and file-backed merging.

Copied from cache_hmr2a_rgb_anchor_v11_raw.py before editing. Original files,
VACE inputs, HMR sources and fixed 2dcf... weights are never modified.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import threading
import time
import uuid


BUILD_FORMAT = 'star-body-hmr2a-raw25-resume-v31'
HMR2A_SHA256 = '2dcf79638109781d1ae5f5c44fee5f55bc83291c210653feead9b7f04fa6f20e'
SIDE_FORMAT = 'divibody-hmr2a-anchor-sidecar-v8'
FIELDS = {'rgb_tokens': ((192, 1280), 'bfloat16'), 'base_rotmat': ((24, 3, 3), 'float32'),
          'base_betas': ((10,), 'float32'), 'base_camera': ((3,), 'float32')}
BYTES_PER_FRAME = 192 * 1280 * 2 + (24 * 3 * 3 + 10 + 3) * 4


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


def json_digest(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def atomic_json(path, value, replace=True):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(jsonable(value), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        os.replace(temporary, path)
    else:
        os.link(temporary, path)  # Atomic no-clobber publication, same filesystem.
        temporary.unlink()


def fingerprint(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError('Expected a file for content fingerprint: ' + str(path))
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}


def fingerprint_files(paths):
    return [fingerprint(path) for path in sorted({Path(path).resolve() for path in paths})]


def memory_snapshot(torch=None, device=None):
    result = {'rss_bytes': None, 'peak_rss_bytes': None}
    if sys.platform.startswith('linux'):
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                result['rss_bytes'] = int(line.split()[1]) * 1024
            elif line.startswith('VmHWM:'):
                result['peak_rss_bytes'] = int(line.split()[1]) * 1024
    elif os.name == 'nt':
        class Counters(ctypes.Structure):
            _fields_ = [('cb', ctypes.c_ulong), ('faults', ctypes.c_ulong),
                        ('peak', ctypes.c_size_t), ('working', ctypes.c_size_t),
                        ('rest', ctypes.c_size_t * 6)]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        if ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(kernel.GetCurrentProcess()),
                                                  ctypes.byref(counters), counters.cb):
            result.update(rss_bytes=int(counters.working), peak_rss_bytes=int(counters.peak))
    if torch is not None and device is not None and device.type == 'cuda':
        result['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
        result['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
    return result


def event(work, name, **values):
    row = {'event': name, 'time_unix': time.time(), **values}
    with (work / 'events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(jsonable(row), sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(jsonable(row), sort_keys=True, allow_nan=False), flush=True)


@contextmanager
def build_lock(work):
    """OS lock is released after process death; no unsafe stale-lock deletion."""
    with (work / 'build.lock').open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b'0')
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def bounded_unlink(path, parent):
    path, parent = Path(path), Path(parent).resolve()
    if path.is_symlink() or path.resolve().parent != parent:
        raise ValueError('Refusing cleanup outside the recorded private directory')
    if path.exists():
        path.unlink()


class PageAdvice:
    """Release clean mmap pages on Linux; report unsupported OS behaviour."""
    def __init__(self):
        self.calls = 0
        self.errors = []
        self.supported = sys.platform.startswith('linux')
        self.libc = ctypes.CDLL(None, use_errno=True) if self.supported else None
        if self.supported:
            for name in ('madvise', 'msync'):
                fn = getattr(self.libc, name)
                fn.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
                fn.restype = ctypes.c_int

    def advise(self, tensor, sync=False, sequential=False):
        if not self.supported or tensor.numel() == 0:
            return
        address, length = tensor.data_ptr(), tensor.numel() * tensor.element_size()
        page_size = os.sysconf('SC_PAGE_SIZE')
        aligned = address - address % page_size
        length = ((address + length - aligned + page_size - 1) // page_size) * page_size
        address = aligned
        if sync and self.libc.msync(address, length, 4) != 0:  # MS_SYNC
            self.errors.append({'operation': 'msync', 'errno': ctypes.get_errno()})
            return
        advice = 2 if sequential else 4  # MADV_SEQUENTIAL / MADV_DONTNEED
        if self.libc.madvise(address, length, advice) != 0:
            self.errors.append({'operation': 'madvise', 'errno': ctypes.get_errno()})
        self.calls += 1

    @contextmanager
    def reclaim_during(self, tensors):
        for tensor in tensors:
            self.advise(tensor, sync=True, sequential=True)
        stop = threading.Event()
        def worker():
            while not stop.wait(.1):
                for tensor in tensors:
                    self.advise(tensor)
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join()


def configure_imports():
    root = Path(__file__).resolve().parents[2]
    for path in reversed((root / 'video_VACE', root / 'video_VACE/divibody_training', root)):
        sys.path.insert(0, str(path))
    # Never add vace/models: wan.py would shadow the actual external Wan package.
    sys.dont_write_bytecode = True
    return root


def frame_uid(sample):
    return f'{sample.sequence}/person_{sample.person_index}/frame_{sample.frame_index:05d}'


class ImageFrames:
    """Top-level picklable dataset adapter; only RGB images reach the encoder."""
    def __init__(self, dataset, indices, first, last):
        self.dataset, self.indices, self.first, self.last = dataset, indices, first, last

    def __len__(self):
        return self.last - self.first

    def __getitem__(self, index):
        return self.dataset[self.indices[self.first + index]]['image']


def audit_vace_coverage(torch, vace, raw_dataset, data_root, allow_development):
    """Reconstruct the original full plan and compare UID/owner semantics exactly."""
    from divibody_training import cache_continuous_3dpw_vace_features_v5 as planner
    from divibody_training.evaluate_standard_3dpw import Standard3DPWDataset
    split = vace['signature']['split']
    stride = int(vace['signature'].get('target_stride', 4 if 'stride4' in vace['signature']['format'] else 8))
    if stride not in (4, 8):
        raise ValueError('Only the audited stride4/stride8 VACE protocols are accepted')
    planner.TARGET_STRIDE = stride
    planner.FORMAT = planner.STRIDE4_FORMAT if stride == 4 else 'divibody-contiguous-vace-video-features-v5'
    standard = Standard3DPWDataset(root=data_root, split=split, image_size=96,
        frame_stride=1, max_samples=None, crop_to_person=True)
    plan = planner.build_v5_window_plan(planner.frame_identities_from_samples(standard.samples))
    planner.validate_v5_window_plan(plan)
    expected = planner.v5_metadata(plan)
    expected_by_uid = {record['window_uid']: index for index, record in enumerate(expected)}
    actual_uids = [record['window_uid'] for record in vace['metadata']]
    if len(actual_uids) != len(set(actual_uids)):
        raise ValueError('VACE window UIDs are duplicated')
    source_development = bool(vace['signature'].get('development_only'))
    if source_development and not allow_development:
        raise ValueError('Development VACE needs explicit --allow-development-cache')
    masks = {'context_edge_repeat_mask': 'context_edge_repeat', 'target_valid_mask': 'target_valid',
             'frame_owner_mask': 'frame_owner', 'velocity_owner_mask': 'velocity_owner',
             'acceleration_owner_mask': 'acceleration_owner'}
    for row, record in enumerate(vace['metadata']):
        if record['window_uid'] not in expected_by_uid:
            raise ValueError('VACE UID is outside the independently rebuilt raw split plan')
        index = expected_by_uid[record['window_uid']]
        for key, value in expected[index].items():
            if key == 'context_image_paths':
                continue  # Absolute root relocation is allowed; UID and content are bound separately.
            if jsonable(record.get(key)) != jsonable(value):
                raise ValueError(f'VACE metadata mismatch at row {row}: {key}')
        for name, attribute in masks.items():
            mask = vace.get(name)
            if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or tuple(mask.shape) != (len(actual_uids), len(getattr(plan.windows[index], attribute))):
                raise ValueError('Invalid VACE mask: ' + name)
            if mask[row].tolist() != list(getattr(plan.windows[index], attribute)):
                raise ValueError(f'VACE ownership differs from original plan at row {row}: {name}')
    owner_uids = [uid for row, record in enumerate(vace['metadata'])
                  for uid, owned in zip(record['target_frame_uids'], vace['frame_owner_mask'][row].tolist()) if owned]
    valid_uids = {uid for row, record in enumerate(vace['metadata'])
                  for uid, valid in zip(record['target_frame_uids'], vace['target_valid_mask'][row].tolist()) if valid}
    if len(owner_uids) != len(set(owner_uids)) or not owner_uids:
        raise ValueError('VACE owners are duplicated or empty')
    raw_uids = {frame_uid(sample) for sample in raw_dataset.samples}
    if not valid_uids <= raw_uids:
        raise ValueError('VACE references identities missing from strict raw RGB data')
    complete = actual_uids == [row['window_uid'] for row in expected] and set(owner_uids) == valid_uids == raw_uids
    if not complete and not allow_development:
        raise ValueError('Formal HMR construction requires the full UID plan and unique complete owner coverage')
    if complete:
        metric_only = {'velocity_pairs': planner._metric_only_records(plan.metric_only_velocity_keys),
                       'acceleration_triplets': planner._metric_only_records(plan.metric_only_acceleration_keys)}
        actual_metric = vace.get('metric_only_cross_window_motion', {})
        if jsonable(actual_metric) != jsonable(metric_only):
            raise ValueError('VACE boundary metric-only relations differ from the original complete plan')
    for name in ('features',):
        for start in range(0, len(vace[name]), 32):
            if not torch.isfinite(vace[name][start:start + 32]).all():
                raise ValueError('Nonfinite VACE features')
    for name, tensor in vace['targets'].items():
        if tensor.is_floating_point():
            for start in range(0, len(tensor), 128):
                if not torch.isfinite(tensor[start:start + 128]).all():
                    raise ValueError('Nonfinite VACE target: ' + name)
    return {'complete_against_full_raw_split_plan': complete, 'source_development_only': source_development,
        'development_only': bool(allow_development or source_development or not complete),
        'target_stride': stride, 'actual_windows': len(actual_uids), 'full_windows': len(expected),
        'owned_frames': len(owner_uids), 'valid_unique_frames': len(valid_uids),
        'raw_split_frames': len(raw_uids), 'original_plan_stats': jsonable(plan.stats),
        'owner_uid_sha256': json_digest(sorted(owner_uids)), 'full_plan_uid_sha256': json_digest([row['window_uid'] for row in expected])}


def source_files(root, hmr_root):
    paths = {Path(__file__).resolve(), Path(__file__).with_name('cache_hmr2a_rgb_anchor_v11_raw.py').resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if not isinstance(filename, (str, os.PathLike)):
            continue
        path = Path(filename)
        if path.is_file() and path.suffix == '.py':
            path = path.resolve()
            if path.is_relative_to(root):
                paths.add(path)
    paths.update((hmr_root / 'hmr2').rglob('*.py'))
    if not any(path.is_relative_to(hmr_root) for path in paths):
        raise ValueError('No HMR2 source package found under --hmr-source-root')
    return paths


def make_contract(args, root, torch, vace, plan, dataset, coverage):
    lookup = {frame_uid(sample): sample for sample in dataset.samples}
    if len(lookup) != len(dataset.samples):
        raise ValueError('Raw dataset has duplicated person-frame identities')
    if any(uid not in lookup for uid in plan.frame_uids):
        raise ValueError('Missing raw-bbox identities required by VACE')
    annotations = sorted(dataset.split_directory.glob('*.pkl'))
    images = {Path(lookup[uid].image_path) for uid in plan.frame_uids}
    hmr_assets = list((args.hmr_data_root / 'smpl').rglob('*'))
    hmr_assets = [path for path in hmr_assets if path.is_file()]
    hmr_assets.extend([args.hmr_data_root / 'SMPL_to_J19.pkl', args.hmr_data_root / 'smpl_mean_params.npz'])
    if not hmr_assets or not (args.hmr_data_root / 'smpl').is_dir():
        raise ValueError('Missing HMR SMPL model assets')
    checkpoint = fingerprint(args.hmr2a_checkpoint)
    if checkpoint['sha256'] != HMR2A_SHA256:
        raise ValueError('Only the fixed audited HMR2a checkpoint SHA256 is accepted')
    return {'format': BUILD_FORMAT, 'parameters': {
        key: jsonable(value) for key, value in vars(args).items()
        if key not in ('resume', 'max_new_frames', 'merge_only')},
        'split': vace['signature']['split'], 'coverage': coverage,
        'vace': fingerprint(args.vace_cache), 'hmr_checkpoint': checkpoint,
        'hmr_model_config': fingerprint(args.hmr_model_config),
        'hmr_assets': fingerprint_files(hmr_assets),
        'source_files': fingerprint_files(source_files(root, args.hmr_source_root)),
        'annotations': fingerprint_files(annotations), 'images': fingerprint_files(images),
        'frame_uids': plan.frame_uids, 'window_uids': plan.window_uids,
        'window_frame_indices_sha256': json_digest(plan.window_frame_indices.tolist()),
        'environment': {'python': sys.version, 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'platform': platform.platform()}}


def check_contract_files(contract):
    records = [contract['vace'], contract['hmr_checkpoint'], contract['hmr_model_config']]
    for key in ('hmr_assets', 'source_files', 'annotations', 'images'):
        records.extend(contract[key])
    for previous in records:
        if fingerprint(previous['path']) != previous:
            raise ValueError('Immutable input/source changed during construction: ' + previous['path'])


def tensor_schema_check(torch, payload, size):
    for key, (shape, dtype) in FIELDS.items():
        tensor = payload.get(key)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (size, *shape) or tensor.dtype != getattr(torch, dtype):
            raise ValueError('Bad shard tensor schema: ' + key)
        for start in range(0, size, 32):
            if not torch.isfinite(tensor[start:start + 32]).all():
                raise ValueError('Nonfinite shard tensor: ' + key)


def validate_shard(torch, path, contract_sha, frame_uids, expected_start=None):
    payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    start, end = payload.get('start'), payload.get('end')
    if payload.get('format') != BUILD_FORMAT + '-shard' or payload.get('contract_sha256') != contract_sha:
        raise ValueError('Shard contract differs: ' + str(path))
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(frame_uids):
        raise ValueError('Invalid shard interval')
    if expected_start is not None and start != expected_start:
        raise ValueError('Shard gap, overlap or out-of-order interval')
    if payload.get('frame_uids') != frame_uids[start:end]:
        raise ValueError('Shard frame identity/order mismatch')
    if path.name != f'shard_{start:08d}_{end:08d}.pt':
        raise ValueError('Shard filename disagrees with its interval')
    tensor_schema_check(torch, payload, end - start)
    return payload


def recover_shards(torch, work, state, contract_sha, frame_uids):
    folder = work / 'shards'
    listed = {row['file']: row for row in state['shards']}
    inflight = state.get('inflight')
    if inflight:
        partial = folder / inflight['partial']
        if partial.name != inflight['partial']:
            raise ValueError('Invalid recorded temporary shard path')
        bounded_unlink(partial, folder)
    result, next_frame = [], 0
    for path in sorted(folder.glob('shard_*.pt')):
        if path.is_symlink():
            raise ValueError('Shard symlinks are not accepted')
        payload = validate_shard(torch, path, contract_sha, frame_uids, next_frame)
        row = {'file': path.name, 'start': payload['start'], 'end': payload['end'],
               'sha256': sha256_file(path), 'bytes': path.stat().st_size,
               'frame_uid_sha256': json_digest(payload['frame_uids'])}
        if path.name in listed and row != listed[path.name]:
            raise ValueError('Previously committed shard contents changed: ' + path.name)
        if path.name not in listed and (not inflight or path.name != inflight['final']):
            raise ValueError('Unrecorded shard cannot be adopted: ' + path.name)
        result.append(row)
        next_frame = payload['end']
        del payload
    if set(listed) - {row['file'] for row in result}:
        raise ValueError('Previously committed shard is missing')
    state.update(shards=result, next_frame=next_frame, inflight=None)
    atomic_json(work / 'state.json', state)
    return next_frame


def save_shard(torch, work, state, contract_sha, start, end, frame_uids, tensors):
    folder = work / 'shards'
    final = folder / f'shard_{start:08d}_{end:08d}.pt'
    partial = folder / (final.name + '.' + state['build_id'] + '.partial')
    if final.exists() or partial.exists():
        raise FileExistsError('Refusing to overwrite a shard or unexamined temporary')
    state['inflight'] = {'partial': partial.name, 'final': final.name}
    atomic_json(work / 'state.json', state)
    payload = {'format': BUILD_FORMAT + '-shard', 'contract_sha256': contract_sha,
        'start': start, 'end': end, 'frame_uids': frame_uids[start:end], **tensors}
    tensor_schema_check(torch, payload, end - start)
    torch.save(payload, str(partial))
    with partial.open('r+b') as stream:
        os.fsync(stream.fileno())
    os.link(partial, final)
    partial.unlink()
    row = {'file': final.name, 'start': start, 'end': end, 'sha256': sha256_file(final),
           'bytes': final.stat().st_size, 'frame_uid_sha256': json_digest(frame_uids[start:end])}
    state['shards'].append(row)
    state.update(next_frame=end, inflight=None)
    atomic_json(work / 'state.json', state)


def cleanup_previous_merge(work, state, output):
    attempt = state.get('merge_attempt')
    if not attempt:
        return
    directory = Path(attempt['directory'])
    temporary = Path(attempt['output_temporary'])
    if (directory.is_symlink() or directory.resolve().parent != work.resolve()
            or not directory.name.startswith('merge_' + state['build_id'] + '_')
            or temporary.resolve().parent != output.parent.resolve()
            or not temporary.name.startswith(output.name + '.' + state['build_id'] + '.')
            or not temporary.name.endswith('.partial')):
        raise ValueError('Untrusted old merge paths; refusing cleanup')
    if directory.exists():
        allowed = {key + '.mmap' for key in FIELDS}
        if any(path.name not in allowed for path in directory.iterdir()):
            raise ValueError('Unexpected file in private merge directory; refusing cleanup')
        for name in allowed:
            bounded_unlink(directory / name, directory)
        directory.rmdir()
    bounded_unlink(temporary, output.parent)
    state['merge_attempt'] = None
    atomic_json(work / 'state.json', state)
    event(work, 'old_private_merge_files_cleaned')


def merge_shards(torch, work, state, contract_sha, frame_uids, base_payload, output, validator):
    """Build ordinary sidecar tensors using mapped storage, never full empty(N)."""
    count = len(frame_uids)
    expected_bytes = count * BYTES_PER_FRAME
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('Refusing to overwrite final output')
    if shutil.disk_usage(work).free < expected_bytes + 256 * 1024**2:
        raise OSError('Insufficient disk space for merge mmap files')
    same_disk = work.stat().st_dev == output.parent.stat().st_dev
    needed = expected_bytes * (2 if same_disk else 1) + 256 * 1024**2
    if shutil.disk_usage(output.parent).free < needed:
        raise OSError('Insufficient disk space for mapped merge and final sidecar')
    merge_dir = work / ('merge_' + state['build_id'] + '_' + uuid.uuid4().hex)
    merge_dir.mkdir(exist_ok=False)
    temporary = output.with_name(output.name + '.' + state['build_id'] + '.' + uuid.uuid4().hex + '.partial')
    mmap_paths, mapped, payload, disk = [], {}, None, None
    advice = PageAdvice()
    state['merge_attempt'] = {'directory': str(merge_dir), 'output_temporary': str(temporary)}
    atomic_json(work / 'state.json', state)
    cleanup = []
    try:
        for key, (shape, dtype) in FIELDS.items():
            path = merge_dir / (key + '.mmap')
            with path.open('xb'):
                pass
            mmap_paths.append(path)
            mapped[key] = torch.from_file(str(path), shared=True, size=count * math.prod(shape),
                                           dtype=getattr(torch, dtype)).reshape(count, *shape)
        for record in state['shards']:
            path = work / 'shards' / record['file']
            if sha256_file(path) != record['sha256']:
                raise ValueError('Shard changed before merging')
            shard = validate_shard(torch, path, contract_sha, frame_uids, record['start'])
            for key in FIELDS:
                mapped[key][record['start']:record['end']].copy_(shard[key])
                advice.advise(mapped[key], sync=True)
            del shard
            event(work, 'merge_copy', frames=record['end'], total=count, memory=memory_snapshot())
        payload = dict(base_payload, **mapped)
        with advice.reclaim_during(list(mapped.values())):
            validator(payload)
            # A filesystem path uses PyTorch's native storage writer. A Python
            # file-like write callback can materialize huge byte strings.
            torch.save(payload, str(temporary))
        with temporary.open('r+b') as stream:
            os.fsync(stream.fileno())
        disk = torch.load(temporary, map_location='cpu', weights_only=False, mmap=True)
        with advice.reclaim_during([disk[key] for key in FIELDS]):
            validator(disk)
        disk.clear()
        disk = None
        digest = sha256_file(temporary)
        os.link(temporary, output)  # Fail atomically if another process created output.
        temporary.unlink()
        return {'output': str(output), 'sha256': digest, 'bytes': output.stat().st_size,
                'file_backed_merge': True, 'madvise_supported': advice.supported,
                'page_advice_calls': advice.calls, 'page_advice_errors': advice.errors,
                'paging_limit': 'OS-managed clean file pages; RSS is observed, not a guaranteed hard cap',
                'memory': memory_snapshot(), 'cleanup': cleanup}
    finally:
        if payload is not None:
            payload.clear()
        if disk is not None:
            disk.clear()
            disk = None
        mapped.clear()
        gc.collect()
        for path in mmap_paths:
            try:
                bounded_unlink(path, merge_dir)
                cleanup.append({'file': str(path), 'removed': True})
            except OSError as error:
                cleanup.append({'file': str(path), 'removed': False, 'reason': str(error)})
        if temporary.exists():
            try:
                bounded_unlink(temporary, output.parent)
            except OSError as error:
                cleanup.append({'file': str(temporary), 'removed': False, 'reason': str(error)})
        try:
            merge_dir.rmdir()
        except OSError:
            pass


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('vace-cache', 'data-root', 'hmr-source-root', 'hmr2a-checkpoint',
                 'hmr-model-config', 'hmr-data-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, help='Private resume directory; default OUTPUT.pt.parts')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--merge-only', action='store_true')
    parser.add_argument('--max-new-frames', type=int, default=0, help='Save and pause after this many NEW frames; 0 finishes all')
    parser.add_argument('--shard-frames', type=int, default=128, choices=(128, 256))
    parser.add_argument('--margin', type=float, default=.25, help='Fixed raw25 protocol; only 0.25 is accepted')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cuda-memory-limit-gib', type=float, default=8.)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--allow-development-cache', action='store_true', help='Diagnostic only; output remains development_only=True')
    args = parser.parse_args(argv)
    if args.margin != .25 or args.batch_size < 1 or args.num_workers < 0 or args.cpu_threads < 1 or args.max_new_frames < 0:
        parser.error('Require raw25, positive batch/threads, nonnegative workers and max-new-frames')
    if not math.isfinite(args.cuda_memory_limit_gib) or args.cuda_memory_limit_gib <= 0 or not 0 <= args.seed < 2**32:
        parser.error('Invalid memory budget or random seed')
    if args.merge_only and (not args.resume or args.max_new_frames):
        parser.error('--merge-only requires --resume and no --max-new-frames')
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    args.work_dir = args.work_dir or args.output.with_name(args.output.name + '.parts')
    if args.output.suffix.lower() != '.pt':
        parser.error('--output must end in .pt')
    if args.output.exists() or args.output.with_suffix('.json').exists():
        parser.error('Output or completion report already exists; never overwrite')
    if args.work_dir.exists() and not args.resume:
        parser.error('Work directory exists; use explicit --resume')
    if args.resume and not (args.work_dir / 'owner.json').is_file():
        parser.error('Resume directory lacks a builder ownership marker')
    return args


def main():
    args = parse_args()
    root = configure_imports()
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from divibody_training.audit_hmr2a_crop_protocol_v11 import RawBBoxHMR3DPWDataset
    from divibody_training.cache_hmr2a_rgb_anchor_v8 import build_anchor_plan, semantic_vace_digest, validate_sidecar, validate_vace_source
    from divibody_training.hmr2a_anchor import HMR2AAnchorExtractor
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Only CPU and CUDA are supported')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable')
        torch.cuda.set_device(device)
        fraction = min(1., args.cuda_memory_limit_gib * 1024**3 / torch.cuda.get_device_properties(device).total_memory)
        torch.cuda.set_per_process_memory_fraction(fraction, device)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
    work = args.work_dir
    if not args.resume:
        work.mkdir(parents=True, exist_ok=False)
    with build_lock(work):
        started = time.perf_counter()
        state = None
        try:
            event(work, 'audit_start', real_encoding=True, diagnostic_only=args.allow_development_cache)
            vace = torch.load(args.vace_cache, map_location='cpu', weights_only=False, mmap=True)
            validate_vace_source(vace)
            if vace['signature']['split'] not in ('train', 'validation'):
                raise ValueError('Only train and validation input caches are accepted; never test')
            plan = build_anchor_plan(vace['metadata'], vace['signature']['split'], 0)
            dataset = RawBBoxHMR3DPWDataset(root=args.data_root, split=plan.split, image_size=256,
                frame_stride=1, max_samples=None, crop_to_person=True, crop_margin=.25)
            coverage = audit_vace_coverage(torch, vace, dataset, args.data_root, args.allow_development_cache)
            contract = make_contract(args, root, torch, vace, plan, dataset, coverage)
            contract_sha = json_digest(contract)
            if args.resume:
                owner = read_json(work / 'owner.json')
                old_contract = read_json(work / 'contract.json')
                if owner.get('format') != BUILD_FORMAT or owner.get('output') != str(args.output):
                    raise ValueError('Resume directory belongs to another build/output')
                if old_contract != jsonable(contract) or owner.get('contract_sha256') != contract_sha:
                    raise ValueError('Resume refused: parameters, source, weights, config, VACE or raw assets changed')
                state = read_json(work / 'state.json')
                if state.get('build_id') != owner['build_id'] or state.get('contract_sha256') != contract_sha:
                    raise ValueError('Resume state ownership/contract mismatch')
            else:
                build_id = uuid.uuid4().hex
                (work / 'shards').mkdir(exist_ok=False)
                atomic_json(work / 'contract.json', contract, replace=False)
                atomic_json(work / 'owner.json', {'format': BUILD_FORMAT, 'build_id': build_id,
                    'output': str(args.output), 'contract_sha256': contract_sha}, replace=False)
                state = {'format': BUILD_FORMAT, 'build_id': build_id, 'contract_sha256': contract_sha,
                         'next_frame': 0, 'shards': [], 'inflight': None, 'status': 'encoding'}
                atomic_json(work / 'state.json', state, replace=False)
            offset = recover_shards(torch, work, state, contract_sha, plan.frame_uids)
            cleanup_previous_merge(work, state, args.output)
            count = len(plan.frame_uids)
            event(work, 'resume_verified', frames=offset, total=count, contract_sha256=contract_sha,
                  coverage=coverage, memory=memory_snapshot(torch, device))
            stop = min(count, offset + args.max_new_frames) if args.max_new_frames else count
            if args.merge_only and offset != count:
                raise ValueError('--merge-only requires every unique frame shard')
            if offset < stop and not args.merge_only:
                lookup = {frame_uid(sample): index for index, sample in enumerate(dataset.samples)}
                indices = [lookup[uid] for uid in plan.frame_uids]
                if shutil.disk_usage(work).free < (stop - offset) * BYTES_PER_FRAME + 256 * 1024**2:
                    raise OSError('Insufficient disk space for requested new frame shards')
                # The exact checkpoint hash was already verified in make_contract.
                from divibody_training.evaluate_standard_3dpw import enable_legacy_chumpy_compatibility
                enable_legacy_chumpy_compatibility()
                with HMR2AAnchorExtractor(source_root=args.hmr_source_root, checkpoint=args.hmr2a_checkpoint,
                        model_config=args.hmr_model_config, hmr2_data_root=args.hmr_data_root,
                        device=device, verify_sha256=False) as extractor:
                    while offset < stop:
                        end = min(stop, offset + args.shard_frames)
                        tensors = {key: torch.empty((end - offset, *shape), dtype=getattr(torch, dtype))
                                   for key, (shape, dtype) in FIELDS.items()}
                        loader = DataLoader(ImageFrames(dataset, indices, offset, end), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == 'cuda',
                            persistent_workers=False, generator=torch.Generator().manual_seed(args.seed))
                        position = 0
                        for images in loader:
                            encoded = extractor.encode(images)
                            size = len(images)
                            for key, (_, dtype) in FIELDS.items():
                                tensors[key][position:position + size].copy_(encoded[key].detach().to('cpu', getattr(torch, dtype)))
                            position += size
                            del encoded
                        if position != end - offset:
                            raise RuntimeError('Incomplete frame shard extraction')
                        save_shard(torch, work, state, contract_sha, offset, end, plan.frame_uids, tensors)
                        del tensors, loader
                        offset = end
                        event(work, 'shard_committed', frames=offset, total=count, memory=memory_snapshot(torch, device))
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            if offset < count:
                state.update(status='paused', elapsed_last_run_seconds=time.perf_counter() - started,
                             memory=memory_snapshot(torch, device))
                atomic_json(work / 'state.json', state)
                event(work, 'paused', frames=offset, total=count, output_created=False)
                return
            event(work, 'premerge_revalidate_inputs')
            check_contract_files(contract)
            signature = {'format': SIDE_FORMAT, 'split': plan.split, 'development_only': coverage['development_only'],
                'anchor_revision': 'raw-bbox-v11', 'builder_revision': BUILD_FORMAT, 'raw_bbox_margin': .25,
                'unique_frames': count, 'windows': len(plan.window_uids),
                'hmr2a_checkpoint': str(args.hmr2a_checkpoint), 'hmr2a_checkpoint_sha256': HMR2A_SHA256,
                'vace_cache': str(args.vace_cache), 'vace_cache_bytes': contract['vace']['bytes'],
                'vace_cache_sha256': contract['vace']['sha256'], 'vace_semantic_sha256': semantic_vace_digest(vace, len(plan.window_uids)),
                'crop_protocol': 'raw rectangular 3DPW poses2d bbox + 0.25 per-side margin; single HMR2 aspect expansion; no pre-square or boundary clamp',
                'rgb_token_dtype': 'bfloat16', 'anchor_dtype': 'float32', 'build_contract_sha256': contract_sha}
            base_payload = {'signature': signature, 'frame_uids': plan.frame_uids, 'window_uids': plan.window_uids,
                            'window_frame_indices': plan.window_frame_indices}
            state['status'] = 'merging'
            atomic_json(work / 'state.json', state)
            merged = merge_shards(torch, work, state, contract_sha, plan.frame_uids, base_payload,
                                  args.output, lambda payload: validate_sidecar(payload, vace))
            result = {'status': 'completed', 'real_hmr_encoding': True, 'benchmark_metrics_computed': False,
                'development_only': coverage['development_only'], 'coverage': coverage, **merged,
                'contract_sha256': contract_sha, 'seed': args.seed, 'frames': count,
                'windows': len(plan.window_uids), 'elapsed_last_run_seconds': time.perf_counter() - started,
                'memory': memory_snapshot(torch, device), 'work_dir': str(work), 'shards_retained_for_resume_audit': True}
            atomic_json(args.output.with_suffix('.json'), result, replace=False)
            state.update(status='completed', output=result)
            atomic_json(work / 'state.json', state)
            event(work, 'completed', **result)
        except BaseException as error:
            if state is not None:
                state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                             last_error={'type': type(error).__name__, 'message': str(error)})
                atomic_json(work / 'state.json', state)
            event(work, 'failed', error_type=type(error).__name__, error=str(error),
                  elapsed_seconds=time.perf_counter() - started)
            raise


if __name__ == '__main__':
    main()
