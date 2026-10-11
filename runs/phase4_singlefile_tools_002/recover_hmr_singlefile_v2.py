"""CPU-only, single-file storage recovery; immutable HMR encoding inputs.

The frozen V32 source is retained separately before this additive helper.
No GPU/model loading, old-state mutation, partial deletion or automatic retry.
Linux Torch 2.5.1 is mandatory for real writes.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib.util
import json
import mmap
import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import sys
import time
import zipfile
import zlib

COPY_SHA = 'f0acd326968e88c4bd5cc4ae5bccc0b15441d08b47387ec76eac1377073c62eb'
COPY_NAME = 'cache_hmr2a_rgb_anchor_v32_scratch.copy.py'
FIELDS = {'rgb_tokens': ((192, 1280), 'bfloat16'), 'base_rotmat': ((24, 3, 3), 'float32'),
          'base_betas': ((10,), 'float32'), 'base_camera': ((3,), 'float32')}
GIB = 1024 ** 3
CONTROLLER_LOCK = '/media/lenovo/data2/mjx/STaR-Body_V31_20261008/runs/full_rebuild_pipeline_001/state/controller.lock'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def record(path):
    p = Path(path)
    s = p.stat()
    require(stat.S_ISREG(s.st_mode) and not p.is_symlink(), 'Expected regular non-symlink file: ' + str(p))
    sha = digest(p)
    after = p.stat()
    require((s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'File changed during hash: ' + str(p))
    return {'path': str(p.resolve()), 'bytes': s.st_size, 'sha256': sha,
            'device': s.st_dev, 'inode': s.st_ino, 'mtime_ns': s.st_mtime_ns}


def write_new_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def checked_directory(path):
    p = Path(path).absolute()
    require(all(not item.is_symlink() for item in (p, *p.parents)), 'Symlink directory rejected: ' + str(p))
    require(p.is_dir(), 'Missing directory: ' + str(p))
    return p.resolve(strict=True)


def copied_module(path):
    require(digest(path) == COPY_SHA, 'Frozen V32 copy differs')
    spec = importlib.util.spec_from_file_location('frozen_v32_recovery_copy', path)
    mod = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(mod)
    require(mod.FIELDS == FIELDS, 'Copied source schema differs')
    return mod


@contextmanager
def readonly_build_lock(parts):
    import fcntl
    path = parts / 'build.lock'
    require(path.is_file() and not path.is_symlink(), 'Original build lock missing')
    with path.open('rb') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def check_torch(torch):
    require(sys.platform.startswith('linux'), 'Real recovery requires Linux MAP_SHARED')
    require(torch.__version__.split('+')[0] == '2.5.1', 'This tool is gated on actual Torch 2.5.1')
    torch.set_num_threads(4)


def inherited_controller_lock():
    """Validate the existing runner lock without opening a second description."""
    import fcntl
    value = os.environ.get('STAR_BODY_CONTROLLER_LOCK_FD')
    require(value is not None and value.isdecimal(), 'Formal recovery requires inherited controller lock FD')
    fd = int(value)
    require(fd >= 3, 'Controller lock FD must be >= 3')
    path = Path(CONTROLLER_LOCK).resolve(strict=True)
    require(os.readlink('/proc/self/fd/' + str(fd)) == str(path), 'Inherited descriptor path differs')
    descriptor, expected = os.fstat(fd), path.stat()
    require(stat.S_ISREG(descriptor.st_mode) and
            (descriptor.st_dev, descriptor.st_ino) == (expected.st_dev, expected.st_ino), 'Inherited lock identity differs')
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd, {'path': str(path), 'fd': fd, 'device': descriptor.st_dev, 'inode': descriptor.st_ino}


def crc_patch_plan(path):
    """Parse bounded ZIP structures, including ZIP64/data descriptors, before edits."""
    with zipfile.ZipFile(path) as archive:
        infos, central = archive.infolist(), archive.start_dir
    require(infos and len({i.filename for i in infos}) == len(infos), 'Empty/duplicate ZIP entries')
    offsets = sorted(info.header_offset for info in infos)
    require(len(set(offsets)) == len(offsets) and all(0 <= offset < central for offset in offsets),
            'Duplicate/out-of-range local ZIP offsets')
    boundaries = dict(zip(offsets, offsets[1:] + [central]))
    patches, ranges, central_cursor = [], [], central
    with Path(path).open('rb') as f:
        for info in infos:
            require(info.compress_type == zipfile.ZIP_STORED and info.compress_size == info.file_size,
                    'Only uncompressed tensor ZIP entries are supported')
            require(not (info.flag_bits & 1), 'Encrypted ZIP entry rejected')
            f.seek(central_cursor)
            header = f.read(46)
            require(len(header) == 46 and header[:4] == b'PK\x01\x02', 'Bad central ZIP header')
            n, e, c = struct.unpack_from('<HHH', header, 28)
            name = f.read(n)
            require(name.decode('utf-8' if info.flag_bits & 0x800 else 'cp437') == info.filename,
                    'Central entry order differs')
            central_crc = central_cursor + 16
            central_cursor += 46 + n + e + c
            f.seek(info.header_offset)
            local = f.read(30)
            require(len(local) == 30 and local[:4] == b'PK\x03\x04', 'Bad local ZIP header')
            flags, method = struct.unpack_from('<HH', local, 6)
            require(flags == info.flag_bits and method == 0, 'Local ZIP flags differ')
            ln, le = struct.unpack_from('<HH', local, 26)
            local_name, extra = f.read(ln), f.read(le)
            require(local_name == name, 'Local filename differs')
            sizes = struct.unpack_from('<II', local, 18)
            zip64 = 0xffffffff in sizes
            if zip64:
                pos, found = 0, False
                while pos < len(extra):
                    require(pos + 4 <= len(extra), 'Truncated ZIP extra')
                    tag, length = struct.unpack_from('<HH', extra, pos)
                    body = extra[pos + 4:pos + 4 + length]
                    require(len(body) == length, 'Truncated ZIP64 extra')
                    if tag == 1:
                        require(length >= 16, 'Missing ZIP64 sizes')
                        us, cs = struct.unpack_from('<QQ', body)
                        require((us, cs) == (info.file_size, info.compress_size) or
                                (flags & 8 and (us, cs) == (0, 0)), 'ZIP64 size mismatch')
                        found = True
                    pos += 4 + length
                require(found, 'ZIP64 extra missing')
            elif not flags & 8:
                require(sizes == (info.compress_size, info.file_size), 'Local size mismatch')
            start = info.header_offset + 30 + ln + le
            end = start + info.file_size
            boundary = boundaries[info.header_offset]
            require(info.header_offset >= 0 and start <= end <= boundary, 'ZIP data outside record')
            positions = [central_crc, info.header_offset + 14]
            descriptor_end = end
            if flags & 8:
                # PyTorch may emit a ZIP64 descriptor for a small tensor whose
                # local-header offset exceeds 4 GiB, while local sizes are zero
                # and extract_version is zero. Infer neither from those fields.
                # Exact physical boundaries disambiguate zero-size entries and
                # a CRC value that happens to equal the optional signature.
                f.seek(end)
                length = boundary - end
                require(length in (12, 16, 20, 24), 'Descriptor has no exact supported boundary')
                raw = f.read(length)
                require(len(raw) == length, 'Truncated descriptor')
                candidates = []
                for signed in (False, True):
                    prefix = 4 if signed else 0
                    if signed and raw[:4] != b'PK\x07\x08':
                        continue
                    for width, fmt in ((8, '<II'), (16, '<QQ')):
                        if prefix + 4 + width != length:
                            continue
                        dsizes = struct.unpack(fmt, raw[prefix + 4:])
                        if dsizes == (info.compress_size, info.file_size):
                            candidates.append(end + prefix)
                require(len(candidates) == 1, 'Descriptor width/signature is invalid or ambiguous')
                crc_offset = candidates[0]
                descriptor_end = boundary
                positions.append(crc_offset)
            ranges.append((info.header_offset, descriptor_end))
            f.seek(start)
            remaining, crc = info.file_size, 0
            while remaining:
                data = f.read(min(8 * 1024 ** 2, remaining))
                require(data, 'Truncated ZIP data')
                crc = zlib.crc32(data, crc)
                remaining -= len(data)
            patches.append({'name': info.filename, 'positions': positions, 'crc32': crc & 0xffffffff,
                            'bytes': info.file_size, 'zip64': zip64, 'descriptor': bool(flags & 8)})
    require(all(a[1] <= b[0] for a, b in zip(sorted(ranges), sorted(ranges)[1:])), 'Overlapping ZIP records')
    flat = [p for row in patches for p in row['positions']]
    require(len(set(flat)) == len(flat), 'Overlapping CRC fields')
    return patches


def repair_crc(path):
    patches = crc_patch_plan(path)
    size = Path(path).stat().st_size
    with Path(path).open('r+b', buffering=0) as f:
        for row in patches:
            for pos in row['positions']:
                f.seek(pos)
                f.write(struct.pack('<I', row['crc32']))
        os.fsync(f.fileno())
    require(Path(path).stat().st_size == size, 'CRC repair changed file length')
    with zipfile.ZipFile(path) as archive:
        require(archive.testzip() is None, 'CRC verification failed')
    return patches


def create_and_fill(torch, copy, path, base, chunks, reserve_bytes=2 * GIB, margin_bytes=256 * 1024 ** 2):
    """Exactly one large destination file; chunks yields (start,end,tensor_dict)."""
    check_torch(torch)
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), 'Output partial already exists')
    count = len(base['frame_uids'])
    total = count * copy.BYTES_PER_FRAME + base['window_frame_indices'].numel() * 8
    available = shutil.disk_usage(path.parent).free
    require(available >= total + reserve_bytes + margin_bytes, 'Insufficient single-file allocation budget')
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        fake = {key: torch.empty((count, *shape), dtype=getattr(torch, dtype), device='cpu')
                for key, (shape, dtype) in FIELDS.items()}
        fake['window_frame_indices'] = torch.empty(tuple(base['window_frame_indices'].shape), dtype=torch.int64)
    skeleton = {key: value for key, value in base.items() if key != 'window_frame_indices'}
    skeleton.update(fake)
    with path.open('xb') as stream, torch.serialization.skip_data(materialize_fake_tensors=True):
        torch.save(skeleton, stream)
        stream.flush()
        os.fsync(stream.fileno())
    del skeleton, fake
    require(hasattr(os, 'posix_fallocate'), 'posix_fallocate is required; no sparse-only budgeting')
    with path.open('r+b', buffering=0) as stream:
        os.posix_fallocate(stream.fileno(), 0, path.stat().st_size)
        os.fsync(stream.fileno())
    require(shutil.disk_usage(path.parent).free >= reserve_bytes, 'Root reserve unavailable after preallocation')
    with torch.serialization.set_default_mmap_options(mmap.MAP_SHARED):
        payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    advice = copy.PageAdvice()
    payload['window_frame_indices'].copy_(base['window_frame_indices'])
    advice.advise(payload['window_frame_indices'], sync=True)
    cursor = 0
    for start, end, source in chunks:
        require(start == cursor and start < end <= count, 'Noncontiguous recovery chunks')
        copy.tensor_schema_check(torch, source, end - start)
        for key in FIELDS:
            payload[key][start:end].copy_(source[key])
            advice.advise(payload[key][start:end], sync=True)
            advice.advise(source[key])
        require(not advice.errors, 'Mapped flush/page advice failed: ' + repr(advice.errors))
        cursor = end
        print(json.dumps({'event': 'filled', 'frames': end, 'total': count}), flush=True)
    require(cursor == count, 'Incomplete frame coverage')
    require(not advice.errors, 'Lookup flush failed')
    del payload
    gc.collect()
    with path.open('r+b', buffering=0) as stream:
        os.fsync(stream.fileno())
    crc = repair_crc(path)
    return {'tensor_bytes': total, 'file_bytes': path.stat().st_size, 'space_before': available,
            'reserve_bytes': reserve_bytes, 'margin_bytes': margin_bytes,
            'preallocated': True, 'crc_records': crc, 'single_large_file': True}


def load_plan(torch, copy, parts, expected):
    inputs = {name: record(parts / name) for name in ('contract.json', 'owner.json', 'state.json')}
    contract, owner, state = (read_json(parts / name) for name in inputs)
    require(canonical(contract) == expected == state['contract_sha256'] == owner['contract_sha256'], 'Contract binding differs')
    require(contract['format'] == owner['format'] == state['format'] == copy.BUILD_FORMAT, 'Build format differs')
    require(owner['build_id'] == state['build_id'] and state.get('inflight') is None, 'Build owner/inflight differs')
    require(contract['split'] in ('train', 'validation'), 'Test recovery is outside this tool')
    frames, windows = contract['frame_uids'], contract['window_uids']
    require(len(frames) == len(set(frames)) and len(windows) == len(set(windows)), 'Duplicate UIDs')
    require(state['next_frame'] == len(frames), 'Encoding has not finished')
    require(owner['output'] == contract['parameters']['output'], 'Logical output mismatch')
    shards = checked_directory(parts / 'shards')
    require({p.name for p in shards.iterdir()} == {row['file'] for row in state['shards']}, 'Shard roster differs')
    cursor = 0
    for row in state['shards']:
        start, end = row['start'], row['end']
        require(start == cursor and start < end <= len(frames), 'Shard ranges differ')
        require(row['file'] == f'shard_{start:08d}_{end:08d}.pt', 'Invalid shard path')
        p = shards / row['file']
        require(not p.is_symlink() and p.is_file() and p.stat().st_size == row['bytes'], 'Shard file differs')
        require(row['frame_uid_sha256'] == canonical(frames[start:end]), 'Shard UID hash differs')
        cursor = end
    require(cursor == len(frames), 'Incomplete shard roster')
    vp = Path(contract['vace']['path'])
    require(vp.stat().st_size == contract['vace']['bytes'], 'VACE size differs')
    vace = torch.load(vp, map_location='cpu', weights_only=False, mmap=True)
    metadata = vace['metadata']
    require([r['window_uid'] for r in metadata] == windows, 'VACE window UIDs differ')
    indices = {uid: i for i, uid in enumerate(frames)}
    lookup = [[indices[uid] for uid in row['target_frame_uids']] for row in metadata]
    require(all(len(row) == 8 for row in lookup) and canonical(lookup) == contract['window_frame_indices_sha256'], 'Lookup binding differs')
    semantic = canonical({'signature': vace['signature'], 'window_uids': windows,
                          'target_frame_uids': [row['target_frame_uids'] for row in metadata]})
    sig = {'format': copy.SIDE_FORMAT, 'split': contract['split'], 'development_only': contract['coverage']['development_only'],
        'anchor_revision': 'raw-bbox-v11', 'builder_revision': copy.BUILD_FORMAT, 'raw_bbox_margin': .25,
        'unique_frames': len(frames), 'windows': len(windows),
        'hmr2a_checkpoint': contract['hmr_checkpoint']['path'], 'hmr2a_checkpoint_sha256': copy.HMR2A_SHA256,
        'vace_cache': contract['vace']['path'], 'vace_cache_bytes': contract['vace']['bytes'],
        'vace_cache_sha256': contract['vace']['sha256'], 'vace_semantic_sha256': semantic,
        'crop_protocol': 'raw rectangular 3DPW poses2d bbox + 0.25 per-side margin; single HMR2 aspect expansion; no pre-square or boundary clamp',
        'rgb_token_dtype': 'bfloat16', 'anchor_dtype': 'float32', 'build_contract_sha256': expected}
    base = {'signature': sig, 'frame_uids': frames, 'window_uids': windows,
            'window_frame_indices': torch.tensor(lookup, dtype=torch.int64)}
    del vace
    return contract, state, base, inputs


def shard_chunks(torch, copy, parts, state, contract_sha, frames, hash_content=True):
    for row in state['shards']:
        path = parts / 'shards' / row['file']
        before = path.stat()
        if hash_content:
            require(digest(path) == row['sha256'], 'Shard SHA differs: ' + row['file'])
        shard = copy.validate_shard(torch, path, contract_sha, frames, row['start'])
        require(shard['end'] == row['end'], 'Shard end differs')
        yield row['start'], row['end'], shard
        after = path.stat()
        require((before.st_ino, before.st_dev, before.st_size, before.st_mtime_ns) ==
                (after.st_ino, after.st_dev, after.st_size, after.st_mtime_ns), 'Shard changed during use')
        del shard


def exact_payload(torch, actual, base, chunks, copy):
    require(actual['signature'] == base['signature'] and actual['frame_uids'] == base['frame_uids']
            and actual['window_uids'] == base['window_uids'], 'Final metadata differs')
    require(actual['window_frame_indices'].dtype == torch.int64 and
            torch.equal(actual['window_frame_indices'], base['window_frame_indices']), 'Final lookup differs')
    for key, (shape, dtype) in FIELDS.items():
        require(tuple(actual[key].shape) == (len(base['frame_uids']), *shape)
                and actual[key].dtype == getattr(torch, dtype), 'Final tensor schema differs')
    advice, count = copy.PageAdvice(), 0
    for start, end, shard in chunks:
        for key in FIELDS:
            for offset in range(start, end, 32):
                stop = min(offset + 32, end)
                a, b = actual[key][offset:stop], shard[key][offset-start:stop-start]
                require(torch.equal(a.view(torch.uint8), b.view(torch.uint8)), 'Bitwise tensor mismatch: ' + key)
                advice.advise(a)
            advice.advise(shard[key])
        count = end
    require(count == len(base['frame_uids']) and not advice.errors, 'Incomplete final validation')


def verify_existing(args):
    import torch
    check_torch(torch)
    copy = copied_module(args.copy_source)
    parts = checked_directory(args.parts_dir)
    contract, state, base, before = load_plan(torch, copy, parts, args.expected_contract_sha256)
    actual = torch.load(args.verify_existing, map_location='cpu', weights_only=False, mmap=True)
    exact_payload(torch, actual, base, shard_chunks(torch, copy, parts, state,
        args.expected_contract_sha256, base['frame_uids']), copy)
    del actual
    with zipfile.ZipFile(args.verify_existing) as archive:
        require(archive.testzip() is None, 'Independent ZIP CRC validation failed')
    require(before == {name: record(parts / name) for name in before}, 'Original metadata changed')
    write_new_json(args.verification_report, {'status': 'passed', 'passed': True,
        'torch_version': torch.__version__, 'frames': len(base['frame_uids']), 'shards': len(state['shards']),
        'tensor_fields': 5, 'crc_valid': True, 'bitwise_equal': True,
        'output': record(args.verify_existing), 'original_metadata': before})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parts-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--expected-contract-sha256', required=True)
    parser.add_argument('--copy-source', type=Path, default=Path(__file__).with_name(COPY_NAME))
    parser.add_argument('--reserve-bytes', type=int, default=2 * GIB)
    parser.add_argument('--publish-alias', action='store_true')
    parser.add_argument('--verify-existing', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--verification-report', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    controller_fd, lock_identity = inherited_controller_lock()
    if args.verify_existing:
        return verify_existing(args)
    import torch
    check_torch(torch)
    require(args.run_dir is not None and args.reserve_bytes >= 2 * GIB, 'New run directory and >=2GiB reserve required')
    parts, parent = checked_directory(args.parts_dir), checked_directory(args.run_dir.parent)
    require(parent.stat().st_dev != parts.stat().st_dev, 'Root output must use a different filesystem from shards')
    run = parent / args.run_dir.name
    require(not run.exists() and not run.is_symlink(), 'Recovery directory must be new')
    copy = copied_module(args.copy_source)
    with readonly_build_lock(parts):
        contract, state, base, before = load_plan(torch, copy, parts, args.expected_contract_sha256)
        logical = Path(contract['parameters']['output']).absolute()
        checked_directory(logical.parent)
        require(not logical.exists() and not logical.is_symlink(), 'Original output already exists')
        source_records = {row['path']: row['sha256'] for row in contract['source_files']}
        for path, sha in source_records.items():
            require(digest(path) == sha, 'Original source changed: ' + path)
        require(digest(contract['vace']['path']) == contract['vace']['sha256'], 'VACE content changed')
        run.mkdir(mode=0o700)
        partial, final = run / 'hmr_train_s8_raw25.partial.pt', run / logical.name
        sources = {'helper': record(__file__), 'copy': record(args.copy_source)}
        write_new_json(run / 'recovery_inputs.json', {'format': 'star-body-hmr-singlefile-recovery-v1',
            'contract_sha256': args.expected_contract_sha256, 'original_metadata': before,
            'original_sources': contract['source_files'], 'recovery_sources': sources,
            'frame_uids': base['frame_uids'], 'window_uids': base['window_uids'],
            'state_shards': state['shards'], 'logical_output': str(logical), 'physical_output': str(final),
            'reserve_bytes': args.reserve_bytes, 'margin_bytes': 256 * 1024 ** 2,
            'model_forward_performed': False, 'original_state_mutated': False,
            'inherited_controller_lock': lock_identity})
        try:
            storage = create_and_fill(torch, copy, partial, base,
                shard_chunks(torch, copy, parts, state, args.expected_contract_sha256, base['frame_uids']), args.reserve_bytes)
            cmd = [sys.executable, str(Path(__file__).resolve()), '--parts-dir', str(parts),
                '--expected-contract-sha256', args.expected_contract_sha256, '--copy-source', str(args.copy_source.resolve()),
                '--verify-existing', str(partial), '--verification-report', str(run / 'independent_verification.json')]
            with (run / 'independent_verification.log').open('xb') as log:
                subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, pass_fds=(controller_fd,))
            require(before == {name: record(parts / name) for name in before}, 'Original metadata changed')
            require(sources == {'helper': record(__file__), 'copy': record(args.copy_source)}, 'Recovery source changed')
            os.link(partial, final)
            partial.unlink()
            fsync_dir(run)
            alias = {'requested': args.publish_alias, 'published': False, 'logical': str(logical), 'target': str(final)}
            if args.publish_alias:
                os.symlink(str(final), logical)
                fsync_dir(logical.parent)
                require(logical.is_symlink() and logical.resolve(strict=True) == final, 'Published alias differs')
                alias.update(published=True, symlink_inode=logical.lstat().st_ino,
                             logical_parent_device=logical.parent.stat().st_dev, target_device=final.stat().st_dev)
            verified = read_json(run / 'independent_verification.json')
            write_new_json(run / 'recovery_receipt.json', {'format': 'star-body-hmr-singlefile-publication-v1',
                'status': 'recovered_pending_original_verifier', 'recovery_passed': True,
                'contract_sha256': args.expected_contract_sha256, 'original_metadata': before,
                'original_sources': contract['source_files'], 'recovery_sources': sources,
                'storage': storage, 'alias': alias, 'output': dict(verified['output'], path=str(final)),
                'verification': record(run / 'independent_verification.json'),
                'frames': len(base['frame_uids']), 'windows': len(base['window_uids']), 'shards': len(state['shards']),
                'torch_version': torch.__version__, 'original_state_mutated': False,
                'inherited_controller_lock': lock_identity,
                'model_forward_performed': False, 'original_full_verifier_required': True,
                'all_original_assets_revalidated_by_recovery': False})
        except BaseException as error:
            write_new_json(run / 'failure.json', {'status': 'failed', 'type': type(error).__name__,
                'error': str(error), 'partial_retained': partial.exists(), 'final_exists': final.exists(),
                'original_state_mutated': False, 'automatic_cleanup_performed': False})
            raise


if __name__ == '__main__':
    main()
