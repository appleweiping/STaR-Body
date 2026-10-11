"""Small stdlib-only recovery/ZIP32/ZIP64 mechanism checks. No real assets."""
import argparse
import importlib.util
import io
import inspect
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import zipfile
import zlib

HERE = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get('STAR_HMR_RECOVERY_SOURCE', HERE / 'recover_hmr_singlefile_v2.py'))
spec = importlib.util.spec_from_file_location('recovery_mechanism_test', SOURCE)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
FIXTURE_EVIDENCE = {}


def offset_only_zip(path, data=b'offset-only descriptor', signed=True, width=64,
                    descriptor_tail=b'', descriptor_sizes=None, crc=0, name=b'archive/data/1'):
    """Tiny physical archive with the real PyTorch offset-only local schema."""
    flags, size = 2056, len(data)
    extra = struct.pack('<HHQ', 1, 8, 0)
    local = struct.pack('<4s5H3I2H', b'PK\x03\x04', 0, flags, 0, 0, 0,
                        0, 0, 0, len(name), len(extra)) + name + extra
    sizes = descriptor_sizes if descriptor_sizes is not None else (size, size)
    dd = (b'PK\x07\x08' if signed else b'') + struct.pack('<I', crc)
    dd += struct.pack('<QQ' if width == 64 else '<II', *sizes) + descriptor_tail
    central_offset = len(local) + size + len(dd)
    central = struct.pack('<4s6H3I5H2I', b'PK\x01\x02', 0, 0, flags,
        0, 0, 0, 0, size, size, len(name), 0, 0, 0, 0, 0, 0) + name
    eocd = struct.pack('<4s4H2IH', b'PK\x05\x06', 0, 0, 1, 1, len(central), central_offset, 0)
    path.write_bytes(local + data + dd + central + eocd)
    return {'data_start': len(local), 'descriptor_start': len(local) + size,
            'crc_position': len(local) + size + (4 if signed else 0), 'central': central_offset}


class SegmentReader:
    """Read-only synthetic address space; large offsets consume no disk space."""
    def __init__(self, segments, size):
        self.segments, self.size, self.pos = segments, size, 0
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def seek(self, pos, whence=0):
        if whence != 0: raise ValueError('Only absolute seeks in this fixture')
        self.pos = pos
        return pos
    def read(self, count):
        count = min(count, self.size - self.pos)
        if count <= 0: return b''
        out = bytearray(count)
        for start, data in self.segments:
            left, right = max(start, self.pos), min(start + len(data), self.pos + count)
            if left < right: out[left - self.pos:right - self.pos] = data[left - start:right - start]
        self.pos += count
        return bytes(out)


def handcrafted_zip(path, zip64=False, descriptor=False, signed=True, data=b'actual bytes not zero'):
    name = b'data/0'
    flags = 8 if descriptor else 0
    crc = zlib.crc32(data) & 0xffffffff
    size = len(data)
    extra = struct.pack('<HHQQ', 1, 16, size, size) if zip64 else b''
    local_size = 0xffffffff if zip64 else (0 if descriptor else size)
    local = struct.pack('<4s5H3I2H', b'PK\x03\x04', 45 if zip64 else 20, flags, 0, 0, 0,
                        0, local_size, local_size, len(name), len(extra)) + name + extra
    dd = b''
    if descriptor:
        dd = (b'PK\x07\x08' if signed else b'') + struct.pack('<I', 0)
        dd += struct.pack('<QQ' if zip64 else '<II', size, size)
    central_offset = len(local) + size + len(dd)
    central = struct.pack('<4s6H3I5H2I', b'PK\x01\x02', 45, 45 if zip64 else 20, flags,
        0, 0, 0, 0, 0xffffffff if zip64 else size, 0xffffffff if zip64 else size,
        len(name), len(extra), 0, 0, 0, 0, 0) + name + extra
    eocd = struct.pack('<4s4H2IH', b'PK\x05\x06', 0, 0, 1, 1, len(central), central_offset, 0)
    path.write_bytes(local + data + dd + central + eocd)
    return {'data': data, 'offset': len(local), 'crc': crc, 'central': central_offset,
            'descriptor': len(local) + size + (4 if signed else 0)}


class Mechanisms(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def check_zip(self, zip64, descriptor, signed=True):
        path = self.root / 'data.pt'
        setup = handcrafted_zip(path, zip64, descriptor, signed)
        before = path.read_bytes()
        rows = r.repair_crc(path)
        after = path.read_bytes()
        self.assertEqual(len(before), len(after))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['zip64'], zip64)
        self.assertEqual(rows[0]['descriptor'], descriptor)
        allowed = {i for p in rows[0]['positions'] for i in range(p, p + 4)}
        self.assertTrue(all(a == b or i in allowed for i, (a, b) in enumerate(zip(before, after))))
        self.assertEqual(after[setup['offset']:setup['offset'] + len(setup['data'])], setup['data'])
        with zipfile.ZipFile(path) as z:
            self.assertIsNone(z.testzip())
            self.assertEqual(z.read('data/0'), setup['data'])
        for pos in rows[0]['positions']:
            self.assertEqual(struct.unpack_from('<I', after, pos)[0], setup['crc'])

    def test_zip32_plain(self): self.check_zip(False, False)
    def test_zip32_signed_descriptor(self): self.check_zip(False, True, True)
    def test_zip32_unsigned_descriptor(self): self.check_zip(False, True, False)
    def test_zip64_plain(self): self.check_zip(True, False)
    def test_zip64_signed_descriptor(self): self.check_zip(True, True, True)
    def test_zip64_unsigned_descriptor(self): self.check_zip(True, True, False)

    def check_offset_only(self, *, data=b'offset-only descriptor', signed=True, width=64, crc=0):
        path = self.root / 'offset.pt'
        setup = offset_only_zip(path, data=data, signed=signed, width=width, crc=crc)
        before = path.read_bytes()
        rows = r.repair_crc(path)
        after = path.read_bytes()
        self.assertFalse(rows[0]['zip64'])  # Existing field means local size sentinel.
        self.assertEqual(rows[0]['positions'][-1], setup['crc_position'])
        positions = {i for p in rows[0]['positions'] for i in range(p, p + 4)}
        self.assertTrue(all(a == b or i in positions for i, (a, b) in enumerate(zip(before, after))))
        self.assertEqual(len(before), len(after))
        with zipfile.ZipFile(path) as archive:
            self.assertEqual(archive.read('archive/data/1'), data)
        restored = bytearray(after)
        for position in rows[0]['positions']:
            restored[position:position + 4] = before[position:position + 4]
        self.assertEqual(bytes(restored), before)

    def test_offset_only_signed64_version_zero(self): self.check_offset_only()
    def test_offset_only_unsigned64_version_zero(self): self.check_offset_only(signed=False)
    def test_offset_only_signed32(self): self.check_offset_only(width=32)
    def test_offset_only_unsigned32(self): self.check_offset_only(signed=False, width=32)
    def test_zero_size_signed64_exact_boundary(self): self.check_offset_only(data=b'')
    def test_zero_size_unsigned64_exact_boundary(self): self.check_offset_only(data=b'', signed=False)
    def test_zero_size_signed32_exact_boundary(self): self.check_offset_only(data=b'', width=32)
    def test_zero_size_unsigned32_exact_boundary(self): self.check_offset_only(data=b'', signed=False, width=32)
    def test_unsigned_crc_matches_signature64(self): self.check_offset_only(signed=False, crc=0x08074b50)
    def test_unsigned_crc_matches_signature32(self): self.check_offset_only(signed=False, width=32, crc=0x08074b50)

    def test_zero_size_unsupported_boundary_rejected_before_write(self):
        path = self.root / 'bad.pt'
        offset_only_zip(path, data=b'', descriptor_tail=b'\x00' * 4)
        before = path.read_bytes()
        with self.assertRaises(ValueError): r.repair_crc(path)
        self.assertEqual(path.read_bytes(), before)

    def test_offset_only_wrong_sizes_rejected_before_write(self):
        path = self.root / 'bad.pt'
        offset_only_zip(path, descriptor_sizes=(33, 33))
        before = path.read_bytes()
        with self.assertRaises(ValueError): r.repair_crc(path)
        self.assertEqual(path.read_bytes(), before)

    def test_offset_only_truncated_descriptor_rejected_before_write(self):
        path = self.root / 'bad.pt'
        setup = offset_only_zip(path)
        raw = bytearray(path.read_bytes())
        # Remove four size bytes and correct only the central offset in EOCD.
        del raw[setup['central'] - 4:setup['central']]
        struct.pack_into('<I', raw, len(raw) - 6, setup['central'] - 4)
        path.write_bytes(raw)
        with self.assertRaises(ValueError): r.repair_crc(path)
        self.assertEqual(path.read_bytes(), bytes(raw))

    def mixed_archive(self, reverse=False, overlap=False, duplicate=False):
        first, second = self.root / 'first.pt', self.root / 'second.pt'
        a = offset_only_zip(first, width=32, name=b'archive/data/0')
        b = offset_only_zip(second, width=64, signed=False)
        ar, br = first.read_bytes(), second.read_bytes()
        ca, cb = bytearray(ar[a['central']:-22]), bytearray(br[b['central']:-22])
        second_offset = 0 if duplicate else (a['data_start'] if overlap else a['central'])
        struct.pack_into('<I', cb, 42, second_offset)
        records = ar[:a['central']] + br[:b['central']]
        central = bytes(cb + ca if reverse else ca + cb)
        eocd = struct.pack('<4s4H2IH', b'PK\x05\x06', 0, 0, 2, 2, len(central), len(records), 0)
        path = self.root / 'mixed.pt'
        path.write_bytes(records + central + eocd)
        return path

    def test_next_local_boundary_mixed_widths_reversed_central_order(self):
        path = self.mixed_archive(reverse=True)
        before = path.read_bytes()
        rows = r.repair_crc(path)
        after = path.read_bytes()
        self.assertEqual([row['name'] for row in rows], ['archive/data/1', 'archive/data/0'])
        allowed = {p + i for row in rows for p in row['positions'] for i in range(4)}
        self.assertTrue(all(a == b or i in allowed for i, (a, b) in enumerate(zip(before, after))))
        with zipfile.ZipFile(path) as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(archive.read('archive/data/0'), archive.read('archive/data/1'))

    def test_next_local_inside_data_rejected_before_write(self):
        path = self.mixed_archive(overlap=True)
        before = path.read_bytes()
        with self.assertRaises(ValueError): r.repair_crc(path)
        self.assertEqual(path.read_bytes(), before)

    def test_duplicate_local_offsets_rejected_before_write(self):
        path = self.mixed_archive(duplicate=True)
        before = path.read_bytes()
        with self.assertRaises(ValueError): r.repair_crc(path)
        self.assertEqual(path.read_bytes(), before)

    def test_real_large_offset_header_layout_with_synthetic_zero_data(self):
        # Exact offsets/name/extra/descriptor from failed_zip64_headers.json.
        # The payload here is synthetic zero bytes, NOT the real tensor.
        offset, size, end, central = 11176138584, 19643040, 11195781728, 11195781752
        name = b'archive/data/1'
        extra = bytes.fromhex('010008005857269a0200000046422c00' + '5a' * 44)
        local = struct.pack('<4s5H3I2H', b'PK\x03\x04', 0, 2056, 0, 0, 0,
                            0, 0, 0, len(name), len(extra)) + name + extra
        self.assertEqual(offset + len(local) + size, end)
        descriptor = bytes.fromhex('504b070800000000a0ba2b0100000000a0ba2b0100000000')
        central_header = struct.pack('<4s6H3I5H2I', b'PK\x01\x02', 0, 0, 2056,
            0, 0, 0, 0, size, size, len(name), 12, 0, 0, 0, 0, 0xffffffff) + name
        info = zipfile.ZipInfo(name.decode())
        info.header_offset, info.file_size, info.compress_size = offset, size, size
        info.flag_bits, info.compress_type = 2056, 0
        archive = SimpleNamespace(infolist=lambda: [info], start_dir=central)
        class ArchiveContext:
            def __enter__(self): return archive
            def __exit__(self, *args): return False
        segments = [(offset, local), (end, descriptor), (central, central_header)]
        with patch.object(r.zipfile, 'ZipFile', return_value=ArchiveContext()), \
             patch.object(Path, 'open', return_value=SegmentReader(segments, central + len(central_header))):
            rows = r.crc_patch_plan(self.root / 'virtual.pt')
        self.assertEqual(rows[0]['positions'], [central + 16, offset + 14, end + 4])
        crc = 0
        for _ in range(size // 4096): crc = zlib.crc32(b'\x00' * 4096, crc)
        crc = zlib.crc32(b'\x00' * (size % 4096), crc)
        self.assertEqual(rows[0]['crc32'], crc & 0xffffffff)
        FIXTURE_EVIDENCE['real_header_layout_synthetic_payload'] = {
            'source_evidence': 'outputs/phase4_evidence/failed_zip64_headers.json',
            'header_offset': offset, 'file_size': size, 'descriptor_width_bits': 64,
            'local_size_fields': [0, 0], 'extract_version': 0,
            'local_extra_hex': extra.hex(), 'real_tensor_read': False,
            'disk_allocation_bytes': 0, 'payload': 'synthetic all-zero bytes'}

    def test_stdlib_force_zip64(self):
        path = self.root / 'force.pt'
        with zipfile.ZipFile(path, 'w') as z:
            with z.open('small', 'w', force_zip64=True) as f:
                f.write(b'abcd')
        # CPython 3.10 adds the ZIP64 extra for force_zip64 but writes normal
        # local sizes for small entries; CPython 3.14 writes both sentinels.
        # Inspect actual bytes independently instead of assuming API semantics.
        raw = path.read_bytes()
        self.assertEqual(raw[:4], b'PK\x03\x04')
        sizes = struct.unpack_from('<II', raw, 18)
        name_length, extra_length = struct.unpack_from('<HH', raw, 26)
        extra = raw[30 + name_length:30 + name_length + extra_length]
        self.assertEqual(len(extra), 20)
        self.assertEqual(struct.unpack('<HHQQ', extra), (1, 16, 4, 4))
        self.assertIn(sizes, ((4, 4), (0xffffffff, 0xffffffff)))
        rows = r.repair_crc(path)
        self.assertEqual(rows[0]['zip64'], sizes == (0xffffffff, 0xffffffff))
        with zipfile.ZipFile(path) as z:
            self.assertIsNone(z.testzip())
            self.assertEqual(z.read('small'), b'abcd')
        FIXTURE_EVIDENCE['stdlib_force_zip64'] = {
            'python_version': sys.version, 'local_size_fields': list(sizes),
            'local_extra_hex': extra.hex(), 'parser_zip64_required': rows[0]['zip64'],
            'zipfile_source_path': zipfile.__file__, 'zipfile_source_sha256': r.digest(zipfile.__file__),
            'FileHeader_source': inspect.getsource(zipfile.ZipInfo.FileHeader)}

    def test_bad_descriptor_rejected_before_write(self):
        path = self.root / 'bad.pt'
        setup = handcrafted_zip(path, True, True)
        raw = bytearray(path.read_bytes())
        struct.pack_into('<Q', raw, setup['descriptor'] + 4, 999)
        path.write_bytes(raw)
        with self.assertRaises(ValueError): r.repair_crc(path)
        self.assertEqual(bytes(raw), path.read_bytes())

    def test_bad_local_flags_rejected(self):
        path = self.root / 'bad.pt'
        handcrafted_zip(path)
        raw = bytearray(path.read_bytes())
        struct.pack_into('<H', raw, 6, 1)
        path.write_bytes(raw)
        with self.assertRaises(ValueError): r.repair_crc(path)
        self.assertEqual(bytes(raw), path.read_bytes())

    def test_compressed_entry_rejected(self):
        path = self.root / 'compressed.pt'
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z: z.writestr('data', b'abc')
        with self.assertRaises(ValueError): r.repair_crc(path)

    def test_duplicate_entry_rejected(self):
        import warnings
        path = self.root / 'duplicate.pt'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with zipfile.ZipFile(path, 'w') as z:
                z.writestr('same', b'abc'); z.writestr('same', b'def')
        with self.assertRaises(ValueError): r.repair_crc(path)

    def test_corrupted_central_rejected(self):
        path = self.root / 'truncated.pt'
        setup = handcrafted_zip(path)
        raw = bytearray(path.read_bytes()); raw[setup['central']] = 0
        path.write_bytes(raw)
        with self.assertRaises((ValueError, zipfile.BadZipFile)): r.repair_crc(path)

    def test_directory_symlink_guard(self):
        with patch.object(Path, 'is_symlink', return_value=True):
            with self.assertRaises(ValueError): r.checked_directory(self.root)

    def test_copy_sha_bound(self):
        copy = Path(os.environ.get('STAR_HMR_RECOVERY_COPY', HERE / r.COPY_NAME))
        self.assertEqual(r.digest(copy), r.COPY_SHA)
        self.assertEqual(r.copied_module(copy).FIELDS, r.FIELDS)

    def test_copy_corruption_rejected(self):
        path = self.root / 'copy.py'; path.write_text('x=1')
        with self.assertRaises(ValueError): r.copied_module(path)

    def test_report_no_overwrite(self):
        path = self.root / 'record.json'; r.write_new_json(path, {'a': 1})
        with self.assertRaises(FileExistsError): r.write_new_json(path, {'a': 2})
        self.assertEqual(r.read_json(path), {'a': 1})

    def test_canonical_order(self):
        self.assertEqual(r.canonical({'a': 1, 'b': 2}), r.canonical({'b': 2, 'a': 1}))

    def test_low_disk_rejected_before_tensor_allocation(self):
        base = {'frame_uids': ['one'], 'window_frame_indices': SimpleNamespace(numel=lambda: 8)}
        with patch.object(r, 'check_torch'), patch.object(r.shutil, 'disk_usage', return_value=SimpleNamespace(free=10)):
            with self.assertRaises(ValueError):
                r.create_and_fill(None, SimpleNamespace(BYTES_PER_FRAME=492436), self.root / 'out.pt', base, [])
        self.assertFalse((self.root / 'out.pt').exists())

    def test_existing_destination_rejected(self):
        path = self.root / 'out.pt'; path.write_bytes(b'keep')
        with patch.object(r, 'check_torch'):
            with self.assertRaises(ValueError): r.create_and_fill(None, None, path, {}, [])
        self.assertEqual(path.read_bytes(), b'keep')

    def test_record_detects_hash_time_mutation(self):
        path = self.root / 'change'; path.write_bytes(b'old')
        def mutate(p):
            p.write_bytes(b'new and longer')
            return 'bad'
        with patch.object(r, 'digest', side_effect=mutate):
            with self.assertRaises(ValueError): r.record(path)

    def test_controller_lock_missing_rejected(self):
        with patch.dict(sys.modules, {'fcntl': SimpleNamespace()}), patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError): r.inherited_controller_lock()

    def test_controller_lock_bad_fd_rejected(self):
        with patch.dict(sys.modules, {'fcntl': SimpleNamespace()}), patch.dict(os.environ, {'STAR_BODY_CONTROLLER_LOCK_FD': '2'}):
            with self.assertRaises(ValueError): r.inherited_controller_lock()

    def test_controller_lock_wrong_identity_rejected(self):
        fake = SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=lambda *a: None)
        with patch.dict(sys.modules, {'fcntl': fake}), patch.dict(os.environ, {'STAR_BODY_CONTROLLER_LOCK_FD': '7'}), \
             patch.object(Path, 'resolve', return_value=Path('/expected/lock')), \
             patch.object(os, 'readlink', return_value='/wrong/lock'):
            with self.assertRaises(ValueError): r.inherited_controller_lock()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Mechanisms))
    report = {'status': 'passed' if result.wasSuccessful() else 'failed', 'passed': result.wasSuccessful(),
        'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'skipped': [str(item) for item in result.skipped], 'synthetic_only': True,
        'real_tensor_parity': False, 'source_sha256': {
            'runs/phase4_singlefile_tools_002/recover_hmr_singlefile_v2.py': r.digest(SOURCE),
            'runs/phase4_singlefile_tools_001/' + r.COPY_NAME: r.digest(Path(os.environ.get('STAR_HMR_RECOVERY_COPY', HERE / r.COPY_NAME))),
            'runs/phase4_singlefile_tools_002/test_hmr_singlefile_v3.py': r.digest(__file__)},
        'fixture_evidence': FIXTURE_EVIDENCE,
        'copy_source_sha256': '2e4f7eb7c964b7b26455c9896aadb2124fcc7374a0eb4660f10908514922302a'}
    r.write_new_json(args.report, report)
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__': main()
