"""Small stdlib-only recovery/ZIP32/ZIP64 mechanism checks. No real assets."""
import argparse
import importlib.util
import io
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
SOURCE = Path(os.environ.get('STAR_HMR_RECOVERY_SOURCE', HERE / 'recover_hmr_singlefile_v1.py'))
spec = importlib.util.spec_from_file_location('recovery_mechanism_test', SOURCE)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


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

    def test_stdlib_force_zip64(self):
        path = self.root / 'force.pt'
        with zipfile.ZipFile(path, 'w') as z:
            with z.open('small', 'w', force_zip64=True) as f:
                f.write(b'abcd')
        self.assertTrue(r.repair_crc(path)[0]['zip64'])

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
            'runs/phase4_singlefile_tools_001/recover_hmr_singlefile_v1.py': r.digest(SOURCE),
            'runs/phase4_singlefile_tools_001/' + r.COPY_NAME: r.digest(Path(os.environ.get('STAR_HMR_RECOVERY_COPY', HERE / r.COPY_NAME))),
            'runs/phase4_singlefile_tools_001/test_hmr_singlefile_v1.py': r.digest(__file__)}}
    r.write_new_json(args.report, report)
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__': main()
