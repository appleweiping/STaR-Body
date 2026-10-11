"""Tiny local ZIP transaction tests; no Torch, SSH, main, jobs or real cache I/O."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid
import zipfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ENTRY = HERE / 'finalize_hmr_partial_v1.py'
spec = importlib.util.spec_from_file_location('independent_finalizer', ENTRY)
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)


def sha(data): return hashlib.sha256(data).hexdigest()


class TransactionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=HERE)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / 'failed.partial.zip'
        self.run = self.root / 'repair'
        self.run.mkdir()
        with zipfile.ZipFile(self.archive, 'x', compression=zipfile.ZIP_STORED) as z:
            z.writestr('archive/data/0', b'actual tiny tensor-like bytes\x00\xff' * 19)
            z.writestr('archive/data/1', b'second payload' * 7)
            z.writestr('archive/data/2', b'')
        self.planned = f.h.crc_patch_plan(self.archive)
        data = bytearray(self.archive.read_bytes())
        for row in self.planned:
            for offset in row['positions']: data[offset:offset+4] = b'\0' * 4
        self.archive.write_bytes(data)
        self.before = bytes(data)
        self.expected = sha(self.before)
        # Directory fsync has no Windows equivalent; retain actual file fsync.
        if sys.platform == 'win32':
            self.dir_sync = patch.object(f.h, 'fsync_dir', lambda _: None)
            self.dir_sync.start()
            self.addCleanup(self.dir_sync.stop)

    def run_repair(self): return f.journal_crc(self.archive, self.run, self.expected)

    def journal(self): return json.loads((self.run / 'crc_undo_journal.json').read_text())

    def test_success_valid_crc_and_durable_order(self):
        synchronized = []
        original_fsync = f.h.fsync_dir
        def fsync_directory(path):
            self.assertEqual(self.archive.read_bytes(), self.before)
            self.assertTrue((Path(path) / 'crc_undo_journal.json').is_file())
            synchronized.append(str(path))
            original_fsync(path)
        with patch.object(f.h, 'fsync_dir', fsync_directory): result = self.run_repair()
        self.assertEqual(synchronized, [str(self.run)])
        self.assertTrue(result['exact_original_reconstructible'])
        self.assertEqual(result['tensor_bytes_rewritten'], 0)
        with zipfile.ZipFile(self.archive) as z: self.assertIsNone(z.testzip())

    def test_only_crc_field_bytes_changed_and_virtual_undo_exact(self):
        result = self.run_repair()
        journal = self.journal()
        after = self.archive.read_bytes()
        allowed = {p['offset'] + i for p in journal['patches'] for i in range(4)}
        changed = {i for i, (a, b) in enumerate(zip(self.before, after)) if a != b}
        self.assertTrue(changed)
        self.assertLessEqual(changed, allowed)
        restored = bytearray(after)
        for p in journal['patches']: restored[p['offset']:p['offset']+4] = bytes.fromhex(p['before_hex'])
        self.assertEqual(bytes(restored), self.before)
        self.assertEqual(f.virtual_sha(self.archive, journal['patches']), self.expected)
        self.assertEqual(result['after']['sha256'], sha(after))

    def test_wrong_expected_sha_refuses_before_journal_or_mutation(self):
        with self.assertRaisesRegex(ValueError, 'SHA differs'):
            f.journal_crc(self.archive, self.run, '0'*64)
        self.assertEqual(self.archive.read_bytes(), self.before)
        self.assertEqual(list(self.run.iterdir()), [])

    def test_invalid_zip_structure_refuses_before_journal_or_mutation(self):
        self.archive.write_bytes(b'not a zip archive')
        bad = self.archive.read_bytes()
        with self.assertRaises(zipfile.BadZipFile):
            f.journal_crc(self.archive, self.run, sha(bad))
        self.assertEqual(self.archive.read_bytes(), bad)
        self.assertEqual(list(self.run.iterdir()), [])

    def test_extra_payload_byte_change_breaks_virtual_undo_sha(self):
        self.run_repair()
        data = bytearray(self.archive.read_bytes())
        index = data.index(b'actual tiny tensor-like bytes')
        data[index] ^= 1
        self.archive.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'Original SHA differs'):
            f.require(f.virtual_sha(self.archive, self.journal()['patches']) == self.expected, 'Original SHA differs')

    def test_unrecorded_crc_field_value_rejected(self):
        self.run_repair()
        journal = self.journal()
        data = bytearray(self.archive.read_bytes())
        offset = journal['patches'][0]['offset']
        data[offset:offset+4] = bytes.fromhex('ffffffff')
        self.archive.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'unrecorded value'): f.virtual_sha(self.archive, journal['patches'])

    def test_duplicate_overlapping_or_negative_positions_rejected(self):
        self.run_repair()
        good = self.journal()['patches']
        for bad in ([good[0], good[0]], [dict(good[0], offset=-1)], [dict(good[0], offset=len(self.before)+1)]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError): f.virtual_sha(self.archive, bad)

    def test_duplicate_journal_refuses_replay_even_with_current_sha(self):
        self.run_repair()
        after = self.archive.read_bytes()
        journal = (self.run / 'crc_undo_journal.json').read_bytes()
        with self.assertRaises(FileExistsError): f.journal_crc(self.archive, self.run, sha(after))
        self.assertEqual(self.archive.read_bytes(), after)
        self.assertEqual((self.run / 'crc_undo_journal.json').read_bytes(), journal)

    def test_failure_after_first_write_keeps_original_evidence_and_no_receipt(self):
        real_open = Path.open
        counts = []
        class FailedWriter:
            def __init__(self, stream): self.stream = stream
            def __enter__(self): return self
            def __exit__(self, *args): self.stream.close()
            def __getattr__(self, name): return getattr(self.stream, name)
            def write(self, data):
                self.stream.write(data)
                counts.append(len(data))
                raise OSError('injected failure immediately after first CRC write')
        def intercepted(path, *args, **kwargs):
            stream = real_open(path, *args, **kwargs)
            return FailedWriter(stream) if args and args[0] == 'r+b' else stream
        with patch.object(Path, 'open', intercepted):
            with self.assertRaisesRegex(OSError, 'injected'): self.run_repair()
        self.assertEqual(counts, [4])
        self.assertTrue(self.archive.exists())
        self.assertTrue((self.run / 'crc_undo_journal.json').is_file())
        self.assertFalse((self.run / 'crc_repair_receipt.json').exists())
        self.assertEqual(f.virtual_sha(self.archive, self.journal()['patches']), self.expected)
        interrupted = self.archive.read_bytes()
        with self.assertRaises(FileExistsError): f.journal_crc(self.archive, self.run, sha(interrupted))
        self.assertEqual(self.archive.read_bytes(), interrupted)

    def test_new_hardlink_publication_does_not_copy_and_refuses_overwrite(self):
        self.run_repair()
        final = self.run / 'final.zip'
        f.os.link(self.archive, final)
        self.assertEqual((self.archive.stat().st_dev, self.archive.stat().st_ino), (final.stat().st_dev, final.stat().st_ino))
        with self.assertRaises(FileExistsError): f.os.link(self.archive, final)
        self.assertEqual(final.read_bytes(), self.archive.read_bytes())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TransactionTests))
    report = dict(status='passed' if result.wasSuccessful() else 'failed', passed=result.wasSuccessful(),
        tests=result.testsRun, failures=len(result.failures), errors=len(result.errors), skipped=[],
        synthetic_only=True, real_tensor_validation=False, real_cache_modified=False, main_executed=False,
        directory_fsync_mocked=sys.platform=='win32', file_fsync_real=True,
        source_sha256={p.name:sha(p.read_bytes()) for p in (ENTRY, Path(f.h.__file__), Path(__file__))})
    output = args.report or HERE / ('finalize_crc_result_' + uuid.uuid4().hex + '.json')
    with output.open('x') as stream: json.dump(report, stream, indent=2)
    print(output)
    raise SystemExit(0 if result.wasSuccessful() else 1)
