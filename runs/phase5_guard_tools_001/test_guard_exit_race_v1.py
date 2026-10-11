"""Synthetic proc-read races only; no real processes, /proc reads or signals."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SOURCE = HERE/'run_memory_disk_guard_v3.py'
ORIGINAL = HERE.parent/'phase3_splitdisk_stage/run_memory_disk_guard_v2.py'
spec = importlib.util.spec_from_file_location('guard_exit_candidate', SOURCE)
guard = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = guard
spec.loader.exec_module(guard)


def stat_text(state, **changes):
    fields = dict(pid=431, ppid=100, pgrp=431, session=431, start_ticks=1200)
    fields.update(changes)
    tail = [state, str(fields['ppid']), str(fields['pgrp']), str(fields['session'])]
    tail += ['0']*15+[str(fields['start_ticks'])]
    return str(fields['pid'])+' (worker) '+' '.join(tail)


UID = 'Uid: 1000 1000 1000 1000\n'
MEM = 'RssAnon: 17 kB\nVmRSS: 25 kB\nVmSwap: 0 kB\n'


class ExitRace(unittest.TestCase):
    def snapshot(self, first='R', second='Z', status=UID, changes=None):
        readings = iter([stat_text(first), status, stat_text(second, **(changes or {}))])
        def read(path, *args, **kwargs):
            self.assertIn(path.name, ('stat', 'status'))
            return next(readings)
        with patch.object(Path, 'read_text', read):
            return guard.process_snapshot(431, Path('/synthetic/proc'))

    def test_running_to_zombie_missing_mm_is_zero_after_identity_check(self):
        result = self.snapshot()
        self.assertEqual(result['state'], 'Z')
        self.assertEqual([result[k+'_bytes'] for k in ('RssAnon','VmRSS','VmSwap')], [0,0,0])
        self.assertEqual(result['uids'], [1000]*4)

    def test_stable_zombie_missing_mm(self):
        self.assertEqual(self.snapshot(first='Z')['RssAnon_bytes'], 0)

    def test_stable_running_fields_remain_exact(self):
        result = self.snapshot(second='R', status=UID+MEM)
        self.assertEqual((result['state'],result['RssAnon_bytes'],result['VmRSS_bytes']), ('R',17*1024,25*1024))

    def test_running_to_zombie_present_memory_not_discarded(self):
        self.assertEqual(self.snapshot(status=UID+MEM)['RssAnon_bytes'], 17*1024)

    def test_live_task_missing_memory_still_invalid(self):
        with self.assertRaises(KeyError): self.snapshot(second='R')

    def test_sleeping_task_missing_memory_still_invalid(self):
        with self.assertRaises(KeyError): self.snapshot(second='S')

    def test_zombie_missing_uid_still_invalid(self):
        with self.assertRaises(KeyError): self.snapshot(status='State: Z (zombie)\n')

    def test_zombie_malformed_present_memory_still_invalid(self):
        for text in ('RssAnon: -1 kB\n','RssAnon: 1 MB\n','RssAnon: x kB\n'):
            with self.subTest(text=text),self.assertRaises(ValueError): self.snapshot(status=UID+text)

    def test_second_zombie_cannot_mask_identity_changes(self):
        for key in ('pid','start_ticks','pgrp','session'):
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'identity changed'):
                self.snapshot(changes={key:9999})

    def test_disappeared_task_propagates_existing_exit_handling(self):
        with patch.object(Path,'read_text',side_effect=[stat_text('R'),UID,FileNotFoundError('exited')]):
            with self.assertRaises(FileNotFoundError): guard.process_snapshot(431,Path('/synthetic/proc'))

    def test_duplicate_status_fields_remain_invalid(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'): self.snapshot(status=UID+UID)

    def test_original_race_is_reproduced(self):
        namespace={'Path':Path,'os':guard.os}
        tree=ast.parse(ORIGINAL.read_bytes())
        names={'require','parse_proc_stat','parse_kib','parse_status','process_snapshot'}
        nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ORIGINAL),'exec'),namespace)
        with patch.object(Path,'read_text',side_effect=[stat_text('R'),UID,stat_text('Z')]):
            with self.assertRaisesRegex(KeyError,'RssAnon'):
                namespace['process_snapshot'](431,Path('/synthetic/proc'))

    def test_all_other_functions_identical_to_original(self):
        def functions(path):
            return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(path.read_bytes()).body
                    if isinstance(node,ast.FunctionDef)}
        old,new=functions(ORIGINAL),functions(SOURCE)
        self.assertEqual(set(old),set(new))
        self.assertEqual([name for name in old if old[name]!=new[name]],['process_snapshot'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ExitRace))
    report=dict(status='passed' if result.wasSuccessful() else 'failed',passed=result.wasSuccessful(),
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=[],
        synthetic_only=True,real_process_spawned=False,real_signal_sent=False,original_failure_receipt_changed=False,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE,ORIGINAL,Path(__file__))})
    with args.report.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    raise SystemExit(0 if result.wasSuccessful() else 1)
