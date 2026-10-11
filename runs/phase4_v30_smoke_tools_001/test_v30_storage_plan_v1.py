"""CPU/stdlib contracts only: no actual prepare, cache load, training or SSH."""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import subprocess
import unittest
from unittest.mock import patch
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import plan_v30_pair_v1 as plan
PROJECT=Path('E:/STaR-Body')


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def functions(path,names,namespace):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    body=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
    exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


class Contracts(unittest.TestCase):
    def test_generator_only_one_line_changed(self):
        old=(PROJECT/'scripts/prepare_rebuilt_experiments_v31.py').read_text()
        new=(HERE/'prepare_rebuilt_experiments_v32_storage.py').read_text()
        self.assertEqual(new,old.replace('str(args.cache_root / filename)',
            'str((args.cache_root / filename).resolve(strict=True))'))
        self.assertEqual(sha(PROJECT/'scripts/prepare_rebuilt_experiments_v31.py'),
            'ddbb55407151d0ea616bdcaa93a3fe44b6d16be4faf7b6e813f50bce143cad05')

    def test_all_four_cache_paths_strictly_resolved(self):
        source=ast.parse((HERE/'prepare_rebuilt_experiments_v32_storage.py').read_text())
        main=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        node=next(n for n in main.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='paths' for t in n.targets))
        calls=[]
        class Cache:
            def __truediv__(self,name): return Entry(name)
        class Entry:
            def __init__(self,name): self.name=name
            def resolve(self,strict):
                calls.append((self.name,strict)); return '/physical/'+self.name
        env={'args':SimpleNamespace(cache_root=Cache()),'CACHE_NAMES':{str(i):str(i)+'.pt' for i in range(4)}}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'paths','exec'),env)
        self.assertEqual(calls,[(str(i)+'.pt',True) for i in range(4)])
        self.assertEqual(env['paths'],{str(i):'/physical/'+str(i)+'.pt' for i in range(4)})

    def test_original_prepared_checker_accepts_new_generator_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); (root/'scripts').mkdir()
            generator=root/'scripts/prepare_rebuilt_experiments_v32_storage.py'; generator.write_text('trusted new source')
            inp=root/'input.json'; inp.write_text('{}')
            cfg=root/'config.json'; cfg.write_text('{"batch-size":4}')
            receipt={'status':'prepared_not_run','test_used_for_selection':False,
                'template':{'path':str(inp),'sha256':sha(inp)},'baseline':{'path':str(inp),'sha256':sha(inp)},
                'source_sha256':{'scripts/prepare_rebuilt_experiments_v32_storage.py':sha(generator)},
                'experiments':{str(i):{'config':{'batch-size':4},'config_file':{'path':str(cfg),'sha256':sha(cfg)}} for i in range(10)}}
            report=root/'plan.json'; report.write_text(json.dumps(receipt))
            env={'__file__':str(root/'scripts/verify_experiment_stage_v31.py'),'argparse':argparse,
                 'hashlib':hashlib,'json':json,'Path':Path}
            functions(PROJECT/'scripts/verify_experiment_stage_v31.py',{'digest','require','normalize','main'},env)
            with patch.object(sys,'argv',['checker','--kind','prepared','--output',str(report)]): env['main']()
            generator.write_text('changed')
            with patch.object(sys,'argv',['checker','--kind','prepared','--output',str(report)]):
                with self.assertRaises(ValueError): env['main']()

    def test_original_config_logic_and_pair_hyperparameters_unchanged(self):
        env={'deepcopy':deepcopy,'Path':Path,'V31_SWITCHES':('frame-shape-delta-scale','local-radius-schedule','deduplicate-shape-consistency')}
        functions(HERE/'prepare_rebuilt_experiments_v32_storage.py',{'build_configs'},env)
        template=json.loads((PROJECT/'configs/star_body_v31.compat.example.json').read_text())
        configs=env['build_configs'](template,{'train-hmr-sidecar':'/root/physical.pt'},Path('/root/training'),
            Path('/baseline.json'),'bf16',{'within':2.,'boundary':2.},22.)
        self.assertEqual(len(configs),10)
        series=[configs['smoke_v30_'+mode]['config'] for mode in ('uninterrupted','paused','resume')]
        stripped=[{k:v for k,v in c.items() if k not in {'output-dir','resume-checkpoint','stop-after'}} for c in series]
        self.assertEqual(stripped[0],stripped[1]); self.assertEqual(stripped[1],stripped[2])
        for c in series:
            self.assertEqual((c['batch-size'],c['amp-dtype'],c['seed'],c['max-steps'],c['eval-every']),(4,'bf16',2026,4,2))
            self.assertEqual((c['max-train-windows'],c['max-val-windows']),(0,0))
            self.assertEqual(c['train-hmr-sidecar'],'/root/physical.pt')
            self.assertFalse(set(env['V31_SWITCHES']) & c.keys())

    def test_five_stages_only(self):
        value=plan.template()
        self.assertFalse(value['ready_for_execution'])
        self.assertEqual([s['name'] for s in value['stages']],['prepare','uninterrupted','paused','resume','compare'])
        self.assertFalse(value['formal_training_authorized']); self.assertFalse(value['v31_training_authorized'])
        for s in value['stages']:
            self.assertFalse(any('train_from_config_v31' in x or '6000' in x for x in s['command']))
            self.assertFalse(any('\\' in x for x in s['command']))

    def test_snapshot_and_exact_original_entry(self):
        s={s['name']:s for s in plan.stages()}
        self.assertIn('scripts/run_paused_training_stage_v31.py',s['paused']['command'])
        self.assertIn('--snapshot-prefix',s['paused']['command'])
        self.assertIn('--checkpoint',s['paused']['check'])
        self.assertIn('scripts/verify_training_resume_v31.py',s['compare']['command'])
        self.assertIn('resume-result',s['compare']['check'])

    def test_disk_budget_no_cross_device_addition(self):
        k,b=plan.ESTIMATED_K,plan.ESTIMATED_B
        self.assertEqual(plan.remaining_budget('uninterrupted'),max(4*plan.GIB,2*plan.GIB+plan.MARGIN+4*k+2*b))
        self.assertGreaterEqual(plan.remaining_budget('prepare'),4*plan.GIB)
        self.assertEqual(plan.remaining_budget('paused',k,b),2*plan.GIB+plan.MARGIN+max(3*k+b,2*k+2*b))
        self.assertEqual(plan.remaining_budget('resume',k,b),2*plan.GIB+plan.MARGIN+max(k,b))
        self.assertEqual(plan.remaining_budget('compare'),2*plan.GIB+plan.MARGIN)

    def test_draft_ast(self):
        for path in HERE.glob('*.py'): ast.parse(path.read_text(encoding='utf-8'))

    def test_physical_gpu_one_and_allocator_ten(self):
        env={'subprocess':subprocess}
        functions(HERE/'run_v30_pair_draft_v1.py',{'require','gpu_available'},env)
        with patch.object(subprocess,'check_output',return_value='0, 999\n1, 15000\n'):
            self.assertEqual(env['gpu_available'](),15000)
        source=(HERE/'run_v30_pair_draft_v1.py').read_text()
        self.assertIn("'CUDA_VISIBLE_DEVICES':'1'",source)
        self.assertIn("'V28D_MEMORY_LIMIT_GIB':'10'",source)

    def test_guard_plan_binding_has_only_path_and_sha(self):
        tree=ast.parse((HERE/'run_v30_pair_draft_v1.py').read_text())
        matches=[node for node in ast.walk(tree) if isinstance(node,ast.Dict) and
            any(isinstance(k,ast.Constant) and k.value=='plan' for k in node.keys)]
        bindings=[node for node in matches if any(isinstance(k,ast.Constant) and k.value=='files' for k in node.keys)]
        self.assertEqual(len(bindings),1)
        value=bindings[0].values[[k.value for k in bindings[0].keys].index('plan')]
        self.assertEqual({k.value for k in value.keys},{'path','sha256'})

    def test_original_checker_json_is_required(self):
        env={'Path':Path,'json':json,'digest':lambda p:'correct'}
        functions(HERE/'run_v30_pair_draft_v1.py',{'require','accepted_stage'},env)
        output=Path('/output.json')
        good={'status':'passed','kind':'training','artifact':str(output),'sha256':'correct','metrics_recomputed':False}
        self.assertEqual(env['accepted_stage'](json.dumps(good),'uninterrupted',output),good)
        for bad in (dict(good,status='failed'),dict(good,sha256='wrong'),dict(good,kind='prepared')):
            with self.assertRaises(ValueError): env['accepted_stage'](json.dumps(bad),'uninterrupted',output)
        with self.assertRaises(ValueError): env['accepted_stage']('exitcode=0','uninterrupted',output)

    def test_resume_budget_measures_paused_and_snapshot(self):
        class FakePath:
            def __init__(self,value): self.value=value
            def __truediv__(self,part): return FakePath(self.value+'/'+str(part))
            def __str__(self): return self.value
            def glob(self,pattern): return []
            def is_file(self): return True
            def is_symlink(self): return False
            def stat(self):
                size=900 if self.value.endswith('v30_step2.pt') else 800 if 'v30_resumed/latest' in self.value else 100
                return SimpleNamespace(st_size=size,st_blocks=2,st_mtime_ns=1,st_ino=2)
        env={'digest':lambda p:'sha'}
        functions(HERE/'run_v30_pair_draft_v1.py',{'require','checkpoint_sizes'},env)
        value=env['checkpoint_sizes'](FakePath('/run'),True)
        self.assertEqual(len(value['files']),5)
        self.assertEqual(value['K'],1024)
        self.assertEqual(value['B'],1024)
        self.assertTrue(any(row['path'].endswith('v30_step2.pt') for row in value['files']))

    def test_current_config_and_asset_bindings_present(self):
        source=(HERE/'run_v30_pair_draft_v1.py').read_text()
        self.assertIn("asset['sha256']==accepted['output_sha256']",source)
        self.assertIn("prepared['baseline']['sha256']==accepted['output_sha256']",source)
        self.assertIn("state.get('generated_bindings',[])",source)
        self.assertIn("state.get('prepared_asset_stats',[])",source)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--report',type=Path,default=HERE/'cpu_contract_tests_v1.json')
    arguments=parser.parse_args()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'status':'passed' if result.wasSuccessful() else 'failed','passed':result.wasSuccessful(),
        'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'skipped':[],
        'synthetic_only':True,'training_started':False,'generator_main_executed':False,
        'sources':{p.name:sha(p) for p in HERE.glob('*.py')}}
    with arguments.report.open('x') as f: json.dump(report,f,indent=2)
    raise SystemExit(0 if result.wasSuccessful() else 1)
