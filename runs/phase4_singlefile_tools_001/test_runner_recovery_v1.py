"""Independent pure-function tests; no runner main, process, or signal calls."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
import uuid

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'run_guarded_recovery_v1.py'
SOURCE_BYTES=SOURCE.read_bytes()
tree=ast.parse(SOURCE_BYTES)
constants={'GIB','PARENT_SHA','GUARD_SHA','TOOLS','RUNNER','HELPER','COPY','GUARD','COPY_SHA','CONTRACT_SHA','LOGICAL','RUN_DIR'}
functions={'require','accepted_check','validate_review','validate_stages'}
nodes=[node for node in tree.body if
       (isinstance(node,ast.FunctionDef) and node.name in functions) or
       (isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id in constants for target in node.targets))]
module=ModuleType('reviewed_recovery_runner_functions')
module.__dict__.update(Path=Path,json=json,sys=sys)
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),module.__dict__)


def fingerprint(path):
    path=Path(path).resolve(); raw=path.read_bytes()
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


class ReviewFixture(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='runner_review_',dir=HERE)
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        self.guard=SimpleNamespace(fingerprint=fingerprint,parse_json=json.loads,
            read_disk_policy=lambda path,sha:json.loads(Path(path).read_bytes()))
        self.sources={module.RUNNER:hashlib.sha256(SOURCE_BYTES).hexdigest(),
            module.HELPER:'a'*64,module.COPY:module.COPY_SHA,module.GUARD:module.GUARD_SHA}
        self.plan=dict(source_sha256=copy.deepcopy(self.sources))
        self.reports={
            'mechanism':dict(status='passed',passed=True,tests=25,failures=0,errors=0,skipped=[],
                source_sha256={module.HELPER:self.sources[module.HELPER]}),
            'runner':dict(status='passed',passed=True,tests=30,failures=0,errors=0,skipped=[],
                source_sha256={module.RUNNER:self.sources[module.RUNNER]}),
            'real_parity':dict(status='passed',passed=True,real_tensor_parity=True,crc_valid=True,
                original_input_unchanged=True,model_forward_performed=False,frames=16,tensor_fields=5,
                torch_version='2.5.1+fixture',source_sha256={module.HELPER:self.sources[module.HELPER]})}
        self.review=dict(status='passed',real_tensor_parity=True,crc_valid=True,
            formal_recovery_started=False,torch_version='2.5.1+fixture',source_sha256=copy.deepcopy(self.sources))
        self.publish_reports()

    def publish_reports(self):
        self.review['checks']={}
        for key,value in self.reports.items():
            path=self.root/(key+'.json');path.write_text(json.dumps(value),encoding='utf-8')
            self.review['checks'][key]=fingerprint(path)

    def validate(self): return module.validate_review(self.guard,self.plan,self.review)
    def reject(self):
        with self.assertRaises((ValueError,KeyError,TypeError,OSError)): self.validate()

    def test_complete_reports_accepted(self):
        self.assertEqual(set(self.validate()),{'mechanism','runner','real_parity'})

    def test_empty_review_source_rejected(self):
        self.review['source_sha256']={};self.reject()

    def test_missing_or_extra_review_sources_rejected(self):
        original=copy.deepcopy(self.review)
        for key in self.sources:
            self.review=copy.deepcopy(original);del self.review['source_sha256'][key]
            with self.subTest(key=key): self.reject()
        self.review=copy.deepcopy(original);self.review['source_sha256']['unreviewed.py']='b'*64;self.reject()

    def test_changed_frozen_dependency_even_if_plan_matches_rejected(self):
        for key in (module.GUARD,module.COPY):
            original=self.plan['source_sha256'][key]
            self.plan['source_sha256'][key]=self.review['source_sha256'][key]='b'*64
            with self.subTest(key=key): self.reject()
            self.plan['source_sha256'][key]=self.review['source_sha256'][key]=original

    def test_empty_missing_or_extra_reports_rejected(self):
        original=copy.deepcopy(self.review['checks'])
        self.review['checks']={};self.reject()
        for key in original:
            self.review['checks']=copy.deepcopy(original);del self.review['checks'][key]
            with self.subTest(key=key): self.reject()
        self.review['checks']=copy.deepcopy(original);self.review['checks']['extra']=original['runner'];self.reject()

    def test_failed_actual_report_even_rehashed_rejected(self):
        for name in self.reports:
            self.reports[name]['passed']=False;self.publish_reports()
            with self.subTest(name=name): self.reject()
            self.reports[name]['passed']=True

    def test_failed_report_status_rejected(self):
        for name in self.reports:
            self.reports[name]['status']='failed';self.publish_reports()
            with self.subTest(name=name): self.reject()
            self.reports[name]['status']='passed'

    def test_report_sources_empty_missing_entry_or_unknown_rejected(self):
        for name in self.reports:
            original=self.reports[name]['source_sha256']
            for value in ({},{module.GUARD:module.GUARD_SHA},{'unknown.py':'a'*64}):
                self.reports[name]['source_sha256']=value;self.publish_reports()
                with self.subTest(name=name,value=value): self.reject()
            self.reports[name]['source_sha256']=original

    def test_report_source_changed_rejected(self):
        self.reports['runner']['source_sha256'][module.RUNNER]='c'*64
        self.publish_reports();self.reject()

    def test_mechanism_and_runner_counts_errors_skips_rejected(self):
        for name in ('mechanism','runner'):
            for key,value in (('tests',0),('tests',True),('failures',1),('errors',1),('skipped',['Linux case'])):
                original=self.reports[name][key];self.reports[name][key]=value;self.publish_reports()
                with self.subTest(name=name,key=key,value=value): self.reject()
                self.reports[name][key]=original

    def test_real_parity_flags_rejected(self):
        for key in ('real_tensor_parity','crc_valid','original_input_unchanged','model_forward_performed'):
            original=self.reports['real_parity'][key];self.reports['real_parity'][key]=not original;self.publish_reports()
            with self.subTest(key=key): self.reject()
            self.reports['real_parity'][key]=original

    def test_real_parity_sizes_and_types_rejected(self):
        for key,value in (('frames',8),('frames',16.0),('tensor_fields',4),('tensor_fields',5.0)):
            original=self.reports['real_parity'][key];self.reports['real_parity'][key]=value;self.publish_reports()
            with self.subTest(key=key,value=value): self.reject()
            self.reports['real_parity'][key]=original

    def test_actual_server_torch_version_required(self):
        self.reports['real_parity']['torch_version']='2.12.0+cpu';self.publish_reports();self.reject()

    def test_top_level_false_or_started_or_wrong_torch_rejected(self):
        for key,value in (('status','failed'),('real_tensor_parity',False),('crc_valid',False),
                          ('formal_recovery_started',True),('torch_version','2.12.0+cpu')):
            original=self.review[key];self.review[key]=value
            with self.subTest(key=key): self.reject()
            self.review[key]=original

    def test_report_hash_changed_rejected(self):
        path=self.root/'real_parity.json';path.write_bytes(path.read_bytes()+b' ');self.reject()

    def test_one_union_report_cannot_replace_three_reports(self):
        union={**self.reports['mechanism'],**self.reports['real_parity'],
               'source_sha256':copy.deepcopy(self.sources)}
        path=self.root/'union.json';path.write_text(json.dumps(union),encoding='utf-8')
        self.review['checks']={key:fingerprint(path) for key in self.reports};self.reject()


class StageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='stage_review_',dir=HERE);self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        self.guard=SimpleNamespace(fingerprint=fingerprint,
            read_disk_policy=lambda path,sha:json.loads(Path(path).read_bytes()))
        commands=[
            [module.sys.executable,str(self.root/module.HELPER),'--parts-dir',module.LOGICAL+'.parts',
             '--run-dir',module.RUN_DIR+'/helper','--expected-contract-sha256',module.CONTRACT_SHA,
             '--copy-source',str(self.root/module.COPY),'--reserve-bytes',str(2*module.GIB),'--publish-alias'],
            ['bash','scripts/run_star_body_v31.sh','scripts/verify_rebuild_stage_v31.py','--kind','hmr',
             '--output',module.LOGICAL,'--split','train','--expected-windows','2909',
             '--expected-frames','22735','--work-dir',module.LOGICAL+'.parts']]
        self.policies=[];self.plan=dict(logical_output=module.LOGICAL,original_work_dir=module.LOGICAL+'.parts',
            run_dir=module.RUN_DIR,physical_output=module.RUN_DIR+'/helper/hmr_train_s8_raw25.pt',stages=[])
        for index,(name,minimum) in enumerate((('recover_hmr',22735*492436+2*module.GIB+256*1024**2),
                                               ('verify_hmr',2*module.GIB+256*1024**2))):
            self.policies.append(dict(format='star-body-disk-policy-v1',directories=[dict(path=module.RUN_DIR,
                device=2066,inode=100,min_start_free_bytes=minimum,min_runtime_free_bytes=2*module.GIB)]))
            self.plan['stages'].append(dict(name=name,command=commands[index]))
        self.publish_policies()

    def publish_policies(self):
        for index,policy in enumerate(self.policies):
            path=self.root/('policy'+str(index)+'.json');path.write_text(json.dumps(policy),encoding='utf-8')
            self.plan['stages'][index]['disk_policy']=fingerprint(path)

    def validate(self): return module.validate_stages(self.plan,self.root,self.guard)
    def reject(self):
        with self.assertRaises((ValueError,KeyError,TypeError,OSError)): self.validate()

    def test_exact_original_verifier_and_recovery_commands_accepted(self): self.assertTrue(self.validate())

    def test_recovery_command_contract_path_source_reserve_changes_rejected(self):
        original=list(self.plan['stages'][0]['command'])
        for key in ('--parts-dir','--run-dir','--expected-contract-sha256','--copy-source','--reserve-bytes'):
            command=list(original);command[command.index(key)+1]='changed';self.plan['stages'][0]['command']=command
            with self.subTest(key=key): self.reject()
        self.plan['stages'][0]['command']=original+['--resume'];self.reject()

    def test_original_verifier_counts_split_workdir_cannot_change(self):
        original=list(self.plan['stages'][1]['command'])
        for key in ('--kind','--output','--split','--expected-windows','--expected-frames','--work-dir'):
            command=list(original);command[command.index(key)+1]='changed';self.plan['stages'][1]['command']=command
            with self.subTest(key=key): self.reject()

    def test_stage_order_extra_stage_or_training_command_rejected(self):
        original=copy.deepcopy(self.plan['stages']);self.plan['stages'].reverse();self.reject()
        self.plan['stages']=copy.deepcopy(original)+[dict(name='train')];self.reject()
        self.plan['stages']=original;self.plan['stages'][0]['command']=['python','train.py'];self.reject()

    def test_dedicated_output_paths_rejected(self):
        for key in ('logical_output','original_work_dir','run_dir','physical_output'):
            original=self.plan[key];self.plan[key]='/unreviewed/output.pt'
            with self.subTest(key=key): self.reject()
            self.plan[key]=original

    def test_reduced_or_increased_budget_rejected(self):
        for index in (0,1):
            for key in ('min_start_free_bytes','min_runtime_free_bytes'):
                for delta in (-1,1):
                    self.policies[index]['directories'][0][key]+=delta;self.publish_policies()
                    with self.subTest(index=index,key=key,delta=delta): self.reject()
                    self.policies[index]['directories'][0][key]-=delta

    def test_policy_other_path_or_extra_filesystem_rejected(self):
        original=copy.deepcopy(self.policies)
        self.policies[0]['directories'][0]['path']='/tmp';self.publish_policies();self.reject()
        self.policies=copy.deepcopy(original);self.policies[0]['directories'].append(dict(original[0]['directories'][0]))
        self.publish_policies();self.reject()

    def test_policy_file_mutation_rejected(self):
        path=Path(self.plan['stages'][0]['disk_policy']['path']);path.write_bytes(path.read_bytes()+b' ');self.reject()


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='acceptance_review_',dir=HERE);self.addCleanup(self.temp.cleanup)
        self.output=Path(self.temp.name).resolve()/'output.pt';self.output.write_bytes(b'synthetic only')
        self.sha=fingerprint(self.output)['sha256']
        self.record=dict(status='passed',kind='hmr',split='train',windows=2909,frames=22735,
            output=str(self.output),output_sha256=self.sha,formal_protocol_satisfied=True,
            development_only=False,cache_development_only=False,read_only=True,optimizer_steps=0)

    def validate(self,record=None): return module.accepted_check(json.dumps(record or self.record),self.output,self.sha)

    def test_exact_full_acceptance_accepted(self): self.assertEqual(self.validate(),self.record)

    def test_partial_validation_wrong_hash_or_output_rejected(self):
        for key,value in (('frames',16),('windows',2),('split','validation'),('output_sha256','0'*64),
                          ('output','other.pt'),('status','diagnostic_passed')):
            record=dict(self.record);record[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): self.validate(record)

    def test_boolean_or_numeric_coercion_rejected(self):
        for key,value in (('formal_protocol_satisfied',1),('development_only',0),('optimizer_steps',False),('frames',22735.0)):
            record=dict(self.record);record[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): self.validate(record)

    def test_duplicate_or_missing_acceptance_rejected(self):
        for text in (json.dumps(self.record)+'\n'+json.dumps(self.record),'warning only'):
            with self.subTest(text=text),self.assertRaises(ValueError): module.accepted_check(text,self.output,self.sha)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    stable=SOURCE.read_bytes()==SOURCE_BYTES
    passed=result.wasSuccessful() and stable
    report=dict(status='passed' if passed else 'failed',passed=passed,tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),skipped=[str(t) for t,_ in result.skipped],
        source_sha256={module.RUNNER:hashlib.sha256(SOURCE_BYTES).hexdigest()},
        test_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        synthetic_only=True,formal_recovery_started=False,runner_main_executed=False,
        real_process_started=False,real_signals_sent=False,source_stable_during_test=stable,
        details=[dict(test=str(t),traceback=trace) for t,trace in result.failures+result.errors])
    output=HERE/('runner_review_'+uuid.uuid4().hex+'.json')
    with output.open('x',encoding='utf-8') as stream: json.dump(report,stream,indent=2)
    print(json.dumps(dict(report=str(output),tests=result.testsRun,passed=passed)))
    raise SystemExit(0 if passed else 1)
