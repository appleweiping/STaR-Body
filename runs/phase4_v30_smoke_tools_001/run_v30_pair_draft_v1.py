"""Review draft: finite old-entry V30 pair using the frozen memory/disk guard.

This draft intentionally requires separately bound successful prerequisite evidence.
No automatic retry, old controller restart, source patch, V31 or formal training.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from plan_v30_pair_v1 import PROJECT, RUN, CACHE, GIB, FLOOR, MARGIN, stages, remaining_budget, ESTIMATED_K, ESTIMATED_B, template

GUARD = 'runs/phase3_splitdisk_tools_001/run_memory_disk_guard_v2.py'
GUARD_SHA = 'e74efb26f5fe6ec2c91a8537737ca0e6b70c692389beb5a961be0ce686a5f31a'
OPERATIONAL = {'output-dir', 'resume-checkpoint', 'stop-after'}


def require(ok, message):
    if not ok: raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8*1024**2), b''): result.update(data)
    return result.hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value; spec.loader.exec_module(value)
    return value


def checkpoint_sizes(run, include_paused=False):
    directory = run / 'training/smoke/v30_uninterrupted'
    entries = [('K',directory/'latest.pt'),('B',directory/'best.pt')]
    folders = [directory]
    if include_paused:
        paused = run/'training/smoke/v30_resumed'
        entries += [('K',paused/'latest.pt'),('B',paused/'best.pt'),('K',run/'evidence/v30_step2.pt')]
        folders += [paused,run/'evidence']
    residuals = [str(p) for folder in folders for p in folder.glob('*.tmp*')]
    require(not residuals,'Temporary checkpoint/snapshot residuals require independent review: '+repr(residuals))
    result = {'K':0,'B':0,'files':[],'temporary_residuals':residuals}
    for key, path in entries:
        st = path.stat()
        require(path.is_file() and not path.is_symlink(), 'Checkpoint is not an owned regular file')
        result['files'].append({'kind':key,'path': str(path), 'logical_bytes': st.st_size,
            'allocated_bytes': st.st_blocks * 512, 'mtime_ns': st.st_mtime_ns, 'inode': st.st_ino,
            'sha256': digest(path)})
        result[key]=max(result[key],st.st_size,st.st_blocks*512)
    return result


def check_configs(run):
    plan = json.loads((run/'configs/experiment_plan.json').read_text())
    require(plan['train_coverage']['frames'] == 22735 and plan['train_coverage']['acceleration_triplets'] == 22481,
            'Not complete training coverage')
    require(plan['validation_coverage']['frames'] == 10413 and plan['validation_coverage']['acceleration_triplets'] == 10226,
            'Not complete validation coverage')
    selected = []
    for mode in ('uninterrupted','paused','resume'):
        spec = plan['experiments']['smoke_v30_'+mode]; cfg = spec['config']
        require(spec['model_version'] == 'v30' and spec['kind'] == 'engine_resume_acceptance', 'Wrong experiment')
        expected = {'batch-size':4, 'val-batch-size':8, 'max-steps':4, 'eval-every':2, 'seed':2026,
            'amp-dtype':'bf16', 'max-train-windows':0, 'max-val-windows':0, 'num-workers':0,
            'verify-cache-sha256':True, 'allow-partial-validation':False}
        require(all(type(cfg.get(k)) is type(v) and cfg[k] == v for k,v in expected.items()), 'Protocol changed')
        require(not {'frame-shape-delta-scale','local-radius-schedule','deduplicate-shape-consistency'} & cfg.keys(), 'V31 option leaked')
        require(cfg['stop-after'] == (2 if mode == 'paused' else 0), 'Wrong pause boundary')
        require(bool(cfg['resume-checkpoint']) == (mode == 'resume'), 'Wrong resume source')
        require(digest(spec['config_file']['path']) == spec['config_file']['sha256'], 'Config SHA changed')
        selected.append({k:v for k,v in cfg.items() if k not in OPERATIONAL})
    require(selected[0] == selected[1] == selected[2], 'Smoke trajectories differ beyond operational arguments')
    return plan


def validate_prerequisites(plan, guard):
    pre = plan['prerequisites']
    expected = {'vace_train':('vace','train',2909,22735), 'hmr_train':('hmr','train',2909,22735),
                'vace_validation':('vace','validation',2578,10413), 'hmr_validation':('hmr','validation',2578,10413)}
    require(set(pre['cache_acceptance']) == set(expected), 'Four full cache acceptances required')
    for key, (kind, split, windows, frames) in expected.items():
        binding = pre['cache_acceptance'][key]
        require(guard.fingerprint(binding['path']) == binding, 'Acceptance evidence changed')
        result = guard.parse_json(Path(binding['path']).read_bytes())
        require(result['status'] == 'passed' and result['kind'] == kind and result['split'] == split
            and result['windows'] == windows and result['frames'] == frames
            and result['formal_protocol_satisfied'] is True and result['read_only'] is True
            and result['development_only'] is False and result['cache_development_only'] is False,
            'Cache acceptance did not pass the full frozen verifier')
        name = (f'vace_{split}_s{8 if split == "train" else 4}.pt' if kind == 'vace'
                else f'hmr_{split}_s{8 if split == "train" else 4}_raw25.pt')
        require(Path(result['output']).resolve(strict=True) == (Path(CACHE)/name).resolve(strict=True), 'Wrong accepted cache')
    for key in ('baseline_acceptance','hmr_recovery_status'):
        binding = pre[key]
        require(guard.fingerprint(binding['path']) == binding, 'Required baseline/recovery receipt changed')
        value = guard.parse_json(Path(binding['path']).read_bytes())
        if key == 'baseline_acceptance':
            require(value['status'] == 'passed' and value['kind'] == 'evaluation' and value['split'] == 'validation'
                and value['frames'] == 10413 and value['formal_protocol_satisfied'] is True, 'Original baseline check missing')
        else:
            require(value['status'] == 'completed' and set(value['completed']) == {'recover_hmr','verify_hmr'}, 'Recovery guard+original check incomplete')
            require(value['acceptance']==pre['cache_acceptance']['hmr_train'],'Recovery acceptance binding differs')


def freeze_prepared(plan, run, guard):
    prepared=check_configs(run)
    for key,binding in plan['prerequisites']['cache_acceptance'].items():
        accepted=guard.parse_json(Path(binding['path']).read_bytes())
        kind,split=key.split('_',1)
        name=('train' if split=='train' else 'val')+('_vace_cache' if kind=='vace' else '_hmr_sidecar')
        asset=prepared['assets'][name]
        require(asset['sha256']==accepted['output_sha256'] and Path(asset['path']).resolve()==Path(accepted['output']).resolve(),
                'Prepared cache differs from original full acceptance: '+key)
    accepted=guard.parse_json(Path(plan['prerequisites']['baseline_acceptance']['path']).read_bytes())
    require(prepared['baseline']['sha256']==accepted['output_sha256']
        and Path(prepared['baseline']['path']).resolve()==Path(accepted['output']).resolve(),'Prepared baseline differs from original acceptance')
    paths=[run/'configs/experiment_plan.json',run/'configs/configuration_copy_provenance.json']
    paths += [Path(spec['config_file']['path']) for spec in prepared['experiments'].values()]
    generated=[guard.fingerprint(path) for path in paths]
    assets=[]
    for asset in prepared['assets'].values():
        path=Path(asset['path']); st=path.stat()
        require(st.st_size==asset['bytes'],'Prepared asset size changed')
        assets.append(dict(asset=asset,device=st.st_dev,inode=st.st_ino,mtime_ns=st.st_mtime_ns))
    return generated,assets


def accepted_stage(text, stage, output):
    parsed=[]
    for line in text.splitlines():
        try: item=json.loads(line)
        except ValueError: continue
        if isinstance(item,dict) and 'kind' in item: parsed.append(item)
    require(len(parsed)==1,'Expected exactly one original stage-check result')
    result=parsed[0]; kind='prepared' if stage=='prepare' else 'resume-result' if stage=='compare' else 'training'
    require(result.get('status')=='passed' and result.get('kind')==kind
        and Path(result.get('artifact','')).resolve()==Path(output).resolve()
        and result.get('sha256')==digest(output) and result.get('metrics_recomputed') is False,'Original stage JSON acceptance differs')
    return result


def gpu_available():
    rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'], text=True)
    values = {int(a.strip()):int(b.strip()) for a,b in (row.split(',') for row in rows.splitlines())}
    require(1 in values, 'Physical GPU 1 missing from snapshot')
    return values[1]


def audit_metric_counts(run, mode):
    directory = run/'training/smoke'/('v30_uninterrupted' if mode == 'uninterrupted' else 'v30_resumed')
    rows = [json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
    expected = {(0,'initial'), (2,'student'), (2,'ema')}
    if mode != 'paused': expected |= {(4,'student'), (4,'ema')}
    require({(row['step'],row['variant']) for row in rows} == expected and len(rows) == len(expected), 'Evaluation trajectory differs')
    require(all(row['validation']['frames'] == 10413 and row['validation']['acceleration_triplets'] == 10226 for row in rows),
            'A training evaluation is partial')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--plan',type=Path,required=True); p.add_argument('--plan-sha256',required=True)
    a=p.parse_args(); require(digest(a.plan)==a.plan_sha256,'Plan changed')
    plan=json.loads(a.plan.read_text()); root=Path(PROJECT); run=Path(RUN)
    require(plan['format']=='star-body-finite-v30-pair-plan-v1' and plan['ready_for_execution'] is True,'Draft is not executable')
    require(plan['project_root']==PROJECT and plan['run_dir']==RUN and plan['stages']==stages(),'Finite path/command plan changed')
    require(plan['resources']==template()['resources'],'Declared resource policy differs from this finite runner')
    require(plan['numerical_protocol_change'] is False and plan['v31_training_authorized'] is False
        and plan['formal_training_authorized'] is False and plan['test_used'] is False,'Scope changed')
    require(digest(root/GUARD)==GUARD_SHA,'Frozen guard changed')
    guard=module(root/GUARD,'finite_v30_guard'); ctl=module(root/'scripts/rebuild_pipeline_v31_v1.py','finite_v30_identity')
    parent=plan['parent_plan']; require(guard.fingerprint(parent['path'])==parent,'Frozen parent plan changed')
    original=guard.parse_json(Path(parent['path']).read_bytes())
    require(all(plan['source_sha256'].get(k)==v for k,v in original['source_sha256'].items()),'Original source freeze removed')
    required_sources={GUARD,'scripts/prepare_rebuilt_experiments_v32_storage.py',
        'runs/phase4_v30_smoke_tools_001/run_v30_pair_draft_v1.py','runs/phase4_v30_smoke_tools_001/plan_v30_pair_v1.py'}
    require(required_sources <= set(plan['source_sha256']),'New storage/orchestration sources not bound')
    require(not (run/'status.json').exists(),'No automatic replay of an existing smoke run')
    run.mkdir(mode=0o700,parents=False,exist_ok=True); (run/'evidence').mkdir(exist_ok=False)
    lock=(root/'runs/full_rebuild_pipeline_001/state/controller.lock').open('r+')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB); fd=lock.fileno(); os.set_inheritable(fd,True)
    identity=guard.check_inherited_lock(fd,root)
    state={'status':'prepared','identity':ctl.process_identity(os.getpid()),'lock':identity,'completed':{},'formal_training_started':False}
    def publish(status,**values):
        state.update(status=status,updated_unix=time.time(),**values); ctl.atomic(run/'status.json',state)
    def fixed():
        require(digest(a.plan)==a.plan_sha256,'Plan changed')
        for name,sha in plan['source_sha256'].items(): require(digest(root/name)==sha,'Source changed: '+name)
        if state.get('prepared_config_plan'):
            require(guard.fingerprint(run/'configs/experiment_plan.json')==state['prepared_config_plan'], 'Prepared config plan changed')
        for item in state.get('generated_bindings',[]): require(guard.fingerprint(item['path'])==item,'Generated config/plan changed')
        for item in state.get('prepared_asset_stats',[]):
            st=Path(item['asset']['path']).stat()
            require((st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==
                (item['device'],item['inode'],item['asset']['bytes'],item['mtime_ns']),'Prepared asset changed since full SHA audit')
        guard.check_inherited_lock(fd,root); validate_prerequisites(plan,guard)
    env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1',
        'PYTHONUNBUFFERED':'1','CUDA_VISIBLE_DEVICES':'1','V28D_MEMORY_LIMIT_GIB':'10','STAR_BODY_CONTROLLER_LOCK_FD':str(fd)}
    try:
        fixed(); k,b=ESTIMATED_K,ESTIMATED_B
        for stage in stages():
            name=stage['name']; minimum=remaining_budget(name,k,b)
            if name!='prepare': check_configs(run)
            if name in ('paused','resume'):
                sizes=checkpoint_sizes(run,include_paused=name=='resume'); k=max(k,sizes['K'])
                b=max(b,sizes['B']); minimum=remaining_budget(name,k,b)
                guard.json_file_new(run/(name+'.checkpoint_budget.json'),dict(measured=sizes,remaining_required_free_bytes=minimum))
            for part,argv in (('run',stage['command']),('check',stage['check'])):
                fixed(); deadline=time.monotonic()+3600; required=minimum if part=='run' else FLOOR+MARGIN
                while True:
                    host=guard.parse_meminfo(Path('/proc/meminfo').read_text()); free=os.statvfs(run).f_bavail*os.statvfs(run).f_frsize
                    gpu=gpu_available() if stage['gpu'] and part=='run' else None
                    publish('waiting_for_resources',stage=name,part=part,child_pid=None,host=host,disk_free=free,gpu_free_mib=gpu)
                    if host>=12*GIB and free>=required and (gpu is None or gpu>=12288): break
                    require(time.monotonic()<deadline,'Finite resource wait expired'); time.sleep(30)
                prefix=run/(name+'.'+part); command=Path(str(prefix)+'.command.json'); bindings=Path(str(prefix)+'.bindings.json')
                guard.json_file_new(command,argv)
                bound_files=[{'path':str(root/x),'sha256':y} for x,y in plan['source_sha256'].items()]
                bound_files += [{'path':item['path'],'sha256':item['sha256']} for item in state.get('generated_bindings',[])]
                guard.json_file_new(bindings,{'plan':{'path':str(a.plan.resolve()),'sha256':a.plan_sha256},'files':bound_files})
                st=run.stat(); policy=Path(str(prefix)+'.disk.json')
                guard.json_file_new(policy,{'format':guard.DISK_FORMAT,'directories':[{'path':str(run),'device':st.st_dev,'inode':st.st_ino,
                    'min_start_free_bytes':required,'min_runtime_free_bytes':FLOOR}]})
                out=Path(str(prefix)+'.guard')
                cmd=[sys.executable,str(root/GUARD),'--command-json',str(command),'--bindings-json',str(bindings),
                    '--expected-command-sha256',digest(command),'--expected-bindings-sha256',digest(bindings),
                    '--project-root',str(root),'--output-dir',str(out),'--inherited-lock-fd',str(fd),
                    '--min-start-available-gib','12','--min-runtime-available-gib','3','--max-child-anon-gib','8',
                    '--poll-seconds','1','--timeout-seconds','7200','--disk-policy-json',str(policy),'--expected-disk-policy-sha256',digest(policy)]
                with Path(str(prefix)+'.log').open('xb') as log:
                    child=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(fd,))
                    publish('running',stage=name,part=part,child_pid=child.pid,child_identity=ctl.process_identity(child.pid))
                    code=child.wait()
                receipt=guard.parse_json((out/'receipt.json').read_bytes())
                require(code==0 and receipt['status']=='completed' and receipt['returncode']==0 and receipt['monitoring_errors']==0,
                        'Guard/stage did not complete; retain all evidence and stop')
                require(receipt['bindings']['command_json']['sha256']==digest(command)
                    and receipt['bindings']['guard']['sha256']==GUARD_SHA,'Guard command/source binding differs')
                if part=='check':
                    accepted=accepted_stage((out/'child.log').read_text(),name,argv[argv.index('--output')+1])
                    guard.json_file_new(Path(str(prefix)+'.acceptance.json'),accepted)
                state['completed'][name+'.'+part]=guard.fingerprint(out/'receipt.json')
            if name=='prepare':
                check_configs(run)
                state['prepared_config_plan']=guard.fingerprint(run/'configs/experiment_plan.json')
                state['generated_bindings'],state['prepared_asset_stats']=freeze_prepared(plan,run,guard)
            if stage['gpu']: audit_metric_counts(run,name)
            publish('stage_complete',stage=name,child_pid=None)
        publish('completed',child_pid=None,optimizer_steps_per_trajectory=4,automatic_formal_dispatch=False)
    except BaseException as error:
        publish('failed',error_type=type(error).__name__,error=str(error)); raise
    finally:
        lock.close()


if __name__=='__main__': main()
