"""One bounded, source-bound determinism probe under the independently tested guard.

No training controller, optimization, checkpoint selection or benchmark dispatch.
"""
import argparse, fcntl, hashlib, importlib.util, json, os, subprocess, sys, time
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path('/media/lenovo/data2/mjx/STaR-Body_V31_20261008')
RUN = ROOT/'runs/phase5_determinism_probe_001'
GUARD = ROOT/'runs/phase5_guard_tools_001/run_memory_disk_guard_v3.py'
GUARD_SHA = '6fd3c7972a96aac21bd2b2b787e03c9b5a149f58e6d7108454fbde49501ee9d8'
GIB = 1024**3

def require(ok, message):
    if not ok: raise ValueError(message)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def module(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True)
    a=p.parse_args();require(a.plan.resolve()==RUN/'plan.json','Unexpected run')
    require(digest(a.plan)==a.plan_sha256,'Plan changed')
    plan=json.loads(a.plan.read_text())
    require(plan['format']=='star-body-determinism-probe-plan-v1' and plan['optimizer_steps']==0
        and plan['validation_test_used'] is False and plan['automatic_training_dispatch'] is False,'Scope mismatch')
    require(digest(GUARD)==GUARD_SHA,'Guard changed')
    guard=module(GUARD,'bounded_probe_guard')
    ctl=module(ROOT/'scripts/rebuild_pipeline_v31_v1.py','bounded_probe_identity')
    require(not (RUN/'status.json').exists(),'No replay of this probe')
    command=plan['command']
    require(command[:3]==['bash','scripts/run_star_body_v31.sh','runs/phase5_determinism_tools_001/probe_determinism_v1.py'], 'Unexpected probe command')
    require('--config' in command and '--prepared-state' in command,'Real train diagnostic required')
    require(command[command.index('--output')+1]==str(RUN/'probe_result.json'),'Unexpected output')
    state=dict(status='prepared',identity=ctl.process_identity(os.getpid()),optimizer_steps=0,validation_test_used=False)
    def publish(status,**values):
        state.update(status=status,updated_unix=time.time(),**values);ctl.atomic(RUN/'status.json',state)
    def fixed():
        require(digest(a.plan)==a.plan_sha256,'Plan changed')
        for path,sha in plan['source_sha256'].items():require(digest(ROOT/path)==sha,'Source changed: '+path)
        for record in plan['bound_evidence']:require(guard.fingerprint(record['path'])==record,'Bound evidence changed')
        for record in plan['asset_stats']:
            st=Path(record['asset']['path']).stat()
            require((st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==
                (record['device'],record['inode'],record['asset']['bytes'],record['mtime_ns']),'Asset changed since prior full hash')
    with (ROOT/'runs/full_rebuild_pipeline_001/state/controller.lock').open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fd=lock.fileno();os.set_inheritable(fd,True)
        state['lock']=guard.check_inherited_lock(fd,ROOT)
        try:
            fixed();deadline=time.monotonic()+3600
            while True:
                host=guard.parse_meminfo(Path('/proc/meminfo').read_text());st=os.statvfs(RUN);free=st.f_bavail*st.f_frsize
                gpu_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
                gpu={int(x.split(',')[0]):int(x.split(',')[1]) for x in gpu_rows.splitlines()}[1]
                publish('waiting_for_resources',host_available_bytes=host,disk_free_bytes=free,gpu_1_free_mib=gpu)
                if host>=12*GIB and free>=6*GIB and gpu>=12288:break
                require(time.monotonic()<deadline,'Resource wait expired');time.sleep(30)
            fixed();guard.json_file_new(RUN/'command.json',command)
            files=[dict(path=str(ROOT/name),sha256=sha) for name,sha in plan['source_sha256'].items()]
            files.extend(dict(path=x['path'],sha256=x['sha256']) for x in plan['bound_evidence'])
            guard.json_file_new(RUN/'bindings.json',dict(plan=dict(path=str(a.plan),sha256=a.plan_sha256),files=files))
            st=RUN.stat();guard.json_file_new(RUN/'disk.json',dict(format=guard.DISK_FORMAT,directories=[
                dict(path=str(RUN),device=st.st_dev,inode=st.st_ino,min_start_free_bytes=6*GIB,min_runtime_free_bytes=4*GIB)]))
            argv=[sys.executable,str(GUARD),'--command-json',str(RUN/'command.json'),'--bindings-json',str(RUN/'bindings.json'),
                '--expected-command-sha256',digest(RUN/'command.json'),'--expected-bindings-sha256',digest(RUN/'bindings.json'),
                '--project-root',str(ROOT),'--output-dir',str(RUN/'probe.guard'),'--inherited-lock-fd',str(fd),
                '--min-start-available-gib','12','--min-runtime-available-gib','3','--max-child-anon-gib','8',
                '--poll-seconds','1','--timeout-seconds','1800','--disk-policy-json',str(RUN/'disk.json'),
                '--expected-disk-policy-sha256',digest(RUN/'disk.json')]
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'1','V28D_MEMORY_LIMIT_GIB':'10','OMP_NUM_THREADS':'4',
                'MKL_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8',
                'STAR_BODY_CONTROLLER_LOCK_FD':str(fd)}
            with (RUN/'guard.log').open('xb') as log:
                child=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env,pass_fds=(fd,))
                publish('running',child_identity=ctl.process_identity(child.pid));rc=child.wait()
            receipt=json.loads((RUN/'probe.guard/receipt.json').read_text())
            state['receipt']=guard.fingerprint(RUN/'probe.guard/receipt.json')
            require(rc==0 and receipt['status']=='completed' and receipt['monitoring_errors']==0,'Probe or monitoring failed')
            require((RUN/'probe_result.json').is_file(),'Probe result missing')
            result=json.loads((RUN/'probe_result.json').read_text())
            require(result['status']=='diagnostic_passed' and result['diagnostic_only'] is True
                and result['optimizer_steps']==0 and result['benchmark_metrics_computed'] is False
                and result['real']['status']=='diagnostic_passed','Diagnostic result did not pass')
            fixed();publish('completed',probe_result=guard.fingerprint(RUN/'probe_result.json'),automatic_training_dispatch=False)
        except BaseException as e:
            publish('failed',error_type=type(e).__name__,error=str(e));raise

if __name__=='__main__':main()
