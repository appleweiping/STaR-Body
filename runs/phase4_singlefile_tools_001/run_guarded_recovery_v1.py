"""Finite CPU-only HMR recovery and original acceptance, with frozen bindings.

This new orchestration file does not alter model, training, cache contracts, or
old failed runs. It uses the previously reviewed guard without modifying it.
"""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
GIB = 1024**3
PARENT_SHA = 'cc5d7362cfc1753d4bb29b5c6b3b0f3c61153f0115d1094083b01066cfb694b9'
GUARD_SHA = 'e74efb26f5fe6ec2c91a8537737ca0e6b70c692389beb5a961be0ce686a5f31a'
TOOLS = 'runs/phase4_singlefile_tools_001/'
RUNNER = TOOLS+'run_guarded_recovery_v1.py'
HELPER = TOOLS+'recover_hmr_singlefile_v1.py'
COPY = TOOLS+'cache_hmr2a_rgb_anchor_v32_scratch.copy.py'
GUARD = 'runs/phase3_splitdisk_tools_001/run_memory_disk_guard_v2.py'
COPY_SHA = 'f0acd326968e88c4bd5cc4ae5bccc0b15441d08b47387ec76eac1377073c62eb'
CONTRACT_SHA = '435ad29249f1d7e3aa9e4488e92584a710612dafd0958f20288682386e6388c2'
LOGICAL = '/media/lenovo/data2/v28d_parallel_20261002/data/star_body_v31_cache_20261008/hmr_train_s8_raw25.pt'
RUN_DIR = '/home/lenovo/star_body_scratch_20261009/phase4_hmr_recovery_001'


def require(value, message):
    if not value:
        raise ValueError(message)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def accepted_check(text, output, output_sha):
    records = []
    for line in text.splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict) and item.get('kind') == 'hmr':
            records.append(item)
    require(len(records) == 1, 'Expected one original HMR acceptance result')
    value = records[0]
    expected = dict(status='passed', kind='hmr', split='train', windows=2909,
        frames=22735, output=str(Path(output).resolve(strict=True)), output_sha256=output_sha,
        formal_protocol_satisfied=True, development_only=False, cache_development_only=False,
        read_only=True, optimizer_steps=0)
    require(all(type(value.get(k)) is type(v) and value[k] == v for k, v in expected.items()),
            'Full original acceptance differs from the registered protocol')
    return value


def validate_review(guard, plan, review):
    require(review.get('status') == 'passed' and review.get('real_tensor_parity') is True
            and review.get('crc_valid') is True and review.get('formal_recovery_started') is False
            and str(review.get('torch_version','')).startswith('2.5.1'), 'Review incomplete')
    needed = {key:plan['source_sha256'][key] for key in (RUNNER, HELPER, COPY, GUARD)}
    require(review.get('source_sha256') == needed, 'Review source set incomplete or changed')
    require(needed[GUARD] == GUARD_SHA and needed[COPY] == COPY_SHA, 'Frozen dependency changed')
    checks = review.get('checks')
    require(isinstance(checks,dict) and set(checks) == {'mechanism','real_parity','runner'},
            'Exactly three mandatory review reports are required')
    require(len({str(Path(item['path']).resolve()) for item in checks.values()}) == 3,
            'Mandatory reports must be distinct files')
    reports = {}
    for name, item in checks.items():
        require(guard.fingerprint(item['path']) == item, 'Review evidence changed')
        report = guard.parse_json(Path(item['path']).read_bytes())
        require(report.get('status') == 'passed' and report.get('passed') is True,
                'Report did not pass: '+name)
        sources = report.get('source_sha256')
        require(isinstance(sources,dict) and bool(sources)
                and all(plan['source_sha256'].get(k) == v for k,v in sources.items()),
                'Report source SHA binding differs: '+name)
        require((RUNNER if name == 'runner' else HELPER) in sources, 'Report entry point not bound')
        if name in ('mechanism','runner'):
            require(type(report.get('tests')) is int and report['tests'] >= 1
                    and type(report.get('failures')) is int and report['failures'] == 0
                    and type(report.get('errors')) is int and report['errors'] == 0
                    and report.get('skipped') == [], 'Incomplete mechanism suite: '+name)
        else:
            require(report.get('real_tensor_parity') is True and report.get('crc_valid') is True
                    and report.get('original_input_unchanged') is True
                    and report.get('model_forward_performed') is False
                    and type(report.get('frames')) is int and report['frames'] == 16
                    and type(report.get('tensor_fields')) is int and report['tensor_fields'] == 5
                    and str(report.get('torch_version','')).startswith('2.5.1'), 'Incomplete real parity')
        reports[name] = report
    return reports


def validate_stages(plan, root, guard):
    require(plan['logical_output'] == LOGICAL and plan['original_work_dir'] == LOGICAL+'.parts'
            and plan['run_dir'] == RUN_DIR, 'Dedicated storage layout changed')
    require(plan['physical_output'] == RUN_DIR+'/helper/hmr_train_s8_raw25.pt', 'Wrong final path')
    require([s['name'] for s in plan['stages']] == ['recover_hmr','verify_hmr'], 'Wrong finite stages')
    expected_recover = [sys.executable,str(root/HELPER),'--parts-dir',LOGICAL+'.parts',
        '--run-dir',RUN_DIR+'/helper','--expected-contract-sha256',CONTRACT_SHA,
        '--copy-source',str(root/COPY),'--reserve-bytes',str(2*GIB),'--publish-alias']
    expected_verify = ['bash','scripts/run_star_body_v31.sh','scripts/verify_rebuild_stage_v31.py',
        '--kind','hmr','--output',LOGICAL,'--split','train','--expected-windows','2909',
        '--expected-frames','22735','--work-dir',LOGICAL+'.parts']
    for stage, command, minimum in zip(plan['stages'],(expected_recover,expected_verify),
            (22735*492436+2*GIB+256*1024**2,2*GIB+256*1024**2)):
        require(stage['command'] == command, 'Reviewed command changed: '+stage['name'])
        bound = stage['disk_policy']
        require(guard.fingerprint(bound['path']) == bound, 'Policy changed')
        policy = guard.read_disk_policy(bound['path'],bound['sha256'])
        require(len(policy['directories']) == 1, 'Only the written root filesystem is budgeted')
        row = policy['directories'][0]
        require(row['path'] == RUN_DIR and row['min_start_free_bytes'] == minimum
                and row['min_runtime_free_bytes'] == 2*GIB, 'Recovery disk policy changed')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    import hashlib
    require(hashlib.sha256(args.plan.read_bytes()).hexdigest() == args.plan_sha256,
            'Plan SHA changed')
    plan = json.loads(args.plan.read_text())
    root = Path(plan['project_root']).resolve(strict=True)
    guard_path = root/'runs/phase3_splitdisk_tools_001/run_memory_disk_guard_v2.py'
    require(hashlib.sha256(guard_path.read_bytes()).hexdigest() == GUARD_SHA, 'Guard SHA changed')
    guard = module(guard_path, 'recovery_memory_disk_guard')
    controller = module(root/'scripts/rebuild_pipeline_v31_v1.py', 'recovery_identity')
    require(plan['format'] == 'star-body-singlefile-recovery-plan-v1', 'Wrong plan type')
    require(plan['numerical_protocol_change'] is False and plan['model_forward'] is False
            and plan['optimizer_steps'] == 0 and plan['test_used'] is False,
            'This runner is only storage recovery')
    parent = root/'runs/full_rebuild_pipeline_003/pipeline_plan.json'
    require(guard.fingerprint(parent)['sha256'] == PARENT_SHA, 'Parent plan changed')
    original = guard.parse_json(parent.read_bytes())
    require(all(plan['source_sha256'].get(k) == v for k, v in original['source_sha256'].items()),
            'Original source binding removed or altered')
    require([s['name'] for s in plan['stages']] == ['recover_hmr', 'verify_hmr'], 'Wrong finite stages')
    require(plan['max_resource_wait_seconds'] == 3600, 'Wait bound changed')
    validate_stages(plan, root, guard)
    run = Path(plan['run_dir']).resolve(strict=True)
    require(not (run/'status.json').exists(), 'Run already has status; do not replay')
    lock_path = root/'runs/full_rebuild_pipeline_001/state/controller.lock'
    lock = lock_path.open('r+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fd = lock.fileno()
    os.set_inheritable(fd, True)
    lock_identity = guard.check_inherited_lock(fd, root)
    state = dict(format='star-body-singlefile-recovery-run-v1', status='prepared',
        identity=controller.process_identity(os.getpid()), cwd=str(Path.cwd().resolve()),
        lock=lock_identity, plan=guard.fingerprint(args.plan), completed={}, started_unix=time.time(),
        old_failed_state_modified=False, automatic_retry=False, full_training_started=False)

    def publish(status, **values):
        state.update(status=status, updated_unix=time.time(), **values)
        controller.atomic(run/'status.json', state)

    def fixed():
        require(guard.fingerprint(args.plan)['sha256'] == args.plan_sha256, 'Plan changed')
        for name, sha in plan['source_sha256'].items():
            require(guard.fingerprint(root/name)['sha256'] == sha, 'Source changed: '+name)
        for item in plan['evidence_bindings']:
            require(guard.fingerprint(item['path']) == item, 'Evidence changed: '+item['path'])
        review = guard.parse_json(Path(plan['review_receipt']['path']).read_bytes())
        require(guard.fingerprint(plan['review_receipt']['path']) == plan['review_receipt'], 'Review changed')
        validate_review(guard, plan, review)
        guard.check_inherited_lock(fd, root)

    environment = {**os.environ, 'PYTHONUNBUFFERED':'1', 'PYTHONDONTWRITEBYTECODE':'1',
        'OMP_NUM_THREADS':'4', 'MKL_NUM_THREADS':'4', 'CUDA_VISIBLE_DEVICES':'',
        'STAR_BODY_CONTROLLER_LOCK_FD':str(fd)}
    try:
        fixed()
        for stage in plan['stages']:
            name = stage['name']
            policy_path = Path(stage['disk_policy']['path'])
            policy_sha = stage['disk_policy']['sha256']
            deadline = time.monotonic()+plan['max_resource_wait_seconds']
            while True:
                policy = guard.read_disk_policy(policy_path, policy_sha)
                disks = guard.disk_sample(policy)
                available = guard.parse_meminfo(Path('/proc/meminfo').read_text())
                publish('waiting_for_resources', current_stage=name, child_pid=None,
                    mem_available_bytes=available, disk_resources=disks)
                if available >= 8*GIB and guard.disk_limit_reason(disks, 'start') is None:
                    break
                require(time.monotonic() < deadline, 'One-hour resource wait expired')
                time.sleep(30)
            fixed()
            command = run/(name+'.command.json')
            bindings = run/(name+'.bindings.json')
            guard.json_file_new(command, stage['command'])
            guard.json_file_new(bindings, dict(plan=dict(path=str(args.plan.resolve()),sha256=args.plan_sha256),
                files=[dict(path=str(root/k),sha256=v) for k,v in plan['source_sha256'].items()]))
            out = run/(name+'.guard')
            argv = [sys.executable,str(guard_path),'--command-json',str(command),'--bindings-json',str(bindings),
                '--expected-command-sha256',controller.digest(command),'--expected-bindings-sha256',controller.digest(bindings),
                '--project-root',str(root),'--output-dir',str(out),'--inherited-lock-fd',str(fd),
                '--min-start-available-gib','8','--min-runtime-available-gib','2','--max-child-anon-gib','4',
                '--poll-seconds','1','--timeout-seconds','7200','--disk-policy-json',str(policy_path),
                '--expected-disk-policy-sha256',policy_sha]
            before = guard.fingerprint(plan['physical_output']) if name == 'verify_hmr' else None
            with (run/(name+'.guard.log')).open('x') as log:
                child = subprocess.Popen(argv,cwd=root,env=environment,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,pass_fds=(fd,))
                publish('running',current_stage=name,child_pid=child.pid,
                    child_identity=controller.process_identity(child.pid),command=argv)
                code = child.wait()
            require(code == 0, 'Guard/child failed; preserve all evidence')
            receipt = guard.parse_json((out/'receipt.json').read_bytes())
            require(receipt['status'] == 'completed' and receipt['returncode'] == 0
                    and receipt['monitoring_errors'] == 0, 'Guard did not certify complete')
            require(receipt['bindings']['command_json']['sha256'] == controller.digest(command)
                    and receipt['bindings']['guard']['sha256'] == GUARD_SHA, 'Guard binding differs')
            fixed()
            if name == 'verify_hmr':
                after = guard.fingerprint(plan['physical_output'])
                require(after == before, 'Verifier changed cache')
                accepted = accepted_check((out/'child.log').read_text(),plan['physical_output'],after['sha256'])
                guard.json_file_new(run/'hmr_train.acceptance.json', dict(**accepted, output_record=after,
                    guard_receipt=guard.fingerprint(out/'receipt.json'),verifier_log=guard.fingerprint(out/'child.log'),
                    plan=guard.fingerprint(args.plan), storage_alias=plan['logical_output'],
                    original_work_dir=plan['original_work_dir']))
            state['completed'][name] = guard.fingerprint(out/'receipt.json')
            publish('stage_completed',child_pid=None,child_identity=None)
        publish('completed',current_stage=None,acceptance=guard.fingerprint(run/'hmr_train.acceptance.json'),
            next_step='Prepare a new experiment plan with explicit relocated-cache acceptance; do not run old 003')
    except (Exception,KeyboardInterrupt) as error:
        publish('failed',error=repr(error),child_pid=state.get('child_pid'))
        raise


if __name__ == '__main__':
    main()
