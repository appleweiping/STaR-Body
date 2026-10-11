"""Finite resumable cache/evaluation stages on an already authorized shared GPU.

Copied before editing from scripts/v30/launch_v30_architecture_v3.py.
Stage commands and outputs are explicit in an immutable JSON plan. This controller
never kills another process, changes GPU settings, or selects a test checkpoint.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def atomic(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def resources(gpu, storage):
    values = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True,
        timeout=10)
    free, utilization = (int(v.strip()) for v in values.strip().split(','))
    memory = int(next(line for line in Path('/proc/meminfo').read_text().splitlines()
                      if line.startswith('MemAvailable:')).split()[1]) * 1024
    return dict(gpu_free_mib=free, gpu_utilization_percent=utilization,
                host_available_bytes=memory, disk_free_bytes=shutil.disk_usage(storage).free)


def process_identity(pid):
    """Linux process birth identity, not a PID alone (which can be reused)."""
    directory = Path('/proc') / str(int(pid))
    try:
        stat = (directory / 'stat').read_text()
        fields = stat[stat.rfind(')') + 2:].split()
        if fields[0] == 'Z':
            return None
        command = [part.decode('utf-8', errors='replace')
                   for part in (directory / 'cmdline').read_bytes().split(b'\0') if part]
        second = (directory / 'stat').read_text()
        second_fields = second[second.rfind(')') + 2:].split()
        if fields[19] != second_fields[19]:
            raise RuntimeError('Process changed while its identity was being read')
        return dict(pid=int(pid), start_ticks=int(fields[19]), cmdline=command,
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    uid=directory.stat().st_uid)
    except FileNotFoundError:
        return None


def refuse_live_previous_child(state):
    pid = state.get('child_pid')
    if not pid:
        return
    live = process_identity(pid)
    if live is None:
        return
    saved = state.get('child_identity')
    if not saved:
        raise RuntimeError(f'Previous child PID {pid} is alive but has no birth identity; refusing a duplicate')
    birth_keys = ('pid', 'start_ticks', 'boot_id', 'uid')
    if any(live.get(key) != saved.get(key) for key in birth_keys):
        # The saved child ended and the operating system reused its PID.
        return
    if live['cmdline'] != saved.get('cmdline'):
        raise RuntimeError(f'Previous child PID {pid} is alive with the same birth identity but changed cmdline; refusing a duplicate')
    raise RuntimeError(f'Previous child PID {pid} is still running; refusing a duplicate')


def resolve_command(values):
    if not isinstance(values, list) or not values or not all(isinstance(value, str) for value in values):
        raise ValueError('Commands must be nonempty arrays of string arguments')
    return [sys.executable if value == '{python}' else value for value in values]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--physical-gpu', type=int, default=1)
    parser.add_argument('--resource-wait-seconds', type=int, default=86400)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    plan = json.loads(args.plan.read_text())
    if plan.get('test_used_for_selection') is not False or not plan.get('stages'):
        raise ValueError('Explicit finite stages and no test selection required')
    if len({stage['name'] for stage in plan['stages']}) != len(plan['stages']):
        raise ValueError('Stage names must be unique')
    for stage in plan['stages']:
        if not stage.get('outputs') or not all(isinstance(value, str) for value in stage['outputs']):
            raise ValueError('Every stage must declare its output files')
        if Path(stage['name']).name != stage['name'] or stage['name'] in ('.', '..'):
            raise ValueError('Stage names must be safe filenames')
        resolve_command(stage['command'])
        if stage.get('completion_check'):
            check = stage['completion_check']
            resolve_command(check['command'])
            if not 0 < check.get('timeout_seconds', 600) <= 86400:
                raise ValueError('completion_check timeout must be in (0,86400] seconds')
    args.work_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.work_dir / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan_sha = digest(args.plan)
    sources = {name: digest(root / name) for name in plan['source_sha256']}
    if sources != plan['source_sha256']:
        raise ValueError('Source has changed since preregistration')
    for prerequisite in plan.get('prerequisites', []):
        if digest(prerequisite['path']) != prerequisite['sha256']:
            raise ValueError('Prerequisite evidence changed')
        evidence = json.loads(Path(prerequisite['path']).read_text())
        if evidence['status'] != prerequisite['required_status']:
            raise ValueError('Prerequisite check did not pass')
    state_path = args.work_dir / 'status.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['plan_sha256'] != plan_sha or state['source_sha256'] != sources:
            raise ValueError('Cannot resume with a different plan or source')
        refuse_live_previous_child(state)
    else:
        state = dict(plan_sha256=plan_sha, source_sha256=sources, completed={}, attempts={},
                     created_unix=time.time(), test_used_for_selection=False, sota_claim=False)
    state.update(controller_pid=os.getpid(), physical_gpu=args.physical_gpu)
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(args.physical_gpu), OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
        V28D_MEMORY_LIMIT_GIB='10')

    def publish(status, **values):
        state.update(status=status, updated_unix=time.time(), **values)
        atomic(state_path, state)

    def output_records(stage):
        records = {}
        for value in stage['outputs']:
            output = Path(value)
            if not output.is_file():
                raise FileNotFoundError(output)
            records[str(output)] = dict(bytes=output.stat().st_size, sha256=digest(output))
        return records

    def verify_outputs(stage, required):
        check = stage.get('completion_check')
        if not check:
            if required:
                raise RuntimeError('Unregistered outputs require a declarative read-only completion_check: ' + stage['name'])
            return None
        command = resolve_command(check['command'])
        log_path = args.work_dir / (stage['name'] + '.verify.' + uuid.uuid4().hex + '.log')
        publish('verifying_outputs', current_stage=stage['name'], verification_log=str(log_path),
                verification_command=command, child_pid=None, child_identity=None)
        before = output_records(stage)
        with log_path.open('x') as stream:
            result = subprocess.run(command, cwd=root, env=environment, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=check.get('timeout_seconds', 600),
                                    pass_fds=(lock.fileno(),))
        if result.returncode:
            publish('verification_failed', current_stage=stage['name'], returncode=result.returncode)
            raise RuntimeError('Read-only completion check failed: ' + str(log_path))
        after = output_records(stage)
        if before != after:
            raise RuntimeError('Read-only completion check changed an output file')
        return dict(command=command, log=str(log_path), returncode=0, outputs_unchanged=True)

    for stage in plan['stages']:
        name = stage['name']
        if name in state['completed']:
            for output, record in state['completed'][name]['outputs'].items():
                if not Path(output).is_file() or digest(output) != record['sha256']:
                    raise ValueError('Completed stage output changed: ' + output)
            continue
        if {key: digest(root / key) for key in sources} != sources:
            raise ValueError('Source changed while pipeline was running')
        present = [Path(value).exists() for value in stage['outputs']]
        if any(present):
            if not all(present):
                publish('incomplete_existing_outputs', current_stage=name, child_pid=None, child_identity=None)
                raise RuntimeError('Only part of the stage outputs exist; refusing to overwrite: ' + name)
            verification = verify_outputs(stage, required=True)
            state['completed'][name] = dict(outputs=output_records(stage), reconciled=True,
                                            completion_check=verification, elapsed_seconds=None,
                                            command=resolve_command(stage['command']))
            publish('stage_reconciled', current_stage=name, child_pid=None, child_identity=None)
            print(json.dumps(dict(event='stage_reconciled', stage=name)), flush=True)
            continue
        deadline = time.monotonic() + args.resource_wait_seconds
        while True:
            try:
                usage = resources(args.physical_gpu, args.work_dir)
            except (OSError, ValueError, StopIteration, subprocess.SubprocessError) as error:
                publish('resource_probe_error', current_stage=name, resource_error=str(error))
                if time.monotonic() >= deadline:
                    raise TimeoutError('Resource checks failed until the wait deadline') from error
                time.sleep(30)
                continue
            publish('waiting_for_resources', current_stage=name, resources=usage)
            if (usage['gpu_free_mib'] >= stage.get('min_gpu_free_mib', 14336)
                and usage['host_available_bytes'] >= stage.get('min_host_available_gib', 16) * 1024**3
                and usage['disk_free_bytes'] >= stage.get('min_disk_free_gib', 40) * 1024**3):
                break
            if time.monotonic() >= deadline:
                publish('resource_timeout', current_stage=name)
                raise TimeoutError('Resource wait exceeded; no other process was modified')
            time.sleep(30)
        attempt = state['attempts'].get(name, 0) + 1
        while (args.work_dir / f'{name}.attempt{attempt:02d}.log').exists():
            attempt += 1
        state['attempts'][name] = attempt
        command = resolve_command(stage['command'])
        if stage.get('resume_marker') and Path(stage['resume_marker']).exists():
            command.append('--resume')
        log_path = args.work_dir / f'{name}.attempt{attempt:02d}.log'
        # Persist the reserved attempt before creating a log or launching a child.
        publish('starting', current_stage=name, command=command, log=str(log_path),
                child_pid=None, child_identity=None)
        with log_path.open('x') as log:
            # An inherited flock also prevents a second controller if this
            # controller dies in the small Popen -> PID-publication interval.
            child = subprocess.Popen(command, cwd=root, env=environment, stdout=log,
                                     stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
            started = time.time()
            publish('running', current_stage=name, child_pid=child.pid, command=command,
                    child_identity=process_identity(child.pid),
                    log=str(log_path), stage_started_unix=started)
            print(json.dumps(dict(event='stage_started', stage=name, pid=child.pid)), flush=True)
            while child.poll() is None:
                time.sleep(15)
                try:
                    usage = resources(args.physical_gpu, args.work_dir)
                    publish('running', resources=usage, resource_error=None)
                except (OSError, ValueError, StopIteration, subprocess.SubprocessError) as error:
                    # A transient monitoring failure must not abandon the child.
                    publish('running', resource_error=str(error))
            code = child.returncode
        if code:
            publish('failed', returncode=code, current_stage=name, child_pid=None, child_identity=None)
            raise RuntimeError(f'Stage {name} exited {code}; inspect {log_path}')
        records = output_records(stage)
        verification = verify_outputs(stage, required=False)
        state['completed'][name] = dict(outputs=records, elapsed_seconds=time.time()-started,
                                        command=command, log=str(log_path), completion_check=verification)
        publish('stage_complete', current_stage=name, child_pid=None, child_identity=None)
        print(json.dumps(dict(event='stage_complete', stage=name)), flush=True)
    publish('complete', current_stage=None, child_pid=None, child_identity=None)


if __name__ == '__main__':
    main()
