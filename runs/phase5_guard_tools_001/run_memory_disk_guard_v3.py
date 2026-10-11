#!/usr/bin/env python3
"""Linux-only resource wrapper; no numerical/evaluation code is modified.

--command-json is a JSON array of argv strings, executed with shell=False.
--bindings-json is required for execution and has this schema:
  {"files": [{"path": "scripts/evaluate_body_v31.py", "sha256": "..."}],
   "plan": {"path": "/absolute/frozen_plan.json", "sha256": "..."}}
Relative binding/argv script paths resolve against --project-root. At least one
explicit .py/.sh argv script must be in files. Bind the complete frozen source
manifest in files, not just its entry point. Expected command/bindings digests
can be pinned by the caller. All bytes and bindings are checked again immediately
before spawning. Outputs are exclusive; failed output directories are retained.

RssAnon is summed over the child's original Linux session (not GPU memory).
Only the original process group created by start_new_session=True can be signaled.
A fresh boot/UID/start-tick/session/group identity check is required before EVERY
signal; after the leader exits a previously witnessed group member can keep the
original group identity pinned. Unknown/reused identities never authorize signals.
Escaped sessions/groups are recorded where visible but never followed or killed.
An invalid memory sample never authorizes a memory abort. Monitoring errors are
recorded, monitoring continues to child exit or the independently valid timeout,
and a nominal child success with monitoring gaps is not certified completed.

--disk-policy-json and --expected-disk-policy-sha256 are mandatory for execution.
Policy format: {"format":"star-body-disk-policy-v1", "directories":[
  {"path":"/absolute/existing/directory", "device":2080, "inode":123,
   "min_start_free_bytes":123456, "min_runtime_free_bytes":12345}]}
Each device appears once: thresholds apply independently, never to summed space.
Directory identity and SHA-pinned policy bytes are rechecked every sample.
Invalid disk samples do not authorize signals; the original timeout still applies.

--self-test runs only pure/simulated CPU mechanisms, including mocked signals.
It never spawns or signals a real process and does not require Linux or assets.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import time

FORMAT = "star-body-memory-disk-guard-v3"
DISK_FORMAT = "star-body-disk-policy-v1"
GIB = 1024 ** 3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(path):
    path = Path(path).resolve(strict=True)
    require(path.is_file(), "Not a regular file: " + str(path))
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            h.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            "File changed while hashing: " + str(path))
    return {"path": str(path), "bytes": after.st_size, "sha256": h.hexdigest()}


def parse_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON field: " + key)
            result[key] = value
        return result
    def reject_constant(value):
        raise ValueError("Nonfinite JSON token: " + value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=reject_constant)


def json_file_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def validate_disk_policy(policy):
    """One independently budgeted directory per device; free space is never added."""
    require(isinstance(policy, dict) and set(policy) == {"format", "directories"}
            and policy["format"] == DISK_FORMAT, "Invalid disk policy format")
    rows = policy["directories"]
    require(isinstance(rows, list) and 1 <= len(rows) <= 16, "Disk policy needs 1..16 directories")
    paths, identities, devices = set(), set(), set()
    fields = {"path", "device", "inode", "min_start_free_bytes", "min_runtime_free_bytes"}
    for row in rows:
        require(isinstance(row, dict) and set(row) == fields, "Invalid disk policy directory fields")
        require(isinstance(row["path"], str) and row["path"] and "\0" not in row["path"]
                and Path(row["path"]).is_absolute(), "Disk policy path must be absolute")
        require(all(type(row[key]) is int for key in fields - {"path"}), "Disk policy numbers must be integers")
        require(row["device"] >= 0 and row["inode"] > 0
                and row["min_start_free_bytes"] >= row["min_runtime_free_bytes"] > 0,
                "Invalid disk identity or positive thresholds")
        identity = (row["device"], row["inode"])
        require(row["path"] not in paths and identity not in identities, "Duplicate disk policy directory")
        require(row["device"] not in devices, "Duplicate device: budget one entry per filesystem, never sum free space")
        paths.add(row["path"]); identities.add(identity); devices.add(row["device"])
    return policy


def disk_directory_snapshot(path):
    """Pin a directory FD while querying user-available statvfs blocks."""
    directory = Path(path)
    require(str(directory.resolve(strict=True)) == path, "Disk directory path is not canonical or changed")
    before = directory.stat(follow_symlinks=False)
    require(stat.S_ISDIR(before.st_mode), "Disk policy path is not a directory")
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        opened = os.fstat(fd)
        values = os.fstatvfs(fd)
        after = directory.stat(follow_symlinks=False)
        require(stat.S_ISDIR(opened.st_mode) and stat.S_ISDIR(after.st_mode)
                and (before.st_dev, before.st_ino) == (opened.st_dev, opened.st_ino)
                == (after.st_dev, after.st_ino), "Disk directory changed during statvfs")
        require(type(values.f_bavail) is int and values.f_bavail >= 0
                and type(values.f_frsize) is int and values.f_frsize > 0, "Invalid statvfs free block sample")
        return {"device": opened.st_dev, "inode": opened.st_ino,
                "free_bytes": values.f_bavail * values.f_frsize}
    finally:
        os.close(fd)


def disk_sample(policy, probe=disk_directory_snapshot):
    validate_disk_policy(policy)
    rows = []
    for row in policy["directories"]:
        observed = probe(row["path"])
        require(observed["device"] == row["device"] and observed["inode"] == row["inode"],
                "Bound disk directory/device identity changed: " + row["path"])
        require(type(observed["free_bytes"]) is int and observed["free_bytes"] >= 0,
                "Invalid disk free bytes")
        rows.append({**row, "free_bytes": observed["free_bytes"]})
    return {"valid": True, "directories": rows}


def read_disk_policy(path, expected_sha256, probe=disk_directory_snapshot):
    require(path is not None and isinstance(expected_sha256, str) and len(expected_sha256) == 64
            and all(ch in "0123456789abcdef" for ch in expected_sha256), "Disk policy requires a pinned SHA256")
    source = Path(path).resolve(strict=True)
    require(source.is_file() and source.stat().st_size <= 65536, "Disk policy must be a small regular JSON file")
    data = source.read_bytes()
    require(digest(data) == expected_sha256, "Disk policy SHA256 changed")
    policy = validate_disk_policy(parse_json(data))
    disk_sample(policy, probe=probe)  # Missing/replaced directories cannot be accepted at bind time.
    return policy


def disk_limit_reason(sample, phase="runtime"):
    require(phase in ("start", "runtime"), "Invalid disk check phase")
    if sample is None:
        return None  # Invalid observations never authorize a disk abort.
    require(sample.get("valid") is True and bool(sample.get("directories")), "Invalid disk sample")
    for row in sample["directories"]:
        if row["free_bytes"] < row["min_" + phase + "_free_bytes"]:
            return "disk_free_below_" + phase + "_limit:device=" + str(row["device"])
    return None


def bound_disk_sample(binding, policy):
    require(fingerprint(binding["path"]) == binding, "Disk policy file identity/content changed")
    current = read_disk_policy(binding["path"], binding["sha256"])
    require(current == policy, "Bound disk policy changed")
    return disk_sample(current)


def resource_sample(birth, binding, policy):
    sample = memory_sample(birth)
    sample["disk"] = bound_disk_sample(binding, policy)
    return sample


def append_event(stream, event, **details):
    stream.write(json.dumps({"utc": utc_now(), "event": event, **details},
                            ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def best_effort_event(stream, event, **details):
    try:
        append_event(stream, event, **details)
    except (OSError, ValueError):
        try:
            print(json.dumps({"event": event, "log_write_failed": True, **details}), file=sys.stderr, flush=True)
        except OSError:
            pass


def peek_returncode(child):
    """Observe exit without reaping: keep the original leader PID pinned until all signals finish."""
    if child.returncode is not None:
        return child.returncode
    status = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    if status is None:
        return None
    return status.si_status if status.si_code == os.CLD_EXITED else -status.si_status


def parse_proc_stat(text):
    """Linux stat comm may contain spaces and right parentheses."""
    left, right = text.find("("), text.rfind(")")
    require(left > 0 and right > left, "Malformed /proc stat comm")
    pid = int(text[:left].strip())
    tail = text[right + 1:].split()
    require(len(tail) >= 20, "Truncated /proc stat")
    result = {"pid": pid, "comm": text[left + 1:right], "state": tail[0],
              "ppid": int(tail[1]), "pgrp": int(tail[2]), "session": int(tail[3]),
              "start_ticks": int(tail[19])}
    require(pid > 0 and result["start_ticks"] >= 0, "Invalid PID/start ticks")
    return result


def parse_kib(value):
    parts = value.split()
    require(len(parts) == 2 and parts[1] == "kB", "Expected Linux kB memory value")
    amount = int(parts[0])
    require(amount >= 0, "Negative memory value")
    return amount * 1024


def parse_status(text, zombie=False):
    fields = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            require(key not in fields, "Duplicate /proc status field")
            fields[key] = value.strip()
    uids = [int(item) for item in fields["Uid"].split()]
    require(len(uids) == 4 and all(item >= 0 for item in uids), "Invalid process UID")
    result = {"uids": uids}
    for field in ("RssAnon", "VmRSS", "VmSwap"):
        result[field + "_bytes"] = (0 if zombie and field not in fields
                                     else parse_kib(fields[field]))
    return result


def parse_meminfo(text):
    matches = [line.split(":", 1)[1].strip() for line in text.splitlines()
               if line.startswith("MemAvailable:")]
    require(len(matches) == 1, "Missing/duplicate MemAvailable")
    return parse_kib(matches[0])


def read_boot_id(proc_root=Path("/proc")):
    value = (proc_root / "sys/kernel/random/boot_id").read_text().strip()
    require(bool(value), "Missing Linux boot ID")
    return value


def process_snapshot(pid, proc_root=Path("/proc"), metadata=False):
    folder = proc_root / str(pid)
    first = parse_proc_stat((folder / "stat").read_text())
    status_text = (folder / "status").read_text()
    result = dict(first)
    if metadata:
        result["cmdline"] = [part.decode("utf-8", "replace") for part in
                             (folder / "cmdline").read_bytes().split(b"\0") if part]
        result["cwd"] = os.readlink(folder / "cwd")
    second = parse_proc_stat((folder / "stat").read_text())
    require(all(first[key] == second[key] for key in
                ("pid", "start_ticks", "pgrp", "session")), "Process identity changed while reading")
    # A task can exit after the first stat read and lose its mm before status.
    # Only a second, same-birth zombie observation permits absent memory fields.
    result.update(state=second["state"], **parse_status(status_text, second["state"] == "Z"))
    return result


def birth_key(item):
    return (item["pid"], item["start_ticks"], tuple(item["uids"]))


def same_birth(left, right):
    return birth_key(left) == birth_key(right)


def authorize_group_signal(birth, observed_boot, members, known):
    """Pure authorization; no stale numeric PID/group can grant permission."""
    if observed_boot != birth["boot_id"]:
        return False, "boot_id_changed"
    pid = birth["pid"]
    if not (pid > 1 and birth["pgrp"] == pid and birth["session"] == pid):
        return False, "birth_was_not_own_session_leader"
    group = [item for item in members if item["pgrp"] == pid]
    if not group:
        return False, "original_group_absent"
    if any(item["session"] != pid or item["uids"] != birth["uids"] for item in group):
        return False, "foreign_session_or_uid"
    leaders = [item for item in members if item["pid"] == pid]
    if leaders and not all(same_birth(item, birth) and item["pgrp"] == pid
                           and item["session"] == pid for item in leaders):
        return False, "leader_pid_reused_or_moved"
    if not any(item["pid"] in known and same_birth(item, known[item["pid"]]) for item in group):
        return False, "no_surviving_birth_witness"
    return True, "verified_original_process_group"


def learn_members(birth, boot_id, members, known):
    allowed, reason = authorize_group_signal(birth, boot_id, members, known)
    if allowed:
        for item in members:
            if item["pgrp"] == birth["pid"] and item["session"] == birth["pid"]:
                known[item["pid"]] = dict(item)
    return allowed, reason


def collect_session(birth, proc_root=Path("/proc")):
    """Complete current original-session sample, including other same-session groups."""
    boot_id = read_boot_id(proc_root)
    require(boot_id == birth["boot_id"], "Boot changed")
    members, errors = [], []
    for folder in proc_root.iterdir():
        if not folder.name.isdigit():
            continue
        try:
            initial = parse_proc_stat((folder / "stat").read_text())
            if not (initial["session"] == birth["pid"] or initial["pgrp"] == birth["pid"]
                    or initial["pid"] == birth["pid"]):
                continue
            item = process_snapshot(initial["pid"], proc_root)
            require(same_birth({**initial, "uids": item["uids"]}, item),
                    "PID changed during session enumeration")
            members.append(item)
        except (FileNotFoundError, ProcessLookupError):
            continue  # A process exited between two proc reads.
        except (OSError, ValueError, KeyError) as error:
            errors.append({"pid": int(folder.name), "error": str(error)})
    require(not errors, "Incomplete session sample: " + json.dumps(errors))
    return boot_id, members


def memory_sample(birth, proc_root=Path("/proc")):
    available = parse_meminfo((proc_root / "meminfo").read_text())
    boot, members = collect_session(birth, proc_root)
    session = [item for item in members if item["session"] == birth["pid"]]
    require(all(item["uids"] == birth["uids"] for item in session), "Session UID changed")
    root = next((item for item in members if item["pid"] == birth["pid"]), None)
    sample = {"mem_available_bytes": available,
              "session_RssAnon_bytes": sum(item["RssAnon_bytes"] for item in session),
              "session_VmRSS_bytes": sum(item["VmRSS_bytes"] for item in session),
              "session_VmSwap_bytes": sum(item["VmSwap_bytes"] for item in session),
              "child": root, "members": members, "boot_id": boot}
    return sample


def limit_reason(sample, elapsed, args):
    if elapsed >= args.timeout_seconds:
        return "timeout"
    if sample is None:
        return None
    if sample["mem_available_bytes"] < args.min_runtime_available_gib * GIB:
        return "mem_available_below_runtime_limit"
    if sample["session_RssAnon_bytes"] > args.max_child_anon_gib * GIB:
        return "child_session_anon_above_limit"
    if "disk" in sample:
        return disk_limit_reason(sample["disk"])
    return None


def guarded_signal(birth, known, sig, snapshot_reader, signal_sender):
    """Injectable for tests; check immediately before sending, never signal a PID directly."""
    boot, members = snapshot_reader()
    allowed, reason = learn_members(birth, boot, members, known)
    result = {"signal": int(sig), "authorized": allowed, "reason": reason,
              "group": birth["pid"], "members": members}
    if allowed:
        try:
            signal_sender(birth["pid"], sig)
            result["sent"] = True
        except ProcessLookupError:
            result.update(sent=False, reason="group_exited_before_signal")
    else:
        result["sent"] = False
    return result


def terminate_owned_group(child, birth, known, events, grace):
    reader = lambda: collect_session(birth)
    outcomes = []
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            result = guarded_signal(birth, known, sig, reader, os.killpg)
        except (OSError, ValueError, KeyError) as error:
            result = {"signal": int(sig), "authorized": False, "sent": False,
                      "reason": "identity_read_or_signal_error", "error": str(error)}
        outcomes.append(result)
        best_effort_event(events, "termination_signal", **result)
        if not result["sent"]:
            break
        deadline = time.monotonic() + (grace if sig == signal.SIGTERM else 5.0)
        while time.monotonic() < deadline:
            try:
                boot, members = reader()
                learn_members(birth, boot, members, known)
                live = [item for item in members if item["pgrp"] == birth["pid"]
                        and item["state"] != "Z"]
                if not live:
                    return outcomes
            except (OSError, ValueError, KeyError) as error:
                best_effort_event(events, "termination_observation_error", error=str(error))
            time.sleep(0.1)
    return outcomes


def supervise_after_error(child, birth, child_started, args, events, known=None,
                          clock=time.monotonic, sleeper=time.sleep, sampler=memory_sample,
                          terminator=terminate_owned_group, peeker=peek_returncode):
    """Keep ownership/lock after a guard error; only valid resource/timeout triggers authorize termination."""
    known = dict(known or ({birth["pid"]: dict(birth)} if birth is not None else {}))
    result = {"degraded_monitoring": True, "status": "process_failed", "signals": []}
    while True:
        try:
            if birth is None:
                try:
                    candidate = process_snapshot(child.pid)
                    require(candidate["pgrp"] == child.pid and candidate["session"] == child.pid
                            and candidate["uids"] == [os.getuid()] * 4, "Unverified child identity")
                    candidate["boot_id"] = read_boot_id()
                    birth = candidate
                    known[child.pid] = dict(birth)
                    best_effort_event(events, "birth_recovered_without_optional_metadata", birth=birth)
                except (OSError, ValueError, KeyError):
                    pass
            sample = None
            try:
                if birth is not None:
                    sample = sampler(birth)
                    allowed, why = learn_members(birth, sample["boot_id"], sample["members"], known)
                    require(allowed or peeker(child) is not None, "Ownership invalid: " + why)
            except (OSError, ValueError, KeyError) as error:
                sample = None
                best_effort_event(events, "degraded_sample_error", error=str(error))
            returncode = peeker(child)
            live = ([] if sample is None else [item for item in sample["members"]
                    if item["session"] == birth["pid"] and item["state"] != "Z"])
            if returncode is not None and (sample is None or not live):
                result.update(returncode=returncode, child_still_running=False,
                              descendants_verified_absent=sample is not None)
                return result
            elapsed = clock() - child_started
            reason = limit_reason(sample, elapsed, args)
            if reason and birth is not None:
                best_effort_event(events, "degraded_resource_abort", reason=reason, elapsed_seconds=elapsed)
                outcomes = terminator(child, birth, known, events, args.term_grace_seconds)
                result["signals"].extend(outcomes)
                result["termination_trigger"] = reason
                # Re-check live members; never report completed after a guard error.
            best_effort_event(events, "degraded_monitoring", elapsed_seconds=elapsed,
                              returncode=returncode, valid=sample is not None, sample=sample)
            sleeper(args.poll_seconds)
        except KeyboardInterrupt:
            best_effort_event(events, "interrupt_during_degraded_monitoring")


def resolve_binding(record, project_root):
    require(isinstance(record, dict) and set(record) == {"path", "sha256"}, "Bad binding record")
    require(isinstance(record["path"], str) and record["path"], "Bad binding path")
    expected = record["sha256"]
    require(isinstance(expected, str) and len(expected) == 64
            and all(ch in "0123456789abcdef" for ch in expected), "Bad expected SHA256")
    path = Path(record["path"])
    if not path.is_absolute():
        path = project_root / path
    actual = fingerprint(path)
    require(actual["sha256"] == expected, "Bound content SHA mismatch: " + str(path))
    return actual


def validate_lock_identity(fd, expected_path, fd_path, fd_stat, path_stat):
    """Pure identity check, separately testable without opening or locking a real file."""
    require(type(fd) is int and fd >= 3, "Inherited lock FD must be >= 3")
    require(str(expected_path) == str(fd_path), "Inherited FD is not the declared controller lock")
    require(stat.S_ISREG(fd_stat.st_mode), "Inherited lock FD is not a regular file")
    require((fd_stat.st_dev, fd_stat.st_ino) == (path_stat.st_dev, path_stat.st_ino),
            "Inherited lock FD inode differs from controller lock path")
    return {"fd": fd, "path": str(expected_path), "device": fd_stat.st_dev, "inode": fd_stat.st_ino}


def check_inherited_lock(fd, project_root):
    if fd is None:
        return None
    import fcntl
    expected = (project_root / "runs/full_rebuild_pipeline_001/state/controller.lock").resolve(strict=True)
    fd_path = os.readlink("/proc/self/fd/" + str(fd))
    identity = validate_lock_identity(fd, expected, fd_path, os.fstat(fd), expected.stat())
    # Same open-file-description flock is retained in this process and passed to
    # the child. Never unlock or close this inherited descriptor in the guard.
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity["exclusive_flock_verified"] = True
    return identity


def read_inputs(args, project_root):
    command_path = Path(args.command_json).resolve(strict=True)
    command_bytes = command_path.read_bytes()
    command_sha = digest(command_bytes)
    if args.expected_command_sha256:
        require(command_sha == args.expected_command_sha256, "Pinned command SHA mismatch")
    argv = parse_json(command_bytes)
    require(isinstance(argv, list) and argv and all(isinstance(item, str) and item
            and "\0" not in item for item in argv), "Command must be a nonempty JSON argv array")
    # Never accept shell command strings that conceal which bound script executes.
    require(not any(item.lower() in ("-c", "/c", "-command", "-encodedcommand") for item in argv[1:]),
            "Inline interpreter/shell command strings are not accepted")
    binding_path = Path(args.bindings_json).resolve(strict=True)
    binding_bytes = binding_path.read_bytes()
    binding_sha = digest(binding_bytes)
    if args.expected_bindings_sha256:
        require(binding_sha == args.expected_bindings_sha256, "Pinned bindings SHA mismatch")
    bindings = parse_json(binding_bytes)
    require(isinstance(bindings, dict) and set(bindings) == {"files", "plan"}, "Bad bindings schema")
    require(isinstance(bindings["files"], list) and bindings["files"], "No bound source files")
    sources = [resolve_binding(item, project_root) for item in bindings["files"]]
    require(len({item["path"] for item in sources}) == len(sources), "Duplicate bound source path")
    plan = resolve_binding(bindings["plan"], project_root)
    script_paths = []
    for item in argv:
        if item.endswith((".py", ".sh")):
            script = Path(item)
            script_paths.append(str((script if script.is_absolute() else project_root / script).resolve(strict=True)))
    require(script_paths and set(script_paths) <= {item["path"] for item in sources},
            "All explicit argv scripts must be bound source files")
    executable = argv[0]
    if "/" in executable or "\\" in executable:
        executable = str((Path(executable) if Path(executable).is_absolute()
                          else project_root / executable).resolve(strict=True))
    else:
        executable = shutil.which(executable)
    require(executable is not None, "Executable not found")
    return {"argv": argv, "resolved_executable": fingerprint(executable),
            "command_json": {"path": str(command_path), "bytes": len(command_bytes), "sha256": command_sha},
            "command_json_content": argv,
            "bindings_json": {"path": str(binding_path), "bytes": len(binding_bytes), "sha256": binding_sha},
            "bindings_json_content": bindings, "sources": sources, "plan": plan,
             "guard": fingerprint(Path(__file__))}


def read_bound_inputs(args, project_root):
    inputs = read_inputs(args, project_root)
    policy = read_disk_policy(args.disk_policy_json, args.expected_disk_policy_sha256)
    record = fingerprint(args.disk_policy_json)
    require(record["sha256"] == args.expected_disk_policy_sha256, "Disk policy changed while binding")
    inputs.update(disk_policy=record, disk_policy_content=policy)
    return inputs


def spawn_child(argv, project_root, child_log, executable, inherited_lock_fd=None, factory=subprocess.Popen):
    return factory(argv, executable=executable, cwd=str(project_root), stdout=child_log,
                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, shell=False,
                   close_fds=True, start_new_session=True,
                   pass_fds=() if inherited_lock_fd is None else (inherited_lock_fd,))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--command-json", type=Path)
    parser.add_argument("--bindings-json", type=Path)
    parser.add_argument("--expected-command-sha256")
    parser.add_argument("--expected-bindings-sha256")
    parser.add_argument("--disk-policy-json", type=Path)
    parser.add_argument("--expected-disk-policy-sha256")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-start-available-gib", type=float, default=8.0)
    parser.add_argument("--min-runtime-available-gib", type=float, default=2.0)
    parser.add_argument("--max-child-anon-gib", type=float, default=4.0)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--term-grace-seconds", type=float, default=5.0)
    parser.add_argument("--inherited-lock-fd", type=int, help="Inherited exclusive controller.lock open-file descriptor")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    for name in ("min_start_available_gib", "min_runtime_available_gib", "max_child_anon_gib",
                 "poll_seconds", "timeout_seconds", "term_grace_seconds"):
        require(math.isfinite(getattr(args, name)) and getattr(args, name) > 0, "Invalid positive limit: " + name)
    require(args.poll_seconds <= 1, "Disk monitoring poll interval must not exceed one second")
    require(args.min_start_available_gib >= args.min_runtime_available_gib, "Start threshold below runtime threshold")
    require(args.inherited_lock_fd is None or args.inherited_lock_fd >= 3, "Inherited lock FD must be >= 3")
    if not args.self_test:
        require(all(getattr(args, name) is not None for name in
                    ("command_json", "bindings_json", "project_root", "output_dir", "disk_policy_json", "expected_disk_policy_sha256")),
                "Execution requires command/bindings JSON, project root, new output directory and SHA-pinned disk policy")
    return args


def execute(args):
    require(sys.platform.startswith("linux"), "Execution is Linux-only; use --self-test for portable mechanism tests")
    root = args.project_root.resolve(strict=True)
    require(root.is_dir(), "Project root is not a directory")
    output = args.output_dir.resolve()
    output.mkdir(parents=False, exist_ok=False)
    started = time.monotonic()
    receipt = {"format": FORMAT, "started_utc": utc_now(), "status": "process_failed",
               "reason": "preflight_not_completed", "child_started": False, "returncode": None,
               "project_root": str(root), "output_dir": str(output),
               "monitoring_scope": "original_child_session_anon_rss_swap; bound_filesystem_free; original_group_signals_only",
               "gpu_queried": False, "numerical_code_modified": False}
    exit_code, child, birth, child_started = 1, None, None, None
    sampler = None
    with (output / "events.jsonl").open("x", encoding="utf-8", newline="\n") as events:
        try:
            inputs = read_bound_inputs(args, root)
            sampler = lambda identity: resource_sample(identity, inputs["disk_policy"], inputs["disk_policy_content"])
            inherited_lock = check_inherited_lock(args.inherited_lock_fd, root)
            boot_id = read_boot_id()
            available = parse_meminfo(Path("/proc/meminfo").read_text())
            receipt["bindings"] = inputs
            receipt["inherited_controller_lock"] = inherited_lock
            receipt["preflight_mem_available_bytes"] = available
            disks = bound_disk_sample(inputs["disk_policy"], inputs["disk_policy_content"])
            receipt["preflight_disk"] = disks
            disk_reason = disk_limit_reason(disks, "start")
            receipt["limits"] = {key: getattr(args, key) for key in
                                 ("min_start_available_gib", "min_runtime_available_gib", "max_child_anon_gib",
                                  "poll_seconds", "timeout_seconds", "term_grace_seconds")}
            json_file_new(output / "run.json", {**receipt, "guard_uid": os.getuid(), "boot_id": boot_id})
            append_event(events, "preflight", mem_available_bytes=available, boot_id=boot_id, disk=disks)
            if available < args.min_start_available_gib * GIB or disk_reason:
                receipt.update(status="resource_abort", reason=disk_reason or "mem_available_below_start_limit")
                exit_code = 75
            else:
                # Binding is immutable from initial review to launch, including guard source.
                require(read_bound_inputs(args, root) == inputs, "Input/source/plan/policy bytes changed before spawn")
                require(check_inherited_lock(args.inherited_lock_fd, root) == inherited_lock,
                        "Inherited controller lock changed before spawn")
                latest_available = parse_meminfo(Path("/proc/meminfo").read_text())
                latest_disks = bound_disk_sample(inputs["disk_policy"], inputs["disk_policy_content"])
                receipt["prelaunch_disk"] = latest_disks
                disk_reason = disk_limit_reason(latest_disks, "start")
                if latest_available < args.min_start_available_gib * GIB or disk_reason:
                    receipt.update(status="resource_abort", reason=disk_reason or "mem_available_below_start_limit",
                                   prelaunch_mem_available_bytes=latest_available)
                    append_event(events, "prelaunch_resource_abort", mem_available_bytes=latest_available, disk=latest_disks)
                    receipt.update(finished_utc=utc_now(), elapsed_seconds=time.monotonic()-started, guard_exit_code=75)
                    json_file_new(output / "receipt.json", receipt)
                    print(json.dumps({"status": receipt["status"], "reason": receipt["reason"],
                                      "receipt": str(output / "receipt.json"), "exit_code": 75}))
                    return 75
                receipt["prelaunch_mem_available_bytes"] = latest_available
                with (output / "child.log").open("xb") as child_log:
                    child = spawn_child(inputs["argv"], root, child_log, inputs["resolved_executable"]["path"],
                                        args.inherited_lock_fd)
                    receipt.update(child_started=True, child_pid=child.pid)
                    child_started = time.monotonic()
                    try:
                        birth = process_snapshot(child.pid, metadata=True)
                        birth["boot_id"] = boot_id
                        require(birth["pgrp"] == child.pid and birth["session"] == child.pid,
                                "Child is not the newly created session/group leader")
                        require(birth["uids"] == [os.getuid()] * 4, "Child UID differs from guard UID")
                        require(birth["cwd"] == str(root), "Child cwd differs from declared project root")
                        json_file_new(output / "birth.json", birth)
                    except (OSError, ValueError, KeyError) as error:
                        # Never guess an identity or signal an unverified process.
                        append_event(events, "birth_verification_failed", error=str(error))
                        receipt.update(reason="birth_verification_failed", error=str(error))
                        birth = None
                    if birth is not None:
                        receipt["birth"] = birth
                        known = {birth["pid"]: dict(birth)}
                        monitoring_errors = 0
                        peak_anon = peak_rss = peak_swap = 0
                        minimum_available = available
                        minimum_disk_free = {str(row["device"]): row["free_bytes"] for row in latest_disks["directories"]}
                        reason = None
                        with (output / "samples.jsonl").open("x", encoding="utf-8", newline="\n") as samples:
                            while True:
                                elapsed = time.monotonic() - child_started
                                sample = None
                                try:
                                    sample = sampler(birth)
                                    live_owned, ownership_reason = learn_members(birth, sample["boot_id"], sample["members"], known)
                                    # A live root without its birth witness means this is not a trustworthy sample.
                                    require(live_owned or peek_returncode(child) is not None, "Ownership check failed: " + ownership_reason)
                                    peak_anon = max(peak_anon, sample["session_RssAnon_bytes"])
                                    peak_rss = max(peak_rss, sample["session_VmRSS_bytes"])
                                    peak_swap = max(peak_swap, sample["session_VmSwap_bytes"])
                                    minimum_available = min(minimum_available, sample["mem_available_bytes"])
                                    for row in sample["disk"]["directories"]:
                                        key = str(row["device"])
                                        minimum_disk_free[key] = min(minimum_disk_free[key], row["free_bytes"])
                                    append_event(samples, "sample", elapsed_seconds=elapsed, valid=True, **sample)
                                except (OSError, ValueError, KeyError) as error:
                                    sample = None
                                    monitoring_errors += 1
                                    append_event(samples, "sample", elapsed_seconds=elapsed, valid=False, error=str(error))
                                returncode = peek_returncode(child)
                                append_event(events, "child_poll", elapsed_seconds=elapsed, returncode=returncode)
                                if returncode is not None:
                                    exit_disk_reason = (disk_limit_reason(sample["disk"])
                                                        if sample is not None else None)
                                    survivors = ([] if sample is None else
                                                 [item for item in sample["members"] if item["pid"] != birth["pid"]
                                                  and item["session"] == birth["pid"] and item["state"] != "Z"])
                                    receipt.update(returncode=returncode,
                                                   status=("resource_abort" if exit_disk_reason else
                                                           "completed" if returncode == 0 and not monitoring_errors and not survivors else "process_failed"),
                                                   reason=("descendants_outlived_leader" if survivors else
                                                           "monitoring_was_incomplete" if monitoring_errors else exit_disk_reason or "child_exit"),
                                                   surviving_session_members=survivors)
                                    if survivors:
                                        receipt["degraded_guard"] = supervise_after_error(
                                            child, birth, child_started, args, events, known, sampler=sampler)
                                    exit_code = 0 if receipt["status"] == "completed" else 75 if exit_disk_reason else 1
                                    break
                                reason = limit_reason(sample, elapsed, args)
                                if reason:
                                    append_event(events, "resource_abort", reason=reason, elapsed_seconds=elapsed)
                                    signals = terminate_owned_group(child, birth, known, events, args.term_grace_seconds)
                                    receipt.update(status="resource_abort", reason=reason, signals=signals,
                                                   returncode=peek_returncode(child), child_still_running=peek_returncode(child) is None)
                                    if receipt["child_still_running"]:
                                        receipt["degraded_guard"] = supervise_after_error(
                                            child, birth, child_started, args, events, known, sampler=sampler)
                                        receipt.update(returncode=peek_returncode(child), child_still_running=peek_returncode(child) is None)
                                    exit_code = 124 if reason == "timeout" else 75
                                    break
                                time.sleep(min(args.poll_seconds, max(0.001, args.timeout_seconds - elapsed)))
                        receipt.update(monitoring_errors=monitoring_errors, peak_session_RssAnon_bytes=peak_anon,
                                       peak_session_VmRSS_bytes=peak_rss, peak_session_VmSwap_bytes=peak_swap,
                                       minimum_mem_available_bytes=minimum_available,
                                       minimum_disk_free_bytes_by_device=minimum_disk_free,
                                       child_elapsed_seconds=time.monotonic() - child_started)
                    else:
                        receipt["degraded_guard"] = supervise_after_error(child, birth, child_started, args, events, sampler=sampler)
                        receipt.update(returncode=peek_returncode(child), child_still_running=peek_returncode(child) is None)
        except (Exception, KeyboardInterrupt) as error:
            exit_code = 1
            receipt.update(status="process_failed", reason="guard_error", error=repr(error))
            if child is not None:
                receipt["degraded_guard"] = supervise_after_error(child, birth, child_started, args, events, sampler=sampler)
                receipt.update(returncode=peek_returncode(child), child_still_running=peek_returncode(child) is None)
            best_effort_event(events, "guard_error", error=repr(error))
        if child is not None and peek_returncode(child) is not None:
            child.poll()  # All permitted signals/monitoring have finished; now reap our child.
        receipt.update(finished_utc=utc_now(), elapsed_seconds=time.monotonic() - started, guard_exit_code=exit_code)
        best_effort_event(events, "finished", status=receipt["status"], reason=receipt["reason"],
                          returncode=receipt["returncode"], guard_exit_code=exit_code)
        json_file_new(output / "receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "reason": receipt["reason"],
                      "receipt": str(output / "receipt.json"), "exit_code": exit_code}))
    return exit_code


def self_test():
    """No live proc reads, Popen, GPU access, or OS signals in this test suite."""
    import copy
    import tempfile
    import unittest
    from types import SimpleNamespace

    class GuardTests(unittest.TestCase):
        def setUp(self):
            self.birth = {"pid": 431, "pgrp": 431, "session": 431, "ppid": 100,
                          "start_ticks": 1200, "uids": [1000] * 4, "state": "S", "boot_id": "boot-a"}
            self.known = {431: dict(self.birth)}

        def test_stat_embedded_parentheses_and_spaces(self):
            tail = ["S", "100", "431", "431"] + ["0"] * 15 + ["1200"] + ["0"] * 4
            got = parse_proc_stat("431 (eval ) worker) " + " ".join(tail))
            self.assertEqual((got["pid"], got["comm"], got["start_ticks"]), (431, "eval ) worker", 1200))

        def test_status_units(self):
            got = parse_status("Uid:\t1000 1000 1000 1000\nRssAnon: 17 kB\nVmRSS: 25 kB\nVmSwap: 3 kB")
            self.assertEqual((got["RssAnon_bytes"], got["VmRSS_bytes"], got["VmSwap_bytes"]), (17*1024, 25*1024, 3*1024))

        def test_zombie_without_memory_fields(self):
            self.assertEqual(parse_status("Uid: 1000 1000 1000 1000", True)["RssAnon_bytes"], 0)

        def test_missing_memory_field_is_invalid(self):
            with self.assertRaises(KeyError):
                parse_status("Uid: 1000 1000 1000 1000\nVmRSS: 25 kB")

        def test_meminfo_missing_or_bad_units(self):
            for text in ("MemFree: 15 kB", "MemAvailable: 15 MB", "MemAvailable: -1 kB", "MemAvailable: 1 kB\nMemAvailable: 2 kB"):
                with self.subTest(text=text), self.assertRaises(ValueError):
                    parse_meminfo(text)
            self.assertEqual(parse_meminfo("MemAvailable: 8123 kB\nMemFree: 1 kB"), 8123*1024)

        def test_limits_and_invalid_sample(self):
            args = SimpleNamespace(timeout_seconds=10, min_runtime_available_gib=2, max_child_anon_gib=4)
            sample = {"mem_available_bytes": 2*GIB, "session_RssAnon_bytes": 4*GIB}
            self.assertIsNone(limit_reason(sample, 9, args))
            self.assertIsNone(limit_reason(None, 9, args))
            self.assertEqual(limit_reason(None, 10, args), "timeout")
            sample["mem_available_bytes"] -= 1
            self.assertEqual(limit_reason(sample, 1, args), "mem_available_below_runtime_limit")
            sample.update(mem_available_bytes=3*GIB, session_RssAnon_bytes=4*GIB+1)
            self.assertEqual(limit_reason(sample, 1, args), "child_session_anon_above_limit")

        def test_own_session_allowed(self):
            self.assertTrue(authorize_group_signal(self.birth, "boot-a", [self.birth], self.known)[0])

        def test_pid_reuse_rejected(self):
            reused = {**self.birth, "start_ticks": 9999}
            self.assertFalse(authorize_group_signal(self.birth, "boot-a", [reused], self.known)[0])

        def test_foreign_group_session_uid_rejected(self):
            for field, value in (("pgrp", 99), ("session", 99), ("uids", [2000]*4)):
                with self.subTest(field=field):
                    self.assertFalse(authorize_group_signal(self.birth, "boot-a", [{**self.birth, field:value}], self.known)[0])

        def test_boot_changed_rejected(self):
            self.assertFalse(authorize_group_signal(self.birth, "boot-b", [self.birth], self.known)[0])

        def test_birth_not_session_leader_rejected(self):
            for field, value in (("pgrp", 99), ("session", 99), ("pid", 1)):
                with self.subTest(field=field):
                    self.assertFalse(authorize_group_signal({**self.birth, field:value}, "boot-a", [self.birth], self.known)[0])

        def test_known_surviving_worker_pins_group(self):
            worker = {**self.birth, "pid": 432, "start_ticks": 1205}
            self.assertTrue(learn_members(self.birth, "boot-a", [self.birth, worker], self.known)[0])
            self.assertTrue(authorize_group_signal(self.birth, "boot-a", [worker], self.known)[0])
            self.assertFalse(authorize_group_signal(self.birth, "boot-a", [{**worker, "start_ticks": 1300}], self.known)[0])

        def test_unknown_group_cannot_be_adopted(self):
            worker = {**self.birth, "pid": 432, "start_ticks": 1205}
            self.assertFalse(learn_members(self.birth, "boot-a", [worker], self.known)[0])
            self.assertNotIn(432, self.known)

        def test_guarded_signal_mock_only(self):
            calls = []
            good = guarded_signal(self.birth, self.known, 15, lambda: ("boot-a", [self.birth]),
                                  lambda group, sig: calls.append((group, sig)))
            self.assertTrue(good["sent"])
            changed = {**self.birth, "start_ticks": 1300}
            bad = guarded_signal(self.birth, self.known, 9, lambda: ("boot-a", [changed]),
                                 lambda group, sig: calls.append((group, sig)))
            self.assertFalse(bad["sent"])
            self.assertEqual(calls, [(431, 15)])

        def test_degraded_monitor_uses_valid_timeout_and_keeps_failed_status(self):
            import io
            state = {"done": False, "signals": 0}
            def sampler(birth):
                member = {**birth, "state": "Z" if state["done"] else "S"}
                return {"boot_id": "boot-a", "members": [member],
                        "mem_available_bytes": 3*GIB, "session_RssAnon_bytes": GIB}
            def terminator(*unused):
                state.update(done=True, signals=state["signals"]+1)
                return [{"synthetic": True, "sent": True}]
            args = SimpleNamespace(timeout_seconds=1, min_runtime_available_gib=2,
                                   max_child_anon_gib=4, term_grace_seconds=.1, poll_seconds=.1)
            result = supervise_after_error(object(), self.birth, 0, args, io.StringIO(),
                clock=lambda: 2, sleeper=lambda seconds: None, sampler=sampler,
                terminator=terminator, peeker=lambda child: -15 if state["done"] else None)
            self.assertEqual((state["signals"], result["returncode"], result["status"]), (1, -15, "process_failed"))

        def test_degraded_monitor_waits_for_surviving_worker_without_early_signal(self):
            import io
            state = {"done": False, "waits": 0}
            worker = {**self.birth, "pid": 432, "start_ticks": 1205}
            def sampler(birth):
                return {"boot_id": "boot-a", "members": [{**birth, "state": "Z"}] + ([] if state["done"] else [worker]),
                        "mem_available_bytes": 3*GIB, "session_RssAnon_bytes": GIB}
            def sleeper(seconds):
                state.update(done=True, waits=state["waits"]+1)
            args = SimpleNamespace(timeout_seconds=10, min_runtime_available_gib=2,
                                   max_child_anon_gib=4, term_grace_seconds=.1, poll_seconds=.1)
            result = supervise_after_error(object(), self.birth, 0, args, io.StringIO(),
                clock=lambda: 2, sleeper=sleeper, sampler=sampler,
                terminator=lambda *unused: self.fail("No valid resource/timeout trigger"), peeker=lambda child: 0)
            self.assertEqual((state["waits"], result["status"], result["returncode"]), (1, "process_failed", 0))

        def test_logging_failure_is_best_effort(self):
            import contextlib
            import io
            class BrokenLog:
                def write(self, text):
                    raise OSError("synthetic disk full")
            errors = io.StringIO()
            with contextlib.redirect_stderr(errors):
                best_effort_event(BrokenLog(), "synthetic_event")
            self.assertIn('"log_write_failed": true', errors.getvalue())

        def test_spawn_is_own_session_no_shell(self):
            calls = []
            token = object()
            def fake(*args, **kwargs):
                calls.append((args, kwargs))
                return token
            self.assertIs(spawn_child(["python", "eval.py"], Path("."), None, "python", factory=fake), token)
            self.assertIs(calls[0][1]["start_new_session"], True)
            self.assertIs(calls[0][1]["shell"], False)

        def test_inherited_lock_passed_without_close(self):
            calls = []
            spawn_child(["python", "eval.py"], Path("."), None, "python", 7,
                        factory=lambda *args, **kwargs: calls.append(kwargs))
            self.assertEqual(calls[0]["pass_fds"], (7,))
            self.assertTrue(calls[0]["start_new_session"])

        def test_lock_path_inode_and_fd_validation(self):
            expected = "/project/runs/full_rebuild_pipeline_001/state/controller.lock"
            info = SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_dev=1, st_ino=55)
            self.assertEqual(validate_lock_identity(7, expected, expected, info, info)["inode"], 55)
            for fd, path, fdinfo in ((2, expected, info), (7, "/foreign/lock", info),
                                     (7, expected, SimpleNamespace(st_mode=stat.S_IFREG, st_dev=1, st_ino=56)),
                                     (7, expected, SimpleNamespace(st_mode=stat.S_IFDIR, st_dev=1, st_ino=55))):
                with self.subTest(fd=fd, path=path), self.assertRaises(ValueError):
                    validate_lock_identity(fd, expected, path, fdinfo, info)

        def test_duplicate_json_and_nonfinite_rejected(self):
            for data in ('{"a":1,"a":2}', '{"a":NaN}'):
                with self.subTest(data=data), self.assertRaises(ValueError):
                    parse_json(data)

        def test_exclusive_output(self):
            with tempfile.TemporaryDirectory(prefix="guard_synthetic_") as name:
                path = Path(name) / "result.json"
                json_file_new(path, {"synthetic": True})
                with self.assertRaises(FileExistsError):
                    json_file_new(path, {"synthetic": False})

        def test_source_command_and_plan_bindings(self):
            with tempfile.TemporaryDirectory(prefix="guard_synthetic_") as name:
                root = Path(name)
                source, plan, command, binding = [root / item for item in ("eval.py", "plan.json", "command.json", "bindings.json")]
                source.write_text("# synthetic only\n")
                plan.write_text('{"synthetic":true}')
                argv = [sys.executable, str(source)]
                command.write_text(json.dumps(argv))
                binding.write_text(json.dumps({"files": [{"path": str(source), "sha256": fingerprint(source)["sha256"]}],
                                               "plan": {"path": str(plan), "sha256": fingerprint(plan)["sha256"]}}))
                args = SimpleNamespace(command_json=command, bindings_json=binding,
                                       expected_command_sha256=digest(command.read_bytes()), expected_bindings_sha256=None)
                first = read_inputs(args, root)
                self.assertEqual(first["argv"], argv)
                plan.write_text('{"synthetic":false}')
                with self.assertRaisesRegex(ValueError, "Bound content SHA mismatch"):
                    read_inputs(args, root)
                plan.write_text('{"synthetic":true}')
                source.write_text("# changed source\n")
                with self.assertRaisesRegex(ValueError, "Bound content SHA mismatch"):
                    read_inputs(args, root)
                source.write_text("# synthetic only\n")
                command.write_text(json.dumps(argv + ["--changed"]))
                with self.assertRaisesRegex(ValueError, "Pinned command SHA mismatch"):
                    read_inputs(args, root)

        def test_default_cli_limits(self):
            args = parse_args(["--self-test"])
            self.assertEqual((args.min_start_available_gib, args.min_runtime_available_gib,
                              args.max_child_anon_gib, args.poll_seconds, args.timeout_seconds), (8, 2, 4, 1, 7200))
            for flag, value in (("--poll-seconds", "inf"), ("--max-child-anon-gib", "nan"),
                                ("--timeout-seconds", "-1")):
                with self.subTest(flag=flag), self.assertRaises(ValueError):
                    parse_args(["--self-test", flag, value])

        def disk_fixture(self):
            root = Path.cwd().resolve()
            return {"format": DISK_FORMAT, "directories": [
                {"path": str(root / "mock-root-scratch"), "device": 2066, "inode": 101,
                 "min_start_free_bytes": 15*GIB, "min_runtime_free_bytes": 4*GIB},
                {"path": str(root / "mock-data2-output"), "device": 2080, "inode": 202,
                 "min_start_free_bytes": 25*GIB, "min_runtime_free_bytes": 4*GIB}]}

        def test_disk_policy_schema_and_duplicate_device(self):
            policy = self.disk_fixture()
            self.assertIs(validate_disk_policy(policy), policy)
            changes = [lambda p: p.update(extra=True), lambda p: p.update(directories=[]),
                       lambda p: p["directories"][0].update(path="relative/path"),
                       lambda p: p["directories"][0].update(device=True),
                       lambda p: p["directories"][0].update(min_runtime_free_bytes=0),
                       lambda p: p["directories"][0].update(min_start_free_bytes=1),
                       lambda p: p["directories"][1].update(device=2066),
                       lambda p: p["directories"][1].update(path=p["directories"][0]["path"])]
            for change in changes:
                bad = copy.deepcopy(policy); change(bad)
                with self.assertRaises(ValueError): validate_disk_policy(bad)

        def test_disk_free_not_summed_and_low_runtime(self):
            policy = self.disk_fixture()
            probe = lambda path: {**next(r for r in policy["directories"] if r["path"] == path), "free_bytes": 30*GIB}
            sample = disk_sample(policy, probe)
            self.assertIsNone(disk_limit_reason(sample, "start"))
            sample["directories"][0]["free_bytes"] = 15*GIB - 1
            self.assertIn("device=2066", disk_limit_reason(sample, "start"))
            sample["directories"][0]["free_bytes"] = 4*GIB
            self.assertIsNone(disk_limit_reason(sample))
            sample["directories"][0]["free_bytes"] -= 1
            args = SimpleNamespace(timeout_seconds=100, min_runtime_available_gib=2, max_child_anon_gib=6)
            memory = {"mem_available_bytes": 3*GIB, "session_RssAnon_bytes": GIB, "disk": sample}
            self.assertIn("disk_free_below_runtime_limit", limit_reason(memory, 1, args))
            self.assertIsNone(disk_limit_reason(None))

        def test_disk_path_device_inode_or_free_invalid(self):
            policy = self.disk_fixture()
            for changes in ({"device": 99}, {"inode": 99}, {"free_bytes": -1}, {"free_bytes": 1.0}):
                def probe(path):
                    return {**next(r for r in policy["directories"] if r["path"] == path), "free_bytes": 30*GIB, **changes}
                with self.assertRaises(ValueError): disk_sample(policy, probe)
            with self.assertRaises(OSError): disk_sample(policy, lambda path: (_ for _ in ()).throw(FileNotFoundError(path)))

        def test_disk_policy_sha_pin_required_and_changed_file_rejected(self):
            policy = self.disk_fixture()
            probe = lambda path: {**next(r for r in policy["directories"] if r["path"] == path), "free_bytes": 30*GIB}
            with tempfile.TemporaryDirectory(prefix="disk_policy_mock_") as folder:
                path = Path(folder) / "policy.json"; path.write_text(json.dumps(policy))
                expected = digest(path.read_bytes())
                self.assertEqual(read_disk_policy(path, expected, probe), policy)
                for sha in (None, "", "0"*64):
                    with self.assertRaises(ValueError): read_disk_policy(path, sha, probe)
                path.write_text(json.dumps({**policy, "unexpected": True}))
                with self.assertRaisesRegex(ValueError, "SHA256 changed"): read_disk_policy(path, expected, probe)
                with self.assertRaises(ValueError): read_disk_policy(path, digest(path.read_bytes()), probe)

        def test_disk_execution_requires_binding_flags(self):
            flags = ["--command-json", "a", "--bindings-json", "b", "--project-root", ".", "--output-dir", "new"]
            with self.assertRaises(ValueError): parse_args(flags)
            with self.assertRaises(ValueError): parse_args(flags + ["--disk-policy-json", "p"])
            with self.assertRaises(ValueError): parse_args(["--self-test", "--poll-seconds", "2"])

        def test_disk_low_does_not_authorize_reused_birth(self):
            policy = self.disk_fixture()
            sample = disk_sample(policy, lambda path: {**next(r for r in policy["directories"] if r["path"] == path), "free_bytes": 1})
            self.assertIsNotNone(disk_limit_reason(sample))
            calls = []
            changed = {**self.birth, "start_ticks": 9999}
            result = guarded_signal(self.birth, self.known, 15, lambda: ("boot-a", [changed]),
                                    lambda *values: calls.append(values))
            self.assertFalse(result["sent"]); self.assertEqual(calls, [])

        def test_disk_invalid_degraded_sample_waits_for_original_timeout(self):
            import io
            state = {"tick": 0, "done": False, "signal_at": []}
            args = SimpleNamespace(timeout_seconds=2, min_runtime_available_gib=2,
                                   max_child_anon_gib=6, term_grace_seconds=.1, poll_seconds=1)
            def sampler(birth): raise ValueError("synthetic replaced disk directory")
            def sleeper(seconds): state["tick"] += 1
            def terminator(*unused):
                state["signal_at"].append(state["tick"]); state["done"] = True
                return [{"synthetic": True, "sent": True}]
            result = supervise_after_error(object(), self.birth, 0, args, io.StringIO(),
                clock=lambda: state["tick"], sleeper=sleeper, sampler=sampler, terminator=terminator,
                peeker=lambda child: -15 if state["done"] else None)
            self.assertEqual(state["signal_at"], [2])
            self.assertEqual(result["status"], "process_failed")
            self.assertEqual(result["termination_trigger"], "timeout")

        def test_disk_valid_low_degraded_sample_uses_original_terminator(self):
            import io
            state = {"done": False, "signals": 0}; policy = self.disk_fixture()
            args = SimpleNamespace(timeout_seconds=100, min_runtime_available_gib=2,
                                   max_child_anon_gib=6, term_grace_seconds=.1, poll_seconds=1)
            def sampler(birth):
                return {"boot_id": "boot-a", "members": [{**birth, "state": "Z" if state["done"] else "S"}],
                    "mem_available_bytes": 3*GIB, "session_RssAnon_bytes": GIB,
                    "disk": disk_sample(policy, lambda path: {**next(r for r in policy["directories"] if r["path"] == path), "free_bytes": 1})}
            def terminator(*unused):
                state.update(done=True, signals=state["signals"]+1)
                return [{"synthetic": True, "sent": True}]
            result = supervise_after_error(object(), self.birth, 0, args, io.StringIO(),
                clock=lambda: 1, sleeper=lambda seconds: None, sampler=sampler, terminator=terminator,
                peeker=lambda child: -15 if state["done"] else None)
            self.assertEqual(state["signals"], 1)
            self.assertEqual(result["status"], "process_failed")
            self.assertIn("disk_free_below_runtime_limit", result["termination_trigger"])

        def test_statvfs_uses_available_blocks_and_checks_directory_replacement(self):
            from unittest import mock
            from contextlib import ExitStack
            with tempfile.TemporaryDirectory(prefix="disk_statvfs_mock_") as folder:
                path = str(Path(folder).resolve()); info = Path(path).stat()
                with ExitStack() as stack:
                    for name in ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC"):
                        stack.enter_context(mock.patch.object(os, name, 0, create=True))
                    stack.enter_context(mock.patch.object(os, "open", return_value=99))
                    stack.enter_context(mock.patch.object(os, "close"))
                    fake_stat = stack.enter_context(mock.patch.object(os, "fstat", return_value=info))
                    stack.enter_context(mock.patch.object(os, "fstatvfs", create=True,
                        return_value=SimpleNamespace(f_bavail=7, f_bfree=999, f_frsize=4096)))
                    self.assertEqual(disk_directory_snapshot(path)["free_bytes"], 7*4096)
                    fake_stat.return_value = SimpleNamespace(st_dev=info.st_dev, st_ino=info.st_ino+1, st_mode=info.st_mode)
                    with self.assertRaisesRegex(ValueError, "changed during statvfs"):
                        disk_directory_snapshot(path)

        def disk_execute_fixture(self, scenario):
            """Exercise the real execution branch with no process, procfs, signal, or disk-space calls."""
            from unittest import mock
            from contextlib import ExitStack, redirect_stdout
            import io
            policy = self.disk_fixture()
            good = disk_sample(policy, lambda path: {**next(r for r in policy["directories"] if r["path"] == path), "free_bytes": 30*GIB})
            low = copy.deepcopy(good); low["directories"][0]["free_bytes"] = 1
            class Child:
                pid = 431
                returncode = None
                def poll(self): self.returncode = 0; return 0
            state = {"terminated": False}
            def terminator(*unused):
                state["terminated"] = True
                return [{"synthetic": True, "sent": True}]
            with tempfile.TemporaryDirectory(prefix="disk_execute_mock_") as folder:
                root = Path(folder).resolve(); output = root / "output"
                args = parse_args(["--self-test", "--max-child-anon-gib", "6"])
                args.project_root = root; args.output_dir = output
                birth = {**self.birth, "cwd": str(root), "cmdline": ["mock"]}
                inputs = {"argv": ["mock"], "resolved_executable": {"path": "mock"},
                          "disk_policy": {"path": "mock-policy", "sha256": "a"*64}, "disk_policy_content": policy}
                snapshot = {"mem_available_bytes": 8*GIB, "session_RssAnon_bytes": GIB,
                            "session_VmRSS_bytes": 2*GIB, "session_VmSwap_bytes": 0,
                            "child": birth, "members": [birth], "boot_id": "boot-a", "disk": good}
                if scenario in ("runtime_low", "exit_low"):
                    snapshot["disk"] = low
                disk_probes = [low] if scenario == "start_low" else [good, low if scenario == "prelaunch_low" else good]
                def peeker(child):
                    return None if scenario == "runtime_low" and not state["terminated"] else 0
                original_read = Path.read_text
                def read(path, *a, **kw):
                    return "MemAvailable: 10485760 kB" if str(path).replace("\\", "/") == "/proc/meminfo" else original_read(path, *a, **kw)
                with ExitStack() as stack:
                    stack.enter_context(mock.patch.object(sys, "platform", "linux"))
                    stack.enter_context(mock.patch.object(os, "getuid", return_value=1000, create=True))
                    stack.enter_context(mock.patch.object(Path, "read_text", read))
                    module = sys.modules[__name__]
                    stack.enter_context(mock.patch.object(module, "read_bound_inputs", return_value=inputs))
                    stack.enter_context(mock.patch.object(module, "read_boot_id", return_value="boot-a"))
                    stack.enter_context(mock.patch.object(module, "bound_disk_sample", side_effect=disk_probes))
                    stack.enter_context(mock.patch.object(module, "process_snapshot", return_value=birth))
                    stack.enter_context(mock.patch.object(module, "resource_sample", side_effect=ValueError("mock bad disk") if scenario == "invalid_sample" else None,
                                                          return_value=snapshot))
                    stack.enter_context(mock.patch.object(module, "peek_returncode", side_effect=peeker))
                    spawn = stack.enter_context(mock.patch.object(module, "spawn_child", return_value=Child()))
                    terminate = stack.enter_context(mock.patch.object(module, "terminate_owned_group", side_effect=terminator))
                    with redirect_stdout(io.StringIO()): code = execute(args)
                receipt = parse_json((output / "receipt.json").read_bytes())
                return code, receipt, spawn.call_count, terminate.call_count

        def test_disk_low_at_start_and_prelaunch_never_spawns(self):
            for scenario in ("start_low", "prelaunch_low"):
                with self.subTest(scenario=scenario):
                    code, result, spawned, signaled = self.disk_execute_fixture(scenario)
                    self.assertEqual((code, spawned, signaled), (75, 0, 0))
                    self.assertIn("disk_free_below_start_limit", result["reason"])

        def test_disk_runtime_low_uses_existing_termination_branch(self):
            code, result, spawned, signaled = self.disk_execute_fixture("runtime_low")
            self.assertEqual((code, spawned, signaled), (75, 1, 1))
            self.assertEqual(result["status"], "resource_abort")
            self.assertIn("disk_free_below_runtime_limit", result["reason"])

        def test_disk_invalid_sample_cannot_certify_success(self):
            code, result, spawned, signaled = self.disk_execute_fixture("invalid_sample")
            self.assertEqual((code, spawned, signaled), (1, 1, 0))
            self.assertEqual(result["status"], "process_failed")
            self.assertEqual(result["monitoring_errors"], 1)

        def test_disk_low_at_child_exit_cannot_certify_success(self):
            code, result, spawned, signaled = self.disk_execute_fixture("exit_low")
            self.assertEqual((code, spawned, signaled), (75, 1, 0))
            self.assertEqual(result["status"], "resource_abort")

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GuardTests))
    print(json.dumps({"format": FORMAT + "-self-test", "synthetic_only": True,
                      "real_process_spawned": False, "real_signal_sent": False,
                      "tests": result.testsRun, "passed": result.wasSuccessful(),
                      "guard_sha256": fingerprint(Path(__file__))["sha256"]}))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    try:
        arguments = parse_args()
        sys.exit(self_test() if arguments.self_test else execute(arguments))
    except (OSError, ValueError) as error:
        print(json.dumps({"status": "process_failed", "reason": "invalid_request", "error": str(error)}), file=sys.stderr)
        sys.exit(1)
