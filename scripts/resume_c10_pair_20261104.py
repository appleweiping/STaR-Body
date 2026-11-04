"""Resume the existing C00/C10 pair, serially, without changing its protocol.

Linux execution only; --dry-run is portable and uses only the stdlib contract.
Deploy this controller OUTSIDE the bound training project: its strict resume
engine includes every scripts/**/*.py file in the original source inventory.
Launch with nohup/setsid and redirected output. No SSH or credentials live here.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback


ARMS = ("C10_joint_feedback", "C00_mean_feedback")
CAPS = ("max-train-windows", "max-val-windows", "max-pose-train-windows", "max-pose-val-windows")
PATH_KEYS = {
    "train-vace-cache", "train-hmr-sidecar", "val-vace-cache", "val-hmr-sidecar",
    "pose-train-vace-dir", "pose-train-hmr-dir", "pose-val-vace-dir", "pose-val-hmr-dir",
    "init-checkpoint", "smpl-model-root", "joint-regressor", "output-dir",
    "validation-manifest", "coco-train-manifest", "resume-checkpoint", "motion-pretrained",
    "train-track-prototypes", "val-track-prototypes", "initial-reference-report",
    "parent-validation-report",
}
WAIT_SECONDS = 7 * 86400
POLL_SECONDS = 30


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_contract(project):
    path = project / "video_VACE/divibody_training/target60_contract_v36.py"
    spec = importlib.util.spec_from_file_location("resume_c10_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config_argv(config, project):
    """Normalize the existing JSON for the project's actual stdlib parser."""
    result = []
    for key, value in config.items():
        if not isinstance(key, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", key):
            raise ValueError("Invalid configuration key: " + repr(key))
        if value is None or value is False:
            continue
        if key == "local-radius-schedule" and isinstance(value, list):
            value = ",".join(str(item) for item in value)
        result.append("--" + key)
        if value is True:
            continue
        if not isinstance(value, (str, int, float)):
            raise ValueError("Non-scalar configuration value: " + key)
        if key in PATH_KEYS:
            path = Path(value).expanduser()
            value = str((path if path.is_absolute() else project / path).resolve())
        result.append(str(value))
    return result


def check_sources(project, bound_path, contract):
    bound = read_json(bound_path)
    if not isinstance(bound, dict) or not bound:
        raise ValueError("bound_source_sha256.json must be a nonempty path/hash map")
    errors = []
    for name, expected in bound.items():
        path = (project / name).resolve()
        if not path.is_relative_to(project) or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
            errors.append("Invalid bound source entry: " + str(name))
        elif not path.is_file():
            errors.append("Missing bound source: " + str(path))
        elif digest(path) != expected.lower():
            errors.append("Bound source differs: " + str(path))
    if errors:
        raise ValueError("; ".join(errors))
    # The resume engine compares the entire inventory, including added files.
    return contract.source_fingerprints()


def inspect_arm(project, experiment, label, contract, source_inventory):
    source = experiment / "configs" / (label + "_formal.json")
    config = read_json(source)
    if not isinstance(config, dict) or not config:
        raise ValueError("Nonempty formal config required: " + str(source))
    required = {"max-steps": 6000, "stop-after": 2000, "device": "cuda:0", "max-cuda-memory-gib": 3.0,
                "v37-joint-feedback": label == "C10_joint_feedback", **{key: 0 for key in CAPS}}
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"{label}: frozen {key} must be {expected!r}")
    run = experiment / "runs" / label
    output = Path(config["output-dir"]).expanduser()
    if not output.is_absolute():
        output = project / output
    if output.resolve() != run.resolve():
        raise ValueError(label + ": output-dir does not name the existing run")
    config = dict(config, **{"resume-checkpoint": str(run / "latest.pt"), "stop-after": 6000})
    parsed = contract.parse_training_args(config_argv(config, project))
    assets = contract.inspect_assets(parsed, fingerprint=False)
    missing = dict(assets["missing"])
    for name in ("run_config.json", "paused.json", "progress.json", "metrics.jsonl", "events.jsonl", "latest.pt"):
        path = run / name
        if not path.is_file():
            missing[name] = str(path)
    errors = list(assets["errors"])
    if (run / "paused.json").is_file() and read_json(run / "paused.json").get("step") != 2000:
        errors.append("Expected the original paused step 2000")
    if (run / "progress.json").is_file():
        progress = read_json(run / "progress.json")
        if progress.get("step") != 2000 or progress.get("complete") is not False:
            errors.append("Expected incomplete progress at step 2000")
    if (run / "completion.json").exists():
        errors.append("Existing completion.json: inspect instead of relaunching")
    if (run / "latest.pt").is_file() and (run / "latest.pt").stat().st_size == 0:
        errors.append("Empty latest.pt")
    if (run / "run_config.json").is_file():
        recorded = read_json(run / "run_config.json")
        if recorded.get("source_files_sha256") != source_inventory:
            errors.append("Original run source inventory differs (including added scripts); controller must live outside the bound project")
        actual = json.loads(json.dumps({key: str(value) if isinstance(value, Path) else value
                                        for key, value in vars(parsed).items()}))
        saved = recorded.get("args", {})
        excluded = {"resume_checkpoint", "stop_after"}
        if {key: value for key, value in actual.items() if key not in excluded} != {key: value for key, value in saved.items() if key not in excluded}:
            errors.append("Parsed configuration differs from original run args beyond resume_checkpoint/stop_after")
        if recorded.get("development_only") is not False:
            errors.append("Original run must be full-data, not development_only")
    return {"label": label, "source_config": str(source), "run_dir": str(run), "config": config,
            "passed": not missing and not errors, "missing": missing, "errors": errors,
            "changes": {"resume-checkpoint": str(run / "latest.pt"), "stop-after": 6000}}


def preflight(project, experiment):
    required = [project / "run_server32_v34.sh", project / "scripts/train_from_config_v36.py",
                project / "video_VACE/divibody_training/target60_contract_v36.py",
                experiment / "bound_source_sha256.json", experiment / "controller.lock"]
    missing = [str(path) for path in required if not path.is_file()]
    plans, errors = [], []
    contract_path = required[2]
    if contract_path.is_file():
        try:
            contract = load_contract(project)
            inventory = check_sources(project, required[3], contract) if required[3].is_file() else None
            for label in ARMS:
                try:
                    plans.append(inspect_arm(project, experiment, label, contract, inventory))
                except (OSError, ValueError, KeyError, TypeError, SystemExit) as error:
                    errors.append(f"{label}: {error}")
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(str(error))
    return {"passed": not missing and not errors and len(plans) == 2 and all(plan["passed"] for plan in plans),
            "missing": missing, "errors": errors, "arms": plans}


def resources(gpu, experiment):
    available = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    output = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True, timeout=30)
    free = int(output.strip())
    disk = shutil.disk_usage(experiment).free
    return {"free_gpu_mib": free, "mem_available_kib": available, "disk_free_bytes": disk}


def verify_completion(run):
    completion = read_json(run / "completion.json")
    progress = read_json(run / "progress.json")
    if completion.get("completed_steps") != 6000 or completion.get("complete") is not True or completion.get("development_only") is not False:
        raise ValueError("Missing full-data step-6000 completion")
    if progress.get("step") != 6000 or progress.get("complete") is not True:
        raise ValueError("Progress does not confirm step 6000")
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines() if line.strip()]
    if not any(row.get("event") == "resumed" and row.get("step") == 2000 for row in events):
        raise ValueError("No resume-from-2000 event")
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    for step in (3000, 4000, 5000, 6000):
        for variant in ("student", "ema"):
            found = [row for row in rows if row.get("step") == step and row.get("variant") == variant]
            if len(found) != 1 or found[0].get("development_only") is not False:
                raise ValueError(f"Missing or duplicated formal validation: {step}/{variant}")
            val = found[0]["validation"]
            if (val.get("frames"), val.get("acceleration_triplets")) != (10413, 10226):
                raise ValueError("Incomplete validation coverage")
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    project, experiment, state_dir = (path.expanduser().resolve() for path in (args.project_root, args.experiment_root, args.state_dir))
    sys.dont_write_bytecode = True
    if args.gpu < 0:
        parser.error("GPU index must be nonnegative")
    if state_dir in (project, experiment) or any(state_dir.is_relative_to(experiment / "runs" / label) for label in ARMS):
        parser.error("Use a separate state directory, not a project root or an original arm directory")
    if args.dry_run:
        report = preflight(project, experiment)
        report.update(dry_run=True, launched=False, state_dir=str(state_dir), gpu=args.gpu,
                      checkpoint_note="Checkpoint payload/optimizer/RNG invariants are verified by the unchanged trainer at launch; this stdlib check does not deserialize torch files.")
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 2
    if not sys.platform.startswith("linux"):
        parser.error("Execution requires Linux flock/nice/proc; Windows supports --dry-run only")
    import fcntl
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / "controller.lock").open("a") as own_lock:
        try:
            fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another controller owns this state directory", file=sys.stderr)
            return 2
        state_path = state_dir / "state.json"
        if state_path.exists():
            print("State already exists; inspect it before choosing a fresh attempt directory", file=sys.stderr)
            return 2
        state = {"status": "waiting_resources", "pid": os.getpid(), "project_root": str(project),
                 "experiment_root": str(experiment), "gpu": args.gpu, "completed": [], "test_used": False}

        def update(**values):
            state.update(values, updated_epoch=time.time())
            atomic_json(state_path, state)

        try:
            update(stage="preflight")
            initial = preflight(project, experiment)
            atomic_json(state_dir / "preflight.json", initial)
            if not initial["passed"]:
                raise ValueError("Preflight failed; see preflight.json for missing files/errors")
            contract = load_contract(project)
            with (experiment / "controller.lock").open("r+") as original_lock:
                start = time.monotonic()
                while True:
                    try:
                        fcntl.flock(original_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() - start > WAIT_SECONDS:
                            raise TimeoutError("Original controller lock wait exceeded seven days")
                        update(status="waiting_resources", stage="original_controller_lock")
                        time.sleep(POLL_SECONDS)
                for label in ARMS:
                    start = time.monotonic()
                    while True:
                        observed = resources(args.gpu, experiment)
                        if observed["disk_free_bytes"] < 12 * 1024**3:
                            raise RuntimeError("Disk below 12 GiB; no training launched")
                        if observed["free_gpu_mib"] >= 4096 and observed["mem_available_kib"] >= 16 * 1024**2:
                            break
                        if time.monotonic() - start > WAIT_SECONDS:
                            raise TimeoutError("GPU/RAM resource wait exceeded seven days")
                        update(status="waiting_resources", stage=label, **observed)
                        time.sleep(POLL_SECONDS)
                    inventory = check_sources(project, experiment / "bound_source_sha256.json", contract)
                    plan = inspect_arm(project, experiment, label, contract, inventory)
                    if not plan["passed"]:
                        raise ValueError(json.dumps(plan))
                    config_path = state_dir / (label + "_resume6000.json")
                    atomic_json(config_path, plan["config"])
                    log_path = state_dir / (label + "_resume6000.log")
                    command = ["nice", "-n", "10", "bash", str(project / "run_server32_v34.sh"),
                               "scripts/train_from_config_v36.py", "--config", str(config_path)]
                    env = os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), V28D_MEMORY_LIMIT_GIB="3",
                               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONDONTWRITEBYTECODE="1")
                    with log_path.open("x", encoding="utf-8") as log:
                        child = subprocess.Popen(command, cwd=project, env=env, stdin=subprocess.DEVNULL,
                                                 stdout=log, stderr=log, start_new_session=True,
                                                 pass_fds=(own_lock.fileno(), original_lock.fileno()))
                        update(status="running", stage=label, child_pid=child.pid, command=command,
                               log=str(log_path), config=str(config_path), **observed)
                        while child.poll() is None:
                            time.sleep(POLL_SECONDS)
                            update(status="running", stage=label, child_pid=child.pid)
                    if child.returncode != 0:
                        raise RuntimeError(f"{label} exited {child.returncode}; inspect {log_path}")
                    receipt = verify_completion(Path(plan["run_dir"]))
                    state["completed"].append({"label": label, "completion": receipt})
                    update(status="waiting_resources", stage="between_arms", child_pid=None)
            update(status="complete", stage="pair_complete", child_pid=None)
            return 0
        except Exception as error:
            update(status="failed", error=str(error), traceback=traceback.format_exc())
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
