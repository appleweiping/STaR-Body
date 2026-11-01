#!/usr/bin/env python3
"""C11 experiment runner: keypoint caches, training matrix, evaluation, test locks, tables.

All heavy work is delegated to the versioned entry points; this script only schedules them.

  python scripts/run_c11_matrix.py --matrix configs/c11/matrix.local.json status
  python scripts/run_c11_matrix.py --matrix ... keypoints --data-root /data/3DPW --source vitpose --gpus 0
  python scripts/run_c11_matrix.py --matrix ... train --tier core --gpus 0,1,2
  python scripts/run_c11_matrix.py --matrix ... evaluate --split validation --gpus 0,1,2
  python scripts/run_c11_matrix.py --matrix ... lock            # validation -> immutable test locks
  python scripts/run_c11_matrix.py --matrix ... evaluate --split test --gpus 0,1,2
  python scripts/run_c11_matrix.py --matrix ... analyze         # tables + LaTeX macros

Global option --dry-run goes BEFORE the subcommand (``--matrix X --dry-run train ...``) and only
prints commands. Finished outputs are never recomputed or overwritten. Training writes a
resumable latest.pt every ``checkpoint_every`` (default 100) steps, so an interrupted run
resumes from there; live runs (lock '<run_dir>.running') are reported as 'running' and skipped.
Pre-registered protocol: every variant/seed is compared at the same fixed step (EMA);
test results are produced only through validation locks and never used for selection.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIERS = ('core', 'ablation', 'optional')


def resolve(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_matrix(path):
    matrix = json.loads(Path(path).read_text(encoding='utf-8'))
    for name in matrix['variants']:
        if not name.replace('-', '').isalnum():
            raise ValueError(f'Variant names must be alphanumeric with dashes: {name}')
    return matrix


def selected_runs(matrix, args):
    tiers = set(args.tier or ['core'])
    names = args.variants or [n for n, v in matrix['variants'].items() if v['tier'] in tiers]
    seeds = args.seeds or matrix['seeds']
    return [(n, s) for n in names for s in seeds]


def run_dir(matrix, variant, seed, smoke=False):
    return resolve(matrix['runs_dir'] + ('_smoke' if smoke else '')) / variant / f's{seed}'


def fixed_checkpoint(matrix, variant, seed):
    return run_dir(matrix, variant, seed) / f'step_{matrix["fixed_step"]:06d}.pt'


def is_running(out):
    """True while train_from_config_v43 holds '<run_dir>.running' with a live PID on this host."""
    import socket
    lock = out.with_name(out.name + '.running')
    if not lock.is_file():
        return False
    try:
        info = json.loads(lock.read_text())
        os.kill(int(info['pid']), 0)
        return info.get('host') == socket.gethostname()
    except (ProcessLookupError, PermissionError, ValueError, KeyError, json.JSONDecodeError):
        return False


def run_status(matrix, variant, seed):
    out = run_dir(matrix, variant, seed)
    if fixed_checkpoint(matrix, variant, seed).is_file():
        return 'done'
    if is_running(out):
        return 'running'
    if (out / 'latest.pt').is_file():
        return 'resumable'
    if out.is_dir() and any(out.iterdir()):
        return 'incomplete'
    return 'pending'


SMOKE = {'stop-after': 2, 'eval-every': 1, 'max-val-windows': 16, 'allow-partial-validation': True}


def train_config(matrix, variant, seed, resume, smoke=False):
    spec = matrix['variants'][variant]
    config = json.loads(resolve(matrix['base_config']).read_text(encoding='utf-8'))
    config.pop('v37-joint-feedback', None)
    for key in [k for k in config if k.startswith('v43-')]:
        config.pop(key)
    config.update(spec['cli'])
    config.update({'seed': seed, 'v43-variant': variant, 'stop-after': matrix['fixed_step'], 'device': 'cuda:0',
                   'v43-checkpoint-every': matrix.get('checkpoint_every', 100),
                   'output-dir': str(run_dir(matrix, variant, seed, smoke))})
    if smoke:  # development-only pipeline check: 2 optimizer steps, 16 validation windows
        config.update(SMOKE)
    if spec['cli'].get('v43-feedback') == 'router':
        config['v43-router-init-seed'] = matrix['router_init_seed_base'] + (seed - matrix['seeds'][0])
    if spec.get('evidence'):
        paths = matrix['keypoints'][spec['evidence']]
        config['train-external-keypoints'] = paths['train']
        config['val-external-keypoints'] = paths['validation']
    if resume:
        config['resume-checkpoint'] = str(run_dir(matrix, variant, seed) / 'latest.pt')
    return config


class GpuQueue:
    """One job per GPU at a time; CUDA_VISIBLE_DEVICES pins the job, which then uses cuda:0."""

    def __init__(self, gpus, log_dir, dry_run):
        self.gpus, self.log_dir, self.dry_run = list(gpus), Path(log_dir), dry_run
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.failures = []

    def run(self, jobs):
        """jobs: list of (name, command list, optional pre-launch callable)."""
        pending = list(jobs)
        running = {}
        while pending or running:
            for gpu in self.gpus:
                if gpu in running or not pending:
                    continue
                name, command, before = pending.pop(0)
                if before is not None:
                    before()
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), V28D_MEMORY_LIMIT_GIB=os.environ.get('V28D_MEMORY_LIMIT_GIB', '20'))
                log = self.log_dir / f'{name}.{datetime.now():%Y%m%d_%H%M%S}.log'
                print(f'[gpu {gpu}] {name}\n  {shlex.join(command)}\n  log: {log}', flush=True)
                if self.dry_run:
                    continue
                stream = log.open('w')
                running[gpu] = (name, subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT), stream)
            if self.dry_run:
                if not pending:
                    break
                continue
            time.sleep(20)
            for gpu, (name, proc, stream) in list(running.items()):
                if proc.poll() is not None:
                    stream.close()
                    print(f'[gpu {gpu}] {name} finished with exit code {proc.returncode}', flush=True)
                    if proc.returncode != 0:
                        self.failures.append(name)
                    del running[gpu]
        if self.failures:
            print('FAILED: ' + ', '.join(self.failures), file=sys.stderr)
        return not self.failures


def cmd_status(matrix, args):
    rows = selected_runs(matrix, args)
    for variant, seed in rows:
        evals = [split for split in ('validation', 'test')
                 if (resolve(matrix['results_dir']) / split / f'{variant}_s{seed}.json').is_file()]
        print(f'{variant:18s} s{seed}  {run_status(matrix, variant, seed):10s}  evaluated: {",".join(evals) or "-"}')


def cmd_keypoints(matrix, args):
    base = json.loads(resolve(matrix['base_config']).read_text(encoding='utf-8'))
    sidecars = {'train': base['train-hmr-sidecar'], 'validation': base['val-hmr-sidecar'],
                'test': matrix['eval']['test']['hmr-sidecar']}
    jobs = []
    for split in args.splits:
        output = resolve(matrix['keypoints'][args.source][split])
        if output.exists():
            print(f'exists, skipped: {output}')
            continue
        command = ['bash', 'scripts/run_star_body_v43.sh', 'scripts/cache_external_keypoints_v43.py',
                   '--data-root', str(resolve(args.data_root)), '--split', split, '--hmr-sidecar', str(resolve(sidecars[split])),
                   '--source', args.source, '--output', str(output), '--device', 'cuda:0']
        if args.source == 'vitpose':
            command += ['--model', args.model, '--batch-size', str(args.batch_size)]
        jobs.append((f'keypoints_{args.source}_{split}', command, None))
    return GpuQueue(args.gpus, resolve(matrix['runs_dir']) / '_logs', args.dry_run).run(jobs)


def cmd_train(matrix, args):
    config_dir = resolve(matrix['runs_dir']) / '_configs'
    config_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for variant, seed in selected_runs(matrix, args):
        state = 'pending' if args.smoke else run_status(matrix, variant, seed)
        if state in ('done', 'running'):
            if state == 'running':
                print(f'still running, skipped: {variant} s{seed}')
            continue
        out = run_dir(matrix, variant, seed, args.smoke)
        if args.smoke and out.exists():
            print(f'smoke output exists, skipped: {out}')
            continue
        before = None
        if state == 'incomplete':
            target = out.with_name(out.name + f'.incomplete_{datetime.now():%Y%m%d_%H%M%S}')
            print(f'moving incomplete run without latest.pt aside: {out} -> {target}')
            if not args.dry_run:
                out.rename(target)
        config = train_config(matrix, variant, seed, resume=state == 'resumable', smoke=args.smoke)
        path = config_dir / f'{variant}_s{seed}.{"smoke." if args.smoke else ""}{"resume." if state == "resumable" else ""}json'
        if not args.dry_run:
            path.write_text(json.dumps(config, indent=2), encoding='utf-8')
        command = ['bash', 'scripts/run_star_body_v43.sh', 'scripts/train_from_config_v43.py', '--config', str(path)]
        jobs.append((f'{"smoke" if args.smoke else "train"}_{variant}_s{seed}', command, before))
    if not jobs:
        print('nothing to train')
        return True
    return GpuQueue(args.gpus, resolve(matrix['runs_dir']) / '_logs', args.dry_run).run(jobs)


def eval_targets(matrix, args):
    """(name, seed, checkpoint, model_version, weights, evidence) for runs and references."""
    targets = []
    for variant, seed in ([] if getattr(args, 'references_only', False) else selected_runs(matrix, args)):
        if run_status(matrix, variant, seed) == 'done' or args.dry_run:
            targets.append((variant, seed, fixed_checkpoint(matrix, variant, seed), 'v43',
                            matrix['checkpoint_weights'], matrix['variants'][variant].get('evidence')))
    if args.references:
        for name, ref in matrix.get('references', {}).items():
            if name.startswith('_') or not ref:
                continue
            targets.append((name, 0, resolve(ref['checkpoint']), ref['model_version'], ref['weights'], None))
    return targets


def lock_path(matrix, name, seed):
    return resolve(matrix['results_dir']) / 'locks' / f'{name}_s{seed}.json'


def cmd_evaluate(matrix, args):
    split = args.split
    out_dir = resolve(matrix['results_dir']) / split
    out_dir.mkdir(parents=True, exist_ok=True)
    common = matrix['eval']
    spec = common[split]
    jobs = []
    for name, seed, checkpoint, version, weights, evidence in eval_targets(matrix, args):
        output = out_dir / f'{name}_s{seed}.json'
        if output.exists():
            if not output.with_suffix('.npz').exists():
                print(f'WARNING: {output.name} exists without its per-frame .npz; delete the JSON to re-evaluate')
            continue
        if output.with_suffix('.npz').exists():
            print(f'WARNING: removing orphan per-frame file without JSON: {output.with_suffix(".npz")}')
            if not args.dry_run:
                output.with_suffix('.npz').unlink()
        command = ['bash', 'scripts/run_star_body_v43.sh', 'scripts/evaluate_target60_v43.py',
                   '--model-version', version, '--checkpoint', str(checkpoint), '--checkpoint-weights', weights,
                   '--split', split, '--vace-cache', str(resolve(spec['vace-cache'])),
                   '--hmr-sidecar', str(resolve(spec['hmr-sidecar'])), '--coverage-manifest', str(resolve(spec['coverage-manifest'])),
                   '--smpl-model-root', str(resolve(common['smpl-model-root'])), '--joint-regressor', str(resolve(common['joint-regressor'])),
                   '--output', str(output), '--per-frame-output', str(output.with_suffix('.npz')),
                   '--amp-dtype', 'none', '--device', 'cuda:0', '--batch-size', str(common.get('batch-size', 8))]
        if version in ('v34', 'v36', 'v43'):
            command += ['--track-prototypes', str(resolve(spec['track-prototypes']))]
        if split == 'test':
            command += ['--gt-cache', str(resolve(spec['gt-cache']))]
            lock = lock_path(matrix, name, seed)
            if version in ('v36', 'v43'):
                if not lock.is_file() and not args.dry_run:
                    print(f'skip {name} s{seed}: no validation selection lock (run the "lock" subcommand first)')
                    continue
                command += ['--selection-lock', str(lock)]
        if evidence:
            command += ['--external-keypoints', str(resolve(matrix['keypoints'][evidence][split]))]
        jobs.append((f'eval_{split}_{name}_s{seed}', command, None))
    if not jobs:
        print('nothing to evaluate')
        return True
    return GpuQueue(args.gpus, resolve(matrix['runs_dir']) / '_logs', args.dry_run).run(jobs)


def cmd_lock(matrix, args):
    ok = True
    for name, seed, checkpoint, version, weights, _ in eval_targets(matrix, args):
        if version not in ('v36', 'v43'):
            continue
        lock = lock_path(matrix, name, seed)
        validation = resolve(matrix['results_dir']) / 'validation' / f'{name}_s{seed}.json'
        if lock.exists():
            continue
        if not validation.is_file() and not args.dry_run:
            print(f'skip {name} s{seed}: validation evaluation missing')
            ok = False
            continue
        command = [sys.executable, str(ROOT / 'scripts/make_selection_lock_v43.py'), '--checkpoint', str(checkpoint),
                   '--checkpoint-weights', weights, '--validation-eval', str(validation), '--output', str(lock),
                   '--rule', f'fixed pre-registered step {matrix["fixed_step"]}; {weights}; identical for all variants']
        print(shlex.join(command))
        if not args.dry_run:
            ok &= subprocess.run(command, cwd=ROOT).returncode == 0
    return ok


def cmd_analyze(matrix, args):
    results = resolve(matrix['results_dir'])
    runs = []
    for split in ('validation', 'test'):
        for path in sorted((results / split).glob('*_s*.json')):
            name, seed = path.stem.rsplit('_s', 1)
            runs.append(dict(variant=name, seed=int(seed), split=split, eval_json=str(path)))
    manifest = dict(control=args.control, bootstrap=args.bootstrap, comparisons=matrix.get('comparisons', []), runs=runs)
    manifest_path = results / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    command = [sys.executable, str(ROOT / 'scripts/analyze_results_c11.py'), '--manifest', str(manifest_path),
               '--out-json', str(ROOT / 'evidence/c11/summary.json'), '--out-tex', str(ROOT / 'paper/latex/generated/results.tex'),
               '--out-md', str(ROOT / 'evidence/c11/summary.md')]
    print(shlex.join(command))
    return args.dry_run or subprocess.run(command, cwd=ROOT).returncode == 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--matrix', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true')
    sub = p.add_subparsers(dest='command', required=True)

    def selection(sp):
        sp.add_argument('--tier', nargs='+', choices=TIERS)
        sp.add_argument('--variants', nargs='+')
        sp.add_argument('--seeds', nargs='+', type=int)

    sp = sub.add_parser('status'); selection(sp)
    sp = sub.add_parser('keypoints')
    sp.add_argument('--data-root', required=True)
    sp.add_argument('--source', choices=('vitpose', 'openpose3dpw'), default='vitpose')
    sp.add_argument('--model', default='usyd-community/vitpose-plus-huge')
    sp.add_argument('--batch-size', type=int, default=64)
    sp.add_argument('--splits', nargs='+', default=['train', 'validation', 'test'])
    sp.add_argument('--gpus', default='0')
    sp = sub.add_parser('train'); selection(sp); sp.add_argument('--gpus', default='0')
    sp.add_argument('--smoke', action='store_true', help='2-step development run per selected variant (runs_dir + _smoke)')
    sp = sub.add_parser('evaluate'); selection(sp); sp.add_argument('--gpus', default='0')
    sp.add_argument('--split', choices=('validation', 'test'), required=True)
    sp.add_argument('--no-references', dest='references', action='store_false')
    sp.add_argument('--references-only', action='store_true', help='Only the archived parent/C10/C00 checkpoints')
    sp = sub.add_parser('lock'); selection(sp); sp.add_argument('--no-references', dest='references', action='store_false')
    sp.add_argument('--references-only', action='store_true')
    sp = sub.add_parser('analyze'); sp.add_argument('--control', default='A1-mean'); sp.add_argument('--bootstrap', type=int, default=20000)
    args = p.parse_args()
    if hasattr(args, 'gpus'):
        args.gpus = [g.strip() for g in str(args.gpus).split(',') if g.strip()]
    matrix = load_matrix(args.matrix)
    if args.command in ('evaluate', 'lock') and args.tier is None and args.variants is None:
        args.tier = list(TIERS)  # evaluate everything that has finished
    handler = dict(status=cmd_status, keypoints=cmd_keypoints, train=cmd_train, evaluate=cmd_evaluate,
                   lock=cmd_lock, analyze=cmd_analyze)[args.command]
    ok = handler(matrix, args)
    raise SystemExit(0 if ok in (None, True) else 1)


if __name__ == '__main__':
    main()
