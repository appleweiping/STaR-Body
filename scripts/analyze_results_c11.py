#!/usr/bin/env python3
"""Aggregate C11 evaluation JSONs into seed statistics, paired comparisons and LaTeX macros.

Input is a manifest JSON (written by scripts/run_c11_matrix.py, or by hand):

    {"control": "A1-mean", "bootstrap": 20000,
     "comparisons": [["B4-ext-joint", "B1-ext-pooled"], ["A2-head-joint", "A1-mean"]],
     "runs": [{"variant": "B4-ext-joint", "seed": 2027, "split": "validation",
               "eval_json": "results/c11/validation/B4-ext-joint_s2027.json"}, ...]}

Every ``eval_json`` must be an unmodified output of scripts/evaluate_target60_v43.py (or the
original v36 evaluator); ``metrics.per_sequence`` is required.

Statistics for each comparison (variant vs control, same split, paired by seed):
  * primary 95% interval and p-value: point +- t_{S-1} * sqrt(seed variance / S + sequence
    variance), where the seed variance is that of the per-seed aggregate differences and the
    sequence variance comes from a cluster bootstrap over sequences of the seed-averaged
    differences (person-frames of one sequence are strongly correlated, so sequences are the
    resampling unit). With S = 3 seeds the t quantile is 4.30, which keeps coverage honest;
  * secondary (JSON only): seed-only paired t interval and the percentile hierarchical bootstrap;
  * per-sequence win counts and whether all seeds agree in sign (@reliable macro).
Nothing is imputed: a missing run yields no macro, and the paper's \\res{} command prints a
red TBD marker instead of a number.

LaTeX macros (paper/latex/generated/results.tex):
  \\res{<variant>@<split>@<metric>}            mean over seeds, 2 decimals
  \\res{<variant>@<split>@<metric>@pm}         mean with +- std (when >1 seed)
  \\res{<variant>@<split>@nseeds}
  \\res{cmp@<variant>@<control>@<split>@<metric>@diff|ci|wins|p|reliable}
  \\res{cmp@<variant>@<control>@<split>@<metric>@seq@<3DPW sequence name>}   per-sequence diff
metric in {mpjpe (H36M-pelvis root), mpjpehip (hip-midpoint root, WHAM/GVHMR), pampjpe,
pve (unaligned), pvepelvis (H36M-pelvis aligned, CameraHMR/ReFit), pvehip (hip-midpoint
aligned, WHAM/GVHMR), accel (mm/frame^2), accelms (m/s^2 at 30 fps)}. The optional
variants exist only for evaluations produced by scripts/evaluate_target60_v43.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

CORE = ('mpjpe_mm', 'pa_mpjpe_mm', 'pve_mm', 'acceleration_error_mm_per_frame2')
OPTIONAL = ('mpjpe_hip_mm', 'pve_pelvis_mm', 'pve_hip_mm', 'acceleration_error_m_per_s2')
SHORT = {'mpjpe_mm': 'mpjpe', 'pa_mpjpe_mm': 'pampjpe', 'pve_mm': 'pve', 'mpjpe_hip_mm': 'mpjpehip',
         'pve_pelvis_mm': 'pvepelvis', 'pve_hip_mm': 'pvehip', 'acceleration_error_mm_per_frame2': 'accel',
         'acceleration_error_m_per_s2': 'accelms'}
NAME_RE = re.compile(r'^[A-Za-z0-9-]+$')
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042}


def t_quantile(df: int) -> float:
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, df))
    except ImportError:
        return T975[max(k for k in T975 if k <= df)] if df >= 1 else float('nan')


def t_pvalue(t_stat: float, df: int):
    try:
        from scipy import stats
        return float(2 * stats.t.sf(abs(t_stat), df))
    except ImportError:
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_run(entry: dict, base: Path) -> dict:
    for key in ('variant', 'seed', 'split', 'eval_json'):
        if key not in entry:
            raise ValueError(f'Run entry lacks {key}: {entry}')
    if not NAME_RE.match(str(entry['variant'])):
        raise ValueError('Variant names must match [A-Za-z0-9-]+ (used in LaTeX macros): ' + str(entry['variant']))
    path = Path(entry['eval_json'])
    path = path if path.is_absolute() else base / path
    data = json.loads(path.read_text(encoding='utf-8'))
    metrics = data['metrics']
    if data.get('split') not in (None, entry['split']):
        raise ValueError(f'{path}: split {data.get("split")} differs from manifest {entry["split"]}')
    if data.get('status') == 'diagnostic_subset' and not entry.get('allow_diagnostic', False):
        raise ValueError(f'{path}: diagnostic subset evaluations cannot enter paper statistics')
    if not metrics.get('per_sequence'):
        raise ValueError(f'{path}: per_sequence metrics are required')
    names = CORE + tuple(m for m in OPTIONAL if m in metrics)
    return dict(variant=str(entry['variant']), seed=int(entry['seed']), split=str(entry['split']),
                weights=entry.get('weights'), path=str(path), sha256=sha256(path),
                metrics={m: float(metrics[m]) for m in names},
                frames=int(metrics['frames']), triplets=int(metrics['acceleration_triplets']),
                sequences=metrics['per_sequence'])


def check_consistent(runs: list[dict]) -> list[str]:
    first = runs[0]
    order = sorted(first['sequences'])
    for run in runs[1:]:
        if (run['frames'], run['triplets']) != (first['frames'], first['triplets']):
            raise ValueError(f'Coverage differs: {run["path"]} vs {first["path"]}')
        if sorted(run['sequences']) != order:
            raise ValueError(f'Sequence set differs: {run["path"]}')
        for name in order:
            a, b = run['sequences'][name], first['sequences'][name]
            if (a['frames'], a['acceleration_triplets']) != (b['frames'], b['acceleration_triplets']):
                raise ValueError(f'Per-sequence coverage differs for {name}: {run["path"]}')
    return order


def common_metrics(runs):
    names = list(CORE)
    names += [m for m in OPTIONAL if all(m in r['metrics'] and all(m in s for s in r['sequences'].values()) for r in runs)]
    return names


def sequence_sums(run, order, metric):
    weight = np.asarray([float(run['sequences'][q]['acceleration_triplets' if metric.startswith('acceleration') else 'frames'])
                         for q in order])
    return np.asarray([float(run['sequences'][q][metric]) for q in order]) * weight, weight


def describe(values) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    out = dict(n=int(arr.size), mean=float(arr.mean()), min=float(arr.min()), max=float(arr.max()))
    out['std'] = float(arr.std(ddof=1)) if arr.size > 1 else None
    return out


def paired(variant_runs, control_runs, order, metrics, n_boot, rng) -> dict:
    seeds = sorted(set(variant_runs) & set(control_runs))
    if not seeds:
        return {}
    result = dict(seeds=seeds)
    for m in metrics:
        diffs = [variant_runs[k]['metrics'][m] - control_runs[k]['metrics'][m] for k in seeds]
        entry = dict(diff=describe(diffs))
        n = len(diffs)
        if n > 1:
            sd = float(np.std(diffs, ddof=1))
            mean = float(np.mean(diffs))
            half = t_quantile(n - 1) * sd / math.sqrt(n)
            entry['seed_t_ci95'] = [mean - half, mean + half]
            entry['seed_t_p'] = t_pvalue(mean / (sd / math.sqrt(n)), n - 1) if sd > 0 else None
        delta = np.stack([sequence_sums(variant_runs[k], order, m)[0] - sequence_sums(control_runs[k], order, m)[0]
                          for k in seeds])                       # [S, Q] error-sum differences
        weight = sequence_sums(control_runs[seeds[0]], order, m)[1]  # identical coverage (checked)
        per_sequence = delta.mean(0) / np.maximum(weight, 1e-12)
        entry['sequences_improved'] = int((per_sequence < 0).sum())
        entry['sequences_total'] = len(order)
        entry['per_sequence_mean_diff'] = {q: float(v) for q, v in zip(order, per_sequence)}
        point = float(delta.mean(0).sum() / weight.sum())
        boots = np.empty(n_boot)
        n_seed, n_seq = delta.shape
        for b in range(n_boot):
            s_idx = rng.integers(0, n_seed, n_seed)
            q_idx = rng.integers(0, n_seq, n_seq)
            boots[b] = delta[s_idx][:, q_idx].mean(0).sum() / weight[q_idx].sum()
        entry['cluster_bootstrap'] = dict(point=point, ci95=[float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
                                          prob_improvement=float((boots < 0).mean()), resamples=int(n_boot),
                                          unit='seeds then sequences (frame/triplet weighted); secondary, '
                                               'too narrow for few seeds')
        # Primary interval: seed variance with a small-sample t (S-1 df) plus sequence-sampling
        # variance from a cluster bootstrap over sequences of the seed-averaged deltas.
        seed_level = delta.sum(1) / weight.sum()                       # per-seed aggregate differences
        seed_var = float(seed_level.var(ddof=1)) / n_seed if n_seed > 1 else 0.
        mean_delta = delta.mean(0)
        seq_boots = np.empty(n_boot)
        for b in range(n_boot):
            q_idx = rng.integers(0, n_seq, n_seq)
            seq_boots[b] = mean_delta[q_idx].sum() / weight[q_idx].sum()
        seq_var = float(seq_boots.var(ddof=1))
        se = math.sqrt(seed_var + seq_var)
        df = n_seed - 1 if n_seed > 1 else None
        crit = t_quantile(df) if df else 1.959964
        if se > 0:
            stat = abs(point) / se
            p_value = t_pvalue(stat, df) if df else math.erfc(stat / math.sqrt(2))
        else:
            p_value = None
        entry['combined'] = dict(point=point, ci95=[point - crit * se, point + crit * se], p=p_value, se=se,
                                 seed_se=math.sqrt(seed_var), sequence_se=math.sqrt(seq_var), df=df,
                                 method='t(S-1) on sqrt(seed variance / S + sequence cluster-bootstrap variance)')
        entry['seeds_agree_in_sign'] = bool(np.all(np.sign(seed_level) == np.sign(point))) if point != 0 else False
        result[m] = entry
    return result


def fmt(value, digits=2):
    return 'nan' if value is None or not math.isfinite(value) else f'{value:.{digits}f}'


def signed(value, digits=2):
    r"""LaTeX signed number with a typographic minus (or plus), e.g. \ensuremath{-}0.34."""
    if value is None or not math.isfinite(value):
        return 'nan'
    return ('\\ensuremath{-}' if value < 0 else '\\ensuremath{+}') + f'{abs(value):.{digits}f}'


def macro(lines, key, value):
    lines.append('\\expandafter\\def\\csname res@%s\\endcsname{%s}' % (key, value))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out-json', type=Path, required=True)
    parser.add_argument('--out-tex', type=Path, help='LaTeX macro file consumed by paper/latex (\\res{...})')
    parser.add_argument('--out-md', type=Path, help='Markdown summary for humans')
    parser.add_argument('--bootstrap', type=int, default=None)
    parser.add_argument('--rng-seed', type=int, default=20261101)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    base = args.manifest.resolve().parent
    if manifest.get('base_dir'):
        base = (base / manifest['base_dir']).resolve()
    control = manifest.get('control')
    n_boot = args.bootstrap or int(manifest.get('bootstrap', 20000))
    rng = np.random.default_rng(args.rng_seed)

    runs = [load_run(entry, base) for entry in manifest['runs']]
    by_split = {}
    for run in runs:
        bucket = by_split.setdefault(run['split'], {}).setdefault(run['variant'], {})
        if run['seed'] in bucket:
            raise ValueError(f'Duplicate run {run["variant"]}/{run["split"]}/seed {run["seed"]}')
        bucket[run['seed']] = run
    pairs = [tuple(p) for p in manifest.get('comparisons', [])]

    summary = dict(manifest=str(args.manifest), control=control, rng_seed=args.rng_seed, bootstrap=n_boot,
                   runs=[{k: r[k] for k in ('variant', 'seed', 'split', 'weights', 'path', 'sha256', 'metrics')} for r in runs],
                   splits={})
    tex = ['% Auto-generated by scripts/analyze_results_c11.py; do not edit by hand.']
    md = [f'# C11 result summary\n\nBootstrap resamples: {n_boot}. Negative Δ = lower error than the control. '
          '95% CIs combine seed variance (t, S-1 df) and sequence-sampling variance (cluster bootstrap over '
          'sequences); * marks reliable differences (CI excludes 0 and all seeds agree in sign).\n']
    for split, variants in sorted(by_split.items()):
        all_runs = [r for v in variants.values() for r in v.values()]
        order = check_consistent(all_runs)
        metrics = common_metrics(all_runs)
        split_pairs = [p for p in pairs if p[0] in variants and p[1] in variants]
        if control in variants:
            split_pairs += [(v, control) for v in sorted(variants) if v != control and (v, control) not in split_pairs]
        out = dict(frames=all_runs[0]['frames'], triplets=all_runs[0]['triplets'], sequences=order,
                   metrics=metrics, variants={}, comparisons={})
        md.append(f'\n## {split}: {out["frames"]} person-frames, {out["triplets"]} triplets, {len(order)} sequences\n')
        header = ' | '.join(SHORT[m] for m in metrics)
        md.append(f'| variant | seeds | {header} |')
        md.append('|---|---:|' + '---:|' * len(metrics))
        for variant, seeds in sorted(variants.items()):
            stats = {m: describe([seeds[k]['metrics'][m] for k in sorted(seeds)]) for m in metrics}
            out['variants'][variant] = dict(seeds=sorted(seeds), stats=stats)
            prefix = f'{variant}@{split}'
            macro(tex, f'{prefix}@nseeds', str(len(seeds)))
            cells = []
            for m in metrics:
                s, short = stats[m], SHORT[m]
                macro(tex, f'{prefix}@{short}', fmt(s['mean']))
                if s['std'] is not None:
                    macro(tex, f'{prefix}@{short}@std', fmt(s['std']))
                    macro(tex, f'{prefix}@{short}@pm', f'{fmt(s["mean"])}{{\\scriptsize$\\pm${fmt(s["std"])}}}')
                    cells.append(f'{fmt(s["mean"])} ± {fmt(s["std"])}')
                else:
                    macro(tex, f'{prefix}@{short}@pm', fmt(s['mean']))
                    cells.append(fmt(s['mean']))
            md.append(f'| {variant} | {len(seeds)} | ' + ' | '.join(cells) + ' |')
        if split_pairs:
            md.append('\n| comparison | seeds | ' + ' | '.join(f'Δ{SHORT[m]} [95% CI] (wins)' for m in metrics) + ' |')
            md.append('|---|---:|' + '---:|' * len(metrics))
        for variant, ctrl in split_pairs:
            result = paired(variants[variant], variants[ctrl], order, metrics, n_boot, rng)
            if not result:
                continue
            out['comparisons'][f'{variant}__vs__{ctrl}'] = result
            cells = []
            for m in metrics:
                p, short = result[m], SHORT[m]
                lo, hi = p['combined']['ci95']
                key = f'cmp@{variant}@{ctrl}@{split}@{short}'
                macro(tex, f'{key}@diff', signed(p['combined']['point']))
                macro(tex, f'{key}@ci', f'[{signed(lo)}, {signed(hi)}]')
                macro(tex, f'{key}@reliable', 'yes' if (hi < 0 or lo > 0) and p['seeds_agree_in_sign'] else 'no')
                for sequence, value in p['per_sequence_mean_diff'].items():
                    macro(tex, f'{key}@seq@{sequence}', signed(value))
                macro(tex, f'{key}@wins', f'{p["sequences_improved"]}/{p["sequences_total"]}')
                if p['combined']['p'] is not None:
                    macro(tex, f'{key}@p', f'{p["combined"]["p"]:.3f}')
                flag = '' if not ((hi < 0 or lo > 0) and p['seeds_agree_in_sign']) else ' *'
                cells.append(f'{p["combined"]["point"]:+.3f} [{lo:+.3f}, {hi:+.3f}] ({p["sequences_improved"]}/{p["sequences_total"]}){flag}')
            md.append(f'| {variant} vs {ctrl} | {len(result["seeds"])} | ' + ' | '.join(cells) + ' |')
        summary['splits'][split] = out

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    if args.out_tex:
        args.out_tex.parent.mkdir(parents=True, exist_ok=True)
        args.out_tex.write_text('\n'.join(tex) + '\n', encoding='utf-8')
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text('\n'.join(md) + '\n', encoding='utf-8')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
