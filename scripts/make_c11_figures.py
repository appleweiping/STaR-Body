#!/usr/bin/env python3
"""Paper figures for C11 from released result files only (no model inference).

  python scripts/make_c11_figures.py --matrix configs/c11/matrix.local.json \
      --summary evidence/c11/summary.json --out-dir paper/latex/figures/generated

Produces (each skipped with a message if its inputs are missing):
  ablation.pdf    paired differences (variant - control, mm) with 95% cluster-bootstrap CIs,
                  validation (circles) and test (squares), MPJPE and PA-MPJPE panels
  sequences.pdf   per-sequence seed-averaged PVE (H36M-pelvis aligned) difference for one
                  comparison on test (default B4-ext-joint vs B1-ext-pooled), sorted
  confidence.pdf  per-frame MPJPE difference (default B4-ext-joint minus A0-none) vs ViTPose
                  mean keypoint confidence quintile on test, with sequence-cluster bootstrap CIs (needs per-frame .npz files
                  from evaluate_target60_v43.py --per-frame-output and the keypoint cache)
Colors: fixed categorical slots (blue = validation, orange = test) plus marker shape, so
identity never relies on color alone.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e4e3df'
LABELS = {'mpjpe_mm': 'Δ MPJPE (mm)', 'pa_mpjpe_mm': 'Δ PA-MPJPE (mm)', 'pve_hip_mm': 'Δ PVE (mm)',
          'pve_pelvis_mm': 'Δ PVE (mm)'}


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def style(plt):
    plt.rcParams.update({'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'DejaVu Sans'})


def ablation(summary, out, plt, metrics=('mpjpe_mm', 'pa_mpjpe_mm')):
    pairs = []
    for split in ('validation', 'test'):
        for key in summary['splits'].get(split, {}).get('comparisons', {}):
            if key not in pairs:
                pairs.append(key)
    if not pairs:
        print('ablation: no comparisons in summary; skipped')
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.9, 0.32 * len(pairs) + 0.9), sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(pairs))[::-1]
    for ax, metric in zip(axes, metrics):
        ax.axvline(0, color=MUTED, lw=1)
        ax.grid(axis='x', color=GRID, lw=.6)
        for offset, split, color, marker in ((.14, 'validation', BLUE, 'o'), (-.14, 'test', ORANGE, 's')):
            comps = summary['splits'].get(split, {}).get('comparisons', {})
            for row, key in zip(y, pairs):
                entry = comps.get(key, {}).get(metric)
                if not entry:
                    continue
                point = entry['combined']['point']
                lo, hi = entry['combined']['ci95']
                ax.plot([lo, hi], [row + offset] * 2, color=color, lw=2, solid_capstyle='round')
                ax.plot(point, row + offset, marker=marker, ms=5, color=color, mec='white', mew=.8, ls='none',
                        label=split if row == y[0] else None)
        ax.set_xlabel(LABELS.get(metric, metric) + '  (← better)')
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([k.replace('__vs__', ' vs ') for k in pairs])
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper right', frameon=False, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(out / 'ablation.pdf')
    plt.close(fig)
    print('wrote', out / 'ablation.pdf')


def sequences(summary, out, plt, comparison, metric='pve_pelvis_mm'):
    entry = summary['splits'].get('test', {}).get('comparisons', {}).get(comparison, {}).get(metric)
    if not entry:
        print(f'sequences: {comparison} not on test; skipped')
        return
    items = sorted(entry['per_sequence_mean_diff'].items(), key=lambda kv: kv[1])
    names = [k.split('_', 1)[-1] for k, _ in items]  # drop the location prefix (downtown_, outdoors_, ...)
    values = np.asarray([v for _, v in items])
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    colors = [BLUE if v < 0 else ORANGE for v in values]
    ax.bar(np.arange(len(values)), values, color=colors, width=.75)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(names, fontsize=5, rotation=90)
    ax.set_ylabel(LABELS.get(metric, metric) + '  (↓ better)')
    ax.set_title(comparison.replace('__vs__', ' vs ') + ' (test; blue = improved)', fontsize=7, color=INK)
    ax.grid(axis='y', color=GRID, lw=.6)
    fig.tight_layout()
    fig.savefig(out / 'sequences.pdf')
    plt.close(fig)
    print('wrote', out / 'sequences.pdf')


def per_frame(results, split, variant, seeds, metric):
    arrays = []
    for seed in seeds:
        path = results / split / f'{variant}_s{seed}.npz'
        if not path.is_file():
            return None, None
        data = np.load(path)
        arrays.append((data['uid'], data[metric]))
    uid = arrays[0][0]
    if any(not np.array_equal(uid, a[0]) for a in arrays):
        raise ValueError('per-frame UID order differs across seeds')
    return uid, np.mean([a[1] for a in arrays], 0)


def confidence(matrix, out, plt, variant, control, metric='mpjpe_mm', bins=5, n_boot=2000):
    import torch
    results = resolve(matrix['results_dir'])
    seeds = matrix['seeds']
    uid, a = per_frame(results, 'test', variant, seeds, metric)
    uid_c, b = per_frame(results, 'test', control, seeds, metric)
    if uid is None or uid_c is None:
        print('confidence: per-frame test files missing; skipped')
        return
    if not np.array_equal(uid, uid_c):
        raise ValueError('variant/control per-frame UIDs differ')
    pack = torch.load(resolve(matrix['keypoints']['vitpose']['test']), map_location='cpu', weights_only=False)
    index = {u: i for i, u in enumerate(pack['frame_uids'])}
    conf = pack['keypoints'][..., 2].float().mean(-1).numpy()
    c = np.asarray([conf[index[u]] for u in uid])
    diff = a - b
    sequence = np.asarray([u.split('/person_')[0] for u in uid])
    edges = np.quantile(c, np.linspace(0, 1, bins + 1))
    rng = np.random.default_rng(0)
    centers, means, los, his = [], [], [], []
    for k in range(bins):
        mask = (c >= edges[k]) & (c <= edges[k + 1] if k == bins - 1 else c < edges[k + 1])
        seqs = np.unique(sequence[mask])
        sums = np.asarray([diff[mask & (sequence == s)].sum() for s in seqs])
        counts = np.asarray([(mask & (sequence == s)).sum() for s in seqs])
        boots = []
        for _ in range(n_boot):
            pick = rng.integers(0, len(seqs), len(seqs))
            boots.append(sums[pick].sum() / max(counts[pick].sum(), 1))
        centers.append(np.median(c[mask]))
        means.append(sums.sum() / counts.sum())
        los.append(np.quantile(boots, .025))
        his.append(np.quantile(boots, .975))
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.axhline(0, color=MUTED, lw=1)
    ax.grid(axis='y', color=GRID, lw=.6)
    ax.fill_between(centers, los, his, color=ORANGE, alpha=.18, lw=0)
    ax.plot(centers, means, color=ORANGE, lw=2, marker='s', ms=5, mec='white', mew=.8)
    ax.set_xlabel('ViTPose confidence (bin median)')
    ax.set_ylabel(LABELS.get(metric, metric))
    ax.set_title(f'{variant} − {control} (test, mean of {len(seeds)} seeds)', fontsize=7, color=INK)
    fig.tight_layout()
    fig.savefig(out / 'confidence.pdf')
    plt.close(fig)
    print('wrote', out / 'confidence.pdf')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--matrix', type=Path, required=True)
    p.add_argument('--summary', type=Path, default=ROOT / 'evidence/c11/summary.json')
    p.add_argument('--out-dir', type=Path, default=ROOT / 'paper/latex/figures/generated')
    p.add_argument('--sequence-comparison', default='B4-ext-joint__vs__B1-ext-pooled')
    p.add_argument('--confidence-variant', default='B4-ext-joint')
    p.add_argument('--confidence-control', default='A0-none')
    args = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    style(plt)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    matrix = json.loads(args.matrix.read_text(encoding='utf-8'))
    summary = json.loads(args.summary.read_text(encoding='utf-8')) if args.summary.is_file() else {'splits': {}}
    ablation(summary, args.out_dir, plt)
    sequences(summary, args.out_dir, plt, args.sequence_comparison)
    confidence(matrix, args.out_dir, plt, args.confidence_variant, args.confidence_control)


if __name__ == '__main__':
    main()
