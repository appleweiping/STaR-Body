#!/usr/bin/env python3
"""Day-0 audit of parent / C00 / C10 checkpoints (CPU, seconds; no data needed).

Answers the questions that decide the C11 paper narrative:
  1. Is the inherited observation head (the source of C10 "landmark evidence") still at its
     zero initialisation?  Code reading says V30 builds it with zero coordinate heads,
     log-variance weight 0 / bias -4, visibility 0, and no later stage trains it (2D loss
     weights are 0 and the V34/V36 contracts forbid them). If so, C10's residual is
     P(x_reference) - P(x_current) (drift from the frozen reference), not image evidence.
  2. Which AdaptationConfig the A5 parent really uses (refinement steps, step bounds, GRU width).
  3. Seeds / steps / configs recorded in each checkpoint (e.g. was C00 run with seed 2026?).

Usage:
  bash scripts/run_star_body_v43.sh scripts/inspect_checkpoint_v43.py \
      --checkpoint assets/checkpoints/refinement_parent.pt runs/c10/best.pt runs/c00/best.pt \
      --output results/c11/checkpoint_audit.json
"""
import argparse
import json
import re
from pathlib import Path

PATTERN = re.compile(r'observation\.(coordinate|log_variance|visibility)')


def describe(tensor):
    import torch
    t = tensor.detach().float()
    return dict(shape=list(t.shape), max_abs=float(t.abs().max()) if t.numel() else 0.,
                mean=float(t.mean()) if t.numel() else 0., unique_if_small=sorted(set(t.flatten().tolist()))[:4]
                if t.numel() <= 4096 and len(set(t.flatten().tolist())) <= 4 else None)


def audit(path):
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    out = dict(path=str(path), format=payload.get('format'), step=payload.get('step'), variant=payload.get('variant'),
               development_only=payload.get('development_only'),
               seed=(payload.get('args') or {}).get('seed'), v36_config=payload.get('v36_config'),
               adaptation_config=payload.get('adaptation_config'),
               validation=payload.get('validation'), observation_head={}, weights_fields=[])
    for field in ('model', 'ema_model'):
        state = payload.get(field)
        if not isinstance(state, dict):
            continue
        out['weights_fields'].append(field)
        heads = {k: describe(v) for k, v in state.items() if PATTERN.search(k) and hasattr(v, 'shape')}
        out['observation_head'][field] = heads
    verdicts = {}
    for field, all_heads in out['observation_head'].items():
        # The frozen V29 reference has its own (possibly trained) observation head; C10 reads
        # the V30 head, whose keys do not pass through '.reference.'.
        heads = {k: v for k, v in all_heads.items() if '.reference.' not in k and not k.startswith('reference.')}
        coord = [v for k, v in heads.items() if '.coordinate' in k]
        logvar_w = [v for k, v in heads.items() if '.log_variance' in k and k.endswith('weight')]
        logvar_b = [v for k, v in heads.items() if '.log_variance' in k and k.endswith('bias')]
        vis = [v for k, v in heads.items() if '.visibility' in k]
        if not heads:
            verdicts[field] = 'no observation head tensors found (check key names)'
            continue
        untouched = (all(v['max_abs'] == 0 for v in coord) and all(v['max_abs'] == 0 for v in logvar_w)
                     and all(abs(v['mean'] + 4) < 1e-6 and v['max_abs'] == 4 for v in logvar_b)
                     and all(v['max_abs'] == 0 for v in vis))
        verdicts[field] = ('UNTRAINED: observed_2d == projection of the frozen reference; logvar == -4; '
                           'visibility == 0.5. C10 "evidence" is reference drift, not image evidence.'
                           if untouched else 'TRAINED (non-initial values found): inspect statistics above')
    out['observation_head_verdict'] = verdicts
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    report = [audit(path) for path in args.checkpoint]
    text = json.dumps(report, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    for row in report:
        print(json.dumps(dict(path=row['path'], format=row['format'], step=row['step'], seed=row['seed'],
                              verdict=row['observation_head_verdict'],
                              refinement={k: (row['adaptation_config'] or {}).get(k) for k in
                                          ('gru_width', 'refinement_steps', 'root_step_degrees', 'body_step_degrees',
                                           'use_gru', 'use_track_shape', 'track_shape_bound')}), indent=2))


if __name__ == '__main__':
    main()
