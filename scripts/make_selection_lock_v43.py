#!/usr/bin/env python3
"""Write an immutable validation selection lock that authorises ONE test evaluation.

The lock binds: checkpoint SHA256, weight field (model / ema_model), architecture config,
optimizer step, and a complete validation evaluation JSON of exactly that checkpoint.
scripts/evaluate_target60_v43.py refuses test evaluation of v36/v43 models without it.

Pre-registration rule used by C11 (state it in the paper): every compared variant is
locked at the same fixed step (default 1,000 new optimizer steps, EMA weights). No test
number is ever used to choose a step, seed, or variant.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--checkpoint-weights', choices=('model', 'ema_model'), required=True)
    p.add_argument('--validation-eval', type=Path, required=True,
                   help='evaluate_target60_v43.py (or v36) JSON for this checkpoint on the full validation split')
    p.add_argument('--rule', default='fixed pre-registered step; EMA weights; same for all variants')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()

    import torch
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    evaluation = json.loads(args.validation_eval.read_text(encoding='utf-8'))
    checkpoint_sha = sha256(args.checkpoint)
    if evaluation.get('split') != 'validation' or evaluation.get('status') != 'complete_matched_project_protocol':
        raise ValueError('Lock requires a complete formal validation evaluation')
    if evaluation['assets']['checkpoint']['sha256'] != checkpoint_sha:
        raise ValueError('Validation evaluation was computed for a different checkpoint file')
    if evaluation['checkpoint']['weights_field'] != args.checkpoint_weights:
        raise ValueError('Validation evaluation used a different weight field')
    metrics = evaluation['metrics']
    lock = dict(format='star-body-selection-lock-v43', selection_split='validation',
                checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_sha,
                checkpoint_weights=args.checkpoint_weights, checkpoint_format=payload.get('format'),
                v36_config=payload.get('v36_config'), step=payload.get('step'),
                variant=payload.get('v43_variant', payload.get('variant')),
                validation_frames=metrics['frames'], validation_triplets=metrics['acceleration_triplets'],
                validation_metrics={k: metrics[k] for k in metrics if k.endswith('_mm') or k.endswith('frame2')},
                validation_eval=str(args.validation_eval.resolve()), validation_eval_sha256=sha256(args.validation_eval),
                rule=args.rule, created_utc=datetime.now(timezone.utc).isoformat(), test_metrics_seen=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(lock, stream, indent=2)
    print(json.dumps(lock, indent=2))


if __name__ == '__main__':
    main()
