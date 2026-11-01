"""C11 (v43) CLI contract: v36 arguments plus evidence-router and external-evidence options.

All original v36 options keep their meaning (target60_contract_v36 is unchanged). The v37
switch ``--v37-joint-feedback`` is replaced by the explicit ``--v43-feedback`` family:

  --v43-feedback none      continuation control without any landmark feedback
  --v43-feedback mean      C00 mean-feedback control (unchanged 4-d pooled feedback)
  --v43-feedback router    evidence router; with all defaults this is exactly C10
  --v43-router-sources     head | external | head+external
  --v43-router-routing     anatomy | none | shuffled
  --v43-router-identity / --v43-router-no-identity
  --v43-router-pooling     joint | mean | concat (mean = capacity-matched pooled control,
                           concat = ordered-vector WHAM-style control)
  --v43-router-init-seed   router initialisation stream (default 370026 = C10)
  --train-external-keypoints / --val-external-keypoints   caches from
                           scripts/cache_external_keypoints_v43.py (required for 'external')
"""
import argparse
from pathlib import Path

from divibody_training.target60_contract_v36 import (  # noqa: F401  (re-exported)
    parse_training_args as parse_v36_args, inspect_assets as inspect_v36_assets, file_fingerprint,
    source_fingerprints)

V43_DEFAULT_INIT_SEED = 370026


def _v43_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--v43-variant', default='unnamed', help='Free-form run label recorded in metadata')
    p.add_argument('--v43-feedback', choices=('none', 'mean', 'router'), required=True)
    p.add_argument('--v43-router-sources', choices=('head', 'external', 'head+external'), default='head')
    p.add_argument('--v43-router-routing', choices=('anatomy', 'none', 'shuffled'), default='anatomy')
    p.add_argument('--v43-router-identity', dest='v43_router_identity', action='store_true', default=True)
    p.add_argument('--v43-router-no-identity', dest='v43_router_identity', action='store_false')
    p.add_argument('--v43-router-pooling', choices=('joint', 'mean', 'concat'), default='joint')
    p.add_argument('--v43-router-init-seed', type=int, default=V43_DEFAULT_INIT_SEED)
    p.add_argument('--v43-external-min-confidence', type=float, default=0.)
    p.add_argument('--v43-checkpoint-every', type=int, default=0,
                   help='Write a resumable latest.pt every N optimizer steps without evaluating (0 = only at evaluations)')
    p.add_argument('--train-external-keypoints', type=Path)
    p.add_argument('--val-external-keypoints', type=Path)
    return p


def parse_training_args(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--v37-joint-feedback' in argv:
        raise SystemExit('v43: use --v43-feedback router (C10 defaults) instead of --v37-joint-feedback')
    extra, rest = _v43_parser().parse_known_args(argv)
    args = parse_v36_args(rest)
    for key, value in vars(extra).items():
        setattr(args, key, value)
    error = validate_v43(args)
    if error:
        raise SystemExit('v43 argument error: ' + error)
    return args


def validate_v43(args):
    if args.v43_feedback == 'router' and args.v36_arm not in ('R2', 'R3a', 'R3b', 'R4'):
        return 'router feedback requires an R2-compatible arm'
    if args.v43_feedback in ('mean', 'none'):
        defaults = dict(v43_router_sources='head', v43_router_routing='anatomy', v43_router_identity=True,
                        v43_router_pooling='joint', v43_router_init_seed=V43_DEFAULT_INIT_SEED)
        changed = [k for k, v in defaults.items() if getattr(args, k) != v]
        if changed:
            return f'router options have no effect with --v43-feedback {args.v43_feedback}: ' + ', '.join(changed)
    if args.v43_router_pooling in ('mean', 'concat') and (args.v43_router_identity or args.v43_router_routing != 'none'):
        return 'global router controls (mean/concat) require --v43-router-no-identity --v43-router-routing none'
    if 'external' in args.v43_router_sources and args.v43_feedback == 'router':
        if not args.train_external_keypoints or not args.val_external_keypoints:
            return 'external evidence requires --train-external-keypoints and --val-external-keypoints'
    elif args.train_external_keypoints or args.val_external_keypoints:
        return 'external keypoint caches given but the configured router does not read external evidence'
    if args.v43_checkpoint_every < 0:
        return '--v43-checkpoint-every must be >= 0'
    if not 0. <= args.v43_external_min_confidence < 1.:
        return '--v43-external-min-confidence must be in [0,1)'
    return None


def uses_external(args):
    return args.v43_feedback == 'router' and 'external' in args.v43_router_sources


def inspect_assets(args, fingerprint=False):
    audit = inspect_v36_assets(args, fingerprint)
    if uses_external(args):
        for key in ('train_external_keypoints', 'val_external_keypoints'):
            path = Path(getattr(args, key))
            audit['required_paths'][key] = str(path)
            if not path.is_file():
                audit['missing'][key] = str(path)
            elif fingerprint and audit['passed']:
                audit['assets'][key] = file_fingerprint(path, content=True)
        audit['passed'] = not audit['missing'] and not audit['errors']
    return audit
