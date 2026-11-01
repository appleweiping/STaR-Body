#!/usr/bin/env python3
"""C11 preflight: verify server assets against the C11 matrix before spending GPU time (CPU only).

Checks, per split (train / validation / test):
  * VACE cache + HMR sidecar exist, declare the split, and align (window_uids, window counts);
  * metadata carries ``target_source_frame_ids`` (otherwise the R2 temporal blocks silently
    see every frame as its own segment);
  * owned-frame / triplet counts match the protocol (validation 10,413 / 10,226,
    test 35,515 / 34,752) and train matches the hard-coded acceleration coefficients
    (Nw 16,933 + Nb 5,548);
  * track prototypes and external keypoint caches exist and cover the sidecar frame UIDs.
Writes a JSON report; exit code 1 if any hard check fails.

  bash scripts/run_star_body.sh scripts/preflight_c11.py --matrix configs/c11/matrix.local.json \
      --output results/c11/preflight.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {'validation': (10413, 10226), 'test': (35515, 34752)}
TRAIN_TRIPLETS = (16933, 5548)


def resolve(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def owned_counts(vace):
    owned = set()
    for row, record in enumerate(vace['metadata']):
        for uid, own, valid in zip(record['target_frame_uids'], vace['frame_owner_mask'][row].tolist(),
                                   vace['target_valid_mask'][row].tolist()):
            if own and valid:
                owned.add(uid)
    triplets = 0
    for uid in owned:
        prefix, frame = uid.rsplit('/frame_', 1)
        i = int(frame)
        if f'{prefix}/frame_{i - 1:05d}' in owned and f'{prefix}/frame_{i + 1:05d}' in owned:
            triplets += 1
    return len(owned), triplets


def check_split(split, vace_path, hmr_path, prototypes, keypoints, report, torch):
    out = report.setdefault(split, {})
    errors = out.setdefault('errors', [])
    for name, path in (('vace', vace_path), ('hmr', hmr_path), ('prototypes', prototypes)):
        if path is None or not resolve(path).is_file():
            errors.append(f'missing {name}: {path}')
    if errors:
        return
    vace = torch.load(resolve(vace_path), map_location='cpu', weights_only=False, mmap=True)
    hmr = torch.load(resolve(hmr_path), map_location='cpu', weights_only=False, mmap=True)
    out['vace_signature'] = {k: vace['signature'].get(k) for k in ('format', 'split', 'feature_only')}
    out['hmr_signature'] = {k: hmr['signature'].get(k) for k in ('format', 'split')} if 'signature' in hmr else None
    if vace['signature'].get('split') != split:
        errors.append(f'VACE cache declares split {vace["signature"].get("split")}')
    windows = len(vace['metadata'])
    out['windows'] = windows
    if len(hmr['window_frame_indices']) != windows:
        errors.append('HMR window count differs from VACE metadata')
    if [r['window_uid'] for r in vace['metadata']] != list(hmr.get('window_uids', [])):
        errors.append('window_uids differ between VACE and HMR caches')
    missing_source = sum(r.get('target_source_frame_ids') is None for r in vace['metadata'])
    out['windows_without_target_source_frame_ids'] = missing_source
    if missing_source:
        errors.append(f'{missing_source} windows lack target_source_frame_ids (temporal blocks would be inert)')
    frames, triplets = owned_counts(vace)
    out['owned_frames'], out['owned_triplets'] = frames, triplets
    if split in EXPECTED and (frames, triplets) != EXPECTED[split]:
        errors.append(f'coverage {frames}/{triplets} != protocol {EXPECTED[split]}')
    if split == 'train' and frames != 22735:
        errors.append(f'train owned frames {frames} != 22,735 (acceleration coefficients assume this cache)')
    pack = torch.load(resolve(prototypes), map_location='cpu', weights_only=False)
    out['prototype_format'] = pack.get('format')
    if pack.get('split') != split:
        errors.append(f'prototype split {pack.get("split")}')
    uids = list(hmr['frame_uids'])
    for source, path in (keypoints or {}).items():
        entry = out.setdefault('keypoints', {})
        if path is None or not resolve(path).is_file():
            entry[source] = 'missing (create with run_c11_matrix.py keypoints)'
            continue
        kp = torch.load(resolve(path), map_location='cpu', weights_only=False)
        covered = set(kp['frame_uids']) >= set(uids)
        conf = kp['keypoints'][..., 2]
        entry[source] = dict(format=kp.get('format'), split=kp.get('split'), covers_sidecar=covered,
                             frames=len(kp['frame_uids']), mean_confidence=float(conf.mean()),
                             frac_conf_gt_0_3=float((conf > .3).float().mean()))
        if not covered or kp.get('split') != split:
            errors.append(f'keypoints {source} do not cover the {split} sidecar')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--matrix', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--splits', nargs='+', default=['train', 'validation', 'test'])
    args = p.parse_args()
    import torch
    matrix = json.loads(args.matrix.read_text(encoding='utf-8'))
    base = json.loads(resolve(matrix['base_config']).read_text(encoding='utf-8'))
    ev = matrix['eval']
    sources = {split: {name: paths.get(split) for name, paths in matrix['keypoints'].items()}
               for split in ('train', 'validation', 'test')}
    plan = dict(train=(base['train-vace-cache'], base['train-hmr-sidecar'], base.get('train-track-prototypes')),
                validation=(base['val-vace-cache'], base['val-hmr-sidecar'], base.get('val-track-prototypes')),
                test=(ev['test']['vace-cache'], ev['test']['hmr-sidecar'], ev['test']['track-prototypes']))
    report = {}
    for split in args.splits:
        try:
            check_split(split, *plan[split], sources[split], report, torch)
        except Exception as error:  # report and continue with other splits
            report.setdefault(split, {}).setdefault('errors', []).append(f'{type(error).__name__}: {error}')
    for key in ('init-checkpoint', 'parent-validation-report', 'initial-reference-report', 'validation-manifest'):
        if not resolve(base[key]).is_file():
            report.setdefault('training_assets', {}).setdefault('errors', []).append(f'missing {key}: {base[key]}')
    if 'test' in args.splits:
        for key in ('gt-cache', 'coverage-manifest'):
            if not resolve(ev['test'][key]).is_file():
                report['test'].setdefault('errors', []).append(f'missing test {key}: {ev["test"][key]}')
    failed = any(section.get('errors') for section in report.values())
    report['passed'] = not failed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    print(json.dumps(report, indent=2, default=str))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
