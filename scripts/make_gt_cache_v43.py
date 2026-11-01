#!/usr/bin/env python3
"""Write the separate UID-indexed 3DPW ground-truth cache required for test evaluation.

Reads ONLY the official 3DPW sequence pickles (poses, betas, genders, cam_poses); no images,
no model. The person-frame set is the same strict set as the HMR crop dataset
(UnifiedHMR3DPWDataset: image_XXXXX.jpg frames), so every sidecar UID is present.
Output keys match what scripts/evaluate_target60_v36.py / _v43.py read:
  sequences, person_id, frame_id, gt_full_pose[N,72], gt_betas[N,10], gt_gender[N] (0 m, 1 f),
  gt_cam_pose[N,4,4], signature{split, format, ...}
This file holds labels: keep it separate from feature caches and never pass it to training.

  bash scripts/run_star_body.sh scripts/make_gt_cache_v43.py --data-root /data/3DPW --split test \
      --output assets/cache/test_gt.pt [--hmr-sidecar assets/cache/test_hmr.pt]
"""
import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--split', choices=('train', 'validation', 'test'), default='test')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--hmr-sidecar', type=Path, help='Optional: verify that every sidecar frame UID is covered')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import numpy as np
    import torch
    from divibody_training.audit_hmr2a_crop_protocol_v11 import RawBBoxHMR3DPWDataset
    from divibody_training.evaluate_unified_hmr2_divibody_3dpw import _normalise_gender, _load_annotation, _to_numpy

    dataset = RawBBoxHMR3DPWDataset(root=args.data_root, split=args.split, image_size=256, frame_stride=1,
                                    max_samples=None, crop_to_person=True, crop_margin=.25)
    rows = dict(sequences=[], person_id=[], frame_id=[], gt_full_pose=[], gt_betas=[], gt_gender=[], gt_cam_pose=[])
    cache = {}
    for sample in dataset.samples:
        if sample.annotation_path not in cache:
            cache.clear()  # samples are grouped by sequence; keep one annotation in memory
            cache[sample.annotation_path] = _load_annotation(sample.annotation_path)
        ann = cache[sample.annotation_path]
        pose = _to_numpy(ann['poses'][sample.person_index][sample.frame_index]).reshape(-1)[:72]
        betas = _to_numpy(ann['betas'][sample.person_index]).reshape(-1)[:10]
        camera = _to_numpy(ann['cam_poses'][sample.frame_index])
        if pose.size != 72 or betas.size != 10 or camera.shape != (4, 4):
            raise ValueError(f'Invalid target for {sample.sequence}/{sample.person_index}/{sample.frame_index}')
        rows['sequences'].append(sample.sequence)
        rows['person_id'].append(int(sample.person_index))
        rows['frame_id'].append(int(sample.frame_index))
        rows['gt_full_pose'].append(pose.astype(np.float32))
        rows['gt_betas'].append(betas.astype(np.float32))
        rows['gt_gender'].append(_normalise_gender(ann['genders'][sample.person_index]))
        rows['gt_cam_pose'].append(camera.astype(np.float32))
    payload = dict(sequences=rows['sequences'], person_id=torch.tensor(rows['person_id']),
                   frame_id=torch.tensor(rows['frame_id']),
                   gt_full_pose=torch.from_numpy(np.stack(rows['gt_full_pose'])),
                   gt_betas=torch.from_numpy(np.stack(rows['gt_betas'])),
                   gt_gender=torch.tensor(rows['gt_gender'], dtype=torch.long),
                   gt_cam_pose=torch.from_numpy(np.stack(rows['gt_cam_pose'])))
    if not all(torch.isfinite(payload[k]).all() for k in ('gt_full_pose', 'gt_betas', 'gt_cam_pose')):
        raise ValueError('Non-finite 3DPW targets')
    uids = [f'{s}/person_{p}/frame_{f:05d}' for s, p, f in zip(rows['sequences'], rows['person_id'], rows['frame_id'])]
    if len(set(uids)) != len(uids):
        raise ValueError('Duplicate person-frame UIDs')
    if args.hmr_sidecar:
        sidecar = torch.load(args.hmr_sidecar, map_location='cpu', weights_only=False, mmap=True)
        missing = set(sidecar['frame_uids']) - set(uids)
        if missing:
            raise ValueError(f'GT cache misses {len(missing)} sidecar UIDs, e.g. {sorted(missing)[:3]}')
    payload['signature'] = dict(format='star-body-3dpw-gt-cache-v43', split=args.split, frames=len(uids),
                                uid_sha256=hashlib.sha256(('\n'.join(sorted(uids)) + '\n').encode()).hexdigest(),
                                source='official 3DPW sequenceFiles pickles', created_utc=datetime.now(timezone.utc).isoformat())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + '.tmp')
    torch.save(payload, tmp)
    tmp.replace(args.output)
    print(payload['signature'])


if __name__ == '__main__':
    main()
