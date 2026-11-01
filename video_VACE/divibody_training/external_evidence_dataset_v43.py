"""Attach cached external 2D keypoints (detector outputs) to 3DPW window samples.

The cache is produced by scripts/cache_external_keypoints_v43.py and is aligned to the
HMR sidecar's ``frame_uids``. Each window reads the same sidecar frame indices that
V8Dataset uses for ``rgb_tokens`` (``sidecar['window_frame_indices']``), so evidence and
image tokens always refer to the same person-frames. No ground truth is read here.
"""
import hashlib

import torch
from torch.utils.data import Dataset

FORMAT = 'star-body-external-keypoints-v43-r1'


def _uid_digest(uids):
    return hashlib.sha256(('\n'.join(uids) + '\n').encode()).hexdigest()


def load_external_keypoints(path, sidecar, split):
    pack = torch.load(path, map_location='cpu', weights_only=False)
    if pack.get('format') != FORMAT:
        raise ValueError(f'{path}: not a {FORMAT} cache')
    if pack.get('split') != split:
        raise ValueError(f'{path}: split {pack.get("split")} != {split}')
    if pack.get('ground_truth_used') is not False:
        raise ValueError(f'{path}: external evidence must declare ground_truth_used=False')
    if pack.get('development_only'):
        raise ValueError(f'{path}: development-only keypoint cache cannot be used for training/evaluation')
    keypoints = pack['keypoints']
    sidecar_uids = list(sidecar['frame_uids'])
    if list(pack['frame_uids']) != sidecar_uids:
        index = {uid: row for row, uid in enumerate(pack['frame_uids'])}
        missing = [uid for uid in sidecar_uids if uid not in index]
        if missing:
            raise ValueError(f'{path}: {len(missing)} sidecar frames lack keypoints, e.g. {missing[:3]}')
        keypoints = keypoints[[index[uid] for uid in sidecar_uids]]
    keypoints = keypoints.float()
    if tuple(keypoints.shape) != (len(sidecar_uids), 17, 3) or not torch.isfinite(keypoints).all():
        raise ValueError(f'{path}: keypoints must be finite [N,17,3]')
    audit = dict(format=FORMAT, source=pack.get('source'), model=pack.get('model'), split=split,
                 frames=len(sidecar_uids), frame_uids_sha256=_uid_digest(sidecar_uids),
                 sidecar_sha256_recorded=pack.get('sidecar_sha256'),
                 mean_confidence=float(keypoints[..., 2].mean()), ground_truth_used=False)
    return keypoints, audit


class ExternalKeypointDataset(Dataset):
    """Wraps V10Dataset (or anything exposing .vace/.sidecar/.lookup) and adds evidence."""

    def __init__(self, dataset, keypoint_path, split):
        self.dataset = dataset
        self.vace = dataset.vace
        self.sidecar = dataset.sidecar
        self.lookup = getattr(dataset, 'lookup', self.sidecar['window_frame_indices'])
        self.keypoints, self.evidence_audit = load_external_keypoints(keypoint_path, self.sidecar, split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample['external_keypoints'] = self.keypoints[self.lookup[index]]
        return sample
