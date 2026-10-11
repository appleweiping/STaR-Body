"""Versioned preprocessing helpers; existing RGB anchors are read-only."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Subset
from divibody_training.train_divibody_rgb_vace_hmr2a_v8 import load_pair, load
from divibody_training.cache_hmr2a_rgb_anchor_v8 import (
    validate_vace_source, validate_sidecar, semantic_vace_digest)
from divibody_training.ram_adaptation_runtime_v13 import V10Dataset

def square_bbox(points, width, height, margin=.25):
    """Integer square in original-image coordinates; PIL provides zero padding."""
    xy = None
    if points is not None:
        points = np.asarray(points)
        if points.ndim == 2 and points.shape[0] == 3:
            xy, conf = points[:2].T, points[2]
        elif points.ndim == 2 and points.shape[1] >= 3:
            xy, conf = points[:,:2], points[:,2]
        if xy is not None:
            xy = xy[np.isfinite(xy).all(1) & np.isfinite(conf) & (conf > 0)]
    reliable = xy is not None and len(xy) >= 2 and bool(((xy.max(0)-xy.min(0)) >= 2).all())
    if reliable:
        low, high = xy.min(0), xy.max(0)
        center = (low+high)/2
        side = math.ceil(float(max(high-low))*(1+2*margin))
    else:
        center, side = np.array([width,height])/2, max(width,height)
    left, top = np.floor(center-side/2).astype(int)
    return (int(left),int(top),int(left+side),int(top+side)), reliable

def remap_anchor(source, vace):
    """Reuse RGB/SMPL by exact UID, never reuse VACE tokens for new contexts."""
    lookup = {uid:i for i,uid in enumerate(source['frame_uids'])}
    try:
        indices = [[lookup[uid] for uid in m['target_frame_uids']] for m in vace['metadata']]
    except KeyError as error:
        raise ValueError('New phase has a frame absent from verified RGB source') from error
    sidecar = dict(source)
    sidecar['window_frame_indices'] = torch.tensor(indices,dtype=torch.long)
    sidecar['window_uids'] = [m['window_uid'] for m in vace['metadata']]
    sidecar['signature'] = dict(source['signature'],
        vace_semantic_sha256=semantic_vace_digest(vace,len(indices)),
        remapping_revision='same-rgb-exact-uid-v14')
    validate_sidecar(sidecar,vace)
    return sidecar

class PhaseBank:
    def __init__(self, manifest_path, seed, max_windows=0):
        path = Path(manifest_path).resolve()
        spec = json.loads(path.read_text())
        self.manifest = spec
        if spec.get('format') != 'v9-training-phases-v14':
            raise ValueError('Unsupported phase manifest')
        def resolve(value):
            p = Path(value)
            return p if p.is_absolute() else path.parent/p
        self.source_vace_path, self.source_hmr_path = map(resolve,
            (spec['source_vace'],spec['source_hmr']))
        base, rgb = load_pair(self.source_vace_path,self.source_hmr_path,'train')
        if rgb['signature'].get('anchor_revision') != 'raw-bbox-v11':
            raise ValueError('Phase reuse requires the unchanged raw-bbox-v11 RGB source')
        if base['signature'].get('development_only'):
            raise ValueError('Source must cover the full training split')
        expected = set(rgb['frame_uids'])
        self.datasets, self.phases, self.paths = [], [], []
        self.development_only = False
        crop_modes = set()
        for entry in spec['phases']:
            p = resolve(entry['vace'])
            vace = load(p)
            validate_vace_source(vace)
            sig = vace['signature']
            phase = sig.get('window_phase')
            if (sig.get('preprocessing_revision') != 'phase-crop-v14' or
                sig.get('split') != 'train' or phase != entry['phase'] or phase not in (0,2,4,6)):
                raise ValueError('Incorrect phase revision, offset or split')
            for key in ('checkpoint','encoder_protocol','feature_layers','image_size'):
                if sig.get(key) != base['signature'].get(key):
                    raise ValueError(f'Phase encoder mismatch: {key}')
            crop_modes.add(sig.get('crop_mode'))
            valid = vace['target_valid_mask']
            if (vace['frame_owner_mask'] & ~valid).any():
                raise ValueError('Invalid frames cannot own supervision')
            for width, name in ((2,'velocity_owner_mask'),(3,'acceleration_owner_mask')):
                allowed = valid.unfold(1,width,1).all(-1)
                if (vace[name] & ~allowed).any():
                    raise ValueError('Invalid temporal primitive supervision')
            owned = []
            for m,mask in zip(vace['metadata'],vace['frame_owner_mask']):
                owned.extend(uid for uid,keep in zip(m['target_frame_uids'],mask) if keep)
            if len(owned) != len(set(owned)):
                raise ValueError('Duplicate supervised frames inside a phase')
            development = bool(sig.get('development_only'))
            if not development and set(owned) != expected:
                raise ValueError('Full phases must own exactly the verified training frames')
            if not set(owned) <= expected:
                raise ValueError('Unexpected frames in phase')
            from divibody_training.cache_vace_phase_v14 import validate_v5_payload
            validate_v5_payload(vace,feature_only=False)
            self.development_only |= development
            self.datasets.append(V10Dataset(vace,remap_anchor(rgb,vace)))
            self.phases.append(phase)
            self.paths.append(str(p.resolve()))
        if sorted(self.phases) not in ([0],[0,2,4,6]) or len(set(self.phases)) != len(self.phases):
            raise ValueError('Provide phase 0 control or exactly phases 0,2,4,6')
        if len(crop_modes) != 1 or not crop_modes <= {'legacy','square_pad'}:
            raise ValueError('All phases must use the same supported crop mode')
        self.crop_mode = next(iter(crop_modes))
        self.seed = seed
        self.budget = min(map(len,self.datasets))
        if max_windows:
            self.budget = min(self.budget,max_windows)
            self.development_only = True
        if self.budget < 1:
            raise ValueError('Empty phase bank')

    def epoch(self, epoch):
        # Deterministic order independent of manifest ordering.
        phase = sorted(self.phases)[epoch % len(self.phases)]
        dataset = self.datasets[self.phases.index(phase)]
        generator = torch.Generator().manual_seed(self.seed+epoch)
        indices = torch.randperm(len(dataset),generator=generator)[:self.budget].tolist()
        return Subset(dataset,indices), phase
