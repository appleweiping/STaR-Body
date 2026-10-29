"""Audited, prediction-only offline track prototypes; targets stay in losses."""
import hashlib,json,re
import torch
from torch.utils.data import Dataset
FORMAT='target60-v30-track-prototypes-v34'
CONTEXT_KEYS=('track_shape_base','track_shape_features')
UID=re.compile(r'^([^/]+)/person_(\d+)/frame_(\d+)$')
def uid_hash(uids):return hashlib.sha256(('\n'.join(sorted(uids))+'\n').encode()).hexdigest()
class TrackPrototypeDataset(Dataset):
    def __init__(self,dataset,prototype_path=None,init_sha256=None,prototype_sha256=None):
        self.dataset=dataset;self.vace=dataset.vace;self.sidecar=getattr(dataset,'sidecar',None)
        if self.sidecar is None:raise ValueError('HMR sidecar required')
        self.rows=None
        identity=uid_hash(self.sidecar['frame_uids'])
        if prototype_path is not None:
            from pathlib import Path
            path=Path(prototype_path);digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if prototype_sha256 and digest!=prototype_sha256:raise ValueError('Prototype content SHA mismatch')
            pack=torch.load(path,map_location='cpu',weights_only=False)
            if pack.get('format')!=FORMAT or pack.get('split')!=self.vace['signature']['split']:raise ValueError('Prototype format/split mismatch')
            if pack.get('source_v30_sha256')!=init_sha256 or pack.get('frame_uids_sha256')!=identity:raise ValueError('Prototype init/UID mismatch')
            if pack.get('ground_truth_inputs') is not False:raise ValueError('Prototype must declare prediction-only inputs')
            if pack.get('vace_signature')!=self.vace['signature'] or pack.get('hmr_signature')!=self.sidecar['signature']:raise ValueError('Prototype feature signatures mismatch')
            self.rows=pack['tracks']
            expected={uid.rsplit('/frame_',1)[0] for uid in self.sidecar['frame_uids']}
            if set(self.rows)!=expected:raise ValueError('Prototype track identities incomplete')
            for value in self.rows.values():
                if value['betas'].shape!=(10,) or value['features'].shape!=(self.sidecar['rgb_tokens'].shape[-1],):raise ValueError('Prototype tensor shape mismatch')
                if not all(torch.isfinite(value[k]).all() for k in ('betas','features')):raise ValueError('Nonfinite prototype')
        self.context_audit=dict(format='target60-track-context-v34',split=self.vace['signature']['split'],
            frame_uids_sha256=identity,prototype_sha256=digest if self.rows is not None else None,
            source_v30_sha256=init_sha256 if self.rows is not None else None,label_free=True,scope='offline')
    def __len__(self):return len(self.dataset)
    def __getitem__(self,index):
        sample=self.dataset[index]
        if self.rows is not None:
            record=self.vace['metadata'][index]
            key=f"{record['sequence']}/person_{int(record['person_id'])}"
            prototype=self.rows[key]
            sample.update(track_shape_base=prototype['betas'],track_shape_features=prototype['features'])
        return sample
