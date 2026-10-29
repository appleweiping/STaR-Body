"""Label-free same-track context and immutable offline shape prototypes.

    A wrapper around either the train V10Dataset or evaluator EvaluationWindows.
    Only sidecar predictions and UID/time metadata are read for new model inputs.
    Ground-truth fields stay in the wrapped sample for losses/metrics only.
"""
from bisect import bisect_left,bisect_right
from collections import defaultdict
import hashlib,json,re
import torch
from torch.utils.data import Dataset

UID=re.compile(r'^([^/]+)/person_(\d+)/frame_(\d+)$')
CONTEXT_KEYS=('memory_rgb_tokens','memory_rotmat','memory_betas','memory_camera',
              'memory_source_frame_ids','memory_annotation_ids','memory_valid',
              'target_annotation_ids','track_shape_base','track_shape_features')


class TrackContextDataset(Dataset):
    def __init__(self,dataset,horizon=32,shape_frames=8,mode='offline',enabled=True):
        if not 1<=horizon<=64 or not 1<=shape_frames<=16 or mode not in ('offline','past_memory'):
            raise ValueError('Invalid context policy')
        self.dataset=dataset;self.vace=dataset.vace
        self.sidecar=getattr(dataset,'sidecar',getattr(dataset,'hmr',None))
        if self.sidecar is None:raise ValueError('HMR sidecar required')
        if self.vace['signature'].get('split')!=self.sidecar['signature'].get('split'):
            raise ValueError('Context cannot cross splits')
        self.horizon=horizon;self.shape_frames=shape_frames;self.mode=mode;self.enabled=enabled
        self.tracks=defaultdict(list);self.frame_meta=[];self.source_ids={}
        lookup=self.sidecar['window_frame_indices']
        for row,record in enumerate(self.vace['metadata']):
            source=record.get('target_source_frame_ids')
            if source is None:source=record['target_frame_ids']
            for col,index in enumerate(lookup[row].tolist()):
                if not bool(self.vace['target_valid_mask'][row,col]):continue
                value=int(source[col])
                if index in self.source_ids and self.source_ids[index]!=value:
                    raise ValueError('HMR UID has conflicting source times')
                self.source_ids[index]=value
        for index,uid in enumerate(self.sidecar['frame_uids']):
            match=UID.fullmatch(uid)
            if match is None or index not in self.source_ids:raise ValueError('Unmapped HMR UID: '+uid)
            seq,person,frame=match.groups();key=(seq,int(person));frame=int(frame)
            self.frame_meta.append((key,frame));self.tracks[key].append((frame,index))
        for key,rows in self.tracks.items():
            rows.sort()
            if len({x[0] for x in rows})!=len(rows):raise ValueError('Duplicate same-track UID')
        self.prototypes={}
        identity=dict(horizon=horizon,shape_frames=shape_frames,mode=mode,enabled=enabled,
            split=self.vace['signature']['split'],frame_uids=self.sidecar['frame_uids'],
            source_ids=sorted(self.source_ids.items()),shape='whole-track robust predicted-beta prototype; offline',
            boundary='contiguous annotation segments; same-track; no GT')
        self.context_audit=dict(format='star-track-context-v33',split=identity['split'],
            identity_sha256=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            horizon=horizon,shape_frames=shape_frames,mode=mode,tracks=len(self.tracks),
            label_free=True,offline_shape_prototypes=True,online_model=False)

    def __len__(self):return len(self.dataset)

    def _prototype(self,key):
        if key not in self.prototypes:
            indices=torch.tensor([i for _,i in self.tracks[key]],dtype=torch.long)
            betas=self.sidecar['base_betas'][indices].float()
            cameras=self.sidecar['base_camera'][indices].float()
            good=torch.isfinite(betas).all(-1)&torch.isfinite(cameras).all(-1)&(cameras[:,0]>0)
            if not bool(good.any()):raise ValueError('Track has no finite predicted shape')
            selected=indices[good];values=betas[good]
            median=values.median(0).values
            distance=(values-median).square().mean(-1).sqrt()
            weight=1/(1+distance)
            base=(values*weight[:,None]).sum(0)/weight.sum()
            # Diverse deterministic reliable frames, independent of requested window.
            reliable=distance.argsort()[:max(1,(len(distance)+1)//2)]
            reliable=selected[reliable].sort().values
            positions=torch.linspace(0,len(reliable)-1,min(self.shape_frames,len(reliable))).round().long().unique()
            anchors=reliable[positions]
            features=torch.stack([self.sidecar['rgb_tokens'][i].float().mean(0) for i in anchors]).mean(0)
            self.prototypes[key]=(base,features)
        return self.prototypes[key]

    def __getitem__(self,index):
        sample=self.dataset[index]
        if not self.enabled:return sample
        record=self.vace['metadata'][index]
        indices=self.sidecar['window_frame_indices'][index]
        valid=self.vace['target_valid_mask'][index]
        real=indices[valid].tolist()
        keys={self.frame_meta[i][0] for i in real}
        if len(keys)!=1:raise ValueError('Window crosses track identity')
        key=next(iter(keys));lo=min(self.frame_meta[i][1] for i in real);hi=max(self.frame_meta[i][1] for i in real)
        rows=self.tracks[key];frames=[x[0] for x in rows]
        left=bisect_left(frames,lo);right=bisect_right(frames,hi)
        # Do not retrieve through annotation gaps, even within one person.
        seg_left=left;seg_right=right
        while seg_left>0 and frames[seg_left]-frames[seg_left-1]==1:seg_left-=1
        while seg_right<len(rows) and frames[seg_right]-frames[seg_right-1]==1:seg_right+=1
        start=max(seg_left,left-self.horizon)
        end=min(seg_right,right+self.horizon) if self.mode=='offline' else right
        center=(lo+hi)/2
        candidates=rows[start:end]
        chosen=sorted(sorted(candidates,key=lambda x:(abs(x[0]-center),x[0]))[:self.horizon])
        if not chosen:raise ValueError('No context frames')
        chosen_indices=[i for _,i in chosen]
        mask=torch.arange(self.horizon)<len(chosen_indices)
        chosen_indices+= [chosen_indices[-1]]*(self.horizon-len(chosen_indices))
        midx=torch.tensor(chosen_indices,dtype=torch.long)
        base,features=self._prototype(key)
        sample.update(memory_rgb_tokens=self.sidecar['rgb_tokens'][midx],
            memory_rotmat=self.sidecar['base_rotmat'][midx],memory_betas=self.sidecar['base_betas'][midx],
            memory_camera=self.sidecar['base_camera'][midx],memory_source_frame_ids=torch.tensor([self.source_ids[i] for i in chosen_indices]),
            memory_annotation_ids=torch.tensor([self.frame_meta[i][1] for i in chosen_indices]),memory_valid=mask,
            target_annotation_ids=torch.tensor(record['target_frame_ids'],dtype=torch.long),
            track_shape_base=base,track_shape_features=features)
        return sample
