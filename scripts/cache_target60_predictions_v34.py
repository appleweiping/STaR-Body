"""Cache immutable native V30 predictions and label-free offline track prototypes."""
import argparse,hashlib,json,random
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from divibody_training.target60_track_dataset_v34 import FORMAT,uid_hash
from divibody_training.target60_runtime_v34 import forward_batch
from models.human_decoder_spatiotemporal_v30 import model_from_payload,FORMAT as V30_FORMAT
from divibody_training.evaluate_standard_3dpw import create_smpl_models
class PredictionWindows(Dataset):
    def __init__(self,vace,sidecar):self.vace=vace;self.sidecar=sidecar
    def __len__(self):return len(self.vace['metadata'])
    def __getitem__(self,index):
        ids=self.sidecar['window_frame_indices'][index];record=self.vace['metadata'][index]
        result={k:self.sidecar[k][ids] for k in ('rgb_tokens','base_rotmat','base_betas','base_camera')}
        result.update(vace_features=self.vace['features'][index],target_valid=self.vace['target_valid_mask'][index],
            frame_owner=self.vace['frame_owner_mask'][index],
            source_frame_ids=torch.tensor(record['target_source_frame_ids'] if record.get('target_source_frame_ids') is not None else record['target_frame_ids'],dtype=torch.long))
        return result
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('checkpoint','vace-cache','hmr-sidecar','smpl-model-root','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--split',choices=('train','validation','test'),required=True)
    p.add_argument('--coverage-manifest',type=Path)
    p.add_argument('--batch-size',type=int,default=1);p.add_argument('--device',default='cuda:0')
    p.add_argument('--max-cuda-memory-gib',type=float,default=3.5);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(4);device=torch.device(a.device)
    random.seed(2026);np.random.seed(2026);torch.manual_seed(2026)
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(min(1.,a.max_cuda_memory_gib*1024**3/torch.cuda.get_device_properties(device).total_memory),device)
    load=lambda x:torch.load(x,map_location='cpu',weights_only=False,mmap=True)
    vace,hmr=load(a.vace_cache),load(a.hmr_sidecar)
    # Reuse independent exact coverage/signature guards, with no GT tensor access.
    import importlib.util
    evaluator=Path(__file__).with_name('evaluate_target60_v34.py')
    spec=importlib.util.spec_from_file_location('target60_cache_audit',evaluator);audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    audit.verify_cache_provenance(vace,hmr,a.vace_cache,a.split,False)
    coverage,owned=audit.audit_coverage(vace,hmr,a.split,a.coverage_manifest,False)
    payload=load(a.checkpoint)
    if payload.get('format')!=V30_FORMAT or not payload.get('step',0)>0 or payload.get('development_only',True):raise ValueError('Actual trained native V30 required')
    model=model_from_payload(payload).to(device).eval()
    models=create_smpl_models(a.smpl_model_root,device)
    for body in models.values():body.requires_grad_(False).eval()
    from hmr2.models.smpl_wrapper import SMPL
    official=SMPL(model_path=str(a.smpl_model_root),gender='neutral',num_betas=10,
        joint_regressor_extra=str(a.smpl_model_root.parent/'SMPL_to_J19.pkl')).to(device).eval().requires_grad_(False)
    model.bind_geometry(models[2]).bind_landmarks(official)
    dataset=PredictionWindows(vace,hmr);lookup={u:i for i,u in enumerate(hmr['frame_uids'])}
    betas=torch.empty(len(lookup),10);seen=set();cursor=0
    with torch.no_grad():
        for batch in DataLoader(dataset,batch_size=a.batch_size,shuffle=False,num_workers=0):
            out=forward_batch(model,batch,device,'none')
            for row in range(len(batch['target_valid'])):
                record=vace['metadata'][cursor+row]
                for col,uid in enumerate(record['target_frame_uids']):
                    if bool(batch['frame_owner'][row,col] and batch['target_valid'][row,col]):
                        if uid in seen:raise ValueError('Duplicate predicted frame owner')
                        seen.add(uid);betas[lookup[uid]]=out['betas'][row,col].cpu()
            cursor+=len(batch['target_valid'])
            if cursor%100<a.batch_size:print(json.dumps(dict(event='prototype_prediction',windows=cursor,frames=len(seen))),flush=True)
    if seen!=owned or len(seen)!=len(lookup):raise ValueError('Prediction prototype coverage incomplete')
    tracks=defaultdict(list)
    for uid,i in lookup.items():tracks[uid.rsplit('/frame_',1)[0]].append(i)
    result={}
    for key,indices in tracks.items():
        idx=torch.tensor(indices);values=betas[idx].float()
        if not torch.isfinite(values).all():raise ValueError('Nonfinite V30 beta')
        median=values.median(0).values;distance=(values-median).square().mean(-1).sqrt()
        weights=1/(1+distance);base=(values*weights[:,None]).sum(0)/weights.sum()
        reliable=distance.argsort()[:max(1,(len(distance)+1)//2)]
        reliable=idx[reliable].sort().values
        positions=torch.linspace(0,len(reliable)-1,min(8,len(reliable))).round().long().unique()
        features=torch.stack([hmr['rgb_tokens'][int(i)].float().mean(0) for i in reliable[positions]]).mean(0)
        result[key]=dict(betas=base,features=features)
    pack=dict(format=FORMAT,split=a.split,source_v30_sha256=sha(a.checkpoint),
        frame_uids_sha256=uid_hash(hmr['frame_uids']),ground_truth_inputs=False,scope='offline',
        vace_signature=vace['signature'],hmr_signature=hmr['signature'],coverage=coverage,tracks=dict(result),
        generator_sha256=sha(Path(__file__)),prediction_precision='FP32',prototype='robust full-track predicted V30 beta and deterministic RGB anchors')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('xb') as f:torch.save(pack,f)
    print(json.dumps(dict(status='complete',output=str(a.output),sha256=sha(a.output),tracks=len(result),frames=len(seen))),flush=True)
if __name__=='__main__':main()
