"""Protocol audit for the full public PoseTrack18 HMR2a baseline.

This additive v2 keeps v1 untouched, caches every prediction, and reports
multiple explicitly named normalizers and joint subsets. None is labelled as
the unpublished RAM protocol.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from divibody_training.evaluate_unified_hmr2_divibody_3dpw import load_hmr_model
from divibody_training.ram_ready_adapter_v1 import PoseTrack18PersonDataset

JOINTS=("nose","head_bottom","head_top","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle")
PT_FROM_OP={0:0,1:1,3:18,4:17,5:5,6:2,7:6,8:3,9:7,10:4,11:12,12:9,13:13,14:10,15:14,16:11}
SUBSETS={"all17":tuple(range(17)),"body15_no_head_segment":(0,3,4,*range(5,17)),"body12_no_face_head":tuple(range(5,17))}

def pose_track_from_openpose(openpose):
    result=openpose.new_zeros((openpose.shape[0],17,2))
    for target,source in PT_FROM_OP.items(): result[:,target]=openpose[:,source]
    result[:,2]=2.0*openpose[:,0]-openpose[:,1]
    return result

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def summarize(distance,visible,denominator):
    valid=visible & torch.isfinite(denominator) & (denominator>1e-8)
    values=(distance/denominator).masked_select(valid)
    return {"PCKn@0.05":float((values<=.05).double().mean()),"PCKn@0.10":float((values<=.10).double().mean()),"mean_normalized_joint_error":float(values.double().mean()),"visible_keypoints":int(values.numel())}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ready-root',type=Path,required=True); p.add_argument('--hmr-source-root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--model-config',type=Path,required=True)
    p.add_argument('--hmr-data-root',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prediction-cache',type=Path,required=True); p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--num-workers',type=int,default=8); p.add_argument('--device',default='cuda:0'); p.add_argument('--amp-dtype',default='bf16')
    args=p.parse_args(); args.hmr_model_config=args.model_config
    if args.output.exists() or args.prediction_cache.exists(): raise FileExistsError('Use new output paths')
    device=torch.device(args.device)
    dataset=PoseTrack18PersonDataset(args.ready_root,split='val',image_size=256,bbox_scale=1.25,min_keypoints=1)
    loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=True,persistent_workers=args.num_workers>0)
    model=load_hmr_model(args,args.checkpoint).to(device).eval()
    mean=torch.tensor((.485,.456,.406),device=device).view(1,3,1,1); std=torch.tensor((.229,.224,.225),device=device).view(1,3,1,1)
    chunks={k:[] for k in ('pred','gt','visible','bbox','crop_side','annotation_id')}; started=time.monotonic(); people=0
    with torch.inference_mode():
        for bi,batch in enumerate(loader):
            images=batch['image'].to(device,non_blocking=True)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda' and args.amp_dtype=='bf16'):
                output=model({'img':(images-mean)/std})
            pred=(pose_track_from_openpose(output['pred_keypoints_2d'][:,:25].float())+.5)*256.
            gt=batch['keypoints'][...,:2].float(); visible=batch['keypoints'][...,2]>0
            chunks['pred'].append(pred.cpu()); chunks['gt'].append(gt); chunks['visible'].append(visible)
            chunks['bbox'].append(batch['bbox_xywh'].float()); chunks['crop_side'].append(batch['crop_side'].float())
            chunks['annotation_id'].append(batch['annotation_id'].long()); people+=len(images)
            if (bi+1)%50==0 or people==len(dataset): print(json.dumps({'processed':people,'total':len(dataset),'seconds':time.monotonic()-started}),flush=True)
    data={k:torch.cat(v) for k,v in chunks.items()}
    if people!=len(dataset) or int(data['visible'].sum())!=dataset.audit()['visible_keypoints']: raise RuntimeError('Incomplete validation coverage')
    args.prediction_cache.parent.mkdir(parents=True,exist_ok=True)
    cache={'format':'hmr2a-posetrack18-public-val-predictions-v2','joint_names':JOINTS,'mapping':PT_FROM_OP,'checkpoint_sha256':sha256(args.checkpoint),**data}
    torch.save(cache,args.prediction_cache)
    distance=torch.linalg.vector_norm(data['pred']-data['gt'],dim=-1)
    bbox_max=data['bbox'][:,2:].amax(-1); bbox_diag=torch.linalg.vector_norm(data['bbox'][:,2:],dim=-1)
    scale_to_original=data['crop_side']/256.
    denominators={
      'gt_bbox_max_side':(bbox_max/scale_to_original)[:,None].expand_as(distance),
      'gt_bbox_diagonal':(bbox_diag/scale_to_original)[:,None].expand_as(distance),
      'crop_side':torch.full_like(distance,256.),
    }
    shoulder=(data['gt'][:,5]+data['gt'][:,6])*.5; hip=(data['gt'][:,11]+data['gt'][:,12])*.5
    torso=torch.linalg.vector_norm(shoulder-hip,dim=-1); torso_ok=data['visible'][:,5]&data['visible'][:,6]&data['visible'][:,11]&data['visible'][:,12]
    denominators['gt_torso_length']=torso[:,None].expand_as(distance)
    head=torch.linalg.vector_norm(data['gt'][:,1]-data['gt'][:,2],dim=-1); head_ok=data['visible'][:,1]&data['visible'][:,2]
    denominators['gt_head_segment']=head[:,None].expand_as(distance)
    results={}
    for normalizer,denominator in denominators.items():
        extra=torch.ones_like(data['visible'])
        if normalizer=='gt_torso_length': extra=torso_ok[:,None].expand_as(extra)
        if normalizer=='gt_head_segment': extra=head_ok[:,None].expand_as(extra)
        results[normalizer]={}
        for subset,indices in SUBSETS.items():
            select=torch.zeros(17,dtype=torch.bool); select[list(indices)]=True
            results[normalizer][subset]=summarize(distance,data['visible']&extra&select[None],denominator)
    per_joint={}
    denominator=denominators['gt_bbox_max_side']
    for j,name in enumerate(JOINTS): per_joint[name]=summarize(distance[:,j:j+1],data['visible'][:,j:j+1],denominator[:,j:j+1])
    payload={'status':'HMR2A_POSETRACK18_PROTOCOL_AUDIT_V2_OK','split':'public PoseTrack18 val','people':people,'prediction_cache':str(args.prediction_cache.resolve()),'prediction_cache_sha256':sha256(args.prediction_cache),'source_sha256':sha256(Path(__file__)),'checkpoint_sha256':sha256(args.checkpoint),'results':results,'per_joint_gt_bbox_max_side':per_joint,'notes':['All runs use GT person boxes enlarged by 1.25 before a 256x256 crop.','Normalizer names are literal diagnostics; none is asserted to be the unpublished RAM protocol.','No 3DPW test data is used.']}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,indent=2),encoding='utf-8'); print(json.dumps(payload,indent=2),flush=True)
if __name__=='__main__': main()
