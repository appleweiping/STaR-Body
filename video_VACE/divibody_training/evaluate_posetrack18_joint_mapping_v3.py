"""Audit PoseTrack18 joint mapping from cached HMR2a SMPL parameters.

Additive evaluation only. It never reads 3DPW test data and does not modify
prior evaluators, caches, checkpoints, or model files.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from divibody_training.ram_pose_stream_v10 import ShuffledPoseDatasetV10

JOINTS=("nose","head_bottom","head_top","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle")
CURRENT=(0,1,-1,18,17,5,2,6,3,7,4,12,9,13,10,14,11)
CURRENT_EXTRA_HEAD=(0,37,38,18,17,5,2,6,3,7,4,12,9,13,10,14,11)
LSP_EXTRA=(0,37,38,18,17,34,33,35,32,36,31,28,27,29,26,30,25)
MAPPINGS={"openpose25_synthetic_head":CURRENT,"openpose_body_lsp_head":CURRENT_EXTRA_HEAD,"lsp_extra_body_head":LSP_EXTRA}

def sha256(path):
 h=hashlib.sha256();
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--vace-dir',type=Path,required=True); p.add_argument('--hmr-dir',type=Path,required=True)
 p.add_argument('--smpl-model-root',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
 p.add_argument('--batch-size',type=int,default=32); p.add_argument('--device',default='cuda:0')
 a=p.parse_args()
 if a.output.exists(): raise FileExistsError(a.output)
 ds=ShuffledPoseDatasetV10(a.vace_dir,a.hmr_dir,seed=2026,max_windows=0)
 if ds.split!='val': raise ValueError('Public PoseTrack val required')
 loader=DataLoader(ds,batch_size=a.batch_size,num_workers=0)
 sysroot=str(a.smpl_model_root.parent.parent.parent/'video_VACE')
 from hmr2.models.smpl_wrapper import SMPL
 smpl=SMPL(model_path=str(a.smpl_model_root),gender='neutral',num_betas=10,joint_regressor_extra=str(a.smpl_model_root.parent/'SMPL_to_J19.pkl')).to(a.device).requires_grad_(False)
 totals={name:{'correct05':torch.zeros(17,dtype=torch.int64),'correct10':torch.zeros(17,dtype=torch.int64),'error':torch.zeros(17,dtype=torch.float64),'count':torch.zeros(17,dtype=torch.int64)} for name in MAPPINGS}
 owned_frames=0
 with torch.inference_mode():
  for step,b in enumerate(loader):
   B,T=b['base_rotmat'].shape[:2]
   rot=b['base_rotmat'].to(a.device).reshape(-1,24,3,3); betas=b['base_betas'].to(a.device).reshape(-1,10)
   joints=smpl(global_orient=rot[:,:1],body_pose=rot[:,1:],betas=betas,pose2rot=False).joints
   cam=b['base_camera'].to(a.device).reshape(-1,3)
   trans=torch.stack((cam[:,1],cam[:,2],10000./(256.*cam[:,0].clamp_min(1e-6))),-1)
   points=joints+trans[:,None]; uv=points[...,:2]/points[...,2:].clamp_min(1e-6)*(5000./256.)
   target=b['keypoints'][...,:2].to(a.device).float()/96.-.5
   visible=(b['keypoints'][...,2].to(a.device)>0)&b['target_valid'].to(a.device)[...,None]&b['frame_owner'].to(a.device)[...,None]
   owned_frames+=int((b['target_valid'].to(a.device)&b['frame_owner'].to(a.device)).sum())
   for name,mapping in MAPPINGS.items():
    pred=uv.new_zeros((B*T,17,2))
    for j,source in enumerate(mapping):
     pred[:,j]=2*uv[:,0]-uv[:,1] if source<0 else uv[:,source]
    pred=pred.reshape(B,T,17,2)
    values=torch.linalg.vector_norm(pred-target,dim=-1)/.8
    for j in range(17):
     x=values[...,j].masked_select(visible[...,j]).cpu()
     totals[name]['correct05'][j]+=int((x<=.05).sum()); totals[name]['correct10'][j]+=int((x<=.10).sum())
     totals[name]['error'][j]+=x.double().sum(); totals[name]['count'][j]+=x.numel()
   if (step+1)%100==0: print(json.dumps({'windows_processed':min((step+1)*a.batch_size,len(ds)),'windows_total':len(ds)}),flush=True)
 results={}
 for name,t in totals.items():
  count=int(t['count'].sum()); c05=int(t['correct05'].sum()); c10=int(t['correct10'].sum())
  per={}
  for j,joint in enumerate(JOINTS):
   n=int(t['count'][j]); per[joint]={'count':n,'PCKn@0.05':None if not n else int(t['correct05'][j])/n,'PCKn@0.10':None if not n else int(t['correct10'][j])/n,'mean_error':None if not n else float(t['error'][j])/n}
  results[name]={'all17':{'visible_keypoints':count,'PCKn@0.05':c05/count,'PCKn@0.10':c10/count,'mean_error':float(t['error'].sum())/count},'per_joint':per}
 payload={'status':'POSETRACK18_JOINT_MAPPING_AUDIT_V3_OK','split':'public PoseTrack18 val development','windows':len(ds),'owned_frames':owned_frames,'mapping_indices':{k:list(v) for k,v in MAPPINGS.items()},'results':results,'source_sha256':sha256(Path(__file__)),'test_data_used':False,'claim':'diagnostic only; no mapping is asserted to be RAM protocol'}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(payload,indent=2),encoding='utf-8'); print(json.dumps(payload,indent=2),flush=True)
if __name__=='__main__': main()
