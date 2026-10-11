"""Evaluate the unchanged HMR2a anchor on COCO2017 val person crops."""
import argparse,json,time
from pathlib import Path
import numpy as np, torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from divibody_training.evaluate_unified_hmr2_divibody_3dpw import load_hmr_model
COCO_FROM_OP=(0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11)
class CocoPersons(Dataset):
 def __init__(self,root,image_size=256,scale=1.25):
  self.root=Path(root); self.size=image_size; self.scale=scale
  x=json.loads((self.root/'annotations/person_keypoints_val2017.json').read_text())
  images={z['id']:z for z in x['images']}; self.samples=[]
  for a in x['annotations']:
   k=np.asarray(a.get('keypoints',[]),np.float32)
   b=np.asarray(a.get('bbox',[]),np.float32)
   if a.get('iscrowd',0) or k.size!=51 or b.size!=4 or b[2]<=1 or b[3]<=1: continue
   k=k.reshape(17,3)
   if not (k[:,2]>0).any(): continue
   im=images[a['image_id']]; path=self.root/'val2017'/im['file_name']
   if path.is_file(): self.samples.append((path,b,k,a['id']))
 def __len__(self): return len(self.samples)
 def __getitem__(self,i):
  path,b,k,aid=self.samples[i]; x,y,w,h=map(float,b); side=max(w,h)*self.scale; cx=x+w/2; cy=y+h/2
  box=(cx-side/2,cy-side/2,cx+side/2,cy+side/2)
  with Image.open(path) as im: rgb=np.asarray(im.convert('RGB').crop(box).resize((self.size,self.size),Image.Resampling.BILINEAR),np.float32)/255
  q=k.copy(); q[:,:2]=(q[:,:2]-np.array(box[:2],np.float32))*self.size/side
  return {'image':torch.from_numpy(rgb.copy()).permute(2,0,1),'keypoints':torch.from_numpy(q),'annotation_id':aid}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--coco-root',type=Path,required=True); p.add_argument('--hmr-source-root',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--model-config',type=Path,required=True); p.add_argument('--hmr-data-root',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--batch-size',type=int,default=128); p.add_argument('--num-workers',type=int,default=4); p.add_argument('--device',default='cuda:0'); a=p.parse_args(); a.hmr_model_config=a.model_config
 d=CocoPersons(a.coco_root); loader=DataLoader(d,batch_size=a.batch_size,num_workers=a.num_workers,pin_memory=True,persistent_workers=a.num_workers>0); dev=torch.device(a.device); m=load_hmr_model(a,a.checkpoint).to(dev).eval(); mean=torch.tensor((.485,.456,.406),device=dev).view(1,3,1,1); std=torch.tensor((.229,.224,.225),device=dev).view(1,3,1,1)
 c05=c10=n=0; es=0.; start=time.monotonic()
 with torch.inference_mode():
  for bi,b in enumerate(loader):
   images=b['image'].to(dev,non_blocking=True)
   with torch.autocast(device_type='cuda',dtype=torch.bfloat16): out=m({'img':(images-mean)/std})
   op=out['pred_keypoints_2d'][:,:25].float(); pred=op[:,COCO_FROM_OP]; pred=(pred+.5)*256; gt=b['keypoints'].to(dev); vis=gt[:,:,2]>0; err=torch.linalg.vector_norm(pred-gt[:,:,:2],dim=-1)/(256/1.25); v=err[vis]; c05+=int((v<=.05).sum()); c10+=int((v<=.1).sum()); n+=v.numel(); es+=float(v.double().sum())
   if (bi+1)%50==0: print(json.dumps({'batches':bi+1,'people':min(len(d),(bi+1)*a.batch_size),'seconds':time.monotonic()-start}),flush=True)
 r={'status':'HMR2A_FULL_COCO2017_VAL_OK','model':'HMR2a anchor','metrics':{'PCKn@0.05':c05/n,'PCKn@0.10':c10/n,'mean_normalized_joint_error':es/n},'coverage':{'people':len(d),'visible_keypoints':n},'normalizer':'max GT bbox side; GT person crop scale=1.25','note':'Anchor baseline; v26 COCO residual requires COCO VACE/HMR cache'}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
if __name__=='__main__': main()
