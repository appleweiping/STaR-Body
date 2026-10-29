"""Train pose completion on explicitly declared train-only canonical rotations.

Supports packs exported from 3DPW train or separately audited AMASS conversions.
Raw SMPL-H/MHR rotations cannot be silently treated as canonical SMPL targets.
"""
import argparse,hashlib,json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'video_VACE'),str(ROOT/'video_VACE/divibody_training')]


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
 return h.hexdigest()


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--steps',type=int,default=50000);p.add_argument('--batch-size',type=int,default=16)
 p.add_argument('--frames',type=int,default=32);p.add_argument('--width',type=int,default=256)
 p.add_argument('--heads',type=int,default=8);p.add_argument('--depth',type=int,default=4)
 p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--seed',type=int,default=2026)
 p.add_argument('--device',default='cuda:0');p.add_argument('--resume',action='store_true')
 p.add_argument('--max-cuda-memory-gib',type=float,default=8.)
 a=p.parse_args()
 import numpy as np,torch
 from models.star_memory_modules_v33 import PoseCompletionPrior
 from models.human_decoder_star_memory_v33 import rotation_6d
 if min(a.steps,a.batch_size,a.width,a.heads,a.depth)<=0 or not 8<=a.frames<=64 or a.width%4 or a.width%a.heads or a.lr<=0 or a.max_cuda_memory_gib<=0:
  raise ValueError('Invalid motion pretraining configuration')
 manifest=json.loads(a.manifest.read_text(encoding='utf-8'))
 if manifest.get('format')!='star-motion-rotations-v33' or manifest.get('split')!='train' or manifest.get('rotation_convention')!='SMPL camera-root and local-body rotmat':
  raise ValueError('Audited train-only canonical SMPL rotation manifest required')
 files=[]
 for row in manifest['files']:
  path=(a.manifest.parent/row['file']).resolve()
  if not path.is_relative_to(a.manifest.parent.resolve()) or sha(path)!=row['sha256']:
   raise ValueError('Motion asset path/hash mismatch')
  if row['frames']>=8:files.append((path,row['frames']))
 if not files:raise ValueError('No eligible motion segments')
 if not a.resume and a.output.exists() and any(a.output.iterdir()):raise FileExistsError('Use new output directory')
 a.output.mkdir(parents=True,exist_ok=True)
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 device=torch.device(a.device)
 if device.type=='cuda':
  cap=torch.cuda.get_device_properties(device).total_memory
  torch.cuda.set_per_process_memory_fraction(min(1.,a.max_cuda_memory_gib*1024**3/cap),device)
 cfg=dict(width=a.width,heads=a.heads,depth=a.depth)
 model=PoseCompletionPrior(**cfg).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
 rng=random.Random(a.seed);step=0;cache={}
 invariant=dict(config=cfg,manifest_sha256=sha(a.manifest),steps=a.steps,frames=a.frames,batch_size=a.batch_size,seed=a.seed,
  lr=a.lr,device=a.device,torch=torch.__version__,numpy=np.__version__,
  source={str(f.relative_to(ROOT)):sha(f) for f in (Path(__file__),ROOT/'video_VACE/models/star_memory_modules_v33.py',ROOT/'video_VACE/models/human_decoder_star_memory_v33.py')})
 if a.resume:
  state=torch.load(a.output/'latest.pt',map_location='cpu',weights_only=False)
  if state.get('invariants')!=invariant:raise ValueError('Motion resume config/assets/code differ')
  model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
  step=state['step'];rng.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng'])
  if device.type=='cuda':torch.cuda.set_rng_state_all(state['cuda_rng'])
  log=a.output/'metrics.jsonl'
  if log.exists():
   rows=[line for line in log.read_text().splitlines() if json.loads(line)['step']<=step]
   log.rename(a.output/('metrics.before_resume_'+str(__import__('time').time_ns())+'.jsonl'))
   log.write_text(''.join(line+'\n' for line in rows))
 def save():
  state=dict(format='star-pose-completion-pretrain-v33',split='train',config=cfg,model=model.state_dict(),optimizer=optimizer.state_dict(),
   step=step,invariants=invariant,python_rng=rng.getstate(),torch_rng=torch.get_rng_state(),
   cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None,not_a_reconstruction_benchmark=True)
  tmp=a.output/'latest.pt.tmp';torch.save(state,tmp);tmp.replace(a.output/'latest.pt')
 def sample():
  path,n=rng.choice(files)
  if path not in cache:
   cache.clear()
   with np.load(path,allow_pickle=False) as data:
    rot=torch.from_numpy(data['rotmat'].copy()).float();ids=torch.from_numpy(data['frame_ids'].copy()).long()
   if rot.shape!=(n,24,3,3) or ids.shape!=(n,) or not torch.isfinite(rot).all() or not bool((ids[1:]>ids[:-1]).all()):
    raise ValueError('Invalid canonical motion pack')
   eye=torch.eye(3)
   if (rot@rot.transpose(-1,-2)-eye).abs().max()>.002 or (torch.linalg.det(rot)-1).abs().max()>.002:
    raise ValueError('Motion target is not SO(3)')
   cache[path]=(rot,ids)
  rot,ids=cache[path];count=min(n,a.frames);start=rng.randrange(n-count+1)
  r=rot[start:start+count];i=ids[start:start+count];valid=torch.arange(a.frames)<count
  if count<a.frames:
   r=torch.cat((r,r[-1:].expand(a.frames-count,-1,-1,-1)));i=torch.cat((i,i[-1:].expand(a.frames-count)))
  return r,i,valid
 while step<a.steps:
  model.train();optimizer.zero_grad(set_to_none=True)
  examples=[sample() for _ in range(a.batch_size)]
  target=torch.stack([x[0] for x in examples]).to(device)
  ids=torch.stack([x[1] for x in examples]).to(device)
  valid=torch.stack([x[2] for x in examples]).to(device)
  masked=(torch.rand(valid.shape,device=device)<.3)&valid
  masked[:,0]=False;masked[:,1]=True
  visible=valid&~masked
  noisy=rotation_6d(target[...,:2,:].flatten(-2)+torch.randn_like(target[...,:2,:].flatten(-2))*.03)
  # Completely hide masked frames; their GT remains only in the loss tensor.
  noisy=torch.where(visible[...,None,None,None],noisy,torch.eye(3,device=device))
  allowed=visible[:,None,:]&(ids[:,:,None]!=ids[:,None,:])&valid[:,:,None]
  _,raw,confidence,available=model(noisy,ids,ids,allowed)
  predicted=rotation_6d(raw.new_tensor([1.,0.,0.,0.,1.,0.])+raw)
  error=(predicted-target).square().mean((-1,-2))
  active=(masked&available)[...,None].expand_as(error)
  pose=error[active].mean()
  calibration=torch.nn.functional.binary_cross_entropy_with_logits(confidence.float(),torch.exp(-error.detach()/.25),reduction='none')[active].mean()
  loss=pose+.01*calibration
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite pretraining loss')
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
  optimizer.step();step+=1
  if step==1 or step%100==0:
   with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,rotation_chordal=float(pose.detach()),loss=float(loss.detach())))+'\n')
  if step%1000==0 or step==a.steps:save()
 with (a.output/'completion.json').open('x') as f:json.dump(dict(status='complete',steps=step,checkpoint=str(a.output/'latest.pt'),reconstruction_metrics=None),f,indent=2)

if __name__=='__main__':main()
