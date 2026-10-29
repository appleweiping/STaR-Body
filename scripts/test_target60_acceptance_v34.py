"""V34 native warm-start and adapter smoke on real train windows, not benchmark."""
import argparse,dataclasses,gc,json,os,time,traceback
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--memory-gib',type=float,default=2.5);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
result={'status':'importing','checks':[],'benchmark':False,'training_started':False,'shape_context':'synthetic zero prototype for branch smoke only; production V30 prototype pack not validated here'}
def write():
 t=a.output/'status.json.tmp';t.write_text(json.dumps(result,indent=2));t.replace(a.output/'status.json')
def check(name,**detail):
 result['checks'].append(dict(name=name,passed=True,**detail));write();print(json.dumps(result['checks'][-1]),flush=True)
write()
try:
 import numpy as np,torch
 from torch.utils.data import default_collate
 from divibody_training.target60_runtime_v34 import V10Dataset,forward_batch
 from divibody_training.target60_track_dataset_v34 import TrackPrototypeDataset
 from divibody_training.target60_contract_v34 import parse_training_args
 from divibody_training.target60_boundary_v34 import build_boundary_pool,boundary_training_loss
 from divibody_training.target60_partition_v34 import geometric_losses,audit_boundary_partition
 from divibody_training.target60_optim_v34 import make_optimizer,ModelEMA
 from divibody_training.evaluate_standard_3dpw import create_smpl_models
 from models.human_decoder_target60_v34 import AdaptationConfig,FORMAT,model_from_payload
 from models.human_decoder_spatiotemporal_v30 import model_from_payload as native_v30
 from hmr2.models.smpl_wrapper import SMPL
 torch.set_num_threads(4);torch.manual_seed(2026)
 device=torch.device('cuda:0')
 torch.cuda.set_per_process_memory_fraction(a.memory_gib*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
 c=json.loads(a.config.read_text());c['output-dir']=str(a.output/'unused');c.pop('initial-reference-report',None)
 cli=[]
 for k,v in c.items():
  if v is False or v is None:continue
  cli+=['--'+k]
  if v is not True:cli+=[str(v)]
 args=parse_training_args(cli)
 load=lambda x:torch.load(x,map_location='cpu',weights_only=False,mmap=True)
 vace,hmr=load(args.train_vace_cache),load(args.train_hmr_sidecar)
 class SmokeContext(TrackPrototypeDataset):
  def __getitem__(self,index):
   s=super().__getitem__(index)
   s.update(track_shape_base=torch.zeros(10),track_shape_features=torch.zeros(1280))
   return s
 dataset=SmokeContext(V10Dataset(vace,hmr))
 pool=build_boundary_pool(dataset);partition=audit_boundary_partition(dataset,pool)
 args.motion_train_windows=len(dataset);args.motion_canonical_within_triplets=partition['canonical_within_triplets']
 check('complete_train_boundary_partition',windows=len(dataset),triplets=partition['full_contiguous_triplets'])
 batch=default_collate([dataset[0]])
 models=create_smpl_models(args.smpl_model_root,device)
 for body in models.values():body.eval().requires_grad_(False)
 official=SMPL(model_path=str(args.smpl_model_root),gender='neutral',num_betas=10,
  joint_regressor_extra=str(args.smpl_model_root.parent/'SMPL_to_J19.pkl')).to(device).eval().requires_grad_(False)
 reg=torch.as_tensor(np.load(args.joint_regressor),device=device,dtype=torch.float32)
 payload=load(args.init_checkpoint)
 original=native_v30(payload).to(device).bind_geometry(models[2]).bind_landmarks(official).eval()
 with torch.no_grad():reference={k:v.detach().cpu() for k,v in forward_batch(original,{k:v for k,v in batch.items() if k not in ('track_shape_base','track_shape_features')},device,'none').items() if k in ('rotmat','betas','camera')}
 del original;gc.collect();torch.cuda.empty_cache()
 for name,flags,pa in [('A1',{},0.),('A2',{},.05),('A3',{'use_track_shape':True},.05),
   ('A4_gru',{'use_gru':True,'freeze_base':True},.05),('combined',{'use_gru':True,'use_track_shape':True,'freeze_base':True},.05)]:
  result['status']='testing_'+name;write();args.pa_weight=pa;args.adapter_warmup_steps=0
  torch.manual_seed(2026);cfg=AdaptationConfig(**flags)
  model=model_from_payload(payload,cfg).to(device).bind_geometry(models[2]).bind_landmarks(official)
  # All tensors of the trained source must be loaded strictly and unchanged.
  for key,value in payload['model'].items():torch.testing.assert_close(model.backbone.state_dict()[key].cpu(),value,rtol=0,atol=0)
  model.eval()
  with torch.no_grad():initial=forward_batch(model,batch,device,'none')
  for key,value in reference.items():torch.testing.assert_close(initial[key].cpu(),value,rtol=0,atol=1e-6)
  check('native_v30_zero_init_equivalence_'+name)
  optimizer,_=make_optimizer(model,args);ema=ModelEMA(model,args.ema_decay);torch.cuda.reset_peak_memory_stats()
  for step in range(2):
   model.train();optimizer.zero_grad(set_to_none=True)
   pred=forward_batch(model,batch,device,'bf16');loss=geometric_losses(pred,batch,models,reg,args,device)
   assert torch.isfinite(loss['total']);loss['total'].backward()
   grads={n:p.grad for n,p in model.named_parameters() if p.grad is not None}
   assert grads and all(torch.isfinite(g).all() for g in grads.values())
   if step==0 and cfg.use_track_shape:assert model.track_shape[-1].weight.grad.abs().sum()>0
   if step==0 and cfg.use_gru:assert sum(h.weight.grad.abs().sum() for h in model.refiner.heads)>0
   torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
   optimizer.step();ema.update(model)
   del pred,loss
  for n,p0 in model.named_parameters():
   if not p0.requires_grad:torch.testing.assert_close(ema.model.state_dict()[n],p0,rtol=0,atol=0)
  optimizer.zero_grad(set_to_none=True)
  boundary,info=boundary_training_loss(model,dataset,pool,models,reg,device,'bf16',step=0,seed=2026,triplets_per_step=1)
  boundary.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
  torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
  model.eval()
  with torch.no_grad():out=forward_batch(model,batch,device,'none')
  r=out['rotmat'];torch.testing.assert_close(r@r.transpose(-1,-2),torch.eye(3,device=device).expand_as(r),rtol=0,atol=1e-4)
  assert (torch.linalg.det(r)>.999).all()
  if name=='combined':
   pack=dict(format=FORMAT,model=model.state_dict(),backbone_spec=model.backbone_spec,
    model_config=dataclasses.asdict(model.config),adaptation_config=dataclasses.asdict(cfg))
   path=a.output/'native_roundtrip.pt';torch.save(pack,path)
   restored=model_from_payload(load(path)).to(device).bind_geometry(models[2]).bind_landmarks(official).eval()
   with torch.no_grad():again=forward_batch(restored,batch,device,'none')
   for key in ('rotmat','betas','camera'):torch.testing.assert_close(out[key],again[key],rtol=0,atol=0)
   check('native_v34_checkpoint_roundtrip')
   del restored,again,pack
  check('real_window_backward_boundary_EMA_SO3_'+name,peak_allocated_bytes=torch.cuda.max_memory_allocated())
  del model,optimizer,ema,out,r,initial,boundary;gc.collect();torch.cuda.empty_cache()
 result['status']='passed';result['completed_utc_epoch']=time.time();write()
except BaseException as exc:
 result.update(status='failed',error=repr(exc),traceback=traceback.format_exc());write();traceback.print_exc();raise
