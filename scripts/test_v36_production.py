"""Real production-window V36 acceptance. No benchmark/claimed training result."""

import argparse,dataclasses,gc,json,os,time,traceback,hashlib

from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)

p.add_argument('--memory-gib',type=float,default=3.);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)

result=dict(status='importing',checks=[],benchmark=False,formal_training_started=False)

def write():

 t=a.output/'status.json.tmp';t.write_text(json.dumps(result,indent=2));t.replace(a.output/'status.json')

def check(name,**values):

 result['checks'].append(dict(name=name,passed=True,**values));write();print(json.dumps(result['checks'][-1]),flush=True)

write()

try:

 import numpy as np,torch

 from torch.utils.data import default_collate

 from divibody_training.target60_runtime_v34 import V10Dataset,forward_batch

 from divibody_training.target60_track_dataset_v34 import TrackPrototypeDataset

 from divibody_training.target60_contract_v36 import parse_training_args

 from divibody_training.target60_boundary_v34 import build_boundary_pool,boundary_training_loss

 from divibody_training.target60_partition_v36 import geometric_losses,audit_boundary_partition

 from divibody_training.target60_partition_v34 import geometric_losses as legacy_loss

 from divibody_training.target60_optim_v36 import make_optimizer,ModelEMA

 from divibody_training.evaluate_standard_3dpw import create_smpl_models

 from models.human_decoder_target60_v34 import model_from_payload as parent_model

 from models.human_decoder_kinematic_v36 import V36Config,FORMAT,model_from_payload

 from hmr2.models.smpl_wrapper import SMPL

 torch.set_num_threads(4);torch.manual_seed(2026);device=torch.device('cuda:0')

 torch.cuda.set_per_process_memory_fraction(a.memory_gib*1024**3/torch.cuda.get_device_properties(0).total_memory,0)

 c=json.loads(a.config.read_text());c['output-dir']=str(a.output/'unused');cli=[]

 for k,v in c.items():

  if v is False or v is None:continue

  cli+=['--'+k]

  if v is not True:cli+=[str(v)]

 requested_joint=bool(c.get('v37-joint-feedback',False));c.pop('v37-joint-feedback',None)
 args=parse_training_args([v for v in cli if v!='--v37-joint-feedback']);load=lambda x:torch.load(x,map_location='cpu',weights_only=False,mmap=True)

 payload=load(args.init_checkpoint)

 vace,hmr=load(args.train_vace_cache),load(args.train_hmr_sidecar)

 dataset=TrackPrototypeDataset(V10Dataset(vace,hmr),args.train_track_prototypes,payload['source_sha256'])

 pool=build_boundary_pool(dataset);partition=audit_boundary_partition(dataset,pool)

 args.motion_train_windows=len(dataset);args.motion_canonical_within_triplets=partition['canonical_within_triplets']

 assert partition['full_contiguous_triplets']==22481

 check('canonical_train_partition',triplets=22481)
 from collections import Counter
 from models.part_time_refiner_v36 import temporal_allowed
 cadence=Counter()
 for record,mask in zip(vace['metadata'],vace['target_valid_mask'].tolist()):
  ids=record.get('target_source_frame_ids')
  if ids is None:ids=record['target_frame_ids']
  for i in range(7):
   if mask[i] and mask[i+1]:cadence[int(ids[i+1])-int(ids[i])]+=1
 assert set(cadence)<={0,2,3} and cadence[2]>0
 check('full_train_source_cadence_audited',histogram=dict(cadence),source_stride=2,source_jitter=1)

 batch=default_collate([dataset[0]])

 bodies=create_smpl_models(args.smpl_model_root,device)

 for body in bodies.values():body.eval().requires_grad_(False)

 official=SMPL(model_path=str(args.smpl_model_root),gender='neutral',num_betas=10,

  joint_regressor_extra=str(args.smpl_model_root.parent/'SMPL_to_J19.pkl')).to(device).eval().requires_grad_(False)

 reg=torch.as_tensor(np.load(args.joint_regressor),device=device,dtype=torch.float32)

 old=parent_model(payload).to(device).bind_geometry(bodies[2]).bind_landmarks(official).eval()

 with torch.no_grad():reference={k:v.cpu() for k,v in forward_batch(old,batch,device,'none').items() if k in ('rotmat','betas','camera')}

 del old;gc.collect();torch.cuda.empty_cache()

 for arm in ('R0','R1','R2','R3a','R3b','R4'):

  result['status']='testing_'+arm;write();args.v36_arm=arm

  cfg=V36Config(arm=arm,temporal=arm in ('R2','R3a','R3b','R4'),skeleton=arm in ('R3a','R3b','R4'),ik=arm in ('R3b','R4'),shape_consensus=arm=='R4',feedback_mode='joint' if requested_joint and arm in ('R2','R3a','R3b','R4') else 'mean')

  cuda_rng=torch.cuda.get_rng_state().clone()
  model=model_from_payload(payload,cfg).to(device).bind_geometry(bodies[2]).bind_landmarks(official).eval()
  assert torch.equal(cuda_rng,torch.cuda.get_rng_state()),'Model construction changed CUDA RNG'
  check('model_construction_preserves_cuda_rng_'+arm)

  with torch.no_grad():initial=forward_batch(model,batch,device,'none')

  for key,value in reference.items():torch.testing.assert_close(initial[key].cpu(),value,rtol=0,atol=2e-6)

  check('trained_parent_compatibility_'+arm)
  if cfg.temporal:
   allowed,independent=temporal_allowed(batch['source_frame_ids'],batch['target_valid'],cfg.source_stride,cfg.source_jitter)
   assert int(allowed.sum())>batch['target_valid'].shape[1]
   check('real_mapped_source_neighbors_allowed_'+arm,allowed_pairs=int(allowed.sum()))

  optimizer,names=make_optimizer(model,args);ema=ModelEMA(model,args.ema_decay)

  torch.cuda.reset_peak_memory_stats()

  # Loss decomposition must equal the legacy objective and its selected gradient.

  if arm=='R0':

   model.train();pred=forward_batch(model,batch,device,'none')

   new=geometric_losses(pred,batch,bodies,reg,args,device)

   old=legacy_loss(pred,batch,bodies,reg,args,device)

   torch.testing.assert_close(new['total'],old['total'],atol=1e-7,rtol=1e-6)

   torch.testing.assert_close(new['space_total']+new['motion_total']+new['regularizer_total'],new['total'],atol=1e-7,rtol=1e-6)

   weight=model.track_shape[-1].weight

   ng=torch.autograd.grad(new['total'],weight,retain_graph=True)[0];og=torch.autograd.grad(old['total'],weight)[0]

   torch.testing.assert_close(ng,og,atol=2e-6,rtol=2e-5)

   check('legacy_loss_scalar_components_gradient_equivalence');del new,old,pred,ng,og

  # Exercise all three trainable phases without claiming optimizer exposure=800.

  for phase in (0,300,800):

   report=model.configure_phase(phase);model.train();optimizer.zero_grad(set_to_none=True)

   pred=forward_batch(model,batch,device,'bf16');loss=geometric_losses(pred,batch,bodies,reg,args,device)

   assert torch.isfinite(loss['total']);loss['total'].backward()

   grads={n:p.grad for n,p in model.named_parameters() if p.grad is not None}

   assert grads and all(torch.isfinite(g).all() for g in grads.values())

   if arm!='R0' and phase>=300:assert any(n.startswith('backbone.body_head.') and g.abs().sum()>0 for n,g in grads.items())

   if arm!='R0' and phase>=800:assert any(n.startswith('backbone.blocks.3.') and g.abs().sum()>0 for n,g in grads.items())

   if cfg.feedback_mode=='joint':
    assert model.refiner.joint_feedback.output.weight.grad is not None
    assert model.refiner.joint_feedback.output.weight.grad.abs().sum()>0
    if phase>=300:
     assert any('joint_feedback.attention' in n and g.abs().sum()>0 for n,g in grads.items())
    check('joint_feedback_gradient_'+arm+'_'+str(phase))
   assert not any(n.startswith('backbone.reference.') for n in grads)

   torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();ema.update(model)

   check('real_optimizer_phase_'+arm+'_'+str(phase),**report)

   del pred,loss,grads

  optimizer.zero_grad(set_to_none=True)

  boundary,_=boundary_training_loss(model,dataset,pool,bodies,reg,device,'bf16',step=0,seed=2026,triplets_per_step=1)

  boundary.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

  del boundary

  model.eval()

  with torch.no_grad():out=forward_batch(model,batch,device,'none')

  r=out['rotmat'];torch.testing.assert_close(r@r.transpose(-1,-2),torch.eye(3,device=device).expand_as(r),atol=2e-4,rtol=0)

  assert (torch.linalg.det(r)>.999).all()

  allowed={'rgb_tokens','vace_features','base_rotmat','base_betas','base_camera','target_valid','source_frame_ids','track_shape_base','track_shape_features'}

  changed={k:(torch.full_like(v,float('nan')) if v.is_floating_point() else torch.zeros_like(v)) if torch.is_tensor(v) and k not in allowed else v for k,v in batch.items()}

  with torch.no_grad():replaced=forward_batch(model,changed,device,'none')

  for key in reference:torch.testing.assert_close(out[key],replaced[key],rtol=0,atol=0)

  pack=dict(format=FORMAT,model=model.state_dict(),step=model.phase_step,v36_config=dataclasses.asdict(cfg),

    parent_spec=model.parent_spec,development_only=True,initial_fallback_only=False)

  path=a.output/(arm+'_development_roundtrip.pt');torch.save(pack,path)

  restored=model_from_payload(load(path)).to(device).bind_geometry(bodies[2]).bind_landmarks(official).eval()

  with torch.no_grad():again=forward_batch(restored,batch,device,'none')

  for key in reference:torch.testing.assert_close(out[key],again[key],rtol=0,atol=0)

  check('GT_invariance_SO3_boundary_EMA_native_roundtrip_'+arm,peak_allocated_bytes=torch.cuda.max_memory_allocated())

  del restored,again,pack,model,optimizer,ema,initial,out,r,changed,replaced;gc.collect();torch.cuda.empty_cache()

 result.update(status='passed',completed_epoch=time.time());write()

except BaseException as exc:

 result.update(status='failed',error=repr(exc),traceback=traceback.format_exc());write();traceback.print_exc();raise

