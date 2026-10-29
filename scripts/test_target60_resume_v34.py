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

 from types import SimpleNamespace
 from torch.utils.data import Subset,DataLoader
 from divibody_training.target60_engine_v34 import execute_run
 from divibody_training.target60_validation_v34 import validate
 from divibody_training.target60_contract_v34 import source_fingerprints
 import hashlib,random
 args.max_steps=4;args.stop_after=2;args.gradient_accumulation=2;args.batch_size=1
 args.eval_every=2;args.warmup_steps=2;args.diagnostic_every=1
 args.output_dir=a.output/'engine_run';args.resume_checkpoint=None;args.pa_weight=.05
 args.adapter_warmup_steps=0
 vv,vh=load(args.val_vace_cache),load(args.val_hmr_sidecar)
 val=TrackPrototypeDataset(V10Dataset(vv,vh))
 loader=DataLoader(Subset(val,[0,1,2,3]),batch_size=1,shuffle=False)
 source_sha=hashlib.sha256(args.init_checkpoint.read_bytes()).hexdigest()
 cfg=AdaptationConfig()
 model=model_from_payload(payload,cfg).to(device).bind_geometry(models[2]).bind_landmarks(official)
 optimizer,_=make_optimizer(model,args);ema=ModelEMA(model,args.ema_decay)
 metadata=dict(format=FORMAT,model_config=dataclasses.asdict(model.config),adaptation_config=dataclasses.asdict(cfg),
  backbone_spec=model.backbone_spec,reference_spec=model.reference_spec,
  source_sha256=source_sha,source_files_sha256=source_fingerprints(),
  asset_manifest={'diagnostic':'real immutable cache files, size/mtime only; no formal claim'},
  cache_hash_policy='diagnostic-size-mtime',environment={'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0)},
  coverage={'complete':False,'manifest_sha256':None},context_audit=dataset.context_audit,
  development_only=True,selection_split='four validation windows, development resume test only',
  args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
 args.output_dir.mkdir()
 def evaluate(network):
  eval_args=SimpleNamespace(**vars(args));eval_args.amp_dtype='none'
  return validate(loader,network,models,reg,eval_args,device),{}
 result['status']='engine_to_2';write()
 execute_run(model,optimizer,ema,SimpleNamespace(dataset=dataset),None,None,models,official,reg,evaluate,args,device,metadata)
 first=load(args.output_dir/'latest.pt')
 assert first['step']==2 and first['ema_updates']==2
 check('actual_engine_saved_optimizer_EMA_streams_RNG_at_2')
 del model,optimizer,ema,first;gc.collect();torch.cuda.empty_cache()
 # A new model/optimizer/EMA instance must restore strict native state.
 model=model_from_payload(payload,cfg).to(device).bind_geometry(models[2]).bind_landmarks(official)
 optimizer,_=make_optimizer(model,args);ema=ModelEMA(model,args.ema_decay)
 args.resume_checkpoint=args.output_dir/'latest.pt';args.stop_after=4
 metadata['args']=metadata['args']|dict(resume_checkpoint=str(args.resume_checkpoint),stop_after=4)
 result['status']='engine_resume_to_4';write()
 execute_run(model,optimizer,ema,SimpleNamespace(dataset=dataset),None,None,models,official,reg,evaluate,args,device,metadata)
 final=load(args.output_dir/'latest.pt')
 assert final['step']==4 and final['ema_updates']==4 and final['stream_states']['geometry']['draws']==8
 assert json.loads((args.output_dir/'completion.json').read_text())['completed_steps']==4
 assert any(json.loads(x).get('event')=='resumed' for x in (args.output_dir/'events.jsonl').read_text().splitlines())
 check('actual_engine_strict_resume_completed_4',optimized_windows=8,ema_updates=4)
 result['status']='passed';result['completed_utc_epoch']=time.time();write()
except BaseException as exc:
 result.update(status='failed',error=repr(exc),traceback=traceback.format_exc());write();traceback.print_exc();raise
