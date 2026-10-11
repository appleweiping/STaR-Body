"""All-decoder fine-tuning: mixed supervision, per-group LR, warmup and EMA."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from divibody_training.evaluate_standard_3dpw import create_smpl_models
from divibody_training.train_divibody_rgb_vace_hmr2a_v8 import load_pair
from divibody_training.train_divibody_rgb_vace_hmr2a_v9_small import spread_subset
from divibody_training.ram_parts_runtime_v22 import (
    V10Dataset,forward_batch,validate,evaluate_pose,pose_losses,geometric_losses,model_from_payload)
from divibody_training.ram_pose_stream_v10 import ShuffledPoseDatasetV10,CyclingPoseStream
from divibody_training.parts_training_utils_v20 import validate_cache_coverage,StepProbe
from divibody_training.parts_training_utils_v22 import make_optimizer,step_lr,ModelEMA
from models.human_decoder_parts_v22 import FORMAT

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('train-vace-cache','train-hmr-sidecar','val-vace-cache','val-hmr-sidecar',
        'pose-train-vace-dir','pose-train-hmr-dir','pose-val-vace-dir','pose-val-hmr-dir',
        'init-checkpoint','smpl-model-root','joint-regressor','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    for name,value in dict(epochs=15,batch_size=4,pose_batch_size=8,val_batch_size=16,
        num_workers=2,seed=2026,patience=6,min_epochs=8,max_train_windows=0,max_val_windows=0,
        max_pose_train_windows=0,max_pose_val_windows=0,diagnostic_every=100).items():
        p.add_argument('--'+name.replace('_','-'),type=int,default=value)
    for name,value in dict(rgb_lr=1e-6,vace_lr=1e-6,heads_lr=1e-6,camera_lr=3e-6,
        gate_lr=1e-5,local_lr=1e-5,weight_decay=1e-4,grad_clip=1.,warmup_epochs=1.,
        ema_decay=.995,pose_2d_weight=.02,pose_deadzone=.01,joint_weight=1.,vertex_weight=.5,
        rotation_weight=.05,shape_weight=.001,velocity_weight=.5,acceleration_weight=.25,
        pve_tolerance_mm=.5,pose_pck_tolerance=.005).items():
        p.add_argument('--'+name.replace('_','-'),type=float,default=value)
    p.add_argument('--amp-dtype',choices=('bf16','none'),default='bf16')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--validation-manifest',type=Path)
    p.add_argument('--allow-partial-validation',action='store_true')
    args=p.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,(float,int)) and (not math.isfinite(value) or value < 0):
            p.error(f'{key} must be finite and nonnegative')
    if min(args.epochs,args.batch_size,args.pose_batch_size,args.val_batch_size,args.patience,args.grad_clip,args.pose_2d_weight) <= 0:
        p.error('Epochs, batches, patience, clipping and pose supervision must be positive')
    if not 0 <= args.ema_decay < 1:
        p.error('EMA decay must be in [0,1)')
    args.anchor_weight=0.
    return args

def append_record(path,row):
    with path.open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)

def selection_score(geometry,pose,baseline_geometry,baseline_pose,args):
    if not all(math.isfinite(v) for v in [*geometry.values(),*pose.values()]):
        raise FloatingPointError('Nonfinite validation metrics')
    if geometry['pve_mm'] > baseline_geometry['pve_mm']+args.pve_tolerance_mm:
        return -float('inf')
    if any(pose[k] < baseline_pose[k]-args.pose_pck_tolerance for k in ('PCKn@0.05','PCKn@0.10')):
        return -float('inf')
    return -geometry['mpjpe_mm']

def train_epoch(model,loader,pose_stream,models,official_smpl,regressor,optimizer,ema,args,device,epoch):
    model.train()
    totals={}
    consumed=0
    started=time.monotonic()
    for step,batch in enumerate(loader):
        global_step=(epoch-1)*len(loader)+step
        step_lr(optimizer,global_step,args.epochs*len(loader),round(args.warmup_epochs*len(loader)))
        optimizer.zero_grad(set_to_none=True)
        prediction=forward_batch(model,batch,device,args.amp_dtype)
        losses=geometric_losses(prediction,batch,models,regressor,args,device)
        # Backward separately limits memory while accumulating a single mixed-data update.
        if not torch.isfinite(losses['total']):
            raise FloatingPointError('Nonfinite geometry loss')
        losses['total'].backward()
        values={key:float(value.detach()) for key,value in losses.items()}
        del prediction,losses
        pose_batch=pose_stream.next_batch()
        pp=forward_batch(model,pose_batch,device,args.amp_dtype)
        pl=pose_losses(pp,pose_batch,official_smpl,args,device)
        weight=args.pose_2d_weight*min(1.,(global_step+1)/max(1,len(loader)))
        pose_loss=weight*pl['pose_2d']
        if not torch.isfinite(pose_loss):
            raise FloatingPointError('Nonfinite PoseTrack loss')
        pose_loss.backward()
        values.update({key:float(value.detach()) for key,value in pl.items()})
        values['optimized_total']=values['total']+float(pose_loss.detach())
        del pp,pl,pose_loss
        diagnose=step==0 or (args.diagnostic_every>0 and step%args.diagnostic_every==0)
        probe=StepProbe(optimizer) if diagnose else None
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip,error_if_nonfinite=True)
        optimizer.step()
        ema.update(model)
        pose_stream.record_consumed(pose_batch)
        consumed+=len(batch['rgb_tokens'])
        values['grad_norm']=float(norm)
        for key,value in values.items():
            totals[key]=totals.get(key,0.)+value
        if probe is not None:
            append_record(args.output_dir/'diagnostics.jsonl',dict(epoch=epoch,step=step,
                groups=probe.finish(),losses=values,elapsed_seconds=time.monotonic()-started))
    return {k:v/len(loader) for k,v in totals.items()},consumed

def main():
    args=parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Use a new output directory; earlier runs are preserved')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device=torch.device(args.device)
    train_vace,train_hmr=load_pair(args.train_vace_cache,args.train_hmr_sidecar,'train')
    val_vace,val_hmr=load_pair(args.val_vace_cache,args.val_hmr_sidecar,'validation')
    coverage=validate_cache_coverage(train_vace,val_vace,args.validation_manifest,args.allow_partial_validation)
    train=spread_subset(V10Dataset(train_vace,train_hmr),args.max_train_windows)
    val=spread_subset(V10Dataset(val_vace,val_hmr),args.max_val_windows)
    train_pose=ShuffledPoseDatasetV10(args.pose_train_vace_dir,args.pose_train_hmr_dir,args.seed,args.max_pose_train_windows)
    val_pose=ShuffledPoseDatasetV10(args.pose_val_vace_dir,args.pose_val_hmr_dir,args.seed,args.max_pose_val_windows)
    if train_pose.split!='train' or val_pose.split!='val':
        raise ValueError('Incorrect PoseTrack split')
    train_loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed),pin_memory=True)
    val_loader=DataLoader(val,batch_size=args.val_batch_size,shuffle=False,num_workers=args.num_workers)
    pose_stream=CyclingPoseStream(DataLoader(train_pose,batch_size=args.pose_batch_size,num_workers=0,
        generator=torch.Generator().manual_seed(args.seed+1)))
    pose_val_loader=DataLoader(val_pose,batch_size=args.val_batch_size,num_workers=0)
    payload=torch.load(args.init_checkpoint,map_location='cpu',weights_only=False)
    model=model_from_payload(payload).to(device)
    models=create_smpl_models(args.smpl_model_root,device)
    for value in models.values():
        value.requires_grad_(False)
    model.bind_geometry(models[2])
    from hmr2.models.smpl_wrapper import SMPL as OfficialSMPL
    official_smpl=OfficialSMPL(model_path=str(args.smpl_model_root),gender='neutral',num_betas=10,
        joint_regressor_extra=str(args.smpl_model_root.parent/'SMPL_to_J19.pkl')).to(device).requires_grad_(False)
    regressor=torch.as_tensor(np.load(args.joint_regressor),dtype=torch.float32,device=device)
    optimizer,names=make_optimizer(model,args)
    ema=ModelEMA(model,decay=args.ema_decay)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    development=not coverage['complete'] or any((args.max_train_windows,args.max_val_windows,
        args.max_pose_train_windows,args.max_pose_val_windows))
    metadata=dict(format=FORMAT,model_config=asdict(model.config),adaptation_config=asdict(model.adaptation_config),
        initialization=str(args.init_checkpoint),source_format=payload['format'],source_epoch=payload.get('epoch'),
        trainable_parameters={name:p.numel() for name,p in model.named_parameters()},
        trainable_count=sum(p.numel() for p in model.parameters()),frozen_student_count=0,
        coverage=coverage,development_only=development,test_used_for_model_selection=False,
        training_windows_3dpw=len(train),training_windows_posetrack=len(train_pose),
        validation_windows_3dpw=len(val),validation_windows_posetrack=len(val_pose),
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    source_hash=hashlib.sha256()
    with args.init_checkpoint.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            source_hash.update(block)
    metadata['source_sha256']=source_hash.hexdigest()
    (args.output_dir/'run_config.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(dict(event='initialized',trainable_count=metadata['trainable_count'],frozen_student_count=0,
        development_only=development,train_windows=len(train),pose_train_windows=len(train_pose))),flush=True)
    def evaluate(network):
        geometry=validate(val_loader,network,models,regressor,args,device)
        pose=evaluate_pose(pose_val_loader,network,official_smpl,args,device)
        return geometry,pose
    def save(filename,network,epoch,variant,geometry,pose,*,resumable=False):
        state=metadata|dict(model=network.state_dict(),epoch=epoch,variant=variant,validation=geometry,
            pose_validation=pose,selection_metric='validation MPJPE with PVE and PCK guards')
        if resumable:
            state.update(optimizer=optimizer.state_dict(),ema_model=ema.model.state_dict(),
                ema_updates=ema.updates,pose_coverage=pose_stream.coverage())
        temporary=args.output_dir/(filename+'.tmp')
        torch.save(state,temporary)
        temporary.replace(args.output_dir/filename)
    baseline_geometry,baseline_pose=evaluate(model)
    best=selection_score(baseline_geometry,baseline_pose,baseline_geometry,baseline_pose,args)
    save('best.pt',model,0,'initial',baseline_geometry,baseline_pose)
    append_record(args.output_dir/'metrics.jsonl',dict(epoch=0,variant='initial',validation=baseline_geometry,
        pose_validation=baseline_pose,accepted=True,development_only=development))
    stale=0
    for epoch in range(1,args.epochs+1):
        losses,consumed=train_epoch(model,train_loader,pose_stream,models,official_smpl,regressor,optimizer,ema,args,device,epoch)
        candidates=[]
        for variant,network in (('student',model),('ema',ema.model)):
            geometry,pose=evaluate(network)
            score=selection_score(geometry,pose,baseline_geometry,baseline_pose,args)
            candidates.append((score,variant,network,geometry,pose))
            append_record(args.output_dir/'metrics.jsonl',dict(epoch=epoch,variant=variant,train=losses,
                validation=geometry,pose_validation=pose,development_only=development,
                geometry_windows_consumed=consumed,pose_coverage=pose_stream.coverage(),accepted=score>best))
        winner=max(candidates,key=lambda row:row[0])
        score,variant,network,geometry,pose=winner
        if score>best:
            best,stale=score,0
            save('best.pt',network,epoch,variant,geometry,pose)
        else:
            stale+=1
        student=candidates[0]
        save('latest.pt',model,epoch,'student',student[3],student[4],resumable=True)
        (args.output_dir/'progress.json').write_text(json.dumps(dict(epoch=epoch,best_mpjpe_mm=-best,
            stale_epochs=stale,pose_coverage=pose_stream.coverage(),complete=False)),encoding='utf-8')
        if epoch>=args.min_epochs and stale>=args.patience:
            break
    (args.output_dir/'completion.json').write_text(json.dumps(dict(completed_epochs=epoch,
        best_mpjpe_mm=-best,early_stopped=epoch<args.epochs,pose_coverage=pose_stream.coverage(),
        development_only=development)),encoding='utf-8')

if __name__=='__main__':
    main()
