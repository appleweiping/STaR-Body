"""v39 copied from v33: corrected native LSP PoseTrack head projection.

PoseTrack supervision can be restricted to camera calibration while 3DPW keeps
ownership of all geometry parameters.  This tests whether the frequent mixed
task gradient conflicts observed in v29/v30 caused the 3D regression.
"""
import argparse
from dataclasses import asdict
import hashlib
import inspect
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
from divibody_training.ram_parts_runtime_v39 import (
    V10Dataset,forward_batch,validate,evaluate_pose,pose_losses,geometric_losses,model_from_payload,CocoTrainingCache)
from divibody_training.ram_pose_stream_v10 import ShuffledPoseDatasetV10,CyclingPoseStream
from divibody_training.parts_training_utils_v20 import validate_cache_coverage,StepProbe
from divibody_training.parts_training_utils_v28 import make_optimizer,step_lr,ModelEMA
from models.human_decoder_parts_v39 import FORMAT, AdaptationConfig

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
        gate_lr=1e-5,local_lr=1e-5,geometry_lr=2e-5,weight_decay=1e-4,grad_clip=1.,warmup_epochs=1.,
        ema_decay=.995,pose_2d_weight=.02,pose_deadzone=.01,pose_projected_weight=.5,
        pose_confidence_weight=.02,joint_weight=1.,vertex_weight=.5,
        max_angle_degrees=8.,stage_2d_weight=.5,observation_weight=.5,uncertainty_weight=.01,
        visibility_weight=.02,anchor_weight=.1,update_regularization=.001,coco_weight=1.,
        rotation_weight=.05,shape_weight=.001,velocity_weight=.5,acceleration_weight=.25,
        pve_tolerance_mm=.5,pose_pck_tolerance=.005).items():
        p.add_argument('--'+name.replace('_','-'),type=float,default=value)
    p.add_argument('--protect-geometry-gradient',action='store_true')
    p.add_argument('--pose-gradient-route',choices=('shared','camera_only'),default='shared')
    p.add_argument('--geometry-gradient-scale',type=float,default=1.)
    p.add_argument('--amp-dtype',choices=('bf16','none'),default='bf16')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--validation-manifest',type=Path)
    p.add_argument('--allow-partial-validation',action='store_true')
    p.add_argument('--eval-only',action='store_true')
    p.add_argument('--eval-part-sweep',action='store_true')
    p.add_argument('--eval-part-angles',default='')
    p.add_argument('--iterations',type=int,default=2,choices=(1,2,3))
    p.add_argument('--coco-train-manifest',type=Path)
    args=p.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,(float,int)) and (not math.isfinite(value) or value < 0):
            p.error(f'{key} must be finite and nonnegative')
    if min(args.epochs,args.batch_size,args.pose_batch_size,args.val_batch_size,args.patience,args.grad_clip,args.pose_2d_weight) <= 0:
        p.error('Epochs, batches, patience, clipping and pose supervision must be positive')
    if not 0 <= args.ema_decay < 1:
        p.error('EMA decay must be in [0,1)')
    if not 0 <= args.geometry_gradient_scale <= 1:
        p.error('geometry-gradient-scale must be in [0,1]')
    if args.eval_part_sweep or args.eval_part_angles:
        p.error('Legacy part-angle sweeps do not apply to v28')
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
    pck_keys=('projected_PCKn@0.05','projected_PCKn@0.10')
    if any(pose[k] < baseline_pose[k]-args.pose_pck_tolerance for k in pck_keys):
        return -float('inf')
    assisted=(pose['PCKn@0.05']+pose['PCKn@0.10']-pose['mean_normalized_joint_error'])
    projected=(pose['projected_PCKn@0.05']+pose['projected_PCKn@0.10']
               -pose['projected_mean_normalized_joint_error'])
    return projected

def merge_task_gradients(parameters, geometry_grads, protect):
    pairs=[(p,g) for p,g in zip(parameters,geometry_grads) if g is not None]
    if not pairs:
        return {'gradient_dot':0.,'gradient_conflict':0.}
    norm=sum(g.float().square().sum() for p,g in pairs)
    dot=sum((p.grad.float()*g.float()).sum() if p.grad is not None else g.new_zeros(()) for p,g in pairs)
    coefficient=torch.minimum(dot/norm.clamp_min(1e-20),dot.new_zeros(())) if protect else dot.new_zeros(())
    for p,g in pairs:
        if p.grad is None:
            p.grad=g*(1-coefficient)
        else:
            p.grad.add_(g*(1-coefficient))
    return {'gradient_dot':float(dot),'gradient_conflict':float(dot<0),'gradient_projection':float(coefficient)}

def route_pose_gradients(model, route):
    """Keep PoseTrack updates off geometry when running the camera-only study."""
    if route == 'shared':
        return 0
    kept=0
    for name,parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        if name.startswith('refiner.camera.'):
            kept += parameter.numel()
        else:
            parameter.grad=None
    if kept == 0:
        raise RuntimeError('camera_only route found no camera gradients')
    return kept


def train_epoch(model,loader,pose_stream,models,official_smpl,regressor,optimizer,ema,args,device,epoch,coco_loader=None):
    model.train()
    totals={}
    consumed=0
    started=time.monotonic()
    coco_iterator=iter(coco_loader) if coco_loader is not None else None
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
        shared=[p for p in model.parameters() if p.requires_grad]
        geometry_grads=[None if p.grad is None else p.grad.detach().clone()*args.geometry_gradient_scale for p in shared]
        optimizer.zero_grad(set_to_none=True)
        del prediction,losses
        pose_batch=pose_stream.next_batch()
        pp=forward_batch(model,pose_batch,device,args.amp_dtype)
        pl=pose_losses(pp,pose_batch,official_smpl,args,device)
        weight=args.pose_2d_weight*min(1.,(global_step+1)/max(1,len(loader)))
        pose_loss=weight*pl['pose_2d']
        if not torch.isfinite(pose_loss):
            raise FloatingPointError('Nonfinite PoseTrack loss')
        pose_loss.backward()
        values['pose_gradient_parameters']=float(route_pose_gradients(model,args.pose_gradient_route))
        values.update({key:float(value.detach()) for key,value in pl.items()})
        values['optimized_total']=values['total']+float(pose_loss.detach())
        del pp,pl,pose_loss
        if coco_iterator is not None:
            try:
                coco_batch=next(coco_iterator)
            except StopIteration:
                coco_iterator=iter(coco_loader)
                coco_batch=next(coco_iterator)
            cp=forward_batch(model,coco_batch,device,args.amp_dtype)
            cl=pose_losses(cp,coco_batch,official_smpl,args,device)
            coco_loss=args.coco_weight*cl['pose_2d']
            if not torch.isfinite(coco_loss):
                raise FloatingPointError('Nonfinite COCO training loss')
            coco_loss.backward()
            values['coco_loss']=float(coco_loss.detach())
            del cp,cl,coco_loss
        values.update(merge_task_gradients(shared,geometry_grads,args.protect_geometry_gradient))
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
    config=AdaptationConfig(max_angle_degrees=args.max_angle_degrees,iterations=args.iterations)
    model=model_from_payload(payload,config).to(device)
    models=create_smpl_models(args.smpl_model_root,device)
    for value in models.values():
        value.requires_grad_(False)
    model.bind_geometry(models[2])
    from hmr2.models.smpl_wrapper import SMPL as OfficialSMPL
    official_smpl=OfficialSMPL(model_path=str(args.smpl_model_root),gender='neutral',num_betas=10,
        joint_regressor_extra=str(args.smpl_model_root.parent/'SMPL_to_J19.pkl')).to(device).requires_grad_(False)
    model.bind_landmarks(official_smpl)
    coco_loader=None
    if args.coco_train_manifest is not None:
        coco_loader=DataLoader(CocoTrainingCache(args.coco_train_manifest),batch_size=args.batch_size,
            shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(args.seed+2))
    regressor=torch.as_tensor(np.load(args.joint_regressor),dtype=torch.float32,device=device)
    optimizer,names=make_optimizer(model,args)
    ema=ModelEMA(model,decay=args.ema_decay)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    development=not coverage['complete'] or any((args.max_train_windows,args.max_val_windows,
        args.max_pose_train_windows,args.max_pose_val_windows))
    metadata=dict(experiment_version='v39_lsp_head_mapping_correction',format=FORMAT,model_config=asdict(model.config),adaptation_config=asdict(model.adaptation_config),
        teacher_config=asdict(model.teacher_config),
        selection_split='3DPW validation + PoseTrack val development',
        coco_training_enabled=coco_loader is not None,
        initialization=str(args.init_checkpoint),source_format=payload['format'],source_epoch=payload.get('epoch'),
        trainable_parameters={name:p.numel() for name,p in model.named_parameters() if p.requires_grad},
        trainable_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
        frozen_student_count=sum(p.numel() for p in model.parameters() if not p.requires_grad),
        coverage=coverage,development_only=development,test_used_for_model_selection=False,
        training_windows_3dpw=len(train),training_windows_posetrack=len(train_pose),
        validation_windows_3dpw=len(val),validation_windows_posetrack=len(val_pose),
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    source_hash=hashlib.sha256()
    with args.init_checkpoint.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            source_hash.update(block)
    metadata['source_sha256']=source_hash.hexdigest()
    source_paths=[Path(__file__),Path(inspect.getfile(type(model))),Path(inspect.getfile(forward_batch)),
                  Path(inspect.getfile(make_optimizer))]
    metadata['source_files_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    (args.output_dir/'run_config.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(dict(event='initialized',trainable_count=metadata['trainable_count'],
        frozen_student_count=metadata['frozen_student_count'],
        development_only=development,train_windows=len(train),pose_train_windows=len(train_pose))),flush=True)
    def evaluate(network):
        geometry=validate(val_loader,network,models,regressor,args,device)
        pose=evaluate_pose(pose_val_loader,network,official_smpl,args,device)
        return geometry,pose
    if args.eval_part_angles:
        candidates=[]
        for index,specification in enumerate(args.eval_part_angles.split(';')):
            angles=[float(value) for value in specification.split(',')]
            if len(angles)!=6 or any(not 0 < value < args.max_angle_degrees for value in angles):
                raise ValueError('Each eval-part-angles candidate needs six values below max-angle-degrees')
            candidates.append((f'custom_{index}',angles))
        for label,angles in candidates:
            values=torch.tensor(angles,device=device)/args.max_angle_degrees
            logits=torch.logit(values.clamp(1e-4,1-1e-4))
            with torch.no_grad():
                model.part_branch.part_logits.copy_(logits)
            geometry,pose=evaluate(model)
            append_record(args.output_dir/'part_combinations.jsonl',dict(
                candidate=label,part_angles_degrees=angles,validation=geometry,
                pose_validation=pose,development_only=development))
        (args.output_dir/'completion.json').write_text(json.dumps(dict(
            completed_epochs=0,part_combinations=True,development_only=development,
            eval_only=True)),encoding='utf-8')
        return
    if args.eval_part_sweep:
        if args.max_angle_degrees <= 12.5:
            raise ValueError('eval-part-sweep requires max-angle-degrees > 12.5')
        groups={'legs':(0,1),'axial':(2,3),'arms':(4,5)}
        candidates=[('baseline',[10.]*6)]
        for label,indices in groups.items():
            for value in (7.5,12.5):
                angles=[10.]*6
                for index in indices:
                    angles[index]=value
                candidates.append((f'{label}_{value:g}',angles))
        for label,angles in candidates:
            values=torch.tensor(angles,device=device)/args.max_angle_degrees
            logits=torch.logit(values.clamp(1e-4,1-1e-4))
            with torch.no_grad():
                model.part_branch.part_logits.copy_(logits)
            geometry,pose=evaluate(model)
            append_record(args.output_dir/'part_sweep.jsonl',dict(
                candidate=label,part_angles_degrees=angles,validation=geometry,
                pose_validation=pose,development_only=development))
        (args.output_dir/'completion.json').write_text(json.dumps(dict(
            completed_epochs=0,part_sweep=True,development_only=development,
            eval_only=True)),encoding='utf-8')
        return
    def save(filename,network,epoch,variant,geometry,pose,*,resumable=False):
        state=metadata|dict(model=network.state_dict(),epoch=epoch,variant=variant,validation=geometry,
            pose_validation=pose,selection_metric='direct SMPL projection only, PVE constrained')
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
    if args.eval_only:
        (args.output_dir/'completion.json').write_text(json.dumps(dict(completed_epochs=0,
            best_pose_score=best,early_stopped=False,pose_coverage=pose_stream.coverage(),
            development_only=development,eval_only=True)),encoding='utf-8')
        return
    stale=0
    for epoch in range(1,args.epochs+1):
        losses,consumed=train_epoch(model,train_loader,pose_stream,models,official_smpl,regressor,optimizer,ema,args,device,epoch,coco_loader)
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
        (args.output_dir/'progress.json').write_text(json.dumps(dict(epoch=epoch,best_pose_score=best,
            stale_epochs=stale,pose_coverage=pose_stream.coverage(),complete=False)),encoding='utf-8')
        if epoch>=args.min_epochs and stale>=args.patience:
            break
    (args.output_dir/'completion.json').write_text(json.dumps(dict(completed_epochs=epoch,
        best_pose_score=best,early_stopped=epoch<args.epochs,pose_coverage=pose_stream.coverage(),
        development_only=development)),encoding='utf-8')

if __name__=='__main__':
    main()
