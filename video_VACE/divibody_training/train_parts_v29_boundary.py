"""v28 shared iterative refiner; mixed 3DPW/PoseTrack and optional COCO-train caches."""
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
from divibody_training.ram_parts_runtime_v29 import (
    V10Dataset,forward_batch,validate,evaluate_pose,pose_losses,geometric_losses,model_from_payload,CocoTrainingCache)
from divibody_training.ram_pose_stream_v10 import ShuffledPoseDatasetV10,CyclingPoseStream
from divibody_training.v29_experiment_utils import validate_coverage
from divibody_training.v29_training_engine_boundary import execute_run
from divibody_training.parts_training_utils_v29 import make_optimizer,step_lr,ModelEMA
from models.human_decoder_parts_v29 import FORMAT, AdaptationConfig
from divibody_training.v29_sequence_evaluation_report import validate

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
    p.add_argument('--amp-dtype',choices=('bf16','none'),default='bf16')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--validation-manifest',type=Path)
    p.add_argument('--allow-partial-validation',action='store_true')
    p.add_argument('--eval-only',action='store_true')
    p.add_argument('--eval-part-sweep',action='store_true')
    p.add_argument('--eval-part-angles',default='')
    p.add_argument('--iterations',type=int,default=2,choices=(1,2,3))
    p.add_argument('--coco-train-manifest',type=Path)
    p.add_argument('--max-steps',type=int,default=6000)
    p.add_argument('--eval-every',type=int,default=1000)
    p.add_argument('--stop-after',type=int,default=0)
    p.add_argument('--resume-checkpoint',type=Path)
    p.add_argument('--coco-warmup-steps',type=int,default=500)
    p.add_argument('--warmup-steps',type=int,default=200)
    p.add_argument('--geometry-only-steps',type=int,default=500)
    p.add_argument('--auxiliary-ramp-steps',type=int,default=500)
    p.add_argument('--auxiliary-grad-ratio',type=float,default=.1)
    p.add_argument('--stage-3d-weight',type=float,default=.25)
    p.add_argument('--shape-consistency-weight',type=float,default=.01)
    p.add_argument('--shape-bound',type=float,default=.5)
    p.add_argument('--mpjpe-tolerance-mm',type=float,default=0.)
    p.add_argument('--pa-mpjpe-tolerance-mm',type=float,default=0.)
    p.add_argument('--strict-3d-selection',action='store_true')
    p.add_argument('--geometry-only-validation',action='store_true')
    p.add_argument('--acceleration-ceiling-mm',type=float,required=True)
    p.add_argument('--boundary-weight',type=float,default=.25)
    p.add_argument('--boundary-triplets-per-step',type=int,default=2)
    args=p.parse_args()
    if args.max_train_windows or args.boundary_triplets_per_step<1:
        p.error('Boundary training requires complete train dataset and positive triplet batch')
    if args.max_steps<1 or args.eval_every<1:
        p.error('Positive step budget and evaluation period required')
    if args.resume_checkpoint and args.resume_checkpoint.resolve() != (args.output_dir/'latest.pt').resolve():
        p.error('Resume uses latest.pt in the same run directory')
    for key,value in vars(args).items():
        if isinstance(value,(float,int)) and (not math.isfinite(value) or value < 0):
            p.error(f'{key} must be finite and nonnegative')
    if min(args.epochs,args.batch_size,args.pose_batch_size,args.val_batch_size,args.patience,args.grad_clip,args.pose_2d_weight) <= 0:
        p.error('Epochs, batches, patience, clipping and pose supervision must be positive')
    if not 0 <= args.ema_decay < 1:
        p.error('EMA decay must be in [0,1)')
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

def main():
    args=parse_args()
    if not args.resume_checkpoint and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Use a new output directory; earlier runs are preserved')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device=torch.device(args.device)
    train_vace,train_hmr=load_pair(args.train_vace_cache,args.train_hmr_sidecar,'train')
    val_vace,val_hmr=load_pair(args.val_vace_cache,args.val_hmr_sidecar,'validation')
    coverage=validate_coverage(train_vace,val_vace,args.validation_manifest,args.allow_partial_validation)
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
    config=AdaptationConfig(max_angle_degrees=args.max_angle_degrees,iterations=args.iterations,shape_bound=args.shape_bound)
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
    metadata=dict(format=FORMAT,model_config=asdict(model.config),adaptation_config=asdict(model.adaptation_config),
        teacher_config=asdict(model.teacher_config),
        selection_split='3DPW validation only' if args.geometry_only_validation else '3DPW validation + PoseTrack val',
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
    (args.output_dir/('resume_config.json' if args.resume_checkpoint else 'run_config.json')).write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(dict(event='initialized',trainable_count=metadata['trainable_count'],
        frozen_student_count=metadata['frozen_student_count'],
        development_only=development,train_windows=len(train),pose_train_windows=len(train_pose))),flush=True)
    def evaluate(network):
        geometry=validate(val_loader,network,models,regressor,args,device)
        print(json.dumps(dict(event='geometry_validation_done',metrics=geometry)),flush=True)
        pose=(dict(projected_PCKn05_unused=0.,**{'projected_PCKn@0.05':0.,'projected_PCKn@0.10':0.,
                    'projected_mean_normalized_joint_error':0.}) if args.geometry_only_validation
              else evaluate_pose(pose_val_loader,network,official_smpl,args,device))
        return geometry,pose
    execute_run(model,optimizer,ema,train_loader,pose_stream.loader,coco_loader,
                models,official_smpl,regressor,evaluate,args,device,metadata)

if __name__=='__main__':
    main()
