"""V31 additive architecture with V30-compatible defaults and bound run provenance."""
import argparse
from dataclasses import asdict
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import platform
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from divibody_training.evaluate_standard_3dpw import create_smpl_models
from divibody_training.train_divibody_rgb_vace_hmr2a_v8 import load_pair
from divibody_training.train_divibody_rgb_vace_hmr2a_v9_small import spread_subset
from divibody_training.ram_parts_runtime_v31 import (
    V10Dataset,forward_batch,validate,evaluate_pose,pose_losses,geometric_losses,model_from_payload,CocoTrainingCache)
from divibody_training.ram_pose_stream_v10 import ShuffledPoseDatasetV10,CyclingPoseStream
from divibody_training.v29_experiment_utils import validate_coverage
from divibody_training.v31_training_engine import execute_run
from divibody_training.parts_training_utils_v31 import make_optimizer,step_lr,ModelEMA
from models.human_decoder_spatiotemporal_v31 import FORMAT, AdaptationConfig
from divibody_training.v29_sequence_evaluation_report import validate

from divibody_training.v31_training_contract import (
    parse_training_args as parse_args, pose_training_enabled, inspect_assets, source_fingerprints)


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
    asset_audit=inspect_assets(args,fingerprint=True)
    if not asset_audit['passed']:
        raise ValueError('V31 asset preflight failed: '+json.dumps(asset_audit))
    train_vace,train_hmr=load_pair(args.train_vace_cache,args.train_hmr_sidecar,'train')
    val_vace,val_hmr=load_pair(args.val_vace_cache,args.val_hmr_sidecar,'validation')
    coverage=validate_coverage(train_vace,val_vace,args.validation_manifest,args.allow_partial_validation)
    train=spread_subset(V10Dataset(train_vace,train_hmr),args.max_train_windows)
    val=spread_subset(V10Dataset(val_vace,val_hmr),args.max_val_windows)
    train_pose=(ShuffledPoseDatasetV10(args.pose_train_vace_dir,args.pose_train_hmr_dir,args.seed,args.max_pose_train_windows)
                if pose_training_enabled(args) else None)
    val_pose=(ShuffledPoseDatasetV10(args.pose_val_vace_dir,args.pose_val_hmr_dir,args.seed,args.max_pose_val_windows)
              if not args.geometry_only_validation else None)
    if (train_pose is not None and train_pose.split!='train') or (val_pose is not None and val_pose.split!='val'):
        raise ValueError('Incorrect PoseTrack split')
    train_loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed),pin_memory=True)
    val_loader=DataLoader(val,batch_size=args.val_batch_size,shuffle=False,num_workers=args.num_workers)
    pose_loader=(DataLoader(train_pose,batch_size=args.pose_batch_size,num_workers=0,
        generator=torch.Generator().manual_seed(args.seed+1)) if train_pose is not None else None)
    pose_val_loader=DataLoader(val_pose,batch_size=args.val_batch_size,num_workers=0) if val_pose is not None else None
    payload=torch.load(args.init_checkpoint,map_location='cpu',weights_only=False)
    config=AdaptationConfig(hidden_dim=args.model_width,num_heads=args.model_heads,depth=args.model_depth,
        iterations=args.iterations,shape_bound=args.shape_bound,use_temporal=not args.disable_temporal,
        use_local_feedback=not args.disable_local_feedback,use_vace=not args.disable_vace,
        frame_shape_delta_scale=args.frame_shape_delta_scale,local_radius_schedule=args.local_radius_schedule)
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
        reference_spec=model.reference_spec,
        selection_split='3DPW validation only' if args.geometry_only_validation else '3DPW validation + PoseTrack val',
        coco_training_enabled=coco_loader is not None,
        initialization=str(args.init_checkpoint),source_format=payload['format'],source_epoch=payload.get('epoch'),
        trainable_parameters={name:p.numel() for name,p in model.named_parameters() if p.requires_grad},
        trainable_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
        frozen_student_count=sum(p.numel() for p in model.parameters() if not p.requires_grad),
        coverage=coverage,development_only=development,test_used_for_model_selection=False,
        training_windows_3dpw=len(train),training_windows_posetrack=len(train_pose) if train_pose is not None else 0,
        validation_windows_3dpw=len(val),validation_windows_posetrack=len(val_pose) if val_pose is not None else 0,
        posetrack_training_enabled=train_pose is not None,posetrack_validation_enabled=val_pose is not None,
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    source_hash=hashlib.sha256()
    with args.init_checkpoint.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            source_hash.update(block)
    metadata['source_sha256']=source_hash.hexdigest()
    metadata['source_files_sha256']=source_fingerprints()
    metadata['asset_manifest']=asset_audit['assets']
    metadata['cache_hash_policy']=asset_audit['cache_hash_policy']
    metadata['environment']=dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,
        numpy=np.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        gpu_name=torch.cuda.get_device_name(device) if device.type=='cuda' else None)
    metadata['geometry_loss_source']='divibody_training.v31_boundary_partition -> ram_parts_runtime_v31'
    metadata['evaluation_protocol']=dict(dataset='3DPW',selection_split='validation',
        joint_set='H36M-regressed J14',joint_root='H36M joint zero pelvis',
        pve_alignment='direct model-centred vertices; no regressed-pelvis removal',
        prediction_gender='neutral',target_gender='annotation male/female',
        acceleration_root='J14 hip midpoint',acceleration_unit='mm/annotation-frame^2',
        prediction_window_frames=8,vace_context_frames=17,
        bbox_provenance='annotation-derived person crops; verify cached preprocessing signatures',
        historical_test_exposure=True,sota_status='not_established')
    (args.output_dir/('resume_config.json' if args.resume_checkpoint else 'run_config.json')).write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(dict(event='initialized',trainable_count=metadata['trainable_count'],
        frozen_student_count=metadata['frozen_student_count'],
        development_only=development,train_windows=len(train),pose_train_windows=len(train_pose) if train_pose is not None else 0)),flush=True)
    def evaluate(network):
        geometry=validate(val_loader,network,models,regressor,args,device)
        print(json.dumps(dict(event='geometry_validation_done',metrics=geometry)),flush=True)
        pose=(dict(projected_PCKn05_unused=0.,**{'projected_PCKn@0.05':0.,'projected_PCKn@0.10':0.,
                    'projected_mean_normalized_joint_error':0.}) if args.geometry_only_validation
              else evaluate_pose(pose_val_loader,network,official_smpl,args,device))
        return geometry,pose
    execute_run(model,optimizer,ema,train_loader,pose_loader,coco_loader,
                models,official_smpl,regressor,evaluate,args,device,metadata)

if __name__=='__main__':
    main()
