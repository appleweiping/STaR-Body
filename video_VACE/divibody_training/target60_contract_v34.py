"""Portable V34 CLI and asset/source manifests; no torch or CUDA imports.

Parser derived from a preserved copy of train_spatiotemporal_v30_v3.py.
Content SHA256 checks on feature caches are opt-in because full caches are large.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def pose_training_enabled(args):
    return not args.eval_only and args.pose_2d_weight > 0 and args.geometry_only_steps < args.max_steps


def parse_radius_schedule(value):
    try:
        values=tuple(float(v.strip()) for v in value.split(',')) if value else ()
    except ValueError as error:
        raise argparse.ArgumentTypeError('Expected comma-separated positive radii') from error
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise argparse.ArgumentTypeError('Local radii must be finite and positive')
    return values


def parse_training_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('train-vace-cache','train-hmr-sidecar','val-vace-cache','val-hmr-sidecar',
        'init-checkpoint','smpl-model-root','joint-regressor','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('pose-train-vace-dir','pose-train-hmr-dir','pose-val-vace-dir','pose-val-hmr-dir'):
        p.add_argument('--'+name,type=Path)
    for name,value in dict(epochs=15,batch_size=4,pose_batch_size=8,val_batch_size=16,
        num_workers=2,seed=2026,patience=6,min_epochs=8,max_train_windows=0,max_val_windows=0,
        max_pose_train_windows=0,max_pose_val_windows=0,diagnostic_every=100).items():
        p.add_argument('--'+name.replace('_','-'),type=int,default=value)
    for name,value in dict(rgb_lr=1e-6,vace_lr=1e-6,heads_lr=1e-6,camera_lr=3e-6,
        gate_lr=1e-5,local_lr=1e-5,geometry_lr=2e-5,weight_decay=1e-4,grad_clip=1.,warmup_epochs=1.,
        ema_decay=.995,pose_2d_weight=.02,pose_deadzone=.01,pose_projected_weight=.5,
        pose_confidence_weight=.02,joint_weight=1.,vertex_weight=.5,
        stage_2d_weight=.5,observation_weight=.5,uncertainty_weight=.01,
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
    p.add_argument('--model-width',type=int,default=256)
    p.add_argument('--model-heads',type=int,default=8)
    p.add_argument('--model-depth',type=int,default=4)
    p.add_argument('--disable-temporal',action='store_true')
    p.add_argument('--disable-local-feedback',action='store_true')
    p.add_argument('--disable-vace',action='store_true')
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
    p.add_argument('--shape-bound',type=float,default=3.)
    p.add_argument('--mpjpe-tolerance-mm',type=float,default=0.)
    p.add_argument('--pa-mpjpe-tolerance-mm',type=float,default=0.)
    p.add_argument('--strict-3d-selection',action='store_true')
    p.add_argument('--geometry-only-validation',action='store_true')
    p.add_argument('--acceleration-ceiling-mm',type=float,required=True)
    p.add_argument('--boundary-weight',type=float,default=.25)
    p.add_argument('--boundary-triplets-per-step',type=int,default=2)
    p.add_argument('--frame-shape-delta-scale',type=float,default=1.)
    p.add_argument('--local-radius-schedule',type=parse_radius_schedule,default=())
    p.add_argument('--deduplicate-shape-consistency',action='store_true')
    p.add_argument('--verify-cache-sha256',action='store_true')
    p.set_defaults(pose_2d_weight=0.,coco_weight=0.,geometry_only_validation=True)
    p.add_argument('--shared-track-shape',action='store_true')
    p.add_argument('--enable-memory',action='store_true')
    p.add_argument('--enable-motion',action='store_true')
    p.add_argument('--memory-horizon',type=int,default=32)
    p.add_argument('--memory-topk',type=int,default=8)
    p.add_argument('--memory-mode',choices=('offline','past_memory'),default='offline')
    p.add_argument('--shape-prototype-frames',type=int,default=8)
    p.add_argument('--feedback-dropout',type=float,default=0.)
    p.add_argument('--motion-depth',type=int,default=4)
    p.add_argument('--motion-lr',type=float,default=1e-4)
    p.add_argument('--motion-aux-weight',type=float,default=.05)
    p.add_argument('--gate-calibration-weight',type=float,default=.01)
    p.add_argument('--motion-pretrained',type=Path)
    p.add_argument('--freeze-motion',action='store_true')
    p.add_argument('--gradient-accumulation',type=int,default=1)
    p.add_argument('--derive-acceleration-ceiling',action='store_true')
    p.add_argument('--max-cuda-memory-gib',type=float,default=10.)
    p.add_argument('--enable-track-shape',action='store_true')
    p.add_argument('--enable-gru-refinement',action='store_true')
    p.add_argument('--track-shape-bound',type=float,default=.3)
    p.add_argument('--gru-width',type=int,default=128)
    p.add_argument('--refinement-steps',type=int,default=3)
    p.add_argument('--root-step-degrees',type=float,default=2.)
    p.add_argument('--body-step-degrees',type=float,default=5.)
    p.add_argument('--freeze-base',action='store_true')
    p.add_argument('--unfreeze-camera',action='store_true')
    p.add_argument('--adapter-warmup-steps',type=int,default=0)
    p.add_argument('--pa-weight',type=float,default=0.)
    p.add_argument('--adapter-lr',type=float,default=5e-5)
    p.add_argument('--target-mpjpe-mm',type=float,default=60.)
    p.add_argument('--target-pa-mpjpe-mm',type=float,default=37.3)
    p.add_argument('--acceleration-tolerance-mm',type=float,default=0.)
    p.add_argument('--initial-reference-report',type=Path)
    p.add_argument('--initial-reproduction-atol-mm',type=float,default=.001)
    p.add_argument('--train-track-prototypes',type=Path)
    p.add_argument('--val-track-prototypes',type=Path)
    p.set_defaults(max_steps=6000,warmup_steps=200,geometry_lr=2e-5,local_lr=2e-5,
        camera_lr=2e-6,gradient_accumulation=16,batch_size=1,val_batch_size=2,
        max_cuda_memory_gib=3.5,geometry_only_steps=6001,coco_warmup_steps=0)
    args=p.parse_args(argv)
    if not 1<=args.gradient_accumulation<=32 or args.max_cuda_memory_gib<=0:
        p.error('Invalid accumulation or CUDA budget')
    if not 1<=args.memory_topk<=args.memory_horizon<=64 or not 1<=args.shape_prototype_frames<=16:
        p.error('Invalid memory/prototype dimensions')
    if not 1<=args.motion_depth<=8 or not 0<=args.feedback_dropout<1 or args.motion_lr<=0:
        p.error('Invalid motion/dropout settings')
    if args.motion_pretrained and not args.enable_motion:
        p.error('Motion pretrained weights require --enable-motion')
    if args.freeze_motion and (not args.enable_motion or not args.motion_pretrained):
        p.error('Frozen motion prior requires explicit pretrained motion weights')
    if args.pose_2d_weight or args.coco_weight or not args.geometry_only_validation:
        p.error('V34 first implementation is pure 3DPW geometry; auxiliary datasets need explicit track-context adapters')
    if args.enable_memory or args.enable_motion or args.shared_track_shape or args.motion_pretrained or args.freeze_motion:
        p.error('V34 memory/motion/absolute-shape settings do not apply to V34')
    if args.freeze_base and not (args.enable_track_shape or args.enable_gru_refinement):
        p.error('Frozen V30 requires a new adapter')
    if args.enable_track_shape and (not args.train_track_prototypes or not args.val_track_prototypes):
        p.error('Track shape requires audited train/validation V30 prediction prototypes')
    if not 0<=args.pa_weight<=1 or not 32<=args.gru_width<=512 or not 1<=args.refinement_steps<=4:
        p.error('Invalid PA/GRU settings')
    if min(args.track_shape_bound,args.root_step_degrees,args.body_step_degrees,args.adapter_lr,args.target_mpjpe_mm,args.target_pa_mpjpe_mm)<=0:
        p.error('Positive Target60 bounds and rates required')
    if args.max_train_windows or args.boundary_triplets_per_step<1:
        p.error('Boundary training requires complete train dataset and positive triplet batch')
    if args.max_steps<1 or args.eval_every<1:
        p.error('Positive step budget and evaluation period required')
    if args.resume_checkpoint and args.resume_checkpoint.resolve() != (args.output_dir/'latest.pt').resolve():
        p.error('Resume uses latest.pt in the same run directory')
    for key,value in vars(args).items():
        if isinstance(value,(float,int)) and (not math.isfinite(value) or value < 0):
            p.error(f'{key} must be finite and nonnegative')
    if min(args.epochs,args.batch_size,args.pose_batch_size,args.val_batch_size,args.patience,args.grad_clip) <= 0:
        p.error('Epochs, batches, patience and clipping must be positive; auxiliary pose weight may be zero')
    if not 0 <= args.ema_decay < 1:
        p.error('EMA decay must be in [0,1)')
    if args.eval_part_sweep or args.eval_part_angles:
        p.error('Legacy part-angle sweeps do not apply to V34')
    if not 0 <= args.frame_shape_delta_scale <= 1:
        p.error('Frame shape delta scale must be in [0,1]')
    if args.local_radius_schedule and len(args.local_radius_schedule) != args.iterations:
        p.error('Local radius schedule length must equal iterations')
    if args.model_width < 32 or args.model_heads < 1 or args.model_width % 4 or args.model_width % args.model_heads:
        p.error('Invalid model width or attention heads')
    if not 1 <= args.model_depth <= 8 or args.shape_bound <= 0:
        p.error('Invalid model depth or shape bound')
    if pose_training_enabled(args) and any(getattr(args,key) is None for key in ('pose_train_vace_dir','pose_train_hmr_dir')):
        p.error('Enabled PoseTrack training requires its VACE and HMR directories')
    if not args.geometry_only_validation and any(getattr(args,key) is None for key in ('pose_val_vace_dir','pose_val_hmr_dir')):
        p.error('PoseTrack validation requires its VACE and HMR directories')
    return args


def file_fingerprint(path, content=True):
    path=Path(path).expanduser().resolve()
    stat=path.stat()
    result=dict(path=str(path),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
    if content:
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):
                digest.update(block)
        result['sha256']=digest.hexdigest()
        # Timestamp is irrelevant when contents are fully bound.
        result.pop('mtime_ns')
    return result


def inspect_assets(args, fingerprint=False):
    required={name:Path(getattr(args,name)) for name in
              ('train_vace_cache','train_hmr_sidecar','val_vace_cache','val_hmr_sidecar',
               'init_checkpoint','joint_regressor')}
    for gender in ('NEUTRAL','MALE','FEMALE'):
        required['smpl_'+gender.lower()]=args.smpl_model_root/('SMPL_'+gender+'.pkl')
    required['smpl_to_j19']=args.smpl_model_root.parent/'SMPL_to_J19.pkl'
    for name in ('train_track_prototypes','val_track_prototypes','initial_reference_report'):
        path=getattr(args,name)
        if path is not None:required[name]=path
    if args.resume_checkpoint:
        required['resume_checkpoint']=args.resume_checkpoint
    if args.validation_manifest:
        required['validation_manifest']=args.validation_manifest
    elif not args.allow_partial_validation:
        required['validation_manifest']=None
    if pose_training_enabled(args):
        for key in ('pose_train_vace_dir','pose_train_hmr_dir'):
            required[key+'_manifest']=getattr(args,key)/'manifest.json'
    if not args.geometry_only_validation:
        for key in ('pose_val_vace_dir','pose_val_hmr_dir'):
            required[key+'_manifest']=getattr(args,key)/'manifest.json'
    if args.coco_train_manifest:
        required['coco_train_manifest']=args.coco_train_manifest
    errors=[]
    # Manifest contents are read as data, and all declared files must exist.
    for key,path in list(required.items()):
        if path is None or not path.is_file():
            continue
        if key.endswith('_manifest'):
            try:
                data=json.loads(path.read_text(encoding='utf-8'))
                if key=='validation_manifest' and data.get('independent_archive_verified'):
                    archive=Path(data['archive_path']).expanduser()
                    if not archive.is_absolute():
                        errors.append('Independent validation archive_path must be absolute to match the preserved evaluator')
                    required['validation_annotation_archive']=archive
                if key.startswith('pose_'):
                    for index,row in enumerate(data['shards']):
                        required[f'{key}_shard_{index}']=path.parent/row['file']
                if key=='coco_train_manifest':
                    for index,item in enumerate(data['files']):
                        required[f'coco_cache_{index}']=path.parent/item
            except (ValueError,KeyError,TypeError,OSError) as error:
                errors.append(f'Invalid manifest {key}: {type(error).__name__}: {error}')
    missing={key:str(path) if path is not None else None for key,path in required.items()
             if path is None or not path.is_file()}
    if args.output_dir.exists() and not args.output_dir.is_dir():
        errors.append('Output path exists and is not a directory')
    elif not args.resume_checkpoint and args.output_dir.is_dir() and any(args.output_dir.iterdir()):
        errors.append('Use a new empty output directory; earlier runs are preserved')
    assets={}
    if fingerprint and not missing and not errors:
        for key,path in required.items():
            if key=='resume_checkpoint':
                continue  # This mutable latest.pt is validated and restored separately.
            cache=(key.endswith('_cache') or key.endswith('_sidecar') or '_shard_' in key or key.startswith('coco_cache_'))
            assets[key]=file_fingerprint(path,content=args.verify_cache_sha256 or not cache)
    return dict(passed=not missing and not errors,missing=missing,errors=errors,
                required_paths={key:str(path) if path is not None else None for key,path in required.items()},
                cache_hash_policy='sha256' if args.verify_cache_sha256 else 'size_mtime_only',assets=assets,
                note='Path/schema preflight only; tensor identity, numerical and GPU checks remain required')


def source_fingerprints():
    root=Path(__file__).resolve().parents[2]
    names=[
        'video_VACE/divibody_training/target60_contract_v34.py',
        'video_VACE/divibody_training/train_target60_v34.py',
        'video_VACE/divibody_training/target60_engine_v34.py',
        'video_VACE/divibody_training/target60_runtime_v34.py',
        'video_VACE/divibody_training/target60_optim_v34.py',
        'video_VACE/divibody_training/target60_partition_v34.py',
        'video_VACE/models/human_decoder_target60_v34.py',
        'video_VACE/models/human_decoder_parts_v29.py',
        'video_VACE/models/human_decoder_parts_v28.py',
        'video_VACE/divibody_training/target60_boundary_v34.py',
        'video_VACE/divibody_training/v29_geometry_policy_stable.py',
        'video_VACE/divibody_training/target60_validation_v34.py',
        'video_VACE/divibody_training/v29_experiment_utils.py',
        'video_VACE/divibody_training/ram_parts_runtime_v29.py',
        'video_VACE/divibody_training/ram_parts_runtime_v27.py',
        'video_VACE/divibody_training/ram_pose_stream_v10.py',
        'video_VACE/divibody_training/train_divibody_rgb_vace_hmr2a_v8.py',
        'video_VACE/divibody_training/cache_hmr2a_rgb_anchor_v8.py',
        'video_VACE/divibody_training/train_divibody_continuous_geometry_v5.py',
        'video_VACE/divibody_training/train_divibody_real_3dpw_geometry.py',
        'video_VACE/divibody_training/geometry_metrics_3dpw.py',
        'video_VACE/divibody_training/evaluate_standard_3dpw.py',
        'scripts/train_from_config_v34.py',
    ]
    names.extend(['video_VACE/divibody_training/target60_track_dataset_v34.py',
                  'video_VACE/divibody_training/target60_pa_loss_v34.py','video_VACE/divibody_training/target60_policy_v34.py',
                  'video_VACE/models/human_decoder_spatiotemporal_v30.py',
                  'scripts/train_from_config_v34.py',
                  'video_VACE/models/star_memory_modules_v33.py',
                  'video_VACE/divibody_training/target60_track_dataset_v34.py'])
    return {name:file_fingerprint(root/name)['sha256'] for name in names}
