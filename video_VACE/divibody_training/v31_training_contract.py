"""Portable V31 CLI and asset/source manifests; no torch or CUDA imports.

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
    args=p.parse_args(argv)
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
        p.error('Legacy part-angle sweeps do not apply to V31')
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
        'video_VACE/divibody_training/v31_training_contract.py',
        'video_VACE/divibody_training/train_spatiotemporal_v31.py',
        'video_VACE/divibody_training/v31_training_engine.py',
        'video_VACE/divibody_training/ram_parts_runtime_v31.py',
        'video_VACE/divibody_training/parts_training_utils_v31.py',
        'video_VACE/divibody_training/v31_boundary_partition.py',
        'video_VACE/models/human_decoder_spatiotemporal_v31.py',
        'video_VACE/models/human_decoder_parts_v29.py',
        'video_VACE/models/human_decoder_parts_v28.py',
        'video_VACE/divibody_training/v29_boundary_training.py',
        'video_VACE/divibody_training/v29_geometry_policy_stable.py',
        'video_VACE/divibody_training/v29_sequence_evaluation_report.py',
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
        'scripts/train_from_config_v31.py',
    ]
    return {name:file_fingerprint(root/name)['sha256'] for name in names}
