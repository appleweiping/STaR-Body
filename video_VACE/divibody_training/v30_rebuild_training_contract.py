"""Portable unchanged-V30 rebuild CLI and asset/source manifests.

Copied from the V31 infrastructure contract; no V31 model/loss switches remain.
Formal runs bind a complete V28 validation report and full cache fingerprints.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def pose_training_enabled(args):
    return not args.eval_only and args.pose_2d_weight > 0 and args.geometry_only_steps < args.max_steps


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
    p.add_argument('--acceleration-ceiling-mm',type=float,
                   help='Diagnostic-only manual value; formal runs derive this from --baseline-report')
    p.add_argument('--baseline-report',type=Path,
                   help='Complete matched V28 validation JSON from evaluate_body_v31.py')
    p.add_argument('--boundary-weight',type=float,default=.25)
    p.add_argument('--boundary-triplets-per-step',type=int,default=2)
    p.add_argument('--verify-cache-sha256',action='store_true')
    args=p.parse_args(argv)
    if args.baseline_report:
        try:
            report=json.loads(args.baseline_report.read_text(encoding='utf-8-sig'))
            if report.get('model_version') != 'v28' or report.get('split') != 'validation':
                raise ValueError('Baseline must be a V28 validation evaluation, never test')
            ceiling=float(report['metrics']['acceleration_error_mm_per_frame2'])
            if not math.isfinite(ceiling) or ceiling < 0:
                raise ValueError('Baseline acceleration must be finite and nonnegative')
            if args.acceleration_ceiling_mm is not None and args.acceleration_ceiling_mm != ceiling:
                raise ValueError('Supplied ceiling differs from the bound V28 validation report')
            args.acceleration_ceiling_mm=ceiling
        except (OSError,ValueError,KeyError,TypeError) as error:
            p.error('Invalid baseline report: '+str(error))
    elif not args.allow_partial_validation:
        p.error('Formal V30 rebuild requires --baseline-report; a historical acceleration constant is insufficient')
    elif args.acceleration_ceiling_mm is None:
        p.error('Diagnostic partial validation needs an explicit acceleration ceiling or V28 baseline report')
    if not args.allow_partial_validation and not args.verify_cache_sha256:
        p.error('Formal matched rebuilding requires --verify-cache-sha256 to bind cache contents')
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
        p.error('Legacy part-angle sweeps do not apply to V30 rebuild')
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
    if args.baseline_report:
        required['baseline_report']=args.baseline_report
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


def verify_baseline_binding(args, asset_audit, coverage, validation_payload):
    """Bind the acceleration ceiling to complete V28 validation on these inputs."""
    if args.baseline_report is None:
        if not args.allow_partial_validation:
            raise ValueError('Formal V30 rebuild requires a matched V28 validation report')
        return dict(complete=False,diagnostic_only=True,source='explicit diagnostic ceiling',
                    acceleration_ceiling_mm=args.acceleration_ceiling_mm)
    report=json.loads(args.baseline_report.read_text(encoding='utf-8-sig'))
    if report.get('model_version')!='v28' or report.get('split')!='validation':
        raise ValueError('The baseline must be V28 validation; test cannot define model selection')
    metrics=report['metrics']
    reference=report.get('coverage',{}).get('manifest') or {}
    complete=(report.get('status')=='complete_matched_project_protocol'
              and metrics.get('frames')==10413 and metrics.get('acceleration_triplets')==10226
              and reference.get('exact_uid_match') is True
              and reference.get('independent_archive_verified') is True
              and coverage.get('complete') is True)
    if not args.allow_partial_validation and not complete:
        raise ValueError('Baseline and current validation must both cover all 10413 frames and 10226 triplets')
    ceiling=metrics['acceleration_error_mm_per_frame2']
    if not math.isfinite(ceiling) or args.acceleration_ceiling_mm != ceiling:
        raise ValueError('The acceleration ceiling changed after parsing the baseline report')
    if report.get('optimizer_steps') != 0 or report.get('model_or_teacher_fitting_performed') is not False:
        raise ValueError('Baseline report must be evaluation-only')
    required_protocol={'mpjpe':'H36M pelvis aligned',
                       'pa_mpjpe':'per-frame Procrustes similarity aligned J14',
                       'pve':'unaligned SMPL vertex Euclidean error',
                       'acceleration_unit':'mm/frame^2'}
    if any(report.get('protocol',{}).get(key)!=value for key,value in required_protocol.items()):
        raise ValueError('Baseline metric protocol differs from the preserved V30 sequence evaluator')
    if report.get('checkpoint',{}).get('format')!='ram-iterative-anchor-routed-v28':
        raise ValueError('Baseline checkpoint is not a native V28 checkpoint')
    if report['checkpoint'].get('weights_field')!='model':
        raise ValueError('Baseline must evaluate the same model field used for V28 initialization')
    if not args.allow_partial_validation and report.get('resources',{}).get('amp_dtype')!=args.amp_dtype:
        raise ValueError('Formal baseline and training-time validation must use the same AMP mode')
    current=asset_audit['assets']; baseline=report['assets']
    pairs=[('init_checkpoint',baseline['checkpoint']),('val_vace_cache',baseline['vace_cache']),
           ('val_hmr_sidecar',baseline['hmr_sidecar']),('joint_regressor',baseline['joint_regressor']),
           ('smpl_to_j19',baseline['joint_regressor_extra'])]
    pairs.extend(('smpl_'+gender.lower(),baseline['smpl'][gender]) for gender in ('NEUTRAL','MALE','FEMALE'))
    matched={}
    for name,previous in pairs:
        actual=current[name]
        if actual.get('sha256') and previous.get('sha256'):
            same=actual['sha256']==previous['sha256'] and actual['bytes']==previous['bytes']
        elif args.allow_partial_validation:
            same=all(actual.get(key)==previous.get(key) for key in ('path','bytes','mtime_ns'))
        else:
            raise ValueError('Formal baseline needs full cache SHA256; evaluate V28 with --hash-large-caches')
        if not same:
            raise ValueError('V28 baseline and V30 rebuild asset differ: '+name)
        matched[name]=actual.get('sha256')
    owners=validation_payload['frame_owner_mask'].tolist()
    uids=[uid for row,record in enumerate(validation_payload['metadata'])
          for uid,owned in zip(record['target_frame_uids'],owners[row]) if owned]
    unique=set(uids)
    triplets=sum(f'{uid.rsplit("/frame_",1)[0]}/frame_{int(uid.rsplit("/frame_",1)[1])-1:05d}' in unique
                 and f'{uid.rsplit("/frame_",1)[0]}/frame_{int(uid.rsplit("/frame_",1)[1])+1:05d}' in unique
                 for uid in unique)
    if len(unique)!=len(uids) or (not args.allow_partial_validation and (len(uids),triplets)!=(10413,10226)):
        raise ValueError('Current validation owned UIDs do not have the complete required frame/triplet coverage')
    identity=hashlib.sha256(('\n'.join(sorted(uids))+'\n').encode()).hexdigest()
    if report['coverage']['owned_uid_sha256']!=identity:
        raise ValueError('V28 baseline and current validation owned UID sets differ')
    return dict(complete=complete,diagnostic_only=not complete,
                report=file_fingerprint(args.baseline_report),matched_asset_sha256=matched,
                validation_owned_uid_sha256=identity,acceleration_ceiling_mm=ceiling,
                checkpoint_selection_split='validation',test_used_for_selection=False)


def source_fingerprints():
    root=Path(__file__).resolve().parents[2]
    names=[
        'video_VACE/divibody_training/v30_rebuild_training_contract.py',
        'video_VACE/divibody_training/train_spatiotemporal_v30_rebuild.py',
        'video_VACE/divibody_training/v30_rebuild_training_engine.py',
        'video_VACE/divibody_training/ram_parts_runtime_v30.py',
        'video_VACE/divibody_training/parts_training_utils_v30.py',
        'video_VACE/divibody_training/v29_boundary_partition.py',
        'video_VACE/models/human_decoder_spatiotemporal_v30.py',
        'video_VACE/models/human_decoder_parts_v29.py',
        'video_VACE/models/human_decoder_parts_v28.py',
        'video_VACE/models/human_decoder_parts_v27.py',
        'video_VACE/models/human_decoder_rgb_vace_v9_isolated.py',
        'video_VACE/models/human_decoder_rgb_vace_v7.py',
        'video_VACE/models/human_decoder_geometry.py',
        'video_VACE/models/human_decoder.py',
        'video_VACE/models/body_projection_v28.py',
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
        'video_VACE/divibody_training/train_divibody_real_3dpw_temporal.py',
        'video_VACE/divibody_training/v28d_gpu_entry.py',
        'video_VACE/divibody_training/geometry_metrics_3dpw.py',
        'video_VACE/divibody_training/evaluate_standard_3dpw.py',
        'scripts/train_from_config_v30_rebuild.py',
    ]
    return {name:file_fingerprint(root/name)['sha256'] for name in names}
