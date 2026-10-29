"""Portable owned-frame V28/V30/V31/V33 evaluation; no optimization or label-based selection.



Copied from evaluate_parts_v28.py before editing. Formal evaluation requires an

independently verified UID manifest and full expected coverage. A small cache is

only accepted with --development-only and always reported as diagnostic_subset.

Native V30-to-V31 checkpoint migration is unsupported; use a native checkpoint

matching the selected model or explicitly request diagnostic V28 initialization.

"""

from __future__ import annotations



import argparse

from dataclasses import asdict

from datetime import datetime, timezone

import hashlib

import importlib

import json

import math

from pathlib import Path

import platform

import random

import re

import sys

import time



sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]

for path in (ROOT / 'video_VACE', ROOT / 'video_VACE/divibody_training', ROOT):

    sys.path.insert(0, str(path))



FORMATS = {'v36':'star-body-joint-feedback-v37-r1','v34':'star-body-target60-v34','v33':'star-body-reliable-memory-v33','v28': 'ram-iterative-anchor-routed-v28',

           'v30': 'spatial-dual-order-smpl-feedback-v30',

           'v31': 'spatial-dual-order-smpl-feedback-v31'}

EXPECTED = {'validation': (10413, 10226), 'test': (35515, 34752)}

UID_PATTERN = re.compile(r'^([^/]+)/person_(\d+)/frame_(\d{5,})$')





def sha256(path):

    digest = hashlib.sha256()

    with Path(path).open('rb') as stream:

        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):

            digest.update(chunk)

    return digest.hexdigest()





def uid_hash(uids):

    return hashlib.sha256(('\n'.join(sorted(uids)) + '\n').encode()).hexdigest()





def asset_record(path, full_hash=True):

    path = Path(path).expanduser().resolve()

    stat = path.stat()

    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,

                sha256=sha256(path) if full_hash else None,

                hash_scope='full_file' if full_hash else 'not_hashed_use_uid_and_signature_audit')





def parse_args():

    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument('--model-version', choices=tuple(FORMATS), required=True)

    p.add_argument('--checkpoint', type=Path, required=True)

    p.add_argument('--track-prototypes',type=Path,help='V34 prediction-only prototype pack for this split')

    p.add_argument('--checkpoint-weights', choices=('model', 'ema_model'), default='model',

                   help='Explicit state field; best.pt normally already stores the selected variant in model.')

    p.add_argument('--allow-initialization', action='store_true',

                   help='Allow a V28 checkpoint to initialize V30/V31 only with --development-only.')

    p.add_argument('--initial-config',type=Path,help='V33 AdaptationConfig JSON, only for explicit diagnostic V28 initialization')

    p.add_argument('--split', choices=('train', 'validation', 'test'), required=True)

    p.add_argument('--vace-cache', type=Path, required=True)

    p.add_argument('--hmr-sidecar', '--hmr-cache', dest='hmr_sidecar', type=Path, required=True)

    p.add_argument('--gt-cache', type=Path, help='Required for test; separate UID-indexed GT cache.')

    p.add_argument('--coverage-manifest', type=Path,

                   help='Independent full UID manifest: historical V28D validation reference or star-body-evaluation-reference-v1.')

    p.add_argument('--development-only', action='store_true',

                   help='Permit partial data and missing independent manifest; output remains diagnostic_subset.')

    p.add_argument('--smpl-model-root', type=Path, required=True)

    p.add_argument('--joint-regressor', type=Path, required=True)

    p.add_argument('--joint-regressor-extra', type=Path,

                   help='Default: sibling SMPL_to_J19.pkl of the SMPL model directory.')

    p.add_argument('--output', type=Path, required=True, help='New JSON file; existing files are never overwritten.')

    p.add_argument('--batch-size', type=int, default=8)

    p.add_argument('--num-workers', type=int, default=0)

    p.add_argument('--device', default='cuda:0')

    p.add_argument('--max-cuda-memory-gib', type=float, default=10.,

                   help='CUDA allocator budget in GiB, default 10; capped at the physical device capacity.')

    p.add_argument('--amp-dtype', choices=('bf16', 'none'), default='none')

    p.add_argument('--seed', type=int, default=20261008)

    p.add_argument('--hash-large-caches', action='store_true', help='Hash every byte of VACE/HMR/GT caches.')

    p.add_argument('--selection-lock',type=Path,help='Required immutable validation selection lock before test')

    args = p.parse_args()

    if args.output.exists():

        p.error('--output already exists; choose a new result path')

    if args.model_version=='v36' and args.split=='test' and args.selection_lock is None:

        p.error('V36 test requires a validation selection lock')

    if args.split == 'test' and args.gt_cache is None:

        p.error('--gt-cache is required for test')

    if not args.development_only and args.coverage_manifest is None:

        p.error('Formal evaluation requires --coverage-manifest; use --development-only for small diagnostic caches')

    if args.allow_initialization and not args.development_only:

        p.error('--allow-initialization requires --development-only')

    if args.batch_size < 1 or args.num_workers < 0:

        p.error('Invalid batch size or worker count')

    if not math.isfinite(args.max_cuda_memory_gib) or args.max_cuda_memory_gib <= 0:

        p.error('--max-cuda-memory-gib must be finite and positive')

    if args.initial_config and (args.model_version!='v33' or not args.allow_initialization or not args.development_only):

        p.error('--initial-config requires diagnostic V28-to-V33 initialization')

    return args





def verify_cache_provenance(vace, sidecar, vace_path, split, development_only):

    """Keep the original immutable feature contract and all declared hashes.



    Relocated inputs are never repaired by editing signatures. A declared full

    VACE hash remains mandatory even when optional large-file hashing is off.

    """

    import torch

    signature, side_signature = vace['signature'], sidecar['signature']

    formats = {'divibody-contiguous-vace-video-features-v5', 'divibody-contiguous-vace-video-features-v6-stride4'}

    if signature.get('format') not in formats or side_signature.get('format') != 'divibody-hmr2a-anchor-sidecar-v8':

        raise ValueError('Unsupported immutable VACE/HMR cache formats')

    if side_signature.get('hmr2a_checkpoint_sha256') != '2dcf79638109781d1ae5f5c44fee5f55bc83291c210653feead9b7f04fa6f20e':

        raise ValueError('HMR 2.0a pretrained checkpoint fingerprint differs from the audited contract')

    if not development_only and any(cache['signature'].get('development_only') for cache in (vace,sidecar)):

        raise ValueError('Formal evaluation refuses development-only cache declarations')

    count = len(vace['metadata'])

    if vace['features'].shape != (count,5,6,6,1536) or vace['features'].dtype != torch.bfloat16:

        raise ValueError('Expected actual BF16 VACE features [windows,5,6,6,1536]')

    frames = len(sidecar['frame_uids'])

    for key,shape in {'rgb_tokens':(frames,192,1280),'base_rotmat':(frames,24,3,3),

                      'base_betas':(frames,10),'base_camera':(frames,3)}.items():

        if sidecar[key].shape != shape or not sidecar[key].is_floating_point():

            raise ValueError('Unexpected HMR tensor layout: '+key)

    if sidecar['rgb_tokens'].dtype != torch.bfloat16:

        raise ValueError('Expected actual BF16 HMR spatial tokens')

    for key,width in (('target_valid_mask',8),('frame_owner_mask',8),('velocity_owner_mask',7),('acceleration_owner_mask',6)):

        if vace[key].shape != (count,width) or vace[key].dtype != torch.bool:

            raise ValueError('Unexpected VACE mask layout: '+key)

    if split == 'test':

        if not signature.get('feature_only') or signature.get('contains_targets') or 'targets' in vace:

            raise ValueError('Test VACE must be feature-only; supply separate UID-indexed GT')

    else:

        if signature.get('feature_only'):

            raise ValueError('Train/validation evaluation expects the original target-bearing cache')

        for key,shape in {'full_pose':(count,8,72),'betas':(count,8,10),'gender_code':(count,8),'cam_pose':(count,8,4,4)}.items():

            if vace['targets'][key].shape != shape:

                raise ValueError('Unexpected target layout: '+key)

    value = dict(signature=signature,window_uids=[record['window_uid'] for record in vace['metadata']],

                 target_frame_uids=[record['target_frame_uids'] for record in vace['metadata']])

    semantic = hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

    if side_signature.get('vace_semantic_sha256') != semantic:

        raise ValueError('VACE semantic fingerprint differs from the immutable HMR sidecar')

    declared_hash = side_signature.get('vace_cache_sha256')

    actual_hash = sha256(vace_path) if declared_hash else None

    if declared_hash and actual_hash != declared_hash:

        raise ValueError('Supplied VACE file differs from its declared HMR-sidecar SHA256')

    source_records = {}

    for label,declared in (('vace',signature),('hmr',side_signature)):

        for key,expected in declared.items():

            source_key = key[:-7] if key.endswith('_sha256') else ''

            if source_key.endswith('_source') and source_key in declared and expected:

                record = asset_record(declared[source_key])

                if record['sha256'] != expected:

                    raise ValueError('Declared source fingerprint differs: '+label+'.'+source_key)

                source_records[label+'.'+source_key] = record

    return dict(semantic_vace_sha256=semantic,declared_vace_file_sha256=declared_hash,

                verified_vace_file_sha256=actual_hash,declared_sources=source_records)





def audit_coverage(vace, sidecar, split, manifest_path, development_only):

    """Reject count-only matches; audit all window, frame, owner, and source IDs."""

    import torch

    if any(cache.get('signature', {}).get('split') != split for cache in (vace, sidecar)):

        raise ValueError('VACE and HMR signatures must match the requested split')

    metadata = vace['metadata']

    owners, valid = vace['frame_owner_mask'], vace['target_valid_mask']

    if owners.dtype != torch.bool or valid.dtype != torch.bool or owners.ndim != 2 or owners.shape != valid.shape:

        raise ValueError('Expected matching Boolean ownership and valid masks [windows,T]')

    if not metadata or len(metadata) != len(owners) or len(metadata) != len(vace['features']) or (owners & ~valid).any() or not valid.any(1).all():

        raise ValueError('Invalid window, ownership, or valid-frame contract')

    if [record['window_uid'] for record in metadata] != list(sidecar['window_uids']):

        raise ValueError('VACE/HMR window UID order mismatch')

    if len(set(sidecar['window_uids'])) != len(metadata):

        raise ValueError('Duplicated window UIDs')

    lookup = sidecar['window_frame_indices']

    if lookup.shape != owners.shape or lookup.dtype not in (torch.int32, torch.int64):

        raise ValueError('Invalid HMR window-frame lookup')

    frame_uids = list(sidecar['frame_uids'])

    if not frame_uids or len(set(frame_uids)) != len(frame_uids):

        raise ValueError('Empty or duplicated HMR frame UID index')

    if int(lookup.min()) < 0 or int(lookup.max()) >= len(frame_uids):

        raise ValueError('HMR frame lookup is out of bounds')

    for key in ('rgb_tokens', 'base_rotmat', 'base_betas', 'base_camera'):

        if len(sidecar[key]) != len(frame_uids):

            raise ValueError('HMR per-frame tensor count mismatch: '+key)

    owned_uids = []

    mapped_windows = 0

    for row, record in enumerate(metadata):

        uids = record['target_frame_uids']

        if len(uids) != owners.shape[1] or any(UID_PATTERN.fullmatch(uid) is None for uid in uids):

            raise ValueError('Malformed target frame UID or window length')

        if any(uid.split('/person_', 1)[0] != record['sequence'] for uid in uids):

            raise ValueError('Target UID sequence differs from metadata')

        if [frame_uids[int(index)] for index in lookup[row]] != list(uids):

            raise ValueError('HMR frame lookup and VACE target UID mismatch')

        annotation_ids = torch.as_tensor(record['target_frame_ids'])

        if (annotation_ids.shape != (owners.shape[1],) or annotation_ids.dtype not in (torch.int32, torch.int64)

                or annotation_ids.tolist() != [int(uid.rsplit('/frame_',1)[1]) for uid in uids]):

            raise ValueError('Annotation frame IDs differ from the target UIDs')

        explicit = record.get('target_source_frame_ids')

        if explicit is not None:

            mapped_windows += 1

        source = torch.as_tensor(explicit if explicit is not None else record['target_frame_ids'])

        if source.shape != (owners.shape[1],) or source.dtype not in (torch.int32, torch.int64):

            raise ValueError('Missing integer source/annotation frame IDs')

        owned_uids.extend(uid for uid, own in zip(uids, owners[row].tolist()) if own)

    if not owned_uids or len(set(owned_uids)) != len(owned_uids):

        raise ValueError('Owned frames must be nonempty and unique')

    owned = set(owned_uids)

    triplets = sum(f'{uid.rsplit("/frame_", 1)[0]}/frame_{int(uid.rsplit("/frame_", 1)[1])-1:05d}' in owned

                   and f'{uid.rsplit("/frame_", 1)[0]}/frame_{int(uid.rsplit("/frame_", 1)[1])+1:05d}' in owned

                   for uid in owned)

    manifest_info = None

    if manifest_path:

        manifest_path = Path(manifest_path)

        manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))

        if manifest.get('format') == 'v28d-validation-reference-v1' and split == 'validation':

            expected_list = manifest['validation_frame_uids']

        elif manifest.get('format') == 'star-body-evaluation-reference-v1' and manifest.get('split') == split:

            expected_list = manifest['frame_uids']

        else:

            raise ValueError('Coverage manifest format or split mismatch')

        if not expected_list or len(expected_list) != len(set(expected_list)) or any(UID_PATTERN.fullmatch(uid) is None for uid in expected_list):

            raise ValueError('Invalid independent UID reference')

        expected = set(expected_list)

        if owned - expected:

            raise ValueError('Cache contains owned UIDs outside the independent reference')

        complete = owned == expected

        if not development_only and not complete:

            raise ValueError('Owned UID set does not equal the full independent manifest')

        independent = manifest.get('independent_archive_verified') is True

        archive_record = None

        if independent:

            archive = Path(manifest['archive_path'])

            if not archive.is_absolute():

                archive = manifest_path.parent / archive

            archive_record = asset_record(archive)

            if archive_record['sha256'] != manifest['archive_sha256']:

                raise ValueError('Independent annotation archive fingerprint mismatch')

        if not development_only and not independent:

            raise ValueError('Formal UID reference must be independently verified against an annotation archive')

        manifest_info = dict(path=str(manifest_path.resolve()), sha256=sha256(manifest_path),

                             exact_uid_match=complete, independent_archive_verified=independent,

                             missing_reference_frames=len(expected-owned), archive=archive_record)

    if not development_only:

        if split in EXPECTED and (len(owned), triplets) != EXPECTED[split]:

            raise ValueError(f'Incomplete {split}: frames/triplets={(len(owned),triplets)}, expected={EXPECTED[split]}')

        if manifest_info is None:

            raise ValueError('Missing independent full UID reference')

    return dict(windows=len(metadata), frames=len(owned), acceleration_triplets=triplets,

                sequences=len({uid.split('/person_', 1)[0] for uid in owned}),

                tracks=len({uid.rsplit('/frame_', 1)[0] for uid in owned}),

                owned_uid_sha256=uid_hash(owned), manifest=manifest_info,

                source_time_identity=dict(mapped_source_id_windows=mapped_windows,

                    annotation_id_fallback_windows=len(metadata)-mapped_windows)), owned





class EvaluationWindows:

    """Map cached features and labels separately; labels never enter model.forward."""

    def __init__(self, vace, sidecar, gt_cache, split):

        import torch

        self.vace, self.sidecar, self.gt = vace, sidecar, gt_cache

        self.gt_indices = None

        if gt_cache is not None:

            declared = gt_cache.get('signature', {}).get('split', gt_cache.get('split'))

            if declared is not None and declared != split:

                raise ValueError('GT cache split declaration mismatch')

            sequences, persons, frames = gt_cache['sequences'], gt_cache['person_id'], gt_cache['frame_id']

            if not (len(sequences) == len(persons) == len(frames)):

                raise ValueError('GT identity arrays have different lengths')

            uids = [f'{seq}/person_{int(pid)}/frame_{int(fid):05d}' for seq,pid,fid in zip(sequences,persons,frames)]

            if len(uids) != len(set(uids)):

                raise ValueError('Duplicate GT person-frame UIDs')

            for key in ('gt_full_pose','gt_betas','gt_gender','gt_cam_pose'):

                if len(gt_cache[key]) != len(uids):

                    raise ValueError('GT tensor and identity count mismatch: '+key)

            lookup = {uid:i for i,uid in enumerate(uids)}

            missing = set(sidecar['frame_uids']) - set(lookup)

            if missing:

                raise ValueError(f'GT misses {len(missing)} HMR frame UIDs')

            self.gt_indices = torch.tensor([lookup[uid] for uid in sidecar['frame_uids']], dtype=torch.long)

        else:

            for key in ('full_pose','betas','gender_code','cam_pose'):

                if len(vace['targets'][key]) != len(vace['metadata']):

                    raise ValueError('Window target count mismatch: '+key)



    def __len__(self):

        return len(self.vace['metadata'])



    def __getitem__(self, index):

        import torch

        frames = self.sidecar['window_frame_indices'][index]

        record = self.vace['metadata'][index]

        sample = {key:self.sidecar[key][frames] for key in ('rgb_tokens','base_rotmat','base_betas','base_camera')}

        sample.update(vace_features=self.vace['features'][index],

                      frame_owner=self.vace['frame_owner_mask'][index],target_valid=self.vace['target_valid_mask'][index],

                      source_frame_ids=torch.as_tensor(record.get('target_source_frame_ids')

                          if record.get('target_source_frame_ids') is not None else record['target_frame_ids'], dtype=torch.long))

        if self.gt_indices is None:

            sample.update({key:self.vace['targets'][key][index] for key in ('full_pose','betas','gender_code','cam_pose')})

        else:

            gt_frames = self.gt_indices[frames]

            sample.update({key:self.gt[gt_key][gt_frames] for key,gt_key in (

                ('full_pose','gt_full_pose'),('betas','gt_betas'),('gender_code','gt_gender'),('cam_pose','gt_cam_pose'))})

        return sample





def forward_batch(model, batch, device, amp_dtype):

    import torch

    if hasattr(model.adaptation_config,'use_track_shape'):

        from divibody_training.target60_runtime_v34 import forward_batch as contextual_forward

        return contextual_forward(model,batch,device,amp_dtype)

    if hasattr(model.adaptation_config,'memory_horizon'):

        from divibody_training.star_memory_runtime_v33 import forward_batch as contextual_forward

        return contextual_forward(model,batch,device,amp_dtype)

    keys = ('rgb_tokens','vace_features','base_rotmat','base_betas','base_camera')

    values = [batch[key].to(device, dtype=torch.float32 if amp_dtype=='none' else batch[key].dtype,

                            non_blocking=True) for key in keys]

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type=='cuda' and amp_dtype=='bf16'):

        return model(*values, frame_valid=batch['target_valid'].to(device),

                     source_frame_ids=batch['source_frame_ids'].to(device), landmark_schema='posetrack')





def main():

    args = parse_args()

    source_files = [Path(__file__),ROOT/'video_VACE/divibody_training/body_sequence_evaluation_v31.py',

                    ROOT/'video_VACE/divibody_training/geometry_metrics_3dpw.py',

                    ROOT/'video_VACE/divibody_training/evaluate_standard_3dpw.py',

                    ROOT/'video_VACE/divibody_training/train_divibody_continuous_geometry_v5.py',

                    ROOT/'video_VACE/divibody_training/train_divibody_real_3dpw_geometry.py',

                    ROOT/'video_VACE/divibody_training/train_divibody_real_3dpw_temporal.py',

                    ROOT/'video_VACE/models/body_projection_v28.py']

    source_files.extend(sorted((ROOT/'video_VACE/models').glob('human_decoder*.py')))

    source_files.extend(sorted((ROOT/'video_VACE/models').glob('star*_v33.py')))

    source_files.extend(sorted((ROOT/'video_VACE/divibody_training').glob('target60*_v34.py')))

    source_files.extend(sorted((ROOT/'video_VACE/divibody_training').glob('star*_v33.py')))

    source_files.extend(sorted((ROOT/'video_VACE').rglob('*v36.py')))

    missing_sources = [str(path) for path in source_files if not path.is_file()]

    if missing_sources:

        raise FileNotFoundError('Evaluation source preflight failed: '+', '.join(missing_sources))

    source_hashes = {str(path.relative_to(ROOT)):sha256(path) for path in source_files}

    import numpy as np

    import torch

    from torch.utils.data import DataLoader

    started = time.perf_counter()

    started_utc = datetime.now(timezone.utc).isoformat()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = torch.device(args.device)

    cuda_memory_fraction = None

    cuda_memory_budget_bytes = None

    if device.type == 'cuda':

        if not torch.cuda.is_available():

            raise RuntimeError('CUDA was requested but is unavailable')

        torch.cuda.manual_seed_all(args.seed)

        capacity = torch.cuda.get_device_properties(device).total_memory

        cuda_memory_budget_bytes = min(int(args.max_cuda_memory_gib * 1024**3), capacity)

        cuda_memory_fraction = cuda_memory_budget_bytes / capacity

        torch.cuda.set_per_process_memory_fraction(cuda_memory_fraction, device)

    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.deterministic = True

    if args.amp_dtype == 'bf16' and device.type != 'cuda':

        raise ValueError('bf16 option is CUDA-only; use --amp-dtype none on CPU')

    load = lambda path: torch.load(path, map_location='cpu', weights_only=False, mmap=True)

    vace, sidecar = load(args.vace_cache), load(args.hmr_sidecar)

    cache_provenance = verify_cache_provenance(vace, sidecar, args.vace_cache, args.split, args.development_only)

    coverage, expected_uids = audit_coverage(vace, sidecar, args.split, args.coverage_manifest, args.development_only)

    data = EvaluationWindows(vace, sidecar, load(args.gt_cache) if args.gt_cache else None, args.split)

    payload = load(args.checkpoint)

    checkpoint_format = payload.get('format')

    if args.model_version=='v36' and args.split=='test':

        lock=json.loads(args.selection_lock.read_text())

        if lock.get('selection_split')!='validation' or lock.get('checkpoint_sha256')!=sha256(args.checkpoint):

            raise ValueError('V36 selection lock does not bind this validation-selected checkpoint')

        if lock.get('v36_config')!=payload.get('v36_config') or lock.get('step')!=payload.get('step'):

            raise ValueError('V36 architecture/step differs from selection lock')

        if (lock.get('validation_frames'),lock.get('validation_triplets'))!=(10413,10226):

            raise ValueError('V36 selection lock lacks complete validation')

        if lock.get('checkpoint_weights')!=args.checkpoint_weights:raise ValueError('Locked weights variant differs')



    if args.model_version in ('v34','v36') and payload.get('development_only',True) and not args.development_only:

        raise ValueError('Development V34 checkpoint is diagnostic; formal benchmark requires full-data trained provenance')

    native_initial = args.model_version in ('v33','v34','v36') and (payload.get('initial_fallback_only',False) or payload.get('step',0)<=0)

    if native_initial and not args.development_only:

        raise ValueError('Native V33 initialization fallback is diagnostic, not a trained benchmark; use --development-only')

    initialization_only = (checkpoint_format == FORMATS['v28'] and args.model_version != 'v28') or (args.model_version=='v34' and checkpoint_format==FORMATS['v30'])

    accepted = {FORMATS[args.model_version]}

    if args.model_version == 'v31' and checkpoint_format == FORMATS['v30']:

        raise ValueError('Native V30-to-V31 checkpoint migration is unsupported; use native V31 or explicit diagnostic V28 initialization')

    if initialization_only and not args.allow_initialization:

        raise ValueError('V28-to-new-model initialization is diagnostic; explicitly pass --allow-initialization --development-only')

    if checkpoint_format not in accepted and not (initialization_only and args.allow_initialization):

        raise ValueError('Checkpoint format does not match the requested model version')

    if args.checkpoint_weights not in payload:

        raise ValueError('Requested checkpoint weight field is missing')

    selected_payload = payload | dict(model=payload[args.checkpoint_weights])

    module_name=('models.human_decoder_kinematic_v36' if args.model_version=='v36' else

                 'models.human_decoder_target60_v34' if args.model_version=='v34' else

                 'models.human_decoder_star_memory_v33' if args.model_version=='v33' else

                 'models.human_decoder_parts_v28' if args.model_version=='v28' else f'models.human_decoder_spatiotemporal_{args.model_version}')

    module=importlib.import_module(module_name)

    adaptation=None

    if args.initial_config:

        if not initialization_only:raise ValueError('Initial config cannot override a native checkpoint')

        adaptation=module.AdaptationConfig(**json.loads(args.initial_config.read_text(encoding='utf-8-sig')))

    model=module.model_from_payload(selected_payload,adaptation).to(device).eval()

    if args.model_version=='v33':

        from divibody_training.star_track_dataset_v33 import TrackContextDataset

        cfg=model.adaptation_config

        data=TrackContextDataset(data,cfg.memory_horizon,cfg.shape_prototype_frames,cfg.memory_mode,

            cfg.use_memory or cfg.use_motion or cfg.shared_track_shape)

    if args.model_version in ('v34','v36'):

        from divibody_training.target60_track_dataset_v34 import TrackPrototypeDataset

        cfg=model.adaptation_config

        if cfg.use_track_shape and args.track_prototypes is None:raise ValueError('V34 shape model needs audited split prototypes')

        data=TrackPrototypeDataset(data,args.track_prototypes if cfg.use_track_shape else None,

            payload.get('prototype_source_v30_sha256') if args.model_version=='v36' else payload.get('source_sha256'))

    # Imports retain the audited project's exact SMPL and metric conventions.

    from divibody_training.evaluate_standard_3dpw import create_smpl_models

    from divibody_training.body_sequence_evaluation_v31 import validate

    from hmr2.models.smpl_wrapper import SMPL

    smpl = create_smpl_models(args.smpl_model_root, device)

    for body in smpl.values():

        body.requires_grad_(False).eval()

    extra = args.joint_regressor_extra or args.smpl_model_root.parent/'SMPL_to_J19.pkl'

    official = SMPL(model_path=str(args.smpl_model_root), gender='neutral', num_betas=10,

                    joint_regressor_extra=str(extra)).to(device).requires_grad_(False).eval()

    model.bind_geometry(smpl[2]).bind_landmarks(official)

    regressor = torch.as_tensor(np.load(args.joint_regressor, allow_pickle=False), dtype=torch.float32, device=device)

    if regressor.shape != (17,6890) or not torch.isfinite(regressor).all():

        raise ValueError('Expected a finite H36M regressor [17,6890]')

    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, drop_last=False,

                        num_workers=args.num_workers, pin_memory=device.type=='cuda', persistent_workers=args.num_workers>0)

    metrics = validate(loader, model, smpl, regressor, args, device,

                       forward_batch=forward_batch, expected_uids=expected_uids)

    if (metrics['frames'],metrics['acceleration_triplets']) != (coverage['frames'],coverage['acceleration_triplets']):

        raise RuntimeError('Evaluated metric counts differ from the audited UID coverage')

    if not args.development_only and args.split in EXPECTED and (metrics['frames'],metrics['acceleration_triplets']) != EXPECTED[args.split]:

        raise RuntimeError('Formal split metric coverage is incomplete')

    resources = metrics.pop('resources')

    resources.update(total_parameter_count=sum(p.numel() for p in model.parameters()),

                     trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),

                     device=str(device),gpu_name=torch.cuda.get_device_name(device) if device.type=='cuda' else None,

                     torch=torch.__version__,cuda_runtime=torch.version.cuda,amp_dtype=args.amp_dtype,

                     cuda_allocator_memory_fraction=cuda_memory_fraction,

                     cuda_allocator_budget_bytes=cuda_memory_budget_bytes)

    assets = {key:asset_record(path, full_hash=args.hash_large_caches) for key,path in (

        ('vace_cache',args.vace_cache),('hmr_sidecar',args.hmr_sidecar))}

    if args.gt_cache:

        assets['gt_cache'] = asset_record(args.gt_cache, full_hash=args.hash_large_caches)

    assets.update(checkpoint=asset_record(args.checkpoint),joint_regressor=asset_record(args.joint_regressor),

                  joint_regressor_extra=asset_record(extra))

    assets['smpl'] = {gender:asset_record(args.smpl_model_root/f'SMPL_{gender}.pkl') for gender in ('NEUTRAL','MALE','FEMALE')}

    result = dict(status='diagnostic_subset' if args.development_only else 'complete_matched_project_protocol',

        model_version=args.model_version,split=args.split,seed=args.seed,started_utc=started_utc,

        model_config=asdict(model.config),adaptation_config=asdict(model.adaptation_config),

        v36_config=asdict(model.v36_config) if hasattr(model,'v36_config') else None,

        elapsed_seconds=time.perf_counter()-started,metrics=metrics,coverage=coverage,resources=resources,

        checkpoint=dict(format=checkpoint_format,weights_field=args.checkpoint_weights,

            saved_variant=payload.get('variant'),step=payload.get('step'),epoch=payload.get('epoch'),

            initialization_only=initialization_only or native_initial,native_v30_to_v31_migration_supported=False,

            development_only=payload.get('development_only'),saved_selection_rule=payload.get('selection_metric'),

            saved_training_seed=payload.get('args',{}).get('seed'),

            saved_initialization=payload.get('initialization'),saved_initialization_sha256=payload.get('source_sha256'),

            saved_selection_split=payload.get('selection_split'),

            rule_for_this_run='Explicit supplied checkpoint and weight field; no checkpoint/hypothesis selection'),

        environment=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,

            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),

            cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,

            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32),

        config={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},

        cache_signatures={'vace':vace['signature'],'hmr':sidecar['signature']},cache_provenance=cache_provenance,assets=assets,

        source_sha256=source_hashes,

        protocol=dict(name='matched_project_3dpw_owned_sequence_v1',joint_regressor='H36M17 to SPIN/LSP J14',

            mpjpe='H36M pelvis aligned',pa_mpjpe='per-frame Procrustes similarity aligned J14',

            pve='unaligned SMPL vertex Euclidean error',predicted_body='neutral SMPL',target_body='gendered SMPL',

            rotation='camera root and local body',distance_unit='mm',acceleration_unit='mm/frame^2',

            acceleration='hip-centered J14 prediction-minus-target second difference; adjacent annotation UIDs across windows',

            bbox='annotation-derived crop protocol inherited from caches; compare external methods only after matching',

            flip_test=False,postprocessing='none',test_used_for_selection_in_this_run=False,

            historical_test_exposure='Project history has observed test; this script cannot erase that exposure'),

        context_audit=getattr(data,'context_audit',None),

        temporal_scope='offline; past-memory ablation does not make VACE or track shape causal',

        label_use='Ground truth used only after forward for evaluation metrics',optimizer_steps=0,

        model_or_teacher_fitting_performed=False,sota_claim=False)

    serialized = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open('x', encoding='utf-8') as stream:

        stream.write(serialized)

    print(serialized, flush=True)





if __name__ == '__main__':

    main()

