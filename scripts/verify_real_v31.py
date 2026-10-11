"""Bounded real train-window forward/backward, one optimizer step and disk restore.

Copied from scripts/v30/verify_real_v30_v2.py before editing. This is an
acceptance check, not a benchmark or a formal training run. No val/test is read.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
from types import SimpleNamespace


# These defaults match the preserved V30 training parser and V31 contract.
LOSS_DEFAULTS = {
    'geometry-lr': 2e-5, 'local-lr': 1e-5, 'camera-lr': 3e-6,
    'gate-lr': 1e-5, 'weight-decay': 1e-4, 'grad-clip': 1.,
    'joint-weight': 1., 'vertex-weight': .5, 'rotation-weight': .05,
    'shape-weight': .001, 'velocity-weight': .5, 'acceleration-weight': .25,
    'anchor-weight': .1, 'update-regularization': .001,
    'stage-3d-weight': .25, 'shape-consistency-weight': .01,
    'deduplicate-shape-consistency': False,
}
ASSET_KEYS = ('train-vace-cache', 'train-hmr-sidecar', 'init-checkpoint',
              'smpl-model-root', 'joint-regressor')
OUTPUT_KEYS = ('rotmat', 'betas', 'camera')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_value(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def state_digest(torch, module):
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def source_hashes(root):
    paths = {Path(__file__).resolve()}
    skipped = []
    for module_name, module in tuple(sys.modules.items()):
        try:
            filename = getattr(module, '__file__', None)
        except (AttributeError, RuntimeError, TypeError) as error:
            skipped.append({'module': module_name, 'reason': 'dynamic file attribute',
                            'error_type': type(error).__name__})
            continue
        if filename is None:
            continue
        if not isinstance(filename, (str, os.PathLike)):
            skipped.append({'module': module_name, 'reason': 'non-path file attribute'})
            continue
        path = Path(filename)
        # torch.classes exposes a synthetic relative _classes.py; resolving it
        # first would invent a project file under the current directory.
        try:
            if not path.is_file():
                skipped.append({'module': module_name, 'file': str(filename),
                                'reason': 'not an actual file'})
                continue
            path = path.resolve(strict=True)
        except (OSError, ValueError) as error:
            skipped.append({'module': module_name, 'file': str(filename),
                            'reason': 'unresolvable file', 'error_type': type(error).__name__})
            continue
        if path.suffix == '.py' and path.is_relative_to(root):
            paths.add(path)
    return ({str(path.relative_to(root)): sha256_file(path) for path in sorted(paths)}, skipped)


def check_originals(root):
    manifest_path = root / 'original_code_hashes.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    for name, expected in manifest.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha256_file(path) != expected.lower():
            raise ValueError('Preserved original source differs: ' + name)
    return {'manifest_sha256': sha256_file(manifest_path), 'files_checked': len(manifest)}


def configure_imports(root):
    paths = [root / 'video_VACE', root / 'video_VACE/divibody_training',
             root]
    # Do not add vace/models: its wan.py shadows the external Wan package.
    for key in ('WAN_SOURCE_ROOT', 'HMR2_SOURCE_ROOT', 'DIVIBODY_SOURCE_ROOT'):
        if os.environ.get(key):
            paths.append(Path(os.environ[key]).expanduser().resolve())
    for path in reversed(paths):
        sys.path.insert(0, str(path))
    sys.dont_write_bytecode = True


def assert_tree_equal(torch, actual, expected, label='checkpoint'):
    if isinstance(actual, torch.Tensor):
        if not isinstance(expected, torch.Tensor):
            raise AssertionError(label + ': different tensor type')
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    elif isinstance(actual, dict):
        if not isinstance(expected, dict) or actual.keys() != expected.keys():
            raise AssertionError(label + ': different keys')
        for key in actual:
            assert_tree_equal(torch, actual[key], expected[key], label + '/' + str(key))
    elif isinstance(actual, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise AssertionError(label + ': different sequence')
        for index, (left, right) in enumerate(zip(actual, expected)):
            assert_tree_equal(torch, left, right, label + '/' + str(index))
    elif actual != expected:
        raise AssertionError(label + ': different scalar')


def load_train_pair_for_verification(vace_path, hmr_path, allow_development):
    """Retain every original cache audit; opt out only of the full-train gate."""
    from divibody_training.train_divibody_rgb_vace_hmr2a_v8 import load, load_pair
    if not allow_development:
        return load_pair(vace_path, hmr_path, 'train')
    from divibody_training.cache_hmr2a_rgb_anchor_v8 import validate_sidecar, validate_vace_source
    vace, hmr = load(vace_path), load(hmr_path)
    validate_vace_source(vace)
    validate_sidecar(hmr, vace)
    if vace['signature']['split'] != 'train' or hmr['signature']['split'] != 'train':
        raise ValueError('Development acceptance still requires exact train cache splits')
    # No signatures, source hashes, frame identities, masks or tensors change.
    return vace, hmr


def run(args, result):
    root = Path(__file__).resolve().parents[1]
    config_path = args.config.expanduser().resolve()
    def reject_nonfinite(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    config = json.loads(config_path.read_text(encoding='utf-8-sig'), parse_constant=reject_nonfinite)
    if not isinstance(config, dict) or not config:
        raise ValueError('A nonempty ordinary training JSON object is required')
    result.update(project_root=str(root), config_path=str(config_path), config=config,
        config_sha256=sha256_file(config_path), model_version=args.model_version,
        split='train', evaluation_scope='one real train window; no benchmark metrics',
        formal_training_started=False, optimizer_step_performed=False, diagnostic_only=True,
        allow_development_cache=args.allow_development_cache,
        random_input_features=False, validation_or_test_loaded=False,
        checkpoint_selection_rule='exactly one smoke-test step; no selection',
        sota_status='not_evaluated', verifier_sha256=sha256_file(__file__),
        current_stage='resolve_train_assets')
    assets = {}
    for key in ASSET_KEYS:
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError('Required training config path: ' + key)
        path = Path(config[key]).expanduser()
        path = (path if path.is_absolute() else root / path).resolve()
        if not path.exists():
            raise FileNotFoundError(str(path))
        if (key == 'smpl-model-root' and not path.is_dir()) or (key != 'smpl-model-root' and not path.is_file()):
            raise ValueError('Wrong asset file/directory type: ' + key)
        assets[key] = path
    extra_regressor = assets['smpl-model-root'].parent / 'SMPL_to_J19.pkl'
    if not extra_regressor.is_file():
        raise FileNotFoundError(str(extra_regressor))
    result['resolved_train_assets'] = {key: str(path) for key, path in assets.items()}
    result['original_source_integrity'] = check_originals(root)
    result['initialization'] = {'path': str(assets['init-checkpoint']), 'sha256': sha256_file(assets['init-checkpoint'])}
    result['asset_hashes'] = {'joint_regressor': sha256_file(assets['joint-regressor']), 'SMPL_to_J19': sha256_file(extra_regressor)}
    configure_imports(root)
    result['current_stage'] = 'import_runtime'
    import numpy as np
    import torch
    from torch.utils.data import default_collate
    model_module = importlib.import_module('models.human_decoder_spatiotemporal_' + args.model_version)
    runtime = importlib.import_module('divibody_training.ram_parts_runtime_' + args.model_version)
    optimizer_module = importlib.import_module('divibody_training.parts_training_utils_' + args.model_version)
    from divibody_training.evaluate_standard_3dpw import create_smpl_models
    from hmr2.models.smpl_wrapper import SMPL
    result['runtime_modules'] = {'model': model_module.__name__, 'runtime': runtime.__name__,
        'loss': runtime.geometric_losses.__module__, 'optimizer': optimizer_module.__name__}
    result['source_files_sha256'], result['non_file_module_sources'] = source_hashes(root)
    seed = args.seed if args.seed is not None else config.get('seed', 2026)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device or config.get('device', 'cpu'))
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Only cpu and cuda devices are supported')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA device is unavailable')
        torch.cuda.set_device(device)
        total_memory = torch.cuda.get_device_properties(device).total_memory
        memory_fraction = min(1., args.cuda_memory_limit_gib * 1024**3 / total_memory)
        torch.cuda.set_per_process_memory_fraction(memory_fraction, device)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    result['seed'] = seed
    result['environment'] = {'python': sys.version, 'pytorch': torch.__version__,
        'numpy': np.__version__, 'cuda_runtime': torch.version.cuda, 'device': str(device),
        'platform': platform.platform(), 'cpu_threads': args.cpu_threads, 'amp_dtype': 'none',
        'cuda_memory_limit_gib': args.cuda_memory_limit_gib if device.type == 'cuda' else None,
        'cuda_allocator_memory_fraction': memory_fraction if device.type == 'cuda' else None,
        'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None}
    effective = LOSS_DEFAULTS | config
    result['effective_loss_and_optimizer_config'] = effective
    loss_args = SimpleNamespace(**{key.replace('-', '_'): value for key, value in effective.items()})
    for key in LOSS_DEFAULTS:
        value = getattr(loss_args, key.replace('-', '_'))
        if key == 'deduplicate-shape-consistency':
            if not isinstance(value, bool):
                raise ValueError(key + ' must be boolean')
        elif isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid nonnegative loss/optimizer value: ' + key)
    if loss_args.grad_clip <= 0:
        raise ValueError('grad-clip must be positive')
    if args.model_version == 'v30' and (config.get('deduplicate-shape-consistency', False)
            or config.get('frame-shape-delta-scale', 1.) != 1. or config.get('local-radius-schedule')):
        raise ValueError('V31 ablations cannot be silently applied to a V30 verification')
    result['current_stage'] = 'load_real_train_pair'
    vace, hmr = load_train_pair_for_verification(assets['train-vace-cache'],
        assets['train-hmr-sidecar'], args.allow_development_cache)
    result['development_cache'] = bool(vace['signature'].get('development_only') or hmr['signature'].get('development_only'))
    if result['development_cache'] and not args.allow_development_cache:
        raise ValueError('A development cache requires explicit --allow-development-cache')
    result['cache_audits'] = {'exact_train_split': True, 'vace_schema': True,
        'sidecar_schema_and_finiteness': True, 'window_and_frame_uids': True,
        'vace_semantic_sha256': True, 'hmr_initialization_sha256': True,
        'full_train_completeness_gate_waived': bool(args.allow_development_cache),
        'source_cache_signatures_modified': False}
    dataset = runtime.V10Dataset(vace, hmr)
    result['cache_signatures'] = json_value({'vace': vace['signature'], 'hmr': hmr['signature']})
    candidates = [args.window_index] if args.window_index is not None else range(len(dataset))
    selected = None
    for index in candidates:
        if not 0 <= index < len(dataset):
            raise ValueError('window-index is outside the train cache')
        record = vace['metadata'][index]
        ids = record.get('target_source_frame_ids')
        ids = record['target_frame_ids'] if ids is None else ids
        valid = vace['target_valid_mask'][index]
        if len(ids) == 8 and len(set(int(value) for value in ids)) == 8 and bool(valid.all()) and bool(vace['frame_owner_mask'][index].any()):
            selected = index
            break
    if selected is None:
        raise ValueError('A real valid distinct 8-frame train window with owned frames is required')
    sample = dataset[selected]
    if sample['rgb_tokens'].shape[0] != 8 or sample['rgb_tokens'].shape[1] != 16 * 12:
        raise ValueError('Expected real HMR2 spatial tokens [8, 192, C]')
    batch = default_collate([sample])
    for key in ('rgb_tokens', 'vace_features'):
        batch[key] = batch[key].float()
        if not torch.isfinite(batch[key]).all():
            raise ValueError('Nonfinite real input feature: ' + key)
    record_json = json_value(vace['metadata'][selected])
    record_text = json.dumps(record_json, sort_keys=True, ensure_ascii=True).encode()
    result['data_window'] = {'index': selected, 'total_train_windows': len(dataset),
        'selection_rule': 'explicit index' if args.window_index is not None else 'first eligible window without label-based selection',
        'metadata': record_json, 'metadata_sha256': hashlib.sha256(record_text).hexdigest(),
        'source_frame_ids': sample['source_frame_ids'].tolist(), 'frames': 8,
        'valid_frames': int(sample['target_valid'].sum()), 'owned_frames': int(sample['frame_owner'].sum()),
        'velocity_pairs': int(sample['velocity_owner'].sum()), 'acceleration_triplets': int(sample['acceleration_owner'].sum())}
    result['current_stage'] = 'construct_zero_initialized_model'
    payload = torch.load(assets['init-checkpoint'], map_location='cpu', weights_only=False)
    if payload.get('format') != 'ram-iterative-anchor-routed-v28':
        raise ValueError('Zero-initialization acceptance requires the original V28 checkpoint, not trained V30/V31')
    result['initialization']['format'] = payload['format']
    adaptation_values = dict(hidden_dim=config.get('model-width', 256), num_heads=config.get('model-heads', 8),
        depth=config.get('model-depth', 4), iterations=config.get('iterations', 2), shape_bound=config.get('shape-bound', 3.),
        use_temporal=not config.get('disable-temporal', False), use_local_feedback=not config.get('disable-local-feedback', False),
        use_vace=not config.get('disable-vace', False))
    if args.model_version == 'v31':
        schedule = config.get('local-radius-schedule', ())
        if isinstance(schedule, str):
            schedule = tuple(float(value) for value in schedule.split(',') if value.strip())
        adaptation_values.update(frame_shape_delta_scale=config.get('frame-shape-delta-scale', 1.), local_radius_schedule=tuple(schedule))
    adaptation = model_module.AdaptationConfig(**adaptation_values)
    result['adaptation_config'] = asdict(adaptation)
    model = model_module.model_from_payload(payload, adaptation).to(device).eval()
    del payload
    smpl_models = create_smpl_models(assets['smpl-model-root'], device)
    for value in smpl_models.values():
        value.requires_grad_(False).eval()
    official = SMPL(model_path=str(assets['smpl-model-root']), gender='neutral', num_betas=10,
        joint_regressor_extra=str(extra_regressor)).to(device).requires_grad_(False).eval()
    model.bind_geometry(smpl_models[2]).bind_landmarks(official)
    regressor = torch.as_tensor(np.load(assets['joint-regressor'], allow_pickle=False), dtype=torch.float32, device=device)

    def forward(module):
        return runtime.forward_batch(module, batch, device, 'none')

    def losses(prediction):
        values = runtime.geometric_losses(prediction, batch, smpl_models, regressor, loss_args, device)
        if any(not bool(torch.isfinite(value).all()) for value in values.values()):
            raise FloatingPointError('Real train geometry loss is nonfinite')
        return values

    def frozen_reference_check():
        if model.reference.training or any(parameter.requires_grad or parameter.grad is not None for parameter in model.reference.parameters()):
            raise AssertionError('Frozen V28 is trainable, in training mode, or received a gradient')

    result['current_stage'] = 'initialization_and_so3'
    with torch.no_grad():
        expected, actual = forward(model.reference), forward(model)
        differences = {}
        for key in OUTPUT_KEYS:
            torch.testing.assert_close(actual[key], expected[key], atol=1e-6, rtol=0)
            differences[key] = float((actual[key] - expected[key]).abs().max())
        rotations = actual['rotmat']
        orthogonality = float((rotations @ rotations.transpose(-1, -2) - torch.eye(3, device=device)).abs().max())
        torch.testing.assert_close(rotations @ rotations.transpose(-1, -2), torch.eye(3, device=device).expand_as(rotations), atol=1e-4, rtol=0)
        torch.testing.assert_close(torch.det(rotations), torch.ones_like(torch.det(rotations)), atol=1e-4, rtol=0)
        for value in (torch.zeros(5, 6, device=device), torch.tensor([[1., 0., 0., 2., 0., 0.]], device=device)):
            recovered = model_module.rotation_6d(value)
            torch.testing.assert_close(recovered @ recovered.transpose(-1, -2), torch.eye(3, device=device).expand_as(recovered), atol=1e-5, rtol=0)
            torch.testing.assert_close(torch.det(recovered), torch.ones_like(torch.det(recovered)), atol=1e-5, rtol=0)
    result.update(initialization_max_difference=differences, orthogonality_max_error=orthogonality)
    del actual, expected
    optimizer, names = optimizer_module.make_optimizer(model, loss_args)
    learned = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    covered = [parameter for group in optimizer.param_groups for parameter in group['params']]
    if (set(names) != set(learned) or len(names) != len(set(names))
            or len(covered) != len({id(parameter) for parameter in covered})
            or {id(parameter) for parameter in covered} != {id(parameter) for parameter in learned.values()}):
        raise AssertionError('Optimizer coverage is incomplete, duplicated, or includes frozen parameters')
    result['optimizer_coverage'] = {'parameter_tensors': len(names),
        'trainable_parameters': sum(parameter.numel() for parameter in learned.values()),
        'total_parameters': sum(parameter.numel() for parameter in model.parameters()), 'names': sorted(names)}
    result['current_stage'] = 'one_unperturbed_real_optimizer_step'
    reference_before = state_digest(torch, model.reference)
    before = {name: parameter.detach().cpu().clone() for name, parameter in learned.items()}
    model.train()
    frozen_reference_check()
    optimizer.zero_grad(set_to_none=True)
    train_losses = losses(forward(model))
    train_losses['total'].backward()
    frozen_reference_check()
    for parameter in learned.values():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError('Nonfinite gradient in actual optimizer step')
    grad_norm = torch.nn.utils.clip_grad_norm_(learned.values(), loss_args.grad_clip)
    if not torch.isfinite(grad_norm) or grad_norm <= 0:
        raise AssertionError('Actual optimizer step has no finite nonzero gradient')
    optimizer.step()
    result.update(optimizer_step_performed=True, optimizer_steps=1)
    changed = [name for name, parameter in learned.items() if not torch.equal(before[name], parameter.detach().cpu())]
    if not changed or any(not torch.isfinite(parameter).all() for parameter in learned.values()):
        raise AssertionError('Optimizer step did not update finite trainable parameters')
    model.eval()
    with torch.no_grad():
        post_losses = losses(forward(model))
    result.update(optimizer_step_performed=True, optimizer_steps=1,
        train_loss_before_step={key: float(value.detach()) for key, value in train_losses.items()},
        train_loss_after_step={key: float(value) for key, value in post_losses.items()},
        optimizer_step_gradient_norm=float(grad_norm), changed_parameter_names=changed)
    del before, train_losses, post_losses
    if state_digest(torch, model.reference) != reference_before:
        raise AssertionError('Frozen reference state changed during optimizer step')

    # Nonzero heads expose trunk gradients. Restore them before saving, and
    # never apply an optimizer step on these diagnostic perturbations.
    result['current_stage'] = 'nonzero_head_gradient_diagnostic'
    heads = (model.body_head, model.root_head, model.shape_head, model.camera_head)
    head_states = [{key: value.detach().clone() for key, value in head.state_dict().items()} for head in heads]
    with torch.no_grad():
        for head in heads:
            head.weight.normal_(std=1e-4)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    diagnostic_loss = losses(forward(model))['total']
    diagnostic_loss.backward()
    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
           for parameter in model.parameters()):
        raise FloatingPointError('Nonfinite nonzero-head diagnostic gradient')
    prefixes = ['rgb_projection.', 'blocks.', 'body_head.', 'root_head.', 'shape_head.']
    if adaptation.use_local_feedback:
        prefixes.append('feedback_projection.')
    if adaptation.use_vace:
        prefixes.append('vace_projection.')
    norms = {}
    for prefix in prefixes:
        grads = [parameter.grad.float().square().sum() for name, parameter in model.named_parameters()
                 if name.startswith(prefix) and parameter.grad is not None]
        norm = float(torch.stack(grads).sum().sqrt()) if grads else 0.
        if not math.isfinite(norm) or norm <= 0:
            raise AssertionError('Missing/nonfinite diagnostic geometry gradient: ' + prefix)
        norms[prefix] = norm
    if adaptation.use_temporal:
        for axis in ('st_time', 'ts_time'):
            grads = [parameter.grad.float().square().sum() for name, parameter in model.named_parameters()
                     if name.startswith('blocks.') and ('.' + axis + '.') in name and parameter.grad is not None]
            norm = float(torch.stack(grads).sum().sqrt()) if grads else 0.
            if not math.isfinite(norm) or norm <= 0:
                raise AssertionError('Missing/nonfinite temporal diagnostic gradient: ' + axis)
            norms['blocks.*.' + axis + '.'] = norm
    frozen_reference_check()
    for head, state in zip(heads, head_states):
        head.load_state_dict(state, strict=True)
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    result['head_perturbation_diagnostic'] = {'gradient_norms': norms, 'head_weight_normal_std': 1e-4,
        'loss': float(diagnostic_loss.detach()), 'optimizer_step_on_perturbed_heads': False,
        'head_parameters_restored_before_save': True}
    del diagnostic_loss, head_states
    result['current_stage'] = 'actual_disk_save_and_strict_restore'
    native = {'format': model_module.FORMAT, 'model_config': asdict(model.config),
        'adaptation_config': asdict(model.adaptation_config), 'reference_spec': model.reference_spec,
        'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': 1,
        'seed': seed, 'config': config, 'split': 'train',
        'purpose': 'bounded acceptance only; not a selected training checkpoint',
        'initialization_sha256': result['initialization']['sha256']}
    checkpoint = args.output / 'verifier_step1.pt'
    with checkpoint.open('xb') as stream:
        torch.save(native, stream)
    disk = torch.load(checkpoint, map_location='cpu', weights_only=False)
    restored = model_module.model_from_payload(disk).to(device).bind_geometry(smpl_models[2]).bind_landmarks(official).eval()
    restored_optimizer, restored_names = optimizer_module.make_optimizer(restored, loss_args)
    restored_optimizer.load_state_dict(disk['optimizer'])
    if names != restored_names:
        raise AssertionError('Restored optimizer name ordering differs')
    assert_tree_equal(torch, model.state_dict(), restored.state_dict(), 'model')
    assert_tree_equal(torch, optimizer.state_dict(), restored_optimizer.state_dict(), 'optimizer')
    with torch.no_grad():
        output, repeat = forward(model), forward(restored)
        for key in OUTPUT_KEYS:
            torch.testing.assert_close(output[key], repeat[key], atol=0, rtol=0)
    malformed = dict(disk, model={key: value for key, value in disk['model'].items() if key != 'body_head.weight'})
    try:
        model_module.model_from_payload(malformed)
    except (RuntimeError, ValueError):
        strict_rejection = True
    else:
        raise AssertionError('Malformed native checkpoint was accepted')
    result['checkpoint'] = {'path': str(checkpoint), 'sha256': sha256_file(checkpoint),
        'bytes': checkpoint.stat().st_size, 'model_state_exact': True, 'optimizer_state_exact': True,
        'prediction_restore_exact': True, 'malformed_native_rejected': strict_rejection,
        'initialization_is_unperturbed_before_step': True}
    result['frozen_reference_gradient_free'] = True
    result['frozen_reference_state_unchanged'] = state_digest(torch, model.reference) == reference_before
    if not result['frozen_reference_state_unchanged']:
        raise AssertionError('Frozen reference changed during diagnostics or restore')
    result['original_source_integrity_after'] = check_originals(root)
    result['source_files_sha256'], result['non_file_module_sources'] = source_hashes(root)
    result['model_input_keys'] = ['rgb_tokens', 'vace_features', 'base_rotmat', 'base_betas',
                                 'base_camera', 'source_frame_ids', 'target_valid']
    result['no_3d_ground_truth_in_model_inputs'] = True
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        result['resources'] = {'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device),
                              'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(device)}
    else:
        result['resources'] = {'peak_cuda_allocated_bytes': None, 'peak_cuda_reserved_bytes': None}
    result['status'], result['current_stage'] = 'passed', 'complete'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='Ordinary training JSON; relative assets resolve from project root')
    parser.add_argument('--output', type=Path, required=True, help='New directory; an existing directory is rejected')
    parser.add_argument('--device', help='cpu, cuda, or cuda:N; defaults to the config device')
    parser.add_argument('--model-version', choices=('v30', 'v31'), default='v31')
    parser.add_argument('--seed', type=int, help='Defaults to config seed, then 2026')
    parser.add_argument('--window-index', type=int, help='Optional train index; otherwise first full distinct 8-frame window')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--allow-development-cache', action='store_true',
                        help='Allow audited train-only development caches for this diagnostic; never a full-training claim')
    parser.add_argument('--cuda-memory-limit-gib', type=float, default=10.,
                        help='PyTorch CUDA allocator budget in GiB (default: 10)')
    args = parser.parse_args()
    if args.cpu_threads < 1 or (args.window_index is not None and args.window_index < 0):
        parser.error('cpu-threads must be positive and window-index nonnegative')
    if not math.isfinite(args.cuda_memory_limit_gib) or args.cuda_memory_limit_gib <= 0:
        parser.error('cuda-memory-limit-gib must be finite and positive')
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result = {'status': 'running', 'command': sys.argv, 'current_stage': 'read_config'}
    try:
        run(args, result)
    except Exception as error:
        result.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        result['elapsed_seconds'] = time.perf_counter() - started
        torch_module = sys.modules.get('torch')
        try:
            if torch_module is not None and torch_module.cuda.is_initialized():
                device_name = result.get('environment', {}).get('device', '')
                if device_name.startswith('cuda'):
                    result['resources'] = {'peak_cuda_allocated_bytes': torch_module.cuda.max_memory_allocated(device_name),
                                          'peak_cuda_reserved_bytes': torch_module.cuda.max_memory_reserved(device_name)}
        except (RuntimeError, AttributeError) as resource_error:
            # Metadata collection must not hide the original CUDA failure or
            # prevent the acceptance JSON from being written.
            result['resource_probe_error'] = str(resource_error)
        with (args.output / 'result.json').open('x', encoding='utf-8') as stream:
            json.dump(json_value(result), stream, ensure_ascii=False, indent=2, allow_nan=False)
        print(json.dumps({'status': result['status'], 'result': str(args.output / 'result.json')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
