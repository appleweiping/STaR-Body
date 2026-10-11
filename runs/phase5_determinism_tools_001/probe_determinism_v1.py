"""Bounded determinism diagnosis, never a training or evaluation acceptance.

Copied from verify_real_v31.py before editing. No optimizer steps or validation
labels. The optional real-input probe uses exactly one declared train window.
"""
import argparse
import ast
from dataclasses import asdict
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
import traceback
from types import SimpleNamespace

LOSS_DEFAULTS = {
    'geometry-lr': 2e-5, 'local-lr': 1e-5, 'camera-lr': 3e-6,
    'gate-lr': 1e-5, 'weight-decay': 1e-4, 'grad-clip': 1.,
    'joint-weight': 1., 'vertex-weight': .5, 'rotation-weight': .05,
    'shape-weight': .001, 'velocity-weight': .5, 'acceleration-weight': .25,
    'anchor-weight': .1, 'update-regularization': .001,
    'stage-3d-weight': .25, 'shape-consistency-weight': .01,
    'deduplicate-shape-consistency': False,
}

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


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def constant(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),
                      object_pairs_hook=pairs, parse_constant=constant)


def pinned_json(path, expected):
    if not expected or sha256_file(path) != expected.lower():
        raise ValueError('Pinned JSON SHA256 mismatch: ' + str(path))
    return read_json(path)


def configure_strict(torch):
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('CUBLAS_WORKSPACE_CONFIG must be :4096:8 before torch import')
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, 'enable_cudnn_sdp'):
        torch.backends.cuda.enable_cudnn_sdp(False)
    return strict_policy(torch)


def strict_policy(torch):
    return dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
        mem_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
        math_sdp=torch.backends.cuda.math_sdp_enabled(),
        cudnn_sdp=(torch.backends.cuda.cudnn_sdp_enabled()
                   if hasattr(torch.backends.cuda, 'cudnn_sdp_enabled') else None),
        CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))


def tensor_digest(torch, value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str((str(value.dtype), list(value.shape))).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def sync(torch, device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def exact_tensors(torch, left, right):
    if left.keys() != right.keys():
        raise AssertionError('Tensor key sets differ')
    different = []
    max_error = 0.
    for name in left:
        a, b = left[name], right[name]
        if a is None or b is None:
            if a is not b:
                different.append(name)
            continue
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            different.append(name)
            if a.shape == b.shape:
                max_error = max(max_error, float((a.double() - b.double()).abs().max()))
    return {'torch_equal': not different, 'different_tensors': different,
            'maximum_absolute_difference': max_error, 'tensor_count': len(left),
            'comparison_tolerance': 0}


def tensor_summary(torch, values):
    result = {}
    for name, value in values.items():
        if value is None:
            result[name] = None
            continue
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError('Nonfinite tensor: ' + name)
        result[name] = {'sha256': tensor_digest(torch, value), 'shape': list(value.shape),
                        'dtype': str(value.dtype), 'norm': float(value.double().norm())}
    return result


def load_original_sampler(root, torch):
    path = root / 'video_VACE/models/human_decoder_parts_v27.py'
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == 'sample_joint_neighborhood')
    namespace = {'torch': torch, 'F': torch.nn.functional}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[node.name], {'path': str(path), 'sha256': sha256_file(path),
                                 'execution': 'exact original function AST; no dataset import'}


def old_operation_probe(function):
    try:
        value = function()
        return {'status': 'backward_completed', 'values': value}
    except RuntimeError as error:
        # Only strict-mode nondeterminism rejection is an expected diagnosis.
        if 'deterministic' not in str(error).lower():
            raise
        return {'status': 'strict_nondeterminism_rejected', 'error_type': type(error).__name__,
                'error': str(error)}


def synthetic_probe(root, torch, device, result):
    from deterministic_ops_v1 import sample_joint_neighborhood, adaptive_avg_pool2d_cpu
    original, source = load_original_sampler(root, torch)
    result['original_sampler_source'] = source
    generator = torch.Generator(device='cpu').manual_seed(2026)
    memory_cpu = torch.randn(8, 256, 16, 12, generator=generator)
    grid_cpu = torch.rand(8, 26, 2, generator=generator) * 2.6 - 1.3
    grid_cpu[:, :5] = torch.tensor([[0., 0.], [-1., -1.], [1., 1.], [-1.2, .5], [.25, -.5]])
    valid_cpu = torch.ones(8, 26, dtype=torch.bool)
    valid_cpu[:, 4] = False
    upstream = torch.randn(8, 26, 9, 256, generator=generator).to(device)
    pool_cpu = torch.randn(5, 256, 6, 6, generator=generator)
    pool_upstream = torch.randn(5, 256, 2, 3, generator=generator).to(device)
    def sampling(function):
        memory = memory_cpu.to(device).clone().requires_grad_(True)
        grid = grid_cpu.to(device).clone().requires_grad_(True)
        valid = valid_cpu.to(device)
        sync(torch, device); start = time.perf_counter()
        output, mask = function(memory, grid, valid)
        sync(torch, device)
        forward_seconds = time.perf_counter() - start
        sync(torch, device); start = time.perf_counter()
        (output * upstream).sum().backward()
        sync(torch, device)
        values = {'output': output.detach().cpu(), 'mask': mask.detach().cpu(),
                  'memory_gradient': memory.grad.detach().cpu(), 'grid_gradient': grid.grad.detach().cpu()}
        return values, {'forward_seconds': forward_seconds,
                        'backward_seconds': time.perf_counter() - start}
    def pooling(function, dtype=torch.float32):
        value = pool_cpu.to(device=device, dtype=dtype).clone().requires_grad_(True)
        sync(torch, device); start = time.perf_counter()
        output = function(value, (2, 3))
        (output * pool_upstream).sum().backward()
        sync(torch, device)
        return {'output': output.detach().cpu(), 'gradient': value.grad.detach().cpu()}, time.perf_counter() - start
    result['synthetic'] = {
        'sampler_shape': list(memory_cpu.shape), 'grid_shape': list(grid_cpu.shape),
        'pool_shape': list(pool_cpu.shape), 'CPU_FP32_transfer_graph_preserved': True,
        'original_grid_sample': old_operation_probe(lambda: tensor_summary(torch, sampling(original)[0])),
        'original_adaptive_avg_pool2d': old_operation_probe(
            lambda: tensor_summary(torch, pooling(torch.nn.functional.adaptive_avg_pool2d)[0]))}
    a, ta = sampling(sample_joint_neighborhood)
    b, tb = sampling(sample_joint_neighborhood)
    sampler_exact = exact_tensors(torch, a, b)
    if not sampler_exact['torch_equal'] or not a['memory_gradient'].any() or not a['grid_gradient'].any():
        raise AssertionError('Candidate sampler repeat/gradient check failed')
    # CPU reference isolates the intended device-only change; CUDA/CPU equality
    # is measured, never assumed to establish the old numerical protocol.
    with torch.no_grad():
        old_forward, old_mask = original(memory_cpu.to(device), grid_cpu.to(device), valid_cpu.to(device))
    measured = exact_tensors(torch, {'output': old_forward.cpu(), 'mask': old_mask.cpu()},
                             {'output': a['output'], 'mask': a['mask']})
    result['synthetic'].update(candidate_sampler_repeat=sampler_exact,
        candidate_sampler_tensors=tensor_summary(torch, a), candidate_sampler_timings=[ta, tb],
        old_vs_candidate_sampler_forward=measured)
    for dtype in (torch.float32, torch.bfloat16):
        a, ta = pooling(adaptive_avg_pool2d_cpu, dtype)
        b, tb = pooling(adaptive_avg_pool2d_cpu, dtype)
        comparison = exact_tensors(torch, a, b)
        if not comparison['torch_equal'] or not a['gradient'].any():
            raise AssertionError('Candidate pool repeat/gradient check failed')
        with torch.no_grad():
            old = torch.nn.functional.adaptive_avg_pool2d(pool_cpu.to(device=device, dtype=dtype), (2, 3))
        result['synthetic']['candidate_pool_' + str(dtype)] = {
            'repeat': comparison, 'tensors': tensor_summary(torch, a), 'seconds': [ta, tb],
            'old_vs_candidate_forward': exact_tensors(torch, {'output': old.cpu()}, {'output': a['output']})}


def verify_prepared_inputs(args, config):
    status = pinned_json(args.prepared_state, args.expected_prepared_state_sha256)
    record = next((value for value in status['generated_bindings']
                   if Path(value['path']).resolve() == args.config.resolve()), None)
    if record is None or record['sha256'] != args.expected_config_sha256.lower() or record['bytes'] != args.config.stat().st_size:
        raise ValueError('Config does not belong to pinned original preparation')
    plan_record = status['prepared_config_plan']
    plan = pinned_json(plan_record['path'], plan_record['sha256'])
    checked = []
    plan_assets = {str(Path(value['path']).resolve()): value for value in plan['assets'].values()}
    if len(plan_assets) != len(plan['assets']):
        raise ValueError('Duplicate prepared asset paths')
    for entry in status['prepared_asset_stats']:
        path = Path(entry['asset']['path']).resolve(strict=True)
        stat = path.stat()
        if plan_assets.get(str(path)) != entry['asset']:
            raise ValueError('Prepared plan asset SHA binding differs from state')
        if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) != (
            entry['device'], entry['inode'], entry['asset']['bytes'], entry['mtime_ns']):
            raise ValueError('Prepared asset stat identity changed: ' + str(path))
        checked.append(entry)
    if len(checked) != len(plan_assets):
        raise ValueError('Prepared state does not cover exact plan asset set')
    for relative, expected in plan['source_sha256'].items():
        source = args.project_root.resolve() / relative
        if not source.resolve().is_relative_to(args.project_root.resolve()) or not source.is_file() or sha256_file(source) != expected:
            raise ValueError('Original prepared source binding changed: ' + relative)
    identities = {str(Path(item['asset']['path']).resolve()): item for item in checked}
    required = [Path(config[key]).resolve(strict=True) for key in
                ('train-vace-cache', 'train-hmr-sidecar', 'init-checkpoint', 'joint-regressor')]
    smpl_root = Path(config['smpl-model-root']).resolve(strict=True)
    required += [smpl_root / ('SMPL_' + sex + '.pkl') for sex in ('NEUTRAL', 'MALE', 'FEMALE')]
    required += [smpl_root.parent / 'SMPL_to_J19.pkl']
    for path in required:
        if str(path.resolve(strict=True)) not in identities:
            raise ValueError('Real probe asset is absent from original prepared stats: ' + str(path))
    return dict(pinned_state_sha256=args.expected_prepared_state_sha256,
        pinned_plan=plan_record, original_preparation_full_hashes_reused=True,
        full_asset_bytes_rehashed_in_this_probe=False,
        stat_checks_are_not_full_hashes=True, original_source_files_rehashed=len(plan['source_sha256']),
        partition=plan['partition'], checked=checked), identities


def module_from_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def real_probe(args, root, torch, device, result):
    import numpy as np
    config = pinned_json(args.config, args.expected_config_sha256)
    asset_audit, asset_identities = verify_prepared_inputs(args, config)
    if config.get('seed') != 2026 or config.get('amp-dtype') != 'bf16':
        raise ValueError('This bounded full-cache probe requires seed 2026 and original BF16 config')
    if config.get('deduplicate-shape-consistency', False) or config.get('frame-shape-delta-scale', 1.) != 1. or config.get('local-radius-schedule'):
        raise ValueError('V31 changes cannot enter V30 diagnostic')
    result['real'] = dict(status='running', asset_audit=asset_audit, config=config,
        config_sha256=args.expected_config_sha256, window_index=args.window_index,
        batch_size=1, original_config_batch_size=config.get('batch-size'),
        optimizer_steps=0, validation_or_test_tensor_files_loaded=False,
        scope='one fixed full-train window; not batch-4 resume equivalence or benchmark',
        head_modes=['original_zero_initialization', 'diagnostic_nonzero_heads_std_1e-4'])
    configure_imports(root)
    from torch.utils.data import default_collate
    from divibody_training import ram_parts_runtime_v30 as runtime
    from divibody_training.train_divibody_rgb_vace_hmr2a_v8 import load_pair
    from divibody_training.v29_boundary_partition import geometric_losses, audit_boundary_partition
    from divibody_training.v29_boundary_training import build_boundary_pool
    from divibody_training.evaluate_standard_3dpw import create_smpl_models
    from hmr2.models.smpl_wrapper import SMPL
    from models import human_decoder_spatiotemporal_v30 as original_model
    candidate = module_from_file(Path(__file__).with_name('human_decoder_spatiotemporal_v30_deterministic_v1.py'),
                                 'star_body_v30_deterministic_probe_candidate')
    vace, hmr = load_pair(Path(config['train-vace-cache']), Path(config['train-hmr-sidecar']), 'train')
    if len(hmr['frame_uids']) != 22735 or len(hmr['window_uids']) != 2909 or vace['signature'].get('development_only'):
        raise ValueError('Expected accepted full train pair with 22735 frames and 2909 windows')
    dataset = runtime.V10Dataset(vace, hmr)
    boundary_pool = build_boundary_pool(dataset)
    partition = audit_boundary_partition(dataset, boundary_pool)
    if partition != asset_audit['partition']:
        raise ValueError('Full train motion partition differs from original prepared audit')
    result['real']['motion_partition'] = partition
    result['real']['boundary_loss_included'] = False
    if not 0 <= args.window_index < len(dataset):
        raise ValueError('Window index out of range')
    sample = dataset[args.window_index]
    if sample['rgb_tokens'].shape[:2] != (8, 192) or not sample['target_valid'].all() or not sample['frame_owner'].any():
        raise ValueError('Declared window must contain 8 valid train frames and owned targets')
    batch = default_collate([sample])
    for key in ('rgb_tokens', 'vace_features'):
        batch[key] = batch[key].float()
        if not torch.isfinite(batch[key]).all():
            raise ValueError('Selected real input has nonfinite features')
    result['real']['selected_metadata'] = json_value(vace['metadata'][args.window_index])
    result['real']['source_frame_ids'] = sample['source_frame_ids'].tolist()
    result['real']['input_tensors'] = tensor_summary(torch, {key: batch[key] for key in runtime.INPUT_KEYS})
    result['real']['cache_signatures'] = json_value({'vace': vace['signature'], 'hmr': hmr['signature']})
    payload = torch.load(config['init-checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    if payload.get('format') != 'ram-iterative-anchor-routed-v28':
        raise ValueError('Real probe requires original V28 initialization')
    smpl_root = Path(config['smpl-model-root'])
    models = create_smpl_models(smpl_root, device)
    for model in models.values():
        model.requires_grad_(False).eval()
    official = SMPL(model_path=str(smpl_root), gender='neutral', num_betas=10,
        joint_regressor_extra=str(smpl_root.parent / 'SMPL_to_J19.pkl')).to(device).requires_grad_(False).eval()
    regressor = torch.as_tensor(np.load(config['joint-regressor'], allow_pickle=False), dtype=torch.float32, device=device)
    effective = LOSS_DEFAULTS | config
    loss_args = SimpleNamespace(**{key.replace('-', '_'): value for key, value in effective.items()})
    loss_args.motion_train_windows = len(dataset)
    loss_args.motion_canonical_within_triplets = partition['canonical_within_triplets']
    adaptation = dict(hidden_dim=config.get('model-width', 256), num_heads=config.get('model-heads', 8),
        depth=config.get('model-depth', 4), iterations=config.get('iterations', 3), shape_bound=config.get('shape-bound', 3.),
        use_temporal=not config.get('disable-temporal', False), use_local_feedback=not config.get('disable-local-feedback', False),
        use_vace=not config.get('disable-vace', False))
    result['real']['adaptation_config'] = adaptation
    result['real']['loss_function'] = geometric_losses.__module__ + '.geometric_losses'
    male = asset_identities[str((smpl_root / 'SMPL_MALE.pkl').resolve())]['asset']['sha256']
    female = asset_identities[str((smpl_root / 'SMPL_FEMALE.pkl').resolve())]['asset']['sha256']
    result['real']['known_asset_limitations'] = {
        'male_female_asset_sha256_equal': male == female,
        'gendered_public_evaluation_protocol_verified': False,
        'meaning': 'Engineering determinism diagnosis only; no benchmark or valid public gender protocol claim'}
    def seed_all():
        random.seed(2026); np.random.seed(2026); torch.manual_seed(2026)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(2026)
    def run_one(module, perturb):
        seed_all()
        model = module.model_from_payload(payload, module.AdaptationConfig(**adaptation)).to(device)
        model.bind_geometry(models[2]).bind_landmarks(official).train()
        if perturb:
            with torch.no_grad():
                for head in (model.body_head, model.root_head, model.shape_head, model.camera_head):
                    head.weight.normal_(std=1e-4)
        # This SHA binds all parameters including the frozen V28 starting state.
        state_hash = hashlib.sha256()
        for name, value in model.state_dict().items():
            state_hash.update(name.encode()); state_hash.update(tensor_digest(torch, value).encode())
        sync(torch, device); start = time.perf_counter()
        values = {}
        try:
            prediction = runtime.forward_batch(model, batch, device, 'bf16')
            for key in ('rotmat', 'betas', 'camera'):
                values['prediction/' + key] = prediction[key].detach().cpu()
            losses = geometric_losses(prediction, batch, models, regressor, loss_args, device)
            for key, value in losses.items():
                values['loss/' + key] = value.detach().cpu()
            sync(torch, device); forward_seconds = time.perf_counter() - start
            start = time.perf_counter()
            losses['total'].backward()
            sync(torch, device)
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    values['gradient/' + name] = None if parameter.grad is None else parameter.grad.detach().cpu()
            if not all('prediction/' + key in values for key in ('rotmat', 'betas', 'camera')) or 'loss/total' not in values:
                raise AssertionError('Candidate forward/loss coverage is incomplete')
            if not any(key.startswith('gradient/') and value is not None for key, value in values.items()):
                raise AssertionError('No real gradients were collected')
            if model.reference.training or any(p.requires_grad or p.grad is not None for p in model.reference.parameters()):
                raise AssertionError('Frozen reference received gradient or changed training mode')
            return values, dict(status='backward_completed', initial_state_sha256=state_hash.hexdigest(),
                tensors=tensor_summary(torch, values), forward_seconds=forward_seconds,
                backward_seconds=time.perf_counter() - start,
                frozen_reference_no_grad=True)
        except RuntimeError as error:
            if module is not original_model or 'deterministic' not in str(error).lower():
                raise
            return values, dict(status='strict_nondeterminism_rejected', initial_state_sha256=state_hash.hexdigest(),
                error_type=type(error).__name__, error=str(error), tensors=tensor_summary(torch, values))
        finally:
            del model
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    result['real']['modes'] = {}
    for perturb in (False, True):
        name = 'diagnostic_nonzero_heads_std_1e-4' if perturb else 'original_zero_initialization'
        print(json.dumps({'event': 'real_window_mode', 'mode': name}), flush=True)
        old, old_report = run_one(original_model, perturb)
        a, report_a = run_one(candidate, perturb)
        b, report_b = run_one(candidate, perturb)
        if len({old_report['initial_state_sha256'], report_a['initial_state_sha256'], report_b['initial_state_sha256']}) != 1:
            raise AssertionError('Fresh models did not start from identical states')
        comparison = exact_tensors(torch, a, b)
        old_forward = {key: value for key, value in old.items() if key.startswith('prediction/')}
        new_forward = {key: a[key] for key in old_forward}
        forward_comparison = (dict(coverage='all_three_body_outputs', **exact_tensors(torch, old_forward, new_forward))
            if len(old_forward) == 3 else dict(coverage='unavailable', reason='Original strict forward failed before all body outputs'))
        mode_report = dict(original_strict=old_report, candidate_runs=[report_a, report_b],
            candidate_repeat=comparison, old_vs_candidate_forward=forward_comparison)
        result['real']['modes'][name] = mode_report
        if not comparison['torch_equal']:
            raise AssertionError('Candidate real repeated forward/backward is not exact')
        if perturb:
            for prefix in ('rgb_projection.', 'vace_projection.', 'feedback_projection.', 'blocks.'):
                grads = [value for key, value in a.items() if key.startswith('gradient/' + prefix) and value is not None]
                if not grads or not any(bool(value.any()) for value in grads):
                    raise AssertionError('Diagnostic heads did not expose trunk gradient: ' + prefix)
        del old, a, b
    # Check original pinned identities again after reading; no cache bytes are written.
    verify_prepared_inputs(args, config)
    result['real']['status'] = 'diagnostic_passed'
    result['project_source_files_sha256'], result['non_file_module_sources'] = source_hashes(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--cuda-memory-limit-gib', type=float, default=10.)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--expected-config-sha256')
    parser.add_argument('--prepared-state', type=Path)
    parser.add_argument('--expected-prepared-state-sha256')
    parser.add_argument('--window-index', type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error('Output must be a new file in an existing directory')
    if not 1 <= args.cpu_threads <= 8 or not 0 < args.cuda_memory_limit_gib <= 10:
        parser.error('Bounded threads 1..8 and CUDA allocator limit (0,10] GiB required')
    if args.config and not all((args.expected_config_sha256, args.prepared_state, args.expected_prepared_state_sha256)):
        parser.error('Real probe requires config and prepared state with external SHA256 pins')
    if not args.config and any((args.expected_config_sha256, args.prepared_state, args.expected_prepared_state_sha256)):
        parser.error('Incomplete real probe arguments')
    root = args.project_root.resolve(strict=True)
    result = dict(format='star-body-bounded-determinism-probe-v1', status='failed',
        diagnostic_only=True, formal_training_started=False, optimizer_steps=0,
        benchmark_metrics_computed=False, old_bitwise_reproduction_claimed=False,
        candidate_numerical_protocol='strict math-SDPA + CPU FP32 grid_sample and adaptive pool; TF32 off',
        scope='synthetic only' if args.config is None else 'synthetic and one real train window',
        source_files={str(path.name): sha256_file(path) for path in (
            Path(__file__), Path(__file__).with_name('deterministic_ops_v1.py'),
            Path(__file__).with_name('human_decoder_spatiotemporal_v30_deterministic_v1.py'))},
        process_environment_before_import={'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG')})
    torch = None; device = None
    try:
        if 'torch' in sys.modules:
            raise RuntimeError('Probe must run as fresh process before torch import')
        existing = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
        if existing not in (None, ':4096:8'):
            raise ValueError('Conflicting preexisting CUBLAS_WORKSPACE_CONFIG')
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        sys.dont_write_bytecode = True
        import torch
        torch.set_num_threads(args.cpu_threads)
        result['policy'] = configure_strict(torch)
        device = torch.device(args.device)
        if device.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('Requested CUDA unavailable')
            torch.cuda.set_device(device)
            total = torch.cuda.get_device_properties(device).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1., args.cuda_memory_limit_gib * 1024**3 / total), device)
            torch.cuda.reset_peak_memory_stats(device)
        result['environment'] = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(), platform=platform.platform(), device=str(device),
            cpu_threads=torch.get_num_threads(), numpy=__import__('numpy').__version__,
            cuda_allocator_limit_gib=args.cuda_memory_limit_gib,
            gpu_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else None)
        print(json.dumps({'event': 'synthetic_start', 'policy': result['policy']}), flush=True)
        synthetic_probe(root, torch, device, result)
        if args.config:
            print(json.dumps({'event': 'real_start', 'window_index': args.window_index}), flush=True)
            real_probe(args, root, torch, device, result)
        if strict_policy(torch) != result['policy']:
            raise AssertionError('Strict numerical policy changed during probe')
        result['status'] = 'diagnostic_passed'
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        if torch is not None and device is not None and device.type == 'cuda' and torch.cuda.is_initialized():
            result['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
            result['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
        print(json.dumps({'status': result['status'], 'output': str(args.output),
                          'sha256': sha256_file(args.output), 'error': result.get('error')}), flush=True)
    return 0 if result['status'] == 'diagnostic_passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
