"""Functional checks with one real train RGB feature and real SMPL targets.

This bounded check is not training or a performance evaluation. The other
seven target slots repeat the same real frame as invalid, unowned padding.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def uid_digest(uids):
    return hashlib.sha256(('\n'.join(uids) + '\n').encode('utf-8')).hexdigest()


def load_real_feature(directory, expected_uid):
    manifest = json.loads((directory / 'manifest.json').read_text())
    result = json.loads((directory / 'result.json').read_text())
    if manifest.get('split') != 'train' or manifest.get('no_test_frames') is not True:
        raise ValueError('Only explicitly train-only teacher evidence may be checked')
    if manifest.get('three_d_targets_not_teacher_inputs') is not True:
        raise ValueError('Teacher must declare that 3D targets were never inputs')
    if manifest.get('bbox_annotation_derived') is not True:
        raise ValueError('The annotation-derived training crop must be disclosed')
    if result.get('status') != 'passed' or result.get('real_rgb') is not True or result.get('random_features') is not False:
        raise ValueError('Teacher extraction did not establish genuine RGB evidence')
    if result.get('manifest') != manifest:
        raise ValueError('Saved manifest and successful extraction result differ')
    selected = manifest['selected_train_uids']
    if len(set(selected)) != len(selected) or uid_digest(selected) != manifest['selected_train_uid_sha256']:
        raise ValueError('Selected teacher UID manifest is inconsistent')
    samples = result['samples']
    if [sample['uid'] for sample in samples] != selected or result['completed_frames'] != len(selected):
        raise ValueError('Teacher extraction is incomplete or has reordered UID evidence')
    rows = [sample for sample in samples if sample['uid'] == expected_uid]
    if len(rows) != 1:
        raise ValueError(f'Train window 0 frame 0 UID has no unique real teacher feature: {expected_uid}')
    row = rows[0]
    path = (directory / row['file']).resolve()
    if directory.resolve() not in path.parents:
        raise ValueError('Teacher NPZ path must stay inside its evidence directory')
    with np.load(path, allow_pickle=False) as arrays:
        pooled = arrays['strong_rgb_pooled'].copy()
    dimension = manifest['strong_rgb_pooled_dim']
    if pooled.dtype != np.float32 or pooled.shape != (dimension,) or not np.isfinite(pooled).all():
        raise ValueError('Expected finite float32 real pooled feature [D]')
    info = row['strong_rgb_feature']
    if info['pooled_dim'] != dimension or info['layout'] != 'BCHW' or info['forward_shape'][0] != 1:
        raise ValueError('Real teacher feature contract is inconsistent')
    return torch.from_numpy(pooled), dict(
        uid=expected_uid, npz=str(path), npz_sha256=sha256(path),
        pooled_dim=dimension, feature_info=info,
        manifest_sha256=sha256(directory / 'manifest.json'),
        selected_train_uid_sha256=manifest['selected_train_uid_sha256'],
        selected_frames=len(selected), selected_sequence_count=manifest['selected_sequence_count'],
        model_backbone_source_sha256=manifest['model_backbone_source_sha256'],
        checkpoint_sha256=manifest['checkpoint_sha256'],
        three_d_targets_not_teacher_inputs=True, bbox_annotation_derived=True,
    )


def one_real_frame_batch(vace, hmr, pooled):
    from divibody_training.ram_parts_runtime_v29 import V10Dataset
    from torch.utils.data import default_collate
    batch = default_collate([V10Dataset(vace, hmr)[0]])
    if not bool(batch['target_valid'][0, 0] and batch['frame_owner'][0, 0]):
        raise ValueError('The first train window must own its first valid frame')
    frames = batch['target_valid'].shape[1]
    # Every padded slot holds the same genuine frame; none owns a target/loss.
    per_frame = ('rgb_tokens', 'base_rotmat', 'base_betas', 'base_camera',
                 'full_pose', 'betas', 'gender_code', 'cam_pose')
    for name in per_frame:
        value = batch[name]
        if value.shape[:2] != (1, frames):
            raise ValueError(f'Unexpected per-frame shape for {name}: {tuple(value.shape)}')
        batch[name] = value[:, :1].expand(-1, frames, *value.shape[2:]).clone()
    batch['target_valid'] = torch.zeros((1, frames), dtype=torch.bool)
    batch['target_valid'][0, 0] = True
    batch['frame_owner'] = batch['target_valid'].clone()
    batch['velocity_owner'] = torch.zeros_like(batch['velocity_owner'])
    batch['acceleration_owner'] = torch.zeros_like(batch['acceleration_owner'])
    source_id = int(vace['metadata'][0]['target_frame_ids'][0])
    batch['source_frame_ids'] = torch.full((1, frames), source_id, dtype=torch.long)
    batch['strong_rgb_pooled'] = pooled.reshape(1, 1, -1).expand(1, frames, -1).clone()
    for name in ('rgb_tokens', 'vace_features'):
        batch[name] = batch[name].float()
    return batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--teacher-features', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu', choices=('cpu', 'cuda:0'))
    options = parser.parse_args()
    options.output.mkdir(exist_ok=False)
    root = options.root.resolve()
    config_path = options.config or root / 'configs/geometry.json'
    checkpoint = options.checkpoint or root / 'training_runs/v29_geometry_20261006/shape_on/best.pt'
    teacher_dir = options.teacher_features or root / 'runs/sam3d_strong_features64_v4_20261006'
    config = json.loads(config_path.read_text())
    torch.manual_seed(2026)
    torch.set_num_threads(4)
    device = torch.device(options.device)

    from divibody_training.evaluate_standard_3dpw import create_smpl_models
    from divibody_training.train_divibody_rgb_vace_hmr2a_v8 import load_pair
    from divibody_training.ram_parts_runtime_v29 import INPUT_KEYS, forward_batch, geometric_losses
    from divibody_training.parts_training_utils_strong_rgb_v29y import make_optimizer
    from models.human_decoder_parts_v29 import model_from_payload as load_v29, FORMAT as V29_FORMAT
    from models.human_decoder_strong_rgb_v29y import model_from_payload, StrongRGBConfig, FORMAT
    from hmr2.models.smpl_wrapper import SMPL

    vace, hmr = load_pair(Path(config['train-vace-cache']), Path(config['train-hmr-sidecar']), 'train')
    uid = vace['metadata'][0]['target_frame_uids'][0]
    pooled, evidence = load_real_feature(teacher_dir, uid)
    batch = one_real_frame_batch(vace, hmr, pooled)
    smpl_root = Path(config['smpl-model-root'])
    models = create_smpl_models(smpl_root, device)
    official = SMPL(
        model_path=str(smpl_root), gender='neutral', num_betas=10,
        joint_regressor_extra=str(smpl_root.parent / 'SMPL_to_J19.pkl'),
    ).to(device).requires_grad_(False).eval()
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload.get('format') != V29_FORMAT:
        raise ValueError('This check specifically verifies zero-init migration of a native V29 checkpoint')
    base = load_v29(payload).bind_geometry(models[2]).bind_landmarks(official).to(device).eval()
    model = model_from_payload(payload, strong_config=StrongRGBConfig(input_dim=evidence['pooled_dim']))
    model.bind_geometry(models[2]).bind_landmarks(official).to(device).eval()

    def strong_forward(network, include_evidence=True):
        values = [batch[name].to(device) for name in INPUT_KEYS]
        kwargs = dict(frame_valid=batch['target_valid'].to(device),
                      source_frame_ids=batch['source_frame_ids'].to(device), landmark_schema='posetrack')
        if include_evidence:
            kwargs['strong_rgb_pooled'] = batch['strong_rgb_pooled'].to(device)
        return network(*values, **kwargs)

    with torch.no_grad():
        reference = forward_batch(base, batch, device, 'none')
        prediction = strong_forward(model)
    differences = {}
    keys = ('rotmat', 'betas', 'camera', 'direct_2d')
    for name in keys:
        torch.testing.assert_close(prediction[name], reference[name], atol=1e-6, rtol=0.)
        differences[name] = float((prediction[name] - reference[name]).abs().max())
    if torch.count_nonzero(prediction['strong_rgb_delta']).item():
        raise ValueError('Migrated strong adapter must have exactly zero initial output')
    # A deterministic manual perturbation checks that the teacher sees raw RGB.
    # This is a functional probe, not a trained checkpoint or performance gain.
    last = model.strong_rgb_adapter[-1]
    initial_weight = last.weight.detach().clone()
    initial_bias = last.bias.detach().clone()
    with torch.no_grad():
        indices = torch.arange(last.weight.numel(), device=device, dtype=torch.float32)
        last.weight.copy_(.01 * torch.sin(indices.reshape_as(last.weight) * .137))
        bias_indices = torch.arange(last.bias.numel(), device=device, dtype=torch.float32)
        last.bias.copy_(.01 * torch.cos(bias_indices * .173))
        perturbed = strong_forward(model)
    if float(perturbed['strong_rgb_delta'].abs().max()) <= 0:
        raise AssertionError('Nonzero adapter probe did not alter the conditioned RGB')
    teacher_differences = {}
    for name in ('rotmat', 'betas', 'camera'):
        left = perturbed['teacher_prediction'][name]
        right = reference['teacher_prediction'][name]
        torch.testing.assert_close(left, right, atol=1e-6, rtol=0.)
        teacher_differences[name] = float((left - right).abs().max())
    geometric_probe_differences = {
        name: float((perturbed[name] - reference[name]).abs().max()) for name in keys
    }
    probe_delta = float(perturbed['strong_rgb_delta'].abs().max())
    # Restore zero-init weights before the gradient and strict roundtrip checks.
    with torch.no_grad():
        last.weight.copy_(initial_weight)
        last.bias.copy_(initial_bias)
    del base, reference, perturbed
    try:
        strong_forward(model, include_evidence=False)
    except ValueError as error:
        if 'strong_rgb_pooled' not in str(error):
            raise
        missing_error = str(error)
    else:
        raise AssertionError('Missing real strong evidence was accepted')

    args = SimpleNamespace(**{key.replace('-', '_'): value for key, value in config.items()})
    optimizer, names = make_optimizer(model, args)
    learned = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    optimizer_ids = [id(parameter) for group in optimizer.param_groups for parameter in group['params']]
    if set(names) != set(learned) or len(optimizer_ids) != len(set(optimizer_ids)) or set(optimizer_ids) != {id(parameter) for parameter in learned.values()}:
        raise AssertionError('Optimizer must cover every trainable parameter exactly once')
    adapter_ids = {id(parameter) for parameter in model.strong_rgb_adapter.parameters()}
    if not adapter_ids.issubset({id(parameter) for group in optimizer.param_groups if group['category'] == 'geometry' for parameter in group['params']}):
        raise AssertionError('Strong adapter must belong to geometry optimizer groups')
    teacher_ids = {id(parameter) for parameter in model.teacher.parameters()}
    if teacher_ids.intersection(optimizer_ids) or any(parameter.requires_grad for parameter in model.teacher.parameters()):
        raise AssertionError('Frozen teacher was included in optimization')

    model.train()
    prediction = strong_forward(model)
    regressor = torch.as_tensor(np.load(config['joint-regressor']), dtype=torch.float32, device=device)
    losses = geometric_losses(prediction, batch, models, regressor, args, device)
    if not torch.isfinite(losses['total']):
        raise FloatingPointError('Geometric loss is nonfinite')
    losses['total'].backward()
    weight_grad = model.strong_rgb_adapter[-1].weight.grad
    if weight_grad is None or not torch.isfinite(weight_grad).all() or float(weight_grad.norm()) <= 0:
        raise AssertionError('Real geometric loss must produce a finite nonzero strong adapter gradient')
    gradient_norm = float(weight_grad.norm())
    if any(parameter.grad is not None for parameter in model.teacher.parameters()):
        raise AssertionError('Frozen teacher accumulated a gradient')
    # Save/restore unchanged parameters; this functional check performs no update.
    model.eval()
    native = model.checkpoint_payload(dict(
        verification_only=True, strong_evidence_uid=uid,
        three_d_targets_not_teacher_inputs=True, test_used_for_model_selection=False,
    ))
    checkpoint_path = options.output / 'strong_rgb_roundtrip.pt'
    torch.save(native, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if loaded['format'] != FORMAT or 'strong_config' not in loaded:
        raise AssertionError('Native checkpoint did not save its complete strong config')
    restored = model_from_payload(loaded).bind_geometry(models[2]).bind_landmarks(official).to(device).eval()
    with torch.no_grad():
        original_prediction = strong_forward(model)
        restored_prediction = strong_forward(restored)
    for name in keys + ('strong_rgb_delta',):
        torch.testing.assert_close(original_prediction[name], restored_prediction[name], atol=1e-6, rtol=0.)
    damaged = loaded | dict(model={name: value for name, value in loaded['model'].items()
                                  if name != 'strong_rgb_adapter.3.weight'})
    try:
        model_from_payload(damaged)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Native strict reload accepted a missing adapter weight')

    result = dict(
        status='passed', verification_only=True, performance_claim=False,
        sota_status='not_established', checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
        real_train_cache=True, real_smpl=True, real_strong_rgb_feature=evidence,
        actual_owned_valid_frames=1, padding_policy='same real train frame repeated in invalid and unowned slots',
        zero_init_v29_equivalence='passed', max_absolute_differences=differences,
        raw_rgb_teacher_invariance='passed', teacher_probe_differences=teacher_differences,
        manual_nonzero_adapter_probe=True, probe_strong_rgb_delta_abs_max=probe_delta,
        geometric_probe_differences=geometric_probe_differences,
        geometric_loss=float(losses['total'].detach()), strong_adapter_gradient_norm=gradient_norm,
        optimizer_coverage='passed', frozen_teacher='passed',
        missing_evidence_rejection=missing_error, native_strict_roundtrip='passed',
        malformed_native_state_rejection='passed', optimizer_steps=0,
        representation_policy='Only real global RGB evidence is consumed; MHR teacher predictions are not SMPL targets',
    )
    (options.output / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
