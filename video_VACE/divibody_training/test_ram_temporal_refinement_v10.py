"""CPU regression checks for v10 temporal refinement and training isolation.

Derived by copying test_ram_geometry_priority_v9.py; the v9 file is unchanged.
Run from video_VACE: python -m divibody_training.test_ram_temporal_refinement_v10
"""
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from models.human_decoder_rgb_vace_v9_isolated import (
    HumanRGBVaceGeometryTransformerV9Isolated, RGBVaceHMR2AConfig,
)
from models.human_decoder_rgb_vace_v10_temporal import (
    HumanRGBVaceGeometryTransformerV10Temporal, TemporalRefinerConfig,
)
from divibody_training.train_ram_temporal_refinement_v10 import make_optimizer, save
from divibody_training.ram_temporal_runtime_v10 import forward_batch, model_from_payload


def _nonzero_gradient(parameter):
    return (parameter.grad is not None and torch.isfinite(parameter.grad).all()
            and bool(parameter.grad.abs().sum() > 0))


def _expect_checkpoint_rejected(model, state):
    try:
        model.load_compatible_state_dict(state)
    except (RuntimeError, ValueError):
        return
    raise AssertionError('A missing or unknown checkpoint key was silently accepted')


def _assert_geometry_close(left, right, mask=None, *, exact=False):
    for key in ('rotmat', 'betas', 'camera'):
        a, b = left[key], right[key]
        if mask is not None:
            a, b = a[mask], b[mask]
        torch.testing.assert_close(a, b, rtol=0, atol=0 if exact else 1e-6)


def _optimizer_names(model, optimizer):
    all_names = {id(parameter): name for name, parameter in model.named_parameters()}
    members = [parameter for group in optimizer.param_groups for parameter in group['params']]
    assert len({id(parameter) for parameter in members}) == len(members), 'Duplicate optimizer parameters'
    names = {all_names[id(parameter)] for parameter in members}
    assert names == {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return names


def main():
    torch.set_num_threads(2)
    torch.manual_seed(1024)
    config = RGBVaceHMR2AConfig(hidden_dim=32, num_heads=4, rgb_input_dim=16,
                               vace_input_dim=16, dropout=0)
    temporal_config = TemporalRefinerConfig(hidden_dim=32, num_heads=4,
                                            num_layers=1, dropout=0)
    old = HumanRGBVaceGeometryTransformerV9Isolated(config).eval()
    # A trained v9 head must survive expansion, not just an untrained anchor.
    with torch.no_grad():
        old.rotation_residual_head.weight.normal_(0, .025)
        old.rotation_residual_head.bias.add_(torch.randn_like(old.rotation_residual_head.bias) * .01)
        old.shape_residual_head.weight.normal_(0, .03)
        old.shape_residual_head.bias.normal_(0, .01)
        old.camera_residual_head.weight.normal_(0, .03)
        old.camera_residual_head.bias.normal_(0, .01)
        old.temporal_gates.fill_(.15)

    def new_model():
        return HumanRGBVaceGeometryTransformerV10Temporal(config, temporal_config).eval()

    model = new_model()
    incompatible = model.load_compatible_state_dict(old.state_dict())
    expected_new_keys = {name for name in model.state_dict() if name.startswith('temporal_refiner.')}
    assert expected_new_keys
    assert set(incompatible.missing_keys) == expected_new_keys
    assert not incompatible.unexpected_keys
    inputs = dict(rgb_tokens=torch.randn(1, 8, 4, 16),
                  vace_features=torch.randn(1, 5, 2, 2, 16),
                  base_rotmat=torch.eye(3).expand(1, 8, 24, 3, 3).clone(),
                  base_betas=torch.randn(1, 8, 10) * .03,
                  base_camera=torch.randn(1, 8, 3) * .05 + 1)
    with torch.no_grad():
        old_prediction, initial = old(**inputs), model(**inputs)
    _assert_geometry_close(old_prediction, initial, exact=True)
    for key in ('temporal_rotation_delta_6d', 'temporal_beta_delta'):
        assert torch.count_nonzero(initial[key]) == 0

    missing_old = dict(old.state_dict())
    missing_old.pop('rotation_residual_head.weight')
    _expect_checkpoint_rejected(new_model(), missing_old)
    extra_old = dict(old.state_dict(), unknown_parameter=torch.zeros(1))
    _expect_checkpoint_rejected(new_model(), extra_old)
    incomplete_v10 = dict(model.state_dict())
    incomplete_v10.pop('temporal_refiner.rotation_delta_head.weight')
    _expect_checkpoint_rejected(new_model(), incomplete_v10)
    shape_mismatch = dict(model.state_dict())
    shape_mismatch['temporal_refiner.rotation_delta_head.weight'] = torch.zeros(1, 1)
    _expect_checkpoint_rejected(new_model(), shape_mismatch)

    args = SimpleNamespace(temporal_lr=1e-2, unfreeze_geometry=False,
                           camera_lr=2e-6, geometry_lr=2e-7, weight_decay=0)
    optimizer, reported_names = make_optimizer(model, args, True)
    names = _optimizer_names(model, optimizer)
    assert names == set(reported_names)
    expected_names = {name for name, _ in model.named_parameters()
                      if name.startswith(('temporal_refiner.', 'camera_residual_head.'))}
    assert names == expected_names
    assert {group['group_name'] for group in optimizer.param_groups} == {'temporal', 'camera'}
    head_prefixes = ('temporal_refiner.rotation_delta_head.', 'temporal_refiner.beta_delta_head.')
    rotation_target = torch.randn_like(initial['rotmat'])
    beta_target = torch.randn_like(initial['betas'])

    def objective(prediction):
        # Unlike ||R||^2, this objective is not constant on SO(3).
        return ((prediction['rotmat'] - rotation_target).square().mean()
                + (prediction['betas'] - beta_target).square().mean())

    objective(model(**inputs)).backward()
    assert _nonzero_gradient(model.temporal_refiner.rotation_delta_head.weight)
    assert _nonzero_gradient(model.temporal_refiner.beta_delta_head.weight)
    assert all(parameter.grad is None for name, parameter in model.named_parameters()
               if not name.startswith('temporal_refiner.'))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    objective(model(**inputs)).backward()
    assert any(_nonzero_gradient(parameter) for name, parameter in model.named_parameters()
               if name.startswith('temporal_refiner.') and not name.startswith(head_prefixes)), (
                   'Temporal encoder remains disconnected after the zero-initialized heads update')
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        learned = model(**inputs)
        altered_rgb = {key: value.clone() for key, value in inputs.items()}
        altered_rgb['rgb_tokens'][:, 0] = torch.randn_like(altered_rgb['rgb_tokens'][:, 0]) * 3
        cross_rgb = model(**altered_rgb)
        altered_smpl = {key: value.clone() for key, value in inputs.items()}
        altered_smpl['base_betas'][:, 0] += torch.randn_like(altered_smpl['base_betas'][:, 0])
        cross_smpl = model(**altered_smpl)
    for changed, source in ((cross_rgb, 'RGB'), (cross_smpl, 'SMPL')):
        magnitude = sum((learned[key][:, 1:] - changed[key][:, 1:]).abs().max().item()
                        for key in ('rotmat', 'betas'))
        assert magnitude > 1e-7, f'{source} input has no effect on other frames: {magnitude}'

    # Include interior holes as well as a trailing pad, to catch velocity leakage.
    valid = torch.tensor([[True, True, True, False, True, True, False, False]])
    padded_inputs = {key: value.clone() for key, value in inputs.items()}
    for key in ('rgb_tokens', 'base_betas', 'base_camera'):
        padded_inputs[key][~valid] = torch.randn_like(padded_inputs[key][~valid]) * 100
    padded_inputs['base_rotmat'][~valid] = torch.tensor([[0., -1., 0.],
                                                       [1., 0., 0.], [0., 0., 1.]])
    with torch.no_grad():
        padded_reference = model(**inputs, frame_valid=valid)
        padded_altered = model(**padded_inputs, frame_valid=valid)
        padded_runtime = forward_batch(model, dict(padded_inputs, target_valid=valid),
                                       torch.device('cpu'), 'none')
    _assert_geometry_close(padded_reference, padded_altered, valid)
    _assert_geometry_close(padded_runtime, padded_altered, exact=True)
    for key in ('temporal_rotation_delta_6d', 'temporal_beta_delta'):
        torch.testing.assert_close(padded_reference[key][valid], padded_altered[key][valid],
                                   rtol=0, atol=1e-6)
        assert torch.count_nonzero(padded_altered[key][~valid]) == 0
    assert all(torch.isfinite(padded_altered[key]).all() for key in ('rotmat', 'betas', 'camera'))

    # Enabling every parameter makes the isolation test independent of freezing.
    model.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    prediction = model(**inputs)
    isolated = model.camera_only_prediction(prediction, inputs['base_camera'])
    isolated['camera'].square().sum().backward()
    assert _nonzero_gradient(model.camera_residual_head.weight)
    assert all(parameter.grad is None for name, parameter in model.named_parameters()
               if not name.startswith('camera_residual_head.'))
    camera_optimizer, reported_names = make_optimizer(model, args, False)
    camera_names = _optimizer_names(model, camera_optimizer)
    assert camera_names == set(reported_names) == {'camera_residual_head.weight', 'camera_residual_head.bias'}
    assert {group['group_name'] for group in camera_optimizer.param_groups} == {'camera'}

    args.unfreeze_geometry = True
    geometry_optimizer, reported_names = make_optimizer(model, args, True)
    geometry_names = _optimizer_names(model, geometry_optimizer)
    assert geometry_names == set(reported_names)
    geometry_prefixes = ('rotation_residual_head.', 'shape_residual_head.',
                         'temporal_refiner.', 'camera_residual_head.')
    assert geometry_names == {name for name, _ in model.named_parameters() if name.startswith(geometry_prefixes)}
    assert {group['group_name'] for group in geometry_optimizer.param_groups} == {'temporal', 'camera', 'geometry'}

    # Exercise the trainer's actual checkpoint schema and inference loader.
    with TemporaryDirectory(prefix='ram_v10_cpu_check_') as directory:
        checkpoint = Path(directory) / 'best.pt'
        save(checkpoint, model, geometry_optimizer, 1, 'geometry', {}, {}, args)
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert payload['model_config'] == asdict(config)
    assert payload['temporal_config'] == asdict(temporal_config)
    assert model.anchor_metadata()['temporal_config'] == asdict(temporal_config)
    restored = model_from_payload(payload).eval()
    assert asdict(restored.temporal_config) == asdict(temporal_config)
    with torch.no_grad():
        _assert_geometry_close(model(**inputs), restored(**inputs), exact=True)
    try:
        model_from_payload(payload, TemporalRefinerConfig(hidden_dim=64, num_heads=4,
                                                          num_layers=1, dropout=0))
    except ValueError:
        pass
    else:
        raise AssertionError('A requested architecture conflicting with the saved v10 config was accepted')
    print('V10_CHECKS_PASS: trained v9 preservation; strict checkpoint validation; '
          'head and encoder gradients; RGB/SMPL cross-frame influence; padding isolation; '
          'camera isolation; optimizer groups; configured checkpoint roundtrip')


if __name__ == '__main__':
    main()
