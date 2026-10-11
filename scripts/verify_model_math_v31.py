"""CPU-only synthetic checks; no SMPL, dataset, trained checkpoint, or metric claim."""
import argparse
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'video_VACE'))

import torch
from models.human_decoder_spatiotemporal_v30 import rotation_6d as original_rotation_6d
from models.human_decoder_spatiotemporal_v31 import (
    AdaptationConfig, AxisAttention, rotation_6d, sample_joint_neighborhood,
    shape_delta_from_values,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=20261008)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    parallel = torch.randn(10000, 3)
    cases = {
        'zero': torch.zeros(4, 6),
        'identity': torch.tensor([[1., 0., 0., 0., 1., 0.]]),
        'parallel_axis': torch.tensor([[1., 0., 0., 2., 0., 0.]]),
        'parallel_scaled': torch.cat((parallel, 100 * parallel), -1),
        'nearly_parallel': torch.cat((parallel, parallel + 1e-5 * torch.randn_like(parallel)), -1),
        'large_finite': torch.randn(1000, 6) * 1e30,
        'random': torch.randn(10000, 6),
    }
    rotation_results = {}
    for name, values in cases.items():
        values = values.clone().requires_grad_(True)
        actual = rotation_6d(values)
        identity = torch.eye(3).expand_as(actual)
        torch.testing.assert_close(actual @ actual.transpose(-1, -2), identity, rtol=0, atol=2e-6)
        torch.testing.assert_close(torch.det(actual), torch.ones(actual.shape[:-2]), rtol=0, atol=2e-6)
        (actual * torch.randn_like(actual)).sum().backward()
        if not torch.isfinite(values.grad).all():
            raise AssertionError(f'Nonfinite gradient for {name}')
        old = original_rotation_6d(values.detach())
        rotation_results[name] = {
            'v30_orthogonality_max': float((old @ old.transpose(-1, -2) - identity).abs().max()),
            'v31_orthogonality_max': float((actual.detach() @ actual.detach().transpose(-1, -2) - identity).abs().max()),
            'v31_det_min': float(torch.det(actual.detach()).min()),
            'v31_det_max': float(torch.det(actual.detach()).max()),
            'v31_gradient_finite': True,
        }
    torch.testing.assert_close(rotation_6d(cases['identity']), torch.eye(3)[None], rtol=0, atol=0)

    cfg = AdaptationConfig(hidden_dim=32, num_heads=4, depth=1)
    cfg.validate()
    values = torch.randn(2, 8, 20, requires_grad=True)
    independent = torch.tensor([[True, True, False, True, True, False, False, False],
                                [True, False, False, False, False, False, False, False]])
    weights = independent.float()[..., None]
    shared = (values[..., :10]*weights).sum(1, keepdim=True)/weights.sum(1, keepdim=True)
    v30_shape = cfg.shape_bound*torch.tanh(shared+values[..., 10:])
    torch.testing.assert_close(shape_delta_from_values(values, independent, cfg), v30_shape, rtol=0, atol=0)
    disabled_cfg = AdaptationConfig(hidden_dim=32, num_heads=4, depth=1, frame_shape_delta_scale=0.)
    disabled_cfg.validate()
    disabled = shape_delta_from_values(values, independent, disabled_cfg)
    changed = values.detach().clone()
    changed[..., 10:] += 1000
    torch.testing.assert_close(disabled, shape_delta_from_values(changed, independent, disabled_cfg), rtol=0, atol=0)
    torch.testing.assert_close(disabled, disabled[:, :1].expand_as(disabled), rtol=0, atol=0)
    disabled.sum().backward()
    if values.grad[..., 10:].count_nonzero().item() != 0 or values.grad[..., :10].abs().sum() == 0:
        raise AssertionError('Incorrect shape ablation gradient routing')

    schedule_cfg = AdaptationConfig(hidden_dim=32, num_heads=4, depth=1, local_radius_schedule=[2., 1., .5])
    schedule_cfg.validate()
    restored_cfg = AdaptationConfig(**json.loads(json.dumps(asdict(schedule_cfg))))
    if asdict(restored_cfg) != asdict(schedule_cfg):
        raise AssertionError('Config JSON roundtrip changed radius schedule')
    memory = torch.randn(1, 3, 16, 12, requires_grad=True)
    grid = torch.tensor([[[0., 0.]]])
    valid = torch.ones(1, 1, dtype=torch.bool)
    expected, _ = sample_joint_neighborhood(memory, grid, valid, 1.)
    for iteration in range(cfg.iterations):
        observed, _ = sample_joint_neighborhood(memory, grid, valid, cfg.radius_at(iteration))
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    scheduled = [sample_joint_neighborhood(memory, grid, valid, schedule_cfg.radius_at(i))[0] for i in range(3)]
    if torch.equal(scheduled[0], scheduled[1]) or torch.equal(scheduled[1], scheduled[2]):
        raise AssertionError('Radius schedule did not change sampled evidence')
    sum(value.square().mean() for value in scheduled).backward()
    if not torch.isfinite(memory.grad).all() or memory.grad.abs().sum() == 0:
        raise AssertionError('Invalid radius-sampling gradients')
    invalid_configs = [dict(frame_shape_delta_scale=-1), dict(frame_shape_delta_scale=float('nan')),
                       dict(local_radius_schedule=(1.,)), dict(local_radius_schedule=(2., 0., .5))]
    for changes in invalid_configs:
        try:
            AdaptationConfig(**changes).validate()
        except ValueError:
            pass
        else:
            raise AssertionError(f'Invalid config accepted: {changes}')

    attention = AxisAttention(cfg, temporal=True).eval()
    tokens = torch.randn(2, 8, 26, cfg.hidden_dim)
    modified = tokens.clone()
    modified[~independent] += 1000 * torch.randn_like(modified[~independent])
    baseline = attention(tokens, independent)
    altered = attention(modified, independent)
    torch.testing.assert_close(baseline[independent], altered[independent], rtol=0, atol=0)
    buffer = io.BytesIO()
    torch.save(attention.state_dict(), buffer)
    buffer.seek(0)
    restored = AxisAttention(cfg, temporal=True).eval()
    restored.load_state_dict(torch.load(buffer, map_location='cpu', weights_only=True), strict=True)
    torch.testing.assert_close(restored(tokens, independent), baseline, rtol=0, atol=0)

    manifest = json.loads((ROOT / 'provenance/v31_model_copy.json').read_text(encoding='utf-8-sig'))
    source_hash = hashlib.sha256((ROOT / manifest['source']).read_bytes()).hexdigest()
    if source_hash != manifest['source_sha256_before']:
        raise AssertionError('Original V30 model source changed')
    result = {
        'status': 'passed', 'device': 'cpu', 'torch': torch.__version__, 'seed': args.seed,
        'scope': 'Synthetic math and attention submodules only; no full model/SMPL/data evaluation',
        'rotation': rotation_results,
        'default_shape_formula_exact': True, 'frame_shape_disabled_output_and_gradient_verified': True,
        'default_radius_samples_exact': True, 'radius_schedule_changes_samples_and_has_finite_gradient': True,
        'config_json_roundtrip_exact': True, 'invalid_configs_rejected': len(invalid_configs),
        'masked_keys_do_not_change_independent_queries': True, 'attention_save_restore_exact': True,
        'original_v30_sha256': source_hash,
        'source_v31_sha256': hashlib.sha256((ROOT / manifest['destination']).read_bytes()).hexdigest(),
        'split': 'none_synthetic', 'initialization_weights': 'seeded_random_submodules',
        'checkpoint_selection': 'none', 'training_performed': False,
        'reconstruction_accuracy_measured': False,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
