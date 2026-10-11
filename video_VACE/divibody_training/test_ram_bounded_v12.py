"""CPU numeric checks of actual v12 classes with a lightweight inherited-path stub.

This checks the new branch, not SMPL/cache integration. Run beside model source.
"""
import ast
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from collections.abc import Mapping
from types import SimpleNamespace
import torch
from torch import nn


class FrozenBaseline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, rgb, vace, rot, betas, camera):
        return dict(rgb_queries=rgb.mean(2), rotmat=rot, betas=betas, camera=camera)


def main():
    torch.manual_seed(17)
    torch.set_num_threads(2)
    path = Path(__file__).with_name('human_decoder_rgb_vace_v12_bounded.py')
    if not path.exists():
        path = Path(__file__).resolve().parent.parent / 'models' / path.name
    source = ast.parse(path.read_text(encoding='utf-8'))
    classes = ast.Module(body=[n for n in source.body if isinstance(n, ast.ClassDef)], type_ignores=[])
    namespace = dict(torch=torch, nn=nn, dataclass=dataclass, asdict=asdict,
                     Optional=Optional, Mapping=Mapping, RGBVaceHMR2AConfig=SimpleNamespace,
                     HumanRGBVaceGeometryTransformerV9Isolated=FrozenBaseline)
    exec(compile(classes, str(path), 'exec'), namespace)
    config = SimpleNamespace(num_betas=10, max_rgb_frames=8, hidden_dim=16)
    temporal = namespace['TemporalRefinerConfig'](hidden_dim=16, num_heads=4,
                                                num_layers=1, dropout=0)
    model = namespace['HumanRGBVaceGeometryTransformerV12Bounded'](config, temporal).eval()
    rgb = torch.randn(2, 4, 3, 16)
    # Non-identity trained baseline rotations.
    raw = torch.randn(2, 4, 24, 3, 3) * .2
    rot = torch.matrix_exp(raw - raw.transpose(-1, -2))
    betas, camera = torch.randn(2, 4, 10), torch.randn(2, 4, 3)
    mask = torch.tensor([[True, True, True, False], [True, True, True, True]])
    args = (rgb, torch.zeros(1), rot, betas, camera)
    initial = model(*args, frame_valid=mask)
    for key, value in [('rotmat', rot), ('betas', betas), ('camera', camera)]:
        torch.testing.assert_close(initial[key], value, rtol=0, atol=0)
    loss = (initial['rotmat'] * torch.randn_like(rot)).sum()
    loss.backward()
    grad = model.temporal_refiner.rotation_delta_head.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    with torch.no_grad():
        model.temporal_refiner.rotation_delta_head.weight.normal_(0, 10)
        model.temporal_refiner.rotation_delta_head.bias.normal_(0, 10)
    model.zero_grad()
    result = model(*args, frame_valid=mask)
    torch.testing.assert_close(result['rotmat'][:, :, 0], rot[:, :, 0], rtol=0, atol=0)
    torch.testing.assert_close(result['betas'], betas, rtol=0, atol=0)
    torch.testing.assert_close(result['rotmat'][~mask], rot[~mask], rtol=0, atol=0)
    relative = result['rotmat'] @ rot.transpose(-1, -2)
    angles = torch.acos(((relative.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2).clamp(-1, 1))
    assert angles.max() <= torch.tensor(5 * torch.pi / 180) + 1e-5
    torch.testing.assert_close(relative @ relative.transpose(-1, -2),
                               torch.eye(3).expand_as(relative), atol=2e-6, rtol=2e-6)
    poisoned = rgb.clone()
    poisoned[~mask] = 1e5
    padded = model(poisoned, *args[1:], frame_valid=mask)
    torch.testing.assert_close(padded['rotmat'][mask], result['rotmat'][mask], rtol=0, atol=0)
    result['rotmat'].sum().backward()
    grad = model.temporal_refiner.blocks[0].linear1.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    clone = namespace['HumanRGBVaceGeometryTransformerV12Bounded'](config, temporal).eval()
    clone.load_compatible_state_dict(model.state_dict())
    torch.testing.assert_close(clone(*args, frame_valid=mask)['rotmat'], result['rotmat'])
    print('V12_BRANCH_PASS: identity initialization, fixed root/shape, 5-degree bound, SO(3), padding, finite head/encoder gradients, state roundtrip')


if __name__ == '__main__':
    main()
