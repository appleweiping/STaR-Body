"""CPU checks for checkpoint compatibility, gradient isolation and selection."""
from types import SimpleNamespace
import torch
from models.human_decoder_rgb_vace_v8_hmr2a import (
    HumanRGBVaceGeometryTransformerV8HMR2A, RGBVaceHMR2AConfig)
from models.human_decoder_rgb_vace_v9_isolated import HumanRGBVaceGeometryTransformerV9Isolated
from divibody_training.train_ram_geometry_priority_v9 import make_optimizer, selection_score


def main():
    torch.set_num_threads(2)
    cfg = RGBVaceHMR2AConfig(hidden_dim=32, num_heads=4, rgb_input_dim=16,
                            vace_input_dim=16, dropout=0)
    old = HumanRGBVaceGeometryTransformerV8HMR2A(cfg).eval()
    new = HumanRGBVaceGeometryTransformerV9Isolated(cfg).eval()
    new.load_state_dict(old.state_dict(), strict=True)
    inputs = dict(rgb_tokens=torch.randn(1,8,4,16),
                  vace_features=torch.randn(1,5,2,2,16),
                  base_rotmat=torch.eye(3).expand(1,8,24,3,3).clone(),
                  base_betas=torch.zeros(1,8,10), base_camera=torch.ones(1,8,3))
    with torch.no_grad():
        a, b = old(**inputs), new(**inputs)
    for k in ('rotmat','betas','camera'):
        torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)
    args = SimpleNamespace(unfreeze_last_vace=True, camera_lr=2e-6,
                           geometry_lr=1e-6, fusion_lr=1e-7, weight_decay=0)
    optimizer, names = make_optimizer(new,args,True)
    assert any(n.startswith('vace_blocks.1.') for n in names)
    assert not any(n.startswith('vace_blocks.0.') for n in names)
    # Nonzero heads ensure this check would catch an accidentally live route.
    with torch.no_grad():
        new.camera_residual_head.weight.normal_(0,0.1)
        new.rotation_residual_head.weight.normal_(0,0.01)
        new.temporal_gates.fill_(0.1)
    prediction = new(**inputs)
    isolated = new.camera_only_prediction(prediction, inputs['base_camera'])
    isolated['camera'].square().sum().backward()
    assert new.camera_residual_head.weight.grad.abs().sum() > 0
    assert all(p.grad is None for n,p in new.named_parameters()
               if not n.startswith('camera_residual_head.'))
    optimizer.zero_grad(set_to_none=True)
    new(**inputs)['rotmat'].square().mul(torch.randn(1,8,24,3,3)).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in new.vace_blocks[-1].parameters())
    baseg = dict(mpjpe_mm=49.285,pve_mm=63.884)
    basep = {'PCKn@0.05':0.6268,'PCKn@0.10':0.8582}
    assert selection_score(basep,dict(mpjpe_mm=49,pve_mm=63.5),basep,baseg,.5) > -49.285
    assert selection_score(basep,dict(mpjpe_mm=48,pve_mm=65),basep,baseg,.5) == float('-inf')
    assert selection_score(dict(basep,**{'PCKn@0.05':.60}),baseg,basep,baseg,.5) == float('-inf')
    print('V9_CHECKS_PASS: identical initialization; isolated camera gradients; fusion gradients; 3D selection guards')

if __name__ == '__main__':
    main()
