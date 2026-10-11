"""Regression checks against the actual v9 implementation (run on server32)."""
from dataclasses import asdict
import argparse
import io
import json
from types import SimpleNamespace
import torch
from models.human_decoder_v9_adaptation_v13 import (
    HumanV9AdaptationV13, AdaptationConfig, RGBVaceHMR2AConfig, FORMAT)
from divibody_training.ram_adaptation_runtime_v13 import model_from_payload
from divibody_training.train_v9_adaptation_v13 import make_optimizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.manual_seed(2026)
    device = torch.device(args.device)
    cfg = RGBVaceHMR2AConfig(rgb_input_dim=16, vace_input_dim=16,
        hidden_dim=32, num_heads=4, rgb_layers=1, vace_layers=1)
    for amp in (False, True):
        model = HumanV9AdaptationV13(cfg, AdaptationConfig('shared')).to(device)
        with torch.no_grad():
            for name, p in model.reference.named_parameters():
                if 'residual_head' in name:
                    p.add_(torch.randn_like(p) * 0.01)
            model.reference.temporal_gates.fill_(0.03)
        inputs = (torch.randn(2,8,4,16,device=device), torch.randn(2,2,2,2,16,device=device),
            torch.eye(3,device=device).expand(2,8,24,3,3).clone(),
            torch.randn(2,8,10,device=device)*0.1, torch.randn(2,8,3,device=device))
        valid = torch.ones(2,8,dtype=torch.bool,device=device)
        valid[0,6:] = False
        model.train()
        assert all(not m.training for m in model.reference.modules())
        frozen = {n:p.clone() for n,p in model.reference.state_dict().items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            baseline = model.reference(*inputs)
            pred = model(*inputs, frame_valid=valid)
            features = model._read_features(*inputs[:2])
        for key in ('rotmat','betas','camera'):
            assert torch.equal(pred[key], baseline[key]), (amp,key,'initial mismatch')
        torch.testing.assert_close(features, baseline['fused_queries'], rtol=0, atol=0)
        optimizer, names = make_optimizer(model, SimpleNamespace(geometry_lr=1e-4,weight_decay=1.))
        loss = (pred['rotmat'] - torch.randn_like(pred['rotmat'])).square().mean()
        loss.backward()
        assert all(p.grad is None for p in model.reference.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.limb_head_delta.parameters())
        assert model.limb_head_delta.weight.grad.abs().sum() > 0
        optimizer.step()
        model.set_position_mix(0.5)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp):
            after = model(*inputs,frame_valid=valid)
        assert torch.equal(after['rotmat'][:,:,:1],baseline['rotmat'][:,:,:1])
        assert torch.equal(after['rotmat'][~valid],baseline['rotmat'][~valid])
        assert not torch.equal(after['rotmat'][:,:,1:],baseline['rotmat'][:,:,1:])
        for key in ('betas','camera'):
            assert torch.equal(after[key],baseline[key])
        for n,p in model.reference.state_dict().items():
            assert torch.equal(p,frozen[n]), n
        matrices=after['rotmat']
        torch.testing.assert_close(matrices.transpose(-1,-2)@matrices,
            torch.eye(3,device=device).expand_as(matrices), rtol=1e-5,atol=1e-5)
        payload = dict(format=FORMAT,model_config=asdict(cfg),
            adaptation_config=asdict(model.adaptation_config),model=model.state_dict())
        buffer = io.BytesIO()
        torch.save(payload,buffer)
        buffer.seek(0)
        restored = model_from_payload(torch.load(buffer,weights_only=False)).to(device)
        assert float(restored.position_mix) == 0.5
        restored.eval()
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp):
            again = restored(*inputs,frame_valid=valid)
        assert torch.equal(after['rotmat'],again['rotmat'])
        model.set_position_mix(1.)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp):
            same1 = model(*inputs,frame_valid=valid)
            same2 = model(*inputs,frame_valid=valid,position_offset=4)
        torch.testing.assert_close(same1['rotmat'],same2['rotmat'],rtol=1e-4,atol=1e-4)
        for wrong in (-1.,2.,float('nan')):
            try: model.set_position_mix(wrong)
            except ValueError: pass
            else: raise AssertionError('Invalid mix accepted')
        print(json.dumps(dict(amp=amp, exact_v9_initialization=True,
            frozen_reference_unchanged=True, gradient_and_roundtrip=True,
            trainable_parameters=names)),flush=True)
    print('V13_REGRESSION_OK',flush=True)

if __name__ == '__main__':
    main()
