"""Check v15 against real v9 modules, including spatial and gradient behavior."""
import io
from dataclasses import asdict
from types import SimpleNamespace
import torch
from models.human_decoder_parts_v15 import (HumanV9AdaptationV15,AdaptationConfig,
    RGBVaceHMR2AConfig,FORMAT,PARTS)
from divibody_training.ram_parts_runtime_v15 import model_from_payload
from divibody_training.train_parts_v15 import make_optimizer

def main():
    device=torch.device('cuda:0')
    assert sorted(j for p in PARTS for j in p)==list(range(1,24))
    for amp in (False,True):
        torch.manual_seed(2026)
        cfg=RGBVaceHMR2AConfig(rgb_input_dim=16,vace_input_dim=16,hidden_dim=32,
                             num_heads=4,rgb_layers=1,vace_layers=1)
        parts=AdaptationConfig(hidden_dim=32,grid_height=2,grid_width=2)
        model=HumanV9AdaptationV15(cfg,parts).to(device)
        with torch.no_grad():
            for n,p in model.reference.named_parameters():
                if 'residual_head' in n:p.add_(torch.randn_like(p)*.01)
        inputs=(torch.randn(2,8,4,16,device=device),torch.randn(2,2,2,2,16,device=device),
                torch.eye(3,device=device).expand(2,8,24,3,3).clone(),
                torch.randn(2,8,10,device=device)*.1,torch.randn(2,8,3,device=device))
        valid=torch.ones(2,8,dtype=torch.bool,device=device);valid[0,6:]=False
        before={n:p.clone() for n,p in model.reference.state_dict().items()}
        model.train()
        assert not any(m.training for m in model.reference.modules())
        optimizer,_=make_optimizer(model,SimpleNamespace(geometry_lr=1e-3,weight_decay=.1))
        target=torch.randn(2,8,24,3,3,device=device)
        for step in range(3):
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=amp):
                out=model(*inputs,frame_valid=valid)
                baseline=model.reference(*inputs)
            if step==0:
                for k in ('rotmat','betas','camera'):assert torch.equal(out[k],baseline[k]),k
            assert torch.equal(out['rotmat'][:,:,:1],baseline['rotmat'][:,:,:1])
            assert torch.equal(out['rotmat'][~valid],baseline['rotmat'][~valid])
            assert torch.equal(out['betas'],baseline['betas'])
            assert torch.equal(out['camera'],baseline['camera'])
            (out['rotmat']-target).square().mean().backward()
            assert all(p.grad is None for p in model.reference.parameters())
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.part_branch.parameters())
            if step>0:assert model.part_branch.queries.grad.abs().sum()>0
            optimizer.step()
        assert all(torch.equal(v,model.reference.state_dict()[n]) for n,v in before.items())
        model.eval()
        with torch.no_grad():
            context=baseline['fused_queries']
            one=model.part_branch(inputs[0],context)
            two=model.part_branch(inputs[0].flip(2),context)
            assert (one-two).abs().max()>0
        payload=dict(format=FORMAT,model_config=asdict(cfg),adaptation_config=asdict(parts),model=model.state_dict())
        stream=io.BytesIO();torch.save(payload,stream);stream.seek(0)
        restored=model_from_payload(torch.load(stream,weights_only=False)).to(device).eval()
        with torch.no_grad():
            assert torch.equal(model(*inputs)['rotmat'],restored(*inputs)['rotmat'])
            for h in model.part_branch.heads:h.bias.fill_(1000)
            saturated=model(*inputs,frame_valid=valid)
            angles=saturated['part_delta_axis_angle'].norm(dim=-1)
            assert float(angles.max())<=torch.pi*10/180+1e-6
            r=saturated['rotmat']
            torch.testing.assert_close(r.transpose(-1,-2)@r,torch.eye(3,device=device).expand_as(r),rtol=1e-5,atol=1e-5)
        print(f'PASS amp={amp}: exact v9, gradients, frozen outputs, spatial sensitivity, angle bound, roundtrip',flush=True)
    print('V15_REGRESSION_OK',flush=True)
if __name__=='__main__':main()
