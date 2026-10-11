"""Geometry-coordinate, gradient and frozen-reference regression on server32."""
import io
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace
import torch
from models.human_decoder_parts_v16 import (HumanV9AdaptationV16,AdaptationConfig,
    RGBVaceHMR2AConfig,FORMAT,project_joint_grid,sample_joint_neighborhood)
from divibody_training.ram_parts_runtime_v16 import model_from_payload
from divibody_training.train_parts_v16 import make_optimizer
from divibody_training.evaluate_standard_3dpw import create_smpl_models
from hmr2.utils.geometry import perspective_projection

def check_coordinates(device):
    # Coordinate ramps independently encode token centers in square-crop pixels.
    yy,xx=torch.meshgrid(torch.arange(16,device=device),torch.arange(12,device=device),indexing='ij')
    memory=torch.stack((32+(xx+.5)*16,(yy+.5)*16)).float()[None]
    camera=torch.tensor([[1.,0.,0.]],device=device)
    # With z=0 and s=1: projected normalized square xy equals joints.xy / 2.
    pixels=torch.tensor([[[88.,104.],[168.,168.]]],device=device)
    joints=torch.cat(((pixels-128.)/128.,torch.zeros(1,2,1,device=device)),dim=-1)
    grid,valid=project_joint_grid(joints,camera)
    samples,mask=sample_joint_neighborhood(memory,grid,valid)
    torch.testing.assert_close(samples[:,:,4],pixels,atol=2e-5,rtol=0)
    assert mask.all()
    translation=torch.tensor([[0.,0.,10000./256.]],device=device)
    square=perspective_projection(joints,translation=translation,
        focal_length=torch.full((1,2),5000.,device=device))/256.
    torch.testing.assert_close(grid,square*grid.new_tensor([8/3,2]))
    bad=joints.clone();bad[0,0,2]=-100.;bad[0,1,0]=float('nan')
    g,v=project_joint_grid(bad,camera)
    sampled,m=sample_joint_neighborhood(memory,g,v)
    assert not m.any() and torch.isfinite(sampled).all() and sampled.abs().sum()==0
    off=torch.full_like(grid,20.)
    sampled,m=sample_joint_neighborhood(memory,off,valid)
    assert not m.any() and sampled.abs().sum()==0
    # An independently moved sampling point must move the returned image coordinate.
    moved=grid.clone();moved[...,0]+=2/12
    sampled,_=sample_joint_neighborhood(memory,moved,valid)
    torch.testing.assert_close(sampled[:,:,4,0],pixels[...,0]+16,atol=2e-5,rtol=0)
    print('COORDINATES_OK: crop, token centers, official projection, invalid depth/NaN/OOB',flush=True)

def main():
    device=torch.device('cuda:0')
    check_coordinates(device)
    root=Path('/mnt/32a56f7d-8270-4d4e-b9cb-fb8857bd7e93/testpro/xmj/sam-3d-body/divibody_vace')
    smpl=create_smpl_models(root/'checkpoints/4DHumans/extracted/data/smpl',device)[2]
    for amp in (False,True):
        torch.manual_seed(2026)
        cfg=RGBVaceHMR2AConfig(rgb_input_dim=16,vace_input_dim=16,hidden_dim=32,
                             num_heads=4,rgb_layers=1,vace_layers=1)
        parts=AdaptationConfig(hidden_dim=32)
        model=HumanV9AdaptationV16(cfg,parts).to(device).bind_geometry(smpl)
        with torch.no_grad():
            for n,p in model.reference.named_parameters():
                if 'residual_head' in n:p.add_(torch.randn_like(p)*.001)
        camera=torch.zeros(2,8,3,device=device);camera[...,0]=1.
        inputs=(torch.randn(2,8,192,16,device=device),torch.randn(2,2,2,2,16,device=device),
                torch.eye(3,device=device).expand(2,8,24,3,3).clone(),
                torch.randn(2,8,10,device=device)*.1,camera)
        valid=torch.ones(2,8,dtype=torch.bool,device=device);valid[0,6:]=False
        before={n:p.clone() for n,p in model.reference.state_dict().items()}
        model.train()
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
            for k in ('betas','camera'):assert torch.equal(out[k],baseline[k])
            (out['rotmat']-target).square().mean().backward()
            assert all(p.grad is None for p in model.reference.parameters())
            assert all(p.grad is None for p in smpl.parameters())
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.part_branch.parameters())
            if step>0:
                assert model.part_branch.local_projection.weight.grad.abs().sum()>0
                assert model.part_branch.queries.grad.abs().sum()>0
            optimizer.step()
        assert all(torch.equal(v,model.reference.state_dict()[n]) for n,v in before.items())
        assert not any('_neutral_smpl' in k for k in model.state_dict())
        model.eval()
        payload=dict(format=FORMAT,model_config=asdict(cfg),adaptation_config=asdict(parts),model=model.state_dict())
        stream=io.BytesIO();torch.save(payload,stream);stream.seek(0)
        restored=model_from_payload(torch.load(stream,weights_only=False)).to(device).bind_geometry(smpl).eval()
        with torch.no_grad():
            assert torch.equal(model(*inputs)['rotmat'],restored(*inputs)['rotmat'])
            for h in model.part_branch.heads:h.bias.fill_(1000)
            saturated=model(*inputs,frame_valid=valid)
            assert float(saturated['part_delta_axis_angle'].norm(dim=-1).max())<=torch.pi*10/180+1e-6
            r=saturated['rotmat']
            torch.testing.assert_close(r.transpose(-1,-2)@r,torch.eye(3,device=device).expand_as(r),rtol=1e-5,atol=1e-5)
        print(f'PASS amp={amp}: exact v9, local gradients, frozen outputs, angle bound, roundtrip',flush=True)
    print('V16_REGRESSION_OK',flush=True)

if __name__=='__main__':main()
