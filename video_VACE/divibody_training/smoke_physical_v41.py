"""CPU-only numerical and real-checkpoint integration checks, unique outputs."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader
from models.human_decoder_parts_v41 import (
    model_from_payload, V41Config, FORMAT, damped_camera_step)
from divibody_training.ram_pose_stream_v41 import ShuffledPoseDatasetV10
from divibody_training.ram_parts_runtime_v41 import forward_batch, pose_losses
from divibody_training.parts_training_utils_v41 import make_optimizer, ModelEMA
from divibody_training.evaluate_standard_3dpw import create_smpl_models

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','smpl-model-root','pose-vace-dir','pose-hmr-dir','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump({'status':'RUNNING'},stream)
    result={'status':'FAILED','device':'cpu','checks':{}}
    try:
        torch.set_num_threads(4); torch.manual_seed(2026)
        from hmr2.models.smpl_wrapper import SMPL
        payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        cfg=V41Config(temporal_enabled=True)
        model=model_from_payload(payload,v41_config=cfg).float()
        models=create_smpl_models(args.smpl_model_root,torch.device('cpu'))
        model.bind_geometry(models[2])
        official=SMPL(model_path=str(args.smpl_model_root),gender='neutral',num_betas=10,
            joint_regressor_extra=str(args.smpl_model_root.parent/'SMPL_to_J19.pkl')).requires_grad_(False)
        model.bind_landmarks(official)
        dataset=ShuffledPoseDatasetV10(args.pose_vace_dir,args.pose_hmr_dir,2026,1)
        batch=next(iter(DataLoader(dataset,batch_size=1,num_workers=0)))
        model.eval()
        with torch.no_grad():
            actual=forward_batch(model,batch,torch.device('cpu'),'none')
            coarse=forward_batch(model.base,batch,torch.device('cpu'),'none')
        deltas={k:float((actual[k]-coarse[k]).abs().max()) for k in ('rotmat','betas','camera','direct_2d')}
        assert all(v==0 for v in deltas.values()),deltas
        result['checks']['zero_init_exact_equivalence']=deltas
        joints=torch.randn(2,3,17,3)*.1
        cam=torch.tensor([1.,.01,-.02]).expand(2,3,3)
        residual=(torch.randn(2,3,17,2)*.01).requires_grad_()
        delta=damped_camera_step(joints,cam,residual,torch.ones(2,3,17,1),.01)
        delta.square().sum().backward()
        assert torch.isfinite(delta).all() and torch.isfinite(residual.grad).all()
        result['checks']['physical_solver_finite_gradient']=True
        train_args=SimpleNamespace(local_lr=1e-4,geometry_lr=1e-4,camera_lr=1e-4,gate_lr=1e-4,
            weight_decay=1e-4,stage_2d_weight=.5,observation_weight=.5,uncertainty_weight=.01,
            visibility_weight=.02,anchor_weight=.1,heatmap_weight=.1,heat_projection_weight=.01,
            geometry_distill_weight=10.,pose_confidence_weight=.02)
        optimizer,names=make_optimizer(model,train_args)
        model.train(); prediction=forward_batch(model,batch,torch.device('cpu'),'none')
        losses=pose_losses(prediction,batch,official,train_args,torch.device('cpu'))
        losses['pose_2d'].backward()
        assert any(p.grad is not None for n,p in model.named_parameters() if n.startswith('evidence.'))
        assert model.evidence.reliability.weight.grad is not None
        assert float(model.evidence.reliability.weight.grad.abs().sum())>0
        assert all(p.grad is None for p in model.base.parameters())
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step(); model.eval()
        with torch.no_grad(): after=forward_batch(model,batch,torch.device('cpu'),'none')
        assert torch.equal(after['rotmat'],coarse['rotmat']) and torch.equal(after['betas'],coarse['betas'])
        assert float((after['camera']-coarse['camera']).abs().max())>0
        result['checks']['training_step_geometry_exact']=True
        optimizer.zero_grad(set_to_none=True)
        bad_batch=dict(batch); bad_batch['keypoints']=batch['keypoints'].clone()
        bad_batch['keypoints'][...,0,:]=float('nan')
        bad_prediction=forward_batch(model,bad_batch,torch.device('cpu'),'none')
        bad_losses=pose_losses(bad_prediction,bad_batch,official,train_args,torch.device('cpu'))
        bad_losses['pose_2d'].backward()
        assert torch.isfinite(bad_losses['pose_2d'])
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        result['checks']['nan_unannotated_gt_isolated']=True
        ema=ModelEMA(model)
        for _ in range(3): ema.update(model)
        assert all(torch.equal(v,model.base.state_dict()[k]) for k,v in ema.model.base.state_dict().items())
        result['checks']['ema_frozen_base_exact']=True
        result['checks']['trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
        result['checks']['losses']={k:float(v.detach()) for k,v in losses.items()}
        state=dict(format=FORMAT,model=model.state_dict(),v41_config=asdict(cfg),
            model_config=asdict(model.config),adaptation_config=asdict(model.adaptation_config),
            teacher_config=asdict(model.teacher_config))
        restored=model_from_payload(state).float().bind_geometry(models[2]).bind_landmarks(official).eval()
        with torch.no_grad(): again=forward_batch(restored,batch,torch.device('cpu'),'none')
        assert all(torch.equal(after[k],again[k]) for k in ('rotmat','betas','camera','direct_2d'))
        result['checks']['strict_checkpoint_roundtrip_exact']=True
        pose_cfg=V41Config(temporal_enabled=True,enable_pose=True)
        try: model_from_payload(state,v41_config=pose_cfg)
        except ValueError: pass
        else: raise AssertionError('Implicit pose migration allowed')
        migrated=model_from_payload(state,v41_config=pose_cfg,allow_pose_migration=True)
        assert migrated.correction.rotation.weight.requires_grad
        result['checks']['explicit_pose_migration_guard']=True
        result['checkpoint']=str(args.checkpoint)
        result['status']='COMPLETED'
    except Exception as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
        print(json.dumps(result,allow_nan=False),flush=True)

if __name__=='__main__': main()
