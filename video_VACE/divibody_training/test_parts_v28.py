"""Regression checks against real v27 weights and real SMPL assets on the server."""
import argparse
import copy
import json
import math
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import torch
from models.human_decoder_parts_v28 import model_from_payload,FORMAT,axis_angle_matrix,frozen_rotation_log
from models.body_projection_v28 import select_landmarks,project_landmarks
from divibody_training.evaluate_standard_3dpw import create_smpl_models
from divibody_training.ram_parts_runtime_v28 import pose_losses
from divibody_training.parts_training_utils_v28 import make_optimizer,ModelEMA
from divibody_training.train_ram_joint_sharded_pose_geometry_v7 import projected_posetrack

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--smpl-model-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.manual_seed(2026)
    torch.set_num_threads(4)
    device=torch.device('cuda:0')
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=model_from_payload(payload).to(device)
    native=create_smpl_models(args.smpl_model_root,device)
    from hmr2.models.smpl_wrapper import SMPL
    official=SMPL(model_path=str(args.smpl_model_root),gender='neutral',num_betas=10,
        joint_regressor_extra=str(args.smpl_model_root.parent/'SMPL_to_J19.pkl')).to(device)
    model.bind_geometry(native[2]).bind_landmarks(official)
    checks={}
    uv=torch.arange(50,dtype=torch.float32,device=device).reshape(1,25,2)
    coco=select_landmarks(uv,'coco'); pt=select_landmarks(uv,'posetrack')
    assert torch.equal(coco[:,1],uv[:,16]) and torch.equal(coco[:,2],uv[:,15])
    assert torch.equal(pt[:,1],uv[:,1]) and torch.equal(pt[:,2],2*uv[:,0]-uv[:,1])
    checks['coco_eyes_vs_posetrack_neck_head']=True
    aa=torch.tensor([[0.,0.,0.],[math.pi,0.,0.],[0.,math.pi-1e-5,0.],[.2,-.4,.1]],device=device)
    r=axis_angle_matrix(aa)
    assert torch.allclose(axis_angle_matrix(frozen_rotation_log(r)),r,atol=2e-5)
    checks['so3_log_identity_and_pi']=True
    cfg=model.config
    b,t=1,8
    x=[torch.randn(b,t,192,cfg.rgb_input_dim,device=device),
       torch.randn(b,2,4,4,cfg.vace_input_dim,device=device),
       torch.eye(3,device=device).reshape(1,1,1,3,3).repeat(b,t,24,1,1),
       torch.zeros(b,t,10,device=device),
       torch.tensor([1.1,.01,-.02],device=device).reshape(1,1,3).repeat(b,t,1)]
    valid=torch.tensor([[True,True,False,False,False,False,False,False]],device=device)
    model.eval()
    with torch.no_grad():
        fallback=model(*x,frame_valid=valid,force_anchor=True)
        for key,input_index in [('rotmat',2),('betas',3),('camera',4)]:
            assert torch.equal(fallback[key],x[input_index]),key
        output=model(*x,frame_valid=valid)
        assert torch.allclose(output['direct_2d'],projected_posetrack(output,official),atol=1e-5)
        assert torch.equal(output['rotmat'][:,2],x[2][:,2])
        static=model(*x,frame_valid=valid,source_frame_ids=torch.zeros(b,t,device=device,dtype=torch.long),landmark_schema='coco')
        assert (static['modality_routes'][...,2]==0).all()
        assert static['independent_frames'].sum()==1
        assert torch.allclose(static['direct_2d'],project_landmarks(static,official,'coco'),atol=1e-5)
        eye=torch.eye(3,device=device)
        assert torch.allclose(output['rotmat']@output['rotmat'].transpose(-1,-2),eye.expand_as(output['rotmat']),atol=1e-4)
        assert (torch.det(output['rotmat'])>.999).all() and (output['camera'][...,0]>0).all()
    checks.update(exact_hmr_anchor_fallback=True,padded_output_fallback=True,
        posetrack_projection_parity=True,static_independent_evidence=True,proper_rotations_and_camera=True)
    try:
        model(*x,frame_valid=torch.zeros_like(valid))
    except ValueError:
        checks['all_invalid_rejected']=True
    else:
        raise AssertionError('All-invalid clip accepted')
    config=SimpleNamespace(local_lr=1e-4,geometry_lr=1e-4,camera_lr=1e-4,gate_lr=1e-4,weight_decay=1e-4,
        stage_2d_weight=.5,observation_weight=.5,uncertainty_weight=.01,visibility_weight=.02,anchor_weight=.1)
    optimizer,_=make_optimizer(model,config)
    target=output['anchor_2d'].detach()+torch.tensor([.02,-.01],device=device)
    gt=torch.cat(((target+.5)*96,torch.ones(b,t,17,1,device=device)),-1)
    # Missing annotations with NaN coordinates must not poison losses/gradients.
    gt[:,:,0,:2]=float('nan'); gt[:,:,0,2]=0
    batch=dict(keypoints=gt,target_valid=valid,frame_owner=valid,
               visibility_targets=torch.ones(b,t,17,device=device))
    model.train()
    gradients={}
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            pred=model(*x,frame_valid=valid)
        loss=pose_losses(pred,batch,official,config,device)['pose_2d']
        assert torch.isfinite(loss)
        loss.backward()
        for name,parameter in model.named_parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all(),name
        assert all(p.grad is None for p in model.teacher.parameters())
        for prefix in ['observation.coordinate.0','observation.log_variance','observation.visibility',
                       'refiner.rotation','refiner.camera','refiner.gate','refiner.route','refiner.vace_projection','refiner.offset']:
            gradients[prefix]=sum(float(p.grad.abs().sum()) for n,p in model.named_parameters()
                                   if n.startswith(prefix) and p.grad is not None)
        optimizer.step()
    assert all(v>0 for v in gradients.values()),gradients
    checks.update(bf16_backward_finite=True,nan_missing_keypoints_safe=True,all_new_paths_receive_gradients=True,frozen_teacher_no_gradients=True)
    ema=ModelEMA(model)
    ema.update(model)
    assert ema.model._official_smpl is official and ema.model.teacher._neutral_smpl is native[2]
    checks['ema_shares_frozen_assets']=True
    saved=dict(format=FORMAT,model_config=asdict(model.config),adaptation_config=asdict(model.adaptation_config),
        teacher_config=asdict(model.teacher_config),model=model.state_dict())
    restored=model_from_payload(saved).to(device).bind_geometry(native[2]).bind_landmarks(official).eval()
    model.eval()
    with torch.no_grad():
        before=model(*x,frame_valid=valid)['direct_2d']
        after=restored(*x,frame_valid=valid)['direct_2d']
    assert torch.equal(before,after)
    checks['strict_checkpoint_roundtrip']=True
    result=dict(status='V28_REGRESSION_OK',checks=checks,gradient_l1=gradients,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/2**20,
        note='Synthetic features with real checkpoint/SMPL; follow with real-cache training smoke.')
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
