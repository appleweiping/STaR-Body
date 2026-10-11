"""Bounded real train-window inference/gradient/checkpoint checks, no training."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import default_collate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.root.resolve();args.output.mkdir(exist_ok=False)
    torch.manual_seed(2026);torch.set_num_threads(4);device=torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(10*1024**3/torch.cuda.get_device_properties(0).total_memory)
    from models.human_decoder_spatiotemporal_v30 import model_from_payload,FORMAT,AdaptationConfig,rotation_6d
    from divibody_training.ram_parts_runtime_v30 import V10Dataset,forward_batch,geometric_losses
    from divibody_training.parts_training_utils_v30 import make_optimizer
    from divibody_training.evaluate_standard_3dpw import create_smpl_models
    from hmr2.models.smpl_wrapper import SMPL
    plan=json.loads((root/'training_runs/v29y_strong_rgb_20261006/plan.json').read_text())['strong_motion_40']
    vace=torch.load(plan['train-vace-cache'],map_location='cpu',weights_only=False,mmap=True)
    hmr=torch.load(plan['train-hmr-sidecar'],map_location='cpu',weights_only=False,mmap=True)
    if vace['signature']['split']!='train' or hmr['signature']['split']!='train':raise ValueError('Train-only inputs required')
    sample=V10Dataset(vace,hmr)[0];batch=default_collate([sample])
    for key in ('rgb_tokens','vace_features'):batch[key]=batch[key].float()
    checkpoint=root.parent/'v28d_parallel/training_runs/coco_full10k_train_20261001/run/best.pt'
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    cfg=AdaptationConfig(iterations=3)
    model=model_from_payload(payload,cfg).to(device).eval()
    models=create_smpl_models(Path(plan['smpl-model-root']),device)
    official=SMPL(model_path=plan['smpl-model-root'],gender='neutral',num_betas=10,
        joint_regressor_extra=str(Path(plan['smpl-model-root']).parent/'SMPL_to_J19.pkl')).to(device).requires_grad_(False).eval()
    model.bind_geometry(models[2]).bind_landmarks(official)
    regressor=torch.tensor(np.load(plan['joint-regressor'],allow_pickle=False),dtype=torch.float32,device=device)
    loss_args=SimpleNamespace(**{key.replace('-','_'):value for key,value in plan.items()})
    with torch.no_grad():
        expected=forward_batch(model.reference,batch,device,'none')
        actual=forward_batch(model,batch,device,'none')
        differences={key:float((actual[key]-expected[key]).abs().max()) for key in ('rotmat','betas','camera')}
        for key in differences:torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=1e-6)
        rr=actual['rotmat'];orthogonality=float((rr@rr.transpose(-1,-2)-torch.eye(3,device=device)).abs().max())
        if orthogonality>1e-4 or not torch.allclose(torch.det(rr),torch.ones_like(torch.det(rr)),atol=1e-4,rtol=0):raise ValueError('Invalid rotations')
        for value in [torch.zeros(5,6,device=device),torch.tensor([[1.,0.,0.,2.,0.,0.]],device=device)]:
            r=rotation_6d(value)
            torch.testing.assert_close(r@r.transpose(-1,-2),torch.eye(3,device=device).expand_as(r),atol=1e-5,rtol=0)
    optimizer,names=make_optimizer(model,loss_args)
    # Nonzero new heads expose trunk/local/time gradients; the production
    # initializer remains zero and no optimizer update is performed here.
    with torch.no_grad():
        for head in (model.body_head,model.root_head,model.shape_head,model.camera_head):head.weight.normal_(std=1e-4)
    model.train();pred=forward_batch(model,batch,device,'none')
    loss=geometric_losses(pred,batch,models,regressor,loss_args,device)['total'];loss.backward()
    norms={prefix:float(sum(p.grad.float().square().sum() for name,p in model.named_parameters()
        if name.startswith(prefix) and p.grad is not None).sqrt()) for prefix in ('rgb_projection.','blocks.','feedback_projection.','body_head.','root_head.','shape_head.')}
    if not torch.isfinite(loss) or any(not np.isfinite(value) or value<=0 for value in norms.values()):raise ValueError('Missing/nonfinite actual geometry gradients')
    if any(p.grad is not None or p.requires_grad for p in model.reference.parameters()):raise ValueError('Frozen reference received gradients')
    native=dict(format=FORMAT,model_config=asdict(model.config),adaptation_config=asdict(cfg),
                reference_spec=model.reference_spec,model=model.state_dict())
    restored=model_from_payload(native).to(device).bind_geometry(models[2]).bind_landmarks(official).eval()
    model.eval()
    with torch.no_grad():
        output=forward_batch(model,batch,device,'none');repeat=forward_batch(restored,batch,device,'none')
        for key in ('rotmat','betas','camera'):torch.testing.assert_close(output[key],repeat[key],atol=0,rtol=0)
    malformed=native|dict(model={key:value for key,value in native['model'].items() if key!='body_head.weight'})
    try:model_from_payload(malformed)
    except (RuntimeError,ValueError):strict_rejection=True
    else:raise ValueError('Malformed native checkpoint accepted')
    original=json.loads((root/'original_code_hashes.json').read_text())
    if any(hashlib.sha256((root/name).read_bytes()).hexdigest()!=value for name,value in original.items()):raise ValueError('Original source changed')
    result=dict(status='passed',frames=int(sample['target_valid'].sum()),split='train',random_input_features=False,
        initialization_max_difference=differences,orthogonality_max_error=orthogonality,gradient_norms=norms,
        frozen_reference_gradient_free=True,native_restore_exact=True,malformed_native_rejected=strict_rejection,
        optimizer_parameters=len(names),optimizer_step_performed=False,formal_training_started=False,
        model_input_keys=['rgb_tokens','vace_features','base_rotmat','base_betas','base_camera','source_frame_ids','target_valid'],
        no_3d_ground_truth_in_model_inputs=True,original_python_files_unchanged=len(original),sota_status='not_established')
    with (args.output/'result.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
