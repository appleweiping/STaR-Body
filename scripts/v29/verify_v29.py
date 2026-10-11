"""Policy unit checks and real SMPL/cache migration/gradient/roundtrip checks."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import torch


def unit():
    from divibody_training.v29_geometry_policy import merge_auxiliary,selection_report
    from divibody_training.v29_sequence_evaluation import sequence_acceleration
    body=torch.nn.Parameter(torch.tensor([1.,2.]));pose=torch.nn.Parameter(torch.ones(3))
    optimizer=torch.optim.AdamW([dict(params=[body],category='geometry'),dict(params=[pose],category='pose')])
    g={body:torch.tensor([1.,0.]),pose:None};a={body:torch.tensor([-2.,3.]),pose:torch.ones(3)}
    merge_auxiliary(optimizer,g,a,.1)
    assert float(torch.dot(body.grad,g[body]))>=1.-1e-6
    assert float((body.grad-g[body]).norm())<=.100001
    torch.testing.assert_close(pose.grad,a[pose])
    g[body]=torch.zeros(2);merge_auxiliary(optimizer,g,a,.1);assert float(body.grad.norm())==0
    args=SimpleNamespace(mpjpe_tolerance_mm=0.,pa_mpjpe_tolerance_mm=0.,pve_tolerance_mm=0.,
                         strict_3d_selection=True,geometry_only_validation=True)
    baseline=dict(mpjpe_mm=60.,pa_mpjpe_mm=40.,pve_mm=75.)
    assert not selection_report(dict(mpjpe_mm=61.,pa_mpjpe_mm=39.,pve_mm=74.),{},baseline,{},args,-3.)['accepted']
    assert selection_report(dict(mpjpe_mm=59.,pa_mpjpe_mm=39.,pve_mm=74.),{},baseline,{},args,-3.)['accepted']
    errors={f's/person_0/frame_{i:05d}':torch.full((14,3),float(i)) for i in range(6)}
    value,count=sequence_acceleration(errors);assert count==4 and value==0
    del errors['s/person_0/frame_00003'];assert sequence_acceleration(errors)[1]==1
    return dict(policy_checks='passed',sequence_boundary_checks='passed',selection_checks='passed')


def real(config,output):
    from divibody_training.evaluate_standard_3dpw import create_smpl_models
    from divibody_training.train_divibody_rgb_vace_hmr2a_v8 import load_pair
    from divibody_training.ram_parts_runtime_v28 import V10Dataset
    from divibody_training.ram_parts_runtime_v29 import forward_batch,geometric_losses
    from models.human_decoder_parts_v28 import model_from_payload as build_original
    from models.human_decoder_parts_v29 import model_from_payload,FORMAT
    from hmr2.models.smpl_wrapper import SMPL
    from torch.utils.data import default_collate
    import numpy as np
    torch.set_num_threads(4);device=torch.device('cpu')
    models=create_smpl_models(Path(config['smpl-model-root']),device)
    smpl_path=Path(config['smpl-model-root'])
    official=SMPL(model_path=str(smpl_path),gender='neutral',num_betas=10,
                  joint_regressor_extra=str(smpl_path.parent/'SMPL_to_J19.pkl'))
    payload=torch.load(config['init-checkpoint'],map_location='cpu',weights_only=False)
    model=model_from_payload(payload).bind_geometry(models[2]).bind_landmarks(official).eval()
    old=build_original(payload).bind_geometry(models[2]).bind_landmarks(official).eval()
    vace,hmr=load_pair(Path(config['train-vace-cache']),Path(config['train-hmr-sidecar']),'train')
    batch=default_collate([V10Dataset(vace,hmr)[0]])
    for k in ('rgb_tokens','vace_features'):batch[k]=batch[k].float()
    with torch.no_grad():
        baseline=forward_batch(old,batch,device,'none');prediction=forward_batch(model,batch,device,'none')
    for key in ('rotmat','betas','camera','direct_2d'):torch.testing.assert_close(prediction[key],baseline[key],atol=1e-6,rtol=0.)
    del old
    args=SimpleNamespace(**{k.replace('-','_'):v for k,v in config.items()})
    from divibody_training.parts_training_utils_v29 import make_optimizer
    optimizer,_=make_optimizer(model,args)
    model.train();prediction=forward_batch(model,batch,device,'none')
    regressor=torch.as_tensor(np.load(config['joint-regressor']),dtype=torch.float32)
    losses=geometric_losses(prediction,batch,models,regressor,args,device)
    assert torch.isfinite(losses['total']);losses['total'].backward()
    shape_gradient=float(model.shape_head[-1].weight.grad.norm())
    assert shape_gradient>0 and torch.isfinite(model.shape_head[-1].weight.grad).all()
    optimizer.step();model.eval()
    with torch.no_grad():prediction=forward_batch(model,batch,device,'none')
    assert float(prediction['shape_correction'].abs().max())<=model.adaptation_config.shape_bound
    r=prediction['rotmat'];torch.testing.assert_close(r@r.transpose(-1,-2),torch.eye(3).expand_as(r),atol=1e-4,rtol=0.)
    migrated=payload|dict(format=FORMAT,adaptation_config=asdict(model.adaptation_config),model=model.state_dict())
    torch.save(migrated,output/'roundtrip.pt')
    restored=model_from_payload(torch.load(output/'roundtrip.pt',map_location='cpu',weights_only=False)).bind_geometry(models[2]).bind_landmarks(official).eval()
    with torch.no_grad():other=forward_batch(restored,batch,device,'none')
    for k in ('rotmat','betas','camera'):torch.testing.assert_close(prediction[k],other[k],atol=1e-6,rtol=0.)
    assert not any(p.requires_grad for p in model.teacher.parameters())
    return dict(real_train_cache=True,real_smpl=True,zero_init_migration='passed',shape_gradient_norm=shape_gradient,
                so3='passed',native_roundtrip='passed',frozen_teacher='passed',performance_claim=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    result=unit();result.update(real(json.loads(args.config.read_text()),args.output));result['status']='passed'
    (args.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
