"""Verify real train-only stitched boundary gradients before starting research."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    c=json.loads(a.config.read_text());device=torch.device('cuda:0');torch.set_num_threads(4)
    from divibody_training.train_divibody_rgb_vace_hmr2a_v8 import load_pair
    from divibody_training.ram_parts_runtime_v29 import V10Dataset
    from divibody_training.v29_boundary_training import build_boundary_pool,boundary_training_loss,_sample_indices
    from divibody_training.evaluate_standard_3dpw import create_smpl_models
    from models.human_decoder_parts_v29 import model_from_payload
    from hmr2.models.smpl_wrapper import SMPL
    v,h=load_pair(Path(c['train-vace-cache']),Path(c['train-hmr-sidecar']),'train')
    data=V10Dataset(v,h);pool=build_boundary_pool(data)
    assert pool.audit['full_contiguous_triplets']==22481
    assert pool.audit['missing_from_any_within_window']==5408
    assert pool.audit['boundary_triplets']==5548
    from divibody_training.v29_boundary_partition import audit_boundary_partition as audit_partition
    partition=audit_partition(data,pool)
    assert partition['canonical_within_triplets']==16933
    assert partition['canonical_within_triplets']+partition['canonical_boundary_triplets']==22481
    ids=_sample_indices(len(pool.triplets),step=7,seed=2026,count=2,identity_sha256=pool.identity_sha256,task='boundary')
    assert ids==_sample_indices(len(pool.triplets),step=7,seed=2026,count=2,identity_sha256=pool.identity_sha256,task='boundary')
    bad={**v,'signature':{**v['signature'],'split':'validation'}}
    try:build_boundary_pool(V10Dataset(bad,h))
    except ValueError:pass
    else:raise AssertionError('Validation labels were not rejected')
    models=create_smpl_models(Path(c['smpl-model-root']),device)
    for m in models.values():m.requires_grad_(False)
    path=Path(c['smpl-model-root']);official=SMPL(model_path=str(path),gender='neutral',num_betas=10,
        joint_regressor_extra=str(path.parent/'SMPL_to_J19.pkl')).to(device).requires_grad_(False)
    payload=torch.load(c['init-checkpoint'],map_location='cpu',weights_only=False)
    model=model_from_payload(payload).to(device).bind_geometry(models[2]).bind_landmarks(official).train()
    regressor=torch.as_tensor(np.load(c['joint-regressor']),dtype=torch.float32,device=device)
    loss,info=boundary_training_loss(model,data,pool,models,regressor,device,'bf16',step=7,seed=2026,triplets_per_step=2)
    assert torch.isfinite(loss);loss.backward()
    norm=sum(float(p.grad.float().square().sum()) for p in model.parameters() if p.requires_grad and p.grad is not None)**.5
    assert norm>0
    result=dict(status='passed',train_only=True,real_rgb_vace_hmr=True,real_smpl=True,
        boundary_audit=pool.audit,partition=partition,pool_identity_sha256=pool.identity_sha256,
        loss_m_per_frame2=float(loss.detach()),gradient_norm=norm,sampled=info,
        deterministic_sampling=True,validation_targets_rejected=True,performance_claim=False)
    (a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
