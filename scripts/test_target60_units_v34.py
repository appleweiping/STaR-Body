"""CPU-only numerical checks for Target60; no training cache or CUDA required."""
import argparse,json,math,sys,traceback
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
import torch
torch.set_num_threads(2)
from divibody_training.target60_pa_loss_v34 import detached_similarity_align
from divibody_training.geometry_metrics_3dpw import batch_similarity_transform
from divibody_training.target60_policy_v34 import selection_report
from types import SimpleNamespace
checks=[]
torch.manual_seed(2026)
pred=torch.randn(5,14,3,dtype=torch.float32,requires_grad=True)
target=torch.randn(5,14,3)
aligned=detached_similarity_align(pred,target)
expected=batch_similarity_transform(pred.detach(),target)
torch.testing.assert_close(aligned,expected,rtol=0,atol=1e-6)
checks.append('detached_PA_transform_matches_original_metric')
loss=torch.linalg.vector_norm(aligned-target,dim=-1).mean();loss.backward()
assert pred.grad is not None and torch.isfinite(pred.grad).all() and pred.grad.abs().sum()>0
checks.append('PA_gradient_finite_and_nonzero')
try:detached_similarity_align(torch.zeros(1,14,3),torch.randn(1,14,3))
except ValueError:pass
else:raise AssertionError('Degenerate PA not rejected')
checks.append('degenerate_PA_rejected_without_dropping_samples')
args=SimpleNamespace(target_mpjpe_mm=60.,target_pa_mpjpe_mm=37.3,mpjpe_tolerance_mm=0.,
 pa_mpjpe_tolerance_mm=0.,pve_tolerance_mm=0.,acceleration_ceiling_mm=21.51,acceleration_tolerance_mm=0.)
baseline=dict(mpjpe_mm=61.416250005,pa_mpjpe_mm=37.78266967,pve_mm=73.2866783,acceleration_error_mm_per_frame2=21.51)
good=baseline|dict(mpjpe_mm=59.9,pa_mpjpe_mm=37.2,pve_mm=73.,acceleration_error_mm_per_frame2=21.4)
r=selection_report(good,{},baseline,{},args,-1.1);assert r['accepted'] and r['target_met']
edge=selection_report(good|dict(mpjpe_mm=60.),{},baseline,{},args,-1.1);assert not edge['target_met']
bad=selection_report(good|dict(pve_mm=74.),{},baseline,{},args,-1.1);assert not bad['accepted']
checks+=['strict_dual_target_boundary','PVE_guard_cannot_be_bypassed']
result=dict(status='passed',checks=checks,device='cpu',torch=torch.__version__,model_forward_verified=False,training_started=False)
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
