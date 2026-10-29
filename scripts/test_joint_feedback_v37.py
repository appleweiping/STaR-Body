"""Standalone CPU acceptance for new joint feedback; no performance claim."""
import json,sys,importlib.util
from pathlib import Path
import torch
p=Path(__file__).resolve().parents[1]/'video_VACE/models/joint_feedback_v37.py'
s=importlib.util.spec_from_file_location('joint_feedback_v37',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
torch.manual_seed(2026);torch.set_num_threads(2)
checks=[]
def check(name):checks.append(name);print(name,flush=True)
b,t=2,8
r=torch.randn(b,t,17,2)*.1;uv=torch.randn_like(r)*.2;lv=torch.full((b,t,17,1),-4.);v=torch.full_like(lv,.5);valid=torch.ones(b,t,dtype=torch.bool)
before=torch.get_rng_state().clone();net=m.make_joint_feedback(128)
assert torch.equal(before,torch.get_rng_state());check('isolated_module_preserves_cpu_rng')
y=net(r,uv,lv,v,valid);assert torch.equal(y,torch.zeros_like(y));check('zero_output_initialization_exact')
y.square().sum().backward();net.zero_grad(set_to_none=True)
# Nonzero downstream derivative at initialization must reach the output layer.
y=net(r,uv,lv,v,valid);(y*torch.randn_like(y)).sum().backward()
assert net.output.weight.grad is not None and net.output.weight.grad.abs().sum()>0;check('first_update_output_gradient_nonzero')
with torch.no_grad():net.output.weight.copy_(torch.eye(128))
net.zero_grad(set_to_none=True)
# Equal opposing residuals vanish in the old mean but remain distinguishable.
a=torch.zeros_like(r);a[:,:,5,0]=.4;a[:,:,6,0]=-.4
assert torch.equal(a.mean(-2),torch.zeros_like(a.mean(-2)))
base=net(torch.zeros_like(r),uv,lv,v,valid);changed=net(a,uv,lv,v,valid)
assert (changed-base).abs().max()>1e-5
assert (changed[:,:,16]-changed[:,:,17]).abs().max()>1e-5
check('opposing_left_right_residuals_preserve_identity')
for schema in ['posetrack','coco']:
 out=net(r,uv,lv,v,valid,schema=schema);assert torch.isfinite(out).all()
 out.square().mean().backward()
assert net.attention.in_proj_weight.grad is not None and net.attention.in_proj_weight.grad.abs().sum()>0
check('both_declared_schemas_and_inner_gradient')
net.eval();mask=valid.clone();mask[:,3]=False
bad=r.clone();bad[:,3]=float('nan');out=net(bad,uv,lv,v,mask)
assert torch.isfinite(out).all() and torch.equal(out[:,3],torch.zeros_like(out[:,3]))
one=r.clone();one[:,:,0]=float('nan');out=net(one,uv,lv,v,valid);assert torch.isfinite(out).all()
all_bad=torch.full_like(r,float('nan'));out=net(all_bad,uv,lv,v,valid)
assert torch.equal(out,torch.zeros_like(out));check('invalid_landmark_padding_all_invalid_finite_zero')
x=net(r,uv,lv,v,valid);rp=r.clone();rp[1,4]+=1;xp=net(rp,uv,lv,v,valid)
torch.testing.assert_close(x[0],xp[0],atol=0,rtol=0);torch.testing.assert_close(x[1,:4],xp[1,:4],atol=0,rtol=0)
torch.testing.assert_close(x[1,5:],xp[1,5:],atol=0,rtol=0)
check('no_cross_person_or_cross_frame_path_in_new_branch')
try:net(r,uv,lv,v,valid,schema='undeclared')
except ValueError:pass
else:raise AssertionError('Unknown schema accepted')
check('unknown_schema_rejected')
out=Path(sys.argv[1]);out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({'status':'passed','checks':checks,'benchmark':False,'optimizer_updates':0},indent=2))
