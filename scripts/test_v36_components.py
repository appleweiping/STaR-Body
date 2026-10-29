"""CPU numerical tests; no model score. Must run under the project torch env."""
import json,torch,importlib.util
from pathlib import Path
root=Path(__file__).resolve().parents[1]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
ik=load('ik_v36',root/'video_VACE/models/kinematic_ik_v36.py')
checks=[]
def passed(name):checks.append(name);print(name,flush=True)
torch.manual_seed(2026);torch.set_num_threads(2)
for a,b in [(torch.tensor([[1.,0,0]]),torch.tensor([[-1.,0,0]])),
            (torch.zeros(1,3),torch.ones(1,3)),(torch.randn(10,3),torch.randn(10,3))]:
 a.requires_grad_();b.requires_grad_();r=ik.align_vectors(a,b)
 assert torch.isfinite(r).all();torch.testing.assert_close(r@r.transpose(-1,-2),torch.eye(3).expand_as(r),atol=2e-4,rtol=0)
 assert (torch.linalg.det(r)>.999).all()
 r.square().sum().backward();assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
passed('vector_alignment_zero_opposite_random_and_backward')
rest=torch.randn(2,24,3)*.2;local=ik.exp_axis(torch.randn(2,24,3)*.1)
target,_=ik.forward_kinematics(local,rest)
solved=ik.residual_ik(local,rest,target,torch.zeros(2,24))
torch.testing.assert_close(local,solved,atol=3e-5,rtol=0)
passed('residual_ik_preserves_matching_pose_and_twist')
target=target.detach().requires_grad_();solved=ik.residual_ik(local,rest,target,torch.zeros(2,24));solved[...,0,1].sum().backward()
assert torch.isfinite(target.grad).all();passed('ik_target_gradient_finite')
# Extract pure mask function without importing the heavy model runtime.
import ast
tree=ast.parse((root/'video_VACE/models/part_time_refiner_v36.py').read_text())
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='temporal_allowed')
space={'torch':torch};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'mask','exec'),space)
mask,ind=space['temporal_allowed'](torch.tensor([[0,1,1,3,4,5,6,7]]),torch.tensor([[1,1,1,1,1,0,1,1]],dtype=torch.bool))
assert not mask[0,0,3] and not mask[0,1,2] and not mask[0,4,6] and mask[0,6,7] and not ind[0,2]
passed('time_mask_duplicates_gaps_padding_isolation')
# A duplicated source frame must not sever the neighboring unique frames.
m,ind=space['temporal_allowed'](torch.tensor([[0,2,2,4,6,8,10,12]]),torch.ones(1,8,dtype=torch.bool),2,1)
assert m[0,0,3] and m[0,1,3] and not m[0,0,2] and not ind[0,2]
passed('audited_delta2_delta3_and_duplicate_bridge')
# Execute only the pure attention component; no heavy vendor/model imports.
block_node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PartTimeBlock')
space['nn']=torch.nn
exec(compile(ast.fix_missing_locations(ast.Module(body=[block_node],type_ignores=[])),'block','exec'),space)
block=space['PartTimeBlock'](32,source_stride=2,source_jitter=1).eval()
with torch.no_grad():
 block.project.weight.copy_(torch.eye(32));block.project.bias.zero_()
 x=torch.randn(1,8,6,32);ids=torch.tensor([[0,2,4,6,8,10,12,14]]);valid=torch.ones(1,8,dtype=torch.bool)
 y=block(x,ids,valid);perturbed=x.clone();perturbed[:,1]+=torch.randn_like(perturbed[:,1])
 changed=block(perturbed,ids,valid)
 assert (y[:,0]-changed[:,0]).abs().max()>1e-5,'Temporal neighbor path must actually respond'
 gap_ids=torch.tensor([[0,2,4,10,12,14,16,18]])
 y=block(x,gap_ids,valid);perturbed=x.clone();perturbed[:,4]+=torch.randn_like(perturbed[:,4])
 changed=block(perturbed,gap_ids,valid)
 torch.testing.assert_close(y[:,:3],changed[:,:3],atol=0,rtol=0)
 valid[:,4]=False;y=block(x,ids,valid);perturbed=x.clone();perturbed[:,4]+=torch.randn_like(perturbed[:,4])
 changed=block(perturbed,ids,valid);torch.testing.assert_close(y[:,:4],changed[:,:4],atol=0,rtol=0)
passed('actual_temporal_sensitivity_and_gap_padding_isolation')

report={'status':'passed','checks':checks,'benchmark':False,'device':'cpu'}
output=Path(__import__('sys').argv[1]);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2))
