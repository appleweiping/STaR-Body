"""CPU regression using real decoders; a synthetic SMPL surface isolates licensed assets.

Local: python test_parts_v20.py --local
Server: python -m divibody_training.test_parts_v20
These tests are not real-data training or official SMPL integration evaluation.
"""
import ast
import copy
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest

if '--local' in sys.argv:
    sys.argv.remove('--local')
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / '.validation_v12'))
    for package, directory in (('models', 'models'), ('divibody_training', 'divibody_training')):
        module = ModuleType(package)
        module.__path__ = [str(Path(__file__).resolve().parent),
                           str(root / 'github_push/divibody-vace' / directory)]
        sys.modules[package] = module

import torch
from torch import nn
from models.human_decoder_parts_v20 import (AdaptationConfig, FORMAT, HumanV9AdaptationV20,
    RGBVaceHMR2AConfig, SpatialPartBranch, model_from_payload)
from models.human_decoder_parts_v19 import HumanV9AdaptationV19, AdaptationConfig as V19Config
from models.human_decoder_rgb_vace_v9_isolated import HumanRGBVaceGeometryTransformerV9Isolated
from divibody_training.parts_training_utils_v20 import (
    make_optimizer, validate_cache_coverage, StepProbe, loss_gradient_contributions,
)


class SyntheticSMPL(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.register_buffer('joints', torch.stack((torch.linspace(-.3,.3,24),
                             torch.linspace(.2,-.2,24), torch.zeros(24)), dim=-1))

    def forward(self, global_orient, body_pose, betas, pose2rot=False):
        return SimpleNamespace(joints=self.scale*self.joints.expand(len(betas),-1,-1))


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(2026)
        self.cfg = RGBVaceHMR2AConfig(rgb_input_dim=8,vace_input_dim=8,hidden_dim=16,
            num_heads=4,rgb_layers=1,vace_layers=2,output_frames=2,dropout=0.)
        self.adapt = AdaptationConfig(hidden_dim=16)
        base = HumanRGBVaceGeometryTransformerV9Isolated(self.cfg).eval()
        with torch.no_grad():
            base.temporal_gates.copy_(torch.tensor([.00035,.0008]))
            # Nonzero pretrained heads are essential to test upstream fusion gradients.
            for name, parameter in base.named_parameters():
                if 'residual_head' in name and name.endswith('weight'):
                    parameter.add_(torch.randn_like(parameter)*.003)
        self.payload = dict(format='ram-geometry-priority-isolated-v9',
                            model_config=asdict(self.cfg), model=copy.deepcopy(base.state_dict()))
        # Match the frozen v19 reference so attention kernel dispatch is identical.
        self.base = base.requires_grad_(False)
        camera = torch.zeros(1,2,3)
        camera[...,0] = 1
        self.inputs = (torch.randn(1,2,192,8),torch.randn(1,2,2,2,8),
                       torch.eye(3).expand(1,2,24,3,3).clone(), torch.zeros(1,2,10), camera)
        self.args = SimpleNamespace(head_lr=1e-3, geometry_lr=1e-3, fusion_lr=1e-4,
                                    gate_lr=1e-3, weight_decay=.1)

    def model(self, **kwargs):
        return model_from_payload(self.payload, replace(self.adapt, **kwargs)).bind_geometry(SyntheticSMPL())

    def test_initialization_exact_fp32_and_bf16(self):
        for dtype in (None, torch.bfloat16):
            model = self.model().eval()
            with torch.no_grad(), torch.autocast('cpu',dtype=dtype or torch.bfloat16,enabled=dtype is not None):
                before = self.base(*self.inputs)
                after = model(*self.inputs)
            for key in ('rotmat','betas','camera','fused_queries','temporal_gate'):
                self.assertTrue(torch.equal(before[key],after[key]), (dtype,key))
            self.assertFalse(model.last_temporal_gate.requires_grad)

    def test_warmup_joint_update_and_earlier_gate_freeze(self):
        model = self.model()
        model.set_fusion_trainable(True)
        optimizer, names = make_optimizer(model, self.args)
        self.assertEqual({g['group_name'] for g in optimizer.param_groups}, {'head','features','fusion','gate'})
        self.assertEqual(next(g['weight_decay'] for g in optimizer.param_groups if g['group_name']=='gate'),0.)
        frozen = {n:p.detach().clone() for n,p in model.named_parameters() if n not in names}
        gate_before = model.last_temporal_gate.detach().clone()
        block_before = [p.detach().clone() for p in model.reference.vace_blocks[-1].parameters()]
        target = torch.randn(1,2,24,3,3)
        model.set_fusion_trainable(False).train()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            out = model(*self.inputs)
            (out['rotmat']-target).square().mean().backward()
            self.assertIsNone(model.last_temporal_gate.grad)
            optimizer.step()
        self.assertTrue(torch.equal(gate_before,model.last_temporal_gate))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(block_before,model.reference.vace_blocks[-1].parameters())))
        self.assertGreater(model.part_branch.local_projection.weight.grad.abs().sum().item(),0)
        model.set_fusion_trainable(True).train()
        optimizer.zero_grad(set_to_none=True)
        model.collect_diagnostics = True
        out = model(*self.inputs)
        (out['rotmat']-target).square().mean().backward()
        self.assertGreater(model.last_temporal_gate.grad.abs().sum().item(),0)
        probe = StepProbe(optimizer)
        optimizer.step()
        updates = probe.finish()
        self.assertGreater(updates['gate']['update_norm'],0)
        self.assertGreater(updates['fusion']['update_norm'],0)
        self.assertGreater(model.fusion_diagnostics['fusion_update_ratio'],0)
        self.assertTrue(all(torch.equal(p,frozen[n]) for n,p in model.named_parameters() if n in frozen))
        self.assertTrue(all(p.grad is None for p in model.teacher_reference.parameters()))
        model.set_fusion_trainable(False)
        gate_after = model.last_temporal_gate.detach().clone()
        optimizer.step()  # Stale gate gradients must not escape the freeze switch.
        self.assertTrue(torch.equal(gate_after,model.last_temporal_gate))

    def test_context_gradient_switch_and_identical_forward(self):
        branch = SpatialPartBranch(8,16,self.adapt)
        with torch.no_grad():
            branch.heads[0].weight.normal_(std=.03)
        detached = SpatialPartBranch(8,16,replace(self.adapt,detach_context=True))
        detached.load_state_dict(branch.state_dict())
        grid = torch.zeros(2,23,2)
        valid = torch.ones(2,23,dtype=torch.bool)
        outputs = []
        for current, enabled in ((branch,True),(detached,False)):
            context = torch.randn(1,2,16) if not outputs else saved_context.clone()
            saved_context = context.detach().clone()
            context.requires_grad_()
            result = current(self.inputs[0],context,self.inputs[2],grid,valid)
            result.square().sum().backward()
            self.assertEqual(context.grad is not None,enabled)
            if enabled:
                self.assertGreater(context.grad.abs().sum().item(),0)
            outputs.append(result.detach())
        self.assertTrue(torch.equal(*outputs))

    def test_v19_migration_and_v20_roundtrip(self):
        old = HumanV9AdaptationV19(self.cfg,V19Config(hidden_dim=16)).bind_geometry(SyntheticSMPL()).eval()
        old.reference.load_state_dict(self.payload['model'])
        old.teacher_reference.load_state_dict(self.payload['model'])
        with torch.no_grad():
            old.part_branch.heads[0].weight.normal_(std=.01)
        checkpoint = dict(format='ram-v9-joint-aligned-v19-coupled', model_config=asdict(self.cfg),
                          adaptation_config=asdict(old.adaptation_config), model=old.state_dict())
        migrated = model_from_payload(checkpoint,self.adapt).bind_geometry(SyntheticSMPL()).eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(old(*self.inputs)['rotmat'],migrated(*self.inputs)['rotmat']))
        stream = io.BytesIO()
        torch.save(dict(format=FORMAT,model_config=asdict(self.cfg),
                   adaptation_config=asdict(self.adapt),model=migrated.state_dict()),stream)
        stream.seek(0)
        restored = model_from_payload(torch.load(stream,weights_only=False)).bind_geometry(SyntheticSMPL()).eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(restored(*self.inputs)['rotmat'],migrated(*self.inputs)['rotmat']))
        self.assertFalse(any('_neutral_smpl' in key for key in migrated.state_dict()))

    def test_ablation_modes_and_bounded_masked_bias(self):
        for mode,expected in (('frozen',{'head','features'}),('block',{'head','features','fusion'})):
            model = self.model(fusion_mode=mode).set_fusion_trainable(True)
            optimizer,_ = make_optimizer(model,self.args)
            self.assertEqual({g['group_name'] for g in optimizer.param_groups},expected)
            self.assertFalse(model.last_temporal_gate.requires_grad)
        model = self.model(local_mode='bias',fusion_mode='frozen').eval()
        model.set_fusion_trainable(True)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),69)
        with torch.no_grad():
            model.part_branch.bias.fill_(1000.)
            valid = torch.tensor([[True,False]])
            out = model(*self.inputs,frame_valid=valid)
            base = model.reference(*self.inputs)
            self.assertTrue(torch.equal(out['rotmat'][~valid],base['rotmat'][~valid]))
            self.assertTrue(torch.equal(out['rotmat'][:,:,:1],base['rotmat'][:,:,:1]))
            self.assertLessEqual(float(out['part_delta_axis_angle'].norm(dim=-1).max()),torch.pi/18+1e-6)
            r = out['rotmat']
            torch.testing.assert_close(r.transpose(-1,-2)@r,torch.eye(3).expand_as(r),atol=1e-5,rtol=1e-5)
            changed = list(self.inputs)
            changed[0] = torch.randn_like(changed[0])
            self.assertTrue(torch.equal(out['part_delta_axis_angle'],model(*changed,frame_valid=valid)['part_delta_axis_angle']))

    def test_actual_epoch_function_and_diagnostic_logs(self):
        # Execute the unmodified trainer function body without importing CUDA-only
        # legacy data dependencies. Only data/SMPL losses are synthetic here.
        source = Path(__file__).with_name('train_parts_v20.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ('train_geometry_epoch','append_record')]
        model = self.model().set_fusion_trainable(True)
        optimizer,_ = make_optimizer(model,self.args)
        target = torch.randn(1,2,24,3,3)
        environment = dict(torch=torch,json=json,StepProbe=StepProbe,
            loss_gradient_contributions=loss_gradient_contributions,
            forward_batch=lambda m,b,d,a:m(*self.inputs,frame_valid=b['target_valid']),
            geometric_losses=lambda p,*args:dict(total=(p['rotmat']-target).square().mean()),
            geometry_distance=lambda p,t,*args:(p['rotmat']-t['rotmat']).square().mean())
        exec(compile(ast.Module(body=functions,type_ignores=[]),'<actual-v20-trainer>','exec'),environment)
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(**vars(self.args), fusion_warmup_epochs=1, diagnostic_every=1,
                    amp_dtype='none',teacher_weight=.05,grad_clip=1.,output_dir=Path(tmp))
            loader = [dict(target_valid=torch.ones(1,2,dtype=torch.bool))]*2
            for epoch in range(2):
                totals,pose_batches = environment['train_geometry_epoch'](
                    model,loader,None,None,optimizer,args,torch.device('cpu'),epoch,None,None)
                self.assertTrue(torch.isfinite(torch.tensor(totals['optimized_total'])))
                self.assertEqual(pose_batches,0)
            logs = [json.loads(line) for line in (Path(tmp)/'diagnostics.jsonl').read_text().splitlines()]
            self.assertEqual(len(logs),4)
            self.assertEqual(logs[0]['groups']['gate']['update_norm'],0.)
            self.assertGreater(logs[-1]['groups']['gate']['update_norm'],0.)
            self.assertIn('teacher_weighted_norm',logs[-1]['primary_3dpw_loss_gradients']['head'])


class CoverageTests(unittest.TestCase):
    def test_full_partial_missing_frames_duplicates_and_manifest(self):
        names = [f'val_{i}' for i in range(12)]
        train_names = [f'train_{i}' for i in range(24)]
        uids = [f'{name}/person_0/frame_00000' for name in names]
        train = dict(metadata=[dict(sequence=n) for n in train_names])
        val = dict(metadata=[dict(sequence=n,target_frame_uids=[u]) for n,u in zip(names,uids)],
                   frame_owner_mask=torch.ones(12,1,dtype=torch.bool),target_valid_mask=torch.ones(12,1,dtype=torch.bool))
        with self.assertRaises(ValueError):
            validate_cache_coverage(train,val)
        self.assertFalse(validate_cache_coverage(train,val,allow_partial=True)['complete'])
        manifest = dict(format='3dpw-validation-manifest-v20',source='synthetic test fixture',
                        train=train_names,validation=names,test=[f'test_{i}' for i in range(24)],validation_frame_uids=uids)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'manifest.json'
            path.write_text(json.dumps(manifest),encoding='utf-8')
            self.assertTrue(validate_cache_coverage(train,val,path)['complete'])
            partial = dict(metadata=val['metadata'][:3],frame_owner_mask=val['frame_owner_mask'][:3],
                           target_valid_mask=val['target_valid_mask'][:3])
            with self.assertRaises(ValueError):
                validate_cache_coverage(train,partial,path)
            report = validate_cache_coverage(train,partial,path,True)
            self.assertEqual(report['missing_frames'],9)
            incomplete = copy.deepcopy(val)
            incomplete['frame_owner_mask'][0,0] = False
            with self.assertRaises(ValueError):
                validate_cache_coverage(train,incomplete,path)
            duplicate = copy.deepcopy(val)
            duplicate['metadata'].append(duplicate['metadata'][0])
            duplicate['frame_owner_mask']=torch.ones(13,1,dtype=torch.bool)
            duplicate['target_valid_mask']=torch.ones(13,1,dtype=torch.bool)
            with self.assertRaises(ValueError):
                validate_cache_coverage(train,duplicate,path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
