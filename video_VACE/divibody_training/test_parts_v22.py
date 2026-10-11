"""v22 CPU regression with real decoders and synthetic SMPL geometry.

Copied from v20 and adapted without modifying any previous experiment file.
Local: python test_parts_v22.py --local
Server: python -m divibody_training.test_parts_v22
Passing these checks is not a real-data or licensed-SMPL performance result.
"""
import copy
from dataclasses import asdict, replace
import io
from pathlib import Path
import sys
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
from models.human_decoder_parts_v22 import (
    AdaptationConfig, FORMAT, HumanV9AdaptationV22, RGBVaceHMR2AConfig,
    SpatialPartBranch, model_from_payload,
)
from models.human_decoder_parts_v19 import HumanV9AdaptationV19, AdaptationConfig as V19Config
from models.human_decoder_rgb_vace_v9_isolated import HumanRGBVaceGeometryTransformerV9Isolated
from divibody_training.parts_training_utils_v22 import make_optimizer,step_lr,ModelEMA


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
            num_heads=4,rgb_layers=2,vace_layers=2,output_frames=2,dropout=0.)
        self.adapt = AdaptationConfig(hidden_dim=16)
        self.base = HumanRGBVaceGeometryTransformerV9Isolated(self.cfg).eval()
        with torch.no_grad():
            self.base.temporal_gates.copy_(torch.tensor([.00035,.0008]))
            # Trained heads and gates are nonzero, enabling upstream gradients.
            for name, parameter in self.base.named_parameters():
                if 'residual_head' in name and name.endswith('weight'):
                    parameter.add_(torch.randn_like(parameter)*.003)
        self.payload = dict(format='ram-geometry-priority-isolated-v9',
                            model_config=asdict(self.cfg), model=copy.deepcopy(self.base.state_dict()))
        camera = torch.zeros(1,2,3)
        camera[...,0] = 1
        self.inputs = (torch.randn(1,2,192,8),torch.randn(1,2,2,2,8),
                       torch.eye(3).expand(1,2,24,3,3).clone(), torch.zeros(1,2,10), camera)

    def model(self):
        return model_from_payload(self.payload,self.adapt).bind_geometry(SyntheticSMPL())

    def assert_grad(self, parameter):
        self.assertIsNotNone(parameter.grad)
        self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(parameter.grad.abs().sum().item(),0.)

    def test_initial_v9_eval_exact_fp32_and_bf16(self):
        for dtype in (None, torch.bfloat16):
            model = self.model().eval()
            with torch.no_grad(), torch.autocast('cpu',dtype=dtype or torch.bfloat16,enabled=dtype is not None):
                expected = self.base(*self.inputs)
                actual = model(*self.inputs)
            for key in ('rotmat','betas','camera','fused_queries','temporal_gate'):
                self.assertTrue(torch.equal(expected[key],actual[key]), (dtype,key))

    def test_all_learned_parameters_trainable_and_no_teacher(self):
        model = self.model().train()
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        self.assertFalse(hasattr(model,'teacher_reference'))
        self.assertFalse(any('teacher' in name or 'parametrizations' in name for name,_ in model.named_parameters()))
        self.assertTrue(all(module.training for module in model.reference.modules()))
        self.assertFalse(model._neutral_smpl.training)
        self.assertFalse(model._neutral_smpl.scale.requires_grad)
        self.assertFalse(any('_neutral_smpl' in key for key in model.state_dict()))
        model.eval()
        self.assertTrue(all(not module.training for module in model.reference.modules()))

    def test_supervised_gradients_and_two_optimizer_steps(self):
        model = self.model().train()
        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0.)
        before = {name: parameter.detach().clone() for name,parameter in model.named_parameters()}
        target_rotation = torch.randn(1,2,24,3,3)
        target_shape = torch.randn(1,2,10)
        target_camera = torch.randn(1,2,3)
        model.collect_diagnostics = True
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = model(*self.inputs)
            loss = ((output['rotmat']-target_rotation).square().mean()
                    + (output['betas']-target_shape).square().mean()
                    + (output['camera']-target_camera).square().mean())
            loss.backward()
            for blocks in (model.reference.rgb_blocks, model.reference.vace_blocks):
                for block in blocks:
                    self.assert_grad(block.cross_attention.in_proj_weight)
            self.assert_grad(model.reference.rgb_projection[1].weight)
            self.assert_grad(model.reference.vace_projection[1].weight)
            self.assert_grad(model.reference.temporal_gates)
            for head in ('rotation_residual_head','shape_residual_head','camera_residual_head'):
                self.assert_grad(getattr(model.reference,head).weight)
            self.assert_grad(model.part_branch.heads[0].weight)
            if step:
                self.assert_grad(model.part_branch.local_projection.weight)
                self.assert_grad(model.part_branch.context_projection.weight)
                self.assert_grad(model.part_branch.rgb_projection[1].weight)
                self.assert_grad(model.part_branch.pose_projection.weight)
            optimizer.step()
        for name in ('reference.rgb_blocks.0.cross_attention.in_proj_weight',
                     'reference.rgb_blocks.1.cross_attention.in_proj_weight',
                     'reference.vace_blocks.0.cross_attention.in_proj_weight',
                     'reference.vace_blocks.1.cross_attention.in_proj_weight',
                     'reference.temporal_gates','reference.camera_residual_head.weight',
                     'part_branch.local_projection.weight'):
            self.assertFalse(torch.equal(before[name],dict(model.named_parameters())[name]),name)
        self.assertEqual(set(model.fusion_diagnostics),{'block_0','block_1'})
        self.assertTrue(all(d['fusion_update_ratio']>0 for d in model.fusion_diagnostics.values()))

    def test_local_branch_context_and_pose_keep_gradients(self):
        branch = SpatialPartBranch(8,16,self.adapt)
        with torch.no_grad():
            branch.heads[0].weight.normal_(std=.03)
        tokens = self.inputs[0].clone().requires_grad_()
        context = torch.randn(1,2,16,requires_grad=True)
        rotations = self.inputs[2].clone().requires_grad_()
        output = branch(tokens,context,rotations,torch.zeros(2,23,2),torch.ones(2,23,dtype=torch.bool))
        output.square().sum().backward()
        for tensor in (tokens,context,rotations):
            self.assert_grad(tensor)

    def test_mask_angle_bound_and_rotation_composition(self):
        model = self.model().eval()
        with torch.no_grad():
            model.part_branch.heads[0].bias.fill_(1000.)
            valid = torch.tensor([[True,False]])
            output = model(*self.inputs,frame_valid=valid)
            baseline = model.reference(*self.inputs)
            self.assertTrue(torch.equal(output['rotmat'][~valid],baseline['rotmat'][~valid]))
            self.assertTrue(torch.equal(output['rotmat'][:,:,:1],baseline['rotmat'][:,:,:1]))
            self.assertEqual(output['part_delta_axis_angle'][~valid].abs().sum().item(),0.)
            self.assertLessEqual(output['part_delta_axis_angle'].norm(dim=-1).max().item(),torch.pi/18+1e-6)
            rotation = output['rotmat']
            torch.testing.assert_close(rotation.transpose(-1,-2)@rotation,
                                       torch.eye(3).expand_as(rotation),atol=1e-5,rtol=1e-5)
        for invalid in (torch.tensor([[False,False]]),torch.ones(1,2),torch.ones(2,dtype=torch.bool)):
            with self.assertRaises(ValueError):
                model(*self.inputs,frame_valid=invalid)

    def test_v19_migration_preserves_prediction_and_drops_teacher(self):
        old = HumanV9AdaptationV19(self.cfg,V19Config(hidden_dim=16)).bind_geometry(SyntheticSMPL()).eval()
        old.reference.load_state_dict(self.payload['model'])
        old.teacher_reference.load_state_dict(self.payload['model'])
        with torch.no_grad():
            old.part_branch.heads[0].weight.normal_(std=.01)
        checkpoint = dict(format='ram-v9-joint-aligned-v19-coupled',model_config=asdict(self.cfg),
                          adaptation_config=asdict(old.adaptation_config),model=old.state_dict())
        migrated = model_from_payload(checkpoint).bind_geometry(SyntheticSMPL()).eval()
        with torch.no_grad():
            expected = old(*self.inputs)
            actual = migrated(*self.inputs)
        for key in ('rotmat','betas','camera','part_delta_axis_angle'):
            torch.testing.assert_close(expected[key],actual[key],atol=2e-6,rtol=2e-6,msg=key)
        self.assertFalse(any('teacher' in name for name in migrated.state_dict()))
        self.assertTrue(all(parameter.requires_grad for parameter in migrated.parameters()))
        malformed = copy.deepcopy(checkpoint)
        malformed['model']['unexpected.weight'] = torch.zeros(1)
        with self.assertRaises(ValueError):
            model_from_payload(malformed)
        with self.assertRaises(ValueError):
            model_from_payload(checkpoint,replace(self.adapt,max_angle_degrees=5.))

    def test_v22_roundtrip_and_strict_formats(self):
        model = self.model().eval()
        with torch.no_grad():
            model.part_branch.heads[0].weight.normal_(std=.01)
        stream = io.BytesIO()
        torch.save(dict(format=FORMAT,model_config=asdict(self.cfg),
                        adaptation_config=asdict(self.adapt),model=model.state_dict()),stream)
        stream.seek(0)
        checkpoint = torch.load(stream,weights_only=False)
        restored = model_from_payload(checkpoint).bind_geometry(SyntheticSMPL()).eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(model(*self.inputs)['rotmat'],restored(*self.inputs)['rotmat']))
        with self.assertRaises(ValueError):
            model_from_payload(dict(checkpoint,format='ram-v9-joint-aligned-v20-gated'))
        missing = copy.deepcopy(checkpoint)
        del missing['model']['reference.temporal_gates']
        with self.assertRaises(RuntimeError):
            model_from_payload(missing)

    def test_optimizer_schedule_and_ema_cover_full_model(self):
        model=self.model()
        args=SimpleNamespace(rgb_lr=1e-6,vace_lr=2e-6,heads_lr=3e-6,camera_lr=4e-6,
            gate_lr=5e-6,local_lr=6e-6,weight_decay=1e-4)
        optimizer,names=make_optimizer(model,args)
        self.assertEqual(set(names),{name for name,_ in model.named_parameters()})
        self.assertEqual({g['category'] for g in optimizer.param_groups},
                         {'rgb','vace','heads','camera','gate','local'})
        first=step_lr(optimizer,0,20,4)
        peak=step_lr(optimizer,3,20,4)
        final=step_lr(optimizer,19,20,4)
        self.assertTrue(all(first[k]<peak[k] for k in peak))
        for key in peak:
            self.assertAlmostEqual(final[key],peak[key]*.1,places=12)
        ema=ModelEMA(model,.9)
        initial={k:v.clone() for k,v in ema.model.state_dict().items()}
        with torch.no_grad():
            next(model.parameters()).add_(1.)
        decay=ema.update(model)
        self.assertGreater(decay,0.)
        self.assertTrue(all(not p.requires_grad for p in ema.model.parameters()))
        self.assertTrue(any(not torch.equal(v,ema.model.state_dict()[k]) for k,v in initial.items()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
