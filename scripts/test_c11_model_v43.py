#!/usr/bin/env python3
"""CPU full-chain checks for the C11 (v43) model; random weights + a fake SMPL, no assets.

Builds the real V29 -> V30 -> A5/V34 -> v43 module chain with random weights and checks:
  1. every C11 variant reproduces the A5 parent exactly at initialisation (zero-init);
  2. a native C10 (v37) checkpoint converted by v43.model_from_payload loads strictly and gives
     the same outputs as the original v36 implementation (up to float32 ulps across instances);
  3. the external-evidence path runs, is required, and changes predictions once trained;
  4. gradients reach the router's output projection.
Implementation checks only - not performance experiments.

Run:  bash scripts/run_star_body_v43.sh scripts/test_c11_model_v43.py
"""
import sys
import types
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'video_VACE'))

from models.human_decoder_parts_v29 import HumanV9AdaptationV29, AdaptationConfig as C29, FORMAT as F29  # noqa: E402
from models.human_decoder_parts_v27 import AdaptationConfig as C27  # noqa: E402
from models.human_decoder_rgb_vace_v8_hmr2a import RGBVaceHMR2AConfig  # noqa: E402
from models.human_decoder_spatiotemporal_v30 import HumanSpatiotemporalV30, AdaptationConfig as C30, FORMAT as F30  # noqa: E402
from models.human_decoder_target60_v34 import HumanTarget60V34, AdaptationConfig as C34, FORMAT as F34  # noqa: E402
from models.human_decoder_kinematic_v36 import HumanKinematicV36, V36Config, FORMAT as F37  # noqa: E402
from models import human_decoder_kinematic_v43 as v43  # noqa: E402
from models.kinematic_ik_v36 import forward_kinematics, exp_axis  # noqa: E402


class FakeSMPL(torch.nn.Module):
    """Differentiable stand-in: 24-joint FK + 21 extra joints (>= 25 for landmark projection)."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(1)
        self.register_buffer('rest', torch.randn(24, 3, generator=g) * .15)
        self.register_buffer('extra', torch.randn(21, 3, generator=g) * .05)
        self.register_buffer('B', torch.randn(10, 24, 3, generator=g) * .01)

    def forward(self, global_orient, body_pose, betas, pose2rot=False):
        local = torch.cat((global_orient, body_pose), 1)
        rest = self.rest[None] + torch.einsum('nb,bjk->njk', betas, self.B)
        pos, _ = forward_kinematics(local, rest)
        joints = torch.cat((pos, pos[:, :21] + self.extra), 1)
        return types.SimpleNamespace(joints=joints, vertices=pos)


def build_parent_payload():
    torch.manual_seed(0)
    v29 = HumanV9AdaptationV29()
    p29 = dict(format=F29, model_config=asdict(RGBVaceHMR2AConfig()), adaptation_config=asdict(C29()),
               teacher_config=asdict(C27()), model=v29.state_dict())
    v30 = HumanSpatiotemporalV30(p29, C30())
    p30 = dict(format=F30, model_config=asdict(v30.config), adaptation_config=asdict(v30.adaptation_config),
               reference_spec=v30.reference_spec, model=v30.state_dict())
    ac = C34(use_track_shape=True, use_gru=True)
    v34 = HumanTarget60V34(p30, ac)
    with torch.no_grad():  # emulate a trained A5 parent (non-zero GRU heads and track shape)
        for head in v34.refiner.heads:
            head.weight.normal_(0, .05)
            head.bias.normal_(0, .05)
        v34.track_shape[-1].weight.normal_(0, .01)
    return dict(format=F34, adaptation_config=asdict(ac), backbone_spec=v34.backbone_spec,
                model_config=asdict(v34.config), reference_spec=v34.reference_spec, model=v34.state_dict(),
                step=1, development_only=False, initial_fallback_only=False)


def inputs(b=1, t=8, seed=3):
    g = torch.Generator().manual_seed(seed)
    values = (torch.randn(b, t, 192, 1280, generator=g), torch.randn(b, 5, 6, 6, 1536, generator=g),
              exp_axis(torch.randn(b, t, 24, 3, generator=g) * .2), torch.randn(b, t, 10, generator=g) * .5,
              torch.tensor([.9, 0., 0.]).expand(b, t, 3).clone())
    kw = dict(frame_valid=torch.ones(b, t, dtype=torch.bool), landmark_schema='posetrack',
              source_frame_ids=torch.tensor([[10, 12, 15, 17, 20, 22, 25, 27]]).expand(b, t).clone(),
              track_shape_base=torch.randn(b, 10, generator=g) * .3, track_shape_features=torch.randn(b, 1280, generator=g))
    keypoints = torch.cat((torch.rand(b, t, 17, 2, generator=g) - .5, torch.rand(b, t, 17, 1, generator=g)), -1)
    return values, kw, keypoints


VARIANTS = {
    'none': dict(feedback_mode='none'),
    'mean': dict(feedback_mode='mean'),
    'c10-head-joint': dict(feedback_mode='router'),
    'head-noid': dict(feedback_mode='router', router_identity=False),
    'head-nobias': dict(feedback_mode='router', router_routing='none'),
    'ext-pooled': dict(feedback_mode='router', router_sources='external', router_pooling='mean',
                       router_identity=False, router_routing='none'),
    'ext-concat': dict(feedback_mode='router', router_sources='external', router_pooling='concat',
                       router_identity=False, router_routing='none'),
    'ext-set': dict(feedback_mode='router', router_sources='external', router_identity=False, router_routing='none'),
    'ext-joint': dict(feedback_mode='router', router_sources='external'),
    'ext-shuffled': dict(feedback_mode='router', router_sources='external', router_routing='shuffled'),
    'both-joint': dict(feedback_mode='router', router_sources='head+external', router_init_seed=2027),
}


class C11ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        cls.parent = build_parent_payload()
        cls.smpl = FakeSMPL()
        cls.values, cls.kw, cls.keypoints = inputs()

    def make(self, **cfg):
        model = v43.model_from_payload(self.parent, v43.V43Config(arm='R2', temporal=True, **cfg))
        return model.bind_geometry(self.smpl).bind_landmarks(self.smpl).eval()

    def run_model(self, model, external=True, **extra):
        kw = dict(self.kw, **extra)
        if external:
            kw['external_keypoints'] = self.keypoints
        return model(*self.values, **kw)

    def test_all_variants_start_at_parent(self):
        for name, cfg in VARIANTS.items():
            with self.subTest(variant=name):
                model = self.make(**cfg)
                with torch.no_grad():
                    full = self.run_model(model)
                    parent = self.run_model(model, force_anchor=True)
                for key in ('rotmat', 'betas', 'camera'):
                    torch.testing.assert_close(full[key], parent[key], rtol=0, atol=0)

    def test_external_required_and_effective(self):
        model = self.make(**VARIANTS['ext-joint'])
        with self.assertRaises(ValueError):
            with torch.no_grad():
                self.run_model(model, external=False)
        with torch.no_grad():
            before = self.run_model(model)['rotmat']
            model.refiner.joint_feedback.output.weight.normal_(0, .2)
            after = self.run_model(model)['rotmat']
            changed = self.keypoints.clone()
            changed[..., 9, :2] += .1  # move the left-wrist detection only
            moved = model(*self.values, **dict(self.kw, external_keypoints=changed))['rotmat']
        self.assertGreater(float((after - before).abs().max()), 1e-6)
        self.assertGreater(float((moved - after).abs().max()), 1e-7)

    def test_c10_checkpoint_conversion_matches_v36(self):
        torch.manual_seed(11)
        old = HumanKinematicV36(self.parent, V36Config(arm='R2', temporal=True, feedback_mode='joint'))
        with torch.no_grad():
            old.refiner.joint_feedback.output.weight.normal_(0, .1)
            for block in old.refiner.temporal:
                block.project.weight.normal_(0, .02)
        payload = dict(format=F37, v36_config=asdict(old.v36_config), parent_spec=old.parent_spec,
                       model=old.state_dict(), step=1000)
        new = v43.model_from_payload(payload)
        self.assertEqual(new.v36_config.feedback_mode, 'router')
        self.assertTrue(new.v36_config.router_config().is_c10_default)
        old = old.bind_geometry(self.smpl).bind_landmarks(self.smpl).eval()
        new = new.bind_geometry(self.smpl).bind_landmarks(self.smpl).eval()
        with torch.no_grad():
            a = old(*self.values, **self.kw)
            b = new(*self.values, **self.kw)
        # Different module instances with identical tensors can differ by float32 ulps because
        # CPU BLAS reductions depend on memory alignment; within one instance outputs are exact.
        for key in ('rotmat', 'betas', 'camera'):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=1e-5)

    def test_router_output_receives_gradient(self):
        for name in ('c10-head-joint', 'ext-joint', 'ext-pooled', 'ext-concat', 'both-joint'):
            with self.subTest(variant=name):
                model = self.make(**VARIANTS[name])
                model.configure_phase(0)
                model.train()
                out = self.run_model(model)
                out['rotmat'].square().sum().backward()
                grad = model.refiner.joint_feedback.output.weight.grad
                self.assertIsNotNone(grad)
                self.assertGreater(float(grad.abs().max()), 0.)

    def test_v43_native_roundtrip(self):
        model = self.make(**VARIANTS['ext-shuffled'])
        payload = dict(format=v43.FORMAT, v36_config=asdict(model.v36_config), parent_spec=model.parent_spec,
                       model=model.state_dict(), step=1000)
        again = v43.model_from_payload(payload).bind_geometry(self.smpl).bind_landmarks(self.smpl).eval()
        with torch.no_grad():
            torch.testing.assert_close(self.run_model(model)['rotmat'], self.run_model(again)['rotmat'], rtol=0, atol=1e-5)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, again.state_dict()[key]), key)


if __name__ == '__main__':
    unittest.main(verbosity=2)
