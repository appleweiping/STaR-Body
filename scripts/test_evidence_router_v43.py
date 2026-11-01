#!/usr/bin/env python3
"""CPU unit checks for the C11 evidence router (no datasets, SMPL files or GPU needed).

Run:  bash scripts/run_star_body.sh scripts/test_evidence_router_v43.py
These are implementation checks, not performance experiments.
"""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'video_VACE'))

from models.joint_feedback_v37 import make_joint_feedback  # noqa: E402
from models.evidence_router_v43 import (  # noqa: E402
    RouterConfig, SHUFFLE, make_evidence_router, external_evidence_features)


def head_inputs(b=2, t=3, seed=0, invalid_frame=True):
    g = torch.Generator().manual_seed(seed)
    residual = torch.randn(b, t, 17, 2, generator=g) * .1
    projected = torch.randn(b, t, 17, 2, generator=g) * .3
    logvar = torch.randn(b, t, 17, 1, generator=g)
    visibility = torch.rand(b, t, 17, 1, generator=g)
    frame_valid = torch.ones(b, t, dtype=torch.bool)
    if invalid_frame:
        frame_valid[0, 1] = False
    residual[1, 2, 4] = float('nan')  # one non-finite slot
    return residual, projected, logvar, visibility, frame_valid


def head_source(residual, projected, logvar, visibility, frame_valid, schema='posetrack'):
    return dict(name='head', schema=schema, residual=residual, projected=projected, logvar=logvar,
                visibility=visibility, frame_valid=frame_valid)


def randomize_output(module, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        module.output.weight.copy_(torch.randn(module.output.weight.shape, generator=g) * .05)


class RouterTests(unittest.TestCase):
    def test_default_equals_v37_parameters_and_function(self):
        old = make_joint_feedback(128)
        new = make_evidence_router(128, RouterConfig())
        self.assertTrue(RouterConfig().is_c10_default)
        old_state, new_state = old.state_dict(), new.state_dict()
        self.assertEqual(list(old_state), list(new_state))
        for key in old_state:
            torch.testing.assert_close(old_state[key], new_state[key], rtol=0, atol=0)
        randomize_output(old)
        new.load_state_dict(old.state_dict(), strict=True)
        for schema in ('posetrack', 'coco'):
            args = head_inputs(seed=3)
            expected = old(*args, schema=schema)
            actual = new([head_source(*args, schema=schema)], batch_shape=args[0].shape[:2])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_zero_init_and_invalid_frames(self):
        for cfg in (RouterConfig(), RouterConfig(routing='none'), RouterConfig(identity=False),
                    RouterConfig(routing='shuffled'), RouterConfig(pooling='mean', identity=False, routing='none'),
                    RouterConfig(pooling='concat', identity=False, routing='none', sources=('head', 'external')),
                    RouterConfig(sources=('head', 'external'))):
            module = make_evidence_router(64, cfg)
            args = head_inputs(seed=5)
            sources = [head_source(*args)]
            if 'external' in cfg.sources:
                kp = torch.rand(2, 3, 17, 3)
                sources.append(external_evidence_features(kp, args[1], args[4]))
            out = module(sources, batch_shape=(2, 3))
            self.assertEqual(tuple(out.shape), (2, 3, 24, 64))
            self.assertEqual(float(out.abs().max()), 0.)  # zero-initialised output projection
            randomize_output(module)
            out = module(sources, batch_shape=(2, 3))
            self.assertTrue(torch.isfinite(out).all())
            self.assertEqual(float(out[0, 1].abs().max()), 0.)  # invalid frame -> exactly zero
            self.assertGreater(float(out[1, 0].abs().max()), 0.)

    def test_no_identity_is_permutation_invariant(self):
        module = make_evidence_router(64, RouterConfig(identity=False, routing='none'))
        randomize_output(module)
        residual, projected, logvar, visibility, valid = head_inputs(seed=7, invalid_frame=False)
        residual = torch.nan_to_num(residual)
        perm = torch.randperm(17, generator=torch.Generator().manual_seed(0))
        a = module([head_source(residual, projected, logvar, visibility, valid)], batch_shape=(2, 3))
        b = module([head_source(residual[:, :, perm], projected[:, :, perm], logvar[:, :, perm],
                                visibility[:, :, perm], valid)], batch_shape=(2, 3))
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        # With identity the same permutation must change the output.
        ident = make_evidence_router(64, RouterConfig(routing='none'))
        randomize_output(ident)
        a = ident([head_source(residual, projected, logvar, visibility, valid)], batch_shape=(2, 3))
        b = ident([head_source(residual[:, :, perm], projected[:, :, perm], logvar[:, :, perm],
                               visibility[:, :, perm], valid)], batch_shape=(2, 3))
        self.assertGreater(float((a - b).abs().max()), 1e-4)

    def test_mean_pooling_broadcasts_and_matches_capacity(self):
        mean = make_evidence_router(128, RouterConfig(pooling='mean', identity=False, routing='none'))
        joint = make_evidence_router(128, RouterConfig())
        randomize_output(mean)
        out = mean([head_source(*head_inputs(seed=9))], batch_shape=(2, 3))
        torch.testing.assert_close(out[:, :, :1].expand_as(out), out)  # identical vector for all joints
        used = lambda m, skip: sum(p.numel() for n, p in m.named_parameters() if not n.startswith(skip))
        joint_params = used(joint, ('pooled',))
        mean_params = used(mean, ('attention', 'landmark', 'pose_joint', 'schema'))
        self.assertLess(abs(mean_params - joint_params) / joint_params, .1)

    def test_concat_control(self):
        concat = make_evidence_router(64, RouterConfig(pooling='concat', identity=False, routing='none'))
        randomize_output(concat)
        residual, projected, logvar, visibility, valid = head_inputs(seed=4, invalid_frame=False)
        residual = torch.nan_to_num(residual)
        a = concat([head_source(residual, projected, logvar, visibility, valid)], batch_shape=(2, 3))
        torch.testing.assert_close(a[:, :, :1].expand_as(a), a)  # global: identical for all joints
        perm = torch.randperm(17, generator=torch.Generator().manual_seed(1))
        b = concat([head_source(residual[:, :, perm], projected[:, :, perm], logvar[:, :, perm],
                                visibility[:, :, perm], valid)], batch_shape=(2, 3))
        self.assertGreater(float((a - b).abs().max()), 1e-5)  # order-sensitive: identity by position
        with self.assertRaises(ValueError):
            RouterConfig(pooling='concat')  # must not combine with identity embeddings

    def test_shuffled_routing_is_derangement(self):
        self.assertEqual(sorted(SHUFFLE), list(range(17)))
        self.assertTrue(all(i != j for i, j in enumerate(SHUFFLE)))
        module = make_evidence_router(32, RouterConfig(routing='shuffled'))
        torch.testing.assert_close(module.route_posetrack, module.bias_posetrack[:, list(SHUFFLE)])

    def test_external_features(self):
        kp = torch.tensor([[[[.1, -.2, 1.], [0., 0., 0.]] + [[.0, .0, .5]] * 15]])
        projected = torch.zeros(1, 1, 17, 2)
        src = external_evidence_features(kp, projected, torch.ones(1, 1, dtype=torch.bool), .0)
        self.assertTrue(bool(src['slot_valid'][0, 0, 0]) and not bool(src['slot_valid'][0, 0, 1]))
        self.assertAlmostEqual(float(src['logvar'][0, 0, 0]), 0., places=6)
        self.assertAlmostEqual(float(src['logvar'][0, 0, 2]), -2 * torch.log(torch.tensor(.5)).item(), places=5)
        torch.testing.assert_close(src['residual'][0, 0, 0], torch.tensor([.1, -.2]))

    def test_init_seed_changes_router_only(self):
        a = make_evidence_router(32, RouterConfig(init_seed=1))
        b = make_evidence_router(32, RouterConfig(init_seed=2))
        self.assertGreater(float((a.token[0].weight - b.token[0].weight).abs().max()), 0.)
        state = torch.random.get_rng_state()
        make_evidence_router(32, RouterConfig(init_seed=3))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))  # global RNG untouched


if __name__ == '__main__':
    unittest.main(verbosity=2)
