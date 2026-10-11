"""CPU smoke checks for v41 evidence; no datasets, checkpoints, or SMPL needed.

Run from repository root: python -m divibody_training.test_joint_evidence_v41
"""
import json
import unittest

import torch

from models.joint_evidence_v41 import JointEvidenceConfig, JointEvidenceRefiner


class JointEvidenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.model = JointEvidenceRefiner(24, JointEvidenceConfig(hidden_dim=32)).eval()

    def sample(self, b=2, t=4):
        tokens = torch.randn(b, t, 192, 24)
        projected = torch.rand(b, t, 17, 2) * .6 - .3
        valid = torch.ones(b, t, dtype=torch.bool)
        return tokens, projected, valid, valid.clone()

    def test_shapes_coordinates_and_finite_gradients(self):
        x, p, valid, independent = self.sample()
        x.requires_grad_()
        p.requires_grad_()
        out = self.model(x, p, valid, independent, torch.arange(4).expand(2, -1))
        expected = dict(observed_uv=(2, 4, 17, 2), raw_observed_uv=(2, 4, 17, 2),
                        confidence=(2, 4, 17, 1), features=(2, 4, 17, 32),
                        spatial_logits=(2, 4, 17, 192), temporal_support=(2, 4, 17, 1))
        for name, shape in expected.items():
            self.assertEqual(tuple(out[name].shape), shape)
            self.assertTrue(torch.isfinite(out[name]).all(), name)
        centers = self.model.square_coordinates
        self.assertTrue(torch.allclose(centers[0], torch.tensor([-.34375, -.46875])))
        self.assertTrue(torch.allclose(centers[-1], torch.tensor([.34375, .46875])))
        self.assertTrue(((out['confidence'] >= 0) & (out['confidence'] <= 1)).all())
        loss = out['observed_uv'].square().mean() + out['features'].square().mean()
        loss = loss + out['spatial_logits'].square().mean() + out['confidence'].mean()
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(p.grad).all())
        grads = [v.grad for v in self.model.parameters() if v.grad is not None]
        self.assertGreater(len(grads), 20)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(float(x.grad.abs().sum()), 0)

    def test_invalid_padding_cannot_pollute_valid_frames(self):
        x, p, valid, independent = self.sample(b=1)
        valid[:, 2:] = False
        independent[:, 2:] = False
        a = self.model(x, p, valid, independent)
        corrupt_x, corrupt_p = x.clone(), p.clone()
        corrupt_x[:, 2:] = float('nan')
        corrupt_p[:, 2:] = float('nan')
        b = self.model(corrupt_x, corrupt_p, valid, independent)
        for name in a:
            self.assertTrue(torch.equal(a[name][:, :2], b[name][:, :2]), name)
            self.assertTrue(torch.isfinite(b[name]).all(), name)
            self.assertEqual(int(torch.count_nonzero(b[name][:, 2:])), 0, name)

    def test_duplicate_sources_are_not_temporal_evidence(self):
        x, p, valid, independent = self.sample(b=1, t=3)
        source = torch.tensor([[100, 102, 102]])
        a = self.model(x, p, valid, independent, source)
        changed_x, changed_p = x.clone(), p.clone()
        changed_x[:, 2] = torch.randn_like(changed_x[:, 2]) * 100
        changed_p[:, 2] = .49
        b = self.model(changed_x, changed_p, valid, independent, source)
        self.assertTrue(torch.equal(a['observed_uv'][:, :2], b['observed_uv'][:, :2]))
        self.assertTrue(torch.equal(a['features'][:, :2], b['features'][:, :2]))
        self.assertEqual(int(torch.count_nonzero(a['confidence'][:, 2])), 0)
        # Explicit independent=False has the same protection without source metadata.
        independent[:, 2] = False
        c = self.model(x, p, valid, independent)
        d = self.model(changed_x, changed_p, valid, independent)
        self.assertTrue(torch.equal(c['observed_uv'][:, :2], d['observed_uv'][:, :2]))

    def test_single_unique_frame_exact_fallback_and_empty_masks(self):
        x, p, valid, independent = self.sample(b=1, t=3)
        source = torch.tensor([[5, 5, 5]])
        out = self.model(x, p, valid, independent, source)
        self.assertTrue(torch.equal(out['observed_uv'], out['raw_observed_uv']))
        self.assertFalse(out['temporal_support'].any())
        self.assertEqual(int(torch.count_nonzero(out['confidence'][:, 1:])), 0)
        independent[:] = False
        empty_keys = self.model(x, p, valid, independent)
        self.assertTrue(torch.equal(empty_keys['observed_uv'], empty_keys['raw_observed_uv']))
        self.assertFalse(empty_keys['confidence'].any())
        valid[:] = False
        empty = self.model(x, p, valid, independent)
        for name, value in empty.items():
            self.assertTrue(torch.isfinite(value).all(), name)
            self.assertEqual(int(torch.count_nonzero(value)), 0, name)
        x, p, valid, independent = self.sample(b=1, t=1)
        single = self.model(x, p, valid, independent, torch.tensor([[9]]))
        self.assertTrue(torch.equal(single['observed_uv'], single['raw_observed_uv']))

    def test_relative_source_time_bias_is_used(self):
        x, p, valid, independent = self.sample(b=1, t=3)
        a = self.model(x, p, valid, independent, torch.tensor([[10, 11, 12]]))
        shifted = self.model(x, p, valid, independent, torch.tensor([[10010, 10011, 10012]]))
        wider = self.model(x, p, valid, independent, torch.tensor([[10, 11, 112]]))
        self.assertTrue(torch.equal(a['observed_uv'], shifted['observed_uv']))
        self.assertFalse(torch.allclose(a['features'], wider['features']))

    def test_input_validation(self):
        x, p, valid, independent = self.sample()
        with self.assertRaises(ValueError):
            self.model(x[:, :, :-1], p, valid, independent)
        with self.assertRaises(ValueError):
            self.model(x, p[:, :, :-1], valid, independent)
        with self.assertRaises(TypeError):
            self.model(x, p, valid.float(), independent)
        with self.assertRaises(ValueError):
            self.model(x, p, valid, independent, torch.zeros(2, 4))
        p[0, 0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            self.model(x, p, valid, independent)


if __name__ == '__main__':
    torch.set_num_threads(2)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(JointEvidenceTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps(dict(version='v41_joint_evidence_cpu_smoke', tests=result.testsRun,
                          failures=len(result.failures), errors=len(result.errors),
                          success=result.wasSuccessful())))
    raise SystemExit(0 if result.wasSuccessful() else 1)
