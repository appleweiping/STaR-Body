"""Focused CPU checks for sample replay, task gradients and selection safety."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from divibody_training.v28d_experiment_utils import MapStream, gradients, subtract, gradient_report, selection_report, validate_coverage, sha256, official_annotation_members


class TinyDataset:
    files = [Path(f'{i}_clean.pt' if i%2==0 else f'{i}_real_occlusion.pt') for i in range(7)]
    def __len__(self): return len(self.files)
    def __getitem__(self, i): return {'id': torch.tensor(i)}


class ExperimentsTest(unittest.TestCase):
    def test_archive_resource_forks_are_not_annotations(self):
        names=['sequenceFiles/train/real.pkl','__MACOSX/sequenceFiles/train/._real.pkl',
               'sequenceFiles/train/._real.pkl','sequenceFiles/validation/other.pkl']
        self.assertEqual(official_annotation_members(names,'train'),[names[0]])

    def test_sample_replay_across_cycle_boundary(self):
        uninterrupted=MapStream(TinyDataset(),3,2026)
        observed=[]
        for _ in range(2):
            _, ids=uninterrupted.next_batch(); uninterrupted.record(ids); observed.extend(ids)
        saved=uninterrupted.state_dict()
        resumed=MapStream(TinyDataset(),3,2026); resumed.load_state_dict(saved)
        for _ in range(6):
            a, ai=uninterrupted.next_batch(); b, bi=resumed.next_batch()
            self.assertEqual(ai,bi); self.assertTrue(torch.equal(a['id'],b['id']))
            uninterrupted.record(ai); resumed.record(bi)
        self.assertEqual(uninterrupted.state_dict(),resumed.state_dict())
        self.assertEqual(resumed.coverage()['unique_optimized'],7)
        self.assertEqual(sum(resumed.coverage()['type_draws'].values()),resumed.draws)

    def test_disabled_branch_does_not_count_optimized_coverage(self):
        stream=MapStream(TinyDataset(),3,2026)
        stream.next_batch()
        self.assertEqual(stream.coverage()['sampled_draws'],3)
        self.assertEqual(stream.coverage()['optimized_draws'],0)
        self.assertEqual(stream.coverage()['fraction'],0.)

    def test_gradient_decomposition_detects_conflict(self):
        p=torch.nn.Parameter(torch.ones(2))
        optimizer=torch.optim.SGD([dict(params=[p],group_name='shared')],lr=.1)
        p.sum().backward(); geometry=gradients(optimizer)
        (-2*p[0]).backward(); cumulative=gradients(optimizer)
        pose=subtract(cumulative,geometry)
        self.assertTrue(torch.equal(pose['shared'],torch.tensor([-2.,0.])))
        result=gradient_report({'geometry':geometry,'pose':pose})['shared']
        self.assertAlmostEqual(result['cosines']['geometry:pose'],-2**-.5,places=6)

    def test_improved_projection_cannot_bypass_pve_gate(self):
        args=SimpleNamespace(pve_tolerance_mm=.5,pose_pck_tolerance=.005)
        baseline={'pve_mm':64.28}; pose={'projected_PCKn@0.05':.66,'projected_PCKn@0.10':.87,'projected_mean_normalized_joint_error':.056}
        candidate=pose|{'projected_PCKn@0.05':.68,'projected_PCKn@0.10':.89}
        result=selection_report({'pve_mm':65.},candidate,baseline,pose,args,1.)
        self.assertFalse(result['accepted']); self.assertIn('pve_above_tolerance',result['reasons'])

    def test_invalid_frame_schema_cannot_pass_validation(self):
        train={'metadata':[{'sequence':'train'}]}
        val={'metadata':[{'sequence':'val','target_frame_uids':['val/person_0/frame_00000']}],
             'frame_owner_mask':torch.tensor([[True,True]]),'target_valid_mask':torch.tensor([[True,True]])}
        with self.assertRaisesRegex(ValueError,'mask width'):
            validate_coverage(train,val,allow_partial=True)

    def test_partial_train_reference_stays_developmental(self):
        import json
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); archive=root/'archive.zip'; archive.write_bytes(b'unit-fixture')
            uid='val/person_0/frame_00000'
            reference=root/'reference.json'
            reference.write_text(json.dumps(dict(format='v28d-validation-reference-v1',
                train=['train','missing_train'],validation=['val'],validation_frame_uids=[uid],
                independent_archive_verified=True,archive_path=str(archive),archive_sha256=sha256(archive),source='unit-fixture')))
            train={'metadata':[{'sequence':'train'}]}
            val={'metadata':[{'sequence':'val','target_frame_uids':[uid]}],
                 'frame_owner_mask':torch.tensor([[True]]),'target_valid_mask':torch.tensor([[True]])}
            result=validate_coverage(train,val,reference,allow_partial=True)
            self.assertFalse(result['complete'])
            self.assertEqual(result['missing_train_sequences'],['missing_train'])
            with self.assertRaisesRegex(ValueError,'Train cache sequences differ'):
                validate_coverage(train,val,reference,allow_partial=False)


if __name__=='__main__': unittest.main()
