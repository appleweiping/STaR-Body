"""Evaluate a frozen validation-selected V29 once; never selects on test."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile
import torch
import numpy as np
from torch.utils.data import DataLoader


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--vace-cache',type=Path,required=True);p.add_argument('--hmr-cache',type=Path,required=True)
    p.add_argument('--annotation-archive',type=Path,required=True);p.add_argument('--image-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    from divibody_training.evaluate_standard_3dpw import Standard3DPWDataset,audit_split,create_smpl_models
    from divibody_training.train_divibody_rgb_vace_hmr2a_v8 import load_pair
    from divibody_training.v29_experiment_utils import official_annotation_members
    from divibody_training.ram_parts_runtime_v29 import V10Dataset,validate
    from models.human_decoder_parts_v29 import model_from_payload
    from hmr2.models.smpl_wrapper import SMPL
    c=json.loads(a.config.read_text());device=torch.device(a.device);torch.set_num_threads(4)
    # Independently enumerate official test annotations and RGB availability.
    if digest(a.annotation_archive)!='889c5faee2c3e3404b5f8e1dbc53c3af857234b6e30e6413938a1acaa7f2be1c':
        raise ValueError('Official archive fingerprint differs from recovered verified archive')
    reference=a.output/'reference';(reference/'sequenceFiles/test').mkdir(parents=True)
    (reference/'imageFiles').mkdir()
    with ZipFile(a.annotation_archive) as archive:
        members=official_annotation_members(archive.namelist(),'test')
        if len(members)!=24:raise ValueError('Expected 24 official test annotations')
        for name in members:
            seq=Path(name).stem
            (reference/'sequenceFiles/test'/Path(name).name).write_bytes(archive.read(name))
            (reference/'imageFiles'/seq).symlink_to((a.image_root/seq).resolve(),target_is_directory=True)
    audit=audit_split(reference,'test')
    data=Standard3DPWDataset(root=reference,split='test',image_size=96,frame_stride=1,max_samples=None,crop_to_person=True)
    expected={f'{s.sequence}/person_{s.person_index}/frame_{s.frame_index:05d}' for s in data.samples}
    if len(expected)!=35515 or audit['missing_unique_rgb_frames']!=0:raise ValueError('Official test incomplete')
    vace,hmr=load_pair(a.vace_cache,a.hmr_cache,'test')
    uids=[uid for row,m in enumerate(vace['metadata']) for uid,owner in zip(m['target_frame_uids'],vace['frame_owner_mask'][row].tolist()) if owner]
    if len(uids)!=len(set(uids)) or set(uids)!=expected:raise ValueError('Test cache ownership differs from official reference')
    if vace['signature'].get('development_only'):raise ValueError('Development test caches are forbidden')
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if payload.get('test_used_for_model_selection') is not False:raise ValueError('Missing no-test-selection provenance')
    train_val=set(payload['coverage']['sequence_names'])
    if train_val&{m['sequence'] for m in vace['metadata']}:raise ValueError('Test and selection sequences overlap')
    freeze=dict(checkpoint_sha256=digest(a.checkpoint),config_sha256=digest(a.config),
                test_cache=str(a.vace_cache),sidecar=str(a.hmr_cache),test_used_for_model_selection=False)
    (a.output/'freeze.json').write_text(json.dumps(freeze,indent=2))
    models=create_smpl_models(Path(c['smpl-model-root']),device)
    assets=Path(c['smpl-model-root']);official=SMPL(model_path=str(assets),gender='neutral',num_betas=10,
                   joint_regressor_extra=str(assets.parent/'SMPL_to_J19.pkl')).to(device)
    model=model_from_payload(payload).to(device).bind_geometry(models[2]).bind_landmarks(official).eval()
    args=SimpleNamespace(amp_dtype='bf16',geometry_only_validation=True)
    regressor=torch.as_tensor(np.load(c['joint-regressor']),dtype=torch.float32,device=device)
    loader=DataLoader(V10Dataset(vace,hmr),batch_size=16,shuffle=False,num_workers=0)
    metrics=validate(loader,model,models,regressor,args,device)
    if metrics['frames']!=35515 or metrics['acceleration_triplets']!=34752:raise ValueError('Incomplete test metric coverage')
    manifest_sha=hashlib.sha256(('\n'.join(sorted(expected))+'\n').encode()).hexdigest()
    result=dict(model='V29',metrics=metrics,full_coverage=True,owned_person_frames=35515,
                test_used_for_model_selection=False,frozen_checkpoint_sha256=freeze['checkpoint_sha256'],
                frame_manifest_sha256=manifest_sha,evaluator_sha256=digest(inspect.getfile(validate)),
                evidence_type='local_official_protocol_reproduction',sota_status='not_established',
                protocol=dict(dataset='3DPW',split='test',joint_set='H36M-regressed J14',root_alignment='SMPL pelvis',
                              pve_alignment='unaligned model-centred vertices',body_representation='SMPL',
                              gender_policy='neutral prediction; gendered annotation target',
                              input_bbox_policy=hmr['signature'].get('crop_protocol','unverified'),
                              temporal_input_frames=8,fps='annotation frame indices; acceleration mm/frame^2',unit='mm'),
                independent_audit=audit,freeze=freeze)
    (a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
