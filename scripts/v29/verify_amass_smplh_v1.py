"""Bounded native SMPLH reconstruction proof; no SMPL pseudo-target creation."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import torch


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def safe_fields(path,names):
    with zipfile.ZipFile(path) as archive:
        if len(archive.namelist())!=len(set(archive.namelist())):raise ValueError('Duplicate members')
        result={}
        for name in names:
            member=name+'.npy';info=archive.getinfo(member)
            if info.file_size>1024**3 or info.flag_bits&1:raise ValueError('Unsupported member')
            value=np.load(io.BytesIO(archive.read(member)),allow_pickle=False)
            if value.dtype.hasobject:raise ValueError('Object arrays forbidden')
            if value.dtype.kind in 'fiu' and not np.isfinite(value).all():raise ValueError('Nonfinite '+name)
            result[name]=value
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False);root=a.root.resolve();torch.set_num_threads(4)
    manifest_path=root/'runs/amass_inventory_v2_20261006/manifest.json'
    audit=json.loads(manifest_path.read_text())
    accepted=[r for r in audit['files'] if r['status']=='accepted' and r['split']=='train']
    if not accepted:raise ValueError('Verified train-only native AMASS data required')
    row=accepted[0];source=Path(row['source_path'])
    if digest(source)!=row['source_sha256']:raise ValueError('Raw AMASS changed')
    data=safe_fields(source,['poses','betas','gender','trans','mocap_framerate','dmpls'])
    gender=str(data['gender'].item()).lower()
    if gender not in ('male','female','neutral'):raise ValueError('Unsupported gender')
    model=Path('../../MotionBERT/data/AMASS/body_models').resolve()/'smplh'/gender/'model.npz'
    declared=json.loads((root/'runs/amass_smplh_asset_contract_v1.json').read_text())
    expected=next(r['sha256'] for r in declared['models'] if r['gender']==gender)
    if digest(model)!=expected:raise ValueError('Native model changed')
    fields=safe_fields(model,['v_template','shapedirs','posedirs','J_regressor','weights','kintree_table','f'])
    if fields['shapedirs'].shape!=(6890,3,16) or fields['J_regressor'].shape!=(52,6890):raise ValueError('Unexpected native SMPLH contract')
    ktree=fields['kintree_table']
    if not np.array_equal(ktree[1],np.arange(52)):raise ValueError('Non-indexed parent mapping requires explicit conversion')
    parents=torch.tensor(ktree[0].astype(np.int64));parents[0]=-1
    if any(not 0<=int(parents[i])<i for i in range(1,52)):raise ValueError('Invalid ordered kinematic tree')
    ids=np.linspace(0,len(data['poses'])-1,num=min(3,len(data['poses'])),dtype=np.int64)
    pose=torch.tensor(data['poses'][ids],dtype=torch.float32,requires_grad=True)
    if pose.shape[1]!=156 or data['betas'].shape!=(16,):raise ValueError('Pose/beta contract mismatch')
    betas=torch.tensor(data['betas'],dtype=torch.float32)[None].expand(len(ids),-1)
    from smplx.lbs import lbs
    tensor=lambda name:torch.tensor(fields[name],dtype=torch.float32)
    args=(betas,pose,tensor('v_template'),tensor('shapedirs'),
        tensor('posedirs').permute(2,0,1).reshape(459,-1),tensor('J_regressor'),parents,tensor('weights'))
    vertices,joints=lbs(*args,pose2rot=True)
    if tuple(vertices.shape)!=(len(ids),6890,3) or tuple(joints.shape)!=(len(ids),52,3):raise ValueError('Unexpected output')
    if not torch.isfinite(vertices).all() or not torch.isfinite(joints).all():raise FloatingPointError('Invalid reconstruction')
    loss=vertices.square().mean();loss.backward()
    if not torch.isfinite(pose.grad).all() or float(pose.grad.norm())<=0:raise ValueError('Native LBS gradient failed')
    with torch.no_grad():repeat,_=lbs(*args,pose2rot=True)
    torch.testing.assert_close(vertices,repeat,rtol=0,atol=0)
    # Compare only articulated parent structure, never declare parameter/mesh equality.
    from divibody_training.evaluate_standard_3dpw import create_smpl_models
    cfg=json.loads((root/'configs/geometry.json').read_text())
    standard=create_smpl_models(Path(cfg['smpl-model-root']),torch.device('cpu'))[2]
    first22_equal=bool(torch.equal(parents[:22],standard.parents.cpu()[:22]))
    array=vertices.detach().cpu().numpy();faces=fields['f'].astype(np.int64)
    np.savez_compressed(a.output/'native_smplh_frames.npz',vertices=array,joints=joints.detach().cpu().numpy(),faces=faces,frame_indices=ids)
    result=dict(status='passed',source_split='train',raw_source=str(source),raw_sha256=row['source_sha256'],
        smplh_model_sha256=expected,manifest_sha256=digest(manifest_path),gender=gender,frames=ids.tolist(),
        native_vertices=6890,native_joints=52,shape_coefficients=16,pose_dimensions=156,
        differentiable_lbs_gradient_norm=float(pose.grad.norm()),repeat_max_difference=float((vertices-repeat).abs().max()),
        first22_parent_structure_matches_smpl=first22_equal,smpl_parameter_or_vertex_equivalence_verified=False,
        dmpl_dynamic_offsets_applied=False,translation_applied=False,world_frame_or_camera_conversion_done=False,
        used_for_model_training=False,rgb_or_feature_cache_created=False)
    with (a.output/'result.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
