"""Measure actual geometry differences between frozen pilot fitting outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();a.output.mkdir(exist_ok=False)
    torch.set_num_threads(4);device=torch.device('cpu')
    original=json.loads((root/'original_code_hashes.json').read_text())
    if any(digest(root/name)!=sha for name,sha in original.items()):raise ValueError('Original changed')
    before=root/'runs/mhr_smpl_anchor_pilot_v3_20261007/fitted_train_anchors.npz'
    after=root/'runs/v30_geometry_teacher_v5_20261007/pilot_smpl/000000.npz'
    old_report=json.loads((before.parent/'result.json').read_text())
    new_manifest=json.loads((after.parent/'manifest.json').read_text())
    if old_report['status']!='complete' or old_report['gt_used_for_fitting'] is not False or new_manifest['complete'] is not True:
        raise ValueError('Completed GT-free pilots required')
    if not any(r['file']==after.name and r['sha256']==digest(after) for r in new_manifest['files']):
        raise ValueError('New fitted artifact SHA differs')
    with np.load(before,allow_pickle=False) as d:old={k:d[k].copy() for k in d.files}
    with np.load(after,allow_pickle=False) as d:new={k:d[k].copy() for k in d.files}
    if not np.array_equal(old['uids'],new['uids']) or len(old['uids'])!=64:raise ValueError('Fixed pilot identity differs')
    config=json.loads((root/'configs/geometry.json').read_text())
    from divibody_training.evaluate_standard_3dpw import create_smpl_models
    from divibody_training.train_divibody_continuous_geometry_v5 import smpl_vertices
    smpl=create_smpl_models(Path(config['smpl-model-root']),device)[2].eval().requires_grad_(False)
    started=time.monotonic()
    with torch.no_grad():
        va=smpl_vertices(smpl,torch.tensor(old['rotmat']).float(),torch.tensor(old['betas']).float())
        vb=smpl_vertices(smpl,torch.tensor(new['rotmat']).float(),torch.tensor(new['betas']).float())
        distance=torch.linalg.vector_norm(va-vb,dim=-1)*1000
        aligned=torch.linalg.vector_norm(va-va.mean(1,keepdim=True)-vb+vb.mean(1,keepdim=True),dim=-1)*1000
        translated=torch.linalg.vector_norm(va+torch.tensor(old['translation'])[:,None]-vb-torch.tensor(new['translation'])[:,None],dim=-1)*1000
        delta=torch.tensor(old['rotmat']).double()@torch.tensor(new['rotmat']).double().transpose(-1,-2)
        angle=torch.acos(((delta.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1))*180/np.pi
    residual=np.asarray(old_report['fitted_surface_error_mm'],dtype=np.float64)
    if residual.shape!=(64,):raise ValueError('Historical fit residual identity differs')
    per_frame=[]
    for i,uid in enumerate(old['uids'].tolist()):
        per_frame.append(dict(uid=uid,native_vertex_mean_difference_mm=float(distance[i].mean()),
            centroid_aligned_vertex_mean_difference_mm=float(aligned[i].mean()),
            translated_surface_mean_difference_mm=float(translated[i].mean()),
            maximum_joint_rotation_difference_degrees=float(angle[i].max()),
            old_surface_residual_mm=float(residual[i]),new_surface_residual_mm=float(new['surface_error_mm'][i])))
    report=dict(status='complete',frames=64,fit_equivalence_established=False,labels_used=False,
        maximum_parameter_difference={k:float(np.abs(old[k]-new[k]).max()) for k in ('rotmat','betas','translation')},
        native_vertex_mean_difference_mm=float(distance.mean()),maximum_frame_mean_native_vertex_difference_mm=float(distance.mean(-1).max()),
        maximum_vertex_difference_mm=float(distance.max()),centroid_aligned_mean_difference_mm=float(aligned.mean()),
        translated_surface_mean_difference_mm=float(translated.mean()),maximum_joint_rotation_difference_degrees=float(angle.max()),
        mean_rotation_difference_degrees=float(angle.mean()),maximum_fit_residual_difference_mm=float(np.abs(residual-new['surface_error_mm']).max()),
        old_residual_mean_mm=float(residual.mean()),new_residual_mean_mm=float(new['surface_error_mm'].mean()),
        old_artifact_sha256=digest(before),new_artifact_sha256=digest(after),new_manifest_sha256=digest(after.parent/'manifest.json'),
        solver_copy_sha256=digest(root/'scripts/v30/mhr_smpl_anchor_v30_v1.py'),
        original_solver_sha256=digest(root/'scripts/v29/mhr_smpl_anchor_v1.py'),
        evaluator_sha256=digest(__file__),per_frame=per_frame,seconds=time.monotonic()-started,sota_status='not_established')
    with (a.output/'result.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='per_frame'}),flush=True)


if __name__=='__main__':main()
