"""Eight fixed train frames: verify mapping, then evaluate GT-free fitting.

All labels are confined to reporting after fitting. This is a train diagnostic,
not validation/test, and cannot establish generalization or SOTA.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import cv2
import numpy as np
import torch

from mhr_smpl_anchor_v1 import mapped_surface, fit_smpl_surface


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(); args.output.mkdir(exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(2026)
    device = torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(8 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    original = json.loads((root / 'original_code_hashes.json').read_text())
    if any(digest(root / name) != sha for name, sha in original.items()):
        raise ValueError('An original project file changed')
    code = root / 'research_dependencies/MHR_conversion_sources_v2_20261007'
    manifest = json.loads((code / 'source_manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('Incomplete official source snapshot')
    for row in manifest['files']:
        if digest(code / row['path']) != row['sha256']:
            raise ValueError('Official conversion source/asset changed')
    official = code / 'tools/mhr_smpl_conversion/conversion.py'
    mapping_path = official.parent / 'assets/mhr2smpl_mapping.npz'
    with np.load(mapping_path, allow_pickle=False) as arrays:
        triangle_ids = torch.tensor(arrays['triangle_ids'].astype(np.int64), device=device)
        barycentric = torch.tensor(arrays['baryc_coords'], dtype=torch.float32, device=device)
    teacher_dir = root / 'runs/sam3d_strong_features64_v4_20261006'
    extraction = json.loads((teacher_dir / 'result.json').read_text())
    declaration = extraction['manifest']
    if extraction['status'] != 'passed' or declaration['split'] != 'train' or declaration['no_test_frames'] is not True:
        raise ValueError('Actual completed train-only teacher inference required')
    checkpoint = root.parent / 'checkpoints/sam-3d-body-dinov3/model.ckpt'
    mhr_path = root.parent / 'checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt'
    if digest(checkpoint) != declaration['checkpoint_sha256'] or digest(mhr_path) != declaration['mhr_sha256']:
        raise ValueError('Teacher model assets changed')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    faces_cpu = state['head_pose.faces'].long()
    jit = torch.jit.load(str(mhr_path), map_location='cpu')
    native_faces = dict(jit.named_buffers())['character_torch.mesh.faces'].long()
    if not torch.equal(faces_cpu, native_faces):
        raise ValueError('Teacher and native MHR face order differs; mapping unsafe')
    faces = faces_cpu.to(device); del jit, state
    config = json.loads((root / 'configs/geometry.json').read_text())
    from divibody_training.ram_parts_runtime_v29 import V10Dataset
    from divibody_training.evaluate_standard_3dpw import create_smpl_models
    from divibody_training.train_divibody_continuous_geometry_v5 import camera_targets, gendered_vertices, smpl_vertices
    from divibody_training.geometry_metrics_3dpw import pelvis_aligned_j14, mean_euclidean_error, pa_mpjpe_per_sample
    plan = json.loads((root / 'training_runs/v29y_strong_rgb_20261006/plan.json').read_text())['strong_motion_40']
    vace = torch.load(plan['train-vace-cache'], map_location='cpu', weights_only=False, mmap=True)
    hmr = torch.load(plan['train-hmr-sidecar'], map_location='cpu', weights_only=False, mmap=True)
    if vace['signature']['split'] != 'train' or hmr['signature']['split'] != 'train':
        raise ValueError('Strict training pair required')
    dataset = V10Dataset(vace, hmr)
    lookup = {}
    for row, metadata in enumerate(vace['metadata']):
        for col, uid in enumerate(metadata['target_frame_uids']):
            if bool(vace['frame_owner_mask'][row, col] and vace['target_valid_mask'][row, col]):
                if uid in lookup:
                    raise ValueError('Duplicate train owner')
                lookup[uid] = row, col
    selected = [extraction['samples'][int(index)] for index in np.linspace(0, len(extraction['samples'])-1, 8, dtype=np.int64)]
    vertices, rotations, shapes, target_labels, evidence = [], [], [], [], []
    reprojection_max = 0.0
    for row in selected:
        index, col = lookup[row['uid']]; metadata = vace['metadata'][index]
        if Path(row['image']).resolve() != Path(metadata['context_image_paths'][4 + col]).resolve():
            raise ValueError('Teacher image does not match actual source frame')
        with np.load(teacher_dir / row['file'], allow_pickle=False) as arrays:
            surface = arrays['pred_vertices'].copy()
            cam = arrays['pred_cam_t'].copy()
            points = arrays['pred_keypoints_3d'] + cam[None]
            image = cv2.imread(row['image'])
            if image is None or np.any(points[:, 2] <= 0):
                raise ValueError('Invalid teacher camera projection')
            projection = points[:, :2] / points[:, 2:] * float(arrays['focal_length'])
            projection += np.array([image.shape[1]/2, image.shape[0]/2])
            difference = float(np.abs(projection-arrays['pred_keypoints_2d']).max())
            reprojection_max = max(reprojection_max, difference)
            if not np.allclose(arrays['bbox'], metadata['context_bbox_xyxy'][4+col], atol=1e-4, rtol=0):
                raise ValueError('Teacher input crop differs')
        sample = dataset[index]
        vertices.append(torch.tensor(surface, device=device))
        rotations.append(sample['base_rotmat'][col]); shapes.append(sample['base_betas'][col])
        target_labels.append({k: sample[k][col] for k in ('full_pose', 'betas', 'gender_code', 'cam_pose')})
        evidence.append(dict(uid=row['uid'], image_sha256=digest(row['image']), npz_sha256=digest(teacher_dir/row['file'])))
    if reprojection_max > 0.05:
        raise ValueError('Teacher coordinate/camera reprojection differs: '+str(reprojection_max))
    source = torch.stack(vertices).float()
    target = mapped_surface(source, faces, triangle_ids, barycentric)
    # Execute only the preserved official mapping method AST, with real assets;
    # no MHR or PyMomentum import or optimizer substitutions are necessary.
    tree = ast.parse(official.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Conversion')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_compute_target_vertices')
    scope = dict(torch=torch, np=np, tqdm=lambda iterable, **kwargs: iterable,
                 ConversionConstants=SimpleNamespace(CENTIMETERS_TO_METERS=.01, METERS_TO_CENTIMETERS=100.),
                 logger=SimpleNamespace(info=lambda *args: None))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(official), 'exec'), scope)
    reference_self = SimpleNamespace(_DEVICE=device, _batch_size=8,
        _mhr_model=SimpleNamespace(character=SimpleNamespace(mesh=SimpleNamespace(faces=faces_cpu.numpy()))),
        _load_surface_mapping_MHR2SMPL=lambda: (triangle_ids.cpu().numpy(), barycentric.cpu().numpy()))
    reference = scope['_compute_target_vertices'](reference_self, source*100., 'mhr2smpl')
    torch.testing.assert_close(target, reference, atol=1e-6, rtol=1e-6)
    models = create_smpl_models(Path(config['smpl-model-root']), device)
    initial_r = torch.stack(rotations).to(device).float(); initial_b = torch.stack(shapes).to(device).float()
    started = time.monotonic()
    fit = fit_smpl_surface(models[2], target, initial_r, initial_b, steps=400)
    np.savez_compressed(args.output/'fitted_train_anchors.npz', rotmat=fit['rotmat'].cpu().numpy(),
                       betas=fit['betas'].cpu().numpy(), translation=fit['translation'].cpu().numpy(),
                       uids=np.asarray([r['uid'] for r in selected]))
    # Development labels only enter here, after the immutable fitting output.
    batch = {k: torch.stack([r[k] for r in target_labels])[:, None] for k in target_labels[0]}
    regressor = torch.tensor(np.load(config['joint-regressor'], allow_pickle=False), dtype=torch.float32, device=device)
    with torch.no_grad():
        gt = gendered_vertices(models, camera_targets(batch, device).reshape(-1, 24, 3, 3),
            batch['betas'].to(device).reshape(-1, 10), batch['gender_code'].to(device).reshape(-1))
        baseline = smpl_vertices(models[2], initial_r, initial_b)
        gj, _ = pelvis_aligned_j14(gt, regressor)
        def metrics(v):
            pj, _ = pelvis_aligned_j14(v, regressor)
            return dict(mpjpe_mm=float(mean_euclidean_error(pj, gj).mean()*1000),
                pa_mpjpe_mm=float(pa_mpjpe_per_sample(pj, gj).mean()*1000),
                pve_mm=float(mean_euclidean_error(v, gt).mean()*1000))
        before, after = metrics(baseline), metrics(fit['vertices'])
    report = dict(status='complete', split='train', frames=8, uids=[r['uid'] for r in selected],
        steps=400, seconds=time.monotonic()-started, teacher_reprojection_max_pixels=reprojection_max,
        official_mapping_max_difference_m=float((target-reference).abs().max()), teacher_native_face_order_equal=True,
        initial_surface_error_mm=fit['initial_surface_error_mm'].cpu().tolist(),
        fitted_surface_error_mm=fit['surface_error_mm'].cpu().tolist(), history=fit['history'],
        hmr_anchor_train_metrics=before, fitted_teacher_train_metrics=after,
        delta={k: after[k]-before[k] for k in before}, evidence=evidence,
        mapping_sha256=digest(mapping_path), official_source_sha256=digest(official),
        sampler='Eight evenly spaced indices of the frozen 64 train-only teacher samples',
        gt_used_for_fitting=False, labels_used_only_for_train_diagnostic=True,
        camera_translation_not_replaced=True, native_smpl_outputs=True,
        official_parameter_solver=False, independent_generalization_evidence=False,
        model_training_started=False, sota_status='not_established', original_python_files_unchanged=len(original))
    with (args.output/'result.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
