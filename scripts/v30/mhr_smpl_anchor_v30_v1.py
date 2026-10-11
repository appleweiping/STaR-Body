"""Bounded, GT-free SMPL fitting to officially mapped SAM3D surfaces.

Mapping follows the preserved official MHR conversion source; the fitting
routine here is a separate research implementation, not the official solver.
SAM pred_vertices are already meters in camera axes. No additional axis flip
is applied to that output. Camera translation stays separate from the anchor.
"""
import numpy as np
import torch


def mapped_surface(vertices_m, faces, triangle_ids, barycentric):
    if vertices_m.ndim != 3 or vertices_m.shape[1:] != (18439, 3):
        raise ValueError('Expected actual SAM3D body vertices [B,18439,3]')
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype != torch.long:
        raise ValueError('Integer MHR face topology required')
    if triangle_ids.shape != (6890,) or barycentric.shape != (6890, 3):
        raise ValueError('Official SMPL correspondence shape differs')
    if not all(torch.isfinite(x).all() for x in (vertices_m, barycentric)):
        raise ValueError('Nonfinite surface mapping')
    if (triangle_ids < 0).any() or (triangle_ids >= len(faces)).any():
        raise ValueError('Triangle indices outside MHR topology')
    if (faces < 0).any() or (faces >= vertices_m.shape[1]).any():
        raise ValueError('Vertex indices outside MHR topology')
    if not torch.allclose(barycentric.sum(-1), torch.ones(6890, device=vertices_m.device), atol=1e-5, rtol=0):
        raise ValueError('Barycentric weights must sum to one')
    return (vertices_m[:, faces[triangle_ids]] * barycentric[None, :, :, None]).sum(2)


def fit_smpl_surface(smpl, target_m, initial_rotmat, initial_betas, steps=400):
    """Only teacher surface and existing HMR predictions are inputs.

    Optimize each frame independently. A per-frame translation aligns surface
    origins, and must not be silently substituted for the HMR camera. This
    routine produces a research anchor, not an exact topology conversion.
    """
    if not 1 <= steps <= 2000:
        raise ValueError('Finite fitting budget 1..2000 required')
    from scipy.spatial.transform import Rotation
    from smplx.lbs import batch_rodrigues
    device = target_m.device
    b = len(target_m)
    if target_m.shape != (b, 6890, 3) or initial_rotmat.shape != (b, 24, 3, 3) or initial_betas.shape != (b, 10):
        raise ValueError('SMPL fitting input shape differs')
    if not all(torch.isfinite(x).all() for x in (target_m, initial_rotmat, initial_betas)):
        raise ValueError('Nonfinite fitting inputs')
    smpl.eval().requires_grad_(False)
    initial_pose = torch.tensor(Rotation.from_matrix(initial_rotmat.detach().cpu().numpy().reshape(-1, 3, 3)).as_rotvec(),
                                dtype=torch.float32, device=device).reshape(b, 72)
    pose = torch.nn.Parameter(initial_pose.clone())
    betas = torch.nn.Parameter(initial_betas.detach().float().clone())
    target = target_m.detach().float()
    def vertices():
        rot = batch_rodrigues(pose.reshape(-1, 3)).reshape(b, 24, 3, 3)
        return smpl(global_orient=rot[:, :1], body_pose=rot[:, 1:], betas=betas, pose2rot=False).vertices, rot
    with torch.no_grad():
        base, _ = vertices()
    initial_translation = target.mean(1) - base.mean(1)
    translation = torch.nn.Parameter(initial_translation.clone())
    faces = torch.tensor(np.asarray(smpl.faces, dtype=np.int64), device=device)
    edges = torch.unique(torch.sort(torch.cat([faces[:, :2], faces[:, 1:], faces[:, [0, 2]]]), dim=1)[0], dim=0)
    target_edges = target[:, edges[:, 0]] - target[:, edges[:, 1]]
    optimizer = torch.optim.Adam([pose, betas, translation], lr=0.02)
    history = []
    initial_error = torch.linalg.vector_norm(base + translation.detach()[:, None] - target, dim=-1).mean(1)
    best_error = initial_error.clone()
    best_pose, best_betas, best_translation = pose.detach().clone(), betas.detach().clone(), translation.detach().clone()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        pred, _ = vertices()
        per_frame = torch.linalg.vector_norm(pred + translation[:, None] - target, dim=-1).mean(1)
        edge_loss = (pred[:, edges[:, 0]] - pred[:, edges[:, 1]] - target_edges).square().mean()
        penalty = 1e-5 * (pose - initial_pose).square().mean() + 1e-5 * betas.square().mean()
        loss = per_frame.mean() + 0.1 * edge_loss + penalty
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite geometry fitting loss')
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in (pose, betas, translation)):
            raise FloatingPointError('Geometry fitting gradient failed')
        with torch.no_grad():
            better = per_frame < best_error
            best_error[better] = per_frame[better]
            for dest, src in ((best_pose, pose), (best_betas, betas), (best_translation, translation)):
                dest[better] = src[better]
        optimizer.step()
        with torch.no_grad():
            betas.clamp_(-5, 5)
        if step + 1 in (150, 300):
            for group in optimizer.param_groups:
                group['lr'] *= 0.3
        if step % 50 == 0 or step == steps - 1:
            history.append(dict(step=step, surface_error_mm=float(per_frame.detach().mean() * 1000)))
    with torch.no_grad():
        pose.copy_(best_pose); betas.copy_(best_betas); translation.copy_(best_translation)
        fitted, rot = vertices()
    return dict(rotmat=rot.detach(), betas=betas.detach(), vertices=fitted.detach(),
                translation=translation.detach(), initial_surface_error_mm=initial_error * 1000,
                surface_error_mm=best_error * 1000, history=history,
                fit_inputs_contain_3d_gt=False, official_parameter_solver=False)
