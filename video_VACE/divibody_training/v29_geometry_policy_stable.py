"""Geometry-first auxiliary gradients and validation checkpoint selection."""
import math
import torch


def merge_auxiliary(optimizer, geometry, auxiliary, ratio):
    if not math.isfinite(ratio) or ratio < 0:
        raise ValueError('Auxiliary gradient ratio must be finite and nonnegative')
    # Protect all body/shape variables together. Observation/camera variables
    # can learn image evidence without an artificial geometry-gradient norm.
    body = [p for group in optimizer.param_groups if group['category'] not in ('pose','camera')
            for p in group['params']]
    active = [p for p in body if geometry[p] is not None]
    active_ids = {id(p) for p in active}
    if active:
        device = active[0].device
        gnorm2 = sum((geometry[p].float().square().sum() for p in active), torch.zeros((),device=device))
        dot = sum(((geometry[p].float()*auxiliary[p].float()).sum() for p in active if auxiliary[p] is not None),
                  torch.zeros((),device=device))
        coefficient = (dot/gnorm2.clamp_min(1e-20)).clamp_max(0.)
        adjusted = {p:(auxiliary[p].clone() if auxiliary[p] is not None else torch.zeros_like(geometry[p]))
                      -coefficient*geometry[p] for p in active}
        anorm = sum((a.float().square().sum() for a in adjusted.values()),torch.zeros((),device=device)).sqrt()
        scale = (ratio*gnorm2.sqrt()/anorm.clamp_min(1e-20)).clamp_max(1.)
        for p in active:
            p.grad = geometry[p]+scale*adjusted[p]
        report=dict(geometry_norm=float(gnorm2.sqrt()),auxiliary_norm_before_cap=float(anorm),
                    cap_scale=float(scale),conflict_removed=bool(dot<0),raw_dot=float(dot))
    else:
        report=dict(geometry_norm=0.,auxiliary_norm_before_cap=0.,cap_scale=0.,conflict_removed=False,raw_dot=0.)
    for group in optimizer.param_groups:
        for p in group['params']:
            if id(p) in active_ids:
                continue
            g,a=geometry[p],auxiliary[p]
            if group['category'] in ('pose','camera'):
                p.grad = (g+a if g is not None and a is not None else g if a is None else a)
            else:
                # No 3D evidence for this body variable: no unconstrained 2D update.
                p.grad = g
    # This constrains the raw gradient, not Adam's preconditioned update or
    # generalization. Strict validation gates are still required.
    return report


def selection_report(geometry, pose, baseline_geometry, baseline_pose, args, best):
    if not all(math.isfinite(v) for v in [*geometry.values(),*pose.values()] if isinstance(v,(int,float))):
        raise FloatingPointError('Nonfinite validation metric')
    deltas={k:geometry[k]-baseline_geometry[k] for k in geometry if isinstance(geometry[k],(int,float))}
    limits={'mpjpe_mm':args.mpjpe_tolerance_mm,'pa_mpjpe_mm':args.pa_mpjpe_tolerance_mm,
            'pve_mm':args.pve_tolerance_mm}
    reasons=[key+'_above_tolerance' for key,tolerance in limits.items() if deltas[key]>tolerance]
    if args.strict_3d_selection and not all(deltas[k]<0 for k in limits):
        reasons.append('not_all_3d_metrics_improved')
    if not args.geometry_only_validation:
        for key in ('projected_PCKn@0.05','projected_PCKn@0.10'):
            if pose[key]<baseline_pose[key]-args.pose_pck_tolerance:reasons.append(key+'_below_tolerance')
    if geometry['acceleration_error_mm_per_frame2']>args.acceleration_ceiling_mm:
        reasons.append('acceleration_above_frozen_validation_ceiling')
    score=-sum(geometry[k]/baseline_geometry[k] for k in limits)
    accepted=not reasons and score>best
    if not reasons and not accepted:reasons.append('geometry_score_not_improved')
    return dict(accepted=accepted,reasons=reasons,score=score,geometry_delta=deltas,
                selection_metric='negative sum of baseline-normalized MPJPE/PA-MPJPE/PVE')
