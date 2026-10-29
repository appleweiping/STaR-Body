"""Pre-registered Target60 validation-only selection; unchanged metrics."""
import math
def selection_report(geometry,pose,baseline_geometry,baseline_pose,args,best):
    keys=('mpjpe_mm','pa_mpjpe_mm','pve_mm','acceleration_error_mm_per_frame2')
    if not all(math.isfinite(geometry[k]) and math.isfinite(baseline_geometry[k]) for k in keys):raise FloatingPointError('Nonfinite target60 metrics')
    ratio=max(geometry['mpjpe_mm']/args.target_mpjpe_mm,geometry['pa_mpjpe_mm']/args.target_pa_mpjpe_mm)
    reasons=[]
    for k,tol in (('mpjpe_mm',args.mpjpe_tolerance_mm),('pa_mpjpe_mm',args.pa_mpjpe_tolerance_mm),('pve_mm',args.pve_tolerance_mm)):
        if geometry[k]>baseline_geometry[k]+tol:reasons.append(k+'_above_initial_v30')
    if geometry['acceleration_error_mm_per_frame2']>args.acceleration_ceiling_mm+args.acceleration_tolerance_mm:reasons.append('acceleration_above_initial_v30')
    score=-ratio
    if score<=best:reasons.append('target60_score_not_improved')
    met=geometry['mpjpe_mm']<args.target_mpjpe_mm and geometry['pa_mpjpe_mm']<args.target_pa_mpjpe_mm
    return dict(score=score,target_ratio=ratio,accepted=not reasons,reasons=reasons,
        target_met=met,protected_target_met=met and not any('above_' in x for x in reasons),
        targets=dict(mpjpe_mm=args.target_mpjpe_mm,pa_mpjpe_mm=args.target_pa_mpjpe_mm),
        selection_metric='negative max(MPJPE/60,PA-MPJPE/37.3), validation only')
