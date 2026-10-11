"""Paired sequence bootstrap for fixed-model results; never selects weights."""
import argparse
import json
import math
from pathlib import Path
import random

KEYS=('mpjpe_mm','pa_mpjpe_mm','pve_mm')


def compare(candidate,baseline,draws=10000,seed=2026):
    if draws<1000:raise ValueError('At least 1000 draws required')
    for item in (candidate,baseline):
        if item.get('test_used_for_model_selection') is not False or not item.get('frozen_checkpoint_sha256'):
            raise ValueError('Missing frozen checkpoint or no-test-selection evidence')
        if not item.get('full_coverage') or item.get('owned_person_frames')!=35515:
            raise ValueError('Complete official test coverage required')
    for key in ('frame_manifest_sha256','evaluator_sha256','protocol'):
        if not candidate.get(key) or candidate[key]!=baseline.get(key):
            raise ValueError('Paired evaluation provenance differs: '+key)
    left=candidate['metrics']['per_sequence'];right=baseline['metrics']['per_sequence']
    if set(left)!=set(right) or len(left)<2:raise ValueError('Complete matching sequences required')
    rows=[]
    for name in sorted(left):
        a,b=left[name],right[name]
        if a['frames']!=b['frames'] or a['frames']<1:raise ValueError('Sequence frame coverage differs')
        deltas=[a[k+'_sum']-b[k+'_sum'] for k in KEYS]
        if not all(math.isfinite(v) for v in deltas):raise ValueError('Nonfinite per-sequence error')
        rows.append((a['frames'],deltas))
    if sum(r[0] for r in rows)!=candidate['metrics']['frames'] or candidate['metrics']['frames']!=baseline['metrics']['frames']:
        raise ValueError('Aggregate frame coverage differs')
    rng=random.Random(seed);samples={k:[] for k in KEYS}
    for _ in range(draws):
        chosen=[rng.choice(rows) for _ in rows];count=sum(row[0] for row in chosen)
        for i,key in enumerate(KEYS):samples[key].append(sum(row[1][i] for row in chosen)/count)
    intervals={}
    for i,key in enumerate(KEYS):
        values=sorted(samples[key])
        intervals[key]=dict(delta=sum(row[1][i] for row in rows)/sum(row[0] for row in rows),
            lower95=values[int(.025*(draws-1))],upper95=values[int(.975*(draws-1))])
    return dict(intervals=intervals,draws=draws,seed=seed,clusters=len(rows),
                sampling_unit='whole sequence, paired between models; person frames weighted inside each draw')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--baseline-name',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();candidate=json.loads(a.candidate.read_text());baseline=json.loads(a.baseline.read_text())
    result=compare(candidate,baseline)
    report=dict(candidate_model=candidate['model'],baseline_model=a.baseline_name,**result,
        sota_status='not_established',test_used_for_model_selection=candidate['test_used_for_model_selection'],
        candidate_checkpoint_sha256=candidate['frozen_checkpoint_sha256'],
        baseline_checkpoint_sha256=baseline['frozen_checkpoint_sha256'],
        evaluator_sha256=candidate['evaluator_sha256'],protocol=candidate['protocol'],
        frame_manifest_sha256=candidate['frame_manifest_sha256'])
    with a.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report))


if __name__=='__main__':main()
