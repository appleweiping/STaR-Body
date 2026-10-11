"""Refuse SOTA claims without complete matched, frozen official-test evidence."""
import argparse
import json
from pathlib import Path

KEYS=('mpjpe_mm','pa_mpjpe_mm','pve_mm')
PROTOCOL=('dataset','split','joint_set','root_alignment','pve_alignment','body_representation',
          'gender_policy','input_bbox_policy','temporal_input_frames','fps','unit')


def assess(candidate,baselines):
    missing=[]
    for name,item in [('candidate',candidate),*[(str(i),b) for i,b in enumerate(baselines)]]:
        if item.get('protocol',{}).get('dataset')!='3DPW' or item.get('protocol',{}).get('split')!='test':
            missing.append(name+':requires_official_3dpw_test')
        if not item.get('full_coverage') or item.get('owned_person_frames')!=35515:
            missing.append(name+':requires_complete_test_coverage')
        if item.get('test_used_for_model_selection') is not False or not item.get('frozen_checkpoint_sha256'):
            missing.append(name+':requires_frozen_checkpoint_without_test_selection')
        if item.get('evidence_type')!='local_official_protocol_reproduction':
            missing.append(name+':paper_table_is_not_a_local_reproduction')
        if not item.get('frame_manifest_sha256') or not item.get('evaluator_sha256'):
            missing.append(name+':missing_evaluation_provenance')
        for k in PROTOCOL:
            if k not in item.get('protocol',{}):missing.append(name+':missing_protocol:'+k)
    if not baselines:missing.append('no_verified_comparator')
    for baseline in baselines:
        if candidate.get('protocol')!=baseline.get('protocol'):
            missing.append('protocol_mismatch:'+baseline.get('model','unknown'))
        for key in ('frame_manifest_sha256','evaluator_sha256'):
            if candidate.get(key)!=baseline.get(key):missing.append('evaluation_mismatch:'+key)
    if missing:return dict(status='not_established',reasons=missing)
    wins={}
    for baseline in baselines:
        delta={k:candidate['metrics'][k]-baseline['metrics'][k] for k in KEYS}
        intervals=candidate.get('paired_sequence_bootstrap',{}).get(baseline['model'],{})
        robust=all(k in intervals and intervals[k]['upper95']<0 for k in KEYS)
        wins[baseline['model']]=dict(deltas=delta,all_metrics_better=all(v<0 for v in delta.values()),
                                    paired_confidence_better=robust)
    established=all(w['all_metrics_better'] and w['paired_confidence_better'] for w in wins.values())
    return dict(status='superior_to_named_comparators' if established else 'not_established',comparisons=wins,
                scope='Only the explicitly named reproduced comparators; no universal latest-SOTA claim')


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--baselines',type=Path,nargs='*',default=[]);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=assess(json.loads(a.candidate.read_text()),[json.loads(b.read_text()) for b in a.baselines])
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
