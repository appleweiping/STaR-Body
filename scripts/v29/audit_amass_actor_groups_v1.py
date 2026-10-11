"""Find named-actor overlap in directory splits; preserve raw AMASS audits."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    raw=a.manifest.read_bytes();m=json.loads(raw.decode('utf-8'))
    if m['format']!='v29-amass-raw-parameter-audit-v2' or m['status']!='complete':raise ValueError('Complete raw parameter audit required')
    groups={};unresolved=[]
    for row in m['files']:
        if row['status']!='accepted_raw_parameters':raise ValueError('Rejected raw parameter found')
        match=re.match(r'^(Female|Male)([0-9]+)(?:[A-Za-z_]|$)',row['subject_directory'],re.I)
        if row['dataset']!='ACCAD' or match is None:
            unresolved.append(dict(source_relative_path=row['source_relative_path'],source_sha256=row['source_sha256'],
                directory_subject=row['subject_directory'],frames=row['frames'],reason='No verified actor alias mapping; hold outside training and validation'))
            continue
        gender=match.group(1).lower()
        if row['gender']!=gender:raise ValueError('Declared actor prefix conflicts with raw model gender')
        actor=f'ACCAD/{gender}_{int(match.group(2))}'
        groups.setdefault(actor,[]).append(row)
    if len(groups)<2:raise ValueError('At least two named actor groups needed')
    ordered=sorted(groups,key=lambda actor:hashlib.sha256(('2026:'+actor).encode()).hexdigest())
    count=max(1,min(len(ordered)-1,int(.8*len(ordered))));training=set(ordered[:count])
    decisions=[];named_rows=[];old_overlap=[]
    for actor in sorted(groups):
        rows=groups[actor];old_splits=sorted({r['split'] for r in rows});split='train' if actor in training else 'validation'
        if old_splits==['train','validation']:old_overlap.append(actor)
        decisions.append(dict(dataset_named_actor=actor,action_directories=sorted({r['subject_directory'] for r in rows}),
            old_splits=old_splits,new_split=split,sequences=len(rows),frames=sum(r['frames'] for r in rows),
            verified_real_identity=False,grouping_basis='declared directory actor prefix, not biometric identity'))
        for row in rows:named_rows.append(row|dict(previous_split=row['split'],split=split,conservative_actor_group=actor))
    report=dict(format='v29-amass-named-actor-regroup-v1',source_manifest_sha256=hashlib.sha256(raw).hexdigest(),
        source_unchanged=True,named_actor_groups=len(groups),old_named_actor_cross_split=old_overlap,
        decisions=decisions,unresolved_sequences=len(unresolved),unresolved_frames=sum(r['frames'] for r in unresolved),
        unresolved=unresolved,named_files=named_rows,
        statistics={split:dict(sequences=sum(r['split']==split for r in named_rows),
            frames=sum(r['frames'] for r in named_rows if r['split']==split),
            actor_groups=sum(r['new_split']==split for r in decisions)) for split in ('train','validation')},
        new_named_actor_cross_split=[],raw_source_repaired=False,used_for_model_training=False,
        needs_raw_beta_fingerprints_and_source_identity_audit=True,ready_for_full_amass_training=False)
    with (a.output/'report.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,ensure_ascii=False)
    lines=['# AMASS 演员分组复核','',f'原目录划分中，{len(old_overlap)} 个明确命名的演员编号跨 train/validation。',
        '',f'可保守分组的命名演员编号 {len(groups)} 个；无法确认映射的动作 {len(unresolved)} 个已标记排除。',
        '', '此结果使用目录声明的演员编号，不等于生物身份验证。原始数据、原审计和已训练的 V29Y 均未修改。',
        '', 'AMASS 尚未用于 V29Y，因此本问题不改变现有 3DPW 验证结果。未来使用动作先验或渲染增广时，应先应用一致的演员分组，并补充原始 betas 指纹和目录来源核对。']
    (a.output/'SUMMARY_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('named_actor_groups','old_named_actor_cross_split','unresolved_sequences','unresolved_frames','statistics')},ensure_ascii=False))


if __name__=='__main__':main()
