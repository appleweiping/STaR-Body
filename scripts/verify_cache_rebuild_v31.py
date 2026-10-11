"""Compare rebuilt real cache tensors by immutable UID without fitting labels."""
import argparse
import hashlib
import json
from pathlib import Path
import time

def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):value.update(chunk)
    return value.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=('vace','hmr'),required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    import torch
    torch.set_num_threads(4)
    started=time.monotonic()
    ref=torch.load(a.reference,map_location='cpu',weights_only=False,mmap=True)
    new=torch.load(a.candidate,map_location='cpu',weights_only=False,mmap=True)
    if ref['signature']['split']!=new['signature']['split']:raise ValueError('Different splits')
    if a.kind=='vace':
        ids=[r['window_uid'] for r in ref['metadata']]
        new_ids=[r['window_uid'] for r in new['metadata']]
        fields=['features','context_edge_repeat_mask','target_valid_mask','frame_owner_mask','velocity_owner_mask','acceleration_owner_mask']
    else:
        ids=ref['frame_uids'];new_ids=new['frame_uids']
        fields=['rgb_tokens','base_rotmat','base_betas','base_camera']
    if len(ids)!=len(set(ids)) or len(new_ids)!=len(set(new_ids)):raise ValueError('Duplicate UID')
    lookup={uid:i for i,uid in enumerate(new_ids)}
    if any(uid not in lookup for uid in ids):raise ValueError('Reference UIDs missing in rebuilt cache')
    locations=[lookup[uid] for uid in ids]
    stats={};same=True
    for name in fields:
        x,y=ref[name],new[name]
        if x.shape[1:]!=y.shape[1:] or x.dtype!=y.dtype:raise ValueError('Tensor contract differs: '+name)
        maximum=total=elements=nonzero=0
        for i,j in enumerate(locations):
            arow,brow=x[i],y[j]
            if not torch.isfinite(arow).all() or not torch.isfinite(brow).all():raise ValueError('Nonfinite '+name)
            diff=(arow.double()-brow.double()).abs()
            maximum=max(maximum,float(diff.max()));total+=float(diff.sum());elements+=diff.numel();nonzero+=int((diff!=0).sum())
        stats[name]={'max_absolute_difference':maximum,'mean_absolute_difference':total/elements,'different_elements':nonzero,'elements':elements,'dtype':str(x.dtype)}
        same= same and nonzero==0
    metadata_checks={}
    if a.kind=='vace':
        keys=('sequence','person_id','target_frame_ids','target_source_frame_ids','target_frame_uids',
              'context_frame_ids','context_source_frame_ids','context_frame_uids','context_image_paths',
              'context_bbox_xyxy','context_crop_applied','noise_seed')
        for key in keys:
            equal=all(ref['metadata'][i].get(key)==new['metadata'][j].get(key) for i,j in enumerate(locations))
            metadata_checks[key]=equal;same=same and equal
        for key in ref.get('targets',{}):
            equal=all(torch.equal(ref['targets'][key][i],new['targets'][key][j]) for i,j in enumerate(locations))
            metadata_checks['targets/'+key]=equal;same=same and equal
    else:
        for key in ('hmr2a_checkpoint_sha256','anchor_revision','raw_bbox_margin','crop_protocol'):
            equal=ref['signature'].get(key)==new['signature'].get(key)
            metadata_checks[key]=equal;same=same and equal
    result={'status':'exact_match' if same else 'differences_require_diagnosis','kind':a.kind,
        'split':ref['signature']['split'],'compared_uids':len(ids),'candidate_uid_count':len(new_ids),
        'tensor_comparison':stats,'metadata_comparison':metadata_checks,
        'reference':str(a.reference),'reference_sha256':digest(a.reference),
        'candidate':str(a.candidate),'candidate_sha256':digest(a.candidate),
        'development_only':new['signature'].get('development_only'),
        'elapsed_seconds':time.monotonic()-started,'training_or_teacher_fitting':False,
        'note':'This is cache numerical/identity parity, not a reconstruction accuracy result.'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result),flush=True)
    raise SystemExit(0 if same else 2)

if __name__=='__main__':main()
