"""Generate independent full split UID references directly from official 3DPW archive."""
import argparse,hashlib,json,pickle,zipfile
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True)
p.add_argument('--split',choices=('train','validation','test'),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
h=hashlib.sha256()
with a.archive.open('rb') as f:
 for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
uids=[];members={}
with zipfile.ZipFile(a.archive) as z:
 names=sorted(n for n in z.namelist() if n.endswith('.pkl') and a.split in n.split('/') and '__MACOSX' not in n.split('/') and not Path(n).name.startswith('._'))
 for name in names:
  raw=z.read(name);members[name]=hashlib.sha256(raw).hexdigest();d=pickle.loads(raw,encoding='latin1');seq=str(d.get('sequence',Path(name).stem))
  n=len(d['img_frame_ids'])
  for person in range(min(len(d['poses']),len(d['trans']),len(d['betas']),len(d['campose_valid']))):
   valid=np.asarray(d['campose_valid'][person]).reshape(-1)
   for frame in range(n):
    if frame>=len(valid) or not bool(valid[frame]):continue
    pose=np.asarray(d['poses'][person][frame]).reshape(-1);cam=np.asarray(d['cam_poses'][frame]);beta=np.asarray(d['betas'][person]).reshape(-1)
    if pose.size<72 or beta.size<10 or cam.shape!=(4,4) or not all(np.isfinite(v).all() for v in (pose[:72],beta[:10],cam)):raise ValueError('Invalid official valid annotation')
    uids.append(f'{seq}/person_{person}/frame_{frame:05d}')
owned=set(uids)
triplets=sum(f'{u.rsplit("/frame_",1)[0]}/frame_{int(u.rsplit("/frame_",1)[1])-1:05d}' in owned and f'{u.rsplit("/frame_",1)[0]}/frame_{int(u.rsplit("/frame_",1)[1])+1:05d}' in owned for u in owned)
expected={'train':(22735,22481),'validation':(10413,10226),'test':(35515,34752)}
if len(uids)!=len(owned) or (len(uids),triplets)!=expected[a.split]:raise ValueError('Official full split coverage differs')
record=dict(format='star-body-evaluation-reference-v1',split=a.split,frame_uids=sorted(uids),frames=len(uids),
 acceleration_triplets=triplets,independent_archive_verified=True,archive_path=str(a.archive.resolve()),
 archive_sha256=h.hexdigest(),member_sha256=members)
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(dict(status='complete',split=a.split,frames=len(uids),triplets=triplets,output=str(a.output))))
