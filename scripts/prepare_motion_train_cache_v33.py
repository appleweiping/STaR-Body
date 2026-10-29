"""Export canonical, train-only camera/local SMPL rotations for motion pretraining.

No model feature cache is modified. This creates a new supervised training pack;
its GT rotations are never used as inputs to the main image reconstruction model.
"""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'video_VACE'),str(ROOT/'video_VACE/divibody_training')]


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
 return h.hexdigest()


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--train-vace-cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 import torch,numpy as np
 from divibody_training.train_divibody_continuous_geometry_v5 import camera_targets
 v=torch.load(a.train_vace_cache,map_location='cpu',weights_only=False,mmap=True)
 if v['signature'].get('split')!='train' or v['signature'].get('development_only'):
  raise ValueError('Full train cache required; validation/test/development caches rejected')
 a.output.mkdir(parents=True,exist_ok=False)
 rotations=camera_targets(v['targets'],torch.device('cpu')).float()
 tracks={}
 for row,m in enumerate(v['metadata']):
  source=m.get('target_source_frame_ids')
  if source is None:source=m['target_frame_ids']
  for col,uid in enumerate(m['target_frame_uids']):
   if not bool(v['frame_owner_mask'][row,col]&v['target_valid_mask'][row,col]):continue
   key=(m['sequence'],int(m['person_id']));fid=int(m['target_frame_ids'][col])
   group=tracks.setdefault(key,{})
   if fid in group:raise ValueError('Duplicate motion owner')
   group[fid]=(rotations[row,col].numpy(),int(source[col]),uid)
 files=[];excluded=0
 for (sequence,person),frames in sorted(tracks.items()):
  ordered=sorted(frames);segments=[];part=[]
  for fid in ordered:
   if part and fid!=part[-1]+1:segments.append(part);part=[]
   part.append(fid)
  if part:segments.append(part)
  for segment,ids in enumerate(segments):
   if len(ids)<8:excluded+=len(ids);continue
   path=a.output/f'{sequence}_p{person}_s{segment:04d}.npz'
   values=np.stack([frames[i][0] for i in ids])
   if not np.isfinite(values).all():raise ValueError('Nonfinite rotation target')
   np.savez(path,rotmat=values,frame_ids=np.asarray([frames[i][1] for i in ids],dtype=np.int64))
   files.append(dict(file=path.name,sha256=sha(path),frames=len(ids),sequence=sequence,person=person,
                     frame_uid_sha256=hashlib.sha256(('\n'.join(frames[i][2] for i in ids)+'\n').encode()).hexdigest()))
 if not files:raise ValueError('No motion segments of at least eight frames')
 manifest=dict(format='star-motion-rotations-v33',split='train',
  rotation_convention='SMPL camera-root and local-body rotmat',source_cache=str(a.train_vace_cache.resolve()),
  source_cache_sha256=sha(a.train_vace_cache),files=files,excluded_short_segment_frames=excluded,
  labels_used_for='Supervised motion pretraining only; never reconstruction model inputs',source_script_sha256=sha(__file__))
 with (a.output/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
 print(json.dumps(dict(manifest=str(a.output/'manifest.json'),segments=len(files),frames=sum(x['frames'] for x in files))))

if __name__=='__main__':main()
