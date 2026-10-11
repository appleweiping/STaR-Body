"""Offline image-level normalizer audit from the immutable v38 prediction cache.

Copied from the v38 evaluator into an additive analysis script. No inference,
training, or 3DPW test data is used.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import torch

def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

def metric(distance,visible,denom):
 ok=visible&torch.isfinite(denom)&(denom>0); x=(distance/denom).masked_select(ok).double()
 return {'PCKn@0.05':float((x<=.05).mean()),'PCKn@0.10':float((x<=.10).mean()),'mean_error':float(x.mean()),'visible_keypoints':x.numel()}

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--prediction-cache',type=Path,required=True); p.add_argument('--annotation-json',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
 if a.output.exists(): raise FileExistsError(a.output)
 d=torch.load(a.prediction_cache,map_location='cpu',weights_only=False)
 ann=json.loads(a.annotation_json.read_text())
 images={int(x['id']):(float(x['width']),float(x['height'])) for x in ann['images']}; anns={int(x['id']):int(x['image_id']) for x in ann['annotations']}
 wh=torch.tensor([images[anns[int(i)]] for i in d['annotation_id']],dtype=torch.float32)
 distance_crop=torch.linalg.vector_norm(d['pred']-d['gt'],dim=-1); distance_original=distance_crop*d['crop_side'][:,None]/256.
 visible=d['visible']; subsets={'all17':tuple(range(17)),'body15_no_head_segment':(0,3,4,*range(5,17)),'body12_no_face_head':tuple(range(5,17))}
 denoms={'original_image_max_side':wh.amax(-1),'original_image_diagonal':torch.linalg.vector_norm(wh,dim=-1),'original_image_sqrt_area':wh.prod(-1).sqrt()}
 results={}
 for name,base in denoms.items():
  results[name]={}
  for subset,idx in subsets.items():
   choose=torch.zeros(17,dtype=torch.bool); choose[list(idx)]=True
   results[name][subset]=metric(distance_original,visible&choose[None],base[:,None].expand_as(distance_original))
 payload={'status':'POSETRACK_IMAGE_NORMALIZER_AUDIT_V4_OK','split':'public PoseTrack18 val','prediction_cache_sha256':sha256(a.prediction_cache),'source_sha256':sha256(Path(__file__)),'results':results,'test_data_used':False,'claim':'diagnostic only; image normalization is not asserted to be RAM protocol'}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(payload,indent=2),encoding='utf-8'); print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
