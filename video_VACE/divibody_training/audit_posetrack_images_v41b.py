"""Read-only PT18 annotations/shared PT21 image compatibility audit, CPU only."""
import argparse,hashlib,json
from pathlib import Path
from PIL import Image,ImageDraw
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('annotation','npz','image-root','output-dir'): p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=False)
    data=json.loads(a.annotation.read_text()); images={x['file_name']:x for x in data['images']}
    with np.load(a.npz,allow_pickle=True) as n: names=sorted(set(str(x) for x in n['imgname']))
    missing=[]; mismatch=[]; no_size=0
    for name in names:
        path=a.image_root/name
        if not path.is_file(): missing.append(name); continue
        with Image.open(path) as im: size=im.size
        record=images[name]
        if not record.get('width') or not record.get('height'): no_size+=1
        elif size!=(record['width'],record['height']): mismatch.append({'image':name,'actual':size,'declared':[record['width'],record['height']]})
    lookup={x['id']:x for x in data['images']}; persons={}
    for ann in data['annotations']:
        name=lookup[ann['image_id']]['file_name']; persons.setdefault(name,[]).append(ann)
    choices=[names[round(i*(len(names)-1)/11)] for i in range(12)]
    canvas=Image.new('RGB',(384*4,280*3),'white'); samples=[]
    for i,name in enumerate(choices):
        path=a.image_root/name
        with Image.open(path) as raw: im=raw.convert('RGB')
        draw=ImageDraw.Draw(im); width=max(2,im.width//400)
        for ann in persons[name]:
            if 'bbox' in ann:
                x,y,w,h=ann['bbox']
                if w>0 and h>0: draw.rectangle((x,y,x+w,y+h),outline='lime',width=width)
            if len(ann.get('keypoints',[]))!=51: continue
            points=np.array(ann['keypoints']).reshape(17,3)
            for j,(px,py,v) in enumerate(points):
                if v>0:
                    radius=max(3,im.width//200); draw.ellipse((px-radius,py-radius,px+radius,py+radius),fill='magenta')
        im.thumbnail((384,248)); x=(i%4)*384; y=(i//4)*280
        canvas.paste(im,(x+(384-im.width)//2,y))
        ImageDraw.Draw(canvas).text((x+3,y+250),name[-55:],fill='black')
        samples.append({'image':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    canvas.save(a.output_dir/'gt_overlay.png')
    result={'status':'COMPATIBLE' if not missing and not mismatch else 'FAILED',
        'source':'PoseTrack18 annotations + shared PoseTrack21 images',
        'image_root_resolved':str((a.image_root/'images').resolve()),
        'annotation_resolved':str(a.annotation.resolve()),'annotation_sha256':hashlib.sha256(a.annotation.read_bytes()).hexdigest(),
        'npz_unique_images':len(names),'missing':missing,'dimension_mismatches':mismatch,
        'metadata_missing_dimensions':no_size,'samples':samples,'byte_identity_against_pt18_reference_verified':False,
        'original_inputs_modified':False}
    result['overlay_missing_bbox_annotations']=sum('bbox' not in ann for name in choices for ann in persons[name])
    (a.output_dir/'image_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('samples','missing','dimension_mismatches')}),flush=True)
    if missing or mismatch: raise RuntimeError('Image compatibility audit failed; no samples dropped')

if __name__=='__main__': main()
