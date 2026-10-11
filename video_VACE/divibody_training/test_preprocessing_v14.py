"""Run invariant checks using the copied production window planner."""
import numpy as np
import argparse
from dataclasses import asdict
from collections import Counter,defaultdict
from PIL import Image
from divibody_training.preprocessing_v14 import square_bbox
from divibody_training import cache_vace_phase_v14 as new
import cache_continuous_3dpw_vace_features_v5 as old
from cache_continuous_3dpw_vace_features import FrameIdentity

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root')
    args=parser.parse_args()
    frames=[]
    for person in range(2):
        for fid in list(range(37))+list(range(50,55)):
            frames.append(FrameIdentity(len(frames),'sequence',person,fid,fid*2,'unused'))
    initial=old.build_v5_window_plan(frames)
    slots=defaultdict(set)
    for phase in [0,2,4,6]:
        new.WINDOW_PHASE=phase
        plan=new.build_v5_window_plan(frames)
        new.validate_v5_window_plan(plan)
        owned=Counter(f.uid for w in plan.windows for f,keep in zip(w.target_frames,w.frame_owner) if keep)
        assert set(owned)=={f.uid for f in frames}
        assert set(owned.values())=={1}
        for w in plan.windows:
            assert w.context_frames[4:12]==w.target_frames
            assert len({f.person_id for f in w.context_frames})==1
            for i,(f,valid) in enumerate(zip(w.target_frames,w.target_valid)):
                if valid: slots[f.uid].add(i)
        if phase==0:
            assert [asdict(w) for w in plan.windows]==[asdict(w) for w in initial.windows]
        print('phase',phase,'windows',len(plan.windows),'owned',len(owned),flush=True)
    assert len(slots['sequence/person_0/frame_00020'])==4
    points=np.array([[0.,70.,1.],[30.,95.,1.],[10.,80.,1.]])
    box,valid=square_bbox(points,100,100)
    assert valid and box[0]<0 and box[2]-box[0]==box[3]-box[1]
    im=Image.new('RGB',(100,100),'white').crop(box).resize((96,96))
    assert np.asarray(im)[:,0].max()==0
    for missing in (None,np.zeros((3,5))):
        box,valid=square_bbox(missing,200,100)
        assert not valid and box[2]-box[0]==box[3]-box[1]==200
    if args.data_root:
        from divibody_training.evaluate_standard_3dpw import Standard3DPWDataset
        base=Standard3DPWDataset(root=args.data_root,split='train',image_size=96)
        new.CROP_MODE='square_pad'
        data=new.V5PersonTubeDataset(base,None,True)
        for index in (0,len(base)//2,len(base)-1):
            data.crop_mode='legacy'
            assert np.array_equal(data._feature_crop(index)['image'].numpy(),base[index]['image'].numpy())
            data.crop_mode='square_pad'
            crop=data._feature_crop(index)
            b=crop['bbox_xyxy']
            assert b[2]-b[0]==b[3]-b[1]
            assert crop['crop_transform'][0,0]==crop['crop_transform'][1,1]
            assert 0<=float(crop['padding_fraction'])<=1
            assert crop['image'].shape==(3,96,96)
        print('REAL_RGB_SQUARE_PAD_OK',flush=True)
    print('V14_PREPROCESSING_TEST_OK',flush=True)

if __name__=='__main__':main()
