"""Build corrected raw-bbox HMR2a anchors with the unchanged v8 tensor contract."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from divibody_training.audit_hmr2a_crop_protocol_v11 import RawBBoxHMR3DPWDataset
from divibody_training.cache_hmr2a_rgb_anchor_v8 import (
    FORMAT,
    HMR2A_SHA256,
    RGB_TOKENS,
    RGB_WIDTH,
    build_anchor_plan,
    semantic_vace_digest,
    sha256_file,
    validate_sidecar,
    validate_vace_source,
)
from divibody_training.hmr2a_anchor import HMR2AAnchorExtractor


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vace-cache",type=Path,required=True)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--hmr-source-root",type=Path,required=True)
    p.add_argument("--hmr2a-checkpoint",type=Path,required=True)
    p.add_argument("--hmr-model-config",type=Path,required=True)
    p.add_argument("--hmr-data-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--margin",type=float,default=0.25)
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--num-workers",type=int,default=8)
    p.add_argument("--device",default="cuda")
    p.add_argument("--skip-vace-sha256",action="store_true")
    return p.parse_args()


class Images(Dataset):
    def __init__(self,dataset,indices):self.dataset=dataset;self.indices=indices
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):return self.dataset[self.indices[index]]["image"]


def frame_uid(sample):
    return f"{sample.sequence}/person_{sample.person_index}/frame_{sample.frame_index:05d}"


def main():
    args=parse_args()
    if not 0 <= args.margin <= 1:raise ValueError("margin must be in [0,1]")
    vace_path=args.vace_cache.resolve()
    vace=torch.load(vace_path,map_location="cpu",weights_only=False,mmap=True)
    validate_vace_source(vace)
    split=vace["signature"]["split"]
    plan=build_anchor_plan(vace["metadata"],split,0)
    dataset=RawBBoxHMR3DPWDataset(root=args.data_root,split=split,image_size=256,
        frame_stride=1,max_samples=None,crop_to_person=True,crop_margin=args.margin)
    lookup={frame_uid(sample):i for i,sample in enumerate(dataset.samples)}
    missing=[uid for uid in plan.frame_uids if uid not in lookup]
    if missing:raise ValueError(f"Missing strict raw-bbox identities: {missing[:5]}")
    loader=DataLoader(Images(dataset,[lookup[x] for x in plan.frame_uids]),
        batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,
        pin_memory=True,persistent_workers=args.num_workers>0)
    count=len(plan.frame_uids)
    rgb=torch.empty(count,RGB_TOKENS,RGB_WIDTH,dtype=torch.bfloat16)
    rot=torch.empty(count,24,3,3);betas=torch.empty(count,10);camera=torch.empty(count,3)
    offset=0
    with HMR2AAnchorExtractor(source_root=args.hmr_source_root,
            checkpoint=args.hmr2a_checkpoint,model_config=args.hmr_model_config,
            hmr2_data_root=args.hmr_data_root,device=torch.device(args.device)) as extractor:
        for images in loader:
            out=extractor.encode(images);n=len(images);s=slice(offset,offset+n)
            rgb[s].copy_(out["rgb_tokens"].to("cpu",torch.bfloat16))
            rot[s].copy_(out["base_rotmat"].float().cpu())
            betas[s].copy_(out["base_betas"].float().cpu())
            camera[s].copy_(out["base_camera"].float().cpu())
            offset+=n;print(f"raw-bbox HMR2a frames {offset}/{count}",flush=True)
    stat=vace_path.stat();windows=len(plan.window_uids)
    payload={
        "signature":{"format":FORMAT,"split":split,"development_only":False,
          "anchor_revision":"raw-bbox-v11","raw_bbox_margin":args.margin,
          "unique_frames":count,"windows":windows,
          "hmr2a_checkpoint":str(args.hmr2a_checkpoint.resolve()),
          "hmr2a_checkpoint_sha256":HMR2A_SHA256,"vace_cache":str(vace_path),
          "vace_cache_bytes":stat.st_size,
          "vace_cache_sha256":None if args.skip_vace_sha256 else sha256_file(vace_path),
          "vace_semantic_sha256":semantic_vace_digest(vace,windows),
          "crop_protocol":f"raw rectangular 3DPW poses2d bbox + {args.margin:.2f} per-side margin; single HMR2 aspect expansion; no pre-square or boundary clamp",
          "rgb_token_dtype":"bfloat16","anchor_dtype":"float32"},
        "frame_uids":plan.frame_uids,"window_uids":plan.window_uids,
        "window_frame_indices":plan.window_frame_indices,"rgb_tokens":rgb,
        "base_rotmat":rot,"base_betas":betas,"base_camera":camera}
    validate_sidecar(payload,vace)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp=args.output.with_suffix(args.output.suffix+".tmp");torch.save(payload,temp);os.replace(temp,args.output)
    print(json.dumps({"status":"HMR2A_RAW_BBOX_V11_OK","split":split,"frames":count,"windows":windows,"output":str(args.output)}))


if __name__=="__main__":main()
