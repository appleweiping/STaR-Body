"""Validation-gated small run from corrected raw-bbox HMR2a anchors."""

import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from divibody_training.evaluate_standard_3dpw import create_smpl_models
from divibody_training.train_divibody_rgb_vace_hmr2a_v8 import V8Dataset, load_pair, validate
from divibody_training.train_divibody_rgb_vace_hmr2a_v9_small import parse_args, spread_subset
from divibody_training.train_divibody_rgb_vace_v7 import forward_batch, geometric_losses
from models.human_decoder_rgb_vace_v8_hmr2a import HumanRGBVaceGeometryTransformerV8HMR2A


def main():
    args=parse_args()  # init-checkpoint remains a required provenance CLI field but is not loaded.
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise FileExistsError("v11 small output exists")
    torch.manual_seed(args.seed);np.random.seed(args.seed);device=torch.device(args.device)
    tv,th=load_pair(args.train_vace_cache,args.train_hmr_sidecar,"train")
    vv,vh=load_pair(args.val_vace_cache,args.val_hmr_sidecar,"validation")
    if th["signature"].get("anchor_revision")!="raw-bbox-v11" or vh["signature"].get("anchor_revision")!="raw-bbox-v11":raise ValueError("v11 requires corrected raw-bbox anchors")
    train=spread_subset(V8Dataset(tv,th),args.max_train_windows);val=spread_subset(V8Dataset(vv,vh),args.max_val_windows)
    model=HumanRGBVaceGeometryTransformerV8HMR2A().to(device)
    smpl=create_smpl_models(args.smpl_model_root,device)
    for x in smpl.values():x.requires_grad_(False)
    reg=torch.as_tensor(np.load(args.joint_regressor),dtype=torch.float32,device=device)
    opts={"num_workers":args.num_workers,"pin_memory":True,"persistent_workers":args.num_workers>0}
    tl=DataLoader(train,batch_size=args.batch_size,shuffle=True,**opts);vl=DataLoader(val,batch_size=args.val_batch_size,shuffle=False,**opts)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True);path=args.output_dir/"metrics.jsonl"
    def save(name,epoch,metric):
        tmp=args.output_dir/(name+".tmp");torch.save({"format":"divibody-v11-raw-small","epoch":epoch,"validation":metric,"model":model.state_dict(),"model_config":vars(model.config),"test_used":False,"small_sample":True,"args":vars(args)},tmp);os.replace(tmp,args.output_dir/name)
    baseline=validate(vl,model,smpl,reg,args,device);best=baseline["mpjpe_mm"];save("best.pt",0,baseline);path.write_text(json.dumps({"epoch":0,"validation":baseline,"exact_raw_anchor":True})+"\n");print(json.dumps({"epoch":0,"validation":baseline,"exact_raw_anchor":True}),flush=True)
    for epoch in range(1,args.epochs+1):
        model.train();totals={};n=0
        for batch in tl:
            opt.zero_grad(set_to_none=True);pred=forward_batch(model,batch,device,args.amp_dtype);loss=geometric_losses(pred,batch,smpl,reg,args,device);loss["total"].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip);opt.step()
            for k,v in loss.items():totals[k]=totals.get(k,0)+float(v.detach())
            n+=1
        metric=validate(vl,model,smpl,reg,args,device);rec={"epoch":epoch,"train":{k:v/n for k,v in totals.items()},"validation":metric,"temporal_gate":model.temporal_gates.detach().tanh().cpu().tolist()};open(path,"a").write(json.dumps(rec)+"\n");save("latest.pt",epoch,metric)
        if metric["mpjpe_mm"]<best:best=metric["mpjpe_mm"];save("best.pt",epoch,metric)
        print(json.dumps(rec),flush=True)


if __name__=="__main__":main()
