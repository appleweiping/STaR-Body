"""Fixed optimizer topology across phase transitions; only requires_grad changes."""
import torch
from divibody_training.target60_optim_v34 import ModelEMA,step_lr
def make_optimizer(model,args):
    buckets={};names=[]
    rates={'adapter':args.adapter_lr,'heads':args.v36_heads_lr,'late':args.v36_late_lr}
    for name,p in model.named_parameters():
        group=model.parameter_category(name)
        if group=='frozen' or (args.v36_arm=='R0' and group!='adapter'):continue
        decay=p.ndim>=2 and not name.endswith('bias')
        buckets.setdefault((group,decay),[]).append(p);names.append(name)
    groups=[dict(params=params,lr=rates[g],base_lr=rates[g],category=g,
        group_name=g+('_decay' if decay else '_no_decay'),weight_decay=args.weight_decay if decay else 0.)
        for (g,decay),params in sorted(buckets.items())]
    if not groups:raise ValueError('No V36 optimizer groups')
    return torch.optim.AdamW(groups),names
