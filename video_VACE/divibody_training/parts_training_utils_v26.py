"""Local-residual optimizer, warmup/cosine schedule and network EMA for v26."""
import copy
import math
import torch

def _category(name):
    if name.startswith('pose2d_branch.'):
        return 'local'
    raise RuntimeError(f'Unexpected trainable v26 parameter: {name}')

def _no_decay(name,parameter,category):
    leaf=name.rsplit('.',1)[-1]
    return category=='gate' or parameter.ndim<2 or leaf=='bias' or 'embedding.' in name or 'norm.' in name

def make_optimizer(model,args):
    learned=[(name,parameter) for name,parameter in model.named_parameters()
             if parameter.requires_grad]
    frozen=[name for name,parameter in model.named_parameters()
            if not parameter.requires_grad]
    if not learned or not frozen or any(not (name.startswith('reference.') or name.startswith('part_branch.')) for name in frozen):
        raise RuntimeError('v26 expects a frozen reference and a trainable local branch')
    rates={'local':args.local_lr}
    if any(not math.isfinite(value) or value<=0 for value in rates.values()):
        raise ValueError('The local learning rate must be finite and positive')
    buckets={}
    names=[]
    for name,parameter in learned:
        category=_category(name)
        decay=not _no_decay(name,parameter,category)
        buckets.setdefault((category,decay),[]).append(parameter)
        names.append(name)
    groups=[]
    for (category,decay),parameters in sorted(buckets.items()):
        lr=rates[category]
        groups.append(dict(params=parameters,lr=lr,base_lr=lr,category=category,
            group_name=f'{category}_{"decay" if decay else "no_decay"}',
            weight_decay=args.weight_decay if decay else 0.))
    if sum(len(group['params']) for group in groups)!=len(learned):
        raise RuntimeError('Optimizer parameter coverage is incomplete')
    return torch.optim.AdamW(groups),names

def step_lr(optimizer,step,total_steps,warmup_steps,min_ratio=.1):
    if total_steps<1 or not 0<=step<total_steps or warmup_steps<0 or not 0<min_ratio<=1:
        raise ValueError('Invalid learning-rate schedule arguments')
    if warmup_steps and step<warmup_steps:
        factor=(step+1)/warmup_steps
    else:
        span=max(1,total_steps-warmup_steps-1)
        progress=min(1.,max(0.,(step-warmup_steps)/span))
        factor=min_ratio+(1-min_ratio)*.5*(1+math.cos(math.pi*progress))
    result={}
    for group in optimizer.param_groups:
        group['lr']=group['base_lr']*factor
        result[group['group_name']]=group['lr']
    return result

class ModelEMA:
    def __init__(self,model,decay=.995):
        if not 0<=decay<1:
            raise ValueError('EMA decay must be in [0,1)')
        geometry=getattr(model,'_neutral_smpl',None)
        self.model=copy.deepcopy(model).eval()
        if geometry is not None:
            object.__setattr__(self.model,'_neutral_smpl',geometry)
        self.model.requires_grad_(False)
        self.decay=float(decay)
        self.updates=0

    @torch.no_grad()
    def update(self,model):
        self.updates+=1
        decay=min(self.decay,(1+self.updates)/(10+self.updates))
        source=model.state_dict()
        target=self.model.state_dict()
        if source.keys()!=target.keys():
            raise RuntimeError('EMA and student state structures differ')
        for name,value in target.items():
            incoming=source[name].detach()
            if value.is_floating_point():
                value.mul_(decay).add_(incoming,alpha=1-decay)
            else:
                value.copy_(incoming)
        return decay
