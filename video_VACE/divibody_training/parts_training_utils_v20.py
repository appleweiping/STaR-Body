"""Optimizer, input-coverage checks, and sampled optimization diagnostics."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import torch


def make_optimizer(model, args):
    names = []
    groups = {'head': [], 'features': [], 'fusion': [], 'gate': []}
    fusion_prefix = f'reference.vace_blocks.{len(model.reference.vace_blocks)-1}.'
    gate_name = 'reference.parametrizations.temporal_gates.0.last_gate'
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name == gate_name:
            group = 'gate'
        elif name.startswith(fusion_prefix):
            group = 'fusion'
        elif name.startswith('part_branch.heads.') or name == 'part_branch.bias':
            group = 'head'
        elif name.startswith('part_branch.'):
            group = 'features'
        else:
            raise RuntimeError(f'Unexpected trainable parameter: {name}')
        groups[group].append(parameter)
        names.append(name)
    if not names:
        raise ValueError('No trainable parameters')
    rates = dict(head=args.head_lr, features=args.geometry_lr,
                 fusion=args.fusion_lr, gate=args.gate_lr)
    if any(not math.isfinite(rate) or rate <= 0 for rate in rates.values()):
        raise ValueError('All group learning rates must be finite and positive')
    specs = [dict(params=params, lr=rates[name], group_name=name,
                  weight_decay=0. if name == 'gate' else args.weight_decay)
             for name, params in groups.items() if params]
    return torch.optim.AdamW(specs), names


def validate_cache_coverage(train_cache, val_cache, manifest_path=None, allow_partial=False):
    """Formal runs require reference split names AND independently audited frame UIDs."""
    train_names = {m['sequence'] for m in train_cache['metadata']}
    val_names = {m['sequence'] for m in val_cache['metadata']}
    if not train_names or not val_names or train_names & val_names:
        raise ValueError('Empty or overlapping train/validation sequences')
    owners = val_cache['frame_owner_mask']
    valid = val_cache['target_valid_mask']
    if owners.dtype != torch.bool or valid.dtype != torch.bool or owners.shape != valid.shape:
        raise ValueError('Invalid validation masks')
    if owners.ndim != 2 or len(owners) != len(val_cache['metadata']) or (owners & ~valid).any():
        raise ValueError('Owned validation frames must be valid and match metadata')
    uids = []
    for row, record in enumerate(val_cache['metadata']):
        frame_uids = record['target_frame_uids']
        if len(frame_uids) != owners.shape[1]:
            raise ValueError('Frame UID and ownership shape mismatch')
        for uid, owned in zip(frame_uids, owners[row].tolist()):
            if not isinstance(uid, str) or not uid.startswith(record['sequence'] + '/'):
                raise ValueError('Invalid person-frame identity')
            if owned:
                uids.append(uid)
    counts = Counter(uids)
    if not uids or any(count != 1 for count in counts.values()):
        raise ValueError('Validation ownership must count each person-frame exactly once')
    result = dict(sequence_names=sorted(val_names), owned_person_frames=len(uids),
                  complete=False, manifest_sha256=None, missing_sequences=[], missing_frames=None)
    if manifest_path is None:
        if not allow_partial:
            raise ValueError('Formal validation requires --validation-manifest; '
                             'use --allow-partial-validation for development only')
        return result
    raw = Path(manifest_path).read_bytes()
    manifest = json.loads(raw.decode('utf-8-sig'))
    if manifest.get('format') != '3dpw-validation-manifest-v20' or not manifest.get('source'):
        raise ValueError('Missing validation-manifest format or source provenance')
    splits = {}
    for split, expected_count in (('train', 24), ('validation', 12), ('test', 24)):
        values = manifest[split]
        if (not isinstance(values, list) or len(values) != expected_count
                or any(not isinstance(v, str) or not v for v in values)
                or len(set(values)) != expected_count):
            raise ValueError(f'Expected {expected_count} unique official {split} sequence names')
        splits[split] = set(values)
    if (splits['train'] & splits['validation'] or splits['train'] & splits['test']
            or splits['validation'] & splits['test']):
        raise ValueError('Manifest splits overlap')
    if train_names != splits['train'] or not val_names <= splits['validation']:
        raise ValueError('Cache sequence names differ from the reference split manifest')
    expected_list = manifest['validation_frame_uids']
    if not isinstance(expected_list, list) or not expected_list or len(set(expected_list)) != len(expected_list):
        raise ValueError('Invalid reference validation frame list')
    if any(not isinstance(uid, str) or '/' not in uid for uid in expected_list):
        raise ValueError('Invalid reference frame identity')
    if {uid.split('/')[0] for uid in expected_list} != splits['validation']:
        raise ValueError('Reference person-frames must cover all validation sequences')
    expected = set(expected_list)
    actual = set(uids)
    if actual - expected:
        raise ValueError('Cache contains person-frames absent from the reference manifest')
    missing = expected - actual
    result.update(complete=not missing and val_names == splits['validation'],
                  manifest_sha256=hashlib.sha256(raw).hexdigest(),
                  missing_sequences=sorted(splits['validation']-val_names), missing_frames=len(missing))
    if not result['complete'] and not allow_partial:
        raise ValueError(f'Incomplete validation: {result["missing_sequences"]}, '
                         f'{len(missing)} missing person-frames')
    return result


class StepProbe:
    """Sampled pre-clip gradients and actual post-AdamW parameter changes."""
    def __init__(self, optimizer):
        self.groups = {}
        self.values = {}
        for group in optimizer.param_groups:
            name = group['group_name']
            params = [p for p in group['params'] if p.requires_grad and p.grad is not None]
            self.groups[name] = [(p, p.detach().clone()) for p in params]
            grad_sq = sum(float(p.grad.detach().float().square().sum()) for p in params)
            norm_sq = sum(float(p.detach().float().square().sum()) for p in params)
            self.values[name] = dict(lr=group['lr'], active_tensors=len(params),
                                     gradient_norm=math.sqrt(grad_sq), parameter_norm=math.sqrt(norm_sq))

    def finish(self):
        for name, pairs in self.groups.items():
            delta = math.sqrt(sum(float((p.detach().float()-old.float()).square().sum()) for p, old in pairs))
            norm = self.values[name]['parameter_norm']
            self.values[name].update(update_norm=delta, relative_update=delta/norm if norm else None)
        return self.values


def loss_gradient_contributions(optimizer, supervised, teacher_weighted):
    """Compare primary 3DPW loss gradients before the ordinary backward pass."""
    entries = [(group['group_name'], p) for group in optimizer.param_groups
               for p in group['params'] if p.requires_grad]
    params = [p for _, p in entries]
    vectors = []
    for loss in (supervised, teacher_weighted):
        vectors.append(torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                       if loss.requires_grad else [None]*len(params))
    accum = {}
    for (name, _), left, right in zip(entries, *vectors):
        values = accum.setdefault(name, [0., 0., 0.])
        if left is not None:
            values[0] += float(left.detach().float().square().sum())
        if right is not None:
            values[1] += float(right.detach().float().square().sum())
        if left is not None and right is not None:
            values[2] += float((left.detach().float()*right.detach().float()).sum())
    return {name: dict(supervised_norm=math.sqrt(a), teacher_weighted_norm=math.sqrt(b),
                       cosine=max(-1., min(1., dot/math.sqrt(a*b))) if a*b > 0 else None)
            for name, (a,b,dot) in accum.items()}
