"""Deterministic cache streams and auditable optimization diagnostics."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import torch
from torch.utils.data import default_collate


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def official_annotation_members(names, split):
    """Ignore macOS resource forks; these are not annotation pickle files."""
    return [name for name in names if name.endswith('.pkl') and split in name.split('/')
            and '__MACOSX' not in name.split('/') and not Path(name).name.startswith('._')]


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


class MapStream:
    """No prefetch: cursors refer to batches actually returned to training."""
    def __init__(self, dataset, batch_size, seed):
        if len(dataset) < 1 or batch_size < 1:
            raise ValueError('Empty stream or invalid batch size')
        self.dataset, self.batch_size, self.seed = dataset, batch_size, seed
        self.cycle = self.cursor = self.draws = self.sampled = 0
        self.seen = set()
        self.type_draws = {'clean': 0, 'occluded': 0, 'other': 0}
        self.order = self._order()

    def _order(self):
        return torch.randperm(len(self.dataset), generator=torch.Generator().manual_seed(
            self.seed + self.cycle)).tolist()

    def next_batch(self):
        if self.cursor == len(self.order):
            self.cycle += 1
            self.cursor = 0
            self.order = self._order()
        ids = self.order[self.cursor:self.cursor + self.batch_size]
        batch = default_collate([self.dataset[i] for i in ids])
        self.cursor += len(ids)
        self.sampled += len(ids)
        return batch, ids

    def record(self, ids):
        self.draws += len(ids)
        self.seen.update(ids)
        files = getattr(self.dataset, 'files', None)
        for i in ids:
            name = str(files[i]) if files else ''
            kind = ('occluded' if name.endswith('_real_occlusion.pt') else
                    'clean' if name.endswith('_clean.pt') else 'other')
            self.type_draws[kind] += 1

    def coverage(self):
        return dict(pool=len(self.dataset), optimized_draws=self.draws,
                    sampled_draws=self.sampled, unique_optimized=len(self.seen),
                    fraction=len(self.seen) / len(self.dataset), cycle=self.cycle,
                    type_draws=self.type_draws.copy())

    def state_dict(self):
        return dict(seed=self.seed, pool=len(self.dataset), batch_size=self.batch_size,
                    cycle=self.cycle, cursor=self.cursor, draws=self.draws,
                    sampled=self.sampled, seen=sorted(self.seen), type_draws=self.type_draws)

    def load_state_dict(self, state):
        if (state['pool'], state['seed'], state['batch_size']) != (
                len(self.dataset), self.seed, self.batch_size):
            raise ValueError('Stream dataset, seed or batch size changed on resume')
        for key in ('cycle', 'cursor', 'draws', 'sampled'):
            setattr(self, key, state[key])
        self.seen, self.type_draws = set(state['seen']), dict(state['type_draws'])
        self.order = self._order()
        if not 0 <= self.cursor <= len(self.order):
            raise ValueError('Invalid stream cursor')


class PoseStream:
    def __init__(self, loader):
        if loader.num_workers != 0 or loader.drop_last:
            raise ValueError('Pose stream requires zero workers and no dropped batches')
        self.loader = loader
        self.cycle = self.cursor = self.draws = 0
        self.seen = set()
        self.loader.dataset.set_cycle(0)
        self.iterator = iter(loader)

    def next_batch(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.cycle += 1
            self.cursor = 0
            self.loader.dataset.set_cycle(self.cycle)
            self.iterator = iter(self.loader)
            batch = next(self.iterator)
        self.cursor += 1
        return batch

    def record(self, batch):
        self.seen.update(batch['window_uid'])
        self.draws += len(batch['window_uid'])

    def coverage(self):
        return dict(pool=len(self.loader.dataset), unique_optimized=len(self.seen),
                    optimized_draws=self.draws, fraction=len(self.seen) / len(self.loader.dataset),
                    cycle=self.cycle)

    def state_dict(self):
        return dict(cycle=self.cycle, cursor=self.cursor, draws=self.draws,
                    seen=sorted(self.seen), pool=len(self.loader.dataset))

    def load_state_dict(self, state):
        if state['pool'] != len(self.loader.dataset):
            raise ValueError('Pose pool changed on resume')
        self.cycle, self.cursor, self.draws = state['cycle'], state['cursor'], state['draws']
        self.seen = set(state['seen'])
        self.loader.dataset.set_cycle(self.cycle)
        self.iterator = iter(self.loader)
        for _ in range(self.cursor):
            next(self.iterator)


def gradients(optimizer):
    return {group['group_name']: torch.cat([
        (p.grad.detach().float().reshape(-1).cpu() if p.grad is not None
         else torch.zeros(p.numel())) for p in group['params']])
        for group in optimizer.param_groups}


def subtract(left, right):
    return {k: left[k] - right[k] for k in left}


def gradient_report(tasks):
    result = {}
    for group in next(iter(tasks.values())):
        vectors = {name: value[group] for name, value in tasks.items()}
        norms = {name: float(v.norm()) for name, v in vectors.items()}
        cosines = {}
        for a, b in itertools.combinations(vectors, 2):
            denom = norms[a] * norms[b]
            cosines[a + ':' + b] = max(-1., min(1., float(vectors[a].dot(vectors[b])) / denom)) if denom else None
        result[group] = dict(weighted_norms=norms, cosines=cosines)
    return result


def selection_report(geometry, pose, baseline_geometry, baseline_pose, args, best):
    reasons = []
    if not all(math.isfinite(v) for v in [*geometry.values(), *pose.values()]):
        raise FloatingPointError('Nonfinite validation metrics')
    if geometry['pve_mm'] > baseline_geometry['pve_mm'] + args.pve_tolerance_mm:
        reasons.append('pve_above_tolerance')
    for key in ('projected_PCKn@0.05', 'projected_PCKn@0.10'):
        if pose[key] < baseline_pose[key] - args.pose_pck_tolerance:
            reasons.append(key + '_below_tolerance')
    score = pose['projected_PCKn@0.05'] + pose['projected_PCKn@0.10'] - pose['projected_mean_normalized_joint_error']
    accepted = not reasons and score > best
    if not reasons and not accepted:
        reasons.append('score_not_improved')
    return dict(accepted=accepted, reasons=reasons, score=score,
                geometry_delta={k: geometry[k] - baseline_geometry[k] for k in geometry},
                pve_limit_mm=baseline_geometry['pve_mm'] + args.pve_tolerance_mm)


def validate_coverage(train, val, manifest_path=None, allow_partial=False):
    train_names = {m['sequence'] for m in train['metadata']}
    val_names = {m['sequence'] for m in val['metadata']}
    if not train_names or not val_names or train_names & val_names:
        raise ValueError('Empty or overlapping train/validation sequences')
    owners, valid = val['frame_owner_mask'], val['target_valid_mask']
    if owners.dtype != torch.bool or valid.dtype != torch.bool or owners.shape != valid.shape or owners.ndim != 2 or len(owners) != len(val['metadata']) or (owners & ~valid).any():
        raise ValueError('Invalid validation ownership')
    for record in val['metadata']:
        ids = record['target_frame_uids']
        if len(ids) != owners.shape[1] or any(not isinstance(uid,str) or not uid.startswith(record['sequence'] + '/') for uid in ids):
            raise ValueError('Frame identities must match sequence and mask width')
    uids = [uid for row, record in enumerate(val['metadata'])
            for uid, owned in zip(record['target_frame_uids'], owners[row].tolist()) if owned]
    if not uids or len(set(uids)) != len(uids):
        raise ValueError('Validation person-frame ownership is duplicated or empty')
    result = dict(complete=False, sequence_names=sorted(val_names), owned_person_frames=len(uids),
                  provenance='unverified', missing_sequences=None, missing_frames=None,
                  manifest_sha256=None)
    if manifest_path:
        reference = json.loads(Path(manifest_path).read_text())
        if reference.get('format') != 'v28d-validation-reference-v1':
            raise ValueError('Use the independently audited V28D validation reference')
        expected = set(reference['validation_frame_uids'])
        if not expected or len(expected) != len(reference['validation_frame_uids']):
            raise ValueError('Empty or duplicated reference frames')
        if set(uids) - expected:
            raise ValueError('Cache contains frames outside the reference')
        ref_names = set(reference['validation'])
        expected_train = set(reference['train'])
        if train_names - expected_train:
            raise ValueError('Train cache contains sequences outside the reference')
        missing_train = sorted(expected_train - train_names)
        if missing_train and not allow_partial:
            raise ValueError('Train cache sequences differ from reference')
        independent = reference.get('independent_archive_verified', False)
        if independent and sha256(reference['archive_path']) != reference['archive_sha256']:
            raise ValueError('Independent archive fingerprint changed')
        result.update(complete=independent and not missing_train and set(uids) == expected and val_names == ref_names,
                      train_reference_complete=not missing_train, missing_train_sequences=missing_train,
                      provenance=reference['source'], missing_sequences=sorted(ref_names - val_names),
                      missing_frames=len(expected - set(uids)), manifest_sha256=sha256(manifest_path))
    if not result['complete'] and not allow_partial:
        raise ValueError('Independent validation reference or complete cache missing; explicitly opt into development validation')
    return result
