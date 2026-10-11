"""Train-only supervision of predictions stitched by their unique frame owner.

Unlike the legacy metric-only boundary list, this pool includes every consecutive
triplet whose canonical frame owners span windows, including stride-4 overlap.
Labels are retained only in a separate target dictionary, never model inputs.
The sampler is a deterministic cyclic permutation of the complete audited pool.
This module intentionally contains no optimizer, training loop, or server calls.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable


Location = tuple[int, int]
FrameKey = tuple[str, int, int]
TripletKey = tuple[str, int, int, int, int]
INPUT_KEYS = ("rgb_tokens", "vace_features", "base_rotmat", "base_betas", "base_camera")
TARGET_KEYS = ("full_pose", "betas", "gender_code", "cam_pose")


@dataclass(frozen=True)
class BoundaryTriplet:
    key: TripletKey
    frame_uids: tuple[str, str, str]
    locations: tuple[Location, Location, Location]
    also_has_within_window_owner: bool


@dataclass(frozen=True)
class OverlapFrame:
    frame_uid: str
    owner: Location
    other_contexts: tuple[Location, ...]


@dataclass(frozen=True)
class BoundaryPool:
    triplets: tuple[BoundaryTriplet, ...]
    overlap_frames: tuple[OverlapFrame, ...]
    audit: dict[str, Any]
    identity_sha256: str
    window_uids: tuple[str, ...]
    payload_object_id: int = 0


def _frame_uid(key: FrameKey) -> str:
    sequence, person, frame = key
    return f"{sequence}/person_{person}/frame_{frame:05d}"


def _audit_identities(metadata, valid_rows, owner_rows, acceleration_rows,
                      *, forbidden_sequences: Iterable[str] = ()) -> BoundaryPool:
    """Pure identity audit, also usable in small tests without importing torch."""
    count = len(metadata)
    if not count or any(len(rows) != count for rows in (valid_rows, owner_rows, acceleration_rows)):
        raise ValueError("Empty or mismatched boundary metadata/masks")
    forbidden = set(forbidden_sequences)
    owners: dict[FrameKey, Location] = {}
    appearances: dict[FrameKey, list[Location]] = {}
    within: dict[TripletKey, Location] = {}
    windows = []
    identities = []
    for window, record in enumerate(metadata):
        sequence = str(record["sequence"])
        person = int(record["person_id"])
        if sequence in forbidden:
            raise ValueError("Train boundary sequence overlaps a forbidden split: " + sequence)
        if not sequence or "/" in sequence or person < 0:
            raise ValueError("Invalid train sequence/person identity")
        uid = str(record["window_uid"])
        if not uid or uid in windows:
            raise ValueError("Empty or duplicate training window UID")
        windows.append(uid)
        ids = record["target_frame_ids"]
        uids = record["target_frame_uids"]
        valid, owner, acceleration = valid_rows[window], owner_rows[window], acceleration_rows[window]
        if any(len(values) != 8 for values in (ids, uids, valid, owner)) or len(acceleration) != 6:
            raise ValueError("Boundary supervision requires eight-target V10 windows")
        if any(type(value) is not bool for values in (valid, owner, acceleration) for value in values):
            raise ValueError("Boundary masks must be Boolean")
        valid_count = sum(valid)
        if not valid_count or valid != [True] * valid_count + [False] * (8 - valid_count):
            raise ValueError("Target-valid mask must be a nonempty true prefix")
        if record.get("target_valid") != valid:
            raise ValueError("Metadata and tensor validity masks disagree")
        frame_ids = [int(value) for value in ids]
        if any(value < 0 for value in frame_ids):
            raise ValueError("Negative annotation frame identity")
        if any(frame_ids[i] != frame_ids[i - 1] + 1 for i in range(1, valid_count)):
            raise ValueError("Training targets cross an annotation-frame gap")
        if any(frame_ids[i] != frame_ids[valid_count - 1] for i in range(valid_count, 8)):
            raise ValueError("Invalid tail must repeat the last real target")
        keys = [(sequence, person, frame) for frame in frame_ids]
        if list(uids) != [_frame_uid(key) for key in keys]:
            raise ValueError("Frame UID does not match sequence/person/annotation frame")
        for local, (key, is_valid, is_owned) in enumerate(zip(keys, valid, owner)):
            if is_owned and not is_valid:
                raise ValueError("Invalid padded frame has an owner")
            if is_valid:
                appearances.setdefault(key, []).append((window, local))
            if is_owned:
                if key in owners:
                    raise ValueError("Duplicate training frame owner: " + _frame_uid(key))
                owners[key] = (window, local)
        for local, is_owned in enumerate(acceleration):
            if not is_owned:
                continue
            if not all(valid[local:local + 3]):
                raise ValueError("Acceleration owner includes padded targets")
            primitive = (sequence, person, *frame_ids[local:local + 3])
            if primitive in within:
                raise ValueError("Duplicate within-window acceleration primitive owner")
            within[primitive] = (window, local)
        identities.append(dict(window_uid=uid, frame_uids=list(uids), valid=valid,
                               frame_owner=owner, acceleration_owner=acceleration))
    if set(owners) != set(appearances):
        missing = sorted(set(appearances) - set(owners))
        raise ValueError("Real train frames have no unique owner: " + str(missing[:5]))
    complete = []
    for sequence, person, center in sorted(owners):
        keys = ((sequence, person, center - 1), (sequence, person, center), (sequence, person, center + 1))
        if all(key in owners for key in keys):
            primitive = (sequence, person, center - 1, center, center + 1)
            complete.append((primitive, keys))
    complete_keys = {primitive for primitive, _ in complete}
    if not set(within).issubset(complete_keys):
        raise ValueError("Within-window primitive is absent from complete owner frame identities")
    triplets = []
    for primitive, keys in complete:
        locations = tuple(owners[key] for key in keys)
        if len({window for window, _ in locations}) > 1:
            triplets.append(BoundaryTriplet(primitive, tuple(_frame_uid(key) for key in keys),
                                           locations, primitive in within))
    overlap = tuple(OverlapFrame(_frame_uid(key), owners[key],
                                 tuple(location for location in locations if location != owners[key]))
                    for key, locations in sorted(appearances.items()) if len(locations) > 1)
    identity_hash = hashlib.sha256(json.dumps(identities, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    audit = dict(train_only=True, windows=count, unique_owned_frames=len(owners),
                 sequences=sorted({key[0] for key in owners}),
                 full_contiguous_triplets=len(complete),
                 within_window_owned_triplets=len(within),
                 boundary_triplets=len(triplets),
                 boundary_also_within_window=sum(record.also_has_within_window_owner for record in triplets),
                 missing_from_any_within_window=len(complete_keys - set(within)),
                 two_window_triplets=sum(len({w for w, _ in record.locations}) == 2 for record in triplets),
                 three_window_triplets=sum(len({w for w, _ in record.locations}) == 3 for record in triplets),
                 overlap_frames=len(overlap), overlap_other_contexts=sum(len(row.other_contexts) for row in overlap),
                 time_unit="annotation frame; identical to sequence_acceleration evaluator",
                 motion_root="J14 left/right hip midpoint", loss_unit="m per annotation frame squared")
    return BoundaryPool(tuple(triplets), overlap, audit, identity_hash, tuple(windows))


def build_boundary_pool(dataset, *, forbidden_sequences: Iterable[str] = ()) -> BoundaryPool:
    """Audit a complete original V10Dataset, its train labels, and sidecar UIDs.

    Existing load_pair should run first to audit feature/sidecar signatures and
    encoder provenance. Subsets are rejected because they can leave real frames
    without their unique owner. Pass the complete train dataset for this branch.
    """
    import torch
    from divibody_training.ram_parts_runtime_v29 import V10Dataset
    if not isinstance(dataset, V10Dataset):
        raise TypeError("Boundary branch requires the complete original V10Dataset, not a Subset")
    vace, sidecar = dataset.vace, dataset.sidecar
    if vace.get("signature", {}).get("split") != "train" or sidecar.get("signature", {}).get("split") != "train":
        raise ValueError("Boundary optimization accepts explicitly declared train caches only")
    if vace["signature"].get("feature_only") is not False or "targets" not in vace:
        raise ValueError("Boundary optimization requires actual train labels")
    count = len(vace["metadata"])
    for name, columns in (("target_valid_mask", 8), ("frame_owner_mask", 8), ("acceleration_owner_mask", 6)):
        value = vace[name]
        if not torch.is_tensor(value) or value.dtype != torch.bool or tuple(value.shape) != (count, columns):
            raise ValueError("Invalid Boolean train mask: " + name)
    shapes = dict(full_pose=(count, 8, 72), betas=(count, 8, 10), gender_code=(count, 8), cam_pose=(count, 8, 4, 4))
    for name, shape in shapes.items():
        value = vace["targets"][name]
        if not torch.is_tensor(value) or tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
            raise ValueError("Invalid or nonfinite train target: " + name)
        if value.requires_grad:
            raise ValueError("Train labels must not require gradients")
    genders = vace["targets"]["gender_code"]
    real_frames = vace["target_valid_mask"]
    official_gender = (genders == 0) | (genders == 1)
    padding_gender = official_gender | (genders == 2)
    if not bool(torch.where(real_frames, official_gender, padding_gender).all()):
        raise ValueError("Real train labels require male/female codes; padding may also use neutral=2")
    pool = _audit_identities(vace["metadata"], vace["target_valid_mask"].tolist(),
                            vace["frame_owner_mask"].tolist(), vace["acceleration_owner_mask"].tolist(),
                            forbidden_sequences=forbidden_sequences)
    lookup = sidecar["window_frame_indices"]
    frame_uids = sidecar["frame_uids"]
    if tuple(lookup.shape) != (count, 8) or lookup.dtype != torch.long:
        raise ValueError("Invalid sidecar train frame lookup")
    if len(frame_uids) != len(set(frame_uids)):
        raise ValueError("Duplicate sidecar frame UID")
    sidecar_window_uids = sidecar.get("window_uids")
    if (not isinstance(sidecar_window_uids, list) or len(sidecar_window_uids) != count
            or any(not isinstance(uid, str) for uid in sidecar_window_uids)
            or sidecar_window_uids != list(pool.window_uids)):
        raise ValueError("Sidecar window UID order differs from VACE")
    previous: dict[str, Location] = {}
    duplicate_target_checks = 0
    for window, record in enumerate(vace["metadata"]):
        for local, uid in enumerate(record["target_frame_uids"]):
            index = int(lookup[window, local])
            if not 0 <= index < len(frame_uids) or frame_uids[index] != uid:
                raise ValueError("Sidecar lookup references a different person/frame UID")
            if not bool(vace["target_valid_mask"][window, local]):
                continue
            if uid in previous:
                old_window, old_local = previous[uid]
                for name in TARGET_KEYS:
                    if not torch.equal(vace["targets"][name][window, local], vace["targets"][name][old_window, old_local]):
                        raise ValueError("Overlapping train windows disagree about GT: " + uid + "/" + name)
                duplicate_target_checks += 1
            else:
                previous[uid] = (window, local)
    old_records = vace.get("metric_only_cross_window_motion", {}).get("acceleration_triplets", [])
    legacy_keys = []
    for record in old_records:
        frame_ids = tuple(int(value) for value in record["frame_ids"])
        if len(frame_ids) != 3 or frame_ids[1] != frame_ids[0] + 1 or frame_ids[2] != frame_ids[1] + 1:
            raise ValueError("Invalid legacy boundary acceleration identity")
        legacy_keys.append((str(record["sequence"]), int(record["person_id"]), *frame_ids))
    if len(legacy_keys) != len(set(legacy_keys)):
        raise ValueError("Duplicate legacy metric-only boundary triplet")
    missing = {row.key for row in pool.triplets if not row.also_has_within_window_owner}
    if set(legacy_keys) != missing:
        raise ValueError("Legacy metric-only list differs from audited missing within-window primitives")
    audit = pool.audit | dict(overlap_gt_identity_checks=duplicate_target_checks,
                              legacy_metric_only_triplets=len(legacy_keys),
                              labels_never_in_model_inputs=True)
    return BoundaryPool(pool.triplets, pool.overlap_frames, audit, pool.identity_sha256,
                        pool.window_uids, id(vace))


def _sample_indices(pool_size: int, *, step: int, seed: int, count: int, identity_sha256: str, task: str) -> list[int]:
    """State-free complete-pool cyclic permutation; every cycle visits each item."""
    if step < 0 or count <= 0 or pool_size <= 0 or count > pool_size:
        raise ValueError("Invalid deterministic boundary/overlap sampling budget")
    key = f"v29:{task}:{seed}:{identity_sha256}".encode()
    digest = hashlib.sha256(key).digest()
    offset = int.from_bytes(digest[:8], "big") % pool_size
    multiplier = int.from_bytes(digest[8:16], "big") % pool_size or 1
    while math.gcd(multiplier, pool_size) != 1:
        multiplier = (multiplier + 1) % pool_size or 1
    return [(offset + multiplier * (step * count + index)) % pool_size for index in range(count)]


def _collate_windows(dataset, pool: BoundaryPool, window_indices: list[int]):
    from torch.utils.data import default_collate
    if id(dataset.vace) != pool.payload_object_id:
        raise ValueError("Boundary pool belongs to a different train payload")
    raw = default_collate([dataset[index] for index in window_indices])
    inputs = {key: raw[key] for key in INPUT_KEYS}
    inputs["target_valid"] = raw["target_valid"]
    for key in ("landmark_schema", "source_frame_ids"):
        if key in raw:
            inputs[key] = raw[key]
    targets = {key: raw[key] for key in TARGET_KEYS}
    return inputs, targets


def _predicted_joints(model, inputs, models, regressor, device, amp_dtype):
    import torch
    from divibody_training.ram_parts_runtime_v29 import forward_batch
    from divibody_training.train_divibody_continuous_geometry_v5 import smpl_vertices
    from divibody_training.geometry_metrics_3dpw import pelvis_aligned_j14, hip_centered_j14
    prediction = forward_batch(model, inputs, device, amp_dtype)
    if not torch.is_grad_enabled():
        raise RuntimeError("Boundary training loss must be computed with gradients enabled")
    with torch.autocast(device_type=device.type, enabled=False):
        rotations, betas = prediction["rotmat"].float(), prediction["betas"].float()
        b, t = rotations.shape[:2]
        if tuple(rotations.shape) != (b, 8, 24, 3, 3) or tuple(betas.shape) != (b, 8, 10):
            raise ValueError("Boundary branch requires V29 per-frame SMPL predictions")
        vertices = smpl_vertices(models[2], rotations.reshape(-1, 24, 3, 3), betas.reshape(-1, 10))
        joints, _ = pelvis_aligned_j14(vertices, regressor)
        return hip_centered_j14(joints).reshape(b, t, 14, 3)


def boundary_training_loss(model, dataset, pool: BoundaryPool, models, regressor,
                           device, amp_dtype, *, step: int, seed: int,
                           triplets_per_step: int = 4):
    """Return (unweighted scalar loss, sampled identities) for optimizer.backward.

    Each sampled primitive contributes once with equal weight. Its three points
    come only from the unique valid frame owner, even in overlapping windows.
    The loss is ||d2(prediction - train_GT)||, not ||d2(prediction)||, so legitimate
    human acceleration is retained. No test/validation labels are accepted.
    """
    import torch
    from divibody_training.train_divibody_continuous_geometry_v5 import camera_targets, gendered_vertices
    from divibody_training.geometry_metrics_3dpw import pelvis_aligned_j14, hip_centered_j14
    selected = _sample_indices(len(pool.triplets), step=step, seed=seed,
                               count=triplets_per_step, identity_sha256=pool.identity_sha256, task="boundary")
    records = [pool.triplets[index] for index in selected]
    windows = sorted({window for record in records for window, _ in record.locations})
    remap = {window: row for row, window in enumerate(windows)}
    inputs, targets = _collate_windows(dataset, pool, windows)
    predicted = _predicted_joints(model, inputs, models, regressor, device, amp_dtype)
    with torch.autocast(device_type=device.type, enabled=False):
        with torch.no_grad():
            # Padding can contain neutral gender=2. Compute official gendered
            # GT only for real frames; no sampled owner ever references padding.
            real_frames = inputs["target_valid"].to(device).reshape(-1)
            rotations = camera_targets(targets, device).reshape(-1, 24, 3, 3)
            betas = targets["betas"].to(device=device, dtype=torch.float32).reshape(-1, 10)
            genders = targets["gender_code"].to(device).reshape(-1)
            gt_vertices = gendered_vertices(models, rotations[real_frames], betas[real_frames], genders[real_frames])
            gt_joints, _ = pelvis_aligned_j14(gt_vertices, regressor)
            truth = gt_joints.new_zeros((len(windows) * 8, 14, 3))
            truth[real_frames] = hip_centered_j14(gt_joints)
            truth = truth.reshape(len(windows), 8, 14, 3)
        error = predicted - truth
        points = torch.stack([torch.stack([error[remap[window], local] for window, local in row.locations])
                              for row in records])
        second_difference = points[:, 0] - 2 * points[:, 1] + points[:, 2]
        loss = torch.linalg.vector_norm(second_difference, dim=-1).mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite train boundary acceleration loss")
    diagnostics = dict(task="boundary_acceleration", step=step, seed=seed,
                       pool_identity_sha256=pool.identity_sha256, pool_size=len(pool.triplets),
                       sampled_triplet_indices=selected, sampled_triplet_keys=[list(row.key) for row in records],
                       sampled_frame_uids=[list(row.frame_uids) for row in records],
                       owner_locations=[[list(location) for location in row.locations] for row in records],
                       windows=windows, window_uids=[pool.window_uids[window] for window in windows],
                       triplets=len(records), equal_primitive_weight=True, draw_offset=step * triplets_per_step,
                       raw_loss_m_per_frame2=float(loss.detach()), raw_loss_mm_per_frame2=float(loss.detach()) * 1000,
                       supervised_split="train", labels_never_in_model_inputs=True)
    return loss, diagnostics


def overlap_consistency_loss(model, dataset, pool: BoundaryPool, models, regressor,
                             device, amp_dtype, *, step: int, seed: int,
                             frames_per_step: int = 4):
    """Optional symmetric 3D consistency across contexts, one weight per UID.

    This is separate from boundary acceleration and GT supervision. All context
    outputs receive gradients; the owner frame is not a frozen pseudo label.
    Each UID's comparisons are averaged before averaging UIDs, so a frame with
    more overlapping contexts does not receive more total weight.
    """
    import torch
    selected = _sample_indices(len(pool.overlap_frames), step=step, seed=seed,
                               count=frames_per_step, identity_sha256=pool.identity_sha256, task="overlap")
    records = [pool.overlap_frames[index] for index in selected]
    windows = sorted({window for record in records for window, _ in (record.owner, *record.other_contexts)})
    remap = {window: row for row, window in enumerate(windows)}
    inputs, _ = _collate_windows(dataset, pool, windows)
    predicted = _predicted_joints(model, inputs, models, regressor, device, amp_dtype)
    with torch.autocast(device_type=device.type, enabled=False):
        values = []
        for record in records:
            window, local = record.owner
            owner = predicted[remap[window], local]
            values.append(torch.stack([torch.linalg.vector_norm(owner - predicted[remap[w], col], dim=-1).mean()
                                       for w, col in record.other_contexts]).mean())
        loss = torch.stack(values).mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite train overlap consistency loss")
    return loss, dict(task="overlap_consistency", step=step, seed=seed,
                      pool_identity_sha256=pool.identity_sha256, pool_size=len(pool.overlap_frames),
                      sampled_frame_indices=selected, sampled_frame_uids=[row.frame_uid for row in records],
                      owner_locations=[list(row.owner) for row in records],
                      other_context_locations=[[list(value) for value in row.other_contexts] for row in records],
                      windows=windows, window_uids=[pool.window_uids[window] for window in windows],
                      equal_frame_weight=True, symmetric_gradients=True,
                      raw_loss_m=float(loss.detach()), supervised_split="train",
                      labels_never_in_model_inputs=True)
