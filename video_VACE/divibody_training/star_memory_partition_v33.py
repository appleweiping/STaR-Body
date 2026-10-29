"""Partition complete train motion primitives by canonical frame-owner context.

The original runtime and cache masks remain untouched. The wrapper passes a
shallow batch copy with a new acceleration mask to the V31 runtime loss (V30-compatible by default),
including all of its intermediate geometry stages.
"""
from __future__ import annotations

import hashlib
import json
import math
from types import SimpleNamespace
from typing import Any


def _key_hash(keys) -> str:
    return hashlib.sha256(json.dumps(sorted(keys), separators=(",", ":")).encode()).hexdigest()


def _partition_identity_sets(metadata, valid_rows, owner_rows, acceleration_rows,
                             boundary_triplets) -> dict[str, Any]:
    """Perform a set-and-location audit, rather than only comparing totals."""
    try:
        from divibody_training.star_memory_boundary_v33 import _audit_identities
    except ModuleNotFoundError:
        from v29_boundary_training import _audit_identities
    # Retain the original strict continuity, UID, padding and unique-owner audit.
    original_audit = _audit_identities(metadata, valid_rows, owner_rows, acceleration_rows)
    owners = {}
    original_within = {}
    filtered_within = {}
    for window, record in enumerate(metadata):
        sequence, person = str(record["sequence"]), int(record["person_id"])
        ids = [int(frame) for frame in record["target_frame_ids"]]
        for local, (valid, owned) in enumerate(zip(valid_rows[window], owner_rows[window])):
            if valid and owned:
                owners[(sequence, person, ids[local])] = (window, local)
        for local, acceleration_owned in enumerate(acceleration_rows[window]):
            if not acceleration_owned:
                continue
            key = (sequence, person, *ids[local:local + 3])
            original_within[key] = (window, local)
            if all(valid_rows[window][local:local + 3]) and all(owner_rows[window][local:local + 3]):
                filtered_within[key] = (window, local)
    complete = set()
    expected_within = {}
    expected_boundary = {}
    for sequence, person, center in sorted(owners):
        frame_keys = ((sequence, person, center - 1), (sequence, person, center), (sequence, person, center + 1))
        if not all(key in owners for key in frame_keys):
            continue
        primitive = (sequence, person, center - 1, center, center + 1)
        complete.add(primitive)
        locations = tuple(owners[key] for key in frame_keys)
        if len({window for window, _ in locations}) > 1:
            expected_boundary[primitive] = locations
        else:
            window = locations[0][0]
            local = locations[0][1]
            if locations != ((window, local), (window, local + 1), (window, local + 2)):
                raise ValueError("Canonical same-window primitive has nonconsecutive local positions")
            if not acceleration_rows[window][local]:
                raise ValueError("Canonical same-window primitive lacks its original acceleration owner: " + str(primitive))
            expected_within[primitive] = (window, local)
    actual_boundary = {}
    for record in boundary_triplets:
        key = tuple(record.key)
        if key in actual_boundary:
            raise ValueError("Duplicate canonical boundary primitive in pool")
        locations = tuple(tuple(location) for location in record.locations)
        if key not in expected_boundary or locations != expected_boundary[key]:
            raise ValueError("Boundary pool primitive or owner locations differ from canonical train frame owners")
        sequence, person, first, middle, last = key
        expected_uids = tuple(f"{sequence}/person_{person}/frame_{frame:05d}" for frame in (first, middle, last))
        if tuple(record.frame_uids) != expected_uids:
            raise ValueError("Boundary record frame UIDs disagree with primitive identity")
        if bool(record.also_has_within_window_owner) != (key in original_within):
            raise ValueError("Boundary pool within-window provenance flag differs from original mask")
        actual_boundary[key] = locations
    if actual_boundary != expected_boundary:
        missing = sorted(set(expected_boundary) - set(actual_boundary))
        raise ValueError("Canonical boundary pool is incomplete: " + str(missing[:5]))
    if filtered_within != expected_within:
        raise ValueError("Filtered within primitives or owner locations differ from canonical same-window primitives")
    within_keys, boundary_keys = set(filtered_within), set(actual_boundary)
    if within_keys & boundary_keys or within_keys | boundary_keys != complete:
        raise ValueError("Canonical within/boundary sets do not exactly partition full contiguous train motion")
    removed = set(original_within) - within_keys
    boundary_also_within = boundary_keys & set(original_within)
    if removed != boundary_also_within:
        raise ValueError("Removed original within primitives differ from already-covered canonical boundaries")
    legacy_missing = complete - set(original_within)
    if legacy_missing != boundary_keys - set(original_within):
        raise ValueError("Legacy missing primitives are not exactly the previously uncovered boundary subset")
    full_count = len(complete)
    if not full_count:
        raise ValueError("Training partition has no complete contiguous motion primitives")
    return dict(train_only=True, exact_set_partition=True, exact_owner_locations=True,
                all_same_window_original_owners_verified=True,
                windows=len(metadata), owned_frames=len(owners),
                full_contiguous_triplets=full_count,
                original_within_triplets=len(original_within),
                canonical_within_triplets=len(within_keys),
                canonical_boundary_triplets=len(boundary_keys),
                original_within_removed=len(removed),
                boundary_also_within_window=len(boundary_also_within),
                legacy_missing_triplets=len(legacy_missing),
                within_fraction=len(within_keys) / full_count,
                boundary_fraction=len(boundary_keys) / full_count,
                complete_keys_sha256=_key_hash(complete),
                within_keys_sha256=_key_hash(within_keys),
                boundary_keys_sha256=_key_hash(boundary_keys),
                pool_identity_sha256=original_audit.identity_sha256,
                loss_normalization="within uniform-window sampling estimate, scaled by n*W/(B*N); boundary uniform sampled-primitive estimate",
                normalization_required=True, normalization_applies_to_intermediate_stages=True,
                single_minibatch_is_exact_full_pool_mean=False,
                combined_estimator="class fractions combine estimates of canonical within and boundary means")


def audit_boundary_partition(dataset, pool) -> dict[str, Any]:
    """Require an existing audited train pool and verify the exact partition."""
    vace, sidecar = dataset.vace, dataset.sidecar
    if vace.get("signature", {}).get("split") != "train" or sidecar.get("signature", {}).get("split") != "train":
        raise ValueError("Boundary partition accepts train caches only")
    if id(vace) != pool.payload_object_id or not pool.audit.get("train_only"):
        raise ValueError("Boundary partition pool is not the audited pool for this train payload")
    audit = _partition_identity_sets(vace["metadata"], vace["target_valid_mask"].tolist(),
                                     vace["frame_owner_mask"].tolist(),
                                     vace["acceleration_owner_mask"].tolist(), pool.triplets)
    if audit["pool_identity_sha256"] != pool.identity_sha256:
        raise ValueError("Train identities or owner masks changed after boundary pool construction")
    expected_counts = {
        "full_contiguous_triplets": pool.audit["full_contiguous_triplets"],
        "canonical_boundary_triplets": pool.audit["boundary_triplets"],
        "boundary_also_within_window": pool.audit["boundary_also_within_window"],
        "legacy_missing_triplets": pool.audit["missing_from_any_within_window"],
    }
    if any(audit[name] != count for name, count in expected_counts.items()):
        raise ValueError("Partition set audit disagrees with previously audited pool counts")
    return audit


def uniform_window_factor(train_windows: int, within_pool: int,
                          batch_windows: int, owned_primitives: int) -> float:
    """Convert an owned-primitive minibatch mean to a uniform-window estimate."""
    values = (train_windows, within_pool, batch_windows, owned_primitives)
    if (any(not isinstance(value, int) or isinstance(value, bool) for value in values)
            or train_windows < batch_windows or batch_windows <= 0
            or within_pool <= 0 or within_pool > 6 * train_windows
            or not 0 <= owned_primitives <= 6 * batch_windows):
        raise ValueError("Canonical motion normalization requires audited W/N and valid B/n counts")
    return owned_primitives * train_windows / (batch_windows * within_pool) if owned_primitives else 0.0


def geometric_losses(prediction, batch, models, regressor, args, device):
    """Run V31 geometry with only canonical same-window acceleration.

    All three frame_owner and target_valid entries must be true. The original
    cache masks, caller's batch and args are not mutated. For uniformly sampled
    train windows, scale the original minibatch owned-primitive mean by n*W/(B*N)
    to estimate the full canonical-within primitive mean. This is an estimate;
    a single minibatch does not contain the full pool. Require explicit W and N.
    """
    import torch
    from divibody_training.star_memory_runtime_v33 import geometric_losses as original_v31_geometry
    owner, valid, acceleration = batch["frame_owner"], batch["target_valid"], batch["acceleration_owner"]
    if (owner.dtype != torch.bool or valid.dtype != torch.bool or acceleration.dtype != torch.bool
            or owner.shape != valid.shape or owner.ndim != 2 or owner.shape[1] != 8
            or acceleration.shape != (owner.shape[0], 6)):
        raise ValueError("Invalid V31 Boolean motion ownership masks")
    owned = owner & valid
    canonical = owned[:, :-2] & owned[:, 1:-1] & owned[:, 2:]
    if bool((canonical & ~acceleration).any()):
        raise ValueError("Canonical same-window triplet is missing its original acceleration_owner")
    filtered = acceleration & canonical
    train_windows = getattr(args, "motion_train_windows", None)
    within_pool = getattr(args, "motion_canonical_within_triplets", None)
    batch_windows, count = owner.shape[0], int(filtered.sum())
    factor = uniform_window_factor(train_windows, within_pool, batch_windows, count)
    normalized_args = SimpleNamespace(**vars(args))
    normalized_args.acceleration_weight = args.acceleration_weight * factor
    adjusted_batch = batch | {"acceleration_owner": filtered}
    result = original_v31_geometry(prediction, adjusted_batch, models, regressor, normalized_args, device)
    result.update(canonical_within_acceleration_primitives=filtered.sum().float(),
                  original_acceleration_primitives=acceleration.sum().float(),
                  removed_noncanonical_within_primitives=(acceleration & ~canonical).sum().float(),
                  acceleration_uniform_window_factor=owner.new_tensor(factor, dtype=torch.float32),
                  acceleration_audited_train_windows=owner.new_tensor(train_windows, dtype=torch.float32),
                  acceleration_audited_within_pool=owner.new_tensor(within_pool, dtype=torch.float32),
                  acceleration_minibatch_windows=owner.new_tensor(batch_windows, dtype=torch.float32),
                  acceleration_base_weight=owner.new_tensor(args.acceleration_weight, dtype=torch.float32),
                  acceleration_effective_weight=owner.new_tensor(normalized_args.acceleration_weight, dtype=torch.float32))
    return result


def class_fraction_weights(total_weight: float, partition_audit: dict[str, Any]) -> dict[str, float]:
    """Combine class estimates after applying the mandatory runtime correction."""
    if not partition_audit.get("exact_set_partition") or not math.isfinite(total_weight) or total_weight < 0:
        raise ValueError("Weights require an exact audited partition and nonnegative coefficient")
    return dict(within=total_weight * partition_audit["within_fraction"],
                boundary=total_weight * partition_audit["boundary_fraction"])
