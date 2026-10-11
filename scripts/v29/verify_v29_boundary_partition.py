"""Pure exact-partition and normalization checks, with no model or training."""
from dataclasses import replace
import json
import math

try:
    from divibody_training.v29_boundary_training import _audit_identities
    from divibody_training.v29_boundary_partition import (
        _partition_identity_sets, class_fraction_weights, uniform_window_factor,
    )
except ModuleNotFoundError:
    from v29_boundary_training import _audit_identities
    from v29_boundary_partition import (
        _partition_identity_sets, class_fraction_weights, uniform_window_factor,
    )


def window(start, uid):
    ids = list(range(start, start + 8))
    return dict(sequence="train_sequence", person_id=0, window_uid=uid,
                target_frame_ids=ids,
                target_frame_uids=[f"train_sequence/person_0/frame_{frame:05d}" for frame in ids],
                target_valid=[True] * 8)


def rejects(function, phrase):
    try:
        function()
    except ValueError as exc:
        assert phrase in str(exc), str(exc)
    else:
        raise AssertionError("Invalid partition accepted: " + phrase)


def main():
    metadata = [window(0, "overlap0"), window(4, "overlap1")]
    valid = [[True] * 8] * 2
    owners = [[True] * 6 + [False] * 2, [False] * 2 + [True] * 6]
    acceleration = [[True] * 5 + [False], [False] + [True] * 5]
    pool = _audit_identities(metadata, valid, owners, acceleration)
    audit = _partition_identity_sets(metadata, valid, owners, acceleration, pool.triplets)
    assert audit["full_contiguous_triplets"] == 10
    assert audit["original_within_triplets"] == 10
    assert audit["canonical_within_triplets"] == 8
    assert audit["canonical_boundary_triplets"] == 2
    assert audit["original_within_removed"] == 2
    assert audit["legacy_missing_triplets"] == 0
    assert class_fraction_weights(2.0, audit) == dict(within=1.6, boundary=0.4)
    # Same boundary count does not prove equal keys/owner locations.
    wrong_location = [replace(pool.triplets[0], locations=((0, 4), (0, 5), (0, 6))), pool.triplets[1]]
    rejects(lambda: _partition_identity_sets(metadata, valid, owners, acceleration, wrong_location), "owner locations differ")
    duplicate = [pool.triplets[0], pool.triplets[0]]
    rejects(lambda: _partition_identity_sets(metadata, valid, owners, acceleration, duplicate), "Duplicate canonical boundary")
    wrong_uid = [replace(pool.triplets[0], frame_uids=("wrong", *pool.triplets[0].frame_uids[1:])), pool.triplets[1]]
    rejects(lambda: _partition_identity_sets(metadata, valid, owners, acceleration, wrong_uid), "frame UIDs disagree")
    rejects(lambda: _partition_identity_sets(metadata, valid, owners, acceleration, pool.triplets[:-1]), "incomplete")
    missing_original = [acceleration[0][:], acceleration[1][:]]
    missing_original[0][0] = False
    rejects(lambda: _partition_identity_sets(metadata, valid, owners, missing_original, pool.triplets), "lacks its original acceleration owner")
    # Uniform-window sampling needs count correction when window primitive
    # counts differ. Averaging both possible one-window estimates is unbiased.
    values = ([1.0, 2.0], [3.0, 4.0, 5.0, 6.0])
    estimates = [sum(row) / len(row) * uniform_window_factor(2, 6, 1, len(row)) for row in values]
    target_mean = sum(sum(row) for row in values) / 6
    assert math.isclose(sum(estimates) / 2, target_mean)
    assert not math.isclose(sum(sum(row) / len(row) for row in values) / 2, target_mean)
    assert uniform_window_factor(2, 6, 1, 0) == 0.0
    rejects(lambda: uniform_window_factor(None, 6, 1, 1), "requires audited")
    rejects(lambda: uniform_window_factor(2, 6, 1, 7), "requires audited")
    print(json.dumps(dict(status="V29_BOUNDARY_PARTITION_OK", partition=audit,
                          one_window_estimates=estimates, exact_pool_mean=target_mean,
                          averaged_estimate=sum(estimates) / 2), indent=2))


if __name__ == "__main__":
    main()
