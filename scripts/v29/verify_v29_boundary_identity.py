"""Pure UID and sampler checks; no model, torch, training, or server connection."""
import json

try:
    from divibody_training.v29_boundary_training import _audit_identities, _sample_indices
except ModuleNotFoundError:
    from v29_boundary_training import _audit_identities, _sample_indices


def window(start, uid):
    ids = list(range(start, start + 8))
    return dict(sequence="train_sequence", person_id=0, window_uid=uid,
                target_frame_ids=ids,
                target_frame_uids=[f"train_sequence/person_0/frame_{frame:05d}" for frame in ids],
                target_valid=[True] * 8)


def must_reject(function, phrase):
    try:
        function()
    except ValueError as exc:
        assert phrase in str(exc), (phrase, str(exc))
    else:
        raise AssertionError("Invalid identities were accepted: " + phrase)


def main():
    valid = [[True] * 8, [True] * 8]
    # Stride-4 cache: all primitive identities have within-window owners,
    # yet the evaluation stitch changes prediction context at frames 5 -> 6.
    metadata = [window(0, "overlap0"), window(4, "overlap1")]
    owners = [[True] * 6 + [False] * 2, [False] * 2 + [True] * 6]
    acceleration = [[True] * 5 + [False], [False] + [True] * 5]
    pool = _audit_identities(metadata, valid, owners, acceleration)
    assert pool.audit["full_contiguous_triplets"] == 10
    assert pool.audit["missing_from_any_within_window"] == 0
    assert pool.audit["boundary_triplets"] == 2
    assert pool.audit["boundary_also_within_window"] == 2
    assert [row.key[2:] for row in pool.triplets] == [(4, 5, 6), (5, 6, 7)]
    assert pool.triplets[0].locations == ((0, 4), (0, 5), (1, 2))
    assert pool.triplets[1].locations == ((0, 5), (1, 2), (1, 3))
    assert len(pool.overlap_frames) == 4
    stride1 = _audit_identities(
        [window(0, "dense0"), window(1, "dense1"), window(2, "dense2")],
        [[True] * 8] * 3,
        [[True] * 5 + [False] * 3,
         [False] * 4 + [True] + [False] * 3,
         [False] * 4 + [True] * 4],
        [[True] * 4 + [False] * 2,
         [False] * 3 + [True] + [False] * 2,
         [False] * 3 + [True] * 3],
    )
    assert stride1.audit["three_window_triplets"] == 1
    assert next(row for row in stride1.triplets if row.key[2:] == (4, 5, 6)).locations == ((0, 4), (1, 4), (2, 4))
    # Nonoverlap stride-8 case recovers the two omitted primitives exactly.
    stride8 = _audit_identities([window(0, "nonoverlap0"), window(8, "nonoverlap1")],
                               valid, [[True] * 8] * 2, [[True] * 6] * 2)
    assert stride8.audit["full_contiguous_triplets"] == 14
    assert stride8.audit["missing_from_any_within_window"] == 2
    assert [row.key[2:] for row in stride8.triplets] == [(6, 7, 8), (7, 8, 9)]
    # A genuine missing annotation frame never creates a synthetic triplet.
    gap = _audit_identities([window(0, "gap0"), window(10, "gap1")],
                           valid, [[True] * 8] * 2, [[True] * 6] * 2)
    assert gap.audit["full_contiguous_triplets"] == 12
    assert not gap.triplets
    must_reject(lambda: _audit_identities(metadata, valid, [[True] * 8] * 2, acceleration), "Duplicate training frame owner")
    bad_owner = [owners[0][:], owners[1][:]]
    bad_owner[0][0] = False
    must_reject(lambda: _audit_identities(metadata, valid, bad_owner, acceleration), "no unique owner")
    wrong_uid = [metadata[0] | dict(target_frame_uids=["wrong"] * 8), metadata[1]]
    must_reject(lambda: _audit_identities(wrong_uid, valid, owners, acceleration), "Frame UID does not match")
    must_reject(lambda: _audit_identities(metadata, valid, owners, acceleration,
                                         forbidden_sequences=["train_sequence"]), "forbidden split")
    duplicate_accel = [acceleration[0][:], acceleration[1][:]]
    duplicate_accel[1][0] = True
    must_reject(lambda: _audit_identities(metadata, valid, owners, duplicate_accel), "Duplicate within-window")
    sampler = dict(pool_size=17, seed=2026, count=1, identity_sha256=pool.identity_sha256, task="boundary")
    first = [_sample_indices(step=step, **sampler)[0] for step in range(17)]
    second = [_sample_indices(step=step, **sampler)[0] for step in range(17, 34)]
    assert sorted(first) == list(range(17))
    assert second == first
    assert _sample_indices(step=999, **sampler) == _sample_indices(step=999, **sampler)
    for size in (1, 2, 3, 8, 17, 90):
        kwargs = sampler | dict(pool_size=size, count=min(size, 4))
        for step in range(20):
            draw = _sample_indices(step=step, **kwargs)
            assert len(draw) == len(set(draw))
    print(json.dumps(dict(status="V29_BOUNDARY_IDENTITY_OK", stride4=pool.audit, stride1=stride1.audit,
                          stride8=stride8.audit, gap=gap.audit, state_free_sampling=True), indent=2))


if __name__ == "__main__":
    main()
