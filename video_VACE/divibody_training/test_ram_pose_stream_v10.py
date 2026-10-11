"""CPU-only synthetic-cache regression checks for v10 training coverage."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

try:
    from divibody_training.ram_pose_stream_v10 import (
        COLLECTION_FORMAT, CyclingPoseStream, ShuffledPoseDatasetV10,
    )
except ModuleNotFoundError as exc:
    if exc.name not in ("divibody_training", "divibody_training.ram_pose_stream_v10"):
        raise
    from ram_pose_stream_v10 import COLLECTION_FORMAT, CyclingPoseStream, ShuffledPoseDatasetV10


def make_cache(root, split="train"):
    vdir, hdir = root / "vace", root / "hmr"
    vdir.mkdir(parents=True)
    hdir.mkdir(parents=True)
    records = []
    uids_all = set()
    for index, count in enumerate((5, 4, 6)):
        uids = [f"{split}/{index}/{row}" for row in range(count)]
        uids_all.update(uids)
        # Two frames per window, separate frame tables in each aligned shard.
        values = (index * 100 + torch.arange(count)).float()
        frames = torch.arange(count * 2).reshape(count, 2)
        per_frame = values.repeat_interleave(2)
        vace = {
            "metadata": [{"window_uid": uid} for uid in uids],
            "features": values[:, None, None],
            "target_keypoints": values[:, None, None].expand(count, 2, 3),
            "frame_owner_mask": torch.ones(count, 2, dtype=torch.bool),
            "acceleration_owner_mask": torch.ones(count, 0, dtype=torch.bool),
            "target_valid_mask": torch.ones(count, 2, dtype=torch.bool),
        }
        hmr = {
            "window_uids": uids, "window_frame_indices": frames,
            "rgb_tokens": per_frame[:, None, None],
            "base_rotmat": per_frame[:, None, None],
            "base_betas": per_frame[:, None], "base_camera": per_frame[:, None],
        }
        name = f"shard_{index}.pt"
        torch.save(vace, vdir / name)
        torch.save(hmr, hdir / name)
        records.append({"index": index, "windows": count, "file": name})
    vid = f"synthetic-{split}"
    vm = {"status": "COMPLETE", "collection_format": COLLECTION_FORMAT,
          "collection_id": vid, "configuration": {"split": split}, "shards": records}
    hm = {"status": "COMPLETE", "configuration": {
        "source_vace_collection_id": vid, "split": split}, "shards": records}
    (vdir / "manifest.json").write_text(json.dumps(vm))
    (hdir / "manifest.json").write_text(json.dumps(hm))
    return vdir, hdir, uids_all


def expect_error(kind, action, fragment):
    try:
        action()
    except kind as exc:
        assert fragment in str(exc), str(exc)
    else:
        raise AssertionError(f"Expected {kind.__name__}: {fragment}")


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="ram_pose_stream_v10_") as tmp:
        vdir, hdir, expected = make_cache(Path(tmp) / "train")
        data = ShuffledPoseDatasetV10(vdir, hdir, seed=2026)
        assert data.split == "train" and data.collection_id == "synthetic-train"
        stream = CyclingPoseStream(DataLoader(data, batch_size=2, num_workers=0))
        # Model epochs end after two pose batches; the stream must not reset.
        epoch_uids = []
        for _epoch in range(2):
            seen = []
            for _step in range(2):
                batch = stream.next_batch()
                before = stream.coverage()
                assert before["draws"] == sum(map(len, epoch_uids)) + len(seen)
                seen.extend(batch["window_uid"])
                stream.record_consumed(batch)
            epoch_uids.append(seen)
        assert not set(epoch_uids[0]) & set(epoch_uids[1])
        assert stream.coverage()["unique_windows"] == 8
        first_cycle = epoch_uids[0] + epoch_uids[1]
        # Includes the final batch of size one (drop_last=False).
        for _ in range(4):
            batch = stream.next_batch()
            first_cycle.extend(batch["window_uid"])
            stream.record_consumed(batch)
        assert len(first_cycle) == 15 and set(first_cycle) == expected
        assert stream.coverage() == {"unique_windows": 15, "draws": 15,
                                     "pool_windows": 15, "coverage_fraction": 1.0, "cycle": 0}
        batch = stream.next_batch()
        assert stream.coverage()["draws"] == 15  # Fetched, not optimized yet.
        assert stream.coverage()["cycle"] == 1
        stream.record_consumed(batch)
        assert stream.coverage()["draws"] == 17
        assert stream.coverage()["unique_windows"] == 15

        replay = ShuffledPoseDatasetV10(vdir, hdir, seed=2026)
        samples = list(replay)
        assert [item["window_uid"] for item in samples] == first_cycle
        for item in samples:
            # Check VACE rows still match the HMR frame table after both shuffles.
            value = item["vace_features"].flatten()[0]
            assert torch.all(item["rgb_tokens"] == value)
            assert torch.all(item["base_camera"] == value)
            assert torch.all(item["keypoints"] == value)
        replay.set_cycle(1)
        cycle1 = [item["window_uid"] for item in replay]
        assert set(cycle1) == expected and cycle1 != first_cycle
        assert cycle1[:2] == batch["window_uid"]

        limited = ShuffledPoseDatasetV10(vdir, hdir, seed=2026, max_windows=5)
        limited0 = [item["window_uid"] for item in limited]
        limited.set_cycle(3)
        limited3 = [item["window_uid"] for item in limited]
        assert len(limited) == 5 and set(limited0) == set(limited3)
        assert len({uid.split("/")[1] for uid in limited0}) > 1
        assert set(limited0) == {item["window_uid"] for item in
                                ShuffledPoseDatasetV10(vdir, hdir, seed=2026, max_windows=5)}
        expect_error(ValueError, lambda: CyclingPoseStream(
            DataLoader(data, batch_size=2, num_workers=1)), "num_workers=0")
        expect_error(ValueError, lambda: CyclingPoseStream(
            DataLoader(data, batch_size=2, drop_last=True)), "drop_last=False")
        module_name = ShuffledPoseDatasetV10.__module__
        with patch(f"{module_name}.get_worker_info", return_value=object()):
            expect_error(RuntimeError, lambda: next(iter(data)), "num_workers=0")

        vval, hval, val_uids = make_cache(Path(tmp) / "validation", split="val")
        validation = ShuffledPoseDatasetV10(vval, hval)
        assert validation.split == "val" and validation.collection_id != data.collection_id
        assert not expected & val_uids
        expect_error(ValueError, lambda: ShuffledPoseDatasetV10(vdir, hval), "identities disagree")
        manifest_path = hval / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["configuration"]["split"] = "train"
        manifest_path.write_text(json.dumps(manifest))
        expect_error(ValueError, lambda: ShuffledPoseDatasetV10(vval, hval), "splits disagree")
    print("V10_POSE_STREAM_CHECKS_PASS: epoch continuity; full coverage; consumed-only counting; deterministic shuffles; fixed pool; alignment; split/worker guards")


if __name__ == "__main__":
    main()
