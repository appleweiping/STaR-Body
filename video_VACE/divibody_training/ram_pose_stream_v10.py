"""Shuffled PoseTrack training stream copied from train_ram_posetrack_sharded_camera_v6.py.

Preserves its manifest/shard checks and tensor batch schema. No encoder or SMPL
imports are needed. Original source remains unchanged.
"""
from __future__ import annotations

import gc
import json
import random
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, get_worker_info

COLLECTION_FORMAT = "ram-posetrack18-contiguous-vace-sharded-v2"


class ShuffledPoseDatasetV10(IterableDataset):
    """One aligned shard pair resident, shuffled once per cycle.

    Use num_workers=0 only. max_windows=0 selects the full collection; a positive
    limit selects a fixed uniform sample across ALL windows, not an epoch prefix.
    """
    def __init__(self, vace_dir, hmr_dir, seed=2026, max_windows=0):
        super().__init__()
        self.vace_dir, self.hmr_dir = Path(vace_dir), Path(hmr_dir)
        vm = json.loads((self.vace_dir / "manifest.json").read_text())
        hm = json.loads((self.hmr_dir / "manifest.json").read_text())
        if vm.get("status") != "COMPLETE" or vm.get("collection_format") != COLLECTION_FORMAT:
            raise ValueError("VACE collection is not complete")
        if hm.get("status") != "COMPLETE":
            raise ValueError("HMR collection is not complete")
        if not vm.get("collection_id") or vm["collection_id"] != hm["configuration"].get("source_vace_collection_id"):
            raise ValueError("VACE/HMR collection identities disagree")
        self.collection_id = vm["collection_id"]
        self.split = vm.get("configuration", {}).get("split")
        hmr_split = hm.get("configuration", {}).get("split")
        if any(split is not None and split not in ("train", "val") for split in (self.split, hmr_split)):
            raise ValueError("Unsupported PoseTrack split")
        if self.split is not None and hmr_split is not None and self.split != hmr_split:
            raise ValueError("VACE/HMR splits disagree")
        self.vace_records = sorted(vm["shards"], key=lambda record: int(record["index"]))
        self.hmr_records = {int(record["index"]): record for record in hm["shards"]}
        indices = [int(record["index"]) for record in self.vace_records]
        if len(set(indices)) != len(indices) or len(self.hmr_records) != len(hm["shards"]):
            raise ValueError("Duplicate shard index in collection")
        if set(self.hmr_records) != set(indices):
            raise ValueError("VACE/HMR shard sets disagree")
        self.total_windows = 0
        for record in self.vace_records:
            count = int(record["windows"])
            hmr_count = self.hmr_records[int(record["index"])].get("windows")
            if count <= 0 or (hmr_count is not None and int(hmr_count) != count):
                raise ValueError("VACE/HMR shard window counts disagree or are empty")
            self.total_windows += count
        if self.total_windows == 0:
            raise ValueError("PoseTrack training collection is empty")
        if int(max_windows) < 0:
            raise ValueError("max_windows must be nonnegative")
        self.seed, self.cycle = int(seed), 0
        self.windows = min(int(max_windows), self.total_windows) if max_windows else self.total_windows
        self._pool_rows = None
        if self.windows < self.total_windows:
            selected = sorted(random.Random(self.seed).sample(range(self.total_windows), self.windows))
            self._pool_rows = {}
            cursor = offset = 0
            for record in self.vace_records:
                end = offset + int(record["windows"])
                rows = []
                while cursor < len(selected) and selected[cursor] < end:
                    rows.append(selected[cursor] - offset)
                    cursor += 1
                if rows:
                    self._pool_rows[int(record["index"])] = rows
                offset = end

    def __len__(self):
        return self.windows

    def set_cycle(self, cycle):
        if int(cycle) < 0:
            raise ValueError("cycle must be nonnegative")
        self.cycle = int(cycle)

    @staticmethod
    def _load(path):
        try:
            return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def __iter__(self):
        if get_worker_info() is not None:
            raise RuntimeError("ShuffledPoseDatasetV10 requires num_workers=0; workers would repeat windows")
        rng = random.Random(self.seed + self.cycle)
        records = [record for record in self.vace_records
                   if self._pool_rows is None or int(record["index"]) in self._pool_rows]
        rng.shuffle(records)
        for record in records:
            index = int(record["index"])
            vace = self._load(self.vace_dir / record["file"])
            hmr = self._load(self.hmr_dir / self.hmr_records[index]["file"])
            try:
                uids = [item["window_uid"] for item in vace["metadata"]]
                if len(uids) != int(record["windows"]):
                    raise ValueError(f"VACE manifest window count disagrees in shard {index}")
                if uids != hmr["window_uids"]:
                    raise ValueError(f"VACE/HMR identities disagree in shard {index}")
                if any(not isinstance(uid, str) or not uid for uid in uids) or len(set(uids)) != len(uids):
                    raise ValueError(f"Invalid or duplicate window UID in shard {index}")
                rows = list(range(len(uids))) if self._pool_rows is None else list(self._pool_rows[index])
                rng.shuffle(rows)
                for row in rows:
                    frames = hmr["window_frame_indices"][row]
                    yield {"rgb_tokens": hmr["rgb_tokens"][frames],
                           "vace_features": vace["features"][row],
                           "base_rotmat": hmr["base_rotmat"][frames],
                           "base_betas": hmr["base_betas"][frames],
                           "base_camera": hmr["base_camera"][frames],
                           "keypoints": vace["target_keypoints"][row],
                           "frame_owner": vace["frame_owner_mask"][row],
                           "acceleration_owner": vace["acceleration_owner_mask"][row],
                           "target_valid": vace["target_valid_mask"][row],
                           "window_uid": uids[row]}
            finally:
                del hmr, vace
                gc.collect()


class CyclingPoseStream:
    """Keep one iterator across epochs; restart only after exhausting the pool.

    Call record_consumed(batch) AFTER a successful optimizer.step(). Fetches and
    validation iterations do not contribute to optimized-window coverage.
    """
    def __init__(self, loader):
        if loader.num_workers != 0:
            raise ValueError("CyclingPoseStream requires num_workers=0")
        if loader.drop_last:
            raise ValueError("CyclingPoseStream requires drop_last=False for complete coverage")
        if not isinstance(loader.dataset, ShuffledPoseDatasetV10):
            raise TypeError("CyclingPoseStream requires ShuffledPoseDatasetV10")
        self.loader, self.dataset = loader, loader.dataset
        self.cycle = 0
        self.dataset.set_cycle(self.cycle)
        self._iterator = iter(loader)
        self._seen = set()
        self._draws = 0

    def next_batch(self):
        try:
            return next(self._iterator)
        except StopIteration:
            self.cycle += 1
            self.dataset.set_cycle(self.cycle)
            self._iterator = iter(self.loader)
            try:
                return next(self._iterator)
            except StopIteration as exc:
                raise RuntimeError("PoseTrack training loader produced no batches") from exc

    def record_consumed(self, batch):
        uids = batch["window_uid"]
        if isinstance(uids, str):
            uids = [uids]
        if not uids or any(not isinstance(uid, str) or not uid for uid in uids):
            raise ValueError("Consumed batch must retain valid window_uid strings")
        self._draws += len(uids)
        self._seen.update(uids)
        if len(self._seen) > len(self.dataset):
            raise RuntimeError("Consumed coverage exceeds the configured training pool")

    def coverage(self):
        return {"unique_windows": len(self._seen), "draws": self._draws,
                "pool_windows": len(self.dataset),
                "coverage_fraction": len(self._seen) / len(self.dataset),
                "cycle": self.cycle}
