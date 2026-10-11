"""Strict real-RGB pooled evidence for separate complete train/validation splits.

Only the NPZ field strong_rgb_pooled is read. MHR predictions in the same archive
are never accessed or supplied as supervision. Actual image paths come from VACE
context metadata, while source_frame_id is retained as a separate time identity.
No split mixing, partial-coverage option, zero padding, or feature fallback exists.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any


FORMAT = "v29-strong-rgb-feature-manifest-v1"
FEATURE_POLICY = "real-sam-dinov3-global-spatial-mean"
RGB_RESOLVER_POLICY = "metadata-context-image-paths-annotation-file-index-v1"
FEATURE_KEY = "strong_rgb_pooled"
INPUT_KEYS = ("rgb_tokens", "vace_features", "base_rotmat", "base_betas", "base_camera")
ENCODER_ASSETS = {"checkpoint", "model_config", "backbone_source", "dinov3_hubconf", "extractor_source"}


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def uid_digest(uids) -> str:
    return hashlib.sha256(("\n".join(uids) + "\n").encode("utf-8")).hexdigest()


def _json_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _require_sha(value, name) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) or value == "0" * 64:
        raise ValueError("Invalid declared SHA256: " + name)
    return value


def _resolve(path, base: Path) -> Path:
    if not isinstance(path, str) or not path:
        raise ValueError("A nonempty explicit artifact path is required")
    value = Path(path)
    return (value if value.is_absolute() else base / value).resolve(strict=True)


def _confined_npz(path, base: Path) -> Path:
    value = _resolve(path, base)
    if not value.is_relative_to(base) or value.suffix.lower() != ".npz" or not value.is_file():
        raise ValueError("Feature NPZ must be a file inside its split manifest directory")
    return value


def _rows(value):
    return value.tolist() if hasattr(value, "tolist") else value


def expected_frame_records(vace: dict[str, Any], expected_split: str) -> dict[str, dict[str, Any]]:
    """Audit complete cache identities without reading any 3D target values."""
    import numpy as np
    signature = vace.get("signature", {})
    if expected_split not in ("train", "validation") or signature.get("split") != expected_split:
        raise ValueError("Strong evidence requires the explicitly matching train or validation split")
    if signature.get("development_only") is not False:
        raise ValueError("Strong evidence requires a complete, non-development source cache")
    metadata = vace["metadata"]
    valid_rows, owner_rows = _rows(vace["target_valid_mask"]), _rows(vace["frame_owner_mask"])
    if not metadata or len(metadata) != len(valid_rows) or len(metadata) != len(owner_rows):
        raise ValueError("Empty or mismatched strong-evidence source metadata")
    frames, owners, window_uids = {}, {}, set()
    for window, record in enumerate(metadata):
        uid = record["window_uid"]
        if not isinstance(uid, str) or not uid or uid in window_uids:
            raise ValueError("Empty or duplicate source window UID")
        window_uids.add(uid)
        sequence, person = record["sequence"], int(record["person_id"])
        if not isinstance(sequence, str) or not sequence or "/" in sequence or person < 0:
            raise ValueError("Invalid source person/sequence identity")
        ids, sources, uids = record["target_frame_ids"], record["target_source_frame_ids"], record["target_frame_uids"]
        paths, boxes = record["context_image_paths"], record["context_bbox_xyxy"]
        valid, owner = valid_rows[window], owner_rows[window]
        if (any(len(value) != 8 for value in (ids, sources, uids, valid, owner))
                or len(paths) != 17 or len(boxes) != 17):
            raise ValueError("Strong evidence requires explicit eight-target/seventeen-context identities")
        if any(type(value) is not bool for row in (valid, owner) for value in row):
            raise ValueError("Source ownership and validity masks must be Boolean")
        count = sum(valid)
        if count < 1 or valid != [True] * count + [False] * (8 - count) or record.get("target_valid") != valid:
            raise ValueError("Source targets require a consistent nonempty valid prefix")
        if any(int(ids[col]) != int(ids[col - 1]) + 1 for col in range(1, count)):
            raise ValueError("Source targets cross an annotation-frame gap")
        for col in range(8):
            annotation_id, source_id = int(ids[col]), int(sources[col])
            expected_uid = f"{sequence}/person_{person}/frame_{annotation_id:05d}"
            if annotation_id < 0 or source_id < 0 or uids[col] != expected_uid:
                raise ValueError("Strong evidence source UID disagrees with its time identity")
            if col >= count and (annotation_id != int(ids[count - 1]) or source_id != int(sources[count - 1])):
                raise ValueError("Padded targets must repeat the last real frame/time identity")
            if owner[col] and not valid[col]:
                raise ValueError("Padded source target has an owner")
            # This policy follows the verified official RGB renumbering. It does
            # not construct a filename from the original acquisition time ID.
            image = Path(paths[4 + col]).resolve(strict=True)
            if (not image.is_file() or image.parent.name != sequence
                    or image.name != f"image_{annotation_id:05d}.jpg"):
                raise ValueError("Source context image path disagrees with the declared annotation-file resolver")
            bbox = np.asarray(boxes[4 + col], dtype=np.float32)
            if bbox.shape != (4,) or not np.isfinite(bbox).all() or not (bbox[2:] > bbox[:2]).all():
                raise ValueError("Invalid source person crop bbox")
            frame = dict(uid=expected_uid, annotation_frame_id=annotation_id,
                         source_frame_id=source_id, image=str(image), bbox_xyxy=bbox.tolist())
            if valid[col]:
                if expected_uid in frames and frames[expected_uid] != frame:
                    raise ValueError("Overlapping source windows disagree about RGB/time/crop identity")
                frames[expected_uid] = frame
            else:
                if expected_uid not in frames or frames[expected_uid] != frame:
                    raise ValueError("Padding must reuse the real tail RGB identity exactly")
            if owner[col]:
                if expected_uid in owners:
                    raise ValueError("Duplicate strong-evidence source frame owner")
                owners[expected_uid] = (window, col)
    if set(owners) != set(frames):
        raise ValueError("Full strong-evidence source frames do not have one unique owner each")
    return {uid: frames[uid] for uid in sorted(frames)}


def _validate_feature_info(info, feature_dim: int):
    if not isinstance(info, dict) or info.get("pooled_dim") != feature_dim or not isinstance(info.get("output_path"), str) or not info["output_path"]:
        raise ValueError("Feature provenance requires an actual backbone output path and exact pooled dimension")
    shape, layout = info.get("forward_shape"), info.get("layout")
    if not isinstance(shape, list) or len(shape) != 4 or any(type(value) is not int or value <= 0 for value in shape) or shape[0] != 1:
        raise ValueError("Strong evidence must come from a declared single-person four-dimensional backbone tensor")
    if layout == "BCHW":
        channel, spatial = shape[1], shape[2:]
    elif layout == "BHWC":
        channel, spatial = shape[3], shape[1:3]
    else:
        raise ValueError("Only explicit BCHW/BHWC global spatial pooling is accepted")
    if channel != feature_dim or min(spatial) < 2:
        raise ValueError("Feature channels/spatial layout disagree with the pooled evidence")


class StrongFeatureStore:
    """Eagerly verify and retain only the complete split's small pooled vectors."""
    def __init__(self, manifest_path, vace, source_vace_cache, *, expected_split: str,
                 expected_dim: int, expected_manifest_sha256: str | None = None):
        import numpy as np
        self.path = Path(manifest_path).resolve(strict=True)
        base = self.path.parent
        manifest_hash = sha256_file(self.path)
        if expected_manifest_sha256 is not None and manifest_hash != _require_sha(expected_manifest_sha256, "manifest"):
            raise ValueError("Strong evidence manifest SHA256 differs from the frozen declaration")
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("format") != FORMAT or data.get("split") != expected_split or expected_split not in ("train", "validation"):
            raise ValueError("Wrong strong evidence manifest format or split")
        if (type(expected_dim) is not int or expected_dim < 1 or data.get("feature_dim") != expected_dim
                or data.get("feature_key") != FEATURE_KEY or data.get("feature_dtype") != "float32"
                or data.get("feature_policy") != FEATURE_POLICY or data.get("rgb_resolver_policy") != RGB_RESOLVER_POLICY):
            raise ValueError("Strong evidence dimensionality/dtype/extraction/resolver policy mismatch")
        flags = {"complete": True, "real_rgb": True, "random_features": False,
                 "placeholder_features": False, "three_d_targets_not_encoder_inputs": True,
                 "mhr_3d_used_as_supervision": False, "test_frames_used": False,
                 "encoder_frozen": True, "encoder_eval": True}
        if any(data.get(name) is not value for name, value in flags.items()):
            raise ValueError("Strong evidence requires explicit complete real-RGB/no-3D/no-fallback provenance")
        if data.get("input_bbox_policy") != "existing-vace-context-bbox-xyxy-at-target-position":
            raise ValueError("Strong evidence crop provenance differs from its source cache")
        cache = Path(source_vace_cache).resolve(strict=True)
        if sha256_file(cache) != _require_sha(data.get("source_vace_cache_sha256"), "source VACE cache"):
            raise ValueError("Strong evidence source VACE cache SHA256 mismatch")
        expected = expected_frame_records(vace, expected_split)
        uids = list(expected)
        if data.get("frame_uids_sha256") != uid_digest(uids) or data.get("source_rgb_mapping_sha256") != _json_digest(list(expected.values())):
            raise ValueError("Strong evidence UID or explicit context RGB mapping digest mismatch")
        encoder = data.get("encoder", {})
        if encoder.get("family") != "sam-3d-body-dinov3" or encoder.get("inference_type") != "body" or encoder.get("pooling") != "global-spatial-mean-float32":
            raise ValueError("Wrong declared strong RGB encoder/pooling provenance")
        assets = encoder.get("assets", {})
        if not isinstance(assets, dict) or not ENCODER_ASSETS.issubset(assets) or set(assets) - ENCODER_ASSETS - {"mhr_asset"}:
            raise ValueError("Strong evidence needs exact model/config/backbone/hub/extractor asset fingerprints")
        verified_assets = {}
        for name, reference in assets.items():
            if not isinstance(reference, dict):
                raise ValueError("Invalid encoder asset reference")
            declared = _require_sha(reference.get("sha256"), name)
            path = _resolve(reference.get("path"), base)
            if not path.is_file() or sha256_file(path) != declared:
                raise ValueError("Strong encoder asset SHA256 mismatch: " + name)
            verified_assets[name] = declared
        self.encoder_fingerprint = _json_digest(dict(family=encoder["family"], assets=verified_assets,
                                                    policy=FEATURE_POLICY, resolver=RGB_RESOLVER_POLICY,
                                                    feature_dim=expected_dim, pooling=encoder["pooling"]))
        rows = data.get("samples")
        if not isinstance(rows, list) or [row.get("uid") for row in rows if isinstance(row, dict)] != uids:
            raise ValueError("Strong evidence samples must cover every exact split UID once, in sorted order")
        if data.get("completed_frames") != len(uids):
            raise ValueError("Strong evidence declared completion count is not complete")
        vectors, paths_seen, image_hashes = [], set(), {}
        layout_info = None
        for row, uid in zip(rows, uids):
            source = expected[uid]
            if (row.get("annotation_frame_id") != source["annotation_frame_id"]
                    or row.get("source_frame_id") != source["source_frame_id"]
                    or _resolve(row.get("image"), base) != Path(source["image"])
                    or row.get("bbox_xyxy") != source["bbox_xyxy"]):
                raise ValueError("Strong feature UID has different annotation/time/image/crop provenance: " + uid)
            image = source["image"]
            if image not in image_hashes:
                image_hashes[image] = sha256_file(image)
            if image_hashes[image] != _require_sha(row.get("image_sha256"), "source RGB image"):
                raise ValueError("Strong feature source image SHA256 mismatch: " + uid)
            path = _confined_npz(row.get("file"), base)
            if path in paths_seen:
                raise ValueError("One person-frame feature NPZ cannot be reused for a different UID")
            paths_seen.add(path)
            if sha256_file(path) != _require_sha(row.get("file_sha256"), "feature NPZ"):
                raise ValueError("Strong feature NPZ SHA256 mismatch: " + uid)
            info = row.get("strong_rgb_feature")
            _validate_feature_info(info, expected_dim)
            signature = {key: info[key] for key in ("output_path", "forward_shape", "layout", "pooled_dim")}
            if layout_info is not None and signature != layout_info:
                raise ValueError("Split features have inconsistent backbone output/layout provenance")
            layout_info = signature
            # Access exactly one NPZ field. MHR arrays are never loaded.
            with np.load(path, allow_pickle=False) as archive:
                if FEATURE_KEY not in archive.files:
                    raise ValueError("Required genuine strong feature is absent: " + uid)
                vector = archive[FEATURE_KEY]
                if vector.dtype != np.dtype("float32") or vector.shape != (expected_dim,) or not np.isfinite(vector).all():
                    raise ValueError("Strong evidence must be an exact finite float32 [D] vector: " + uid)
                if not np.count_nonzero(vector):
                    raise ValueError("Zero/placeholder strong evidence is forbidden: " + uid)
                vectors.append(vector.copy())
        self.vectors = np.stack(vectors)
        self.vectors.setflags(write=False)
        self.uid_to_index = {uid: index for index, uid in enumerate(uids)}
        self.split, self.feature_dim = expected_split, expected_dim
        self.vace_object_id = id(vace)
        self.mapping = expected
        self.audit = dict(format=FORMAT, split=expected_split, complete=True, frames=len(uids),
                          manifest_sha256=manifest_hash, source_vace_cache_sha256=data["source_vace_cache_sha256"],
                          frame_uids_sha256=uid_digest(uids), source_rgb_mapping_sha256=data["source_rgb_mapping_sha256"],
                          encoder_fingerprint=self.encoder_fingerprint, feature_dim=expected_dim,
                          source_images=len(image_hashes), source_ids_are_separate_time_identity=True,
                          feature_policy=FEATURE_POLICY, rgb_resolver_policy=RGB_RESOLVER_POLICY,
                          mhr_arrays_read=False, missing_feature_fallback=False,
                          all_npz_image_encoder_hashes_verified=True, backbone_layout=layout_info)
        self.fingerprint = _json_digest(self.audit)

    def window_features(self, record):
        import numpy as np
        uids = record["target_frame_uids"]
        if len(uids) != 8 or any(uid not in self.uid_to_index for uid in uids):
            raise ValueError("A window contains a missing strong feature UID; fallback is forbidden")
        return np.stack([self.vectors[self.uid_to_index[uid]] for uid in uids])


class StrongRGBFeatureDataset:
    """Composition wrapper; build strict V10 boundary pool on base_dataset first."""
    def __init__(self, base_dataset, store: StrongFeatureStore):
        if (not hasattr(base_dataset, "vace") or not hasattr(base_dataset, "sidecar")
                or id(base_dataset.vace) != store.vace_object_id
                or base_dataset.vace["signature"].get("split") != store.split
                or base_dataset.sidecar["signature"].get("split") != store.split
                or len(base_dataset) != len(base_dataset.vace["metadata"])):
            raise ValueError("Strong RGB wrapper requires the exact complete source V10 dataset and split")
        self.base_dataset, self.store = base_dataset, store
        self.vace, self.sidecar, self.lookup = base_dataset.vace, base_dataset.sidecar, base_dataset.lookup

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        import torch
        sample = dict(self.base_dataset[index])
        if FEATURE_KEY in sample:
            raise ValueError("Source dataset already supplies an ambiguous strong feature")
        vector = self.store.window_features(self.vace["metadata"][index])
        sample[FEATURE_KEY] = torch.from_numpy(vector)
        return sample


def wrap_strong_train_validation(train_dataset, validation_dataset,
                                  train_manifest, validation_manifest,
                                  train_vace_cache, validation_vace_cache, *, expected_dim):
    """Require separate complete manifests and a single matched encoder policy."""
    if Path(train_manifest).resolve() == Path(validation_manifest).resolve():
        raise ValueError("Train and validation require separate strong feature manifests")
    train = StrongFeatureStore(train_manifest, train_dataset.vace, train_vace_cache,
                               expected_split="train", expected_dim=expected_dim)
    validation = StrongFeatureStore(validation_manifest, validation_dataset.vace, validation_vace_cache,
                                    expected_split="validation", expected_dim=expected_dim)
    if train.encoder_fingerprint != validation.encoder_fingerprint or train.audit["backbone_layout"] != validation.audit["backbone_layout"]:
        raise ValueError("Train and validation must use the same actual frozen strong RGB encoder/layout")
    train_sequences = {row["uid"].split("/", 1)[0] for row in train.mapping.values()}
    val_sequences = {row["uid"].split("/", 1)[0] for row in validation.mapping.values()}
    if train_sequences & val_sequences or set(train.uid_to_index) & set(validation.uid_to_index):
        raise ValueError("Strong feature train/validation sequence or UID leakage")
    return (StrongRGBFeatureDataset(train_dataset, train), StrongRGBFeatureDataset(validation_dataset, validation),
            dict(train=train.audit, validation=validation.audit,
                 train_fingerprint=train.fingerprint, validation_fingerprint=validation.fingerprint,
                 encoder_fingerprint=train.encoder_fingerprint, sequence_overlap=[]))


def forward_strong_batch(model, batch, device, amp_dtype, position_offset=0):
    """Explicit strong input forwarding; never send original 3D labels to model."""
    import torch
    if amp_dtype not in ("bf16", "none") or FEATURE_KEY not in batch:
        raise ValueError("Explicit real strong RGB evidence and supported AMP mode are mandatory")
    strong = batch[FEATURE_KEY]
    if not torch.is_tensor(strong) or not torch.is_floating_point(strong):
        raise TypeError("Strong batch evidence must be a floating tensor")
    if not hasattr(model, "strong_config") or tuple(strong.shape) != (*batch["rgb_tokens"].shape[:2], model.strong_config.input_dim):
        raise ValueError("Strong batch dimensions differ from the actual checkpoint config")
    if not bool(torch.isfinite(strong).all()):
        raise FloatingPointError("Nonfinite strong evidence in a training/validation batch")
    use_amp = device.type == "cuda" and amp_dtype == "bf16"
    values = [batch[key].to(device, non_blocking=True) for key in INPUT_KEYS]
    if not use_amp:
        values = [value.float() for value in values]
    valid = batch.get("target_valid")
    valid = valid.to(device, dtype=torch.bool) if valid is not None else None
    schema = batch.get("landmark_schema", "posetrack")
    if isinstance(schema, (list, tuple)):
        if len(set(schema)) != 1:
            raise ValueError("A strong evidence batch must use one landmark schema")
        schema = schema[0]
    source_ids = batch.get("source_frame_ids")
    source_ids = source_ids.to(device) if source_ids is not None else None
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
        return model(*values, frame_valid=valid, position_offset=position_offset,
                     landmark_schema=schema, source_frame_ids=source_ids,
                     strong_rgb_pooled=strong.to(device=device, dtype=torch.float32, non_blocking=True))
