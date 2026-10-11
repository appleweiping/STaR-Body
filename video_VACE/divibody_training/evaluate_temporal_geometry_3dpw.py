"""Evaluate the additive v3 geometry decoder with standard full-split 3DPW metrics.

Copied from ``evaluate_temporal_3dpw.py``; the original evaluator and model files
remain unchanged. Predictions contain all 24 SMPL rotations in camera/local
coordinates, so camera extrinsics are applied to ground truth only.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from evaluate_standard_3dpw import (
    EXPECTED_TEST_ACCELERATION_TRIPLETS,
    H36M_TO_J14,
    Standard3DPWDataset,
    _normalise_gender,
    audit_split,
    create_smpl_models,
    enforce_complete_test,
    per_sequence_summary,
    sha256_file,
)
from geometry_metrics_3dpw import (
    hip_centered_j14,
    mean_euclidean_error,
    pa_mpjpe_per_sample,
    pelvis_aligned_j14,
    sequence_acceleration_error,
)
from models.human_decoder_geometry import (
    GeometryDecoderConfig,
    HumanGeometryQueryTransformer,
)
from real_3dpw_dataset import _load_annotation
from train_divibody_real_3dpw_temporal import axis_angle_to_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a v3 geometry DiviBody checkpoint on strict 3DPW features"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--smpl-model-root", type=Path, required=True)
    parser.add_argument("--joint-regressor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--smpl-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-cache-hash", action="store_true")
    return parser.parse_args()


def validate_checkpoint_protocol(
    checkpoint: Dict[str, Any], split: str | None = None
) -> int:
    """Validate the v3 contract, returning the trained temporal context length.

    The v3 protocol is intentionally independent of v2's axis-angle metadata.
    ``max_latent_frames`` is an embedding capacity, not the trained clip length.
    """

    if checkpoint.get("format") != "divibody-real-3dpw-temporal-geometry-v3":
        raise ValueError("Not a v3 geometry DiviBody checkpoint")
    if checkpoint.get("target_coordinate_system") != "camera":
        raise ValueError("Geometry v3 requires camera-coordinate predictions")
    protocol = checkpoint.get("pose_protocol", {})
    if not isinstance(protocol, dict) or protocol.get("id") != (
        "3dpw_camera_smpl24_geometry_v3"
    ):
        raise ValueError("Incompatible geometry v3 pose protocol")
    config = checkpoint.get("decoder_config", {})
    saved_args = checkpoint.get("args", {})
    if not isinstance(config, dict) or not isinstance(saved_args, dict):
        raise ValueError("Missing geometry decoder configuration or training args")
    if split == "test" and (
        checkpoint.get("development_only", False)
        or saved_args.get("smoke_max_steps", 0)
    ):
        raise ValueError("Development-only geometry checkpoints cannot run formal test")
    clip_length = saved_args.get("clip_length")
    if isinstance(clip_length, bool) or not isinstance(clip_length, int):
        raise ValueError("Checkpoint args must include integer clip_length")
    capacity = config.get("max_latent_frames", 128)
    if clip_length < 3 or clip_length > capacity:
        raise ValueError("Geometry clip_length is outside decoder temporal capacity")
    if config.get("input_dim", 1536) != 1536:
        raise ValueError("Geometry v3 requires 1536-channel VACE features")
    return clip_length


class CenteredFrameClipDataset(Dataset):
    """Return one deterministic centered temporal context per real frame."""

    def __init__(self, payload: Dict[str, Any], clip_length: int) -> None:
        self.features = payload["features"]
        self.metadata = payload["metadata"]
        if self.features.ndim != 3 or self.features.shape[-1] != 1536:
            raise ValueError(f"Invalid cached features {tuple(self.features.shape)}")
        if len(self.metadata) != self.features.shape[0]:
            raise ValueError("Feature/metadata lengths differ")
        self.clip_length = int(clip_length)
        if self.clip_length < 3:
            raise ValueError("Temporal checkpoint clip length must be at least 3")

        self.sequences = [str(record["sequence"]) for record in self.metadata]
        self.person_ids = [int(record["person_id"]) for record in self.metadata]
        self.frame_ids = [int(record["frame_id"]) for record in self.metadata]
        tracks: dict[tuple[str, int], list[int]] = defaultdict(list)
        for index, identity in enumerate(zip(self.sequences, self.person_ids)):
            tracks[identity].append(index)

        requests: list[tuple[torch.Tensor, int, int]] = []
        short_segment_frames = 0
        segment_count = 0
        gap_count = 0
        for track, indices_list in sorted(tracks.items()):
            indices = sorted(indices_list, key=lambda value: self.frame_ids[value])
            frames = [self.frame_ids[index] for index in indices]
            if len(frames) != len(set(frames)):
                raise ValueError(f"Duplicate frame in feature track {track}")
            boundaries = []
            for position in range(1, len(indices)):
                if frames[position] != frames[position - 1] + 1:
                    boundaries.append(position)
                    gap_count += 1
            segment_start = 0
            for segment_end in (*boundaries, len(indices)):
                segment = indices[segment_start:segment_end]
                segment_start = segment_end
                segment_count += 1
                if len(segment) >= self.clip_length:
                    for position, target_index in enumerate(segment):
                        start = min(
                            max(position - self.clip_length // 2, 0),
                            len(segment) - self.clip_length,
                        )
                        clip = segment[start : start + self.clip_length]
                        requests.append(
                            (
                                torch.as_tensor(clip, dtype=torch.long),
                                target_index,
                                position - start,
                            )
                        )
                else:
                    short_segment_frames += len(segment)
                    pad_total = self.clip_length - len(segment)
                    for position, target_index in enumerate(segment):
                        pad_left = min(
                            max(self.clip_length // 2 - position, 0), pad_total
                        )
                        pad_right = pad_total - pad_left
                        clip = (
                            [segment[0]] * pad_left
                            + segment
                            + [segment[-1]] * pad_right
                        )
                        requests.append(
                            (
                                torch.as_tensor(clip, dtype=torch.long),
                                target_index,
                                pad_left + position,
                            )
                        )
        target_indices = [request[1] for request in requests]
        if len(requests) != len(self.metadata) or sorted(target_indices) != list(
            range(len(self.metadata))
        ):
            raise RuntimeError("Centered stitch requests do not cover every frame once")
        self.requests = requests
        self.stats = {
            "frames": len(self.metadata),
            "tracks": len(tracks),
            "segments": segment_count,
            "frame_gaps": gap_count,
            "clip_length": self.clip_length,
            "requests": len(requests),
            "short_segment_frames_edge_repeated": short_segment_frames,
            "stitch": "one centered prediction per frame; clamp at long-run boundaries",
        }

    def __len__(self) -> int:
        return len(self.requests)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        frame_indices, target_index, local_position = self.requests[index]
        return {
            "features": self.features.index_select(0, frame_indices),
            "target_index": torch.tensor(target_index, dtype=torch.long),
            "local_position": torch.tensor(local_position, dtype=torch.long),
        }


def _normalise_feature_backend(value: Any) -> str:
    backend = str(value).strip().lower()
    return "vace" if "vace" in backend else backend


def validate_feature_compatibility(
    checkpoint: Dict[str, Any], payload: Dict[str, Any]
) -> None:
    """Reject shape-compatible caches produced with different VACE settings."""

    trained = checkpoint.get("train_cache_signature", {})
    cached = payload.get("signature", {})
    if not isinstance(trained, dict) or not isinstance(cached, dict):
        raise ValueError("Checkpoint or feature cache lacks a usable signature")
    required = ("image_size", "feature_layers", "vace_timestep", "feature_backend")
    for label, signature in (("train", trained), ("evaluation", cached)):
        if any(signature.get(key) is None for key in required):
            raise ValueError(f"Incomplete {label} feature signature")
    if trained.get("split") != "train":
        raise ValueError("Checkpoint feature signature must identify the train split")
    mismatches = []
    for key in ("image_size", "feature_layers", "vace_timestep"):
        trained_value = trained.get(key)
        cached_value = cached.get(key)
        if key == "feature_layers":
            trained_value = list(trained_value or [])
            cached_value = list(cached_value or [])
        if trained_value != cached_value:
            mismatches.append(
                f"{key}: trained={trained_value!r}, cache={cached_value!r}"
            )
    if _normalise_feature_backend(trained.get("feature_backend")) != (
        _normalise_feature_backend(cached.get("feature_backend"))
    ):
        mismatches.append(
            "feature_backend: "
            f"trained={trained.get('feature_backend')!r}, "
            f"cache={cached.get('feature_backend')!r}"
        )
    if mismatches:
        raise ValueError(
            "Incompatible temporal feature cache: " + "; ".join(mismatches)
        )


def validate_cache_and_alignment(
    payload: Dict[str, Any], data_root: Path, split: str
) -> Dict[str, Any]:
    if split not in ("validation", "test"):
        raise ValueError("Geometry evaluation accepts only validation or test splits")
    signature = payload.get("signature", {})
    if signature.get("format") != "divibody-strict-vace-features-v2":
        raise ValueError("Temporal evaluation requires a strict v2 VACE cache")
    if signature.get("split") != split:
        raise ValueError(f"Feature cache split is not {split!r}")
    required_targets = ("full_pose", "gender_code", "betas")
    if any(name not in payload.get("targets", {}) for name in required_targets):
        raise ValueError("Strict cache lacks full SMPL evaluation targets")
    image_size = signature.get("image_size", 0)
    if not isinstance(image_size, int) or image_size <= 0 or image_size % 16:
        raise ValueError("Strict feature image_size must be a positive multiple of 16")
    features = payload.get("features")
    metadata = payload.get("metadata", [])
    expected_shape = (len(metadata), (image_size // 16) ** 2, 1536)
    if (
        not isinstance(features, torch.Tensor)
        or tuple(features.shape) != expected_shape
    ):
        raise ValueError(f"Strict feature shape must be {expected_shape}")
    if signature.get("strict_rgb_resolver") != "image_{annotation_frame_index:05d}":
        raise ValueError("Strict cache lacks the annotation-index RGB resolver")

    audit = audit_split(data_root, split)
    if split == "test":
        enforce_complete_test(audit, allow_incomplete=False)
    completeness_failures = []
    if audit["available_valid_person_frames"] != audit["expected_valid_person_frames"]:
        completeness_failures.append("not all annotated person frames have RGB")
    if audit["missing_unique_rgb_frames"] != 0:
        completeness_failures.append("one or more required RGB frames are missing")
    if audit["invalid_gender_tracks"]:
        completeness_failures.append("one or more tracks have invalid gender metadata")
    if audit["invalid_camera_frame_count"]:
        completeness_failures.append("one or more frames have invalid camera metadata")
    if completeness_failures:
        raise RuntimeError(
            f"Incomplete or invalid {split} split: " + "; ".join(completeness_failures)
        )
    dataset = Standard3DPWDataset(
        root=data_root,
        split=split,
        image_size=int(signature["image_size"]),
        frame_stride=1,
        max_samples=None,
        crop_to_person=True,
    )
    metadata = payload["metadata"]
    cache_keys = [
        (str(item["sequence"]), int(item["person_id"]), int(item["frame_id"]))
        for item in metadata
    ]
    expected_keys = [
        (sample.sequence, sample.person_index, sample.frame_index)
        for sample in dataset.samples
    ]
    if cache_keys != expected_keys:
        raise RuntimeError(
            "Strict feature cache order/content differs from dataset index"
        )
    if len(cache_keys) != len(set(cache_keys)):
        raise RuntimeError("Strict feature cache contains duplicate identity keys")
    if len(cache_keys) != int(audit["available_valid_person_frames"]):
        raise RuntimeError("Strict feature cache does not have complete split coverage")
    # Identity keys alone cannot detect stale/reordered targets or a camera
    # convention error. Check cached GT against source annotations, without ever
    # giving it to the prediction DataLoader or model.
    validate_cached_annotation_values(payload, dataset.samples)
    crop_fallbacks = sum(not bool(item.get("crop_applied", False)) for item in metadata)
    return {
        **audit,
        "crop_fallback_person_frames": crop_fallbacks,
        "cached_targets_checked_against_source_annotations": len(metadata),
    }


def validate_cached_annotation_values(
    payload: Dict[str, Any], samples: list[Any]
) -> None:
    """Require cached full pose/shape/gender/extrinsics to match original GT."""

    targets = payload["targets"]
    count = len(samples)
    full_pose = np.asarray(targets["full_pose"], dtype=np.float32)
    betas = np.asarray(targets["betas"], dtype=np.float32)
    gender = np.asarray(targets["gender_code"])
    if full_pose.shape != (count, 72) or betas.shape != (count, 10):
        raise ValueError("Invalid cached full SMPL target shapes")
    if gender.shape != (count,) or not np.issubdtype(gender.dtype, np.integer):
        raise ValueError("Strict cache gender codes must be one integer per frame")
    metadata = payload["metadata"]
    for index, sample in enumerate(samples):
        annotation = _load_annotation(sample.annotation_path)
        person, frame = sample.person_index, sample.frame_index
        expected_pose = np.asarray(
            annotation["poses"][person][frame], np.float32
        ).reshape(-1)[:72]
        expected_betas = np.asarray(annotation["betas"][person], np.float32).reshape(
            -1
        )[:10]
        expected_camera = np.asarray(annotation["cam_poses"][frame], np.float32)
        actual_camera = np.asarray(metadata[index]["cam_pose"], np.float32)
        expected_gender = _normalise_gender(annotation["genders"][person])
        if (
            not np.allclose(full_pose[index], expected_pose, atol=1e-6, rtol=0)
            or not np.allclose(betas[index], expected_betas, atol=1e-6, rtol=0)
            or int(gender[index]) != expected_gender
            or actual_camera.shape != (4, 4)
            or not np.allclose(actual_camera, expected_camera, atol=1e-6, rtol=0)
        ):
            raise ValueError(
                "Cached SMPL targets/camera disagree with source annotation at "
                f"{sample.sequence}/{person}/{frame}"
            )
        if Path(str(metadata[index]["image_path"])).stem != f"image_{frame:05d}":
            raise ValueError(
                "Cached RGB does not use the strict annotation frame index"
            )


def run_temporal_predictions(
    decoder: HumanGeometryQueryTransformer,
    dataset: CenteredFrameClipDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    amp_dtype: str = "bf16",
) -> tuple[np.ndarray, np.ndarray]:
    if amp_dtype not in ("bf16", "none"):
        raise ValueError("Geometry inference amp_dtype must be bf16 or none")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    sample_count = len(dataset)
    pred_rotmat = np.empty((sample_count, 24, 3, 3), dtype=np.float32)
    pred_betas = np.empty((sample_count, 10), dtype=np.float32)
    seen = np.zeros(sample_count, dtype=bool)
    decoder.eval()
    for step, batch in enumerate(loader, start=1):
        features = batch["features"].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        if not bool(torch.isfinite(features).all()):
            raise ValueError("Non-finite geometry inference features")
        batch_count, frames, tokens, channels = features.shape
        grid_width = math.isqrt(tokens)
        if grid_width * grid_width != tokens:
            raise RuntimeError("Cached spatial token count is not square")
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda" and amp_dtype == "bf16",
            ),
        ):
            output = decoder(
                features.reshape(batch_count, frames * tokens, channels),
                grid_size=(frames, grid_width, grid_width),
                target_frames=frames,
            )
        local = batch["local_position"].to(device)
        row = torch.arange(batch_count, device=device)
        if output["rotmat"].shape != (batch_count, frames, 24, 3, 3):
            raise ValueError(
                "Geometry decoder must return all 24 SMPL rotation matrices"
            )
        if output["betas"].shape != (batch_count, 10):
            raise ValueError("Geometry decoder betas must have shape [B,10]")
        rotations = output["rotmat"][row, local].float()
        target_indices = batch["target_index"].numpy()
        if seen[target_indices].any():
            raise RuntimeError("Temporal stitch attempted to write a frame twice")
        pred_rotmat[target_indices] = rotations.cpu().numpy()
        pred_betas[target_indices] = output["betas"].float().cpu().numpy()
        seen[target_indices] = True
        if step % 50 == 0 or step == len(loader):
            print(
                json.dumps(
                    {
                        "phase": "temporal_geometry_centered_prediction",
                        "step": step,
                        "steps": len(loader),
                        "frames": int(seen.sum()),
                    }
                ),
                flush=True,
            )
    if not seen.all():
        raise RuntimeError(f"Temporal stitch missed {int((~seen).sum())} frames")
    return pred_rotmat, pred_betas


def validate_geometry_inputs(
    pred_rotmat: np.ndarray,
    pred_betas: np.ndarray,
    gt_pose: np.ndarray,
    gt_betas: np.ndarray,
    gender_codes: np.ndarray,
    camera_rotations: np.ndarray,
) -> None:
    sample_count = len(pred_rotmat)
    if not (
        pred_rotmat.shape == (sample_count, 24, 3, 3)
        and pred_betas.shape == (sample_count, 10)
        and gt_pose.shape == (sample_count, 72)
        and gt_betas.shape == (sample_count, 10)
        and gender_codes.shape == (sample_count,)
        and camera_rotations.shape == (sample_count, 3, 3)
    ):
        raise ValueError("Strict cache evaluation target shapes are invalid")
    if not all(
        np.isfinite(array).all()
        for array in (
            pred_rotmat,
            pred_betas,
            gt_pose,
            gt_betas,
            camera_rotations,
        )
    ):
        raise ValueError("Temporal prediction or SMPL evaluation target is non-finite")
    if not np.isin(gender_codes, (0, 1, 2)).all():
        invalid = sorted(set(gender_codes.tolist()) - {0, 1, 2})
        raise ValueError(f"Invalid gender codes in strict feature cache: {invalid}")
    for name, rotations in (
        ("predicted SMPL", pred_rotmat),
        ("camera", camera_rotations),
    ):
        gram = np.swapaxes(rotations, -1, -2) @ rotations
        if not np.allclose(gram, np.eye(3), atol=1e-4, rtol=1e-4) or not np.allclose(
            np.linalg.det(rotations), 1.0, atol=1e-4, rtol=1e-4
        ):
            raise ValueError(f"Invalid {name} rotation matrices; expected proper SO(3)")


def compute_geometry_metrics(
    pred_rotmat: np.ndarray,
    pred_betas: np.ndarray,
    payload: Dict[str, Any],
    smpl_root: Path,
    joint_regressor_path: Path,
    smpl_batch_size: int,
    device: torch.device,
    target_coordinate_system: str = "camera",
) -> tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    if target_coordinate_system != "camera":
        raise ValueError(
            f"Unsupported pose coordinate system {target_coordinate_system!r}"
        )
    gt_pose = np.asarray(payload["targets"]["full_pose"], dtype=np.float32)
    gt_betas = np.asarray(payload["targets"]["betas"], dtype=np.float32)
    gender_codes = np.asarray(payload["targets"]["gender_code"], dtype=np.int64)
    metadata = payload["metadata"]
    camera_rotations = np.asarray(
        [record["cam_pose"] for record in metadata], dtype=np.float32
    )[:, :3, :3]
    sample_count = len(pred_rotmat)
    validate_geometry_inputs(
        pred_rotmat,
        pred_betas,
        gt_pose,
        gt_betas,
        gender_codes,
        camera_rotations,
    )

    regressor_np = np.load(joint_regressor_path)
    if regressor_np.shape != (17, 6890) or not np.isfinite(regressor_np).all():
        raise ValueError("Invalid H36M joint regressor")
    regressor = torch.from_numpy(regressor_np).to(device=device, dtype=torch.float32)
    smpl_models = create_smpl_models(smpl_root, device)
    metric_arrays = {
        name: np.empty(sample_count, dtype=np.float64)
        for name in (
            "mpjpe_mm",
            "pa_mpjpe_mm",
            "pve_mm",
            "pelvis_aligned_pve_mm",
        )
    }
    pred_camera = np.empty((sample_count, 14, 3), dtype=np.float64)
    gt_camera = np.empty((sample_count, 14, 3), dtype=np.float64)

    for start in range(0, sample_count, smpl_batch_size):
        end = min(start + smpl_batch_size, sample_count)
        selection = slice(start, end)
        predicted_rotation = torch.from_numpy(pred_rotmat[selection]).to(device)
        predicted_betas = torch.from_numpy(pred_betas[selection]).to(device)
        target_pose = torch.from_numpy(gt_pose[selection]).to(device)
        target_betas = torch.from_numpy(gt_betas[selection]).to(device)
        genders = torch.from_numpy(gender_codes[selection]).to(device)
        with (
            torch.inference_mode(),
            torch.autocast(device_type=device.type, enabled=False),
        ):
            pred_vertices = smpl_models[2](
                global_orient=predicted_rotation[:, :1],
                body_pose=predicted_rotation[:, 1:],
                betas=predicted_betas,
                pose2rot=False,
            ).vertices
            gt_vertices = torch.empty_like(pred_vertices)
            filled = torch.zeros(len(genders), dtype=torch.bool, device=device)
            target_rotation = axis_angle_to_matrix(target_pose.reshape(-1, 24, 3))
            camera_batch = torch.from_numpy(camera_rotations[selection]).to(device)
            target_rotation[:, 0] = camera_batch @ target_rotation[:, 0]
            for gender_code in (0, 1, 2):
                indices = torch.nonzero(
                    genders == gender_code, as_tuple=False
                ).flatten()
                if not len(indices):
                    continue
                rotations = target_rotation.index_select(0, indices)
                vertices = smpl_models[gender_code](
                    global_orient=rotations[:, :1],
                    body_pose=rotations[:, 1:],
                    betas=target_betas.index_select(0, indices),
                    pose2rot=False,
                ).vertices
                gt_vertices.index_copy_(0, indices, vertices)
                filled.index_fill_(0, indices, True)
            if not bool(filled.all()):
                raise RuntimeError("Failed to create every gendered GT SMPL mesh")
            pred_j14, pred_pelvis = pelvis_aligned_j14(pred_vertices, regressor)
            gt_j14, gt_pelvis = pelvis_aligned_j14(gt_vertices, regressor)
            metric_arrays["mpjpe_mm"][selection] = (
                mean_euclidean_error(pred_j14, gt_j14).cpu().numpy() * 1000.0
            )
            metric_arrays["pa_mpjpe_mm"][selection] = (
                pa_mpjpe_per_sample(pred_j14, gt_j14).cpu().numpy() * 1000.0
            )
            metric_arrays["pve_mm"][selection] = (
                mean_euclidean_error(pred_vertices, gt_vertices).cpu().numpy() * 1000.0
            )
            metric_arrays["pelvis_aligned_pve_mm"][selection] = (
                mean_euclidean_error(
                    pred_vertices - pred_pelvis, gt_vertices - gt_pelvis
                )
                .cpu()
                .numpy()
                * 1000.0
            )
            pred_j14 = hip_centered_j14(pred_j14)
            gt_j14 = hip_centered_j14(gt_j14)
            pred_camera[selection] = pred_j14.cpu().numpy()
            gt_camera[selection] = gt_j14.cpu().numpy()
        if end % (smpl_batch_size * 25) == 0 or end == sample_count:
            print(
                json.dumps(
                    {
                        "phase": "temporal_smpl_metrics",
                        "frames": end,
                        "total": sample_count,
                    }
                ),
                flush=True,
            )
    if not all(np.isfinite(values).all() for values in metric_arrays.values()):
        raise FloatingPointError("Temporal geometry metrics contain non-finite values")
    return metric_arrays, pred_camera, gt_camera


def main() -> None:
    args = parse_args()
    if args.output.suffix.lower() != ".json":
        raise ValueError("output must end in .json")
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        raise FileExistsError("Refusing to overwrite an existing geometry evaluation")
    if args.batch_size <= 0 or args.smpl_batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Invalid batch size or worker count")
    if not math.isfinite(args.fps) or args.fps <= 0:
        raise ValueError("fps must be positive")
    started = time.monotonic()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
    clip_length = validate_checkpoint_protocol(checkpoint, args.split)
    target_coordinate_system = "camera"
    amp_dtype = checkpoint["args"].get("amp_dtype", "bf16")
    development_only = bool(
        checkpoint.get("development_only", False)
        or checkpoint["args"].get("smoke_max_steps", 0)
    )
    config = GeometryDecoderConfig(**checkpoint["decoder_config"])
    decoder = HumanGeometryQueryTransformer(config).to(device).eval()
    decoder.load_state_dict(checkpoint["decoder"], strict=True)

    try:
        payload = torch.load(args.feature_cache, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(args.feature_cache, map_location="cpu")
    validate_feature_compatibility(checkpoint, payload)
    audit = validate_cache_and_alignment(payload, args.data_root, args.split)
    dataset = CenteredFrameClipDataset(payload, clip_length)
    pred_rotmat, pred_betas = run_temporal_predictions(
        decoder, dataset, args.batch_size, args.num_workers, device, amp_dtype
    )
    metric_arrays, pred_camera, gt_camera = compute_geometry_metrics(
        pred_rotmat,
        pred_betas,
        payload,
        args.smpl_model_root,
        args.joint_regressor,
        args.smpl_batch_size,
        device,
        target_coordinate_system,
    )
    metadata = payload["metadata"]
    sequences = [str(record["sequence"]) for record in metadata]
    person_ids = [int(record["person_id"]) for record in metadata]
    frame_ids = [int(record["frame_id"]) for record in metadata]
    acceleration = sequence_acceleration_error(
        pred_camera, gt_camera, sequences, person_ids, frame_ids
    )
    expected_triplets = int(audit["expected_acceleration_triplets"])
    if len(acceleration.errors) != expected_triplets:
        raise RuntimeError(
            f"Temporal {args.split} has {len(acceleration.errors)} acceleration "
            f"triplets, expected {expected_triplets}"
        )
    if not len(acceleration.errors):
        raise RuntimeError(f"Temporal {args.split} has no valid acceleration triplets")
    if (
        args.split == "test"
        and expected_triplets != EXPECTED_TEST_ACCELERATION_TRIPLETS
    ):
        raise RuntimeError("Canonical test acceleration count changed unexpectedly")
    acceleration_mm = acceleration.errors * 1000.0
    acceleration_physical = acceleration.errors * (args.fps**2)
    metrics = {name: float(values.mean()) for name, values in metric_arrays.items()}
    metrics.update(
        {
            "acceleration_error_mm_per_frame2": float(acceleration_mm.mean()),
            "acceleration_error_physical_m_per_s2": float(acceleration_physical.mean()),
        }
    )
    coverage = f"complete_{args.split}"
    crop_fallbacks = int(audit["crop_fallback_person_frames"])
    oracle_crop_frames = len(dataset) - crop_fallbacks
    result = {
        "status": "DIVIBODY_TEMPORAL_GEOMETRY_3DPW_EVALUATION_OK",
        "development_only": development_only,
        "coverage": coverage,
        "split": args.split,
        "evaluated_person_frames": len(dataset),
        "acceleration_valid_triplets": int(len(acceleration.errors)),
        "metrics": metrics,
        "checkpoint": {
            "path": str(args.checkpoint.expanduser().resolve()),
            "sha256": sha256_file(args.checkpoint),
            "epoch": int(checkpoint["epoch"]),
            "best_validation_mpjpe_mm": (
                float(checkpoint["best_val"]) if "best_val" in checkpoint else None
            ),
            "selection_metric": checkpoint.get("selection_metric"),
            "checkpoint_selection": checkpoint.get("checkpoint_selection"),
            "format": checkpoint["format"],
            "target_coordinate_system": target_coordinate_system,
            "pose_protocol": checkpoint["pose_protocol"],
            "development_only": development_only,
        },
        "feature_cache": {
            "path": str(args.feature_cache.expanduser().resolve()),
            "sha256": (
                None if args.skip_cache_hash else sha256_file(args.feature_cache)
            ),
            "signature": payload["signature"],
        },
        "dataset_preflight": audit,
        "temporal_stitch": dataset.stats,
        "per_sequence": per_sequence_summary(sequences, metric_arrays),
        "per_track_acceleration": [
            {
                **record,
                "mean_error_mm_per_frame2": (
                    None
                    if record["mean_error_m_per_frame2"] is None
                    else float(record["mean_error_m_per_frame2"]) * 1000.0
                ),
            }
            for record in acceleration.per_track
        ],
        "protocol": {
            "id": (
                f"3dpw_temporal{clip_length}_centered_"
                "oracle_track_camera_smpl24_geometry_spin14_v3"
            ),
            "context": (
                f"{clip_length} consecutive frames from the same sequence/person run"
            ),
            "stitch": "one centered-context prediction per real frame; no test-time averaging",
            "short_runs": "edge-repeat only within the same person/run",
            "causal": False,
            "oracle_track": True,
            "test_used_for_model_selection": checkpoint.get(
                "test_used_for_model_selection"
            ),
            "test_set_previously_observed": checkpoint.get(
                "test_set_previously_observed"
            ),
            "evaluator_uses_targets_for_prediction": False,
            "decoder_amp_dtype": amp_dtype,
            "joint_topology": "H36M-regressed J14 from 6890-vertex SMPL",
            "h36m_to_j14": list(H36M_TO_J14),
            "mpjpe": "H36M-root-aligned J14 mean Euclidean error",
            "pa_mpjpe": "per-frame scale+rotation+translation Procrustes J14 error",
            "pve": "direct corresponding neutral-pred/gendered-GT SMPL vertex error",
            "acceleration": (
                "VIBE-hip-rooted camera-frame J14 consecutive second difference; "
                "poses already generated in camera coordinates"
            ),
            "crop": (
                "oracle 3DPW poses2d person crop when valid; deterministic "
                "whole-image fallback otherwise"
            ),
            "oracle_crop_person_frames": oracle_crop_frames,
            "whole_image_fallback_person_frames": crop_fallbacks,
            "prediction_pose": "all 24 SMPL rotation matrices; no padded end joints",
            "pose_coordinate_system": target_coordinate_system,
            "camera_transform": "GT root only: R_camera_from_world @ R_global_world",
            "prediction_camera_transform": "none; already camera-aligned",
        },
        "assets": {
            "joint_regressor": str(args.joint_regressor.expanduser().resolve()),
            "joint_regressor_sha256": sha256_file(args.joint_regressor),
            "smpl_model_root": str(args.smpl_model_root.expanduser().resolve()),
            "smpl_model_sha256": {
                gender: sha256_file(args.smpl_model_root / f"SMPL_{gender.upper()}.pkl")
                for gender in ("neutral", "male", "female")
            },
        },
        "raw_arrays": str(args.output.with_suffix(".npz").resolve()),
        "runtime_seconds": time.monotonic() - started,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "limitations": [
            "Evaluation uses oracle person tracks, future frames, and poses2d crops where valid; it is offline and non-causal.",
            "3DPW has no hand or face ground truth; all metrics are body-only.",
            "Only the root is camera-frame; the remaining 23 SMPL rotations are local joint rotations.",
            "No test-time target fitting, gender input, camera correction, or target-dependent prediction is used.",
        ]
        + (
            [
                f"Invalid poses2d boxes caused deterministic whole-image fallback for {crop_fallbacks} person-frames."
            ]
            if crop_fallbacks
            else []
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        sequence=np.asarray(sequences),
        person_id=np.asarray(person_ids, dtype=np.int64),
        frame_id=np.asarray(frame_ids, dtype=np.int64),
        pred_rotmat24_camera=pred_rotmat,
        pred_betas=pred_betas,
        pred_camera_root_aligned_j14=pred_camera.astype(np.float32),
        gt_camera_root_aligned_j14=gt_camera.astype(np.float32),
        acceleration_center_sample_index=acceleration.center_sample_indices,
        acceleration_error_mm_per_frame2=acceleration_mm,
        **metric_arrays,
    )
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
