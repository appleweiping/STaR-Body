"""Unified full-3DPW benchmark for HMR 2.0a, HMR 2.0b, and DiviBody v6.

This additive evaluator does not edit any existing model/evaluator.  HMR models
share one strict image/GT index and the exact same SMPL/H36M-J14 metric code.
The validation-selected DiviBody v6 checkpoint is re-evaluated by the existing
overlap evaluator, then all three freshly produced JSON results are aggregated.

The HMR prediction caches are intentionally separate from metrics.  They make
the expensive ViT-H inference resumable and allow metric-protocol audits without
running the image encoder again.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

try:  # Package execution: python -m divibody_training...
    from .geometry_metrics_3dpw import (
        hip_centered_j14,
        mean_euclidean_error,
        pa_mpjpe_per_sample,
        pelvis_aligned_j14,
        sequence_acceleration_error,
    )
    from .real_3dpw_dataset import Real3DPWDataset, _load_annotation, _to_numpy
except ImportError:  # Direct execution from the divibody_training directory.
    from geometry_metrics_3dpw import (
        hip_centered_j14,
        mean_euclidean_error,
        pa_mpjpe_per_sample,
        pelvis_aligned_j14,
        sequence_acceleration_error,
    )
    from real_3dpw_dataset import Real3DPWDataset, _load_annotation, _to_numpy


FORMAT = "unified-hmr2-divibody-3dpw-v1"
PREDICTION_FORMAT = "unified-hmr2-3dpw-predictions-v1"
EXPECTED_TEST_PERSON_FRAMES = 35_515
EXPECTED_TEST_ACCELERATION_TRIPLETS = 34_752
EXPECTED_TEST_SEQUENCES = 24
EXPECTED_TEST_TRACKS = 37
HMR_IMAGE_SIZE = 256
HMR_TARGET_ASPECT = 192.0 / 256.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("hmr2a", "hmr2b", "divibody", "aggregate", "all"),
        default="all",
    )
    parser.add_argument("--hmr-source-root", type=Path, required=True)
    parser.add_argument("--hmr2a-checkpoint", type=Path, required=True)
    parser.add_argument("--hmr2b-checkpoint", type=Path, required=True)
    parser.add_argument("--hmr-model-config", type=Path, required=True)
    parser.add_argument("--hmr-data-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--smpl-model-root", type=Path, required=True)
    parser.add_argument("--joint-regressor", type=Path, required=True)
    parser.add_argument("--divibody-cache", type=Path, required=True)
    parser.add_argument("--divibody-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hmr-batch-size", type=int, default=8)
    parser.add_argument("--metric-batch-size", type=int, default=128)
    parser.add_argument("--divibody-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every-batches", type=int, default=100)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--allow-incomplete-test", action="store_true")
    parser.add_argument("--amp-dtype", choices=("bf16", "none"), default="bf16")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_gender(value: Any) -> int:
    text = str(value).strip().lower()
    if text.startswith("m"):
        return 0
    if text.startswith("f"):
        return 1
    raise ValueError(f"Unsupported 3DPW gender: {value!r}")


def hmr_bbox_size(bbox_xyxy: Iterable[float]) -> tuple[float, float, float]:
    """Return official-style centre and square pre-crop size for HMR2 ViT.

    HMR2 first expands a detection to the 192:256 model aspect ratio, takes the
    maximum expanded side as a square 256 crop, and finally removes 32 pixels
    from each side inside the model.  This function mirrors that geometry.
    """

    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    width, height = x2 - x1, y2 - y1
    if width <= 1.0 or height <= 1.0:
        raise ValueError(f"Degenerate HMR bbox: {(x1, y1, x2, y2)}")
    if width / height < HMR_TARGET_ASPECT:
        width = height * HMR_TARGET_ASPECT
    else:
        height = width / HMR_TARGET_ASPECT
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, max(width, height)


class UnifiedHMR3DPWDataset(Real3DPWDataset):
    """Strict 3DPW person frames with HMR2-compatible RGB and full GT."""

    def _build_index(self) -> list[Any]:
        requested_limit = self.max_samples
        self.max_samples = None
        try:
            candidates = super()._build_index()
        finally:
            self.max_samples = requested_limit
        strict = [
            sample
            for sample in candidates
            if Path(sample.image_path).stem == f"image_{sample.frame_index:05d}"
        ]
        return strict if requested_limit is None else strict[:requested_limit]

    def __getitem__(self, index: int) -> dict[str, Any]:
        reference = self.samples[index]
        annotation = _load_annotation(reference.annotation_path)
        pose = _to_numpy(
            annotation["poses"][reference.person_index][reference.frame_index]
        ).reshape(-1)
        betas = _to_numpy(annotation["betas"][reference.person_index]).reshape(-1)
        camera = _to_numpy(annotation["cam_poses"][reference.frame_index])
        if pose.size < 72 or betas.size < 10 or camera.shape != (4, 4):
            raise ValueError("Invalid 3DPW SMPL/camera target")
        if not (
            np.isfinite(pose[:72]).all()
            and np.isfinite(betas[:10]).all()
            and np.isfinite(camera).all()
        ):
            raise ValueError("Non-finite 3DPW target")

        with Image.open(reference.image_path) as image_file:
            image = image_file.convert("RGB")
            width, height = image.size
            bbox = self._person_bbox(
                annotation,
                reference.person_index,
                reference.frame_index,
                width,
                height,
            )
            if bbox is None:
                bbox = (0.0, 0.0, float(width), float(height))
            center_x, center_y, side = hmr_bbox_size(bbox)
            crop = (
                center_x - side / 2.0,
                center_y - side / 2.0,
                center_x + side / 2.0,
                center_y + side / 2.0,
            )
            image = image.crop(crop)
            try:
                resampling = Image.Resampling.BILINEAR
            except AttributeError:
                resampling = Image.BILINEAR
            image = image.resize((HMR_IMAGE_SIZE, HMR_IMAGE_SIZE), resampling)
            rgb = np.asarray(image, dtype=np.float32) / 255.0

        gender = annotation.get("genders", ())[reference.person_index]
        return {
            "image": torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1),
            "full_pose": torch.from_numpy(pose[:72].astype(np.float32, copy=True)),
            "betas": torch.from_numpy(betas[:10].astype(np.float32, copy=True)),
            "gender_code": torch.tensor(_normalise_gender(gender), dtype=torch.long),
            "cam_pose": torch.from_numpy(camera.astype(np.float32, copy=True)),
            "sequence": reference.sequence,
            "person_id": torch.tensor(reference.person_index, dtype=torch.long),
            "frame_id": torch.tensor(reference.frame_index, dtype=torch.long),
            "bbox_xyxy": torch.tensor(bbox, dtype=torch.float32),
            "hmr_crop_side": torch.tensor(side, dtype=torch.float32),
        }


def dataset_signature(dataset: UnifiedHMR3DPWDataset) -> dict[str, Any]:
    identities = [
        (sample.sequence, sample.person_index, sample.frame_index)
        for sample in dataset.samples
    ]
    return {
        "split": dataset.split,
        "person_frames": len(dataset),
        "sequences": len({value[0] for value in identities}),
        "tracks": len({value[:2] for value in identities}),
        "unique_identities": len(set(identities)),
        "crop": (
            "oracle 3DPW poses2d bbox + 25% margin; expand to 192:256 HMR2 "
            "aspect; square 256 pre-crop; model centre-width crop to 192"
        ),
        "strict_rgb_resolver": "image_{annotation_frame_index:05d}",
    }


def validate_complete_test(signature: dict[str, Any], allow_incomplete: bool) -> None:
    expected = {
        "person_frames": EXPECTED_TEST_PERSON_FRAMES,
        "sequences": EXPECTED_TEST_SEQUENCES,
        "tracks": EXPECTED_TEST_TRACKS,
        "unique_identities": EXPECTED_TEST_PERSON_FRAMES,
    }
    failures = [
        f"{key}={signature[key]} expected={value}"
        for key, value in expected.items()
        if int(signature[key]) != value
    ]
    if failures and not allow_incomplete:
        raise RuntimeError("Incomplete 3DPW test: " + "; ".join(failures))


def load_hmr_model(args: argparse.Namespace, checkpoint: Path) -> torch.nn.Module:
    source_text = str(args.hmr_source_root.expanduser().resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    from hmr2.configs import get_config
    from hmr2.models import HMR2

    cfg = get_config(str(args.hmr_model_config), update_cachedir=False)
    cfg.defrost()
    cfg.MODEL.BBOX_SHAPE = [192, 256]
    cfg.SMPL.MODEL_PATH = str(args.hmr_data_root / "smpl")
    cfg.SMPL.JOINT_REGRESSOR_EXTRA = str(args.hmr_data_root / "SMPL_to_J19.pkl")
    cfg.SMPL.MEAN_PARAMS = str(args.hmr_data_root / "smpl_mean_params.npz")
    cfg.freeze()
    model = HMR2.load_from_checkpoint(
        str(checkpoint),
        strict=True,
        cfg=cfg,
        map_location="cpu",
        init_renderer=False,
    ).eval()
    model.requires_grad_(False)
    return model


def _empty_prediction_buffers(count: int) -> dict[str, Any]:
    return {
        "rotmat": torch.empty(count, 24, 3, 3, dtype=torch.float32),
        "betas": torch.empty(count, 10, dtype=torch.float32),
        "camera": torch.empty(count, 3, dtype=torch.float32),
        "gt_full_pose": torch.empty(count, 72, dtype=torch.float32),
        "gt_betas": torch.empty(count, 10, dtype=torch.float32),
        "gt_gender": torch.empty(count, dtype=torch.long),
        "gt_cam_pose": torch.empty(count, 4, 4, dtype=torch.float32),
        "person_id": torch.empty(count, dtype=torch.long),
        "frame_id": torch.empty(count, dtype=torch.long),
        "sequences": [None] * count,
    }


def _save_partial(
    path: Path,
    *,
    model_name: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    signature: dict[str, Any],
    completed: int,
    buffers: dict[str, Any],
) -> None:
    payload = {
        "format": PREDICTION_FORMAT,
        "model_name": model_name,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_signature": signature,
        "completed": completed,
        **{
            key: (value[:completed] if torch.is_tensor(value) else value[:completed])
            for key, value in buffers.items()
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def predict_hmr(
    args: argparse.Namespace,
    *,
    model_name: str,
    checkpoint: Path,
    dataset: UnifiedHMR3DPWDataset,
    prediction_path: Path,
) -> dict[str, Any]:
    signature = dataset_signature(dataset)
    checkpoint_sha = sha256_file(checkpoint)
    partial_path = prediction_path.with_suffix(".partial.pt")
    count = len(dataset)
    buffers = _empty_prediction_buffers(count)
    completed = 0
    if prediction_path.is_file():
        complete = torch.load(prediction_path, map_location="cpu", weights_only=False)
        if (
            complete.get("format") == PREDICTION_FORMAT
            and complete.get("model_name") == model_name
            and complete.get("checkpoint_sha256") == checkpoint_sha
            and int(complete.get("completed", -1)) == count
            and complete.get("dataset_signature") == signature
        ):
            return complete
        raise RuntimeError(f"Existing incompatible prediction cache: {prediction_path}")
    if partial_path.is_file():
        partial = torch.load(partial_path, map_location="cpu", weights_only=False)
        if not (
            partial.get("format") == PREDICTION_FORMAT
            and partial.get("model_name") == model_name
            and partial.get("checkpoint_sha256") == checkpoint_sha
            and partial.get("dataset_signature") == signature
        ):
            raise RuntimeError(f"Incompatible partial cache: {partial_path}")
        completed = int(partial["completed"])
        for key, buffer in buffers.items():
            if torch.is_tensor(buffer):
                buffer[:completed].copy_(partial[key])
            else:
                buffer[:completed] = partial[key]
        print(json.dumps({"phase": "resume_hmr", "model": model_name,
                          "completed": completed, "total": count}), flush=True)

    device = torch.device(args.device)
    model = load_hmr_model(args, checkpoint).to(device)
    loader = DataLoader(
        dataset,
        batch_size=args.hmr_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).reshape(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).reshape(1, 3, 1, 1)
    started = time.monotonic()
    cursor = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            batch_count = int(batch["image"].shape[0])
            stop = cursor + batch_count
            if stop <= completed:
                cursor = stop
                continue
            if cursor < completed:
                raise RuntimeError("Partial cache does not end at a batch boundary")
            images = batch["image"].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            images = (images - mean) / std
            autocast_enabled = device.type == "cuda" and args.amp_dtype == "bf16"
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                feature_map = model.backbone(images[:, :, :, 32:-32])
                smpl_params, camera, _ = model.smpl_head(feature_map)
            rotmat = torch.cat(
                (smpl_params["global_orient"], smpl_params["body_pose"]), dim=1
            )
            values = {
                "rotmat": rotmat.float().cpu(),
                "betas": smpl_params["betas"].float().cpu(),
                "camera": camera.float().cpu(),
                "gt_full_pose": batch["full_pose"],
                "gt_betas": batch["betas"],
                "gt_gender": batch["gender_code"],
                "gt_cam_pose": batch["cam_pose"],
                "person_id": batch["person_id"],
                "frame_id": batch["frame_id"],
            }
            for key, value in values.items():
                buffers[key][cursor:stop].copy_(value)
            buffers["sequences"][cursor:stop] = list(batch["sequence"])
            cursor = stop
            completed = cursor
            if (
                (batch_index + 1) % args.save_every_batches == 0
                or completed == count
            ):
                _save_partial(
                    partial_path,
                    model_name=model_name,
                    checkpoint=checkpoint,
                    checkpoint_sha256=checkpoint_sha,
                    signature=signature,
                    completed=completed,
                    buffers=buffers,
                )
                print(
                    json.dumps(
                        {
                            "phase": "hmr_inference",
                            "model": model_name,
                            "completed": completed,
                            "total": count,
                            "seconds": time.monotonic() - started,
                            "peak_cuda_mb": (
                                torch.cuda.max_memory_allocated(device) / 1024**2
                                if device.type == "cuda" else 0.0
                            ),
                        }
                    ),
                    flush=True,
                )
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    os.replace(partial_path, prediction_path)
    return torch.load(prediction_path, map_location="cpu", weights_only=False)


def create_smpl_models(root: Path, device: torch.device) -> dict[int, torch.nn.Module]:
    import smplx

    expected = [root / f"SMPL_{gender}.pkl" for gender in ("NEUTRAL", "MALE", "FEMALE")]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing SMPL assets: " + ", ".join(missing))
    models = {
        code: smplx.SMPL(
            str(root), gender=gender, ext="pkl",
            create_betas=False, create_global_orient=False,
            create_body_pose=False, create_transl=False,
        ).eval().to(device)
        for code, gender in ((2, "neutral"), (0, "male"), (1, "female"))
    }
    for model in models.values():
        model.requires_grad_(False)
    return models


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    from smplx.lbs import batch_rodrigues

    original = axis_angle.shape[:-1]
    return batch_rodrigues(axis_angle.reshape(-1, 3)).reshape(*original, 3, 3)


def smpl_vertices(
    model: torch.nn.Module, rotations: torch.Tensor, betas: torch.Tensor
) -> torch.Tensor:
    return model(
        global_orient=rotations[:, [0]],
        body_pose=rotations[:, 1:],
        betas=betas,
        pose2rot=False,
    ).vertices


def gendered_vertices(
    models: dict[int, torch.nn.Module],
    rotations: torch.Tensor,
    betas: torch.Tensor,
    genders: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty(
        rotations.shape[0], 6890, 3, device=rotations.device, dtype=torch.float32
    )
    for gender in (0, 1):
        indices = torch.nonzero(genders == gender, as_tuple=False).flatten()
        if len(indices):
            output.index_copy_(
                0,
                indices,
                smpl_vertices(
                    models[gender],
                    rotations.index_select(0, indices),
                    betas.index_select(0, indices),
                ),
            )
    return output


def evaluate_hmr_predictions(
    args: argparse.Namespace, prediction: dict[str, Any], output_path: Path
) -> dict[str, Any]:
    count = int(prediction["completed"])
    if count != int(prediction["dataset_signature"]["person_frames"]):
        raise RuntimeError("Prediction cache is incomplete")
    device = torch.device(args.device)
    models = create_smpl_models(args.smpl_model_root, device)
    regressor_np = np.load(args.joint_regressor)
    if regressor_np.shape != (17, 6890) or not np.isfinite(regressor_np).all():
        raise ValueError("Invalid H36M joint regressor")
    regressor = torch.as_tensor(regressor_np, device=device, dtype=torch.float32)
    metrics: dict[str, list[np.ndarray]] = {
        name: [] for name in ("mpjpe_mm", "pa_mpjpe_mm", "pve_mm")
    }
    per_sequence: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: {name: [] for name in metrics}
    )
    pred_motion: list[np.ndarray] = []
    target_motion: list[np.ndarray] = []
    for start in range(0, count, args.metric_batch_size):
        stop = min(start + args.metric_batch_size, count)
        pred_rot = prediction["rotmat"][start:stop].to(device)
        pred_betas = prediction["betas"][start:stop].to(device)
        full_pose = prediction["gt_full_pose"][start:stop].to(device)
        gt_rot = axis_angle_to_matrix(full_pose.reshape(-1, 24, 3))
        camera_rotation = prediction["gt_cam_pose"][start:stop, :3, :3].to(device)
        gt_rot[:, 0] = camera_rotation @ gt_rot[:, 0]
        gt_betas = prediction["gt_betas"][start:stop].to(device)
        genders = prediction["gt_gender"][start:stop].to(device)
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            pred_vertices = smpl_vertices(models[2], pred_rot, pred_betas)
            gt_vertices = gendered_vertices(models, gt_rot, gt_betas, genders)
            pred_joints, _ = pelvis_aligned_j14(pred_vertices, regressor)
            gt_joints, _ = pelvis_aligned_j14(gt_vertices, regressor)
            values = {
                "mpjpe_mm": mean_euclidean_error(pred_joints, gt_joints) * 1000.0,
                "pa_mpjpe_mm": pa_mpjpe_per_sample(pred_joints, gt_joints) * 1000.0,
                "pve_mm": mean_euclidean_error(pred_vertices, gt_vertices) * 1000.0,
            }
            pred_motion.extend(hip_centered_j14(pred_joints).cpu().numpy())
            target_motion.extend(hip_centered_j14(gt_joints).cpu().numpy())
        sequences = prediction["sequences"][start:stop]
        for name, tensor in values.items():
            array = tensor.cpu().numpy()
            metrics[name].append(array)
            for sequence in sorted(set(sequences)):
                mask = np.asarray([value == sequence for value in sequences])
                per_sequence[sequence][name].append(array[mask])

    acceleration = sequence_acceleration_error(
        np.asarray(pred_motion),
        np.asarray(target_motion),
        prediction["sequences"],
        prediction["person_id"].tolist(),
        prediction["frame_id"].tolist(),
    )
    if (
        not args.allow_incomplete_test
        and len(acceleration.errors) != EXPECTED_TEST_ACCELERATION_TRIPLETS
    ):
        raise RuntimeError(
            f"Acceleration coverage {len(acceleration.errors)} != "
            f"{EXPECTED_TEST_ACCELERATION_TRIPLETS}"
        )
    result = {
        "status": "UNIFIED_HMR2_3DPW_EVALUATION_OK",
        "format": FORMAT,
        "model": prediction["model_name"],
        "metrics": {
            **{
                name: float(np.concatenate(chunks).astype(np.float64).mean())
                for name, chunks in metrics.items()
            },
            "acceleration_error_mm_per_frame2": float(
                acceleration.errors.astype(np.float64).mean() * 1000.0
            ),
            "acceleration_error_physical_m_per_s2": float(
                acceleration.errors.astype(np.float64).mean() * args.fps**2
            ),
        },
        "coverage": {
            **prediction["dataset_signature"],
            "acceleration_valid_triplets": int(len(acceleration.errors)),
        },
        "checkpoint": {
            "path": prediction["checkpoint"],
            "sha256": prediction["checkpoint_sha256"],
        },
        "inference_precision": "fp32" if args.amp_dtype == "none" else "bf16",
        "protocol": metric_protocol(),
        "per_sequence": {
            sequence: {
                name: float(np.concatenate(chunks).astype(np.float64).mean())
                for name, chunks in values.items()
            }
            for sequence, values in sorted(per_sequence.items())
        },
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    del models
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def metric_protocol() -> dict[str, str]:
    return {
        "mpjpe": "H36M-regressed J14, H36M pelvis aligned, mean Euclidean error",
        "pa_mpjpe": "same J14 after per-frame scale+proper-rotation+translation Procrustes",
        "pve": "unaligned 6890-vertex neutral prediction versus gendered 3DPW GT",
        "acceleration": (
            "hip-midpoint-rooted J14 second-difference error; only same-person "
            "consecutive frame triplets"
        ),
        "distance_unit": "millimetres",
        "acceleration_unit": "millimetres/frame^2",
    }


def run_divibody(args: argparse.Namespace, output_path: Path) -> dict[str, Any]:
    script = Path(__file__).with_name("evaluate_continuous_geometry_v6_overlap.py")
    command = [
        sys.executable,
        str(script),
        "--cache", str(args.divibody_cache),
        "--data-root", str(args.data_root),
        "--checkpoint", str(args.divibody_checkpoint),
        "--smpl-model-root", str(args.smpl_model_root),
        "--joint-regressor", str(args.joint_regressor),
        "--output", str(output_path),
        "--batch-size", str(args.divibody_batch_size),
        "--device", args.device,
        "--amp-dtype", args.amp_dtype,
        "--fusion-weighting", "triangular",
        "--fps", str(args.fps),
    ]
    subprocess.run(command, check=True)
    return json.loads(output_path.read_text(encoding="utf-8"))


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "HMR 2.0a": args.output_dir / "hmr2a_test.json",
        "HMR 2.0b": args.output_dir / "hmr2b_test.json",
        "DiviBody v6": args.output_dir / "divibody_v6_test.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Cannot aggregate missing results: " + ", ".join(missing))
    raw = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    rows = {
        name: {
            key: float(result["metrics"][key])
            for key in (
                "mpjpe_mm", "pa_mpjpe_mm", "pve_mm",
                "acceleration_error_mm_per_frame2",
            )
        }
        for name, result in raw.items()
    }
    coverage = {
        name: int(
            result["coverage"].get(
                "person_frames",
                result["coverage"].get("unique_person_frames", -1),
            )
        )
        for name, result in raw.items()
    }
    if not args.allow_incomplete_test and set(coverage.values()) != {
        EXPECTED_TEST_PERSON_FRAMES
    }:
        raise RuntimeError(f"Cross-model coverage mismatch: {coverage}")
    result = {
        "status": "UNIFIED_HMR2_DIVIBODY_FULL_3DPW_OK",
        "format": FORMAT,
        "models": rows,
        "coverage_person_frames": coverage,
        "protocol": metric_protocol(),
        "inference_precision": "fp32" if args.amp_dtype == "none" else "bf16",
        "lower_is_better": True,
        "source_results": {name: str(path.resolve()) for name, path in paths.items()},
        "test_used_for_model_selection": False,
    }
    output = args.output_dir / "unified_full_3dpw_test.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    args = parse_args()
    if min(
        args.hmr_batch_size,
        args.metric_batch_size,
        args.divibody_batch_size,
        args.num_workers + 1,
        args.save_every_batches,
    ) <= 0:
        raise ValueError("Batch/worker/save arguments must be valid positive values")
    if args.fps <= 0 or args.max_samples < 0:
        raise ValueError("fps must be positive and max-samples non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    limit: Optional[int] = args.max_samples or None
    dataset = UnifiedHMR3DPWDataset(
        root=args.data_root,
        split="test",
        image_size=HMR_IMAGE_SIZE,
        frame_stride=1,
        max_samples=limit,
        crop_to_person=True,
        crop_margin=0.25,
    )
    signature = dataset_signature(dataset)
    validate_complete_test(signature, args.allow_incomplete_test)
    (args.output_dir / "dataset_audit.json").write_text(
        json.dumps(signature, indent=2), encoding="utf-8"
    )

    stages = ("hmr2a", "hmr2b", "divibody", "aggregate") if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage in ("hmr2a", "hmr2b"):
            model_name = "HMR 2.0a" if stage == "hmr2a" else "HMR 2.0b"
            checkpoint = (
                args.hmr2a_checkpoint if stage == "hmr2a" else args.hmr2b_checkpoint
            )
            prediction = predict_hmr(
                args,
                model_name=model_name,
                checkpoint=checkpoint,
                dataset=dataset,
                prediction_path=args.output_dir / f"{stage}_predictions.pt",
            )
            result = evaluate_hmr_predictions(
                args, prediction, args.output_dir / f"{stage}_test.json"
            )
            print(json.dumps(result), flush=True)
        elif stage == "divibody":
            if args.max_samples:
                raise ValueError("DiviBody formal cache cannot use --max-samples")
            result = run_divibody(args, args.output_dir / "divibody_v6_test.json")
            print(json.dumps(result), flush=True)
        else:
            aggregate(args)


if __name__ == "__main__":
    main()
