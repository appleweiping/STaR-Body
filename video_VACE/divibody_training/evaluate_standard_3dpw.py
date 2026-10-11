"""Evaluate a trained DiviBody decoder with standard SMPL 3DPW metrics.

This is a new, additive evaluator.  It leaves every original VACE and DiviBody
model file unchanged.  The primary protocol follows the common SPIN/HMR
14-joint convention and reports sequence-safe VIBE-style acceleration error.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from geometry_metrics_3dpw import (
    H36M_TO_J14,
    hip_centered_j14,
    mean_euclidean_error,
    pa_mpjpe_per_sample,
    pelvis_aligned_j14,
    rotate_points,
    sequence_acceleration_error,
)
from models.human_decoder import HumanDecoderConfig, HumanQueryTransformer
from real_3dpw_dataset import (
    Real3DPWDataset,
    _find_split_directory,
    _load_annotation,
)
from real_vace_feature_encoder import FrozenVaceFeatureEncoder


EXPECTED_TEST_SEQUENCES = 24
EXPECTED_TEST_TRACKS = 37
EXPECTED_TEST_ANNOTATION_FRAMES = 26_240
EXPECTED_TEST_PERSON_FRAMES = 35_515
EXPECTED_TEST_UNIQUE_VALID_RGB_FRAMES = 24_547
EXPECTED_TEST_ACCELERATION_TRIPLETS = 34_752
EXPECTED_TEST_SEQUENCE_NAMES = frozenset(
    {
        "downtown_arguing_00",
        "downtown_bar_00",
        "downtown_bus_00",
        "downtown_cafe_00",
        "downtown_car_00",
        "downtown_crossStreets_00",
        "downtown_downstairs_00",
        "downtown_enterShop_00",
        "downtown_rampAndStairs_00",
        "downtown_runForBus_00",
        "downtown_runForBus_01",
        "downtown_sitOnStairs_00",
        "downtown_stairs_00",
        "downtown_upstairs_00",
        "downtown_walkBridge_01",
        "downtown_walking_00",
        "downtown_walkUphill_00",
        "downtown_warmWelcome_00",
        "downtown_weeklyMarket_00",
        "downtown_windowShopping_00",
        "flat_guitar_01",
        "flat_packBags_00",
        "office_phoneCall_00",
        "outdoors_fencing_01",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standard MPJPE/PA-MPJPE/PVE/acceleration evaluation on 3DPW"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vace-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--smpl-model-root",
        type=Path,
        required=True,
        help="Directory containing SMPL_NEUTRAL/MALE/FEMALE.pkl",
    )
    parser.add_argument("--joint-regressor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--allow-incomplete-test",
        action="store_true",
        help="Development only: do not use the resulting subset as a test benchmark",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_gender(value: Any) -> int:
    gender = str(value).strip().lower()
    if gender in ("m", "male"):
        return 0
    if gender in ("f", "female"):
        return 1
    raise ValueError(f"Unsupported or missing 3DPW gender label: {value!r}")


def _strict_frame_image_path(
    image_directory: Path, frame_index: int
) -> Optional[Path]:
    """Resolve only the official ``image_{annotation_index}`` RGB frame."""

    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = image_directory / f"image_{frame_index:05d}{suffix}"
        if candidate.is_file():
            return candidate
    return None


class Standard3DPWDataset(Real3DPWDataset):
    """Extend the existing image adapter with full SMPL pose and gender."""

    def _build_index(self) -> list[Any]:
        """Use only exact annotation-frame RGB names, never 60 Hz source IDs.

        The inherited adapter supports a source-frame fallback for imperfect
        exploratory datasets.  That is unsafe for a benchmark because
        ``source_frame_id`` addresses the 60 Hz IMU stream and can name a
        different 30 Hz RGB frame.  Build the complete parent index first,
        filter it strictly, and only then apply the requested sample limit.
        """

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
        if requested_limit is not None:
            strict = strict[:requested_limit]
        return strict

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = super().__getitem__(index)
        reference = self.samples[index]
        annotation = _load_annotation(reference.annotation_path)
        pose = np.asarray(
            annotation["poses"][reference.person_index][reference.frame_index],
            dtype=np.float32,
        ).reshape(-1)
        if pose.size < 72 or not np.isfinite(pose[:72]).all():
            raise ValueError(
                f"Invalid full SMPL pose at {reference.sequence}, "
                f"person={reference.person_index}, frame={reference.frame_index}"
            )
        genders = annotation.get("genders", ())
        if reference.person_index >= len(genders):
            raise ValueError(
                f"Missing gender at {reference.sequence}, "
                f"person={reference.person_index}"
            )
        gender = genders[reference.person_index]
        item["full_pose"] = torch.from_numpy(pose[:72].copy())
        item["gender_code"] = torch.tensor(
            _normalise_gender(gender), dtype=torch.long
        )
        return item


def audit_split(root: Path, split: str) -> Dict[str, Any]:
    """Count expected and image-backed samples before loading the 7 GB VACE model."""

    root = root.expanduser().resolve()
    split_directory = _find_split_directory(root, split)
    annotation_paths = sorted(split_directory.glob("*.pkl"))
    expected_person_frames = 0
    available_person_frames = 0
    total_frames = 0
    track_count = 0
    required_image_keys: set[tuple[str, int]] = set()
    available_image_keys: set[tuple[str, int]] = set()
    missing_examples: list[dict[str, Any]] = []
    invalid_gender_tracks: list[dict[str, Any]] = []
    invalid_camera_frames: list[dict[str, Any]] = []
    invalid_camera_frame_count = 0
    discovered_sequence_names: list[str] = []
    valid_track_frames: dict[tuple[str, int], list[int]] = defaultdict(list)

    for annotation_path in annotation_paths:
        annotation = _load_annotation(str(annotation_path))
        sequence = str(annotation.get("sequence", annotation_path.stem))
        discovered_sequence_names.append(sequence)
        image_directory = root / "imageFiles" / sequence
        source_ids = np.asarray(annotation.get("img_frame_ids", ())).reshape(-1)
        camera_poses = np.asarray(annotation.get("cam_poses", ()))
        poses = annotation.get("poses", ())
        translations = annotation.get("trans", ())
        shapes = annotation.get("betas", ())
        validity = annotation.get("campose_valid", ())
        people = min(len(poses), len(translations), len(shapes), len(validity))
        genders = annotation.get("genders", ())
        for person_index in range(people):
            try:
                if person_index >= len(genders):
                    raise ValueError("missing")
                _normalise_gender(genders[person_index])
            except ValueError:
                invalid_gender_tracks.append(
                    {
                        "sequence": sequence,
                        "person_id": person_index,
                        "label": (
                            None
                            if person_index >= len(genders)
                            else str(genders[person_index])
                        ),
                    }
                )
        total_frames += len(source_ids)
        track_count += people

        for frame_index, source_frame_id_value in enumerate(source_ids):
            if frame_index >= len(camera_poses):
                continue
            camera_pose = np.asarray(camera_poses[frame_index])
            camera_valid = camera_pose.shape == (4, 4) and np.isfinite(
                camera_pose
            ).all()
            if camera_valid:
                rotation = camera_pose[:3, :3]
                camera_valid = bool(
                    np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                    and np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
                    and np.allclose(
                        camera_pose[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-7
                    )
                )
            if not camera_valid:
                invalid_camera_frame_count += 1
                if len(invalid_camera_frames) < 20:
                    invalid_camera_frames.append(
                        {"sequence": sequence, "frame_index": frame_index}
                    )
                continue
            source_frame_id = int(source_frame_id_value)
            valid_people = 0
            valid_person_ids: list[int] = []
            for person_index in range(people):
                valid = np.asarray(validity[person_index]).reshape(-1)
                person_pose = np.asarray(poses[person_index])
                person_translation = np.asarray(translations[person_index])
                person_shape = np.asarray(shapes[person_index]).reshape(-1)
                if (
                    frame_index >= len(valid)
                    or not bool(valid[frame_index])
                    or frame_index >= len(person_pose)
                    or frame_index >= len(person_translation)
                    or person_pose.shape[-1] < 72
                    or person_translation.shape[-1] < 3
                    or person_shape.size < 10
                ):
                    continue
                values = (
                    person_pose[frame_index, :72],
                    person_translation[frame_index, :3],
                    person_shape[:10],
                )
                if not all(np.isfinite(value).all() for value in values):
                    continue
                valid_people += 1
                valid_person_ids.append(person_index)
            if not valid_people:
                continue

            key = (sequence, frame_index)
            required_image_keys.add(key)
            for person_index in valid_person_ids:
                valid_track_frames[(sequence, person_index)].append(frame_index)
            image_path: Optional[Path] = _strict_frame_image_path(
                image_directory, frame_index
            )
            expected_person_frames += valid_people
            if image_path is not None:
                available_image_keys.add(key)
                available_person_frames += valid_people
            elif len(missing_examples) < 20:
                missing_examples.append(
                    {
                        "sequence": sequence,
                        "frame_index": frame_index,
                        "source_frame_id": source_frame_id,
                        "valid_people": valid_people,
                    }
                )

    expected_acceleration_triplets = 0
    for frames_list in valid_track_frames.values():
        frames = np.asarray(sorted(frames_list), dtype=np.int64)
        if len(frames) >= 3:
            expected_acceleration_triplets += int(
                (
                    (frames[1:-1] - frames[:-2] == 1)
                    & (frames[2:] - frames[1:-1] == 1)
                ).sum()
            )

    return {
        "split": split,
        "annotation_directory": str(split_directory),
        "sequences": len(annotation_paths),
        "sequence_names": sorted(discovered_sequence_names),
        "tracks": track_count,
        "annotation_frames": total_frames,
        "expected_valid_person_frames": expected_person_frames,
        "available_valid_person_frames": available_person_frames,
        "required_unique_rgb_frames": len(required_image_keys),
        "available_unique_rgb_frames": len(available_image_keys),
        "missing_unique_rgb_frames": len(required_image_keys - available_image_keys),
        "missing_examples": missing_examples,
        "invalid_gender_tracks": invalid_gender_tracks,
        "invalid_camera_frame_count": invalid_camera_frame_count,
        "invalid_camera_examples": invalid_camera_frames,
        "expected_acceleration_triplets": expected_acceleration_triplets,
    }


def enforce_complete_test(audit: Dict[str, Any], allow_incomplete: bool) -> None:
    if audit["split"] != "test" or allow_incomplete:
        return
    failures = []
    expected = {
        "sequences": EXPECTED_TEST_SEQUENCES,
        "tracks": EXPECTED_TEST_TRACKS,
        "annotation_frames": EXPECTED_TEST_ANNOTATION_FRAMES,
        "expected_valid_person_frames": EXPECTED_TEST_PERSON_FRAMES,
        "required_unique_rgb_frames": EXPECTED_TEST_UNIQUE_VALID_RGB_FRAMES,
        "expected_acceleration_triplets": EXPECTED_TEST_ACCELERATION_TRIPLETS,
    }
    for name, expected_value in expected.items():
        if int(audit[name]) != expected_value:
            failures.append(f"{name}={audit[name]} (expected {expected_value})")
    if int(audit["missing_unique_rgb_frames"]) != 0:
        failures.append(
            f"missing_unique_rgb_frames={audit['missing_unique_rgb_frames']}"
        )
    if int(audit["available_valid_person_frames"]) != int(
        audit["expected_valid_person_frames"]
    ):
        failures.append("not every valid person-frame has RGB")
    discovered_names = set(audit["sequence_names"])
    if discovered_names != EXPECTED_TEST_SEQUENCE_NAMES:
        failures.append(
            "test sequence set mismatch: "
            f"missing={sorted(EXPECTED_TEST_SEQUENCE_NAMES - discovered_names)}, "
            f"unexpected={sorted(discovered_names - EXPECTED_TEST_SEQUENCE_NAMES)}"
        )
    if audit["invalid_gender_tracks"]:
        failures.append(
            f"invalid_gender_tracks={len(audit['invalid_gender_tracks'])}"
        )
    if int(audit["invalid_camera_frame_count"]) != 0:
        failures.append(
            f"invalid_camera_frame_count={audit['invalid_camera_frame_count']}"
        )
    if failures:
        raise RuntimeError(
            "Incomplete 3DPW test split; refusing to publish a partial benchmark: "
            + "; ".join(failures)
            + "\n"
            + json.dumps(audit, indent=2, ensure_ascii=False)
        )


def enable_legacy_chumpy_compatibility() -> None:
    """Allow licensed SMPL v1 pickle files to load on modern Python/NumPy."""

    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    aliases = {
        "bool": np.bool_,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
        "unicode": str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def create_smpl_models(root: Path, device: torch.device) -> Dict[int, torch.nn.Module]:
    enable_legacy_chumpy_compatibility()
    try:
        import chumpy  # noqa: F401  # required while unpickling legacy SMPL files
        import smplx
    except ImportError as error:
        raise RuntimeError(
            "SMPL evaluation requires smplx and chumpy==0.70. Install chumpy "
            "with: pip install --no-build-isolation chumpy==0.70"
        ) from error

    root = root.expanduser().resolve()
    expected_files = {
        "neutral": root / "SMPL_NEUTRAL.pkl",
        "male": root / "SMPL_MALE.pkl",
        "female": root / "SMPL_FEMALE.pkl",
    }
    missing = [str(path) for path in expected_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing licensed SMPL files: " + ", ".join(missing))

    models: Dict[int, torch.nn.Module] = {}
    for code, gender in ((2, "neutral"), (0, "male"), (1, "female")):
        model = smplx.SMPL(
            str(root),
            gender=gender,
            ext="pkl",
            create_betas=False,
            create_global_orient=False,
            create_body_pose=False,
            create_transl=False,
        ).eval().to(device)
        models[code] = model
    neutral_faces = models[2].faces_tensor
    for code, label in ((0, "male"), (1, "female"), (2, "neutral")):
        model = models[code]
        if int(model.v_template.shape[0]) != 6890:
            raise ValueError(
                f"{label} SMPL has {model.v_template.shape[0]} vertices, expected 6890"
            )
        if not torch.equal(model.faces_tensor, neutral_faces):
            raise ValueError(f"{label} SMPL topology differs from neutral SMPL")
    return models


def validate_checkpoint_metadata(checkpoint: Dict[str, Any]) -> None:
    """Fail before loading large assets when a checkpoint is incompatible."""

    if checkpoint.get("format") != "divibody-real-3dpw-v1":
        raise ValueError(
            f"Unsupported checkpoint format: {checkpoint.get('format')!r}"
        )
    if checkpoint.get("feature_backend") != "vace":
        raise ValueError("This evaluator requires a frozen-VACE decoder checkpoint")
    required = (
        "decoder",
        "decoder_config",
        "args",
        "feature_layers",
        "vace_timestep",
        "epoch",
        "best_val",
    )
    missing = [name for name in required if name not in checkpoint]
    if missing:
        raise ValueError(f"Checkpoint is missing fields: {', '.join(missing)}")
    config = checkpoint["decoder_config"]
    if not isinstance(config, dict):
        raise TypeError("checkpoint decoder_config must be a dictionary")
    expected_dimensions = {"input_dim": 1536, "body_pose_dim": 63, "num_betas": 10}
    for name, expected in expected_dimensions.items():
        if int(config.get(name, -1)) != expected:
            raise ValueError(
                f"checkpoint decoder_config[{name!r}]={config.get(name)!r}; "
                f"expected {expected}"
            )
    training_args = checkpoint["args"]
    if not isinstance(training_args, dict) or "image_size" not in training_args:
        raise ValueError("checkpoint args must contain image_size")
    image_size = int(training_args["image_size"])
    if image_size <= 0 or image_size % 16:
        raise ValueError(f"Invalid checkpoint image_size={image_size}")
    feature_layers = checkpoint["feature_layers"]
    if not feature_layers or not all(int(value) >= 0 for value in feature_layers):
        raise ValueError(f"Invalid feature_layers={feature_layers!r}")


def per_sequence_summary(
    sequences: list[str], metric_arrays: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, float]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, sequence in enumerate(sequences):
        grouped[sequence].append(index)
    result: Dict[str, Dict[str, float]] = {}
    for sequence, indices in sorted(grouped.items()):
        selection = np.asarray(indices, dtype=np.int64)
        result[sequence] = {"samples": int(len(selection))}
        for name, values in metric_arrays.items():
            result[sequence][name] = float(values[selection].mean())
    return result


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.max_samples < 0:
        raise ValueError("batch-size must be positive; workers/max-samples non-negative")
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if args.output.suffix.lower() != ".json":
        raise ValueError("--output must end in .json so it cannot overwrite raw .npz")

    started = time.monotonic()
    audit = audit_split(args.data_root, args.split)
    print(json.dumps({"dataset_preflight": audit}, indent=2, ensure_ascii=False), flush=True)
    enforce_complete_test(audit, args.allow_incomplete_test)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_metadata(checkpoint)

    decoder = HumanQueryTransformer(
        HumanDecoderConfig(**checkpoint["decoder_config"])
    ).to(device).eval()
    decoder.load_state_dict(checkpoint["decoder"], strict=True)
    training_args = checkpoint["args"]
    image_size = int(training_args["image_size"])
    feature_layers = tuple(int(value) for value in checkpoint["feature_layers"])
    timestep = int(checkpoint["vace_timestep"])

    maximum = None if args.max_samples == 0 else args.max_samples
    dataset = Standard3DPWDataset(
        root=args.data_root,
        split=args.split,
        image_size=image_size,
        frame_stride=1,
        max_samples=maximum,
        crop_to_person=True,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"The real 3DPW {args.split} dataset is empty")
    if args.split == "test" and args.max_samples == 0:
        expected = int(audit["expected_valid_person_frames"])
        if len(dataset) != expected:
            raise RuntimeError(
                f"Dataset index has {len(dataset)} samples, preflight expected {expected}"
            )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    joint_regressor_path = args.joint_regressor.expanduser().resolve()
    joint_regressor_np = np.load(joint_regressor_path)
    if joint_regressor_np.shape != (17, 6890):
        raise ValueError(
            f"Expected J_regressor_h36m shape (17,6890), got {joint_regressor_np.shape}"
        )
    if not np.isfinite(joint_regressor_np).all():
        raise ValueError("The H36M joint regressor contains non-finite values")
    joint_regressor = torch.from_numpy(joint_regressor_np).to(
        device=device, dtype=torch.float32
    )

    smpl_models = create_smpl_models(args.smpl_model_root, device)
    encoder = FrozenVaceFeatureEncoder(
        args.vace_checkpoint_dir,
        device=device,
        image_size=image_size,
        feature_layers=feature_layers,
        timestep=timestep,
    )
    grid_size = (1, image_size // 16, image_size // 16)

    sequence_names: list[str] = []
    person_ids: list[int] = []
    frame_ids: list[int] = []
    metric_chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    pred_camera_joints: list[np.ndarray] = []
    gt_camera_joints: list[np.ndarray] = []
    predicted_pose_chunks: list[np.ndarray] = []
    predicted_beta_chunks: list[np.ndarray] = []
    evaluated = 0

    try:
        for step, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            features = encoder.encode(images)
            amp_enabled = device.type == "cuda"
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                output = decoder(
                    features, grid_size=grid_size, target_frames=1
                )

            pred_global = output["global_orient"][:, 0].float()
            pred_body63 = output["body_pose"][:, 0].float()
            pred_body69 = torch.cat(
                (
                    pred_body63,
                    torch.zeros(
                        pred_body63.shape[0],
                        6,
                        device=device,
                        dtype=torch.float32,
                    ),
                ),
                dim=1,
            )
            pred_betas = output["betas"].float()
            gt_pose = batch["full_pose"].to(
                device, dtype=torch.float32, non_blocking=True
            )
            gt_betas = batch["betas"].to(
                device, dtype=torch.float32, non_blocking=True
            )
            gender_codes = batch["gender_code"].to(device)

            with torch.inference_mode(), torch.autocast(
                device_type=device.type, enabled=False
            ):
                pred_vertices = smpl_models[2](
                    global_orient=pred_global,
                    body_pose=pred_body69,
                    betas=pred_betas,
                ).vertices
                gt_vertices = torch.empty_like(pred_vertices)
                for gender_code in (0, 1, 2):
                    indices = torch.nonzero(
                        gender_codes == gender_code, as_tuple=False
                    ).flatten()
                    if not len(indices):
                        continue
                    gender_output = smpl_models[gender_code](
                        global_orient=gt_pose.index_select(0, indices)[:, :3],
                        body_pose=gt_pose.index_select(0, indices)[:, 3:72],
                        betas=gt_betas.index_select(0, indices),
                    ).vertices
                    gt_vertices.index_copy_(0, indices, gender_output)

                if not (
                    torch.isfinite(pred_vertices).all()
                    and torch.isfinite(gt_vertices).all()
                ):
                    raise FloatingPointError(f"Non-finite SMPL mesh at step {step}")

                pred_j14, pred_pelvis = pelvis_aligned_j14(
                    pred_vertices, joint_regressor
                )
                gt_j14, gt_pelvis = pelvis_aligned_j14(
                    gt_vertices, joint_regressor
                )
                mpjpe = mean_euclidean_error(pred_j14, gt_j14)
                pa_mpjpe = pa_mpjpe_per_sample(pred_j14, gt_j14)
                pve = mean_euclidean_error(pred_vertices, gt_vertices)
                pelvis_pve = mean_euclidean_error(
                    pred_vertices - pred_pelvis,
                    gt_vertices - gt_pelvis,
                )

                camera_rotations = batch["cam_pose"][:, :3, :3].to(
                    device, dtype=torch.float32, non_blocking=True
                )
                # VIBE uses the midpoint of J14 left/right hips (indices 2, 3)
                # as its dynamic root.  This differs slightly from SPIN's H36M
                # joint-zero root used by the primary framewise MPJPE above.
                pred_j14_camera = rotate_points(
                    hip_centered_j14(pred_j14), camera_rotations
                )
                gt_j14_camera = rotate_points(
                    hip_centered_j14(gt_j14), camera_rotations
                )

            metric_chunks["mpjpe_mm"].append(mpjpe.cpu().numpy() * 1000.0)
            metric_chunks["pa_mpjpe_mm"].append(
                pa_mpjpe.cpu().numpy() * 1000.0
            )
            metric_chunks["pve_mm"].append(pve.cpu().numpy() * 1000.0)
            metric_chunks["pelvis_aligned_pve_mm"].append(
                pelvis_pve.cpu().numpy() * 1000.0
            )
            pred_camera_joints.append(pred_j14_camera.cpu().numpy())
            gt_camera_joints.append(gt_j14_camera.cpu().numpy())
            predicted_pose_chunks.append(
                torch.cat((pred_global, pred_body69), dim=1).cpu().numpy()
            )
            predicted_beta_chunks.append(pred_betas.cpu().numpy())
            sequence_names.extend(str(value) for value in batch["sequence"])
            person_ids.extend(int(value) for value in batch["person_id"])
            frame_ids.extend(int(value) for value in batch["frame_id"])
            evaluated += images.shape[0]
            if step % 50 == 0 or step == len(loader):
                print(
                    json.dumps(
                        {
                            "phase": "standard_3dpw_evaluation",
                            "step": step,
                            "steps": len(loader),
                            "samples": evaluated,
                        }
                    ),
                    flush=True,
                )
    finally:
        encoder.close()

    metric_arrays = {
        name: np.concatenate(chunks).astype(np.float64, copy=False)
        for name, chunks in metric_chunks.items()
    }
    pred_camera = np.concatenate(pred_camera_joints).astype(np.float64)
    gt_camera = np.concatenate(gt_camera_joints).astype(np.float64)
    acceleration = sequence_acceleration_error(
        pred_camera,
        gt_camera,
        sequence_names,
        person_ids,
        frame_ids,
    )
    if not len(acceleration.errors):
        raise RuntimeError("No consecutive three-frame windows for acceleration error")
    if (
        args.split == "test"
        and args.max_samples == 0
        and not args.allow_incomplete_test
        and len(acceleration.errors) != EXPECTED_TEST_ACCELERATION_TRIPLETS
    ):
        raise RuntimeError(
            f"Complete test produced {len(acceleration.errors)} acceleration "
            f"triplets, expected {EXPECTED_TEST_ACCELERATION_TRIPLETS}"
        )
    acceleration_mm_per_frame2 = acceleration.errors * 1000.0
    acceleration_m_per_s2 = acceleration.errors * (args.fps**2)

    coverage = (
        "complete_test"
        if args.split == "test"
        and args.max_samples == 0
        and not args.allow_incomplete_test
        else "subset_or_non_test"
    )
    primary_metrics = {
        name: float(values.mean()) for name, values in metric_arrays.items()
    }
    primary_metrics.update(
        {
            "acceleration_error_mm_per_frame2": float(
                acceleration_mm_per_frame2.mean()
            ),
            "acceleration_error_physical_m_per_s2": float(
                acceleration_m_per_s2.mean()
            ),
        }
    )
    smpl_root = args.smpl_model_root.expanduser().resolve()
    body_model_hashes = {
        gender: sha256_file(smpl_root / f"SMPL_{gender.upper()}.pkl")
        for gender in ("neutral", "male", "female")
    }
    result = {
        "status": "DIVIBODY_STANDARD_3DPW_EVALUATION_OK",
        "coverage": coverage,
        "split": args.split,
        "evaluated_person_frames": evaluated,
        "dataset_preflight": audit,
        "checkpoint": {
            "path": str(args.checkpoint.expanduser().resolve()),
            "sha256": sha256_file(args.checkpoint.expanduser().resolve()),
            "epoch": int(checkpoint["epoch"]),
            "best_validation_total": float(checkpoint["best_val"]),
        },
        "feature_backend": "official frozen Wan2.1-VACE",
        "feature_layers": list(feature_layers),
        "vace_timestep": timestep,
        "image_size": image_size,
        "metrics": primary_metrics,
        "acceleration_valid_triplets": int(len(acceleration.errors)),
        "per_sequence": per_sequence_summary(sequence_names, metric_arrays),
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
            "id": "3dpw_spin14_gendered_smpl_plus_sequence_safe_accel_v1",
            "joint_topology": "H36M-regressed J14 from 6890-vertex SMPL",
            "h36m_to_j14": list(H36M_TO_J14),
            "root": "H36M regressor joint 0, independently removed from prediction and GT",
            "mpjpe": "mean J14 Euclidean error after root alignment",
            "pa_mpjpe": "MPJPE after per-frame scale+proper-rotation+translation Procrustes",
            "pve": "direct corresponding-vertex error; create_transl=False; no Procrustes",
            "pve_comparability_warning": "decoder predicts 63D for SMPL joints 1..21; joints 22..23 are fixed to zero, while GT uses all 69D body pose",
            "pelvis_aligned_pve": "auxiliary corresponding-vertex error after H36M pelvis removal",
            "prediction_body_model": "neutral SMPL; decoder body_pose63 plus two zero SMPL end-joint rotations",
            "ground_truth_body_model": "3DPW gendered SMPL; complete 72D annotation pose",
            "translation": "ignored for root-relative body metrics",
            "acceleration": "camera-rotated, VIBE hip-midpoint-rooted J14 second-difference error; only same-person consecutive frame triplets",
            "acceleration_root": "midpoint of J14 left/right hips at indices 2 and 3",
            "acceleration_reported_unit": "mm/frame^2 (VIBE-compatible finite difference)",
            "physical_acceleration": f"also reported in m/s^2 using fps={args.fps}",
            "crop": "oracle person crop from 3DPW ground-truth poses2d with 25 percent margin",
            "not_official_challenge_protocol": True,
        },
        "assets": {
            "joint_regressor": str(joint_regressor_path),
            "joint_regressor_sha256": sha256_file(joint_regressor_path),
            "smpl_model_root": str(smpl_root),
            "smpl_model_sha256": body_model_hashes,
        },
        "raw_arrays": str(args.output.with_suffix(".npz").resolve()),
        "runtime_seconds": time.monotonic() - started,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "limitations": [
            "3DPW has no hand or face ground truth; all four metrics are body-only.",
            "The trained decoder predicts 21 SMPL-compatible body joints; the final two SMPL end-joint rotations are zero.",
            "Oracle ground-truth person crops are used, so this is not detector/tracker evaluation.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        sequence=np.asarray(sequence_names),
        person_id=np.asarray(person_ids, dtype=np.int64),
        frame_id=np.asarray(frame_ids, dtype=np.int64),
        pred_pose72=np.concatenate(predicted_pose_chunks).astype(np.float32),
        pred_betas=np.concatenate(predicted_beta_chunks).astype(np.float32),
        pred_camera_root_aligned_j14=pred_camera.astype(np.float32),
        gt_camera_root_aligned_j14=gt_camera.astype(np.float32),
        acceleration_center_sample_index=acceleration.center_sample_indices,
        acceleration_error_mm_per_frame2=acceleration_mm_per_frame2,
        **metric_arrays,
    )
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
