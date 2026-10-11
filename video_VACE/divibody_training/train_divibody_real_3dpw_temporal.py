"""Train a temporal decoder on consecutive cached 3DPW VACE features.

This is an additive experiment.  The original framewise trainer and model
sources remain unchanged.  Clips never cross sequence/person tracks or frame
gaps, and temporal losses match ground-truth motion instead of forcing motion
to zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.human_decoder import HumanDecoderConfig, HumanQueryTransformer


TARGET_NAMES = ("global_orient", "body_pose", "betas", "transl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the temporal VACE human decoder on consecutive 3DPW clips"
    )
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--train-clip-stride", type=int, default=4)
    parser.add_argument("--val-clip-stride", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--decoder-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--rotation-weight", type=float, default=1.0)
    parser.add_argument("--axis-angle-weight", type=float, default=0.25)
    parser.add_argument("--shape-weight", type=float, default=0.1)
    parser.add_argument("--translation-weight", type=float, default=0.0)
    parser.add_argument("--velocity-weight", type=float, default=0.5)
    parser.add_argument("--acceleration-weight", type=float, default=0.25)
    parser.add_argument("--unsupported-head-reg", type=float, default=1.0e-4)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-interval", type=int, default=25)
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TemporalClipDataset(Dataset):
    """Consecutive clips built from one frozen per-frame feature cache."""

    def __init__(
        self,
        cache_path: Path,
        expected_split: str,
        clip_length: int,
        clip_stride: int,
    ) -> None:
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(cache_path, map_location="cpu")
        self.cache_path = cache_path.expanduser().resolve()
        self.signature = payload.get("signature", {})
        if self.signature.get("format") != "divibody-frozen-vace-features-v1":
            raise ValueError(f"Unsupported cache format in {cache_path}")
        if self.signature.get("split") != expected_split:
            raise ValueError(
                f"Cache split {self.signature.get('split')!r} is not {expected_split!r}"
            )
        self.features = payload["features"]
        self.targets = {name: payload["targets"][name] for name in TARGET_NAMES}
        self.metadata = payload.get("metadata", [])
        if self.features.ndim != 3 or self.features.shape[-1] != 1536:
            raise ValueError(f"Invalid feature shape {tuple(self.features.shape)}")
        sample_count = self.features.shape[0]
        if len(self.metadata) != sample_count:
            raise ValueError("Feature cache must contain metadata for every frame")
        if any(value.shape[0] != sample_count for value in self.targets.values()):
            raise ValueError("Cached targets and features have different lengths")
        if clip_length < 3 or clip_stride <= 0:
            raise ValueError("clip_length must be >=3 and clip_stride must be positive")
        self.clip_length = int(clip_length)
        self.clip_stride = int(clip_stride)

        tracks: dict[tuple[str, int], list[int]] = defaultdict(list)
        self.sequences: list[str] = []
        self.person_ids: list[int] = []
        self.frame_ids: list[int] = []
        for index, record in enumerate(self.metadata):
            sequence = str(record["sequence"])
            person_id = int(record["person_id"])
            frame_id = int(record["frame_id"])
            tracks[(sequence, person_id)].append(index)
            self.sequences.append(sequence)
            self.person_ids.append(person_id)
            self.frame_ids.append(frame_id)

        clips: list[torch.Tensor] = []
        segment_count = 0
        gap_count = 0
        for track, indices_list in sorted(tracks.items()):
            indices = sorted(indices_list, key=lambda value: self.frame_ids[value])
            frames = [self.frame_ids[index] for index in indices]
            if len(frames) != len(set(frames)):
                raise ValueError(f"Duplicate frame in cache track {track}")
            segment_start = 0
            boundaries = []
            for position in range(1, len(indices)):
                if frames[position] != frames[position - 1] + 1:
                    boundaries.append(position)
                    gap_count += 1
            for segment_end in (*boundaries, len(indices)):
                segment = indices[segment_start:segment_end]
                segment_start = segment_end
                segment_count += 1
                if len(segment) < self.clip_length:
                    continue
                last_start = len(segment) - self.clip_length
                starts = list(range(0, last_start + 1, self.clip_stride))
                if starts[-1] != last_start:
                    starts.append(last_start)
                clips.extend(
                    torch.as_tensor(
                        segment[start : start + self.clip_length], dtype=torch.long
                    )
                    for start in starts
                )
        if not clips:
            raise RuntimeError(f"No consecutive clips could be built from {cache_path}")
        self.clips = clips
        self.stats = {
            "split": expected_split,
            "frames": sample_count,
            "tracks": len(tracks),
            "segments": segment_count,
            "frame_gaps": gap_count,
            "clip_length": self.clip_length,
            "clip_stride": self.clip_stride,
            "clips": len(self.clips),
            "sequence_names": sorted(set(self.sequences)),
        }

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        indices = self.clips[index]
        result: Dict[str, Any] = {
            "features": self.features.index_select(0, indices),
            "global_orient": self.targets["global_orient"].index_select(0, indices),
            "body_pose": self.targets["body_pose"].index_select(0, indices),
            "transl": self.targets["transl"].index_select(0, indices),
            "betas": self.targets["betas"][int(indices[0])],
            "sample_indices": indices,
        }
        return result


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Differentiable Rodrigues transform with stable small-angle factors."""

    if axis_angle.shape[-1] != 3:
        raise ValueError("axis_angle must end in dimension 3")
    x_value, y_value, z_value = axis_angle.unbind(dim=-1)
    zero = torch.zeros_like(x_value)
    skew = torch.stack(
        (
            zero,
            -z_value,
            y_value,
            z_value,
            zero,
            -x_value,
            -y_value,
            x_value,
            zero,
        ),
        dim=-1,
    ).reshape(*axis_angle.shape[:-1], 3, 3)
    theta = torch.linalg.vector_norm(axis_angle, dim=-1)
    coefficient_a = torch.sinc(theta / math.pi)
    coefficient_b = 0.5 * torch.sinc(theta / (2.0 * math.pi)).square()
    identity = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    return (
        identity
        + coefficient_a[..., None, None] * skew
        + coefficient_b[..., None, None] * (skew @ skew)
    )


def temporal_losses(
    prediction: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    target_global = batch["global_orient"].to(device=device, dtype=torch.float32)
    target_body = batch["body_pose"].to(device=device, dtype=torch.float32)
    target_betas = batch["betas"].to(device=device, dtype=torch.float32)
    target_transl = batch["transl"].to(device=device, dtype=torch.float32)
    pred_global = prediction["global_orient"].float()
    pred_body = prediction["body_pose"].float()
    pred_betas = prediction["betas"].float()
    pred_transl = prediction["transl"].float()

    pred_pose = torch.cat((pred_global, pred_body), dim=-1).reshape(
        pred_global.shape[0], pred_global.shape[1], 22, 3
    )
    target_pose = torch.cat((target_global, target_body), dim=-1).reshape_as(pred_pose)
    pred_rotations = axis_angle_to_matrix(pred_pose)
    target_rotations = axis_angle_to_matrix(target_pose)

    rotation = F.mse_loss(pred_rotations, target_rotations)
    body_axis_angle = F.smooth_l1_loss(pred_body, target_body)
    global_axis_angle = F.smooth_l1_loss(pred_global, target_global)
    shape = F.smooth_l1_loss(pred_betas, target_betas)
    translation = F.smooth_l1_loss(pred_transl, target_transl)

    pred_velocity = pred_rotations[:, 1:] - pred_rotations[:, :-1]
    target_velocity = target_rotations[:, 1:] - target_rotations[:, :-1]
    velocity = F.smooth_l1_loss(pred_velocity, target_velocity)
    pred_acceleration = pred_velocity[:, 1:] - pred_velocity[:, :-1]
    target_acceleration = target_velocity[:, 1:] - target_velocity[:, :-1]
    acceleration = F.smooth_l1_loss(pred_acceleration, target_acceleration)

    unsupported = (
        prediction["left_hand_pose"].float().square().mean()
        + prediction["right_hand_pose"].float().square().mean()
        + prediction["jaw_pose"].float().square().mean()
        + prediction["expression"].float().square().mean()
        + prediction["object_state"].float().square().mean()
    )
    total = (
        args.rotation_weight * rotation
        + args.axis_angle_weight * (body_axis_angle + global_axis_angle)
        + args.shape_weight * shape
        + args.translation_weight * translation
        + args.velocity_weight * velocity
        + args.acceleration_weight * acceleration
        + args.unsupported_head_reg * unsupported
    )
    return {
        "total": total,
        "rotation_chordal": rotation,
        "body_axis_angle": body_axis_angle,
        "global_axis_angle": global_axis_angle,
        "shape": shape,
        "translation": translation,
        "rotation_velocity": velocity,
        "rotation_acceleration": acceleration,
        "unsupported_regularizer": unsupported,
    }


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def amp_settings(
    args: argparse.Namespace, device: torch.device
) -> tuple[bool, torch.dtype]:
    enabled = device.type == "cuda" and args.amp_dtype != "none"
    dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    return enabled, dtype


def run_epoch(
    loader: DataLoader,
    decoder: HumanQueryTransformer,
    optimizer: Optional[torch.optim.Optimizer],
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
    phase: str,
) -> Dict[str, float]:
    is_train = optimizer is not None
    decoder.train(is_train)
    totals: Dict[str, float] = {}
    clip_count = 0
    started = time.monotonic()
    amp_enabled, amp_dtype = amp_settings(args, device)
    grad_context = torch.enable_grad if is_train else torch.no_grad
    with grad_context():
        for step, batch in enumerate(loader, start=1):
            features = batch["features"].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            batch_size, frames, tokens, channels = features.shape
            grid_width = math.isqrt(tokens)
            if grid_width * grid_width != tokens:
                raise RuntimeError(
                    f"Temporal feature token count {tokens} is not square"
                )
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                prediction = decoder(
                    features.reshape(batch_size, frames * tokens, channels),
                    grid_size=(frames, grid_width, grid_width),
                    target_frames=frames,
                )
                losses = temporal_losses(prediction, batch, args, device)
                loss = losses["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite {phase} loss at epoch={epoch}, step={step}"
                )
            if is_train:
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.grad_clip)
                optimizer.step()
            clip_count += batch_size
            for name, value in losses.items():
                totals[name] = (
                    totals.get(name, 0.0) + float(value.detach()) * batch_size
                )
            if step % args.log_interval == 0 or step == len(loader):
                print(
                    json.dumps(
                        {
                            "phase": phase,
                            "epoch": epoch,
                            "step": step,
                            "steps": len(loader),
                            "clips": clip_count,
                            "loss": float(loss.detach()),
                        }
                    ),
                    flush=True,
                )
    return {
        **{name: value / clip_count for name, value in totals.items()},
        "clips": float(clip_count),
        "seconds": time.monotonic() - started,
    }


def initialise_from_frame_checkpoint(
    decoder: HumanQueryTransformer, path: Path, seed: int
) -> Dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    source = checkpoint["decoder"]
    target = decoder.state_dict()
    copied = []
    expanded = []
    for name, value in source.items():
        if name not in target:
            continue
        if value.shape == target[name].shape:
            target[name] = value
            copied.append(name)
        elif (
            name in ("source_time_embedding.weight", "query_time_embedding.weight")
            and value.ndim == 2
            and value.shape[0] == 1
            and value.shape[1] == target[name].shape[1]
        ):
            generator = torch.Generator().manual_seed(seed)
            noise = (
                torch.randn(target[name].shape, generator=generator, dtype=value.dtype)
                * 1.0e-3
            )
            target[name] = value.expand_as(target[name]).clone() + noise
            expanded.append(name)
    decoder.load_state_dict(target, strict=True)
    return {
        "path": str(path.expanduser().resolve()),
        "sha256": sha256_file(path),
        "format": checkpoint.get("format"),
        "epoch": checkpoint.get("epoch"),
        "copied_tensor_count": len(copied),
        "expanded_time_embeddings": expanded,
    }


def save_checkpoint(
    path: Path,
    decoder: HumanQueryTransformer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val: float,
    validation: Dict[str, float],
    config: HumanDecoderConfig,
    args: argparse.Namespace,
    train_dataset: TemporalClipDataset,
    val_dataset: TemporalClipDataset,
    initialization: Optional[Dict[str, Any]],
) -> None:
    payload = {
        "format": "divibody-real-3dpw-temporal-v1",
        "epoch": epoch,
        "best_val": best_val,
        "decoder_config": vars(config),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "validation": validation,
        "args": json_ready(vars(args)),
        "train_cache_signature": train_dataset.signature,
        "val_cache_signature": val_dataset.signature,
        "train_clip_stats": train_dataset.stats,
        "val_clip_stats": val_dataset.stats,
        "initialization": initialization,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("epochs/batch-size must be positive and workers non-negative")
    if args.hidden_dim % args.decoder_heads:
        raise ValueError("hidden-dim must be divisible by decoder-heads")
    if args.log_interval <= 0:
        raise ValueError("log-interval must be positive")
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    train_dataset = TemporalClipDataset(
        args.train_cache, "train", args.clip_length, args.train_clip_stride
    )
    val_dataset = TemporalClipDataset(
        args.val_cache, "validation", args.clip_length, args.val_clip_stride
    )
    overlap = sorted(
        set(train_dataset.stats["sequence_names"])
        & set(val_dataset.stats["sequence_names"])
    )
    if overlap:
        raise RuntimeError(f"Train/validation sequence leakage: {overlap}")

    train_loader = make_loader(
        train_dataset, args.batch_size, True, args.num_workers, device
    )
    val_loader = make_loader(
        val_dataset, args.batch_size, False, args.num_workers, device
    )
    config = HumanDecoderConfig(
        input_dim=1536,
        hidden_dim=args.hidden_dim,
        num_heads=args.decoder_heads,
        num_layers=args.decoder_layers,
        dropout=args.dropout,
        body_pose_dim=63,
        max_latent_frames=args.clip_length,
    )
    decoder = HumanQueryTransformer(config).to(device)
    initialization = (
        initialise_from_frame_checkpoint(decoder, args.init_checkpoint, args.seed)
        if args.init_checkpoint is not None
        else None
    )
    optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "status": "temporal_training_started",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "decoder_config": vars(config),
        "train_clip_stats": train_dataset.stats,
        "val_clip_stats": val_dataset.stats,
        "train_validation_sequence_overlap": overlap,
        "initialization": initialization,
        "args": json_ready(vars(args)),
        "loss_protocol": {
            "frame_rotation": "22 SMPL rotations, chordal matrix loss",
            "frame_axis_angle": "auxiliary SmoothL1 on global plus 21 body joints",
            "temporal_velocity": "predicted rotation-matrix delta matched to GT",
            "temporal_acceleration": "predicted second rotation delta matched to GT",
            "translation_in_primary_loss": args.translation_weight > 0,
        },
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(run_config, indent=2, ensure_ascii=False), flush=True)

    metrics_path = args.output_dir / "metrics.jsonl"
    best_val = math.inf
    final_validation: Dict[str, float] = {}
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            train_loader, decoder, optimizer, args, device, epoch, "train"
        )
        val_metrics = run_epoch(
            val_loader, decoder, None, args, device, epoch, "validation"
        )
        final_validation = val_metrics
        record = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        improved = val_metrics["total"] < best_val
        if improved:
            best_val = val_metrics["total"]
        save_checkpoint(
            args.output_dir / "latest.pt",
            decoder,
            optimizer,
            epoch,
            best_val,
            val_metrics,
            config,
            args,
            train_dataset,
            val_dataset,
            initialization,
        )
        if improved:
            save_checkpoint(
                args.output_dir / "best.pt",
                decoder,
                optimizer,
                epoch,
                best_val,
                val_metrics,
                config,
                args,
                train_dataset,
                val_dataset,
                initialization,
            )
        print(json.dumps(record, indent=2, ensure_ascii=False), flush=True)

    result = {
        "status": "DIVIBODY_TEMPORAL_3DPW_TRAINING_OK",
        "completed_epochs": args.epochs,
        "best_validation_total": best_val,
        "final_validation": final_validation,
        "best_checkpoint": str((args.output_dir / "best.pt").resolve()),
        "latest_checkpoint": str((args.output_dir / "latest.pt").resolve()),
        "metrics": str(metrics_path.resolve()),
    }
    (args.output_dir / "final_validation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
