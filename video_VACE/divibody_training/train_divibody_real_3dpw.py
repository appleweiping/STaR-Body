"""Train the separate whole-body decoder on real 3DPW images and SMPL labels.

The default feature backend freezes the official Wan2.1-VACE VAE and DiT and
only optimizes ``HumanQueryTransformer``.  A trainable ResNet-18 (or a small
CNN when torchvision is unavailable) is provided as a fast baseline.  The
original VACE and decoder source files are never modified by this program.

3DPW contains SMPL body supervision, but no SMPL-X hand, face, or expression
ground truth.  Consequently, only global orientation, 21-joint body pose,
betas, and translation are supervised.  Unsupported output heads receive a
small zero-centred regularizer and must not be reported as supervised results.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from models.human_decoder import HumanDecoderConfig, HumanQueryTransformer
from real_3dpw_dataset import Real3DPWDataset


class CachedFeatureDataset(Dataset):
    """CPU-resident frozen VACE features paired with real SMPL labels."""

    target_names = ("global_orient", "body_pose", "betas", "transl")

    def __init__(self, payload: Dict[str, Any]) -> None:
        self.features = payload["features"]
        self.targets = {name: payload["targets"][name] for name in self.target_names}
        self.metadata = payload.get("metadata", [])
        if self.features.ndim != 3 or self.features.shape[-1] != 1536:
            raise ValueError(
                f"Invalid cached VACE feature shape: {tuple(self.features.shape)}"
            )
        size = self.features.shape[0]
        if any(value.shape[0] != size for value in self.targets.values()):
            raise ValueError("Cached feature and target counts differ")

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = {"features": self.features[index]}
        item.update({name: value[index] for name, value in self.targets.items()})
        return item


class ResNet18Tokens(nn.Module):
    """Return the final ResNet-18 feature map as spatial decoder tokens."""

    output_dim = 512
    stride = 32

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT
        filename = Path(urlparse(weights.url).path).name
        cached = Path(torch.hub.get_dir()) / "checkpoints" / filename
        model = resnet18(weights=None)
        if cached.is_file():
            try:
                state = torch.load(cached, map_location="cpu", weights_only=True)
            except TypeError:  # PyTorch before weights_only was introduced.
                state = torch.load(cached, map_location="cpu")
            model.load_state_dict(state)
            self.initialization = f"ImageNet cached weights: {cached}"
        else:
            self.initialization = (
                "random initialization (ImageNet weights were not already cached; "
                "no network download was attempted)"
            )
        self.backbone = nn.Sequential(*list(model.children())[:-2])
        self.register_buffer(
            "image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feature_map = self.backbone((images - self.image_mean) / self.image_std)
        return feature_map.flatten(2).transpose(1, 2)


class TinyImageTokens(nn.Module):
    """Small trainable fallback that emits a stride-16 spatial feature grid."""

    output_dim = 128
    stride = 16

    def __init__(self) -> None:
        super().__init__()
        layers = []
        channels = (3, 32, 64, 96, self.output_dim)
        for input_channels, output_channels in zip(channels[:-1], channels[1:]):
            layers.extend(
                [
                    nn.Conv2d(
                        input_channels,
                        output_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(8, output_channels),
                    nn.GELU(),
                ]
            )
        self.backbone = nn.Sequential(*layers)
        self.initialization = "random initialization"

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feature_map = self.backbone(images)
        return feature_map.flatten(2).transpose(1, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train HumanQueryTransformer on real 3DPW RGB/SMPL pairs"
    )
    parser.add_argument(
        "--data-root",
        "--dataset-root",
        dest="dataset_root",
        type=Path,
        default=Path("datasets/3DPW"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("models/Wan2.1-VACE-1.3B"),
        help="Official VACE checkpoint directory; used only by --feature-backend vace",
    )
    parser.add_argument(
        "--feature-backend", choices=("vace", "resnet18", "tiny"), default="vace"
    )
    parser.add_argument("--feature-layers", type=int, nargs="+", default=(7, 14, 22))
    parser.add_argument("--vace-timestep", type=int, default=500)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument(
        "--train-frame-stride", "--frame-stride", dest="train_frame_stride", type=int, default=8
    )
    parser.add_argument("--val-frame-stride", type=int, default=8)
    parser.add_argument(
        "--max-train-samples", "--max-train", dest="max_train", type=int, default=512
    )
    parser.add_argument(
        "--max-val-samples", "--max-val", dest="max_val", type=int, default=128
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--decoder-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", "--learning-rate", dest="learning_rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--body-weight", type=float, default=1.0)
    parser.add_argument("--orient-weight", type=float, default=1.0)
    parser.add_argument("--shape-weight", type=float, default=0.1)
    parser.add_argument("--transl-weight", type=float, default=1.0)
    parser.add_argument("--unsupported-head-reg", type=float, default=1.0e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/divibody_real_3dpw")
    )
    parser.add_argument(
        "--cache-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache frozen VACE tokens and real labels in output-dir (default: enabled)",
    )
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def create_dataset(
    root: Path,
    split: str,
    image_size: int,
    frame_stride: int,
    max_samples: int,
) -> Real3DPWDataset:
    return Real3DPWDataset(
        root=root,
        split=split,
        image_size=image_size,
        frame_stride=frame_stride,
        max_samples=None if max_samples <= 0 else max_samples,
    )


def build_datasets(args: argparse.Namespace) -> Tuple[Dataset, Dataset, str]:
    train_dataset = create_dataset(
        args.dataset_root,
        args.train_split,
        args.image_size,
        args.train_frame_stride,
        args.max_train,
    )
    if len(train_dataset) == 0:
        raise RuntimeError(
            f"Training split {args.train_split!r} produced no 3DPW samples at "
            f"{args.dataset_root}"
        )

    fallback_reason: Optional[str] = None
    try:
        val_dataset: Dataset = create_dataset(
            args.dataset_root,
            args.val_split,
            args.image_size,
            args.val_frame_stride,
            args.max_val,
        )
        if len(val_dataset) == 0:
            fallback_reason = f"split {args.val_split!r} was empty"
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        fallback_reason = f"split {args.val_split!r} unavailable: {error}"
        val_dataset = Subset(train_dataset, [])

    if fallback_reason is not None:
        if len(train_dataset) == 1:
            # A one-sample smoke run cannot provide a disjoint validation set.
            val_dataset = Subset(train_dataset, [0])
            split_note = fallback_reason + "; reused the only train sample for smoke validation"
        else:
            requested_val = args.max_val if args.max_val > 0 else max(1, len(train_dataset) // 10)
            val_size = min(max(1, requested_val), len(train_dataset) - 1)
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = random_split(
                train_dataset,
                (train_size, val_size),
                generator=torch.Generator().manual_seed(args.seed),
            )
            split_note = fallback_reason + "; used a deterministic held-out train subset"
    else:
        split_note = f"native {args.val_split!r} split"

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError(
            f"Non-empty train and validation loaders are required; got "
            f"train={len(train_dataset)}, val={len(val_dataset)}"
        )
    return train_dataset, val_dataset, split_note


def worker_seed(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


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
        drop_last=False,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_seed,
    )


def feature_cache_signature(
    args: argparse.Namespace, role: str, dataset_size: int, split_note: str
) -> Dict[str, Any]:
    return {
        "format": "divibody-frozen-vace-features-v1",
        "role": role,
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_size": dataset_size,
        "split": args.train_split if role == "train" else args.val_split,
        "split_note": split_note,
        "frame_stride": (
            args.train_frame_stride if role == "train" else args.val_frame_stride
        ),
        "image_size": args.image_size,
        "feature_backend": args.feature_backend,
        "feature_layers": list(args.feature_layers),
        "vace_timestep": args.vace_timestep,
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "seed": args.seed,
    }


def load_feature_cache(
    path: Path, expected_signature: Dict[str, Any]
) -> Optional[CachedFeatureDataset]:
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("signature") != expected_signature:
        print(f"Ignoring stale feature cache with a different signature: {path}", flush=True)
        return None
    dataset = CachedFeatureDataset(payload)
    print(
        f"Loaded {len(dataset)} frozen VACE feature/label pairs from {path}",
        flush=True,
    )
    return dataset


def metadata_item(value: Any, index: int) -> Any:
    if torch.is_tensor(value):
        item = value[index].detach().cpu()
        return item.item() if item.ndim == 0 else item.tolist()
    if isinstance(value, np.ndarray):
        return value[index].tolist()
    if isinstance(value, (list, tuple)):
        return json_ready(value[index])
    return json_ready(value)


def precompute_feature_cache(
    path: Path,
    dataset: Dataset,
    signature: Dict[str, Any],
    encoder: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> CachedFeatureDataset:
    loader = make_loader(dataset, args.batch_size, False, args.num_workers, device)
    feature_chunks = []
    target_chunks: Dict[str, list[torch.Tensor]] = {
        name: [] for name in CachedFeatureDataset.target_names
    }
    metadata = []
    excluded = {"image", *CachedFeatureDataset.target_names}
    started = time.monotonic()
    seen = 0
    encoder.eval()
    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, dtype=torch.float32, non_blocking=True)
        features = encoder.encode(images)
        expected_tokens = (args.image_size // 16) ** 2
        if tuple(features.shape[1:]) != (expected_tokens, 1536):
            raise RuntimeError(
                f"Frozen VACE emitted {tuple(features.shape)}, expected "
                f"[B,{expected_tokens},1536]"
            )
        feature_chunks.append(features.to(device="cpu", dtype=torch.bfloat16))
        for name in target_chunks:
            target_chunks[name].append(batch[name].detach().cpu().float())

        for index in range(images.shape[0]):
            metadata.append(
                {
                    key: metadata_item(value, index)
                    for key, value in batch.items()
                    if key not in excluded
                }
            )
        seen += images.shape[0]
        if step % args.log_interval == 0 or step == len(loader):
            print(
                json.dumps(
                    {
                        "phase": "cache_features",
                        "file": str(path),
                        "step": step,
                        "steps": len(loader),
                        "samples": seen,
                        "seconds": time.monotonic() - started,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    if not feature_chunks:
        raise RuntimeError(f"Cannot cache an empty dataset: {path}")
    payload = {
        "signature": signature,
        "features": torch.cat(feature_chunks),
        "targets": {
            name: torch.cat(chunks) for name, chunks in target_chunks.items()
        },
        "metadata": metadata,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    result = CachedFeatureDataset(payload)
    print(
        f"Saved {len(result)} real-image frozen VACE features to {path}", flush=True
    )
    return result


def build_feature_encoder(
    args: argparse.Namespace, device: torch.device
) -> Tuple[nn.Module, int, int, bool, str]:
    if args.feature_backend == "vace":
        if device.type != "cuda":
            raise RuntimeError("The frozen VACE backend requires --device cuda")
        from real_vace_feature_encoder import FrozenVaceFeatureEncoder

        encoder = FrozenVaceFeatureEncoder(
            checkpoint_dir=args.checkpoint_dir,
            device=device,
            image_size=args.image_size,
            feature_layers=args.feature_layers,
            timestep=args.vace_timestep,
        )
        return encoder, 1536, 16, False, "official frozen Wan2.1-VACE VAE + DiT"

    if args.feature_backend == "resnet18":
        try:
            encoder = ResNet18Tokens()
        except (ImportError, OSError, RuntimeError) as error:
            print(
                f"torchvision ResNet-18 unavailable ({error}); using TinyImageTokens",
                flush=True,
            )
            encoder = TinyImageTokens()
        encoder = encoder.to(device)
        return (
            encoder,
            encoder.output_dim,
            encoder.stride,
            True,
            f"{encoder.__class__.__name__}: {encoder.initialization}",
        )

    encoder = TinyImageTokens().to(device)
    return encoder, encoder.output_dim, encoder.stride, True, "trainable lightweight CNN"


def amp_settings(
    args: argparse.Namespace, device: torch.device
) -> Tuple[bool, torch.dtype, bool]:
    enabled = device.type == "cuda" and args.amp_dtype != "none"
    dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    use_scaler = enabled and dtype == torch.float16
    return enabled, dtype, use_scaler


def extract_tokens(
    encoder: nn.Module,
    images: torch.Tensor,
    backend: str,
    trainable: bool,
) -> torch.Tensor:
    if backend == "vace":
        return encoder.encode(images)
    if trainable:
        return encoder(images)
    with torch.no_grad():
        return encoder(images)


def decoder_predictions(
    decoder: HumanQueryTransformer,
    features: torch.Tensor,
    image_size: Tuple[int, int],
    stride: int,
) -> Dict[str, torch.Tensor]:
    height, width = image_size
    grid_h, grid_w = height // stride, width // stride
    expected = grid_h * grid_w
    if features.ndim != 3:
        raise RuntimeError(f"Feature encoder returned {tuple(features.shape)}, expected [B,L,C]")
    if features.shape[1] != expected:
        # Keep the error actionable if an upstream VAE changes its tokenization.
        square = math.isqrt(features.shape[1])
        if square * square == features.shape[1]:
            grid_h = grid_w = square
        else:
            raise RuntimeError(
                f"Feature tokens={features.shape[1]} do not match image grid "
                f"{grid_h}x{grid_w} for stride {stride}"
            )
    return decoder(features, grid_size=(1, grid_h, grid_w), target_frames=1)


def supervised_losses(
    prediction: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    target = {
        name: batch[name].to(device, dtype=torch.float32, non_blocking=True)
        for name in ("global_orient", "body_pose", "betas", "transl")
    }
    predicted = {
        "global_orient": prediction["global_orient"][:, 0].float(),
        "body_pose": prediction["body_pose"][:, 0].float(),
        "betas": prediction["betas"].float(),
        "transl": prediction["transl"][:, 0].float(),
    }
    losses = {
        "body_pose": F.smooth_l1_loss(predicted["body_pose"], target["body_pose"]),
        "global_orient": F.smooth_l1_loss(
            predicted["global_orient"], target["global_orient"]
        ),
        "betas": F.smooth_l1_loss(predicted["betas"], target["betas"]),
        "transl": F.smooth_l1_loss(predicted["transl"], target["transl"]),
    }
    unsupported = (
        prediction["left_hand_pose"].float().square().mean()
        + prediction["right_hand_pose"].float().square().mean()
        + prediction["jaw_pose"].float().square().mean()
        + prediction["expression"].float().square().mean()
        + prediction["object_state"].float().square().mean()
    )
    losses["unsupported_regularizer"] = unsupported
    losses["total"] = (
        args.body_weight * losses["body_pose"]
        + args.orient_weight * losses["global_orient"]
        + args.shape_weight * losses["betas"]
        + args.transl_weight * losses["transl"]
        + args.unsupported_head_reg * unsupported
    )

    for name in predicted:
        difference = predicted[name] - target[name]
        losses[f"{name}_squared_error"] = difference.square().mean()
        losses[f"{name}_absolute_error"] = difference.abs().mean()
    return losses


def set_modes(decoder: nn.Module, encoder: nn.Module, trainable_encoder: bool, train: bool) -> None:
    decoder.train(train)
    if trainable_encoder:
        encoder.train(train)
    else:
        encoder.eval()


def run_epoch(
    loader: DataLoader,
    decoder: HumanQueryTransformer,
    encoder: nn.Module,
    encoder_stride: int,
    trainable_encoder: bool,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
    phase: str,
) -> Dict[str, float]:
    is_train = optimizer is not None
    set_modes(decoder, encoder, trainable_encoder, is_train)
    amp_enabled, amp_dtype, _ = amp_settings(args, device)
    totals: Dict[str, float] = {}
    sample_count = 0
    started = time.monotonic()

    grad_context = torch.enable_grad if is_train else torch.no_grad
    with grad_context():
        for step, batch in enumerate(loader, start=1):
            cached_features = batch.get("features")
            if cached_features is None:
                images = batch["image"].to(
                    device, dtype=torch.float32, non_blocking=True
                )
                if images.ndim != 4 or images.shape[1] != 3:
                    raise RuntimeError(
                        f"Dataset image has invalid shape {tuple(images.shape)}"
                    )
                batch_size = images.shape[0]
            else:
                images = None
                features = cached_features.to(
                    device, dtype=torch.float32, non_blocking=True
                )
                batch_size = features.shape[0]
            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                if images is not None:
                    features = extract_tokens(
                        encoder, images, args.feature_backend, trainable_encoder
                    )
                prediction = decoder_predictions(
                    decoder,
                    features,
                    (args.image_size, args.image_size),
                    encoder_stride,
                )
                losses = supervised_losses(prediction, batch, args, device)
                loss = losses["total"]

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite {phase} loss at epoch={epoch}, step={step}: {loss.item()}"
                )
            if is_train:
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    parameters: Iterable[torch.Tensor] = (
                        parameter
                        for group in optimizer.param_groups
                        for parameter in group["params"]
                        if parameter.grad is not None
                    )
                    torch.nn.utils.clip_grad_norm_(list(parameters), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()

            sample_count += batch_size
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size

            if step % args.log_interval == 0 or step == len(loader):
                print(
                    json.dumps(
                        {
                            "phase": phase,
                            "epoch": epoch,
                            "step": step,
                            "steps": len(loader),
                            "samples": sample_count,
                            "loss": float(loss.detach()),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    if sample_count == 0:
        raise RuntimeError(f"{phase} loader was empty")
    metrics = {name: value / sample_count for name, value in totals.items()}
    for name in ("body_pose", "global_orient", "betas", "transl"):
        metrics[f"{name}_rmse"] = math.sqrt(metrics.pop(f"{name}_squared_error"))
        metrics[f"{name}_mae"] = metrics.pop(f"{name}_absolute_error")
    metrics["samples"] = float(sample_count)
    metrics["seconds"] = time.monotonic() - started
    return metrics


def save_checkpoint(
    path: Path,
    decoder: HumanQueryTransformer,
    encoder: nn.Module,
    trainable_encoder: bool,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_val: float,
    args: argparse.Namespace,
    val_metrics: Dict[str, float],
    decoder_config: HumanDecoderConfig,
) -> None:
    checkpoint = {
        "format": "divibody-real-3dpw-v1",
        "epoch": epoch,
        "best_val": best_val,
        "feature_backend": args.feature_backend,
        "feature_layers": list(args.feature_layers),
        "vace_timestep": args.vace_timestep,
        "decoder_config": vars(decoder_config),
        "decoder": decoder.state_dict(),
        "encoder": encoder.state_dict() if trainable_encoder else None,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "args": json_ready(vars(args)),
        "validation": val_metrics,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: Path,
    decoder: HumanQueryTransformer,
    encoder: nn.Module,
    trainable_encoder: bool,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    backend: str,
    feature_layers: Iterable[int],
    device: torch.device,
) -> Tuple[int, float]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("feature_backend") != backend:
        raise ValueError(
            f"Resume backend {checkpoint.get('feature_backend')!r} does not match {backend!r}"
        )
    if list(checkpoint.get("feature_layers", [])) != list(feature_layers):
        raise ValueError(
            f"Resume feature layers {checkpoint.get('feature_layers')} do not match "
            f"{list(feature_layers)}"
        )
    decoder.load_state_dict(checkpoint["decoder"])
    if trainable_encoder:
        if checkpoint.get("encoder") is None:
            raise ValueError("Resume checkpoint does not contain trainable encoder weights")
        encoder.load_state_dict(checkpoint["encoder"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    scaler.load_state_dict(checkpoint.get("scaler", {}))
    return int(checkpoint["epoch"]) + 1, float(checkpoint.get("best_val", math.inf))


def main() -> None:
    args = parse_args()
    if args.image_size <= 0 or args.image_size % 16:
        raise ValueError("--image-size must be a positive multiple of 16")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--epochs and --batch-size must be positive")
    if args.num_workers < 0 or args.log_interval <= 0:
        raise ValueError("--num-workers must be non-negative and --log-interval positive")
    if args.hidden_dim % args.decoder_heads:
        raise ValueError("--hidden-dim must be divisible by --decoder-heads")
    if args.train_frame_stride <= 0 or args.val_frame_stride <= 0:
        raise ValueError("Frame strides must be positive")

    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, split_note = build_datasets(args)
    features_cached = args.cache_features and args.feature_backend == "vace"
    train_cache_path = args.output_dir / "features_train.pt"
    val_cache_path = args.output_dir / "features_val.pt"
    train_signature = feature_cache_signature(
        args, "train", len(train_dataset), split_note
    )
    val_signature = feature_cache_signature(args, "val", len(val_dataset), split_note)
    cached_train = (
        load_feature_cache(train_cache_path, train_signature)
        if features_cached
        else None
    )
    cached_val = (
        load_feature_cache(val_cache_path, val_signature) if features_cached else None
    )

    if features_cached and cached_train is not None and cached_val is not None:
        encoder = nn.Identity().to(device)
        input_dim, encoder_stride, trainable_encoder = 1536, 16, False
        encoder_description = "official frozen Wan2.1-VACE features loaded from cache"
    else:
        encoder, input_dim, encoder_stride, trainable_encoder, encoder_description = (
            build_feature_encoder(args, device)
        )
        if features_cached:
            if cached_train is None:
                cached_train = precompute_feature_cache(
                    train_cache_path,
                    train_dataset,
                    train_signature,
                    encoder,
                    args,
                    device,
                )
            if cached_val is None:
                cached_val = precompute_feature_cache(
                    val_cache_path,
                    val_dataset,
                    val_signature,
                    encoder,
                    args,
                    device,
                )
            close = getattr(encoder, "close", None)
            if callable(close):
                close()
            del encoder
            gc.collect()
            torch.cuda.empty_cache()
            encoder = nn.Identity().to(device)
            encoder_description += "; cached once as CPU bfloat16 tensors"

    if features_cached:
        if cached_train is None or cached_val is None:
            raise AssertionError("VACE feature cache construction did not finish")
        train_dataset, val_dataset = cached_train, cached_val
    elif args.cache_features:
        print(
            "--cache-features applies only to frozen VACE; the trainable CNN is "
            "evaluated every optimization step",
            flush=True,
        )

    train_loader = make_loader(
        train_dataset, args.batch_size, True, args.num_workers, device
    )
    val_loader = make_loader(
        val_dataset, args.batch_size, False, args.num_workers, device
    )
    config = HumanDecoderConfig(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.decoder_heads,
        num_layers=args.decoder_layers,
        dropout=args.dropout,
        max_latent_frames=1,
    )
    decoder = HumanQueryTransformer(config).to(device)
    parameters = list(decoder.parameters())
    if trainable_encoder:
        parameters.extend(parameter for parameter in encoder.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    _, _, use_scaler = amp_settings(args, device)
    try:
        scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    start_epoch, best_val = 1, math.inf
    if args.resume is not None:
        start_epoch, best_val = load_checkpoint(
            args.resume,
            decoder,
            encoder,
            trainable_encoder,
            optimizer,
            scaler,
            args.feature_backend,
            args.feature_layers,
            device,
        )
        print(f"Resumed {args.resume} at epoch {start_epoch}", flush=True)

    run_metadata = {
        "status": "training_started",
        "dataset_root": str(args.dataset_root.resolve()),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "validation_split": split_note,
        "feature_backend": args.feature_backend,
        "feature_encoder": encoder_description,
        "trainable_encoder": trainable_encoder,
        "decoder_config": vars(config),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "supervised_outputs": ["global_orient", "body_pose", "betas", "transl"],
        "unsupervised_outputs": [
            "left_hand_pose",
            "right_hand_pose",
            "jaw_pose",
            "expression",
            "object_state",
        ],
        "args": json_ready(vars(args)),
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(run_metadata, indent=2, ensure_ascii=False), flush=True)

    metrics_path = args.output_dir / "metrics.jsonl"
    final_validation: Dict[str, float] = {}
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            train_loader,
            decoder,
            encoder,
            encoder_stride,
            trainable_encoder,
            optimizer,
            scaler,
            args,
            device,
            epoch,
            "train",
        )
        val_metrics = run_epoch(
            val_loader,
            decoder,
            encoder,
            encoder_stride,
            trainable_encoder,
            None,
            scaler,
            args,
            device,
            epoch,
            "validation",
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
            encoder,
            trainable_encoder,
            optimizer,
            scaler,
            epoch,
            best_val,
            args,
            val_metrics,
            config,
        )
        if improved:
            save_checkpoint(
                args.output_dir / "best.pt",
                decoder,
                encoder,
                trainable_encoder,
                optimizer,
                scaler,
                epoch,
                best_val,
                args,
                val_metrics,
                config,
            )
        print(json.dumps(record, indent=2, ensure_ascii=False), flush=True)

    result = {
        "status": "DIVIBODY_REAL_3DPW_TRAINING_OK",
        "completed_epochs": args.epochs,
        "best_validation_total": best_val,
        "final_validation": final_validation,
        "latest_checkpoint": str((args.output_dir / "latest.pt").resolve()),
        "best_checkpoint": str((args.output_dir / "best.pt").resolve()),
        "metrics": str(metrics_path.resolve()),
        "feature_backend": args.feature_backend,
        "supervision": "real 3DPW RGB plus SMPL pose/shape/translation",
    }
    (args.output_dir / "final_validation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    close = getattr(encoder, "close", None)
    if callable(close):
        close()


if __name__ == "__main__":
    main()
