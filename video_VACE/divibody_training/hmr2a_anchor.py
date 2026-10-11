"""Strict, read-only HMR 2.0a anchor loader for DiviBody v8.

The official 4D-Humans source tree is imported at runtime and is never edited.
The original checkpoint is loaded without applying Lightning's on-disk upgrade.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import torch


HMR2A_SHA256 = "2dcf79638109781d1ae5f5c44fee5f55bc83291c210653feead9b7f04fa6f20e"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class HMR2AAnchorExtractor:
    """Load HMR 2.0a and emit v8 RGB tokens plus SMPL anchor predictions."""

    def __init__(
        self,
        *,
        source_root: Path,
        checkpoint: Path,
        model_config: Path,
        hmr2_data_root: Path,
        device: torch.device,
        verify_sha256: bool = True,
    ) -> None:
        paths = (source_root, checkpoint, model_config, hmr2_data_root)
        if any(not Path(path).exists() for path in paths):
            missing = [str(path) for path in paths if not Path(path).exists()]
            raise FileNotFoundError(f"Missing HMR 2.0a resources: {missing}")
        if verify_sha256:
            actual = sha256_file(checkpoint)
            if actual != HMR2A_SHA256:
                raise ValueError(f"HMR 2.0a SHA-256 mismatch: {actual}")

        source_text = str(source_root.resolve())
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        from hmr2.configs import get_config
        from hmr2.models import HMR2

        cfg = get_config(str(model_config), update_cachedir=False)
        cfg.defrost()
        cfg.SMPL.MODEL_PATH = str(hmr2_data_root / "smpl")
        cfg.SMPL.JOINT_REGRESSOR_EXTRA = str(
            hmr2_data_root / "SMPL_to_J19.pkl"
        )
        cfg.SMPL.MEAN_PARAMS = str(hmr2_data_root / "smpl_mean_params.npz")
        cfg.freeze()
        self.model = HMR2.load_from_checkpoint(
            str(checkpoint),
            strict=True,
            cfg=cfg,
            map_location="cpu",
            init_renderer=False,
        ).eval()
        self.model.requires_grad_(False)
        self.device = device
        self.model.to(device)

    @torch.inference_mode()
    def encode(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Infer anchors from square 256x256 RGB crops in the [0, 1] range."""

        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 256, 256):
            raise ValueError("images must be RGB crops [B,3,256,256]")
        if not images.is_floating_point():
            raise TypeError("images must use a floating point dtype")
        if not torch.isfinite(images).all():
            raise ValueError("images contain non-finite values")
        if float(images.min()) < 0.0 or float(images.max()) > 1.0:
            raise ValueError("images must be scaled to [0, 1]")
        images = images.to(self.device, dtype=torch.float32, non_blocking=True)
        mean = images.new_tensor((0.485, 0.456, 0.406)).reshape(1, 3, 1, 1)
        std = images.new_tensor((0.229, 0.224, 0.225)).reshape(1, 3, 1, 1)
        images = (images - mean) / std
        feature_map = self.model.backbone(images[:, :, :, 32:-32])
        if tuple(feature_map.shape[1:]) != (1280, 16, 12):
            raise RuntimeError(f"Unexpected HMR2a feature map {feature_map.shape}")
        smpl_params, camera, _ = self.model.smpl_head(feature_map)
        rotmat = torch.cat(
            (smpl_params["global_orient"], smpl_params["body_pose"]), dim=1
        )
        smpl_output = self.model.smpl(
            **{key: value.float() for key, value in smpl_params.items()},
            pose2rot=False,
        )
        return {
            "rgb_tokens": feature_map.flatten(2).transpose(1, 2),
            "base_rotmat": rotmat,
            "base_betas": smpl_params["betas"],
            "base_camera": camera,
            "base_vertices": smpl_output.vertices,
            "base_joints": smpl_output.joints,
        }

    def close(self) -> None:
        self.model.to("cpu")

    def __enter__(self) -> "HMR2AAnchorExtractor":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
