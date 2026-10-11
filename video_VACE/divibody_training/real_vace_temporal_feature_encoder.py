"""Frozen official Wan/VACE features from genuine continuous RGB clips.

This additive encoder is copied from ``real_vace_feature_encoder.py``.  The
single-frame encoder remains unchanged.  Wan's causal VAE consumes one frame
followed by groups of four, so only ``4n+1`` frame clips are accepted; rejecting
other lengths prevents the upstream VAE from silently dropping tail frames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from real_vace_feature_encoder import FrozenVaceFeatureEncoder


def wan_latent_frames(frame_count: int) -> int:
    """Return the exact Wan latent length for a complete ``4n+1`` RGB clip."""

    if isinstance(frame_count, bool) or not isinstance(frame_count, int):
        raise TypeError("frame_count must be an integer")
    if frame_count < 1 or (frame_count - 1) % 4:
        raise ValueError(
            "Wan temporal clips must contain 4n+1 frames; lengths such as 8 "
            "silently lose their final three frames in the official VAE"
        )
    return 1 + (frame_count - 1) // 4


def reshape_temporal_tokens(
    tokens: torch.Tensor,
    grid_size: tuple[int, int, int],
) -> torch.Tensor:
    """Validate Wan row-major ``F,H,W`` tokens and expose that grid explicitly."""

    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens [B,L,C], got {tuple(tokens.shape)}")
    latent_frames, grid_height, grid_width = grid_size
    if min(grid_size) < 1:
        raise ValueError("Temporal token grid entries must be positive")
    expected = latent_frames * grid_height * grid_width
    if tokens.shape[1] != expected:
        raise ValueError(
            f"Token length {tokens.shape[1]} does not match F*H*W={expected}"
        )
    # VACE uses ``u.flatten(2).transpose(1, 2)`` on [B,C,F,H,W], so the
    # contiguous order is time-major, then row, then column.
    return tokens.reshape(
        tokens.shape[0], latent_frames, grid_height, grid_width, tokens.shape[2]
    )


class FrozenTemporalVaceFeatureEncoder(FrozenVaceFeatureEncoder):
    """Encode real ``4n+1`` RGB clips with frozen official Wan VAE/VACE weights.

    Input is ``[batch, frames, 3, height, width]`` in ``[0,1]``.  Output is
    ``[batch, latent_frames, token_height, token_width, 1536]``.  No temporal
    averaging is performed: the three latent time slices of a 9-frame clip are
    retained independently for the downstream human decoder.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        device: str | torch.device = "cuda",
        image_size: int = 96,
        feature_layers: Iterable[int] = (7, 14, 22),
        timestep: int = 500,
    ) -> None:
        super().__init__(
            checkpoint_dir=checkpoint_dir,
            device=device,
            image_size=image_size,
            feature_layers=feature_layers,
            timestep=timestep,
        )
        # The inherited single-frame cache is keyed only by H,W.  A temporal
        # zero-video latent must also include T or a wrong-length control could
        # be reused after a smoke test.
        self._zero_latents: dict[tuple[int, int, int], torch.Tensor] = {}

    def _zero_latent(self, video: torch.Tensor) -> torch.Tensor:
        key = (int(video.shape[1]), int(video.shape[2]), int(video.shape[3]))
        if key not in self._zero_latents:
            self._zero_latents[key] = self.vae.encode([torch.zeros_like(video)])[0]
        return self._zero_latents[key]

    def encode(self, clips: torch.Tensor) -> torch.Tensor:
        """Return time-preserving VACE features for continuous RGB clips."""

        if clips.ndim != 5 or clips.shape[2] != 3:
            raise ValueError(
                f"Expected RGB clips [B,T,3,H,W], got {tuple(clips.shape)}"
            )
        batch_size, frame_count, _, height, width = clips.shape
        expected_latent_frames = wan_latent_frames(int(frame_count))
        if batch_size < 1:
            raise ValueError("A temporal feature batch cannot be empty")
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"Expected configured image size {self.image_size}x{self.image_size}, "
                f"got {height}x{width}"
            )
        if height % 16 or width % 16:
            raise ValueError(
                f"Clip dimensions must be divisible by 16: {(height, width)}"
            )
        if not bool(torch.isfinite(clips).all()):
            raise ValueError("RGB clips contain NaN or infinity")
        minimum, maximum = float(clips.min()), float(clips.max())
        if minimum < 0.0 or maximum > 1.0:
            raise ValueError(
                f"RGB clips must be in [0,1], observed [{minimum}, {maximum}]"
            )

        self._captured.clear()
        clips = clips.to(
            self.device_name, dtype=torch.float32, non_blocking=True
        )
        videos = [clip.permute(1, 0, 2, 3).mul(2).sub(1) for clip in clips]

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            batched_latents = self.vae.model.encode(
                torch.stack(videos, dim=0), self.vae.scale
            ).float()
            if batched_latents.ndim != 5 or batched_latents.shape[0] != batch_size:
                raise RuntimeError(
                    f"Unexpected Wan VAE output {tuple(batched_latents.shape)}"
                )
            if batched_latents.shape[2] != expected_latent_frames:
                raise RuntimeError(
                    "Wan VAE temporal compression mismatch: expected "
                    f"{expected_latent_frames}, got {batched_latents.shape[2]}"
                )
            latents = list(batched_latents.unbind(dim=0))
            controls = []
            for video, latent in zip(videos, latents):
                invisible = self._zero_latent(video)
                if invisible.shape != latent.shape:
                    raise RuntimeError("Visible and zero-video VAE latents differ")
                mask = torch.zeros(
                    64,
                    latent.shape[1],
                    latent.shape[2],
                    latent.shape[3],
                    device=self.device_name,
                    dtype=latent.dtype,
                )
                control = torch.cat((latent, invisible, mask), dim=0)
                if control.shape[0] != int(self.dit.vace_in_dim):
                    raise RuntimeError(
                        f"VACE control has {control.shape[0]} channels; model expects "
                        f"{self.dit.vace_in_dim}"
                    )
                controls.append(control)

            patch_t, patch_h, patch_w = map(int, self.dit.patch_size)
            if patch_t != 1:
                raise RuntimeError(
                    f"Unsupported temporal DiT patch size {self.dit.patch_size}"
                )
            latent_h, latent_w = map(int, batched_latents.shape[-2:])
            if latent_h % patch_h or latent_w % patch_w:
                raise RuntimeError("Wan latent grid is not divisible by DiT patches")
            grid_size = (
                expected_latent_frames,
                latent_h // patch_h,
                latent_w // patch_w,
            )
            sequence_length = math_product(grid_size)
            empty_text = torch.zeros(
                1, 4096, device=self.device_name, dtype=torch.bfloat16
            )
            self.dit(
                x=latents,
                t=torch.full(
                    (batch_size,),
                    self.timestep,
                    device=self.device_name,
                    dtype=torch.long,
                ),
                vace_context=controls,
                context=[empty_text for _ in latents],
                seq_len=sequence_length,
            )

            if sorted(self._captured) != sorted(self.feature_layers):
                raise RuntimeError(
                    f"Missing VACE features: expected {self.feature_layers}, "
                    f"captured {sorted(self._captured)}"
                )
            averaged = torch.stack(
                [self._captured[layer].float() for layer in self.feature_layers]
            ).mean(dim=0)
            if averaged.shape != (batch_size, sequence_length, self.dit.dim):
                raise RuntimeError(
                    f"Unexpected VACE token shape {tuple(averaged.shape)}"
                )
            result = reshape_temporal_tokens(averaged, grid_size)

        # Clone outside inference mode so a trainable decoder can save this
        # frozen tensor for backward without inference-tensor restrictions.
        return result.detach().clone()

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        return self.encode(clips)


def math_product(values: tuple[int, int, int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result
