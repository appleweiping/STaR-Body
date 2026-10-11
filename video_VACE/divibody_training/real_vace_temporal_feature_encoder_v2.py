"""Protocol-aware copy of the continuous-video VACE feature encoder.

``real_vace_temporal_feature_encoder.py`` is intentionally unchanged.  This
version makes the diffusion state and text condition explicit so that coherent
``t=0/clean`` and ``t>0/flow-noised`` protocols can be compared against the old
``t=500/clean/zero-text`` baseline without mixing cache identities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import torch

from real_vace_temporal_feature_encoder import (
    FrozenTemporalVaceFeatureEncoder,
    math_product,
    reshape_temporal_tokens,
    wan_latent_frames,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


DIFFUSION_PROTOCOLS = ("clean", "flow_noised")
TEXT_CACHE_FORMAT = "divibody-official-wan-umt5-embedding-v1"


def validate_noise_seeds(
    values: Sequence[int] | torch.Tensor | None,
    batch_size: int,
    required: bool,
) -> tuple[int, ...] | None:
    """Validate per-clip seeds so features do not depend on batch composition."""

    if values is None:
        if required:
            raise ValueError(
                "flow_noised features require one deterministic noise seed per clip"
            )
        return None
    if isinstance(values, torch.Tensor):
        if values.ndim != 1:
            raise ValueError("noise_seeds tensor must be one-dimensional")
        values = values.detach().cpu().tolist()
    seeds = tuple(int(value) for value in values)
    if len(seeds) != batch_size:
        raise ValueError(
            f"Expected {batch_size} noise seeds, received {len(seeds)}"
        )
    if any(seed < 0 or seed >= 2**63 for seed in seeds):
        raise ValueError("noise seeds must lie in [0, 2**63)")
    return seeds


class ProtocolAwareTemporalVaceFeatureEncoder(FrozenTemporalVaceFeatureEncoder):
    """Frozen continuous VACE features with explicit noise/text conditioning."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        device: str | torch.device = "cuda",
        image_size: int = 96,
        feature_layers: Iterable[int] = (7, 14, 22),
        timestep: int = 0,
        diffusion_input_protocol: str = "clean",
        text_embedding_path: str | Path | None = None,
    ) -> None:
        if diffusion_input_protocol not in DIFFUSION_PROTOCOLS:
            raise ValueError(
                f"diffusion_input_protocol must be one of {DIFFUSION_PROTOCOLS}"
            )
        if isinstance(timestep, bool) or not isinstance(timestep, int):
            raise TypeError("timestep must be an integer")
        if timestep < 0 or timestep >= 1000:
            raise ValueError("Wan timestep must lie in [0,1000)")
        super().__init__(
            checkpoint_dir=checkpoint_dir,
            device=device,
            image_size=image_size,
            feature_layers=feature_layers,
            timestep=timestep,
        )
        self.diffusion_input_protocol = diffusion_input_protocol
        self.text_embedding_path = (
            None
            if text_embedding_path is None
            else Path(text_embedding_path).expanduser().resolve()
        )
        self._text_embedding, self.text_protocol, self.text_prompt = (
            self._load_text_embedding(self.text_embedding_path)
        )
        self.scheduler = (
            FlowUniPCMultistepScheduler(
                num_train_timesteps=1000,
                shift=1,
                use_dynamic_shifting=False,
            )
            if self.diffusion_input_protocol == "flow_noised"
            else None
        )

    @staticmethod
    def _load_text_embedding(
        path: Path | None,
    ) -> tuple[torch.Tensor, str, str | None]:
        if path is None:
            return torch.zeros(1, 4096, dtype=torch.bfloat16), "zero_token", None
        if not path.is_file():
            raise FileNotFoundError(f"Missing Wan text embedding cache: {path}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if payload.get("format") != TEXT_CACHE_FORMAT:
            raise ValueError("Not an official cached Wan UMT5 embedding")
        embedding = payload.get("embedding")
        if (
            not isinstance(embedding, torch.Tensor)
            or embedding.ndim != 2
            or embedding.shape[1] != 4096
            or embedding.shape[0] < 1
            or embedding.shape[0] > 512
            or not bool(torch.isfinite(embedding.float()).all())
        ):
            raise ValueError("Invalid cached Wan UMT5 embedding tensor")
        prompt = payload.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("Wan UMT5 cache does not record its prompt")
        return embedding.to(dtype=torch.bfloat16).contiguous(), "official_umt5", prompt

    @staticmethod
    def _deterministic_noise(
        reference: torch.Tensor,
        seeds: tuple[int, ...],
    ) -> torch.Tensor:
        chunks = []
        sample_shape = tuple(reference.shape[1:])
        for seed in seeds:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            chunks.append(
                torch.randn(
                    sample_shape,
                    generator=generator,
                    device="cpu",
                    dtype=torch.float32,
                )
            )
        return torch.stack(chunks).to(
            device=reference.device, dtype=reference.dtype, non_blocking=True
        )

    def protocol_signature(self) -> dict[str, object]:
        return {
            "diffusion_input_protocol": self.diffusion_input_protocol,
            "vace_timestep": self.timestep,
            "noise_protocol": (
                "per-clip CPU torch.Generator standard-normal seed; official "
                "FlowUniPCMultistepScheduler.add_noise, shift=1"
                if self.diffusion_input_protocol == "flow_noised"
                else "none"
            ),
            "text_protocol": self.text_protocol,
            "text_prompt": self.text_prompt,
            "text_embedding_path": (
                None
                if self.text_embedding_path is None
                else str(self.text_embedding_path)
            ),
            "mask_protocol": (
                "all-black retained RGB: concat(VAE(real), VAE(zero), zeros64)"
            ),
            "vae_batching": "batched official WanVAE_.encode",
            "feature_layer_reduction": "incremental_fp32_arithmetic_mean",
        }

    def encode(
        self,
        clips: torch.Tensor,
        noise_seeds: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``[B,Fz,Htok,Wtok,1536]`` protocol-labelled features."""

        if clips.ndim != 5 or clips.shape[2] != 3:
            raise ValueError(
                f"Expected RGB clips [B,T,3,H,W], got {tuple(clips.shape)}"
            )
        batch_size, frame_count, _, height, width = clips.shape
        expected_latent_frames = wan_latent_frames(int(frame_count))
        seeds = validate_noise_seeds(
            noise_seeds,
            int(batch_size),
            required=self.diffusion_input_protocol == "flow_noised",
        )
        if batch_size < 1:
            raise ValueError("A temporal feature batch cannot be empty")
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"Expected configured image size {self.image_size}x{self.image_size}, "
                f"got {height}x{width}"
            )
        if height % 16 or width % 16:
            raise ValueError("Clip spatial dimensions must be divisible by 16")
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
            clean_batch = self.vae.model.encode(
                torch.stack(videos, dim=0), self.vae.scale
            ).float()
            if (
                clean_batch.ndim != 5
                or clean_batch.shape[0] != batch_size
                or clean_batch.shape[2] != expected_latent_frames
            ):
                raise RuntimeError(
                    f"Unexpected continuous Wan VAE output {tuple(clean_batch.shape)}"
                )
            clean_latents = list(clean_batch.unbind(dim=0))
            if self.diffusion_input_protocol == "flow_noised":
                assert seeds is not None and self.scheduler is not None
                noise = self._deterministic_noise(clean_batch, seeds)
                diffusion_batch = self.scheduler.add_noise(
                    clean_batch,
                    noise,
                    torch.full(
                        (batch_size,),
                        self.timestep,
                        device=self.device_name,
                        dtype=torch.long,
                    ),
                )
            else:
                diffusion_batch = clean_batch
            diffusion_latents = list(diffusion_batch.unbind(dim=0))

            controls = []
            for video, clean_latent in zip(videos, clean_latents):
                invisible = self._zero_latent(video)
                if invisible.shape != clean_latent.shape:
                    raise RuntimeError("Visible and zero-video VAE latents differ")
                mask = torch.zeros(
                    64,
                    *clean_latent.shape[1:],
                    device=self.device_name,
                    dtype=clean_latent.dtype,
                )
                control = torch.cat((clean_latent, invisible, mask), dim=0)
                if control.shape[0] != int(self.dit.vace_in_dim):
                    raise RuntimeError("VACE control channel count mismatch")
                controls.append(control)

            patch_t, patch_h, patch_w = map(int, self.dit.patch_size)
            if patch_t != 1:
                raise RuntimeError("Only Wan temporal patch size 1 is supported")
            latent_h, latent_w = map(int, clean_batch.shape[-2:])
            if latent_h % patch_h or latent_w % patch_w:
                raise RuntimeError("Wan latent grid is not divisible by DiT patches")
            grid_size = (
                expected_latent_frames,
                latent_h // patch_h,
                latent_w // patch_w,
            )
            sequence_length = math_product(grid_size)
            context = self._text_embedding.to(
                device=self.device_name, dtype=torch.bfloat16, non_blocking=True
            )
            self.dit(
                x=diffusion_latents,
                t=torch.full(
                    (batch_size,),
                    self.timestep,
                    device=self.device_name,
                    dtype=torch.long,
                ),
                vace_context=controls,
                context=[context for _ in clean_latents],
                seq_len=sequence_length,
            )
            if sorted(self._captured) != sorted(self.feature_layers):
                raise RuntimeError(
                    f"Missing VACE hooks: captured {sorted(self._captured)}"
                )
            averaged = None
            for layer in self.feature_layers:
                value = self._captured[layer].float()
                averaged = value.clone() if averaged is None else averaged.add_(value)
            assert averaged is not None
            averaged.div_(len(self.feature_layers))
            if averaged.shape != (batch_size, sequence_length, self.dit.dim):
                raise RuntimeError(
                    f"Unexpected VACE token shape {tuple(averaged.shape)}"
                )
            result = reshape_temporal_tokens(averaged, grid_size)
        return result.detach().clone()

    def forward(
        self,
        clips: torch.Tensor,
        noise_seeds: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encode(clips, noise_seeds=noise_seeds)
