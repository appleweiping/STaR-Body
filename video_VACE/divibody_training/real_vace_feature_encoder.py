"""Frozen official Wan/VACE visual features for real RGB training images."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

import run_vace_sdpa as compatibility
from vace.models.wan.modules.model import VaceWanModel
from wan.modules.vae import WanVAE


class FrozenVaceFeatureEncoder(nn.Module):
    """Encode real RGB frames with the official frozen Wan VAE and VACE DiT.

    Images are represented as one-frame videos. Their actual VAE latents are
    provided both as the diffusion input and as the visible VACE conditioning,
    while the text condition remains empty. Intermediate official-pretrained
    DiT block outputs are averaged and returned as differentiable-decoder input.
    Neither official checkpoint nor original model source is modified.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        device: str | torch.device = "cuda",
        image_size: int = 128,
        feature_layers: Iterable[int] = (7, 14, 22),
        timestep: int = 500,
    ) -> None:
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device_name = torch.device(device)
        self.image_size = image_size
        self.feature_layers = tuple(feature_layers)
        self.timestep = timestep

        if self.device_name.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Frozen VACE feature extraction requires CUDA")
        if self.image_size % 16:
            raise ValueError("image_size must be divisible by 16")
        if not self.feature_layers:
            raise ValueError("At least one VACE feature layer is required")

        compatibility.wan_modules.flash_attention = compatibility.scaled_dot_product_attention
        compatibility.wan_attention.flash_attention = compatibility.scaled_dot_product_attention
        compatibility.wan_model.flash_attention = compatibility.scaled_dot_product_attention
        compatibility.wan_clip.flash_attention = compatibility.scaled_dot_product_attention

        print("Loading frozen official Wan VAE weights...", flush=True)
        self.vae = WanVAE(
            vae_pth=str(self.checkpoint_dir / "Wan2.1_VAE.pth"),
            dtype=torch.bfloat16,
            device=str(self.device_name),
        )

        print("Loading frozen official VACE diffusion-transformer weights...", flush=True)
        self.dit = (
            VaceWanModel.from_pretrained(
                str(self.checkpoint_dir),
                torch_dtype=torch.bfloat16,
            )
            .eval()
            .requires_grad_(False)
            .to(self.device_name)
        )
        if min(self.feature_layers) < 0 or max(self.feature_layers) >= len(self.dit.blocks):
            raise ValueError(f"Invalid VACE block indices: {self.feature_layers}")

        self._captured: dict[int, torch.Tensor] = {}
        self._hooks = [
            self.dit.blocks[layer].register_forward_hook(self._make_hook(layer))
            for layer in self.feature_layers
        ]
        self._zero_latents: dict[tuple[int, int], torch.Tensor] = {}
        self.eval().requires_grad_(False)

    def _make_hook(self, layer: int):
        def capture(_module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
            self._captured[layer] = output.detach()

        return capture

    def _zero_latent(self, video: torch.Tensor) -> torch.Tensor:
        shape = tuple(video.shape[-2:])
        if shape not in self._zero_latents:
            self._zero_latents[shape] = self.vae.encode([torch.zeros_like(video)])[0]
        return self._zero_latents[shape]

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Return `[batch, (height / 16) * (width / 16), 1536]` features."""

        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected RGB batch [B,3,H,W], got {tuple(images.shape)}")
        height, width = images.shape[-2:]
        if height % 16 or width % 16:
            raise ValueError(f"Image dimensions must be divisible by 16: {(height, width)}")

        self._captured.clear()
        images = images.to(self.device_name, dtype=torch.float32, non_blocking=True)
        videos = [(image.mul(2).sub(1)).unsqueeze(1) for image in images]

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # The public WanVAE.encode loops over videos one at a time. Its
            # underlying official encoder supports a batch directly, which is
            # substantially faster for real-data training while preserving the
            # same checkpoint, scale normalization, and frozen parameters.
            batched_latents = self.vae.model.encode(
                torch.stack(videos, dim=0), self.vae.scale
            ).float()
            latents = list(batched_latents.unbind(dim=0))
            controls = []
            for video, latent in zip(videos, latents):
                invisible = self._zero_latent(video)
                mask = torch.zeros(
                    64,
                    latent.shape[1],
                    latent.shape[2],
                    latent.shape[3],
                    device=self.device_name,
                    dtype=latent.dtype,
                )
                controls.append(torch.cat((latent, invisible, mask), dim=0))

            sequence_length = (height // 16) * (width // 16)
            empty_text = torch.zeros(
                1, 4096, device=self.device_name, dtype=torch.bfloat16
            )
            self.dit(
                x=latents,
                t=torch.full(
                    (len(latents),),
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

        # Clone outside inference mode: decoder weight gradients must be allowed
        # to save this frozen visual input during their backward pass.
        return averaged.detach().clone()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode(images)

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
