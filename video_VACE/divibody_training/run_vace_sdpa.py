"""Launch a VACE script with PyTorch SDPA when flash-attention is unavailable."""

from __future__ import annotations

import os
import runpy
import sys

import torch
import torch.nn.functional as F

import wan.modules as wan_modules
import wan.modules.attention as wan_attention
import wan.modules.clip as wan_clip
import wan.modules.model as wan_model


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: torch.Tensor | None = None,
    k_lens: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    q_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    version: int | None = None,
    fa_version: int | None = None,
) -> torch.Tensor:
    """Match Wan flash_attention's interface using native PyTorch attention."""

    del deterministic, version, fa_version

    output_dtype = q.dtype
    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)

    if q_scale is not None:
        q = q * q_scale

    if q.size(1) != k.size(1):
        if q.size(1) % k.size(1):
            raise ValueError("Query and key head counts are incompatible")
        repeats = q.size(1) // k.size(1)
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

    if q_lens is not None and bool(torch.all(q_lens >= q.size(-2))):
        q_lens = None
    if k_lens is not None and bool(torch.all(k_lens >= k.size(-2))):
        k_lens = None

    mask = None
    if q_lens is not None or k_lens is not None or window_size != (-1, -1):
        q_index = torch.arange(q.size(-2), device=q.device)[None, :, None]
        k_index = torch.arange(k.size(-2), device=q.device)[None, None, :]
        mask = torch.ones(
            (q.size(0), q.size(-2), k.size(-2)),
            dtype=torch.bool,
            device=q.device,
        )
        if q_lens is not None:
            mask &= q_index < q_lens.to(q.device)[:, None, None]
        if k_lens is not None:
            mask &= k_index < k_lens.to(q.device)[:, None, None]

        left, right = window_size
        if left >= 0:
            mask &= k_index >= q_index + k.size(-2) - q.size(-2) - left
        if right >= 0:
            mask &= k_index <= q_index + k.size(-2) - q.size(-2) + right
        if causal:
            mask &= k_index <= q_index
            causal = False
        mask = mask[:, None]

    result = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=dropout_p,
        is_causal=causal,
        scale=softmax_scale,
    )
    return result.transpose(1, 2).contiguous().to(output_dtype)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: run_vace_sdpa.py VACE_SCRIPT [VACE_ARGS ...]")

    wan_modules.flash_attention = scaled_dot_product_attention
    wan_attention.flash_attention = scaled_dot_product_attention
    wan_model.flash_attention = scaled_dot_product_attention
    wan_clip.flash_attention = scaled_dot_product_attention

    script = os.path.abspath(sys.argv[1])
    sys.argv = sys.argv[1:]
    sys.path.insert(0, os.path.dirname(script))
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
