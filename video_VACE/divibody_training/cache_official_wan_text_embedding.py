"""Cache an official Wan UMT5 prompt embedding for frozen feature extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

import run_vace_sdpa as compatibility
from wan.modules.t5 import T5EncoderModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--text-length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite text cache: {args.output}")
    if args.text_length <= 0:
        raise ValueError("text-length must be positive")
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    checkpoint = checkpoint_dir / "models_t5_umt5-xxl-enc-bf16.pth"
    tokenizer = checkpoint_dir / "google" / "umt5-xxl"
    if not checkpoint.is_file() or not tokenizer.is_dir():
        raise FileNotFoundError("Wan UMT5 checkpoint/tokenizer assets are incomplete")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    # The official attention kernels are patched in the same way as every
    # existing VACE entry point in this repository.
    compatibility.wan_modules.flash_attention = compatibility.scaled_dot_product_attention
    compatibility.wan_attention.flash_attention = compatibility.scaled_dot_product_attention
    compatibility.wan_model.flash_attention = compatibility.scaled_dot_product_attention
    compatibility.wan_clip.flash_attention = compatibility.scaled_dot_product_attention

    started = time.monotonic()
    encoder = T5EncoderModel(
        text_len=args.text_length,
        dtype=torch.bfloat16,
        # Match the official VACE constructor: creating UMT5 directly on CUDA
        # first allocates FP32 parameters and can transiently exceed 24 GiB.
        device=torch.device("cpu"),
        checkpoint_path=str(checkpoint),
        tokenizer_path=str(tokenizer),
    )
    encoder.model.to(device)
    with torch.inference_mode():
        embedding = encoder([args.prompt], device)[0].detach().to(
            device="cpu", dtype=torch.bfloat16
        )
    if embedding.ndim != 2 or embedding.shape[1] != 4096:
        raise RuntimeError(f"Unexpected Wan text embedding {tuple(embedding.shape)}")
    if not bool(torch.isfinite(embedding.float()).all()):
        raise FloatingPointError("Wan text embedding is non-finite")

    payload = {
        "format": "divibody-official-wan-umt5-embedding-v1",
        "prompt": args.prompt,
        "embedding": embedding,
        "embedding_shape": list(embedding.shape),
        "embedding_dtype": str(embedding.dtype),
        "text_length_capacity": args.text_length,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "tokenizer_path": str(tokenizer),
        "runtime_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, args.output)
    result = {
        key: value for key, value in payload.items() if key != "embedding"
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
