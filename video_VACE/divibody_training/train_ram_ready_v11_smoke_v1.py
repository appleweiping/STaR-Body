"""Validate a RAM ready tree, then launch an additive V11 small-sample run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from divibody_training.ram_ready_adapter_v1 import RAMReadyPaths


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ready-root", type=Path, required=True)
    p.add_argument("--train-vace-cache", type=Path, required=True)
    p.add_argument("--train-hmr-sidecar", type=Path, required=True)
    p.add_argument("--val-vace-cache", type=Path, required=True)
    p.add_argument("--val-hmr-sidecar", type=Path, required=True)
    p.add_argument("--init-checkpoint", type=Path, required=True)
    p.add_argument("--smpl-model-root", type=Path, required=True)
    p.add_argument("--joint-regressor", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-train-windows", type=int, default=64)
    p.add_argument("--max-val-windows", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    ready = RAMReadyPaths.discover(args.ready_root)
    if not (ready.threedpw / "imageFiles").is_dir():
        raise FileNotFoundError("Ready 3DPW imageFiles missing")
    for value in (
        args.train_vace_cache, args.train_hmr_sidecar, args.val_vace_cache,
        args.val_hmr_sidecar, args.init_checkpoint, args.joint_regressor,
    ):
        if not value.is_file():
            raise FileNotFoundError(value)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    command = [
        sys.executable, "-m", "divibody_training.train_divibody_rgb_vace_hmr2a_v11_raw_small",
        "--train-vace-cache", str(args.train_vace_cache),
        "--train-hmr-sidecar", str(args.train_hmr_sidecar),
        "--val-vace-cache", str(args.val_vace_cache),
        "--val-hmr-sidecar", str(args.val_hmr_sidecar),
        "--init-checkpoint", str(args.init_checkpoint),
        "--smpl-model-root", str(args.smpl_model_root),
        "--joint-regressor", str(args.joint_regressor),
        "--output-dir", str(args.output_dir),
        "--epochs", str(args.epochs),
        "--max-train-windows", str(args.max_train_windows),
        "--max-val-windows", str(args.max_val_windows),
        "--batch-size", str(args.batch_size),
        "--val-batch-size", str(max(args.batch_size, 16)),
        "--num-workers", "2", "--device", args.device,
    ]
    manifest = {
        "format": "ram-ready-v11-smoke-v1",
        "ready_root": str(ready.root),
        "resolved_3dpw": str(ready.threedpw.resolve()),
        "command": command,
        "existing_sources_modified": False,
    }
    environment = os.environ.copy()
    source_root = Path(__file__).resolve().parents[1]
    module_root = source_root / "divibody_training"
    project_root = source_root.parent
    wan_root = project_root / "dependencies" / "Wan2.1"
    current_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (
            str(source_root), str(module_root), str(project_root), str(wan_root), current_pythonpath
        ) if value
    )
    subprocess.run(command, check=True, cwd=source_root, env=environment)
    (args.output_dir / "ram_ready_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
