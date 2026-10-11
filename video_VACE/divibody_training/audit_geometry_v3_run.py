"""Independently check persisted selection, full coverage and NPZ metric means.

This audits numerical artifacts; it is not a second model inference engine and
does not imply exact equivalence to the HMR2 paper's published evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_divibody_real_3dpw_temporal import sha256_file


METRICS = (
    "mpjpe_mm", "pa_mpjpe_mm", "pve_mm", "pelvis_aligned_pve_mm",
    "acceleration_error_mm_per_frame2",
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit_run(run_dir: Path) -> dict:
    config = read_json(run_dir / "run_config.json")
    final = read_json(run_dir / "final_validation.json")
    if final["status"] != "DIVIBODY_GEOMETRY_3DPW_TRAINING_OK":
        raise ValueError("Training did not complete successfully")
    if final.get("development_only") or config.get("development_only"):
        raise ValueError("Cannot audit a smoke experiment as a formal run")
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    epochs = [r["epoch"] for r in records]
    if epochs != list(range(final["completed_epochs"] + 1)):
        raise ValueError("Missing or duplicated training epochs")
    if any(r["validation"]["frames"] != 3162 for r in records):
        raise ValueError("Selection did not use complete validation")
    selected = min(records, key=lambda r: r["validation"]["mpjpe_mm"])
    if selected["epoch"] != final["best_epoch"]:
        raise ValueError("Final checkpoint was not selected by minimum validation MPJPE")
    checkpoint_path = run_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["epoch"] != selected["epoch"]:
        raise ValueError("Saved checkpoint epoch differs from validation selection")
    if checkpoint.get("test_used_for_model_selection") is not False:
        raise ValueError("Checkpoint does not state validation-only model selection")
    if checkpoint["train_cache_signature"]["split"] != "train":
        raise ValueError("Wrong training split")
    if checkpoint["val_cache_signature"]["split"] != "validation":
        raise ValueError("Wrong selection split")
    digest = sha256_file(checkpoint_path)
    split_reports = {}
    for split, expected_frames, expected_triplets in (
        ("validation", 3162, 3110), ("test", 35515, 34752)
    ):
        report_path = run_dir / f"geometry_3dpw_{split}.json"
        array_path = report_path.with_suffix(".npz")
        report = read_json(report_path)
        if report["coverage"] != f"complete_{split}" or report.get("development_only"):
            raise ValueError(f"Not a complete formal {split} evaluation")
        if report["evaluated_person_frames"] != expected_frames:
            raise ValueError(f"Incorrect {split} frame coverage")
        if report["acceleration_valid_triplets"] != expected_triplets:
            raise ValueError(f"Incorrect {split} triplet coverage")
        if report["checkpoint"]["sha256"] != digest:
            raise ValueError(f"{split} evaluated a different checkpoint")
        with np.load(array_path, allow_pickle=False) as arrays:
            keys = list(zip(arrays["sequence"].tolist(), arrays["person_id"].tolist(),
                            arrays["frame_id"].tolist()))
            if len(keys) != expected_frames or len(set(keys)) != expected_frames:
                raise ValueError(f"Duplicated or missing {split} prediction identities")
            if arrays["pred_rotmat24_camera"].shape != (expected_frames, 24, 3, 3):
                raise ValueError("Full 24-joint rotations were not saved")
            if len(arrays["acceleration_center_sample_index"]) != expected_triplets:
                raise ValueError("Acceleration center indices have wrong size")
            recomputed = {}
            for metric in METRICS:
                values = arrays[metric]
                expected_size = expected_triplets if metric.startswith("acceleration") else expected_frames
                if values.shape != (expected_size,) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid raw {split}/{metric} array")
                mean = float(values.mean())
                if abs(mean - report["metrics"][metric]) > 1e-9:
                    raise ValueError(f"Raw NPZ mean disagrees with {split}/{metric}")
                recomputed[metric] = mean
        validation_delta = None
        if split == "validation":
            validation_delta = {
                metric: recomputed[metric] - selected["validation"][metric]
                for metric in ("mpjpe_mm", "pa_mpjpe_mm", "pve_mm")
            }
            if max(abs(value) for value in validation_delta.values()) > 0.001:
                raise ValueError("Independent validation differs from training selection")
        split_reports[split] = {
            "frames": expected_frames, "acceleration_triplets": expected_triplets,
            "metrics_recomputed_from_npz": recomputed,
            "training_validation_delta_mm": validation_delta,
            "report_sha256": sha256_file(report_path),
            "raw_npz_sha256": sha256_file(array_path),
        }
    return {
        "status": "DIVIBODY_GEOMETRY_V3_RUN_AUDIT_OK",
        "run_dir": str(run_dir.resolve()), "completed_epochs": final["completed_epochs"],
        "selected_epoch": selected["epoch"], "selection_metric": "full_validation_mpjpe_mm",
        "test_used_for_model_selection": False,
        "test_set_previously_observed": checkpoint.get("test_set_previously_observed"),
        "checkpoint_sha256": digest, "initialization": config["initialization"],
        "initialization_validation": records[0]["validation"],
        "splits": split_reports,
        "checks": [
            "complete ordered epoch history; first minimum validation MPJPE wins",
            "3162 validation frames at every selection epoch",
            "fixed checkpoint hash for validation and test",
            "35515 distinct test person-frame keys and 34752 acceleration triplets",
            "all raw per-sample NPZ means equal reported JSON to 1e-9 mm",
            "independent full validation agrees with training selection to 0.001 mm",
        ],
        "scope": "saved-result consistency audit, not HMR2 official-protocol reproduction",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Audit output already exists")
    report = audit_run(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
