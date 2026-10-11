#!/usr/bin/env python3
"""Fail-fast audit for the complete official 3DPW train/validation splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_standard_3dpw import audit_split, enforce_complete_test


EXPECTED = {
    "train": {
        "sequences": 24,
        "tracks": 34,
        "annotation_frames": 17_904,
        "expected_valid_person_frames": 22_735,
        "available_valid_person_frames": 22_735,
        "required_unique_rgb_frames": 17_642,
        "available_unique_rgb_frames": 17_642,
        "missing_unique_rgb_frames": 0,
        "expected_acceleration_triplets": 22_481,
    },
    "validation": {
        "sequences": 3,
        "tracks": 5,
        "annotation_frames": 2_106,
        "expected_valid_person_frames": 3_162,
        "available_valid_person_frames": 3_162,
        "required_unique_rgb_frames": 2_073,
        "available_unique_rgb_frames": 2_073,
        "missing_unique_rgb_frames": 0,
        "expected_acceleration_triplets": 3_110,
    },
    "test": {
        "sequences": 24,
        "tracks": 37,
        "annotation_frames": 26_240,
        "expected_valid_person_frames": 35_515,
        "available_valid_person_frames": 35_515,
        "required_unique_rgb_frames": 24_547,
        "available_unique_rgb_frames": 24_547,
        "missing_unique_rgb_frames": 0,
        "expected_acceleration_triplets": 34_752,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify complete official 3DPW train/validation/test data"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audits = {split: audit_split(args.data_root, split) for split in EXPECTED}
    errors: list[str] = []
    for split, expected in EXPECTED.items():
        audit = audits[split]
        for key, expected_value in expected.items():
            if audit[key] != expected_value:
                errors.append(
                    f"{split}.{key}={audit[key]!r}, expected {expected_value!r}"
                )
        if audit["invalid_gender_tracks"]:
            errors.append(f"{split} has invalid gender metadata")
        if audit["invalid_camera_frame_count"]:
            errors.append(f"{split} has invalid camera frames")

    try:
        enforce_complete_test(audits["test"], allow_incomplete=False)
    except RuntimeError as error:
        errors.append(str(error))

    sequence_sets = {
        split: set(audit["sequence_names"]) for split, audit in audits.items()
    }
    overlaps = {
        "train_validation": sorted(
            sequence_sets["train"] & sequence_sets["validation"]
        ),
        "train_test": sorted(sequence_sets["train"] & sequence_sets["test"]),
        "validation_test": sorted(sequence_sets["validation"] & sequence_sets["test"]),
    }
    for pair, overlap in overlaps.items():
        if overlap:
            errors.append(f"{pair} sequence overlap: {overlap}")

    result: dict[str, Any] = {
        "status": (
            "DIVIBODY_FULL_3DPW_TRAINING_DATA_OK"
            if not errors
            else "DIVIBODY_FULL_3DPW_TRAINING_DATA_INVALID"
        ),
        "data_root": str(args.data_root.expanduser().resolve()),
        "splits": audits,
        "sequence_overlaps": overlaps,
        "errors": errors,
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if errors:
        raise RuntimeError("Complete 3DPW training-data audit failed")


if __name__ == "__main__":
    main()
