"""CPU-only wrapper for unmodified HMR2 PoseTrack preprocessing and audit.

Call this module from video_VACE on the server. The official preprocessing
source and original annotations are read only. Its coco_extract function runs
against byte-identical annotation copies in one exclusively created directory.
No image copies, model imports, GPU use, evaluation-rule tuning or downloads.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

POSETRACK_TO_HMR44 = (0, 37, 38, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11)
OFFICIAL_RELATIVE_SOURCE = Path("hmr2/datasets/preprocess/posetrack_to_npz.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "image-root", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    annotations = parser.add_mutually_exclusive_group(required=True)
    annotations.add_argument("--annotations-dir", type=Path,
                             help="Raw val-only directory of per-video JSON files")
    annotations.add_argument("--annotation-file", type=Path,
                             help="One merged val JSON; never pass a directory also containing train JSON")
    parser.add_argument("--expected-source-sha256",
                        help="Optional trusted SHA256 pin for the official preprocessing file")
    args = parser.parse_args(argv)
    for name in ("source_root", "annotations_dir", "annotation_file", "image_root", "output_dir"):
        path = getattr(args, name)
        if path is not None:
            setattr(args, name, path.expanduser().resolve())
    if args.expected_source_sha256 is not None:
        args.expected_source_sha256 = args.expected_source_sha256.lower()
        if len(args.expected_source_sha256) != 64 or any(
                char not in "0123456789abcdef" for char in args.expected_source_sha256):
            parser.error("expected-source-sha256 must be a 64-character hex digest")
    return args


def check_official_recipe(source_path: Path) -> dict[str, Any]:
    """Reject an unexpected local preprocessing variant, without editing it."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == "coco_extract"]
    if len(functions) != 1:
        raise ValueError("Expected exactly one official coco_extract function")
    function = functions[0]
    assignments: dict[str, list[Any]] = {}
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in (
                    "scaleFactor", "SPLIT", "posetrack_to_op_extra"):
                assignments.setdefault(target.id, []).append(ast.literal_eval(node.value))
    expected = {"scaleFactor": [1.2], "SPLIT": ["val"],
                "posetrack_to_op_extra": [list(POSETRACK_TO_HMR44)]}
    if assignments != expected:
        raise ValueError(f"Unexpected official preprocessing constants: {assignments}")
    tests = [ast.unparse(node.test) for node in ast.walk(function) if isinstance(node, ast.If)]
    body_filter = "sum(keypoints[5:, 2] > 0) < 12"
    if body_filter not in tests:
        raise ValueError("Official twelve-annotated-body-joint filter was not found")
    return {"function": "coco_extract", "split": "val", "scale_factor": 1.2,
            "source_body_filter": body_filter, "mapping_17_to_44": list(POSETRACK_TO_HMR44),
            "source_not_modified": True}


def prepare_and_audit(args: argparse.Namespace, report: dict[str, Any]) -> None:
    import numpy as np

    if not args.source_root.is_dir():
        raise FileNotFoundError("source-root must already exist")
    if args.annotation_file is not None:
        if not args.annotation_file.is_file():
            raise FileNotFoundError(f"Annotation file not found: {args.annotation_file}")
        source_jsons = [args.annotation_file]
    else:
        if not args.annotations_dir.is_dir():
            raise FileNotFoundError(f"Annotation directory not found: {args.annotations_dir}")
        source_jsons = list(args.annotations_dir.glob("*.json"))
    if not args.image_root.is_dir():
        raise FileNotFoundError(f"Image root does not exist: {args.image_root}")
    official_path = (args.source_root / OFFICIAL_RELATIVE_SOURCE).resolve()
    if not official_path.is_relative_to(args.source_root) or not official_path.is_file():
        raise FileNotFoundError(f"Official source is not inside the selected source tree: {official_path}")
    official_hash = sha256_file(official_path)
    report["official_preprocess_source"] = {"path": str(official_path), "sha256": official_hash,
                                            "expected_sha256": args.expected_source_sha256}
    if args.expected_source_sha256 and args.expected_source_sha256 != official_hash:
        raise ValueError("Official preprocessing source SHA256 does not match the trusted pin")
    report["recipe"] = check_official_recipe(official_path)
    if not source_jsons:
        raise FileNotFoundError(f"No *.json annotation files found in {args.annotations_dir}")
    layout_root = args.output_dir / "official_input_layout"
    staged_annotations = layout_root / "posetrack_data/annotations/val"
    staged_annotations.mkdir(parents=True, exist_ok=False)
    input_manifest: list[dict[str, Any]] = []
    for path in source_jsons:
        destination = staged_annotations / path.name
        with path.open("rb") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target)
        digest = sha256_file(path)
        if sha256_file(destination) != digest:
            raise RuntimeError(f"Staged annotation copy differs from its original: {path}")
        input_manifest.append({"original_path": str(path), "staged_name": path.name,
                               "sha256": digest, "bytes": path.stat().st_size})
    report["annotation_inputs"] = input_manifest
    # Match the official glob iteration over its input layout; never reorder NPZ rows.
    staged_jsons = list(staged_annotations.glob("*.json"))
    expected_rows: list[dict[str, Any]] = []
    raw_person_count = raw_image_count = 0
    for json_path in staged_jsons:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        images = {image["id"]: image for image in payload["images"]}
        raw_image_count += len(payload["images"])
        for annotation in payload["annotations"]:
            raw_person_count += 1
            keypoints = np.reshape(annotation["keypoints"], (17, 3)).copy()
            keypoints[keypoints[:, 2] > 0, 2] = 1
            if sum(keypoints[5:, 2] > 0) < 12:
                continue
            bbox = annotation["bbox"]
            image_name = str(images[annotation["image_id"]]["file_name"])
            expected_rows.append({
                "annotation_file": json_path.name, "annotation_id": annotation.get("id"),
                "image_id": annotation["image_id"], "track_id": annotation.get("track_id"),
                "image_name": image_name, "bbox": bbox, "keypoints": keypoints,
                "center": [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2],
                "scale": 1.2 * np.array([bbox[2], bbox[3]]),
            })
    if not expected_rows:
        raise RuntimeError("No person rows satisfy the unchanged official body12 filter")
    report["coverage"] = {"annotation_json_files": len(staged_jsons),
                          "raw_image_entries": raw_image_count,
                          "raw_person_annotations": raw_person_count,
                          "official_body12_selected_persons": len(expected_rows),
                          "body12_filter_excluded_persons": raw_person_count - len(expected_rows)}
    metadata_dir = args.output_dir / "official_metadata"
    metadata_dir.mkdir(exist_ok=False)
    npz_path = metadata_dir / "posetrack_2018_val.npz"
    if npz_path.exists():
        raise FileExistsError(f"Output NPZ already exists: {npz_path}")
    specification = importlib.util.spec_from_file_location("official_posetrack_preprocess_v41", official_path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load official preprocessing source: {official_path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    # This is the unchanged official function, not a new NPZ-writing recipe.
    module.coco_extract(str(layout_root), str(metadata_dir))
    if sha256_file(official_path) != official_hash:
        raise RuntimeError("Official source changed during preprocessing")
    if not npz_path.is_file():
        raise RuntimeError("Official preprocessing did not create posetrack_2018_val.npz")
    report["generated_npz"] = {"path": str(npz_path), "sha256": sha256_file(npz_path),
                               "bytes": npz_path.stat().st_size,
                               "origin": "unmodified official coco_extract; not released NPZ bytes"}
    with np.load(npz_path, allow_pickle=False) as data:
        count = len(expected_rows)
        required = {"imgname", "center", "scale", "part", "body_keypoints_2d", "extra_keypoints_2d"}
        if not required.issubset(data.files):
            raise RuntimeError(f"Official NPZ missing keys: {required - set(data.files)}")
        np.testing.assert_array_equal(data["imgname"], [row["image_name"] for row in expected_rows])
        np.testing.assert_allclose(data["center"], [row["center"] for row in expected_rows], rtol=0, atol=0)
        np.testing.assert_allclose(data["scale"], [row["scale"] for row in expected_rows], rtol=0, atol=0)
        np.testing.assert_array_equal(data["part"], np.stack([row["keypoints"] for row in expected_rows]))
        targets = np.concatenate((data["body_keypoints_2d"], data["extra_keypoints_2d"]), axis=1)
        expected_targets = np.zeros((count, 44, 3))
        expected_targets[:, list(POSETRACK_TO_HMR44)] = np.stack([row["keypoints"] for row in expected_rows])
        np.testing.assert_array_equal(targets, expected_targets)
        if targets.shape != (count, 44, 3) or not np.isfinite(targets).all():
            raise RuntimeError("Invalid official 44-point GT array")
        if data["scale"].shape != (count, 2) or not np.isfinite(data["scale"]).all() or not (data["scale"] > 0).all():
            raise RuntimeError("Official NPZ has invalid pixel scale metadata")
        if not np.isfinite(data["center"]).all():
            raise RuntimeError("Official NPZ has non-finite centers")
        gt_counts = (targets[..., 2] > 0.5).sum(0)
        missing = [row["image_name"] for row in expected_rows
                   if not (args.image_root / row["image_name"]).is_file()]
        report["coverage"].update({"npz_person_entries": len(data["imgname"]),
                                   "unique_image_names": len(set(row["image_name"] for row in expected_rows)),
                                   "gt_counts_44": gt_counts.tolist(),
                                   "gt_keypoints": int(gt_counts.sum()),
                                   "active_gt_joints": int((gt_counts > 0).sum()),
                                   "missing_selected_image_entries": len(missing),
                                   "missing_image_examples": missing[:20]})
        report["array_audit"] = {key: {"shape": list(data[key].shape), "dtype": str(data[key].dtype)}
                                 for key in sorted(required)}
        report["checks"] = {"person_order_matches_source_jsons": True,
                            "center_matches_original_gt_boxes": True,
                            "pixel_scale_matches_1_2_times_gt_width_height": True,
                            "44_point_mapping_matches_official": True,
                            "no_extra_filters_or_missing_image_drops": True,
                            "gt_confidence_rule": "source v>0 becomes 1; official PCK uses >0.5"}
        if missing:
            raise FileNotFoundError(f"{len(missing)} selected NPZ image entries are missing; none were dropped")
    with (args.output_dir / "person_manifest.jsonl").open("x", encoding="utf-8") as stream:
        for idx, row in enumerate(expected_rows):
            identity = {key: row[key] for key in ("annotation_file", "annotation_id", "image_id", "track_id", "image_name")}
            stream.write(json.dumps({"npz_idx": idx, **identity}, ensure_ascii=False) + "\n")
    report["status"] = "HMR2_OFFICIAL_POSETRACK18_METADATA_AUDIT_OK"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # A dedicated directory is never reused, even following a crash.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report: dict[str, Any] = {
        "status": "HMR2_OFFICIAL_POSETRACK18_METADATA_AUDIT_FAILED",
        "split": "PoseTrack18 val development", "device": "CPU only",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "wrapper_sha256": sha256_file(Path(__file__)), "original_files_modified": False,
        "test_used_for_model_selection": False, "ram_protocol_equality_claimed": False,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    exit_code = 0
    try:
        prepare_and_audit(args, report)
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "HMR2_OFFICIAL_POSETRACK18_METADATA_AUDIT_FAILED"
        report["exception"] = {"type": type(exc).__name__, "message": str(exc),
                               "traceback": traceback.format_exc()}
        exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        print(report["exception"]["traceback"], file=sys.stderr, flush=True)
    report["elapsed_seconds"] = time.monotonic() - started
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json_new(args.output_dir / "metadata_audit.json", report)
    print(json.dumps({"status": report["status"], "output_dir": str(args.output_dir),
                      "generated_npz": report.get("generated_npz"),
                      "coverage": report.get("coverage")}, allow_nan=False), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
