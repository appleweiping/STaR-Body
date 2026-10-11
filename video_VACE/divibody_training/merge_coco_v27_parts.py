"""Merge disjoint COCO v27 evaluator outputs using their exact sufficient statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def recover_totals(metric: dict) -> dict[str, list]:
    totals = {key: [] for key in ("correct05", "correct10", "error_sum", "count")}
    for item in metric["per_temporal_position"]:
        count = int(item["visible_keypoints"])
        totals["count"].append(count)
        totals["correct05"].append(round(float(item["PCKn@0.05"]) * count))
        totals["correct10"].append(round(float(item["PCKn@0.10"]) * count))
        totals["error_sum"].append(float(item["mean_normalized_joint_error"]) * count)
    return totals


def summarize(totals: dict[str, list]) -> dict:
    positions = []
    for index in range(8):
        count = totals["count"][index]
        positions.append(
            {
                "position": index,
                "PCKn@0.05": totals["correct05"][index] / count,
                "PCKn@0.10": totals["correct10"][index] / count,
                "mean_normalized_joint_error": totals["error_sum"][index] / count,
                "visible_keypoints": count,
            }
        )
    count = sum(totals["count"])
    return {
        "PCKn@0.05": sum(totals["correct05"]) / count,
        "PCKn@0.10": sum(totals["correct10"]) / count,
        "mean_normalized_joint_error": sum(totals["error_sum"]) / count,
        "visible_keypoint_predictions": count,
        "per_temporal_position": positions,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    parts = [json.loads(path.read_text(encoding="utf-8")) for path in args.parts]
    reference = parts[0]
    for part in parts[1:]:
        if part["checkpoint"] != reference["checkpoint"]:
            raise ValueError("Parts use different checkpoints")
        if part["protocol"] != reference["protocol"]:
            raise ValueError("Parts use different protocols")
    metrics = {}
    for name in ("anchor", "direct", "assisted"):
        recovered = [recover_totals(part["metrics"][name]) for part in parts]
        totals = {
            key: [sum(values[key][i] for values in recovered) for i in range(8)]
            for key in ("correct05", "correct10", "error_sum", "count")
        }
        metrics[name] = summarize(totals)
    result = {
        "status": "RAM_V27_COCO2017_VAL_REPEATFRAME_OK",
        "checkpoint": reference["checkpoint"],
        "checkpoint_epoch": reference["checkpoint_epoch"],
        "checkpoint_variant": reference.get("checkpoint_variant"),
        "metrics": metrics,
        "coverage": {
            "people": sum(part["coverage"]["people"] for part in parts),
            "dataset_people": reference["coverage"]["dataset_people"],
            "development_only": False,
        },
        "protocol": reference["protocol"],
        "test_used_for_model_selection": False,
        "merged_parts": [str(path.resolve()) for path in args.parts],
    }
    if result["coverage"]["people"] != result["coverage"]["dataset_people"]:
        raise ValueError("Merged parts do not cover the complete dataset")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
