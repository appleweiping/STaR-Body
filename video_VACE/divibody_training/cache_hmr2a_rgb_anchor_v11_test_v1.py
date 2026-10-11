"""Test-only additive wrapper for the immutable V11 raw-bbox cache builder.

The original builder intentionally accepts only train/validation.  This wrapper
relaxes only that split guard so an already audited, feature-only 3DPW test VACE
cache can receive HMR2a RGB tokens and anchors.  All original alignment and
sidecar validation remains active; the original source is not modified.
"""

from __future__ import annotations

from divibody_training import cache_hmr2a_rgb_anchor_v11_raw as original
from divibody_training import cache_hmr2a_rgb_anchor_v8 as contract


def validate_test_vace_source(payload):
    signature = payload.get("signature", {})
    if signature.get("split") != "test":
        raise ValueError("This additive wrapper accepts only a 3DPW test cache")
    if not signature.get("feature_only") or signature.get("contains_targets"):
        raise ValueError("Expected a target-free VACE test feature cache")
    if signature.get("development_only"):
        raise ValueError("Development-only test cache is forbidden")
    required = {
        "features", "frame_owner_mask", "velocity_owner_mask",
        "acceleration_owner_mask", "metadata",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Incomplete VACE test cache: {sorted(missing)}")
    if signature.get("format") != "divibody-contiguous-vace-video-features-v6-stride4":
        raise ValueError(f"Unexpected test VACE format: {signature.get('format')}")
    preflight = signature.get("dataset_preflight", {})
    if int(preflight.get("available_valid_person_frames", -1)) != 35515:
        raise ValueError("Expected the complete 35,515-person-frame 3DPW test set")


def build_test_anchor_plan(metadata, split, max_windows=0):
    if split != "test":
        raise ValueError("This additive plan is test-only")
    plan = contract.build_anchor_plan(metadata, "validation", max_windows)
    return contract.AnchorPlan(
        split="test",
        frame_uids=plan.frame_uids,
        window_frame_indices=plan.window_frame_indices,
        window_uids=plan.window_uids,
        development_only=plan.development_only,
    )


def validate_test_sidecar(payload, vace_payload=None):
    """Run tensor checks with a masked split, then audit unmodified test provenance."""
    payload_signature = payload["signature"]
    payload_split = payload_signature["split"]
    try:
        payload_signature["split"] = "validation"
        contract.validate_sidecar(payload, None)
    finally:
        payload_signature["split"] = payload_split
    if vace_payload is None:
        return
    metadata = vace_payload["metadata"][: len(payload["window_uids"])]
    if payload_signature["split"] != vace_payload["signature"]["split"]:
        raise ValueError("HMR sidecar and VACE test splits differ")
    if payload["window_uids"] != [record["window_uid"] for record in metadata]:
        raise ValueError("HMR sidecar and VACE test window order differ")
    frame_uids = payload["frame_uids"]
    actual = [[frame_uids[index] for index in row]
              for row in payload["window_frame_indices"].tolist()]
    if actual != [record["target_frame_uids"] for record in metadata]:
        raise ValueError("HMR sidecar and VACE test frame identities differ")
    digest = contract.semantic_vace_digest(vace_payload, len(payload["window_uids"]))
    if payload_signature["vace_semantic_sha256"] != digest:
        raise ValueError("Unmodified test VACE semantic provenance mismatch")


def main() -> None:
    original.validate_vace_source = validate_test_vace_source
    original.build_anchor_plan = build_test_anchor_plan
    original.validate_sidecar = validate_test_sidecar
    original.main()


if __name__ == "__main__":
    main()
