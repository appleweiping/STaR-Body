"""Geometry metrics for the common SMPL-based 3DPW evaluation protocol.

The functions in this module are deliberately independent of the DiviBody and
VACE model code.  Distances are computed in metres internally.  Callers are
responsible for converting the returned values to millimetres when reporting
paper metrics.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch


# The 14 LSP joints selected from the 17 joints produced by the common H36M
# vertex regressor.  This is the mapping used by SPIN for 3DPW evaluation.
H36M_TO_J14 = (6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10)


def regress_h36m_joints(
    vertices: torch.Tensor, regressor: torch.Tensor
) -> torch.Tensor:
    """Regress 17 H36M joints from corresponding 6,890-vertex SMPL meshes."""

    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError(f"Expected vertices [B,V,3], got {tuple(vertices.shape)}")
    if regressor.ndim != 2 or regressor.shape[0] != 17:
        raise ValueError(f"Expected a [17,V] regressor, got {tuple(regressor.shape)}")
    if vertices.shape[1] != regressor.shape[1]:
        raise ValueError(
            f"Vertex count {vertices.shape[1]} does not match regressor "
            f"width {regressor.shape[1]}"
        )
    return torch.einsum("jv,bvc->bjc", regressor, vertices)


def pelvis_aligned_j14(
    vertices: torch.Tensor, regressor: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return SPIN-compatible J14 and the H36M-regressor pelvis joint.

    SPIN subtracts H36M joint zero from all 17 regressed joints and then takes
    ``H36M_TO_J14``.  The pelvis is returned separately so the same root can be
    used to align vertices for an explicitly named auxiliary PVE.
    """

    joints17 = regress_h36m_joints(vertices, regressor)
    pelvis = joints17[:, [0], :]
    indices = torch.as_tensor(H36M_TO_J14, device=vertices.device)
    return joints17.index_select(1, indices) - pelvis, pelvis


def rotate_points(points: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    """Apply a batch of 3x3 rotations to row-vector point sets."""

    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"Expected points [B,N,3], got {tuple(points.shape)}")
    if rotations.shape != (points.shape[0], 3, 3):
        raise ValueError(
            f"Expected rotations {(points.shape[0], 3, 3)}, "
            f"got {tuple(rotations.shape)}"
        )
    return torch.einsum("bij,bnj->bni", rotations, points)


def hip_centered_j14(joints: torch.Tensor) -> torch.Tensor:
    """Root-align LSP-ordered J14 with the left/right hip midpoint.

    VIBE's published evaluator applies this root definition before its pose and
    acceleration metrics.  In the J14 mapping used here the two hip indices are
    2 and 3.
    """

    if joints.ndim != 3 or joints.shape[1:] != (14, 3):
        raise ValueError(f"Expected J14 [B,14,3], got {tuple(joints.shape)}")
    hip_midpoint = joints[:, [2, 3], :].mean(dim=1, keepdim=True)
    return joints - hip_midpoint


def batch_similarity_transform(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Align ``prediction`` to ``target`` with a proper similarity transform.

    A separate scale, rotation, and translation is estimated for every sample.
    The implementation follows the orthogonal Procrustes procedure used by
    SPIN/VIBE while using ``torch.linalg.svd`` for current PyTorch versions.
    """

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError(
            "prediction and target must have the same [B,N,D] shape, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if prediction.shape[-1] not in (2, 3):
        raise ValueError("Only 2D or 3D point sets are supported")

    source = prediction.transpose(1, 2)
    destination = target.transpose(1, 2)
    source_mean = source.mean(dim=2, keepdim=True)
    destination_mean = destination.mean(dim=2, keepdim=True)
    source_centered = source - source_mean
    destination_centered = destination - destination_mean
    source_variance = source_centered.square().sum(dim=(1, 2))
    if torch.any(source_variance <= eps):
        raise ValueError("Cannot Procrustes-align a degenerate point set")

    covariance = source_centered @ destination_centered.transpose(1, 2)
    u_matrix, _, vh_matrix = torch.linalg.svd(covariance)
    dimension = prediction.shape[-1]
    correction = torch.eye(
        dimension, device=prediction.device, dtype=prediction.dtype
    ).expand(prediction.shape[0], -1, -1).clone()
    determinant = torch.linalg.det(u_matrix @ vh_matrix)
    determinant = torch.where(
        determinant < 0,
        -torch.ones_like(determinant),
        torch.ones_like(determinant),
    )
    correction[:, -1, -1] = determinant
    rotation = vh_matrix.transpose(1, 2) @ correction @ u_matrix.transpose(1, 2)
    scale = torch.diagonal(rotation @ covariance, dim1=1, dim2=2).sum(1)
    scale = scale / source_variance
    translation = destination_mean - scale[:, None, None] * (
        rotation @ source_mean
    )
    aligned = scale[:, None, None] * (rotation @ source) + translation
    return aligned.transpose(1, 2)


def mean_euclidean_error(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Return one mean point/vertex Euclidean error per sample."""

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must have the same [B,N,3] shape")
    return torch.linalg.vector_norm(prediction - target, dim=-1).mean(dim=-1)


def pa_mpjpe_per_sample(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Return per-sample MPJPE after full similarity Procrustes alignment."""

    aligned = batch_similarity_transform(prediction, target)
    return mean_euclidean_error(aligned, target)


@dataclass(frozen=True)
class AccelerationResult:
    """Sequence-safe finite-difference acceleration errors in metres/frame^2."""

    errors: np.ndarray
    center_sample_indices: np.ndarray
    per_track: tuple[dict[str, object], ...]


def sequence_acceleration_error(
    prediction: np.ndarray,
    target: np.ndarray,
    sequences: Sequence[str],
    person_ids: Iterable[int],
    frame_ids: Iterable[int],
) -> AccelerationResult:
    """Compute acceleration error without crossing tracks or missing frames.

    For a valid consecutive triplet the second difference is
    ``X[t-1] - 2*X[t] + X[t+1]``.  Each returned scalar is averaged over the
    joints of one triplet.  Inputs are expected to be pelvis-centred already.
    """

    predicted = np.asarray(prediction)
    ground_truth = np.asarray(target)
    if predicted.shape != ground_truth.shape or predicted.ndim != 3:
        raise ValueError("prediction and target must share an [N,J,3] shape")
    sample_count = predicted.shape[0]
    sequence_array = np.asarray(sequences)
    person_array = np.asarray(tuple(person_ids), dtype=np.int64)
    frame_array = np.asarray(tuple(frame_ids), dtype=np.int64)
    if not (
        len(sequence_array)
        == len(person_array)
        == len(frame_array)
        == sample_count
    ):
        raise ValueError("Track metadata length does not match the joint arrays")

    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for sample_index, (sequence, person_id) in enumerate(
        zip(sequence_array.tolist(), person_array.tolist())
    ):
        groups[(str(sequence), int(person_id))].append(sample_index)

    all_errors: list[np.ndarray] = []
    all_centers: list[np.ndarray] = []
    track_records: list[dict[str, object]] = []
    for (sequence, person_id), indices_list in sorted(groups.items()):
        indices = np.asarray(indices_list, dtype=np.int64)
        order = np.argsort(frame_array[indices], kind="stable")
        indices = indices[order]
        frames = frame_array[indices]
        if len(np.unique(frames)) != len(frames):
            raise ValueError(f"Duplicate frame in track {sequence}/{person_id}")

        if len(indices) < 3:
            valid = np.zeros(0, dtype=bool)
        else:
            valid = (frames[1:-1] - frames[:-2] == 1) & (
                frames[2:] - frames[1:-1] == 1
            )
        centers = indices[1:-1][valid]
        if len(centers):
            pred_track = predicted[indices]
            gt_track = ground_truth[indices]
            pred_acceleration = (
                pred_track[:-2] - 2.0 * pred_track[1:-1] + pred_track[2:]
            )[valid]
            gt_acceleration = (
                gt_track[:-2] - 2.0 * gt_track[1:-1] + gt_track[2:]
            )[valid]
            errors = np.linalg.norm(
                pred_acceleration - gt_acceleration, axis=-1
            ).mean(axis=-1)
            all_errors.append(errors)
            all_centers.append(centers)
            mean_error = float(errors.mean())
        else:
            mean_error = None
        track_records.append(
            {
                "sequence": sequence,
                "person_id": person_id,
                "samples": int(len(indices)),
                "first_frame": int(frames[0]) if len(frames) else None,
                "last_frame": int(frames[-1]) if len(frames) else None,
                "valid_triplets": int(len(centers)),
                "mean_error_m_per_frame2": mean_error,
            }
        )

    errors_array = (
        np.concatenate(all_errors).astype(np.float64, copy=False)
        if all_errors
        else np.empty(0, dtype=np.float64)
    )
    centers_array = (
        np.concatenate(all_centers).astype(np.int64, copy=False)
        if all_centers
        else np.empty(0, dtype=np.int64)
    )
    return AccelerationResult(
        errors=errors_array,
        center_sample_indices=centers_array,
        per_track=tuple(track_records),
    )
