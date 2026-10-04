"""Feature-specific masks and spherical non-maximum suppression."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from ._models import DeduplicationResult


def _is_torch(value: Any) -> bool:
    module = type(value).__module__
    return module == "torch" or module.startswith("torch.")


def _boolean_mask(value: Any, name: str, *, like: Any | None = None):
    if _is_torch(value):
        import torch

        if value.dtype != torch.bool:
            raise TypeError(f"{name} must have boolean dtype")
        if like is not None:
            if not _is_torch(like):
                raise TypeError(f"{name} must use the same backend")
            if value.device != like.device:
                raise ValueError(f"{name} must use the same Torch device")
        return value.clone()
    if not isinstance(value, np.ndarray) or value.dtype != np.bool_:
        raise TypeError(f"{name} must be a boolean numpy.ndarray or torch.Tensor")
    if like is not None and _is_torch(like):
        raise TypeError(f"{name} must use the same backend")
    return value.copy()


def gnomonic_feature_mask(
    projection_result: Any,
    *,
    edge_margin_px: int,
    validity_mask: Any | None = None,
    user_mask: Any | None = None,
):
    """Combine geometric support, data validity, edge margin and user mask.

    The input may be a canonical ``ProjectionResult`` or a workflow-created
    ``GnomonicFace``. The result preserves the mask backend and device.
    """

    if isinstance(edge_margin_px, bool) or not isinstance(edge_margin_px, Integral):
        raise TypeError("edge_margin_px must be an integer")
    margin = int(edge_margin_px)
    if margin < 0:
        raise ValueError("edge_margin_px must be non-negative")

    support = getattr(projection_result, "support_mask", None)
    if support is None:
        raise TypeError("projection_result must expose an explicit support_mask")
    combined = _boolean_mask(support, "support_mask")
    inferred_validity = getattr(projection_result, "validity_mask", None)
    if inferred_validity is None and hasattr(projection_result, "validity"):
        try:
            inferred_validity = projection_result.validity("image")
        except (KeyError, TypeError):
            inferred_validity = None
    for name, candidate in (
        (
            "validity_mask",
            validity_mask if validity_mask is not None else inferred_validity,
        ),
        ("user_mask", user_mask),
    ):
        if candidate is None:
            continue
        mask = _boolean_mask(candidate, name, like=combined)
        if tuple(mask.shape) != tuple(combined.shape):
            raise ValueError(
                f"{name} must have shape {tuple(combined.shape)}; got {tuple(mask.shape)}"
            )
        combined &= mask

    height, width = combined.shape[-2:]
    if margin * 2 >= height or margin * 2 >= width:
        if _is_torch(combined):
            combined.zero_()
        else:
            combined[...] = False
        return combined
    if margin:
        combined[..., :margin, :] = False
        combined[..., height - margin :, :] = False
        combined[..., :, :margin] = False
        combined[..., :, width - margin :] = False
    return combined


def deduplicate_spherical_keypoints(
    bearings_xyz: Any,
    responses: Any,
    *,
    angular_threshold_rad: float,
) -> DeduplicationResult:
    """Apply deterministic angular NMS to already-detected keypoints.

    Higher response wins. Exact ties are resolved by the original source
    order. Detection, descriptor generation, and nearest-neighbour matching
    remain wholly outside this function.
    """

    bearings = _to_numpy(bearings_xyz)
    scores = _to_numpy(responses)
    if bearings.ndim != 2 or bearings.shape[1] != 3:
        raise ValueError("bearings_xyz must have shape (N, 3)")
    if scores.ndim != 1 or scores.shape[0] != bearings.shape[0]:
        raise ValueError("responses must have shape (N,)")
    if isinstance(angular_threshold_rad, bool) or not isinstance(
        angular_threshold_rad, Real
    ):
        raise TypeError("angular_threshold_rad must be a real number")
    threshold = float(angular_threshold_rad)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= math.pi:
        raise ValueError("angular_threshold_rad must be finite and in [0, pi]")
    if not np.isfinite(scores).all():
        raise ValueError("responses must be finite")
    norms = np.linalg.norm(bearings, axis=1)
    if not np.isfinite(bearings).all() or np.any(norms <= 0):
        raise ValueError("bearings_xyz must contain finite non-zero vectors")
    unit = bearings.astype(np.float64, copy=False) / norms[:, None]
    count = len(unit)
    if count == 0:
        return DeduplicationResult(
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            (),
            (),
        )

    rank = np.lexsort((np.arange(count), -scores.astype(np.float64)))
    tree = cKDTree(unit)
    chord = 2.0 * math.sin(threshold / 2.0)
    assigned = np.full(count, -1, dtype=np.int64)
    distance = np.full(count, np.nan, dtype=np.float64)
    ranked_groups: list[tuple[int, tuple[int, ...], tuple[float, ...]]] = []
    for selected in rank:
        if assigned[selected] >= 0:
            continue
        candidates = tree.query_ball_point(unit[selected], chord + 1e-15)
        direct: list[tuple[int, float]] = []
        for candidate in candidates:
            if assigned[candidate] >= 0:
                continue
            angle = math.acos(
                float(np.clip(unit[selected] @ unit[candidate], -1.0, 1.0))
            )
            if angle <= threshold + 1e-15:
                direct.append((int(candidate), angle))
        direct.sort(key=lambda item: (item[1], item[0]))
        members = tuple(item[0] for item in direct)
        angles = tuple(item[1] for item in direct)
        ranked_groups.append((int(selected), members, angles))
        provisional = len(ranked_groups) - 1
        for member, angle in direct:
            assigned[member] = provisional
            distance[member] = angle

    ordered_groups = sorted(ranked_groups, key=lambda item: item[0])
    group_index = np.empty(count, dtype=np.int64)
    duplicate_groups = []
    selected_indices = []
    reasons = []
    for final_index, (selected, members, angles) in enumerate(ordered_groups):
        selected_indices.append(selected)
        ordered_members = (selected,) + tuple(
            item for item in members if item != selected
        )
        duplicate_groups.append(ordered_members)
        reasons.append("highest-response-then-source-order")
        angle_by_member = dict(zip(members, angles))
        for member in ordered_members:
            group_index[member] = final_index
            distance[member] = angle_by_member.get(member, 0.0)
    return DeduplicationResult(
        selected_indices=np.asarray(selected_indices, dtype=np.int64),
        group_index_for_input=group_index,
        angular_distance_to_selected_rad=distance,
        duplicate_groups=tuple(duplicate_groups),
        selection_reasons=tuple(reasons),
    )


def _to_numpy(value: Any) -> np.ndarray:
    if _is_torch(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)
