"""Shared confidence composition for panoramic range fusion."""

from __future__ import annotations

from typing import Any

import numpy as np


def compose_fusion_confidence(
    sum_confidence_taper: Any,
    sum_taper: Any,
    radial_range: Any,
    disagreement: Any,
    validity: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return matching, consistency, and combined confidence components.

    The edge taper controls how a projected face contributes to range fusion.
    Dividing by the accumulated taper removes that geometric blending weight
    from the reported matching confidence.  Range disagreement remains an
    independent multiplicative consistency term.
    """

    weighted = np.asarray(sum_confidence_taper, dtype=np.float64)
    taper = np.asarray(sum_taper, dtype=np.float64)
    radial = np.asarray(radial_range, dtype=np.float64)
    spread = np.asarray(disagreement, dtype=np.float64)
    valid = np.asarray(validity)
    if weighted.ndim != 2:
        raise ValueError("confidence accumulators must have shape HW")
    shape = weighted.shape
    if taper.shape != shape or radial.shape != shape or spread.shape != shape:
        raise ValueError("confidence inputs must share shape HW")
    if valid.dtype != np.bool_ or valid.shape != shape:
        raise ValueError("validity must be a boolean mask with shape HW")
    if np.any(~np.isfinite(weighted)) or np.any(weighted < 0.0):
        raise ValueError("sum_confidence_taper must be finite and non-negative")
    if np.any(~np.isfinite(taper)) or np.any(taper < 0.0):
        raise ValueError("sum_taper must be finite and non-negative")
    if np.any(valid & (~np.isfinite(radial) | (radial <= 0.0))):
        raise ValueError("valid radial range must be finite and positive")
    if np.any(valid & (~np.isfinite(spread) | (spread < 0.0))):
        raise ValueError("valid disagreement must be finite and non-negative")

    matching = np.zeros(shape, dtype=np.float32)
    has_taper = taper > np.finfo(np.float64).eps
    matching[has_taper] = np.clip(
        weighted[has_taper] / taper[has_taper], 0.0, 1.0
    ).astype(np.float32)
    matching[~valid] = 0.0

    consistency = np.zeros(shape, dtype=np.float32)
    relative_disagreement = np.zeros(shape, dtype=np.float64)
    relative_disagreement[valid] = spread[valid] / radial[valid]
    consistency[valid] = np.exp(-relative_disagreement[valid]).astype(np.float32)

    confidence = (matching * consistency).astype(np.float32)
    confidence[~valid] = 0.0
    return matching, consistency, confidence
