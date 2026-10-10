"""Prompt and multimask utilities independent of a SAM implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True, slots=True)
class SegmentPrompt:
    points_xy: tuple[tuple[float, float], ...]
    labels: tuple[int, ...]
    box_xyxy: tuple[float, float, float, float] | None = None
    prior_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self.points_xy or len(self.points_xy) != len(self.labels):
            raise ValueError("points and labels must be non-empty and aligned")
        if any(label not in {0, 1} for label in self.labels):
            raise ValueError("point labels must be zero or one")
        if self.prior_mask is not None:
            prior = np.array(self.prior_mask, dtype=bool, copy=True)
            if prior.ndim != 2:
                raise ValueError("prior_mask must be HW")
            prior.setflags(write=False)
            object.__setattr__(self, "prior_mask", prior)


def pairwise_mask_iou(masks: np.ndarray) -> np.ndarray:
    values = np.asarray(masks, dtype=bool)
    if values.shape[0] != 3 or values.ndim != 3:
        raise ValueError("masks must have shape (3,H,W)")
    result = np.eye(3, dtype=np.float64)
    for first in range(3):
        for second in range(first + 1, 3):
            union = np.count_nonzero(values[first] | values[second])
            intersection = np.count_nonzero(values[first] & values[second])
            score = intersection / union if union else 1.0
            result[first, second] = result[second, first] = score
    return result


def maximum_pairwise_iou(masks: np.ndarray) -> float:
    matrix = pairwise_mask_iou(masks)
    return float(np.max(matrix[np.triu_indices(3, 1)]))


def majority_and_envelope(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(masks, dtype=bool)
    if values.shape[0] != 3 or values.ndim != 3:
        raise ValueError("masks must have shape (3,H,W)")
    return values.sum(axis=0) >= 2, values.any(axis=0)


def mask_stability_score(
    logits: np.ndarray,
    *,
    threshold: float = 0.0,
    offset: float = 1.0,
) -> float:
    """Measure threshold stability using the standard intersection/union rule."""

    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("logits must be a finite HW array")
    strict = values > threshold + offset
    relaxed = values > threshold - offset
    union = np.count_nonzero(relaxed)
    return float(np.count_nonzero(strict) / union) if union else 1.0


def connected_component_at(
    mask: np.ndarray, point_xy: tuple[float, float]
) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2 or not values.any():
        raise ValueError("mask must be a non-empty HW array")
    labels, _ = ndimage.label(values, structure=np.ones((3, 3), dtype=np.uint8))
    height, width = values.shape
    x = int(np.clip(round(point_xy[0]), 0, width - 1))
    y = int(np.clip(round(point_xy[1]), 0, height - 1))
    label = int(labels[y, x])
    if label == 0:
        ys, xs = np.where(values)
        nearest = int(np.argmin((xs - point_xy[0]) ** 2 + (ys - point_xy[1]) ** 2))
        label = int(labels[ys[nearest], xs[nearest]])
    return labels == label


def prompt_from_mask(
    mask: np.ndarray,
    anchor_xy: tuple[float, float],
    *,
    positive_count: int = 3,
    padding: int = 12,
    negative_offset: int = 10,
) -> SegmentPrompt:
    """Build box, foreground/background points, and prior for continuation."""

    component = connected_component_at(mask, anchor_xy)
    height, width = component.shape
    ys, xs = np.where(component)
    x1 = max(0, int(xs.min()) - padding)
    y1 = max(0, int(ys.min()) - padding)
    x2 = min(width - 1, int(xs.max()) + padding)
    y2 = min(height - 1, int(ys.max()) + padding)
    distance = ndimage.distance_transform_edt(component)
    working = distance.copy()
    positives: list[tuple[float, float]] = []
    ax = int(np.clip(round(anchor_xy[0]), 0, width - 1))
    ay = int(np.clip(round(anchor_xy[1]), 0, height - 1))
    if component[ay, ax]:
        positives.append((float(anchor_xy[0]), float(anchor_xy[1])))
    yy, xx = np.ogrid[:height, :width]
    radius = max(8, int(round(min(height, width) * 0.04)))
    while len(positives) < positive_count and float(working.max()) > 0:
        y, x = np.unravel_index(int(np.argmax(working)), working.shape)
        positives.append((float(x), float(y)))
        working[(xx - x) ** 2 + (yy - y) ** 2 <= radius**2] = 0
    middle_x = 0.5 * (x1 + x2)
    middle_y = 0.5 * (y1 + y2)
    candidates = (
        (max(0.0, x1 - negative_offset), middle_y),
        (min(width - 1.0, x2 + negative_offset), middle_y),
        (middle_x, max(0.0, y1 - negative_offset)),
        (middle_x, min(height - 1.0, y2 + negative_offset)),
    )
    negatives = tuple(
        (float(x), float(y))
        for x, y in candidates
        if not component[
            int(np.clip(round(y), 0, height - 1)),
            int(np.clip(round(x), 0, width - 1)),
        ]
    )
    return SegmentPrompt(
        tuple(positives) + negatives,
        (1,) * len(positives) + (0,) * len(negatives),
        (float(x1), float(y1), float(x2), float(y2)),
        component,
    )


def prompt_grid(
    size_hw: tuple[int, int], grid_size: int = 8
) -> tuple[SegmentPrompt, ...]:
    """Create deterministic point-only prompts with a five-percent margin."""

    height, width = size_hw
    if height < 1 or width < 1 or grid_size < 1:
        raise ValueError("shape and grid_size must be positive")
    x_values = np.linspace(0.05 * (width - 1), 0.95 * (width - 1), grid_size)
    y_values = np.linspace(0.05 * (height - 1), 0.95 * (height - 1), grid_size)
    return tuple(
        SegmentPrompt(((float(x), float(y)),), (1,)) for y in y_values for x in x_values
    )


def consensus_frontiers(
    consensus: np.ndarray,
    *,
    band_pixels: int,
    minimum_pixels: int,
) -> tuple[tuple[str, tuple[float, float]], ...]:
    """Return chart borders reached by a consensus mask."""

    mask = np.asarray(consensus, dtype=bool)
    if mask.ndim != 2 or not 0 < band_pixels < min(mask.shape) // 2:
        raise ValueError("invalid mask or frontier band")
    height, width = mask.shape
    regions = {
        "top": (slice(0, band_pixels), slice(None)),
        "bottom": (slice(height - band_pixels, height), slice(None)),
        "left": (slice(None), slice(0, band_pixels)),
        "right": (slice(None), slice(width - band_pixels, width)),
    }
    result: list[tuple[str, tuple[float, float]]] = []
    for side, region in regions.items():
        ys, xs = np.where(mask[region])
        if len(ys) < minimum_pixels:
            continue
        if side == "bottom":
            ys += height - band_pixels
        if side == "right":
            xs += width - band_pixels
        if side == "top":
            point = (float(np.median(xs)), 0.0)
        elif side == "bottom":
            point = (float(np.median(xs)), float(height - 1))
        elif side == "left":
            point = (0.0, float(np.median(ys)))
        else:
            point = (float(width - 1), float(np.median(ys)))
        result.append((side, point))
    return tuple(result)
