"""Local edge-aware densification of confident two-view stereo samples."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class SurfaceDensificationOptions:
    """Conservative local propagation settings for observed surfaces."""

    minimum_seed_confidence: float = 0.12
    maximum_distance_px: float = 8.0
    iterations: int = 8
    color_sigma: float = 0.10
    confidence_decay: float = 0.92
    candidate_agreement_log: float = 0.18
    candidate_blend: float = 0.65
    min_range_m: float = 0.3
    max_range_m: float = 15.0

    def __post_init__(self) -> None:
        for name in (
            "minimum_seed_confidence",
            "maximum_distance_px",
            "color_sigma",
            "confidence_decay",
            "candidate_agreement_log",
            "min_range_m",
            "max_range_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.minimum_seed_confidence > 1.0:
            raise ValueError("minimum_seed_confidence must not exceed one")
        if self.confidence_decay > 1.0:
            raise ValueError("confidence_decay must not exceed one")
        if not 0.0 <= self.candidate_blend <= 1.0:
            raise ValueError("candidate_blend must lie in [0, 1]")
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if self.max_range_m <= self.min_range_m:
            raise ValueError("max_range_m must exceed min_range_m")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def densify_stereo_surface(
    radial_range_m: Any,
    confidence: Any,
    rgb_hwc: Any,
    support_hw: Any,
    *,
    hard_seed_hw: Any | None = None,
    options: SurfaceDensificationOptions | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Propagate confident log-range seeds over a bounded color-aware region.

    Propagation is deliberately local.  A pixel can become valid only when it
    lies within ``maximum_distance_px`` of a stereo/BA seed and is reached by
    the four-neighbour diffusion.  Longitude wraps at the ERP seam; latitude
    never wraps.  Original seeds remain bit-exact throughout.
    """

    settings = options or SurfaceDensificationOptions()
    radial = np.asarray(radial_range_m, dtype=np.float32)
    score = np.asarray(confidence, dtype=np.float32)
    rgb = np.asarray(rgb_hwc)
    support = np.asarray(support_hw, dtype=bool)
    if radial.ndim != 2 or score.shape != radial.shape or support.shape != radial.shape:
        raise ValueError("range, confidence and support must share shape HW")
    if rgb.shape != (*radial.shape, 3):
        raise ValueError("rgb_hwc must match the range shape")
    hard = (
        np.zeros(radial.shape, dtype=bool)
        if hard_seed_hw is None
        else np.asarray(hard_seed_hw, dtype=bool)
    )
    if hard.shape != radial.shape:
        raise ValueError("hard_seed_hw must match the range shape")
    finite = (
        np.isfinite(radial)
        & (radial >= settings.min_range_m)
        & (radial <= settings.max_range_m)
    )
    seeds = support & finite & ((score >= settings.minimum_seed_confidence) | hard)
    if not np.any(seeds):
        empty_range = np.full(radial.shape, np.nan, dtype=np.float32)
        empty_confidence = np.zeros(radial.shape, dtype=np.float32)
        return (
            empty_range,
            seeds,
            empty_confidence,
            {
                "options": settings.to_dict(),
                "seed_pixels": 0,
                "dense_pixels": 0,
                "dense_fraction": 0.0,
                "iterations_completed": 0,
            },
        )

    # Tile longitude so the Euclidean distance gate respects the ERP seam.
    width = radial.shape[1]
    tiled_nonseed = np.tile((~seeds).astype(np.uint8), (1, 3))
    tiled_distance = cv2.distanceTransform(tiled_nonseed, cv2.DIST_L2, 5)
    distance = tiled_distance[:, width : 2 * width]
    allowed = support & (distance <= settings.maximum_distance_px)
    colors = rgb.astype(np.float32)
    if np.issubdtype(rgb.dtype, np.integer):
        colors /= float(np.iinfo(rgb.dtype).max)
    colors = np.clip(colors, 0.0, 1.0)
    log_range = np.zeros(radial.shape, dtype=np.float32)
    log_range[seeds] = np.log(radial[seeds])
    propagated_confidence = np.zeros(radial.shape, dtype=np.float32)
    propagated_confidence[seeds] = np.maximum(score[seeds], 1e-3)
    propagated_confidence[hard & seeds] = 1.0
    known = seeds.copy()
    iterations_completed = 0
    accepted_candidate_pixels = 0

    for iteration in range(settings.iterations):
        numerator = np.zeros(radial.shape, dtype=np.float32)
        denominator = np.zeros(radial.shape, dtype=np.float32)
        for direction in ("left", "right", "up", "down"):
            neighbor_known = _shift(known, direction, fill=False)
            neighbor_value = _shift(log_range, direction, fill=0.0)
            neighbor_confidence = _shift(propagated_confidence, direction, fill=0.0)
            neighbor_color = _shift(colors, direction, fill=0.0)
            color_distance = np.mean(np.abs(colors - neighbor_color), axis=2)
            edge_weight = np.exp(-color_distance / settings.color_sigma).astype(
                np.float32
            )
            weight = neighbor_confidence * edge_weight * neighbor_known
            numerator += weight * neighbor_value
            denominator += weight
        frontier = allowed & ~known & (denominator > 1e-5)
        if not np.any(frontier):
            break
        propagated = numerator / np.maximum(denominator, 1e-8)
        candidate_log = np.zeros(radial.shape, dtype=np.float32)
        candidate_log[finite] = np.log(radial[finite])
        candidate_agrees = (
            frontier
            & finite
            & (score > 0.0)
            & (np.abs(candidate_log - propagated) <= settings.candidate_agreement_log)
        )
        log_range[frontier] = propagated[frontier]
        log_range[candidate_agrees] = (
            settings.candidate_blend * candidate_log[candidate_agrees]
            + (1.0 - settings.candidate_blend) * propagated[candidate_agrees]
        )
        propagated_confidence[frontier] = np.minimum(
            1.0,
            denominator[frontier] * (settings.confidence_decay / 4.0),
        )
        propagated_confidence[candidate_agrees] = np.maximum(
            propagated_confidence[candidate_agrees], score[candidate_agrees]
        )
        accepted_candidate_pixels += int(candidate_agrees.sum())
        known[frontier] = True
        iterations_completed = iteration + 1

    dense = np.full(radial.shape, np.nan, dtype=np.float32)
    dense[known] = np.exp(log_range[known]).astype(np.float32)
    dense[seeds] = radial[seeds]
    propagated_confidence[seeds] = np.maximum(
        propagated_confidence[seeds], score[seeds]
    )
    report = {
        "options": settings.to_dict(),
        "seed_pixels": int(seeds.sum()),
        "seed_fraction": float(seeds.mean()),
        "dense_pixels": int(known.sum()),
        "dense_fraction": float(known.mean()),
        "iterations_completed": iterations_completed,
        "accepted_low_confidence_candidates": accepted_candidate_pixels,
    }
    return dense, known, propagated_confidence, report


def _shift(array: np.ndarray, direction: str, *, fill: Any) -> np.ndarray:
    if direction == "left":
        return np.roll(array, 1, axis=1)
    if direction == "right":
        return np.roll(array, -1, axis=1)
    shifted = np.empty_like(array)
    if direction == "up":
        shifted[0] = fill
        shifted[1:] = array[:-1]
        return shifted
    if direction == "down":
        shifted[-1] = fill
        shifted[:-1] = array[1:]
        return shifted
    raise ValueError(f"unsupported direction: {direction}")


__all__ = ["SurfaceDensificationOptions", "densify_stereo_surface"]
