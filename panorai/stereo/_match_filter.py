"""Dense radial-range evidence for filtering spherical feature matches."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from panorai.features import (
    SphericalBearingCorrespondences,
    SphericalFeatureMatches,
)
from panorai.geometry import rays_to_erp_pixels

from ._dense import SphericalStereoResult, _validate_pose

_INTERFACE = "panorai-dense-guided-match-filter/v1-experimental"


@dataclass(frozen=True, slots=True)
class DenseMatchFilterOptions:
    """Thresholds for one-way dense validation of existing matches.

    Continuous radial range is sampled bilinearly with horizontal ERP seam
    wrapping. Invalid neighbours are excluded and the remaining weights are
    renormalized only when their sum reaches ``min_valid_weight``.
    """

    max_angular_error_deg: float = 1.5
    min_dense_confidence: float = 0.0
    min_valid_weight: float = 0.5

    def __post_init__(self) -> None:
        _finite_interval(
            "max_angular_error_deg",
            self.max_angular_error_deg,
            lower=0.0,
            upper=180.0,
            lower_inclusive=False,
        )
        _finite_interval(
            "min_dense_confidence",
            self.min_dense_confidence,
            lower=0.0,
            upper=1.0,
        )
        _finite_interval(
            "min_valid_weight",
            self.min_valid_weight,
            lower=0.0,
            upper=1.0,
            lower_inclusive=False,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DenseMatchFilterResult:
    """Immutable match classification from dense geometric evidence.

    ``accepted_mask`` and ``rejected_mask`` partition matches for which valid
    dense evidence exists. ``unsupported_mask`` includes absent dense support
    and malformed or explicitly invalid input matches; ``input_valid_mask``
    distinguishes those two cases. No input match or pose value is modified.
    """

    input_valid_mask: np.ndarray
    dense_supported_mask: np.ndarray
    accepted_mask: np.ndarray
    rejected_mask: np.ndarray
    angular_residual_rad: np.ndarray
    sampled_range: np.ndarray
    sampled_confidence: np.ndarray
    sampled_valid_weight: np.ndarray
    bearings_a: np.ndarray
    bearings_b: np.ndarray
    predicted_bearings_b: np.ndarray
    options: DenseMatchFilterOptions
    interface: str = _INTERFACE
    stability: str = "experimental"

    def __post_init__(self) -> None:
        input_valid = _readonly_vector(
            self.input_valid_mask, np.bool_, "input_valid_mask"
        )
        supported = _readonly_vector(
            self.dense_supported_mask, np.bool_, "dense_supported_mask"
        )
        accepted = _readonly_vector(self.accepted_mask, np.bool_, "accepted_mask")
        rejected = _readonly_vector(self.rejected_mask, np.bool_, "rejected_mask")
        residual = _readonly_vector(
            self.angular_residual_rad, np.float64, "angular_residual_rad"
        )
        sampled_range = _readonly_vector(
            self.sampled_range, np.float64, "sampled_range"
        )
        sampled_confidence = _readonly_vector(
            self.sampled_confidence, np.float64, "sampled_confidence"
        )
        valid_weight = _readonly_vector(
            self.sampled_valid_weight, np.float64, "sampled_valid_weight"
        )
        bearings_a = np.array(self.bearings_a, dtype=np.float64, copy=True)
        bearings_b = np.array(self.bearings_b, dtype=np.float64, copy=True)
        predicted = np.array(self.predicted_bearings_b, dtype=np.float64, copy=True)
        count = input_valid.shape[0]
        arrays = (
            supported,
            accepted,
            rejected,
            residual,
            sampled_range,
            sampled_confidence,
            valid_weight,
        )
        if any(array.shape != (count,) for array in arrays):
            raise ValueError("all dense match-filter vectors must have shape (N,)")
        if not all(
            value.shape == (count, 3) for value in (bearings_a, bearings_b, predicted)
        ):
            raise ValueError("bearing arrays must have shape (N, 3)")
        if np.any(supported & ~input_valid):
            raise ValueError("dense support must be a subset of valid input matches")
        if np.any(accepted & rejected):
            raise ValueError("accepted and rejected masks must be disjoint")
        if not np.array_equal(accepted | rejected, supported):
            raise ValueError("accepted and rejected masks must partition dense support")
        if np.any((valid_weight < 0.0) | (valid_weight > 1.0)):
            raise ValueError("sampled_valid_weight must be in [0, 1]")
        bearings_a.setflags(write=False)
        bearings_b.setflags(write=False)
        predicted.setflags(write=False)
        object.__setattr__(self, "input_valid_mask", input_valid)
        object.__setattr__(self, "dense_supported_mask", supported)
        object.__setattr__(self, "accepted_mask", accepted)
        object.__setattr__(self, "rejected_mask", rejected)
        object.__setattr__(self, "angular_residual_rad", residual)
        object.__setattr__(self, "sampled_range", sampled_range)
        object.__setattr__(self, "sampled_confidence", sampled_confidence)
        object.__setattr__(self, "sampled_valid_weight", valid_weight)
        object.__setattr__(self, "bearings_a", bearings_a)
        object.__setattr__(self, "bearings_b", bearings_b)
        object.__setattr__(self, "predicted_bearings_b", predicted)

    @property
    def unsupported_mask(self) -> np.ndarray:
        result = np.logical_not(self.dense_supported_mask)
        result.setflags(write=False)
        return result

    @property
    def accepted_indices(self) -> np.ndarray:
        result = np.flatnonzero(self.accepted_mask)
        result.setflags(write=False)
        return result

    def to_bearing_correspondences(self) -> SphericalBearingCorrespondences:
        """Return unchanged bearings with acceptance as explicit validity."""

        valid = self.accepted_mask.copy()
        return SphericalBearingCorrespondences(
            bearings_a=np.array(self.bearings_a, copy=True),
            bearings_b=np.array(self.bearings_b, copy=True),
            weights=valid.astype(np.float32),
            valid=valid,
        )

    def describe(self) -> dict[str, Any]:
        count = self.input_valid_mask.shape[0]
        return {
            "interface": self.interface,
            "stability": self.stability,
            "match_count": int(count),
            "input_valid_count": int(self.input_valid_mask.sum()),
            "dense_supported_count": int(self.dense_supported_mask.sum()),
            "accepted_count": int(self.accepted_mask.sum()),
            "rejected_count": int(self.rejected_mask.sum()),
            "unsupported_count": int(self.unsupported_mask.sum()),
            "options": self.options.to_dict(),
        }


def filter_matches_by_dense_range(
    matches: SphericalFeatureMatches,
    dense_result: SphericalStereoResult,
    rotation_b_from_a: np.ndarray,
    translation_b_from_a: np.ndarray,
    *,
    options: DenseMatchFilterOptions | None = None,
) -> DenseMatchFilterResult:
    """Classify existing spherical matches using dense radial range.

    For each valid source bearing, the function samples radial range in camera
    A, forms ``X_a = range * bearing_a``, transforms it with
    ``X_b = R_b_from_a @ X_a + t_b_from_a``, and compares the predicted target
    bearing to the existing matched bearing. The operation is deliberately
    one-way: it neither changes the match coordinates nor optimizes the pose.
    """

    if not isinstance(matches, SphericalFeatureMatches):
        raise TypeError("matches must be a SphericalFeatureMatches object")
    if not isinstance(dense_result, SphericalStereoResult):
        raise TypeError("dense_result must be a SphericalStereoResult object")
    settings = options or DenseMatchFilterOptions()
    rotation, translation = _validate_pose(rotation_b_from_a, translation_b_from_a)
    if np.linalg.norm(translation) <= 1e-9:
        raise ValueError("translation_b_from_a must have non-zero scale")

    bearings_a, finite_a = _normalize_bearings(matches.bearings_a)
    bearings_b, finite_b = _normalize_bearings(matches.bearings_b)
    input_valid = np.asarray(matches.valid, dtype=bool) & finite_a & finite_b

    pixels = rays_to_erp_pixels(bearings_a, dense_result.range.shape)
    sampled_range, sampled_confidence, valid_weight = _sample_dense_result(
        dense_result, np.asarray(pixels.pixels_xy, dtype=np.float64)
    )
    dense_supported = (
        input_valid
        & np.asarray(pixels.valid, dtype=bool)
        & (valid_weight >= settings.min_valid_weight)
        & np.isfinite(sampled_range)
        & (sampled_range > 0.0)
        & np.isfinite(sampled_confidence)
        & (sampled_confidence >= settings.min_dense_confidence)
    )

    predicted = np.full_like(bearings_b, np.nan, dtype=np.float64)
    if np.any(dense_supported):
        points_a = bearings_a[dense_supported] * sampled_range[dense_supported, None]
        points_b = points_a @ rotation.T + translation
        point_norm = np.linalg.norm(points_b, axis=1)
        finite_prediction = np.isfinite(points_b).all(axis=1) & (point_norm > 0.0)
        supported_indices = np.flatnonzero(dense_supported)
        failed_indices = supported_indices[~finite_prediction]
        dense_supported[failed_indices] = False
        good_indices = supported_indices[finite_prediction]
        predicted[good_indices] = (
            points_b[finite_prediction] / point_norm[finite_prediction, None]
        )

    residual = np.full(len(matches), np.nan, dtype=np.float64)
    if np.any(dense_supported):
        dot = np.sum(predicted[dense_supported] * bearings_b[dense_supported], axis=1)
        residual[dense_supported] = np.arccos(np.clip(dot, -1.0, 1.0))
    threshold = math.radians(settings.max_angular_error_deg)
    accepted = dense_supported & np.isfinite(residual) & (residual <= threshold)
    rejected = dense_supported & ~accepted
    return DenseMatchFilterResult(
        input_valid_mask=input_valid,
        dense_supported_mask=dense_supported,
        accepted_mask=accepted,
        rejected_mask=rejected,
        angular_residual_rad=residual,
        sampled_range=sampled_range,
        sampled_confidence=sampled_confidence,
        sampled_valid_weight=valid_weight,
        bearings_a=matches.bearings_a,
        bearings_b=matches.bearings_b,
        predicted_bearings_b=predicted,
        options=settings,
    )


def _normalize_bearings(value: object) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("match bearings must have shape (N, 3)")
    norms = np.linalg.norm(array, axis=1)
    valid = np.isfinite(array).all(axis=1) & np.isfinite(norms) & (norms > 0.0)
    normalized = np.full_like(array, np.nan)
    normalized[valid] = array[valid] / norms[valid, None]
    return normalized, valid


def _sample_dense_result(
    dense_result: SphericalStereoResult, pixels_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = dense_result.range.shape
    x = pixels_xy[:, 0]
    y = pixels_xy[:, 1]
    finite_pixels = np.isfinite(pixels_xy).all(axis=1)
    safe_x = np.where(finite_pixels, x, 0.0)
    safe_y = np.where(finite_pixels, y, 0.0)
    x0 = np.floor(safe_x).astype(np.int64)
    y0 = np.floor(safe_y).astype(np.int64)
    dx = safe_x - x0
    dy = safe_y - y0
    weighted_range = np.zeros(x.shape, dtype=np.float64)
    weighted_confidence = np.zeros(x.shape, dtype=np.float64)
    valid_weight = np.zeros(x.shape, dtype=np.float64)
    for offset_x, offset_y, weight in (
        (0, 0, (1.0 - dx) * (1.0 - dy)),
        (1, 0, dx * (1.0 - dy)),
        (0, 1, (1.0 - dx) * dy),
        (1, 1, dx * dy),
    ):
        neighbour_y = y0 + offset_y
        neighbour_x = np.mod(x0 + offset_x, width)
        inside = finite_pixels & (neighbour_y >= 0) & (neighbour_y < height)
        safe_y = np.clip(neighbour_y, 0, height - 1)
        ranges = dense_result.range[safe_y, neighbour_x]
        confidence = dense_result.confidence[safe_y, neighbour_x]
        neighbour_valid = (
            inside
            & dense_result.validity_mask[safe_y, neighbour_x]
            & np.isfinite(ranges)
            & (ranges > 0.0)
            & np.isfinite(confidence)
        )
        accepted_weight = np.where(neighbour_valid, weight, 0.0)
        valid_weight += accepted_weight
        weighted_range += accepted_weight * np.where(neighbour_valid, ranges, 0.0)
        weighted_confidence += accepted_weight * np.where(
            neighbour_valid, confidence, 0.0
        )

    sampled_range = np.full(x.shape, np.nan, dtype=np.float64)
    sampled_confidence = np.full(x.shape, np.nan, dtype=np.float64)
    has_weight = valid_weight > 0.0
    sampled_range[has_weight] = weighted_range[has_weight] / valid_weight[has_weight]
    sampled_confidence[has_weight] = (
        weighted_confidence[has_weight] / valid_weight[has_weight]
    )
    return sampled_range, sampled_confidence, np.clip(valid_weight, 0.0, 1.0)


def _readonly_vector(value: object, dtype: Any, name: str) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must have shape (N,)")
    array.setflags(write=False)
    return array


def _finite_interval(
    name: str,
    value: object,
    *,
    lower: float,
    upper: float,
    lower_inclusive: bool = True,
) -> None:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number")
    numeric = float(value)
    lower_ok = numeric >= lower if lower_inclusive else numeric > lower
    if not math.isfinite(numeric) or not lower_ok or numeric > upper:
        left = "[" if lower_inclusive else "("
        raise ValueError(f"{name} must be in {left}{lower}, {upper}]")
