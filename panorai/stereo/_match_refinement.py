"""Provenance-preserving spherical subpixel match refinement."""

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

from ._dense import _prepare_image, _seam_safe_remap, _validate_pose
from ._match_filter import DenseMatchFilterResult

_INTERFACE = "panorai-spherical-match-refinement/v1-experimental"


@dataclass(frozen=True, slots=True)
class SphericalMatchRefinementOptions:
    """Configuration for tangent-patch target-bearing refinement."""

    patch_radius_px: int = 2
    search_radius_px: float = 0.6
    search_step_px: float = 0.1
    min_patch_support: float = 0.9
    min_patch_std: float = 0.01
    min_cost_improvement: float = 0.02
    max_angular_shift_deg: float = 1.5

    def __post_init__(self) -> None:
        if (
            isinstance(self.patch_radius_px, bool)
            or not isinstance(self.patch_radius_px, (int, np.integer))
            or self.patch_radius_px < 1
        ):
            raise ValueError("patch_radius_px must be a positive integer")
        _positive_finite("search_radius_px", self.search_radius_px)
        _positive_finite("search_step_px", self.search_step_px)
        if self.search_step_px > self.search_radius_px:
            raise ValueError("search_step_px must not exceed search_radius_px")
        _closed_fraction("min_patch_support", self.min_patch_support)
        _closed_fraction("min_patch_std", self.min_patch_std)
        _nonnegative_finite("min_cost_improvement", self.min_cost_improvement)
        _positive_finite("max_angular_shift_deg", self.max_angular_shift_deg)
        if self.max_angular_shift_deg > 180.0:
            raise ValueError("max_angular_shift_deg must not exceed 180")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MatchRefinementProvenance:
    """Inputs and selection policy for one refinement result."""

    source_checksums: tuple[str, str]
    source_match_interface: str
    dense_filter_interface: str
    method: str = "dense-center-range-pose-warped-zncc-grid"
    selection_reason: str = "minimum-cost-with-explicit-improvement-gate"
    interface: str = _INTERFACE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SphericalMatchRefinementResult:
    """Immutable original/refined target bearings and photometric evidence."""

    bearings_a: np.ndarray
    original_bearings_b: np.ndarray
    refined_bearings_b: np.ndarray
    eligible_mask: np.ndarray
    evaluated_mask: np.ndarray
    applied_mask: np.ndarray
    original_cost: np.ndarray
    candidate_cost: np.ndarray
    cost_improvement: np.ndarray
    tangent_offset_rad: np.ndarray
    angular_shift_rad: np.ndarray
    options: SphericalMatchRefinementOptions
    provenance: MatchRefinementProvenance
    interface: str = _INTERFACE
    stability: str = "experimental"

    def __post_init__(self) -> None:
        bearings_a = _readonly_matrix(self.bearings_a, 3, "bearings_a")
        original = _readonly_matrix(self.original_bearings_b, 3, "original_bearings_b")
        refined = _readonly_matrix(self.refined_bearings_b, 3, "refined_bearings_b")
        count = bearings_a.shape[0]
        if original.shape != (count, 3) or refined.shape != (count, 3):
            raise ValueError("all bearing arrays must have shape (N, 3)")
        eligible = _readonly_vector(self.eligible_mask, np.bool_, "eligible_mask")
        evaluated = _readonly_vector(self.evaluated_mask, np.bool_, "evaluated_mask")
        applied = _readonly_vector(self.applied_mask, np.bool_, "applied_mask")
        original_cost = _readonly_vector(
            self.original_cost, np.float64, "original_cost"
        )
        candidate_cost = _readonly_vector(
            self.candidate_cost, np.float64, "candidate_cost"
        )
        improvement = _readonly_vector(
            self.cost_improvement, np.float64, "cost_improvement"
        )
        offsets = _readonly_matrix(self.tangent_offset_rad, 2, "tangent_offset_rad")
        shift = _readonly_vector(
            self.angular_shift_rad, np.float64, "angular_shift_rad"
        )
        vectors = (
            eligible,
            evaluated,
            applied,
            original_cost,
            candidate_cost,
            improvement,
            shift,
        )
        if any(value.shape != (count,) for value in vectors):
            raise ValueError("all refinement vectors must have shape (N,)")
        if offsets.shape != (count, 2):
            raise ValueError("tangent_offset_rad must have shape (N, 2)")
        if np.any(evaluated & ~eligible) or np.any(applied & ~evaluated):
            raise ValueError("applied must be a subset of evaluated and eligible")
        unchanged = ~applied
        if not np.array_equal(refined[unchanged], original[unchanged]):
            raise ValueError("unapplied rows must preserve original target bearings")
        object.__setattr__(self, "bearings_a", bearings_a)
        object.__setattr__(self, "original_bearings_b", original)
        object.__setattr__(self, "refined_bearings_b", refined)
        object.__setattr__(self, "eligible_mask", eligible)
        object.__setattr__(self, "evaluated_mask", evaluated)
        object.__setattr__(self, "applied_mask", applied)
        object.__setattr__(self, "original_cost", original_cost)
        object.__setattr__(self, "candidate_cost", candidate_cost)
        object.__setattr__(self, "cost_improvement", improvement)
        object.__setattr__(self, "tangent_offset_rad", offsets)
        object.__setattr__(self, "angular_shift_rad", shift)

    def to_bearing_correspondences(self) -> SphericalBearingCorrespondences:
        """Expose filtered bearings, refined only where the gate applied."""

        valid = self.eligible_mask.copy()
        return SphericalBearingCorrespondences(
            bearings_a=np.array(self.bearings_a, copy=True),
            bearings_b=np.array(self.refined_bearings_b, copy=True),
            weights=valid.astype(np.float32),
            valid=valid,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "stability": self.stability,
            "match_count": int(self.eligible_mask.size),
            "eligible_count": int(self.eligible_mask.sum()),
            "evaluated_count": int(self.evaluated_mask.sum()),
            "applied_count": int(self.applied_mask.sum()),
            "options": self.options.to_dict(),
            "provenance": self.provenance.to_dict(),
        }


def refine_matches_on_sphere(
    reference_erp: np.ndarray,
    target_erp: np.ndarray,
    matches: SphericalFeatureMatches,
    dense_filter: DenseMatchFilterResult,
    rotation_b_from_a: np.ndarray,
    translation_b_from_a: np.ndarray,
    *,
    options: SphericalMatchRefinementOptions | None = None,
) -> SphericalMatchRefinementResult:
    """Refine accepted target bearings with transported spherical patches.

    Search coordinates are angular offsets in the tangent plane around the
    dense-predicted target bearing. Source patch rays are lifted with the
    center's dense range and transformed by the complete supplied ``R,t``;
    candidate offsets then shift that locally pose-warped target patch. Only a
    finite, supported candidate with explicit ZNCC-cost improvement and a
    bounded shift replaces the original target bearing.
    """

    if not isinstance(matches, SphericalFeatureMatches):
        raise TypeError("matches must be a SphericalFeatureMatches object")
    if not isinstance(dense_filter, DenseMatchFilterResult):
        raise TypeError("dense_filter must be a DenseMatchFilterResult object")
    if len(matches) != dense_filter.accepted_mask.size:
        raise ValueError("matches and dense_filter must have the same length")
    if not np.array_equal(
        matches.bearings_a, dense_filter.bearings_a
    ) or not np.array_equal(matches.bearings_b, dense_filter.bearings_b):
        raise ValueError("dense_filter was not produced from these match bearings")
    settings = options or SphericalMatchRefinementOptions()
    rotation, translation = _validate_pose(rotation_b_from_a, translation_b_from_a)
    if np.linalg.norm(translation) <= 1e-9:
        raise ValueError("translation_b_from_a must have non-zero scale")
    reference = _prepare_image(reference_erp, "reference_erp")
    target = _prepare_image(target_erp, "target_erp")
    if reference.shape != target.shape:
        raise ValueError("reference_erp and target_erp must have the same HW shape")

    count = len(matches)
    original = np.array(matches.bearings_b, dtype=np.float64, copy=True)
    refined = original.copy()
    eligible = dense_filter.accepted_mask.copy()
    evaluated = np.zeros(count, dtype=bool)
    applied = np.zeros(count, dtype=bool)
    original_cost = np.full(count, np.nan, dtype=np.float64)
    candidate_cost = np.full(count, np.nan, dtype=np.float64)
    improvement = np.full(count, np.nan, dtype=np.float64)
    offsets = np.full((count, 2), np.nan, dtype=np.float64)
    shift = np.zeros(count, dtype=np.float64)

    angular_pixel = math.pi / reference.shape[0]
    patch_offsets = _square_offsets(settings.patch_radius_px) * angular_pixel
    search_offsets = _search_offsets(settings) * angular_pixel
    normalized_a, valid_a = _normalize_bearings(matches.bearings_a)
    normalized_b, valid_b = _normalize_bearings(matches.bearings_b)
    eligible &= valid_a & valid_b

    for index in np.flatnonzero(eligible):
        prediction = dense_filter.predicted_bearings_b[index]
        if not np.isfinite(prediction).all():
            continue
        source_basis = _tangent_basis(normalized_a[index])
        source_rays = _patch_rays(normalized_a[index], source_basis, patch_offsets)
        reference_patch, reference_support = _sample_rays(
            reference, source_rays[None, :, :]
        )
        prediction_basis = _tangent_basis(prediction)
        candidate_centers = _offset_bearings(
            prediction, prediction_basis, search_offsets
        )
        range_value = dense_filter.sampled_range[index]
        points_b = (source_rays * range_value) @ rotation.T + translation
        base_target_rays = points_b / np.linalg.norm(points_b, axis=1, keepdims=True)
        candidate_patch_rays = _shift_patch_rays(
            base_target_rays, prediction_basis, search_offsets
        )
        target_patches, target_support = _sample_rays(target, candidate_patch_rays)
        costs = _zncc_costs(
            reference_patch[0],
            reference_support[0],
            target_patches,
            target_support,
            settings,
        )

        original_offset = _tangent_coordinates(
            prediction, prediction_basis, normalized_b[index]
        )
        original_patch_rays = _shift_patch_rays(
            base_target_rays, prediction_basis, original_offset[None, :]
        )
        original_patch, original_support = _sample_rays(target, original_patch_rays)
        baseline_cost = _zncc_costs(
            reference_patch[0],
            reference_support[0],
            original_patch,
            original_support,
            settings,
        )[0]
        if not np.isfinite(baseline_cost) or not np.isfinite(costs).any():
            continue

        best_index = int(np.nanargmin(costs))
        best_cost = float(costs[best_index])
        best_bearing = candidate_centers[best_index]
        best_shift = _angular_distance(normalized_b[index], best_bearing)
        gain = float(baseline_cost - best_cost)
        evaluated[index] = True
        original_cost[index] = baseline_cost
        candidate_cost[index] = best_cost
        improvement[index] = gain
        offsets[index] = search_offsets[best_index]
        if gain >= settings.min_cost_improvement and best_shift <= math.radians(
            settings.max_angular_shift_deg
        ):
            applied[index] = True
            refined[index] = best_bearing
            shift[index] = best_shift

    provenance = MatchRefinementProvenance(
        source_checksums=matches.provenance.source_checksums,
        source_match_interface=matches.interface,
        dense_filter_interface=dense_filter.interface,
    )
    return SphericalMatchRefinementResult(
        bearings_a=matches.bearings_a,
        original_bearings_b=matches.bearings_b,
        refined_bearings_b=refined,
        eligible_mask=eligible,
        evaluated_mask=evaluated,
        applied_mask=applied,
        original_cost=original_cost,
        candidate_cost=candidate_cost,
        cost_improvement=improvement,
        tangent_offset_rad=offsets,
        angular_shift_rad=shift,
        options=settings,
        provenance=provenance,
    )


def _square_offsets(radius: int) -> np.ndarray:
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    x, y = np.meshgrid(coordinates, coordinates)
    return np.stack((x.ravel(), y.ravel()), axis=1)


def _search_offsets(options: SphericalMatchRefinementOptions) -> np.ndarray:
    count = int(math.floor(options.search_radius_px / options.search_step_px))
    coordinates = np.arange(-count, count + 1, dtype=np.float64)
    coordinates *= options.search_step_px
    x, y = np.meshgrid(coordinates, coordinates)
    offsets = np.stack((x.ravel(), y.ravel()), axis=1)
    radius = np.linalg.norm(offsets, axis=1)
    return offsets[radius <= options.search_radius_px + 1e-12]


def _normalize_bearings(value: object) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(value, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1)
    valid = np.isfinite(array).all(axis=1) & np.isfinite(norms) & (norms > 0.0)
    result = np.full_like(array, np.nan)
    result[valid] = array[valid] / norms[valid, None]
    return result, valid


def _tangent_basis(bearing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    up = np.asarray((0.0, 1.0, 0.0))
    east = np.cross(up, bearing)
    if np.linalg.norm(east) <= 1e-8:
        east = np.cross(np.asarray((0.0, 0.0, 1.0)), bearing)
    east /= np.linalg.norm(east)
    north = np.cross(bearing, east)
    north /= np.linalg.norm(north)
    return east, north


def _offset_bearings(
    center: np.ndarray,
    basis: tuple[np.ndarray, np.ndarray],
    offsets_rad: np.ndarray,
) -> np.ndarray:
    values = (
        center[None, :]
        + offsets_rad[:, :1] * basis[0][None, :]
        + offsets_rad[:, 1:] * basis[1][None, :]
    )
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def _patch_rays(
    center: np.ndarray,
    basis: tuple[np.ndarray, np.ndarray],
    offsets_rad: np.ndarray,
) -> np.ndarray:
    rays = (
        center[None, :]
        + offsets_rad[:, :1] * basis[0][None, :]
        + offsets_rad[:, 1:] * basis[1][None, :]
    )
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def _shift_patch_rays(
    base_rays: np.ndarray,
    basis: tuple[np.ndarray, np.ndarray],
    offsets_rad: np.ndarray,
) -> np.ndarray:
    rays = (
        base_rays[None, :, :]
        + offsets_rad[:, None, :1] * basis[0][None, None, :]
        + offsets_rad[:, None, 1:] * basis[1][None, None, :]
    )
    return rays / np.linalg.norm(rays, axis=2, keepdims=True)


def _tangent_coordinates(
    center: np.ndarray,
    basis: tuple[np.ndarray, np.ndarray],
    bearing: np.ndarray,
) -> np.ndarray:
    denominator = float(center @ bearing)
    if not math.isfinite(denominator) or denominator <= 0.0:
        return np.asarray((math.nan, math.nan))
    return np.asarray(
        ((basis[0] @ bearing) / denominator, (basis[1] @ bearing) / denominator)
    )


def _sample_rays(image: np.ndarray, rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pixels = rays_to_erp_pixels(rays, image.shape)
    sampled, support = _seam_safe_remap(image, pixels.pixels_xy)
    return sampled[..., 0].astype(np.float64), support & pixels.valid


def _zncc_costs(
    reference: np.ndarray,
    reference_support: np.ndarray,
    targets: np.ndarray,
    target_support: np.ndarray,
    options: SphericalMatchRefinementOptions,
) -> np.ndarray:
    support = target_support & reference_support[None, :]
    count = support.sum(axis=1)
    minimum = math.ceil(options.min_patch_support * reference.size)
    safe_count = np.maximum(count, 1)
    reference_values = np.where(support, reference[None, :], 0.0)
    target_values = np.where(support, targets, 0.0)
    reference_mean = reference_values.sum(axis=1) / safe_count
    target_mean = target_values.sum(axis=1) / safe_count
    centered_reference = np.where(
        support, reference[None, :] - reference_mean[:, None], 0.0
    )
    centered_target = np.where(support, targets - target_mean[:, None], 0.0)
    reference_std = np.sqrt(
        np.sum(centered_reference * centered_reference, axis=1) / safe_count
    )
    target_std = np.sqrt(np.sum(centered_target * centered_target, axis=1) / safe_count)
    denominator = reference_std * target_std
    covariance = np.sum(centered_reference * centered_target, axis=1) / safe_count
    valid = (
        (count >= minimum)
        & np.isfinite(denominator)
        & (reference_std >= options.min_patch_std)
        & (target_std >= options.min_patch_std)
        & (denominator > 0.0)
    )
    costs = np.full(targets.shape[0], np.nan, dtype=np.float64)
    costs[valid] = 1.0 - np.clip(covariance[valid] / denominator[valid], -1.0, 1.0)
    return costs


def _angular_distance(left: np.ndarray, right: np.ndarray) -> float:
    return math.acos(float(np.clip(left @ right, -1.0, 1.0)))


def _readonly_vector(value: object, dtype: Any, name: str) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must have shape (N,)")
    array.setflags(write=False)
    return array


def _readonly_matrix(value: object, width: int, name: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape (N, {width})")
    array.setflags(write=False)
    return array


def _positive_finite(name: str, value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be positive and finite")


def _nonnegative_finite(name: str, value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be non-negative and finite")


def _closed_fraction(name: str, value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} must be finite and in [0, 1]")
