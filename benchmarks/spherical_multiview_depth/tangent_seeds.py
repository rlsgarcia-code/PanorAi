"""Sparse tangent-descriptor matches as metric radial-depth seeds.

This module deliberately separates three pieces of evidence:

* a descriptor-neutral spherical bearing correspondence;
* fixed metric ``R,t`` used only to test radial-range hypotheses; and
* conservative propagation inside the angular support of the descriptor.

The feature detector and descriptor adapter live in :mod:`panorai.features`.
They are not reimplemented here.  In particular, the P74 experiment uses the
existing spherical DoG detector, 48x48 tangent patches, and RootSIFT adapter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from panorai.geometry import rays_to_erp_pixels


INTERFACE = "panorai-experimental-tangent-depth-seeds/v1"


@dataclass(frozen=True, slots=True)
class TangentDepthSeedOptions:
    """Gates for depth search around a frozen monocular radial-range prior."""

    min_range_m: float = 0.3
    max_range_m: float = 15.0
    hypotheses: int = 65
    range_factor: float = 1.5
    maximum_reprojection_error_deg: float = 0.35
    minimum_parallax_deg: float = 0.35
    maximum_ray_miss_m: float = 0.12
    require_interior_minimum: bool = True
    descriptor_radius_sigmas: float = 6.0
    minimum_propagation_radius_deg: float = 0.20
    maximum_propagation_radius_deg: float = 4.0
    propagation_color_sigma: float = 0.12
    minimum_propagation_weight: float = 0.05

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_range_m) or self.min_range_m <= 0.0:
            raise ValueError("min_range_m must be positive and finite")
        if not math.isfinite(self.max_range_m) or self.max_range_m <= self.min_range_m:
            raise ValueError("max_range_m must exceed min_range_m")
        if (
            isinstance(self.hypotheses, bool)
            or not isinstance(self.hypotheses, int)
            or self.hypotheses < 3
            or self.hypotheses % 2 == 0
        ):
            raise ValueError("hypotheses must be an odd integer of at least 3")
        if not math.isfinite(self.range_factor) or self.range_factor <= 1.0:
            raise ValueError("range_factor must exceed 1")
        for name in (
            "maximum_reprojection_error_deg",
            "minimum_parallax_deg",
            "maximum_ray_miss_m",
            "descriptor_radius_sigmas",
            "minimum_propagation_radius_deg",
            "maximum_propagation_radius_deg",
            "propagation_color_sigma",
            "minimum_propagation_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.maximum_propagation_radius_deg < self.minimum_propagation_radius_deg:
            raise ValueError(
                "maximum_propagation_radius_deg must not be below the minimum"
            )
        if not isinstance(self.require_interior_minimum, bool):
            raise TypeError("require_interior_minimum must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TangentDepthSeedResult:
    """Auditable sparse depth solutions aligned with input correspondences."""

    target_bearings: np.ndarray
    source_bearings: np.ndarray
    target_pixels_xy: np.ndarray
    prior_range_m: np.ndarray
    solved_range_m: np.ndarray
    source_range_m: np.ndarray
    accepted: np.ndarray
    reprojection_error_deg: np.ndarray
    parallax_deg: np.ndarray
    ray_miss_m: np.ndarray
    best_hypothesis_index: np.ndarray
    confidence: np.ndarray
    target_scale_deg: np.ndarray
    descriptor_distance: np.ndarray
    options: TangentDepthSeedOptions
    descriptor_adapter: str
    pose_source: str
    interface: str = INTERFACE

    def __post_init__(self) -> None:
        count = np.asarray(self.accepted).size
        for name in ("target_bearings", "source_bearings"):
            value = np.asarray(getattr(self, name))
            if value.shape != (count, 3):
                raise ValueError(f"{name} must have shape (N, 3)")
        if np.asarray(self.target_pixels_xy).shape != (count, 2):
            raise ValueError("target_pixels_xy must have shape (N, 2)")
        for name in (
            "prior_range_m",
            "solved_range_m",
            "source_range_m",
            "accepted",
            "reprojection_error_deg",
            "parallax_deg",
            "ray_miss_m",
            "best_hypothesis_index",
            "confidence",
            "target_scale_deg",
            "descriptor_distance",
        ):
            if np.asarray(getattr(self, name)).shape != (count,):
                raise ValueError(f"{name} must have shape (N,)")

    def describe(self) -> dict[str, Any]:
        accepted = np.asarray(self.accepted, dtype=bool)
        return {
            "interface": self.interface,
            "match_count": int(accepted.size),
            "accepted_count": int(accepted.sum()),
            "descriptor_adapter": self.descriptor_adapter,
            "pose_source": self.pose_source,
            "options": self.options.to_dict(),
            "median_reprojection_error_deg": _finite_median(
                np.asarray(self.reprojection_error_deg)[accepted]
            ),
            "median_parallax_deg": _finite_median(
                np.asarray(self.parallax_deg)[accepted]
            ),
            "median_ray_miss_m": _finite_median(np.asarray(self.ray_miss_m)[accepted]),
            "median_abs_log_correction": _finite_median(
                np.abs(
                    np.log(
                        np.asarray(self.solved_range_m)[accepted]
                        / np.asarray(self.prior_range_m)[accepted]
                    )
                )
            ),
        }


@dataclass(frozen=True, slots=True)
class PropagatedDepthResult:
    """Dense map with sparse corrections confined to descriptor support."""

    radial_range_m: np.ndarray
    log_residual: np.ndarray
    accumulated_weight: np.ndarray
    changed_mask: np.ndarray
    contributing_seed_count: int
    interface: str = INTERFACE


def sample_erp_scalar(image_hw: Any, bearings: Any) -> tuple[np.ndarray, np.ndarray]:
    """Bilinearly sample an ERP scalar with longitude wrap and explicit support."""

    image = np.asarray(image_hw, dtype=np.float64)
    rays = np.asarray(bearings, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError("image_hw must have shape (H, W)")
    if rays.ndim != 2 or rays.shape[1] != 3:
        raise ValueError("bearings must have shape (N, 3)")
    norm = np.linalg.norm(rays, axis=1)
    valid = np.isfinite(rays).all(axis=1) & np.isfinite(norm) & (norm > 0.0)
    normalized = np.zeros_like(rays)
    normalized[valid] = rays[valid] / norm[valid, None]
    pixels = rays_to_erp_pixels(normalized, image.shape)
    x = pixels.pixels_xy[:, 0]
    y = pixels.pixels_xy[:, 1]
    valid &= pixels.valid & np.isfinite(x) & np.isfinite(y)
    height, width = image.shape
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = x - x0
    wy = y - y0
    vertical = (y0 >= 0) & (y1 < height)
    valid &= vertical
    x0 %= width
    x1 %= width
    y0 = np.clip(y0, 0, height - 1)
    y1 = np.clip(y1, 0, height - 1)
    values = (
        image[y0, x0] * (1.0 - wx) * (1.0 - wy)
        + image[y0, x1] * wx * (1.0 - wy)
        + image[y1, x0] * (1.0 - wx) * wy
        + image[y1, x1] * wx * wy
    )
    valid &= np.isfinite(values)
    values[~valid] = np.nan
    return values, pixels.pixels_xy


def solve_tangent_depth_seeds(
    seed_range_hw: Any,
    target_bearings: Any,
    source_bearings: Any,
    rotation_source_from_target: Any,
    translation_source_from_target_m: Any,
    *,
    target_scale_deg: Any | None = None,
    descriptor_distance: Any | None = None,
    match_validity: Any | None = None,
    options: TangentDepthSeedOptions | None = None,
    descriptor_adapter: str = "panorai-tangent-opencv-descriptor/v2:rootsift",
    pose_source: str = "registered-metric",
) -> TangentDepthSeedResult:
    """Resolve radial range around a prior for fixed spherical matches and pose.

    Hypotheses are uniform in log inverse range.  The discrete search identifies
    the local basin around the prior; a two-ray least-squares triangulation then
    provides the continuous range, but is accepted only if it lies inside that
    winning basin and all reprojection, parallax, cheirality, and ray-miss gates
    pass.  This keeps the result tied to the requested prior distribution rather
    than silently replacing it with unconstrained triangulation.
    """

    settings = options or TangentDepthSeedOptions()
    prior_map = np.asarray(seed_range_hw, dtype=np.float64)
    if prior_map.ndim != 2:
        raise ValueError("seed_range_hw must have shape (H, W)")
    target, target_valid = _normalize_bearings(target_bearings)
    source, source_valid = _normalize_bearings(source_bearings)
    if target.shape != source.shape:
        raise ValueError("target and source bearings must have the same shape")
    count = target.shape[0]
    rotation, translation = _validate_pose(
        rotation_source_from_target, translation_source_from_target_m
    )
    scales = _vector_or_default(target_scale_deg, count, 1.0, "target_scale_deg")
    distances = _vector_or_default(
        descriptor_distance, count, math.nan, "descriptor_distance"
    )
    validity = target_valid & source_valid
    if match_validity is not None:
        match_mask = np.asarray(match_validity, dtype=bool)
        if match_mask.shape != (count,):
            raise ValueError("match_validity must have shape (N,)")
        validity &= match_mask

    prior, pixels = sample_erp_scalar(prior_map, target)
    validity &= (
        np.isfinite(prior)
        & (prior >= settings.min_range_m)
        & (prior <= settings.max_range_m)
    )
    offsets = np.linspace(
        -math.log(settings.range_factor),
        math.log(settings.range_factor),
        settings.hypotheses,
        dtype=np.float64,
    )
    inverse = np.reciprocal(np.maximum(prior[:, None], 1e-12)) * np.exp(
        offsets[None, :]
    )
    hypotheses = np.clip(
        np.reciprocal(inverse), settings.min_range_m, settings.max_range_m
    )
    rotated_target = target @ rotation.T
    points = rotated_target[:, None, :] * hypotheses[:, :, None] + translation
    norms = np.linalg.norm(points, axis=2)
    projected = points / np.maximum(norms[:, :, None], 1e-12)
    cosine = np.einsum("nhc,nc->nh", projected, source)
    errors = np.arccos(np.clip(cosine, -1.0, 1.0))
    errors[~validity] = np.inf
    best_index = np.argmin(errors, axis=1)

    # Continuous closest points on the two bearing rays.
    solved = np.full(count, np.nan, dtype=np.float64)
    source_range = np.full(count, np.nan, dtype=np.float64)
    ray_miss = np.full(count, np.nan, dtype=np.float64)
    for index in np.flatnonzero(validity):
        design = np.column_stack((rotated_target[index], -source[index]))
        solution, *_ = np.linalg.lstsq(design, -translation, rcond=None)
        solved[index] = solution[0]
        source_range[index] = solution[1]
        ray_miss[index] = float(np.linalg.norm(design @ solution + translation))

    continuous_points = rotated_target * solved[:, None] + translation
    continuous_norm = np.linalg.norm(continuous_points, axis=1)
    continuous_projected = continuous_points / np.maximum(
        continuous_norm[:, None], 1e-12
    )
    reprojection = np.degrees(
        np.arccos(
            np.clip(np.einsum("nc,nc->n", continuous_projected, source), -1.0, 1.0)
        )
    )
    parallax = np.degrees(
        np.arccos(np.clip(np.einsum("nc,nc->n", rotated_target, source), -1.0, 1.0))
    )
    lower = np.minimum(hypotheses[:, 0], hypotheses[:, -1])
    upper = np.maximum(hypotheses[:, 0], hypotheses[:, -1])
    accepted = (
        validity
        & np.isfinite(solved)
        & np.isfinite(source_range)
        & (solved >= settings.min_range_m)
        & (solved <= settings.max_range_m)
        & (source_range > 0.0)
        & (solved >= lower)
        & (solved <= upper)
        & (reprojection <= settings.maximum_reprojection_error_deg)
        & (parallax >= settings.minimum_parallax_deg)
        & (ray_miss <= settings.maximum_ray_miss_m)
    )
    if settings.require_interior_minimum:
        accepted &= (best_index > 0) & (best_index < settings.hypotheses - 1)
        left = hypotheses[np.arange(count), np.maximum(best_index - 1, 0)]
        right = hypotheses[
            np.arange(count), np.minimum(best_index + 1, settings.hypotheses - 1)
        ]
        accepted &= solved >= np.minimum(left, right)
        accepted &= solved <= np.maximum(left, right)

    geometric = np.clip(
        1.0 - reprojection / settings.maximum_reprojection_error_deg, 0.0, 1.0
    )
    parallax_confidence = np.clip(
        parallax / max(2.0 * settings.minimum_parallax_deg, 1e-12), 0.0, 1.0
    )
    miss_confidence = np.clip(1.0 - ray_miss / settings.maximum_ray_miss_m, 0.0, 1.0)
    confidence = geometric * parallax_confidence * miss_confidence
    confidence[~accepted] = 0.0

    return TangentDepthSeedResult(
        target_bearings=target,
        source_bearings=source,
        target_pixels_xy=pixels,
        prior_range_m=prior,
        solved_range_m=solved,
        source_range_m=source_range,
        accepted=accepted,
        reprojection_error_deg=reprojection,
        parallax_deg=parallax,
        ray_miss_m=ray_miss,
        best_hypothesis_index=best_index,
        confidence=confidence,
        target_scale_deg=scales,
        descriptor_distance=distances,
        options=settings,
        descriptor_adapter=descriptor_adapter,
        pose_source=pose_source,
    )


def propagate_tangent_depth_seeds(
    seed_range_hw: Any,
    target_rgb: Any,
    results: tuple[TangentDepthSeedResult, ...],
    *,
    options: TangentDepthSeedOptions | None = None,
) -> PropagatedDepthResult:
    """Splat accepted log-range corrections inside tangent descriptor support.

    No image or depth resize is performed.  Spatial weights use exact spherical
    angular distance; the RGB factor prevents a seed from freely crossing a
    strong appearance boundary.  Pixels without sufficient accumulated weight
    remain bit-identical to the input depth map.
    """

    settings = options or TangentDepthSeedOptions()
    seed_map = np.asarray(seed_range_hw, dtype=np.float32)
    rgb = np.asarray(target_rgb)
    if seed_map.ndim != 2:
        raise ValueError("seed_range_hw must have shape (H, W)")
    if rgb.shape != (*seed_map.shape, 3):
        raise ValueError("target_rgb must have shape (H, W, 3)")
    rgb_float = rgb.astype(np.float32)
    if np.issubdtype(rgb.dtype, np.integer):
        rgb_float /= float(np.iinfo(rgb.dtype).max)
    if (
        not np.isfinite(rgb_float).all()
        or rgb_float.min() < 0.0
        or rgb_float.max() > 1.0
    ):
        raise ValueError("target_rgb must be finite and lie in [0, 1]")

    height, width = seed_map.shape
    weighted_delta = np.zeros(seed_map.shape, dtype=np.float64)
    accumulated = np.zeros(seed_map.shape, dtype=np.float64)
    contributing = 0
    latitude_centers = (
        math.pi / 2.0 - (np.arange(height, dtype=np.float64) + 0.5) / height * math.pi
    )
    for result in results:
        if result.options != settings:
            raise ValueError("all seed results must use the propagation options")
        for index in np.flatnonzero(result.accepted):
            correction = math.log(
                float(result.solved_range_m[index] / result.prior_range_m[index])
            )
            if not math.isfinite(correction):
                continue
            center = result.target_bearings[index]
            center_y = float(np.clip(center[1], -1.0, 1.0))
            center_lat = math.asin(center_y)
            center_lon = math.atan2(float(center[0]), float(center[2]))
            radius_deg = float(
                np.clip(
                    result.target_scale_deg[index] * settings.descriptor_radius_sigmas,
                    settings.minimum_propagation_radius_deg,
                    settings.maximum_propagation_radius_deg,
                )
            )
            radius_rad = math.radians(radius_deg)
            row_radius = max(1, int(math.ceil(radius_rad / math.pi * height)))
            center_row = int(
                round((math.pi / 2.0 - center_lat) / math.pi * height - 0.5)
            )
            row_start = max(0, center_row - row_radius)
            row_stop = min(height, center_row + row_radius + 1)
            if row_start >= row_stop:
                continue
            local_lat = latitude_centers[row_start:row_stop]
            minimum_cos = max(
                1e-3,
                float(np.min(np.abs(np.cos(local_lat)))) if local_lat.size else 1e-3,
            )
            column_radius = min(
                width // 2,
                max(
                    1,
                    int(math.ceil(radius_rad / (2.0 * math.pi) * width / minimum_cos)),
                ),
            )
            center_col = int(
                round((center_lon + math.pi) / (2.0 * math.pi) * width - 0.5)
            )
            unwrapped_columns = np.arange(
                center_col - column_radius, center_col + column_radius + 1
            )
            columns = np.mod(unwrapped_columns, width)
            longitude = (unwrapped_columns.astype(np.float64) + 0.5) / width * (
                2.0 * math.pi
            ) - math.pi
            lon_delta = (longitude - center_lon + math.pi) % (2.0 * math.pi) - math.pi
            cosine = (
                np.sin(local_lat)[:, None] * center_y
                + np.cos(local_lat)[:, None]
                * math.cos(center_lat)
                * np.cos(lon_delta)[None, :]
            )
            angle = np.arccos(np.clip(cosine, -1.0, 1.0))
            inside = angle <= radius_rad
            if not inside.any():
                continue
            spatial_sigma = max(radius_rad / 2.0, 1e-8)
            spatial = np.exp(-0.5 * (angle / spatial_sigma) ** 2)
            target_x = int(round(result.target_pixels_xy[index, 0])) % width
            target_y = int(
                np.clip(round(result.target_pixels_xy[index, 1]), 0, height - 1)
            )
            center_color = rgb_float[target_y, target_x]
            local_color = rgb_float[row_start:row_stop][:, columns]
            color_distance = np.mean((local_color - center_color) ** 2, axis=2)
            color = np.exp(
                -0.5 * color_distance / max(settings.propagation_color_sigma**2, 1e-12)
            )
            weight = result.confidence[index] * spatial * color * inside
            row_indices = np.arange(row_start, row_stop)[:, None]
            weighted_delta[row_indices, columns[None, :]] += weight * correction
            accumulated[row_indices, columns[None, :]] += weight
            contributing += 1

    changed = accumulated >= settings.minimum_propagation_weight
    log_residual = np.zeros(seed_map.shape, dtype=np.float32)
    log_residual[changed] = (weighted_delta[changed] / accumulated[changed]).astype(
        np.float32
    )
    refined = seed_map.copy()
    refined[changed] = np.clip(
        seed_map[changed] * np.exp(log_residual[changed]),
        settings.min_range_m,
        settings.max_range_m,
    )
    return PropagatedDepthResult(
        radial_range_m=refined,
        log_residual=log_residual,
        accumulated_weight=accumulated.astype(np.float32),
        changed_mask=changed,
        contributing_seed_count=contributing,
    )


def _normalize_bearings(value: Any) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("bearings must have shape (N, 3)")
    norm = np.linalg.norm(array, axis=1)
    valid = np.isfinite(array).all(axis=1) & np.isfinite(norm) & (norm > 0.0)
    result = np.zeros_like(array)
    result[valid] = array[valid] / norm[valid, None]
    return result, valid


def _validate_pose(rotation: Any, translation: Any) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(rotation, dtype=np.float64)
    vector = np.asarray(translation, dtype=np.float64)
    if matrix.shape != (3, 3) or vector.shape != (3,):
        raise ValueError("rotation must be 3x3 and translation must have shape (3,)")
    if not np.isfinite(matrix).all() or not np.isfinite(vector).all():
        raise ValueError("pose must be finite")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-7, rtol=0.0):
        raise ValueError("rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(matrix)), 1.0, abs_tol=1e-7):
        raise ValueError("rotation must have determinant +1")
    if np.linalg.norm(vector) <= 1e-9:
        raise ValueError("translation must have non-zero metric scale")
    return matrix, vector


def _vector_or_default(
    value: Any | None, count: int, default: float, name: str
) -> np.ndarray:
    if value is None:
        return np.full(count, default, dtype=np.float64)
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (count,):
        raise ValueError(f"{name} must have shape (N,)")
    return array


def _finite_median(value: np.ndarray) -> float | None:
    finite = np.asarray(value)[np.isfinite(value)]
    return None if not finite.size else float(np.median(finite))


__all__ = [
    "INTERFACE",
    "PropagatedDepthResult",
    "TangentDepthSeedOptions",
    "TangentDepthSeedResult",
    "propagate_tangent_depth_seeds",
    "sample_erp_scalar",
    "solve_tangent_depth_seeds",
]
