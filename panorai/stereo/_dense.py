"""Direct plane-sweep stereo on equirectangular panoramas.

For each reference ERP pixel and radial-range hypothesis, the implementation
forms a 3D point along the canonical PanorAi bearing, transforms it into the
second camera, and samples the second ERP with horizontal seam wrapping.  The
current Experimental implementation uses locally normalized intensity and
gradient costs, local cost aggregation, winner-take-all inverse range, and
sub-hypothesis parabolic refinement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any

import cv2
import numpy as np

from panorai.geometry import erp_pixels_to_rays, rays_to_erp_pixels
from panorai.image_processing import spherical_box_blur, spherical_gradient

_INTERFACE = "panorai-spherical-dense-stereo/v1-experimental"
_EPS = np.finfo(np.float32).eps


@dataclass(frozen=True, slots=True)
class SphericalStereoOptions:
    """Configuration for direct spherical inverse-range plane sweep.

    ``min_range`` and ``max_range`` use the same unit as the supplied metric
    translation.  Results are metres only when ``translation_b_from_a`` is in
    metres.  ``window_size`` defines an odd tangent-plane spherical
    neighbourhood.  ``filter_backend='auto'`` selects PanorAi's first-party
    C++ spherical-convolution kernel when it is available and otherwise uses
    the NumPy reference implementation. ``pyramid_levels > 1`` enables an
    adaptive broad-coarse/local-fine inverse-range search without changing the
    supplied pose.
    """

    min_range: float = 0.30
    max_range: float = 12.0
    num_hypotheses: int = 96
    window_size: int = 7
    pole_margin_fraction: float = 0.04
    min_texture_std: float = 0.015
    min_confidence: float = 0.015
    max_matching_cost: float = 0.65
    intensity_weight: float = 0.75
    gradient_weight: float = 0.25
    smoothness_penalty_small: float = 0.02
    smoothness_penalty_large: float = 0.10
    bidirectional_consistency: bool = True
    consistency_relative_tolerance: float = 0.05
    consistency_absolute_tolerance: float = 0.05
    filter_backend: str = "auto"
    pyramid_levels: int = 1
    refinement_hypotheses: int = 9
    refinement_radius_steps: float = 4.0

    def __post_init__(self) -> None:
        _positive_finite("min_range", self.min_range)
        _positive_finite("max_range", self.max_range)
        if self.max_range <= self.min_range:
            raise ValueError("max_range must be greater than min_range")
        if isinstance(self.num_hypotheses, bool) or not isinstance(
            self.num_hypotheses, (int, np.integer)
        ):
            raise TypeError("num_hypotheses must be an integer")
        if self.num_hypotheses < 3:
            raise ValueError("num_hypotheses must be at least 3")
        if (
            isinstance(self.window_size, bool)
            or not isinstance(self.window_size, (int, np.integer))
            or self.window_size <= 0
            or self.window_size % 2 == 0
        ):
            raise ValueError("window_size must be a positive odd integer")
        _closed_fraction("pole_margin_fraction", self.pole_margin_fraction)
        _closed_fraction("min_texture_std", self.min_texture_std)
        _closed_fraction("min_confidence", self.min_confidence)
        _closed_fraction("max_matching_cost", self.max_matching_cost)
        _closed_fraction("intensity_weight", self.intensity_weight)
        _closed_fraction("gradient_weight", self.gradient_weight)
        if not math.isclose(
            self.intensity_weight + self.gradient_weight, 1.0, abs_tol=1e-12
        ):
            raise ValueError("intensity_weight and gradient_weight must sum to 1")
        _nonnegative_finite("smoothness_penalty_small", self.smoothness_penalty_small)
        _nonnegative_finite("smoothness_penalty_large", self.smoothness_penalty_large)
        if self.smoothness_penalty_large < self.smoothness_penalty_small:
            raise ValueError(
                "smoothness_penalty_large must be at least smoothness_penalty_small"
            )
        if not isinstance(self.bidirectional_consistency, bool):
            raise TypeError("bidirectional_consistency must be a bool")
        _closed_fraction(
            "consistency_relative_tolerance", self.consistency_relative_tolerance
        )
        _nonnegative_finite(
            "consistency_absolute_tolerance", self.consistency_absolute_tolerance
        )
        if self.filter_backend not in {"auto", "numpy", "native"}:
            raise ValueError("filter_backend must be 'auto', 'numpy', or 'native'")
        if (
            isinstance(self.pyramid_levels, bool)
            or not isinstance(self.pyramid_levels, (int, np.integer))
            or self.pyramid_levels < 1
        ):
            raise ValueError("pyramid_levels must be a positive integer")
        if (
            isinstance(self.refinement_hypotheses, bool)
            or not isinstance(self.refinement_hypotheses, (int, np.integer))
            or self.refinement_hypotheses < 3
            or self.refinement_hypotheses % 2 == 0
        ):
            raise ValueError(
                "refinement_hypotheses must be an odd integer of at least 3"
            )
        _positive_finite("refinement_radius_steps", self.refinement_radius_steps)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SphericalStereoResult:
    """Dense radial-range estimate in the reference camera frame.

    Invalid pixels contain ``NaN`` in ``range`` and are identified explicitly
    by ``validity_mask``.  Confidence is the normalized best-versus-second-best
    cost separation and is not a calibrated probability. In adaptive mode,
    ``inverse_range_hypotheses`` contains normalized offsets interpreted with
    the per-pixel ``inverse_range_center`` and ``inverse_range_radius`` maps.
    """

    range: np.ndarray
    validity_mask: np.ndarray
    confidence: np.ndarray
    matching_cost: np.ndarray
    hypothesis_index: np.ndarray
    inverse_range_hypotheses: np.ndarray
    options: SphericalStereoOptions
    interface: str = _INTERFACE
    quantity: str = "radial_range"
    hypothesis_mode: str = "absolute_inverse_range"
    inverse_range_center: np.ndarray | None = None
    inverse_range_radius: np.ndarray | None = None

    def __post_init__(self) -> None:
        range_map = _readonly_image(self.range, np.float32, "range")
        validity = _readonly_image(self.validity_mask, np.bool_, "validity_mask")
        confidence = _readonly_image(self.confidence, np.float32, "confidence")
        cost = _readonly_image(self.matching_cost, np.float32, "matching_cost")
        indices = _readonly_image(self.hypothesis_index, np.int32, "hypothesis_index")
        if not (
            range_map.shape
            == validity.shape
            == confidence.shape
            == cost.shape
            == indices.shape
        ):
            raise ValueError("all result maps must have the same HW shape")
        hypotheses = np.array(
            self.inverse_range_hypotheses, dtype=np.float32, copy=True
        )
        if hypotheses.ndim != 1 or hypotheses.size < 3:
            raise ValueError("inverse_range_hypotheses must have shape (D,), D >= 3")
        if not np.all(np.isfinite(hypotheses)) or not np.all(np.diff(hypotheses) > 0):
            raise ValueError("inverse_range_hypotheses must be finite and increasing")
        hypotheses.setflags(write=False)
        if self.hypothesis_mode not in {
            "absolute_inverse_range",
            "normalized_inverse_range_offset",
        }:
            raise ValueError("unsupported hypothesis_mode")
        center = self.inverse_range_center
        radius = self.inverse_range_radius
        if self.hypothesis_mode == "absolute_inverse_range":
            if center is not None or radius is not None:
                raise ValueError(
                    "absolute hypotheses cannot have inverse-range center/radius maps"
                )
        else:
            if center is None or radius is None:
                raise ValueError(
                    "normalized offset hypotheses require center and radius maps"
                )
            center = _readonly_image(center, np.float32, "inverse_range_center")
            radius = _readonly_image(radius, np.float32, "inverse_range_radius")
            if center.shape != range_map.shape or radius.shape != range_map.shape:
                raise ValueError("inverse-range center/radius must match result shape")
            if not np.all(np.isfinite(center)) or not np.all(radius > 0):
                raise ValueError(
                    "inverse-range center/radius must be finite and positive"
                )
        object.__setattr__(self, "range", range_map)
        object.__setattr__(self, "validity_mask", validity)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "matching_cost", cost)
        object.__setattr__(self, "hypothesis_index", indices)
        object.__setattr__(self, "inverse_range_hypotheses", hypotheses)
        object.__setattr__(self, "inverse_range_center", center)
        object.__setattr__(self, "inverse_range_radius", radius)

    @property
    def range_map(self) -> np.ndarray:
        """Descriptive alias for the radial ``range`` map."""

        return self.range

    def describe(self) -> dict[str, Any]:
        valid = self.validity_mask
        return {
            "interface": self.interface,
            "stability": "experimental",
            "quantity": self.quantity,
            "translation_units": "inherited-by-output-range",
            "shape_hw": list(self.range.shape),
            "valid_fraction": float(valid.mean()),
            "hypothesis_mode": self.hypothesis_mode,
            "options": self.options.to_dict(),
        }


class SphericalDenseStereo:
    """Reusable façade for Experimental direct spherical dense stereo."""

    def __init__(self, options: SphericalStereoOptions | None = None) -> None:
        self.options = options or SphericalStereoOptions()

    def estimate(
        self,
        reference_erp: np.ndarray,
        target_erp: np.ndarray,
        rotation_b_from_a: np.ndarray,
        translation_b_from_a: np.ndarray,
    ) -> SphericalStereoResult:
        return estimate_spherical_range(
            reference_erp,
            target_erp,
            rotation_b_from_a,
            translation_b_from_a,
            options=self.options,
        )


def estimate_spherical_range(
    reference_erp: np.ndarray,
    target_erp: np.ndarray,
    rotation_b_from_a: np.ndarray,
    translation_b_from_a: np.ndarray,
    *,
    options: SphericalStereoOptions | None = None,
) -> SphericalStereoResult:
    """Estimate reference-camera radial range from two central ERPs.

    The pose convention is ``X_b = R_b_from_a @ X_a + t_b_from_a``.  Both
    panoramas must use the same ERP shape and PanorAi's canonical camera frame.
    """

    settings = options or SphericalStereoOptions()
    if settings.bidirectional_consistency:
        one_way = replace(settings, bidirectional_consistency=False)
        forward = estimate_spherical_range(
            reference_erp,
            target_erp,
            rotation_b_from_a,
            translation_b_from_a,
            options=one_way,
        )
        rotation = np.asarray(rotation_b_from_a, dtype=np.float64)
        translation = np.asarray(translation_b_from_a, dtype=np.float64)
        inverse_rotation = rotation.T
        inverse_translation = -inverse_rotation @ translation
        reverse = estimate_spherical_range(
            target_erp,
            reference_erp,
            inverse_rotation,
            inverse_translation,
            options=one_way,
        )
        return _apply_bidirectional_consistency(
            forward, reverse, rotation, translation, settings
        )
    first = _prepare_image(reference_erp, "reference_erp")
    second = _prepare_image(target_erp, "target_erp")
    if first.shape != second.shape:
        raise ValueError("reference_erp and target_erp must have the same HW shape")
    rotation, translation = _validate_pose(rotation_b_from_a, translation_b_from_a)
    if np.linalg.norm(translation) <= 1e-9:
        raise ValueError("translation_b_from_a must have non-zero metric scale")
    if settings.pyramid_levels > 1:
        return _estimate_adaptive_prepared(
            first,
            second,
            rotation,
            translation,
            settings,
        )

    height, width = first.shape
    rays = _erp_ray_lattice((height, width))
    first_features, texture = _matching_features(
        first, settings.window_size, settings.filter_backend
    )
    second_features, _ = _matching_features(
        second, settings.window_size, settings.filter_backend
    )
    inverse_ranges = np.linspace(
        1.0 / settings.max_range,
        1.0 / settings.min_range,
        settings.num_hypotheses,
        dtype=np.float32,
    )
    volume_hwd = np.empty((height, width, settings.num_hypotheses), dtype=np.float32)

    for index, inverse_range in enumerate(inverse_ranges):
        radial_range = 1.0 / float(inverse_range)
        points_a = rays * radial_range
        points_b = points_a @ rotation.T + translation
        projected = rays_to_erp_pixels(points_b, (height, width))
        warped, support = _seam_safe_remap(second_features, projected.pixels_xy)
        cost = _feature_cost(first_features, warped, settings)
        volume_hwd[..., index] = np.where(support, cost, 1.0)

    # All hypotheses share the same spherical sampling map. Treating D as the
    # channel dimension lets the native kernel reuse that geometry instead of
    # rebuilding it once per hypothesis.
    filtered_hwd = _spherical_box_filter(
        volume_hwd, settings.window_size, settings.filter_backend
    )
    del volume_hwd
    volume = np.moveaxis(filtered_hwd, -1, 0)

    aggregated = _aggregate_cost_volume(volume, first, settings)
    del volume, filtered_hwd
    indices, best, second_best = _best_two_costs(aggregated)
    confidence = np.clip(
        (second_best - best) / np.maximum(second_best, _EPS), 0.0, 1.0
    ).astype(np.float32)

    refined_inverse = _refine_inverse_range(aggregated, indices, inverse_ranges)
    range_map = np.reciprocal(refined_inverse).astype(np.float32)
    pole_margin = int(math.ceil(height * settings.pole_margin_fraction))
    latitude_support = np.ones((height, width), dtype=bool)
    if pole_margin:
        latitude_support[:pole_margin] = False
        latitude_support[-pole_margin:] = False
    interior_hypothesis = (indices > 0) & (indices < settings.num_hypotheses - 1)
    validity = (
        latitude_support
        & interior_hypothesis
        & np.isfinite(range_map)
        & (texture >= settings.min_texture_std)
        & (confidence >= settings.min_confidence)
        & (best <= settings.max_matching_cost)
    )
    range_map = np.where(validity, range_map, np.nan).astype(np.float32)
    return SphericalStereoResult(
        range=range_map,
        validity_mask=validity,
        confidence=confidence,
        matching_cost=best.astype(np.float32),
        hypothesis_index=indices,
        inverse_range_hypotheses=inverse_ranges,
        options=settings,
    )


def _estimate_adaptive_prepared(
    first: np.ndarray,
    second: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    settings: SphericalStereoOptions,
) -> SphericalStereoResult:
    """Run a wide coarse sweep followed by local inverse-range refinements."""
    height, width = first.shape
    maximum_levels = 1
    while height // (2**maximum_levels) >= 8 and width // (2**maximum_levels) >= 16:
        maximum_levels += 1
    levels = min(settings.pyramid_levels, maximum_levels)
    scale = 2 ** (levels - 1)
    coarse_shape = (height // scale, width // scale)
    coarse_first = cv2.resize(
        first,
        (coarse_shape[1], coarse_shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    coarse_second = cv2.resize(
        second,
        (coarse_shape[1], coarse_shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    coarse_settings = replace(settings, pyramid_levels=1)
    result = estimate_spherical_range(
        coarse_first,
        coarse_second,
        rotation,
        translation,
        options=coarse_settings,
    )
    inverse_range = _result_inverse_range(result)
    confidence = result.confidence
    boundary = (result.hypothesis_index == 0) | (
        result.hypothesis_index == result.inverse_range_hypotheses.size - 1
    )
    global_step = float(
        result.inverse_range_hypotheses[1] - result.inverse_range_hypotheses[0]
    )
    base_radius = settings.refinement_radius_steps * global_step

    for level in range(levels - 2, -1, -1):
        level_scale = 2**level
        level_shape = (height // level_scale, width // level_scale)
        if level_scale == 1:
            level_first = first
            level_second = second
        else:
            level_first = cv2.resize(
                first,
                (level_shape[1], level_shape[0]),
                interpolation=cv2.INTER_AREA,
            )
            level_second = cv2.resize(
                second,
                (level_shape[1], level_shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        center = cv2.resize(
            inverse_range,
            (level_shape[1], level_shape[0]),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        level_confidence = cv2.resize(
            confidence,
            (level_shape[1], level_shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        level_boundary = cv2.resize(
            boundary.astype(np.uint8),
            (level_shape[1], level_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        certainty = np.clip(level_confidence / 0.25, 0.0, 1.0)
        radius = base_radius * (1.0 + 2.0 * (1.0 - certainty))
        radius[level_boundary] *= 2.0
        minimum_inverse = 1.0 / settings.max_range
        maximum_inverse = 1.0 / settings.min_range
        inverse_span = maximum_inverse - minimum_inverse
        radius = np.clip(radius, global_step, 0.5 * inverse_span).astype(np.float32)
        # Keep every normalized label distinct. Clipping each candidate would
        # duplicate boundary hypotheses and could make a clipped endpoint look
        # like a confident interior optimum.
        center = np.clip(
            center,
            minimum_inverse + radius,
            maximum_inverse - radius,
        ).astype(np.float32)
        result = _estimate_local_prepared(
            level_first,
            level_second,
            rotation,
            translation,
            center,
            radius,
            settings,
        )
        inverse_range = _result_inverse_range(result)
        confidence = result.confidence
        boundary = (result.hypothesis_index == 0) | (
            result.hypothesis_index == settings.refinement_hypotheses - 1
        )
        base_radius *= 0.5
    return result


def _estimate_local_prepared(
    first: np.ndarray,
    second: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    center: np.ndarray,
    radius: np.ndarray,
    settings: SphericalStereoOptions,
) -> SphericalStereoResult:
    """Refine a per-pixel inverse-range prior with shared normalized offsets."""
    height, width = first.shape
    rays = _erp_ray_lattice((height, width))
    first_features, texture = _matching_features(
        first, settings.window_size, settings.filter_backend
    )
    second_features, _ = _matching_features(
        second, settings.window_size, settings.filter_backend
    )
    offsets = np.linspace(
        -1.0,
        1.0,
        settings.refinement_hypotheses,
        dtype=np.float32,
    )
    minimum_inverse = 1.0 / settings.max_range
    maximum_inverse = 1.0 / settings.min_range
    volume_hwd = np.empty(
        (height, width, settings.refinement_hypotheses), dtype=np.float32
    )
    for index, offset in enumerate(offsets):
        inverse_range = np.clip(
            center + float(offset) * radius,
            minimum_inverse,
            maximum_inverse,
        )
        points_a = rays / inverse_range[..., None]
        points_b = points_a @ rotation.T + translation
        projected = rays_to_erp_pixels(points_b, (height, width))
        warped, support = _seam_safe_remap(second_features, projected.pixels_xy)
        cost = _feature_cost(first_features, warped, settings)
        volume_hwd[..., index] = np.where(support, cost, 1.0)

    filtered_hwd = _spherical_box_filter(
        volume_hwd, settings.window_size, settings.filter_backend
    )
    del volume_hwd
    volume = np.moveaxis(filtered_hwd, -1, 0)
    aggregated = _aggregate_cost_volume(volume, first, settings)
    del volume, filtered_hwd
    indices, best, second_best = _best_two_costs(aggregated)
    confidence = np.clip(
        (second_best - best) / np.maximum(second_best, _EPS), 0.0, 1.0
    ).astype(np.float32)
    refined_offset = _refine_labels(aggregated, indices, offsets)
    refined_inverse = np.clip(
        center + refined_offset * radius,
        minimum_inverse,
        maximum_inverse,
    )
    range_map = np.reciprocal(refined_inverse).astype(np.float32)
    pole_margin = int(math.ceil(height * settings.pole_margin_fraction))
    latitude_support = np.ones((height, width), dtype=bool)
    if pole_margin:
        latitude_support[:pole_margin] = False
        latitude_support[-pole_margin:] = False
    interior_hypothesis = (indices > 0) & (indices < settings.refinement_hypotheses - 1)
    validity = (
        latitude_support
        & interior_hypothesis
        & np.isfinite(range_map)
        & (texture >= settings.min_texture_std)
        & (confidence >= settings.min_confidence)
        & (best <= settings.max_matching_cost)
    )
    range_map = np.where(validity, range_map, np.nan).astype(np.float32)
    return SphericalStereoResult(
        range=range_map,
        validity_mask=validity,
        confidence=confidence,
        matching_cost=best.astype(np.float32),
        hypothesis_index=indices,
        inverse_range_hypotheses=offsets,
        options=settings,
        hypothesis_mode="normalized_inverse_range_offset",
        inverse_range_center=center,
        inverse_range_radius=radius,
    )


def _result_inverse_range(result: SphericalStereoResult) -> np.ndarray:
    """Recover the selected inverse range, including pixels later invalidated."""
    labels = result.inverse_range_hypotheses[result.hypothesis_index]
    if result.hypothesis_mode == "absolute_inverse_range":
        selected = labels
    else:
        assert result.inverse_range_center is not None
        assert result.inverse_range_radius is not None
        selected = result.inverse_range_center + labels * result.inverse_range_radius
    valid = result.validity_mask & np.isfinite(result.range) & (result.range > 0)
    return np.where(valid, 1.0 / result.range, selected).astype(np.float32)


def _positive_finite(name: str, value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be positive and finite")


def _closed_fraction(name: str, value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} must be finite and in [0, 1]")


def _nonnegative_finite(name: str, value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be non-negative and finite")


def _readonly_image(value: object, dtype: Any, name: str) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape HW")
    array.setflags(write=False)
    return array


def _prepare_image(value: object, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if value.ndim == 2:
        image = value
    elif value.ndim == 3 and value.shape[2] in {1, 3, 4}:
        if value.shape[2] == 1:
            image = value[..., 0]
        else:
            rgb = value[..., :3].astype(np.float32)
            image = rgb @ np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
    else:
        raise ValueError(f"{name} must use HW, HW1, HWC-RGB, or HWC-RGBA layout")
    image = np.asarray(image, dtype=np.float32)
    if not np.all(np.isfinite(image)):
        raise ValueError(f"{name} must contain only finite values")
    if np.issubdtype(value.dtype, np.integer):
        maximum = float(np.iinfo(value.dtype).max)
        image = image / maximum
    elif image.size and (float(image.min()) < 0.0 or float(image.max()) > 1.0):
        raise ValueError(f"floating {name} values must be in [0, 1]")
    if image.shape[0] < 8 or image.shape[1] < 16:
        raise ValueError(f"{name} must be at least 8x16")
    return np.ascontiguousarray(image, dtype=np.float32)


def _validate_pose(
    rotation: object, translation: object
) -> tuple[np.ndarray, np.ndarray]:
    R = np.asarray(rotation, dtype=np.float64)
    t = np.asarray(translation, dtype=np.float64)
    if R.shape != (3, 3) or t.shape != (3,):
        raise ValueError("rotation must be 3x3 and translation must have shape (3,)")
    if not np.all(np.isfinite(R)) or not np.all(np.isfinite(t)):
        raise ValueError("pose must contain only finite values")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6):
        raise ValueError("rotation must be orthonormal")
    if not np.isclose(np.linalg.det(R), 1.0, atol=1e-6):
        raise ValueError("rotation must have determinant +1")
    return R, t


def _erp_ray_lattice(shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    y, x = np.indices((height, width), dtype=np.float32)
    pixels = np.stack((x, y), axis=-1)
    return np.asarray(erp_pixels_to_rays(pixels, shape_hw), dtype=np.float32)


def _spherical_box_filter(
    image: np.ndarray, window_size: int, backend: str
) -> np.ndarray:
    return np.asarray(
        spherical_box_blur(
            image.astype(np.float32, copy=False),
            ksize=window_size,
            backend=backend,
        ),
        dtype=np.float32,
    )


def _matching_features(
    image: np.ndarray, window_size: int, backend: str
) -> tuple[np.ndarray, np.ndarray]:
    local_mean = _spherical_box_filter(image, window_size, backend)
    local_square_mean = _spherical_box_filter(image * image, window_size, backend)
    variance = np.maximum(local_square_mean - local_mean * local_mean, 0.0)
    local_std = np.sqrt(variance, dtype=np.float32)
    normalized = np.clip((image - local_mean) / np.maximum(local_std, 0.02), -3.0, 3.0)
    gradient = spherical_gradient(image, backend=backend)
    gradient_x = np.asarray(gradient.east, dtype=np.float32)
    gradient_y = np.asarray(gradient.north, dtype=np.float32)
    gradient_scale = np.percentile(
        np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y), 90
    )
    scale = max(float(gradient_scale), 0.05)
    features = np.stack(
        (
            normalized / 3.0,
            np.clip(gradient_x / scale, -2.0, 2.0) / 2.0,
            np.clip(gradient_y / scale, -2.0, 2.0) / 2.0,
        ),
        axis=-1,
    ).astype(np.float32)
    return features, local_std.astype(np.float32)


def _seam_safe_remap(
    image: np.ndarray, pixels_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    padded = np.concatenate((image[:, -1:], image, image[:, :1]), axis=1)
    map_x = np.asarray(pixels_xy[..., 0], dtype=np.float32) + 1.0
    map_y = np.asarray(pixels_xy[..., 1], dtype=np.float32)
    support = (
        np.isfinite(map_x)
        & np.isfinite(map_y)
        & (map_y >= 0.0)
        & (map_y <= height - 1.0)
    )
    safe_x = np.where(support, map_x, 0.0).astype(np.float32)
    safe_y = np.where(support, map_y, 0.0).astype(np.float32)
    warped = cv2.remap(
        padded,
        safe_x,
        safe_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if warped.ndim == 2:
        warped = warped[..., None]
    return warped, support


def _sample_wrapped_nearest(image: np.ndarray, pixels_xy: np.ndarray) -> np.ndarray:
    height, _ = image.shape
    padded = np.concatenate((image[:, -1:], image, image[:, :1]), axis=1)
    map_x = np.asarray(pixels_xy[..., 0], dtype=np.float32) + 1.0
    map_y = np.asarray(pixels_xy[..., 1], dtype=np.float32)
    support = (
        np.isfinite(map_x)
        & np.isfinite(map_y)
        & (map_y >= 0.0)
        & (map_y <= height - 1.0)
    )
    sampled = cv2.remap(
        padded,
        np.where(support, map_x, 0.0).astype(np.float32),
        np.where(support, map_y, 0.0).astype(np.float32),
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return np.where(support, sampled, np.nan).astype(np.float32)


def _apply_bidirectional_consistency(
    forward: SphericalStereoResult,
    reverse: SphericalStereoResult,
    rotation_b_from_a: np.ndarray,
    translation_b_from_a: np.ndarray,
    options: SphericalStereoOptions,
) -> SphericalStereoResult:
    shape_hw = forward.range.shape
    rays = _erp_ray_lattice(shape_hw)
    points_a = rays * forward.range[..., None]
    points_b = points_a @ rotation_b_from_a.T + translation_b_from_a
    projected = rays_to_erp_pixels(points_b, shape_hw)
    reverse_range = _sample_wrapped_nearest(reverse.range, projected.pixels_xy)
    reverse_confidence = _sample_wrapped_nearest(
        reverse.confidence, projected.pixels_xy
    )
    tolerance = (
        options.consistency_absolute_tolerance
        + options.consistency_relative_tolerance * reverse_range
    )
    consistent = (
        forward.validity_mask
        & np.isfinite(reverse_range)
        & (np.abs(projected.ranges - reverse_range) <= tolerance)
    )
    confidence = np.minimum(forward.confidence, reverse_confidence)
    confidence = np.where(np.isfinite(confidence), confidence, 0.0).astype(np.float32)
    return SphericalStereoResult(
        range=np.where(consistent, forward.range, np.nan).astype(np.float32),
        validity_mask=consistent,
        confidence=confidence,
        matching_cost=forward.matching_cost,
        hypothesis_index=forward.hypothesis_index,
        inverse_range_hypotheses=forward.inverse_range_hypotheses,
        options=options,
        hypothesis_mode=forward.hypothesis_mode,
        inverse_range_center=forward.inverse_range_center,
        inverse_range_radius=forward.inverse_range_radius,
    )


def _feature_cost(
    first: np.ndarray,
    second: np.ndarray,
    options: SphericalStereoOptions,
) -> np.ndarray:
    intensity = np.minimum(np.abs(first[..., 0] - second[..., 0]), 1.0)
    gradient = np.minimum(
        0.5
        * (
            np.abs(first[..., 1] - second[..., 1])
            + np.abs(first[..., 2] - second[..., 2])
        ),
        1.0,
    )
    return (
        options.intensity_weight * intensity + options.gradient_weight * gradient
    ).astype(np.float32)


def _aggregate_cost_volume(
    volume: np.ndarray,
    guidance: np.ndarray,
    options: SphericalStereoOptions,
) -> np.ndarray:
    """Aggregate inverse-range costs along four edge-aware ERP paths."""

    if options.smoothness_penalty_large == 0.0:
        return volume
    aggregated = np.zeros_like(volume)
    for axis, reverse in ((1, False), (1, True), (0, False), (0, True)):
        _accumulate_path(
            aggregated,
            volume,
            guidance,
            axis=axis,
            reverse=reverse,
            penalty_small=options.smoothness_penalty_small,
            penalty_large=options.smoothness_penalty_large,
        )
    return aggregated * 0.25


def _accumulate_path(
    output: np.ndarray,
    volume: np.ndarray,
    guidance: np.ndarray,
    *,
    axis: int,
    reverse: bool,
    penalty_small: float,
    penalty_large: float,
) -> None:
    depth_count, height, width = volume.shape
    length = width if axis == 1 else height
    order = range(length - 1, -1, -1) if reverse else range(length)
    previous: np.ndarray | None = None
    previous_position: int | None = None
    for position in order:
        if axis == 1:
            local = volume[:, :, position].T
        else:
            local = volume[:, position, :].T
        if previous is None:
            current = local.copy()
        else:
            minimum = previous.min(axis=1)
            same = previous
            lower = np.empty_like(previous)
            lower[:, 0] = np.inf
            lower[:, 1:] = previous[:, :-1] + penalty_small
            upper = np.empty_like(previous)
            upper[:, -1] = np.inf
            upper[:, :-1] = previous[:, 1:] + penalty_small
            if axis == 1:
                edge = np.abs(guidance[:, position] - guidance[:, previous_position])
            else:
                edge = np.abs(guidance[position, :] - guidance[previous_position, :])
            adaptive_large = np.maximum(
                penalty_small, penalty_large / (1.0 + 8.0 * edge)
            )
            jump = minimum[:, None] + adaptive_large[:, None]
            current = local + np.minimum(
                np.minimum(same, lower), np.minimum(upper, jump)
            )
            current -= minimum[:, None]
        if axis == 1:
            output[:, :, position] += current.T
        else:
            output[:, position, :] += current.T
        previous = current
        previous_position = position


def _refine_inverse_range(
    volume: np.ndarray, indices: np.ndarray, hypotheses: np.ndarray
) -> np.ndarray:
    return _refine_labels(volume, indices, hypotheses)


def _refine_labels(
    volume: np.ndarray, indices: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    depth_count = volume.shape[0]
    center = np.take_along_axis(volume, indices[None, ...], axis=0)[0]
    previous_index = np.maximum(indices - 1, 0)
    next_index = np.minimum(indices + 1, depth_count - 1)
    previous = np.take_along_axis(volume, previous_index[None, ...], axis=0)[0]
    following = np.take_along_axis(volume, next_index[None, ...], axis=0)[0]
    denominator = previous - 2.0 * center + following
    delta = np.zeros_like(center, dtype=np.float32)
    conditioned = (indices > 0) & (indices < depth_count - 1) & (denominator > 1e-6)
    delta[conditioned] = np.clip(
        0.5
        * (previous[conditioned] - following[conditioned])
        / denominator[conditioned],
        -0.5,
        0.5,
    )
    step = float(labels[1] - labels[0])
    return labels[indices] + delta * step


def _best_two_costs(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return winner indices and the two smallest costs without copying D*H*W."""
    _, height, width = volume.shape
    best = np.full((height, width), np.inf, dtype=np.float32)
    second = np.full((height, width), np.inf, dtype=np.float32)
    indices = np.zeros((height, width), dtype=np.int32)
    for index, cost in enumerate(volume):
        winner = cost < best
        np.minimum(second, cost, out=second, where=~winner)
        second[winner] = best[winner]
        best[winner] = cost[winner]
        indices[winner] = index
    return indices, best, second
