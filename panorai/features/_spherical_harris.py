"""Experimental Harris corner detection in local spherical tangent frames."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from typing import Any

import numpy as np

from panorai.geometry import erp_pixels_to_rays
from panorai.image_processing import (
    spherical_box_blur,
    spherical_gaussian_blur,
    spherical_sobel,
)
from panorai.image_processing._sampling import (
    row_chunks,
    sample_tangent,
    validate_step,
)

from ._extractor import (
    _array_checksum,
    _as_panorama,
    _panorai_commit,
    _panorai_version,
)
from ._geometry import deduplicate_spherical_keypoints
from ._spherical_detector import SphericalKeypoint

SPHERICAL_HARRIS_INTERFACE = "panorai-spherical-harris-detector/v1"


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_real(
    value: float,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


@dataclass(frozen=True, slots=True)
class SphericalHarrisConfig:
    """Configuration for spherical Harris response and keypoint selection.

    Gradients and structure-tensor smoothing are evaluated in the local
    east/north tangent frame at every ERP pixel. ``angular_step_deg=None``
    uses one source-row angular sample (``180 / height`` degrees).
    """

    block_size: int = 5
    block_sigma_px: float = 1.2
    gradient_operator: str = "sobel"
    harris_k: float = 0.04
    threshold_rel: float = 0.01
    threshold_abs: float | None = None
    nms_radius_px: int = 3
    max_keypoints: int = 1000
    minimum_valid_support_fraction: float = 0.99
    selection_policy: str = "equal-area-round-robin"
    selection_grid_shape: tuple[int, int] = (12, 24)
    angular_step_deg: float | None = None
    backend: str = "auto"

    def __post_init__(self) -> None:
        block_size = _positive_integer(self.block_size, "block_size")
        if block_size < 3 or block_size % 2 == 0:
            raise ValueError("block_size must be an odd integer of at least 3")
        object.__setattr__(self, "block_size", block_size)
        object.__setattr__(
            self,
            "block_sigma_px",
            _finite_real(self.block_sigma_px, "block_sigma_px", minimum=1e-12),
        )
        operator = str(self.gradient_operator).strip().lower()
        if operator not in {"sobel", "scharr"}:
            raise ValueError("gradient_operator must be 'sobel' or 'scharr'")
        object.__setattr__(self, "gradient_operator", operator)
        harris_k = _finite_real(self.harris_k, "harris_k", minimum=0.0)
        if not 0.0 < harris_k < 0.25:
            raise ValueError("harris_k must be in the open interval (0, 0.25)")
        object.__setattr__(self, "harris_k", harris_k)
        object.__setattr__(
            self,
            "threshold_rel",
            _finite_real(self.threshold_rel, "threshold_rel", minimum=0.0, maximum=1.0),
        )
        if self.threshold_abs is not None:
            object.__setattr__(
                self,
                "threshold_abs",
                _finite_real(self.threshold_abs, "threshold_abs", minimum=0.0),
            )
        object.__setattr__(
            self,
            "nms_radius_px",
            _positive_integer(self.nms_radius_px, "nms_radius_px"),
        )
        object.__setattr__(
            self,
            "max_keypoints",
            _positive_integer(self.max_keypoints, "max_keypoints"),
        )
        object.__setattr__(
            self,
            "minimum_valid_support_fraction",
            _finite_real(
                self.minimum_valid_support_fraction,
                "minimum_valid_support_fraction",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        policy = str(self.selection_policy).strip().lower()
        if policy not in {"response", "equal-area-round-robin"}:
            raise ValueError(
                "selection_policy must be 'response' or 'equal-area-round-robin'"
            )
        object.__setattr__(self, "selection_policy", policy)
        if (
            not isinstance(self.selection_grid_shape, tuple)
            or len(self.selection_grid_shape) != 2
        ):
            raise TypeError("selection_grid_shape must be a two-integer tuple")
        object.__setattr__(
            self,
            "selection_grid_shape",
            tuple(
                _positive_integer(item, "selection_grid_shape item")
                for item in self.selection_grid_shape
            ),
        )
        if self.angular_step_deg is not None:
            step = _finite_real(
                self.angular_step_deg,
                "angular_step_deg",
                minimum=np.finfo(float).tiny,
            )
            if step >= 180.0:
                raise ValueError("angular_step_deg must be smaller than 180")
            object.__setattr__(self, "angular_step_deg", step)
        backend = str(self.backend).strip().lower()
        if backend not in {"auto", "numpy", "native"}:
            raise ValueError("backend must be 'auto', 'numpy', or 'native'")
        object.__setattr__(self, "backend", backend)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            {
                "interface": SPHERICAL_HARRIS_INTERFACE,
                "stability": "experimental",
                "response": "det(M)-harris_k*trace(M)^2",
                "gradient_frame": "local-east-north-tangent",
                "coordinate_convention": "ERP pixel centers, top-left origin",
                "scale_units": "degrees",
                "descriptor": None,
            }
        )
        return result


@dataclass(frozen=True, slots=True)
class SphericalHarrisResponse:
    """Harris value, geometric/data support, and validity on the source ERP."""

    response: np.ndarray
    support_fraction: np.ndarray
    validity_mask: np.ndarray
    interface: str = SPHERICAL_HARRIS_INTERFACE
    stability: str = "experimental"

    def __post_init__(self) -> None:
        response = np.asarray(self.response)
        support = np.asarray(self.support_fraction)
        validity = np.asarray(self.validity_mask)
        if response.ndim != 2:
            raise ValueError("response must use HW layout")
        if support.shape != response.shape or validity.shape != response.shape:
            raise ValueError("response, support_fraction and validity_mask must match")
        if not np.issubdtype(response.dtype, np.floating):
            raise TypeError("response must use floating dtype")
        if support.dtype != np.float32:
            raise TypeError("support_fraction must use float32 dtype")
        if validity.dtype != np.bool_:
            raise TypeError("validity_mask must use boolean dtype")
        if not np.isfinite(support).all() or np.any((support < 0) | (support > 1)):
            raise ValueError("support_fraction must be finite and in [0, 1]")
        if not np.isfinite(response[validity]).all():
            raise ValueError("response must be finite wherever valid")
        if not np.isnan(response[~validity]).all():
            raise ValueError("response must be NaN wherever invalid")


@dataclass(frozen=True, slots=True)
class SphericalHarrisDiagnostics:
    """Candidate counts through thresholding and spherical NMS."""

    positive_response_count: int
    threshold_candidate_count: int
    local_maximum_count: int
    unique_after_angular_nms: int
    output_after_budget: int
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SphericalHarrisKeypointSet:
    """Descriptor-free Harris corners with spherical coordinates and provenance."""

    panorama_id: str
    keypoints: tuple[SphericalKeypoint, ...]
    source_shape_hw: tuple[int, int]
    source_checksum: str
    config: SphericalHarrisConfig
    generating_commit: str | None
    projection_backend_version: str
    diagnostics: SphericalHarrisDiagnostics
    interface: str = SPHERICAL_HARRIS_INTERFACE
    stability: str = "experimental"

    def __len__(self) -> int:
        return len(self.keypoints)

    @property
    def bearings(self) -> np.ndarray:
        return (
            np.stack([item.bearing_xyz for item in self.keypoints])
            if self.keypoints
            else np.empty((0, 3), dtype=np.float64)
        )

    @property
    def source_erp_xy(self) -> np.ndarray:
        return (
            np.stack([item.source_erp_xy for item in self.keypoints])
            if self.keypoints
            else np.empty((0, 2), dtype=np.float64)
        )

    @property
    def responses(self) -> np.ndarray:
        return np.asarray([item.response for item in self.keypoints], dtype=np.float64)

    @property
    def scales_deg(self) -> np.ndarray:
        return np.asarray([item.scale_deg for item in self.keypoints], dtype=np.float64)

    @property
    def valid_support_fractions(self) -> np.ndarray:
        return np.asarray(
            [item.valid_support_fraction for item in self.keypoints], dtype=np.float64
        )

    def describe(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "stability": self.stability,
            "panorama_id": self.panorama_id,
            "keypoint_count": len(self),
            "source_shape_hw": self.source_shape_hw,
            "source_checksum": self.source_checksum,
            "config": self.config.to_dict(),
            "generating_commit": self.generating_commit,
            "projection_backend_version": self.projection_backend_version,
            "diagnostics": self.diagnostics.to_dict(),
        }


def _grayscale(image: np.ndarray, validity: np.ndarray) -> np.ndarray:
    if image.ndim not in {2, 3}:
        raise ValueError("panorama must use HW or HWC layout")
    if image.ndim == 3 and image.shape[2] not in {1, 3, 4}:
        raise ValueError("HWC panorama must have 1, 3, or 4 channels")
    if not (
        np.issubdtype(image.dtype, np.integer)
        or np.issubdtype(image.dtype, np.floating)
    ) or np.iscomplexobj(image):
        raise TypeError("panorama must use a real integer or floating dtype")
    values = image.astype(np.float32, copy=False)
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        scale = float(info.max) if info.max > 0 else 1.0
        values = values / scale
    else:
        active = values[validity] if values.ndim == 2 else values[validity, :]
        if active.size and not np.isfinite(active).all():
            raise ValueError("floating panorama must be finite inside valid support")
        if active.size:
            minimum = float(active.min())
            maximum = float(active.max())
            if minimum < 0.0 or maximum > 255.0:
                raise ValueError(
                    "floating panorama values must be in [0, 1] or [0, 255]"
                )
            if maximum > 1.0:
                values = values / 255.0
    if values.ndim == 3:
        if values.shape[2] == 1:
            values = values[..., 0]
        else:
            values = (
                values[..., 0] * np.float32(0.299)
                + values[..., 1] * np.float32(0.587)
                + values[..., 2] * np.float32(0.114)
            )
    result = np.asarray(values, dtype=np.float32).copy()
    result[~validity] = 0.0
    return result


def _validity_for_panorama(
    panorama: Any, validity_mask: np.ndarray | None
) -> tuple[Any, np.ndarray, np.ndarray]:
    wrapped = _as_panorama(panorama)
    if not isinstance(wrapped.image, np.ndarray):
        raise TypeError("spherical Harris detection currently requires NumPy input")
    image = wrapped.image
    if image.ndim not in {2, 3} or image.shape[0] < 8 or image.shape[1] < 16:
        raise ValueError("panorama must use HW/HWC layout and be at least 8x16")
    validity = np.asarray(wrapped.validity("image"), dtype=bool)
    if validity.shape != image.shape[:2]:
        raise ValueError("panorama validity must match its spatial shape")
    if wrapped.support_mask is not None:
        support = np.asarray(wrapped.support_mask)
        if support.dtype != np.bool_ or support.shape != image.shape[:2]:
            raise TypeError("panorama support_mask must be a matching boolean array")
        validity &= support
    if validity_mask is not None:
        if not isinstance(validity_mask, np.ndarray) or validity_mask.dtype != np.bool_:
            raise TypeError("validity_mask must be a boolean NumPy array")
        if validity_mask.shape != image.shape[:2]:
            raise ValueError(
                f"validity_mask must have shape {image.shape[:2]}; got {validity_mask.shape}"
            )
        validity &= validity_mask
    return wrapped, image, validity


def spherical_harris_response(
    panorama: Any,
    config: SphericalHarrisConfig | None = None,
    *,
    validity_mask: np.ndarray | None = None,
) -> SphericalHarrisResponse:
    """Compute a support-aware Harris response on an ERP panorama.

    Image intensities are converted to ``[0, 1]`` grayscale. Invalid samples
    never become valid numeric zero: zero-fill is confined to intermediate
    filtering, and the returned response is NaN outside explicit validity.
    """

    resolved = config or SphericalHarrisConfig()
    _, image, validity = _validity_for_panorama(panorama, validity_mask)
    gray = _grayscale(image, validity)
    options = {
        "angular_step_deg": resolved.angular_step_deg,
        "backend": resolved.backend,
    }
    east = spherical_sobel(
        gray,
        direction="east",
        operator=resolved.gradient_operator,
        normalize_by_angle=False,
        **options,
    ).astype(np.float32, copy=False)
    north = spherical_sobel(
        gray,
        direction="north",
        operator=resolved.gradient_operator,
        normalize_by_angle=False,
        **options,
    ).astype(np.float32, copy=False)

    gradient_support = spherical_box_blur(
        validity.astype(np.float32), 3, normalize=True, **options
    )
    gradient_valid = gradient_support >= (1.0 - 1e-6)
    east = np.where(gradient_valid, east, 0.0)
    north = np.where(gradient_valid, north, 0.0)
    tensor_support = gradient_valid.astype(np.float32)
    denominator = spherical_gaussian_blur(
        tensor_support,
        resolved.block_size,
        resolved.block_sigma_px,
        **options,
    ).astype(np.float64, copy=False)

    def moment(values: np.ndarray) -> np.ndarray:
        numerator = spherical_gaussian_blur(
            values.astype(np.float32, copy=False) * tensor_support,
            resolved.block_size,
            resolved.block_sigma_px,
            **options,
        ).astype(np.float64, copy=False)
        output = np.zeros_like(numerator)
        np.divide(numerator, denominator, out=output, where=denominator > 1e-12)
        return output

    m_ee = moment(east * east)
    m_en = moment(east * north)
    m_nn = moment(north * north)
    support_fraction = spherical_box_blur(
        tensor_support,
        resolved.block_size,
        normalize=True,
        **options,
    ).astype(np.float32, copy=False)
    support_fraction = np.clip(support_fraction, 0.0, 1.0).astype(
        np.float32, copy=False
    )
    output_validity = (
        validity
        & (denominator > 1e-12)
        & (support_fraction + 1e-6 >= resolved.minimum_valid_support_fraction)
    )
    trace = m_ee + m_nn
    response = m_ee * m_nn - m_en * m_en - resolved.harris_k * trace * trace
    response = response.astype(np.float32)
    response[~output_validity] = np.nan
    return SphericalHarrisResponse(
        response=response,
        support_fraction=support_fraction,
        validity_mask=output_validity.astype(bool, copy=False),
    )


def _local_maxima(
    response: np.ndarray,
    candidates: np.ndarray,
    radius_px: int,
    angular_step_rad: float,
) -> np.ndarray:
    source = np.where(np.isfinite(response), response, -np.inf)
    maxima = candidates.copy()
    height, width = response.shape
    offsets = [
        (dx, dy)
        for dy in range(-radius_px, radius_px + 1)
        for dx in range(-radius_px, radius_px + 1)
        if (dx or dy) and dx * dx + dy * dy <= radius_px * radius_px
    ]
    for rows in row_chunks(height, width):
        keep = maxima[rows].copy()
        center = source[rows]
        for dx, dy in offsets:
            neighbour = sample_tangent(
                source,
                rows,
                dx * angular_step_rad,
                dy * angular_step_rad,
                interpolation="nearest",
            )
            keep &= center >= neighbour
        maxima[rows] = keep
    return maxima


def _response_order(xy: np.ndarray, responses: np.ndarray) -> np.ndarray:
    return np.lexsort((xy[:, 0], xy[:, 1], -responses))


def _select_budget(
    xy: np.ndarray,
    bearings: np.ndarray,
    responses: np.ndarray,
    config: SphericalHarrisConfig,
) -> np.ndarray:
    count = len(xy)
    if count <= config.max_keypoints:
        return _response_order(xy, responses)
    if config.selection_policy == "response":
        return _response_order(xy, responses)[: config.max_keypoints]

    latitude_bins, longitude_bins = config.selection_grid_shape
    longitude = np.arctan2(bearings[:, 0], bearings[:, 2])
    longitude_index = (
        np.floor((longitude + np.pi) / (2.0 * np.pi) * longitude_bins).astype(np.int64)
        % longitude_bins
    )
    equal_area_index = np.clip(
        np.floor((bearings[:, 1] + 1.0) * 0.5 * latitude_bins).astype(np.int64),
        0,
        latitude_bins - 1,
    )
    cell = equal_area_index * longitude_bins + longitude_index
    ordered = _response_order(xy, responses)
    queues = [
        ordered[cell[ordered] == index].tolist()
        for index in range(latitude_bins * longitude_bins)
    ]
    selected: list[int] = []
    depth = 0
    while len(selected) < config.max_keypoints:
        added = False
        for queue in queues:
            if depth < len(queue):
                selected.append(queue[depth])
                added = True
                if len(selected) == config.max_keypoints:
                    break
        if not added:
            break
        depth += 1
    return np.asarray(selected, dtype=np.int64)


class SphericalHarrisDetector:
    """Detect Harris corners using local east/north spherical derivatives."""

    def __init__(self, config: SphericalHarrisConfig | None = None) -> None:
        self.config = config or SphericalHarrisConfig()

    def response(
        self, panorama: Any, *, validity_mask: np.ndarray | None = None
    ) -> SphericalHarrisResponse:
        return spherical_harris_response(
            panorama, self.config, validity_mask=validity_mask
        )

    def detect(
        self,
        panorama: Any,
        *,
        panorama_id: str | None = None,
        validity_mask: np.ndarray | None = None,
    ) -> SphericalHarrisKeypointSet:
        _, image, _ = _validity_for_panorama(panorama, validity_mask)
        response_map = self.response(panorama, validity_mask=validity_mask)
        response = response_map.response
        # Products and tangent interpolation are evaluated in float32. Values
        # below eps**2 are indistinguishable from round-off in the quadratic
        # Harris response and must not turn an ideal one-dimensional edge into
        # a corner merely because the relative threshold scales to that noise.
        numeric_floor = float(np.finfo(np.float32).eps ** 2)
        positive = response_map.validity_mask & (response > numeric_floor)
        positive_count = int(np.count_nonzero(positive))
        maximum = float(np.max(response[positive])) if positive_count else 0.0
        relative = self.config.threshold_rel * maximum
        threshold = max(relative, self.config.threshold_abs or 0.0, numeric_floor)
        candidates = positive & (response >= threshold)
        threshold_count = int(np.count_nonzero(candidates))
        step = validate_step(self.config.angular_step_deg, image.shape[0])
        maxima = _local_maxima(response, candidates, self.config.nms_radius_px, step)
        y, x = np.nonzero(maxima)
        xy = np.column_stack((x, y)).astype(np.float64)
        scores = response[y, x].astype(np.float64)
        bearings = np.asarray(erp_pixels_to_rays(xy, image.shape[:2]), dtype=np.float64)
        if len(xy):
            deduplicated = deduplicate_spherical_keypoints(
                bearings,
                scores,
                angular_threshold_rad=self.config.nms_radius_px * step,
            )
            unique = deduplicated.selected_indices
            xy = xy[unique]
            scores = scores[unique]
            bearings = bearings[unique]
            support = response_map.support_fraction[y, x][unique].astype(np.float64)
        else:
            support = np.empty(0, dtype=np.float64)
        unique_count = len(xy)
        budget = _select_budget(xy, bearings, scores, self.config)
        xy = xy[budget]
        scores = scores[budget]
        bearings = bearings[budget]
        support = support[budget]
        scale_deg = self.config.block_sigma_px * math.degrees(step)
        keypoints = tuple(
            SphericalKeypoint(
                source_erp_xy=pixel,
                bearing_xyz=bearing,
                response=float(score),
                scale_deg=scale_deg,
                octave=0,
                level=0,
                valid_support_fraction=float(valid_fraction),
            )
            for pixel, bearing, score, valid_fraction in zip(
                xy, bearings, scores, support
            )
        )
        checksum = _array_checksum(image)
        return SphericalHarrisKeypointSet(
            panorama_id=panorama_id or f"panorama-{checksum[:16]}",
            keypoints=keypoints,
            source_shape_hw=tuple(image.shape[:2]),
            source_checksum=checksum,
            config=self.config,
            generating_commit=_panorai_commit(),
            projection_backend_version=_panorai_version(),
            diagnostics=SphericalHarrisDiagnostics(
                positive_response_count=positive_count,
                threshold_candidate_count=threshold_count,
                local_maximum_count=int(np.count_nonzero(maxima)),
                unique_after_angular_nms=unique_count,
                output_after_budget=len(keypoints),
                threshold=threshold,
            ),
        )


__all__ = [
    "SPHERICAL_HARRIS_INTERFACE",
    "SphericalHarrisConfig",
    "SphericalHarrisDetector",
    "SphericalHarrisDiagnostics",
    "SphericalHarrisKeypointSet",
    "SphericalHarrisResponse",
    "spherical_harris_response",
]
