"""Immutable data contracts for classical rectified stereo pipelines."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


def _readonly(value: Any, dtype: Any, name: str, ndim: int) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    array.setflags(write=False)
    return array


def _boolean_mask(value: Any | None, shape: tuple[int, int], name: str) -> np.ndarray:
    if value is None:
        result = np.ones(shape, dtype=bool)
    else:
        result = np.array(value, copy=True)
        if result.dtype != np.bool_:
            raise TypeError(f"{name} must be boolean")
        if result.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    result.setflags(write=False)
    return result


def _gray_float(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if value.ndim == 2:
        gray = value
    elif value.ndim == 3 and value.shape[2] in {1, 3, 4}:
        if value.shape[2] == 1:
            gray = value[..., 0]
        else:
            rgb = value[..., :3].astype(np.float32)
            gray = rgb @ np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
    else:
        raise ValueError(f"{name} must use HW, HW1, HWC-RGB, or HWC-RGBA layout")
    gray = np.asarray(gray, dtype=np.float32)
    if np.issubdtype(value.dtype, np.integer):
        gray = gray / float(np.iinfo(value.dtype).max)
    elif gray.size and (float(gray.min()) < 0.0 or float(gray.max()) > 1.0):
        raise ValueError(f"floating {name} values must be in [0, 1]")
    if not np.all(np.isfinite(gray)):
        raise ValueError(f"{name} must contain only finite values")
    result = np.ascontiguousarray(gray, dtype=np.float32)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class RectifiedStereoGeometry:
    """Metric geometry for a rectified pinhole pair.

    Pixel coordinates use PanorAi's pixel-centre convention.  ``baseline`` and
    returned radial range use the same unit.  Disparity is ``x_left-x_right``.
    """

    focal_length_x: float
    baseline: float
    focal_length_y: float | None = None
    principal_point_x: float | None = None
    principal_point_y: float | None = None

    def __post_init__(self) -> None:
        for name in ("focal_length_x", "baseline"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.focal_length_y is not None and (
            not math.isfinite(float(self.focal_length_y))
            or float(self.focal_length_y) <= 0.0
        ):
            raise ValueError("focal_length_y must be positive and finite")
        for name in ("principal_point_x", "principal_point_y"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")

    def radial_range(self, disparity: np.ndarray, validity: np.ndarray) -> np.ndarray:
        """Convert rectified disparity to radial range from the left centre."""

        if disparity.ndim != 2 or validity.shape != disparity.shape:
            raise ValueError("disparity and validity must share one HW shape")
        height, width = disparity.shape
        cx = (
            0.5 * (width - 1)
            if self.principal_point_x is None
            else float(self.principal_point_x)
        )
        cy = (
            0.5 * (height - 1)
            if self.principal_point_y is None
            else float(self.principal_point_y)
        )
        fy = float(self.focal_length_y or self.focal_length_x)
        yy, xx = np.indices((height, width), dtype=np.float64)
        ray_norm = np.sqrt(
            ((xx - cx) / float(self.focal_length_x)) ** 2 + ((yy - cy) / fy) ** 2 + 1.0
        )
        safe = validity & np.isfinite(disparity) & (disparity > 0.0)
        z = np.full(disparity.shape, np.nan, dtype=np.float64)
        z[safe] = float(self.focal_length_x) * float(self.baseline) / disparity[safe]
        return np.where(safe, z * ray_norm, np.nan).astype(np.float32)


@dataclass(frozen=True, slots=True)
class RectifiedStereoPair:
    """Prepared rectified pair with explicit support and optional geometry."""

    reference: np.ndarray
    source: np.ndarray
    reference_support: np.ndarray | None = None
    source_support: np.ndarray | None = None
    geometry: RectifiedStereoGeometry | None = None

    def __post_init__(self) -> None:
        reference = _gray_float(self.reference, "reference")
        source = _gray_float(self.source, "source")
        if reference.shape != source.shape:
            raise ValueError("reference and source must have the same HW shape")
        if reference.shape[0] < 8 or reference.shape[1] < 16:
            raise ValueError("rectified stereo images must be at least 8x16")
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "source", source)
        object.__setattr__(
            self,
            "reference_support",
            _boolean_mask(self.reference_support, reference.shape, "reference_support"),
        )
        object.__setattr__(
            self,
            "source_support",
            _boolean_mask(self.source_support, source.shape, "source_support"),
        )

    @property
    def shape_hw(self) -> tuple[int, int]:
        return int(self.reference.shape[0]), int(self.reference.shape[1])


@dataclass(frozen=True, slots=True)
class StereoCostVolume:
    """Bidirectional dense matching costs for one rectified pair."""

    pair: RectifiedStereoPair
    disparities: np.ndarray
    left: np.ndarray
    right: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        disparities = _readonly(self.disparities, np.float32, "disparities", 1)
        if disparities.size < 2 or not np.all(np.diff(disparities) > 0):
            raise ValueError("disparities must be strictly increasing")
        left = _readonly(self.left, np.float32, "left cost volume", 3)
        right = _readonly(self.right, np.float32, "right cost volume", 3)
        expected = (*self.pair.shape_hw, disparities.size)
        if left.shape != expected or right.shape != expected:
            raise ValueError(f"cost volumes must have shape {expected}")
        object.__setattr__(self, "disparities", disparities)
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def replaced(
        self,
        *,
        left: np.ndarray,
        right: np.ndarray,
        metadata: Mapping[str, Any] | None = None,
    ) -> "StereoCostVolume":
        details = dict(self.metadata)
        if metadata:
            details.update(metadata)
        return StereoCostVolume(self.pair, self.disparities, left, right, details)


@dataclass(frozen=True, slots=True)
class StereoDisparityCandidate:
    """Selected left/right disparities before consistency refinement."""

    pair: RectifiedStereoPair
    left: np.ndarray
    right: np.ndarray
    left_cost: np.ndarray
    left_second_cost: np.ndarray
    backend_validity: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        shape = self.pair.shape_hw
        for name, dtype in (
            ("left", np.float32),
            ("right", np.float32),
            ("left_cost", np.float32),
            ("left_second_cost", np.float32),
            ("backend_validity", np.bool_),
        ):
            value = _readonly(getattr(self, name), dtype, name, 2)
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class ClassicalStereoResult:
    """Dense rectified disparity and optional radial range.

    Invalid disparity and range samples are ``NaN``.  Validity, occlusion and
    mismatch are explicit and are never inferred from numeric zero.
    """

    disparity: np.ndarray
    validity: np.ndarray
    confidence: np.ndarray
    matching_cost: np.ndarray
    occlusion: np.ndarray
    mismatch: np.ndarray
    radial_range: np.ndarray | None
    method: str
    diagnostics: Mapping[str, Any]
    interface: str = "panorai-classical-dense-stereo/v1-experimental"

    def __post_init__(self) -> None:
        disparity = _readonly(self.disparity, np.float32, "disparity", 2)
        shape = disparity.shape
        for name, dtype in (
            ("validity", np.bool_),
            ("confidence", np.float32),
            ("matching_cost", np.float32),
            ("occlusion", np.bool_),
            ("mismatch", np.bool_),
        ):
            value = _readonly(getattr(self, name), dtype, name, 2)
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            object.__setattr__(self, name, value)
        if self.radial_range is not None:
            radial = _readonly(self.radial_range, np.float32, "radial_range", 2)
            if radial.shape != shape:
                raise ValueError("radial_range must match disparity shape")
            if np.any(self.validity & ~np.isfinite(radial)):
                raise ValueError(
                    "radial_range must be finite wherever validity is true"
                )
            object.__setattr__(self, "radial_range", radial)
        object.__setattr__(self, "disparity", disparity)
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics))
        )

    def describe(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "stability": "experimental",
            "method": self.method,
            "shape_hw": list(self.disparity.shape),
            "valid_fraction": float(self.validity.mean()),
            "has_metric_geometry": self.radial_range is not None,
            "quantity": "disparity_px",
            "optional_quantity": "radial_range",
            "diagnostics": dict(self.diagnostics),
        }


__all__ = [
    "ClassicalStereoResult",
    "RectifiedStereoGeometry",
    "RectifiedStereoPair",
    "StereoCostVolume",
    "StereoDisparityCandidate",
]
