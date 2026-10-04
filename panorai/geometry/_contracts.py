"""Immutable public contracts for :mod:`panorai.geometry`."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any, Generic

from ._typing import ArrayT, ShapeHW


def _shape_hw(value: object, name: str) -> ShapeHW:
    if isinstance(value, bool):
        raise TypeError(f"{name} values must be integers, not booleans")
    if isinstance(value, Integral):
        value = (value, value)
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must contain exactly (height, width)")
    try:
        dimensions = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must contain exactly (height, width)") from exc
    if len(dimensions) != 2:
        raise ValueError(f"{name} must contain (height, width)")
    if any(
        isinstance(item, bool) or not isinstance(item, Integral) for item in dimensions
    ):
        raise TypeError(f"{name} values must be integers, not truncated")
    shape = (int(dimensions[0]), int(dimensions[1]))
    if shape[0] <= 0 or shape[1] <= 0:
        raise ValueError(f"{name} values must be positive")
    return shape


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _periodic_degrees(value: float) -> float:
    normalized = (value + 180.0) % 360.0 - 180.0
    return 0.0 if normalized == 0.0 else normalized


@dataclass(frozen=True, slots=True)
class GnomonicSpec:
    """Canonical description of a rectangular gnomonic view.

    Angles are expressed in degrees. ``output_shape_hw`` follows ``(H, W)``.
    """

    center_lat_deg: float = 0.0
    center_lon_deg: float = 0.0
    hfov_deg: float = 90.0
    vfov_deg: float = 90.0
    roll_deg: float = 0.0
    output_shape_hw: tuple[int, int] = (1024, 1024)

    def __post_init__(self) -> None:
        latitude = _finite_float(self.center_lat_deg, "center_lat_deg")
        longitude = _finite_float(self.center_lon_deg, "center_lon_deg")
        horizontal_fov = _finite_float(self.hfov_deg, "hfov_deg")
        vertical_fov = _finite_float(self.vfov_deg, "vfov_deg")
        roll = _finite_float(self.roll_deg, "roll_deg")
        if not -90.0 <= latitude <= 90.0:
            raise ValueError("center_lat_deg must be between -90 and 90 degrees")
        for name, value in (("hfov_deg", horizontal_fov), ("vfov_deg", vertical_fov)):
            if not 0.0 < value < 180.0:
                raise ValueError(f"{name} must be between 0 and 180 degrees")
        object.__setattr__(self, "center_lat_deg", latitude)
        object.__setattr__(self, "center_lon_deg", _periodic_degrees(longitude))
        object.__setattr__(self, "hfov_deg", horizontal_fov)
        object.__setattr__(self, "vfov_deg", vertical_fov)
        object.__setattr__(self, "roll_deg", _periodic_degrees(roll))
        object.__setattr__(
            self, "output_shape_hw", _shape_hw(self.output_shape_hw, "output_shape_hw")
        )

    @classmethod
    def from_config(cls, config: Any) -> "GnomonicSpec":
        """Convert a legacy ``GnomonicConfig`` without requiring new fields."""

        fov = float(config.fov_deg)
        hfov = getattr(config, "hfov_deg", None)
        vfov = getattr(config, "vfov_deg", None)
        roll = getattr(config, "roll_deg", 0.0)
        return cls(
            center_lat_deg=float(config.phi1_deg),
            center_lon_deg=float(config.lam0_deg),
            hfov_deg=fov if hfov is None else float(hfov),
            vfov_deg=fov if vfov is None else float(vfov),
            roll_deg=float(roll),
            output_shape_hw=(int(config.y_points), int(config.x_points)),
        )


@dataclass(frozen=True, slots=True)
class CubemapSpec:
    """Canonical cubemap configuration with a shared face shape."""

    face_shape_hw: tuple[int, int] = (1024, 1024)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "face_shape_hw", _shape_hw(self.face_shape_hw, "face_shape_hw")
        )


@dataclass(frozen=True, slots=True)
class ProjectionResult(Generic[ArrayT]):
    """Projected data and masks with support kept distinct from validity.

    ``validity_mask`` and ``valid_weight`` are populated by the opt-in
    validity-normalized bilinear mode. Their defaults preserve construction
    and access of the original two-field result contract.
    ``source_pixels_xy`` is an optional raster-to-source coordinate map.
    """

    data: ArrayT
    support_mask: ArrayT
    validity_mask: ArrayT | None = None
    valid_weight: ArrayT | None = None
    source_pixels_xy: ArrayT | None = None


@dataclass(frozen=True, slots=True)
class ERPPointProjection(Generic[ArrayT]):
    """Cartesian rays or points projected onto ERP pixel coordinates."""

    pixels_xy: ArrayT
    ranges: ArrayT
    valid: ArrayT


@dataclass(frozen=True, slots=True)
class GnomonicRayProjection(Generic[ArrayT]):
    """Gnomonic pixels converted to panorama-frame unit rays."""

    rays_xyz: ArrayT
    valid: ArrayT


@dataclass(frozen=True, slots=True)
class GnomonicPointProjection(Generic[ArrayT]):
    """Panorama-frame rays projected onto a gnomonic raster."""

    pixels_xy: ArrayT
    ranges: ArrayT
    valid: ArrayT


@dataclass(frozen=True, slots=True)
class GnomonicFaceGeometry(Generic[ArrayT]):
    """Explicit pinhole geometry and provenance for one gnomonic face.

    ``K`` follows the usual image convention of ``+x`` right and ``+y`` down.
    The columns of ``R_panorama_from_face`` are therefore the face-camera
    right, down, and forward axes expressed in PanorAi's panorama frame. The
    y-axis convention change makes this an orthogonal direction transform
    with determinant -1, not an SO(3) pose; relative face transforms are
    proper rotations.
    """

    face_id: str
    spec: GnomonicSpec
    K: ArrayT
    R_panorama_from_face: ArrayT
    support_mask: ArrayT
    source_pixels_xy: ArrayT | None = None
