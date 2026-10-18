"""Intrinsic mathematical morphology on canonical ERP rasters.

The structuring element is a disk in each pixel centre's tangent plane.  The
exponential map carries its samples onto the sphere, so the footprint radius
is a great-circle angle rather than an ERP pixel radius.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real
from typing import Any, ClassVar, Literal

import numpy as np


MorphologyReduction = Literal["max", "min"]
_DEFAULT_MAX_SAMPLED_ELEMENTS = 4_194_304


def _finite_positive(value: Any, *, name: str, upper: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    if upper is not None and result >= upper:
        raise ValueError(f"{name} must be smaller than {upper:g}")
    return result


@dataclass(frozen=True, slots=True)
class GeodesicDisk:
    """An isotropic spherical structuring element.

    ``radius_deg`` is intrinsic great-circle distance.  By default samples use
    the ERP's north/east angular lattice spacing.  ``angular_step_deg`` can
    request a fixed, resolution-independent sampling step.
    """

    radius_deg: float
    angular_step_deg: float | None = None

    interface: ClassVar[str] = "panorai-spherical-morphology/v1"
    stability: ClassVar[str] = "stable"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "radius_deg",
            _finite_positive(self.radius_deg, name="radius_deg", upper=180.0),
        )
        if self.angular_step_deg is not None:
            object.__setattr__(
                self,
                "angular_step_deg",
                _finite_positive(
                    self.angular_step_deg,
                    name="angular_step_deg",
                    upper=180.0,
                ),
            )

    def kernel(self, shape_hw: tuple[int, int]) -> tuple[tuple[int, int], np.ndarray]:
        """Return the odd bounding kernel and active disk samples for an ERP."""

        if len(shape_hw) != 2 or any(int(item) < 2 for item in shape_hw):
            raise ValueError("shape_hw must contain two integers of at least 2")
        height, width = (int(item) for item in shape_hw)
        radius = math.radians(self.radius_deg)
        if self.angular_step_deg is None:
            north_step = math.pi / height
            east_step = 2.0 * math.pi / width
        else:
            north_step = east_step = math.radians(self.angular_step_deg)
        north_radius = math.ceil(radius / north_step)
        east_radius = math.ceil(radius / east_step)
        north = np.arange(-north_radius, north_radius + 1, dtype=np.float64)
        east = np.arange(-east_radius, east_radius + 1, dtype=np.float64)
        north_offset, east_offset = np.meshgrid(
            north * north_step,
            east * east_step,
            indexing="ij",
        )
        tolerance = 16.0 * np.finfo(np.float64).eps * max(1.0, radius)
        active = np.hypot(north_offset, east_offset) <= radius + tolerance
        active[north_radius, east_radius] = True
        active.setflags(write=False)
        return active.shape, active

    def describe(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "interface": self.interface,
            "stability": self.stability,
            "parameters": asdict(self),
        }


def _coerce_disk(value: GeodesicDisk | Real) -> GeodesicDisk:
    if isinstance(value, GeodesicDisk):
        return value
    return GeodesicDisk(value)


def _validate_values(values: Any) -> np.ndarray:
    if not isinstance(values, np.ndarray):
        raise TypeError("spherical morphology requires a NumPy array")
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("spherical morphology requires scalar HW layout")
    if values.dtype == np.bool_:
        return values
    if values.dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise TypeError(
            "spherical morphology supports boolean masks and float32/float64 "
            "scalar fields; categorical labels and NumPy HWC channel vectors have no "
            "canonical ordering"
        )
    return values


def _reduce(
    values: Any,
    footprint: GeodesicDisk | Real,
    *,
    reduction: MorphologyReduction,
    max_sampled_elements: int | None = None,
) -> np.ndarray:
    from panorai.core.numpy._modules import _tangent_coordinates

    source = _validate_values(values)
    disk = _coerce_disk(footprint)
    kernel_size, active = disk.kernel(source.shape)
    height, width = source.shape
    active_flat = active.reshape(-1)
    active_count = int(active_flat.sum())
    if max_sampled_elements is None:
        rows_per_chunk = height
    else:
        if isinstance(max_sampled_elements, bool) or not isinstance(
            max_sampled_elements, int
        ):
            raise TypeError("max_sampled_elements must be an integer or None")
        if max_sampled_elements < 1:
            raise ValueError("max_sampled_elements must be positive")
        rows_per_chunk = max(1, max_sampled_elements // (active_count * width))

    source_flat = source.reshape(-1)
    result = source_flat.copy()
    reducer = np.maximum.at if reduction == "max" else np.minimum.at
    columns = np.arange(width, dtype=np.int64)[None, :]
    for row_start in range(0, height, rows_per_chunk):
        row_stop = min(height, row_start + rows_per_chunk)
        map_x, map_y = _tangent_coordinates(
            source.shape,
            source.shape,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            dilation=(1, 1),
            angular_step_deg=disk.angular_step_deg,
            rows=slice(row_start, row_stop),
        )
        xx = np.floor(map_x[active_flat] + 0.5).astype(np.int64) % width
        yy = np.clip(
            np.floor(map_y[active_flat] + 0.5).astype(np.int64),
            0,
            height - 1,
        )
        target = np.broadcast_to(
            np.arange(row_start, row_stop, dtype=np.int64)[:, None] * width + columns,
            (active_count, row_stop - row_start, width),
        ).reshape(-1)
        neighbour = (yy * width + xx).reshape(-1)

        # Symmetrizing the sampled relation makes dilation and erosion an
        # adjoint pair on the finite ERP lattice.  Opening/closing are therefore
        # genuinely idempotent instead of only visually approximate near poles.
        reducer(result, target, source_flat[neighbour])
        reducer(result, neighbour, source_flat[target])
    return result.reshape(source.shape)


def dilate(
    values: Any,
    footprint: GeodesicDisk | Real,
    *,
    max_sampled_elements: int | None = _DEFAULT_MAX_SAMPLED_ELEMENTS,
) -> np.ndarray:
    """Return intrinsic grayscale or binary dilation by a geodesic disk."""

    return _reduce(
        values,
        footprint,
        reduction="max",
        max_sampled_elements=max_sampled_elements,
    )


def erode(
    values: Any,
    footprint: GeodesicDisk | Real,
    *,
    max_sampled_elements: int | None = _DEFAULT_MAX_SAMPLED_ELEMENTS,
) -> np.ndarray:
    """Return intrinsic grayscale or binary erosion by a geodesic disk."""

    return _reduce(
        values,
        footprint,
        reduction="min",
        max_sampled_elements=max_sampled_elements,
    )


def opening(
    values: Any,
    footprint: GeodesicDisk | Real,
    *,
    max_sampled_elements: int | None = _DEFAULT_MAX_SAMPLED_ELEMENTS,
) -> np.ndarray:
    """Erode then dilate with the same intrinsic disk."""

    disk = _coerce_disk(footprint)
    return dilate(
        erode(values, disk, max_sampled_elements=max_sampled_elements),
        disk,
        max_sampled_elements=max_sampled_elements,
    )


def closing(
    values: Any,
    footprint: GeodesicDisk | Real,
    *,
    max_sampled_elements: int | None = _DEFAULT_MAX_SAMPLED_ELEMENTS,
) -> np.ndarray:
    """Dilate then erode with the same intrinsic disk."""

    disk = _coerce_disk(footprint)
    return erode(
        dilate(values, disk, max_sampled_elements=max_sampled_elements),
        disk,
        max_sampled_elements=max_sampled_elements,
    )


@dataclass(frozen=True, slots=True)
class _MorphologyOperation:
    footprint: GeodesicDisk | float
    max_sampled_elements: int | None = _DEFAULT_MAX_SAMPLED_ELEMENTS

    operation_name: ClassVar[str]
    interface: ClassVar[str] = "panorai-spherical-morphology/v1"
    stability: ClassVar[str] = "stable"
    spherical_operation: ClassVar[bool] = True
    canonical_erp_output: ClassVar[bool] = True
    preserves_shadow_state: ClassVar[bool] = True
    interpolation: ClassVar[str] = "nearest"

    def __post_init__(self) -> None:
        object.__setattr__(self, "footprint", _coerce_disk(self.footprint))
        if self.max_sampled_elements is not None:
            value = self.max_sampled_elements
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("max_sampled_elements must be an integer or None")
            if value < 1:
                raise ValueError("max_sampled_elements must be positive")

    def __call__(self, values: Any) -> np.ndarray:
        function = {
            "dilate": dilate,
            "erode": erode,
            "opening": opening,
            "closing": closing,
        }[self.operation_name]
        return function(
            values,
            self.footprint,
            max_sampled_elements=self.max_sampled_elements,
        )

    def transform_mask(self, mask: Any) -> np.ndarray:
        result = erode(
            np.asarray(mask),
            self.footprint,
            max_sampled_elements=self.max_sampled_elements,
        )
        if self.operation_name in {"opening", "closing"}:
            result = erode(
                result,
                self.footprint,
                max_sampled_elements=self.max_sampled_elements,
            )
        return result

    def apply_with_masks(
        self,
        values: Any,
        support: Any,
        validity: Any,
        *,
        invalid_policy: str,
        min_valid_weight: float | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if invalid_policy != "propagate" or min_valid_weight is not None:
            raise ValueError(
                "Stable spherical morphology uses strict validity propagation"
            )
        return (
            self(values),
            self.transform_mask(support),
            self.transform_mask(validity),
        )

    def transform_grid(self, input_grid: Any) -> Any:
        return input_grid.with_transform(
            operation=self.operation_name,
            output_shape_hw=input_grid.shape_hw,
            domain=input_grid.domain,
            alignment=input_grid.alignment,
            first_center_px_yx=input_grid.first_center_px_yx,
            pixel_stride_yx=input_grid.pixel_stride_yx,
            parameters={
                "footprint": self.footprint.describe()["parameters"],
                "max_sampled_elements": self.max_sampled_elements,
            },
        )

    def describe(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "interface": self.interface,
            "stability": self.stability,
            "parameters": {
                "footprint": self.footprint.describe(),
                "max_sampled_elements": self.max_sampled_elements,
            },
        }


class Dilate(_MorphologyOperation):
    """Configured intrinsic grayscale or binary dilation."""

    operation_name = "dilate"


class Erode(_MorphologyOperation):
    """Configured intrinsic grayscale or binary erosion."""

    operation_name = "erode"


class Opening(_MorphologyOperation):
    """Configured intrinsic opening (erosion followed by dilation)."""

    operation_name = "opening"


class Closing(_MorphologyOperation):
    """Configured intrinsic closing (dilation followed by erosion)."""

    operation_name = "closing"


__all__ = [
    "Closing",
    "Dilate",
    "Erode",
    "GeodesicDisk",
    "Opening",
    "closing",
    "dilate",
    "erode",
    "opening",
]
