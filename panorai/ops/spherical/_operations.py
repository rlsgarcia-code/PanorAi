"""Configured sphere-native operations for use inside ``OnSphere``."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from panorai.image_processing import (
    spherical_bilateral_filter,
    spherical_box_blur,
    spherical_canny,
    spherical_filter2d,
    spherical_gaussian_blur,
    spherical_gradient,
    spherical_laplacian,
    spherical_median_blur,
    spherical_resize,
    spherical_rotate,
    spherical_sobel,
)


def _masked_values(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    expanded = mask if values.ndim == 2 else mask[..., None]
    return np.where(expanded, values, 0.0)


def _finalize_renormalized(
    numerator: np.ndarray,
    valid_weight: np.ndarray,
    support_weight: np.ndarray,
    *,
    min_valid_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    support = support_weight > np.finfo(np.float64).eps
    validity = support & (valid_weight >= min_valid_weight)
    denominator = valid_weight if numerator.ndim == 2 else valid_weight[..., None]
    result = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.result_type(numerator, float)),
        where=denominator > 0.0,
    )
    expanded_validity = validity if result.ndim == 2 else validity[..., None]
    result = np.where(expanded_validity, result, np.nan)
    return result, support, validity


def _boolean_mask(mask: Any) -> np.ndarray:
    value = np.asarray(mask)
    if value.dtype != np.bool_ or value.ndim != 2:
        raise TypeError("spherical operation masks must use NumPy boolean HW layout")
    return value


def _strict_neighborhood_mask(
    mask: Any,
    ksize: int | tuple[int, int],
    *,
    angular_step_deg: float | None,
) -> np.ndarray:
    source = _boolean_mask(mask)
    shape = (ksize, ksize) if isinstance(ksize, int) else ksize
    kernel = np.ones(shape, dtype=np.float64)
    covered = spherical_filter2d(
        source.astype(np.float64),
        kernel,
        angular_step_deg=angular_step_deg,
        backend="numpy",
    )
    return covered >= float(kernel.size) - 1e-9


def _identity_grid(input_grid: Any, operation: str, parameters: dict[str, Any]) -> Any:
    return input_grid.with_transform(
        operation=operation,
        output_shape_hw=input_grid.shape_hw,
        domain=input_grid.domain,
        alignment=input_grid.alignment,
        first_center_px_yx=input_grid.first_center_px_yx,
        pixel_stride_yx=input_grid.pixel_stride_yx,
        parameters=parameters,
    )


@dataclass(frozen=True, slots=True)
class Resample:
    """Configured sampling of one ERP at arbitrary canonical bearings."""

    rays_xyz: Any
    interpolation: str = "bilinear"

    def __post_init__(self) -> None:
        rays = np.asarray(self.rays_xyz)
        if rays.ndim < 1 or rays.shape[-1] != 3:
            raise ValueError("rays_xyz must have shape (..., 3)")
        if not np.issubdtype(rays.dtype, np.floating):
            raise TypeError("rays_xyz must use a floating dtype")
        if not np.isfinite(rays).all():
            raise ValueError("rays_xyz must contain only finite values")
        rays = rays.copy()
        rays.setflags(write=False)
        object.__setattr__(self, "rays_xyz", rays)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        from .functional import resample

        return resample(image, self.rays_xyz, interpolation=self.interpolation)

    def describe(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "ray_shape": self.rays_xyz.shape,
            "interpolation": self.interpolation,
        }


@dataclass(frozen=True, slots=True)
class AreaAverage:
    """Configured exact solid-angle mean of an ERP raster."""

    mask: Any | None = None

    def __post_init__(self) -> None:
        if self.mask is not None:
            mask = _boolean_mask(self.mask).copy()
            mask.setflags(write=False)
            object.__setattr__(self, "mask", mask)

    def __call__(self, values: np.ndarray) -> Any:
        from .functional import area_average

        return area_average(values, mask=self.mask)

    def describe(self) -> dict[str, Any]:
        return {"type": type(self).__name__, "has_mask": self.mask is not None}


@dataclass(frozen=True, slots=True)
class AreaHistogram:
    """Configured exact solid-angle histogram of one-channel ERP values."""

    bins: Any = 10
    value_range: tuple[float, float] | None = None
    mask: Any | None = None
    density: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.bins, np.ndarray):
            bins = np.asarray(self.bins).copy()
            bins.setflags(write=False)
            object.__setattr__(self, "bins", bins)
        if self.mask is not None:
            mask = _boolean_mask(self.mask).copy()
            mask.setflags(write=False)
            object.__setattr__(self, "mask", mask)

    def __call__(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from .functional import area_histogram

        return area_histogram(
            values,
            bins=self.bins,
            range=self.value_range,
            mask=self.mask,
            density=self.density,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "bins": (
                self.bins.tolist() if isinstance(self.bins, np.ndarray) else self.bins
            ),
            "value_range": self.value_range,
            "has_mask": self.mask is not None,
            "density": self.density,
        }


@dataclass(frozen=True, slots=True)
class Filter2D:
    """Configured tangent-plane correlation with an explicit fixed kernel."""

    kernel: Any
    angular_step_deg: float | None = None
    backend: str = "auto"

    spherical_operation = True
    canonical_erp_output = True
    preserves_shadow_state = True

    def __post_init__(self) -> None:
        values = np.asarray(self.kernel, dtype=np.float64)
        if values.ndim != 2 or min(values.shape) < 1:
            raise ValueError("kernel must be a non-empty two-dimensional array")
        if any(size % 2 == 0 for size in values.shape):
            raise ValueError("kernel dimensions must be odd")
        if not np.isfinite(values).all():
            raise ValueError("kernel values must be finite")
        values = values.copy()
        values.setflags(write=False)
        object.__setattr__(self, "kernel", values)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return spherical_filter2d(
            image,
            self.kernel,
            angular_step_deg=self.angular_step_deg,
            backend=self.backend,
        )

    def transform_mask(self, mask: Any) -> np.ndarray:
        return _strict_neighborhood_mask(
            mask, self.kernel.shape, angular_step_deg=self.angular_step_deg
        )

    def transform_grid(self, input_grid: Any) -> Any:
        return _identity_grid(
            input_grid,
            "filter2d",
            {
                "kernel": tuple(
                    tuple(float(item) for item in row) for row in self.kernel
                ),
                "angular_step_deg": self.angular_step_deg,
                "backend": self.backend,
            },
        )

    def describe(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "parameters": {
                "kernel": self.kernel.tolist(),
                "angular_step_deg": self.angular_step_deg,
                "backend": self.backend,
            },
        }


@dataclass(frozen=True, slots=True)
class BoxBlur:
    """Configured normalized or unnormalized spherical box filter."""

    ksize: int = 3
    normalize: bool = True
    angular_step_deg: float | None = None
    backend: str = "auto"

    spherical_operation = True
    canonical_erp_output = True
    preserves_shadow_state = True

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return spherical_box_blur(
            image,
            self.ksize,
            normalize=self.normalize,
            angular_step_deg=self.angular_step_deg,
            backend=self.backend,
        )

    def transform_mask(self, mask: Any) -> np.ndarray:
        return _strict_neighborhood_mask(
            mask, self.ksize, angular_step_deg=self.angular_step_deg
        )

    def transform_grid(self, input_grid: Any) -> Any:
        return _identity_grid(input_grid, "box_blur", asdict(self))

    def apply_with_masks(
        self,
        values: Any,
        support: Any,
        validity: Any,
        *,
        invalid_policy: str,
        min_valid_weight: float | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        source = np.asarray(values)
        source_support = _boolean_mask(support)
        source_validity = _boolean_mask(validity)
        if invalid_policy == "propagate":
            return (
                self(source),
                self.transform_mask(source_support),
                self.transform_mask(source_validity),
            )
        if not self.normalize:
            raise ValueError("renormalize requires BoxBlur(normalize=True)")
        if min_valid_weight is None:
            raise ValueError("renormalize requires min_valid_weight")
        usable = source_support & source_validity
        numerator = self(_masked_values(source, usable))
        valid_weight = self(usable.astype(np.float64))
        support_weight = self(source_support.astype(np.float64))
        return _finalize_renormalized(
            numerator,
            valid_weight,
            support_weight,
            min_valid_weight=min_valid_weight,
        )

    def describe(self) -> dict[str, Any]:
        return {"type": type(self).__name__, "parameters": asdict(self)}


@dataclass(frozen=True, slots=True)
class MedianBlur:
    """Configured geodesic median filter."""

    ksize: int = 3
    angular_step_deg: float | None = None

    spherical_operation = True
    canonical_erp_output = True
    preserves_shadow_state = True

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return spherical_median_blur(
            image, self.ksize, angular_step_deg=self.angular_step_deg
        )

    def transform_mask(self, mask: Any) -> np.ndarray:
        return _strict_neighborhood_mask(
            mask, self.ksize, angular_step_deg=self.angular_step_deg
        )

    def transform_grid(self, input_grid: Any) -> Any:
        return _identity_grid(input_grid, "median_blur", asdict(self))

    def describe(self) -> dict[str, Any]:
        return {"type": type(self).__name__, "parameters": asdict(self)}


@dataclass(frozen=True, slots=True)
class BilateralFilter:
    """Configured edge-preserving geodesic bilateral filter."""

    ksize: int = 5
    sigma_color: float = 25.0
    sigma_space: float = 1.5
    angular_step_deg: float | None = None

    spherical_operation = True
    canonical_erp_output = True
    preserves_shadow_state = True

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return spherical_bilateral_filter(
            image,
            self.ksize,
            self.sigma_color,
            self.sigma_space,
            angular_step_deg=self.angular_step_deg,
        )

    def transform_mask(self, mask: Any) -> np.ndarray:
        return _strict_neighborhood_mask(
            mask, self.ksize, angular_step_deg=self.angular_step_deg
        )

    def transform_grid(self, input_grid: Any) -> Any:
        return _identity_grid(input_grid, "bilateral_filter", asdict(self))

    def describe(self) -> dict[str, Any]:
        return {"type": type(self).__name__, "parameters": asdict(self)}


@dataclass(frozen=True, slots=True)
class Sobel:
    """Configured east or north spherical Sobel/Scharr derivative."""

    direction: str = "east"
    operator: str = "sobel"
    angular_step_deg: float | None = None
    normalize_by_angle: bool = False
    backend: str = "auto"

    spherical_operation = True
    canonical_erp_output = True
    preserves_shadow_state = True

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return spherical_sobel(
            image,
            direction=self.direction,
            operator=self.operator,
            angular_step_deg=self.angular_step_deg,
            normalize_by_angle=self.normalize_by_angle,
            backend=self.backend,
        )

    def transform_mask(self, mask: Any) -> np.ndarray:
        return _strict_neighborhood_mask(
            mask, 3, angular_step_deg=self.angular_step_deg
        )

    def transform_grid(self, input_grid: Any) -> Any:
        return _identity_grid(input_grid, "sobel", asdict(self))

    def describe(self) -> dict[str, Any]:
        return {"type": type(self).__name__, "parameters": asdict(self)}


@dataclass(frozen=True, slots=True)
class Gradient:
    """Configured component of the spherical Sobel/Scharr gradient."""

    component: str = "magnitude"
    operator: str = "sobel"
    angular_step_deg: float | None = None
    normalize_by_angle: bool = False
    backend: str = "auto"

    spherical_operation = True
    canonical_erp_output = True
    preserves_shadow_state = True

    def __post_init__(self) -> None:
        if self.component not in {"east", "north", "magnitude", "orientation"}:
            raise ValueError(
                "component must be 'east', 'north', 'magnitude', or 'orientation'"
            )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        result = spherical_gradient(
            image,
            operator=self.operator,
            angular_step_deg=self.angular_step_deg,
            normalize_by_angle=self.normalize_by_angle,
            backend=self.backend,
        )
        return getattr(result, self.component)

    def transform_mask(self, mask: Any) -> np.ndarray:
        return _strict_neighborhood_mask(
            mask, 3, angular_step_deg=self.angular_step_deg
        )

    def transform_grid(self, input_grid: Any) -> Any:
        return _identity_grid(input_grid, "gradient", asdict(self))

    def describe(self) -> dict[str, Any]:
        return {"type": type(self).__name__, "parameters": asdict(self)}


@dataclass(frozen=True, slots=True)
class Laplacian:
    """Configured four-neighbour spherical Laplacian."""

    angular_step_deg: float | None = None
    normalize_by_angle: bool = False
    backend: str = "auto"

    spherical_operation = True
    canonical_erp_output = True
    preserves_shadow_state = True

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return spherical_laplacian(
            image,
            angular_step_deg=self.angular_step_deg,
            normalize_by_angle=self.normalize_by_angle,
            backend=self.backend,
        )

    def transform_mask(self, mask: Any) -> np.ndarray:
        return _strict_neighborhood_mask(
            mask, 3, angular_step_deg=self.angular_step_deg
        )

    def transform_grid(self, input_grid: Any) -> Any:
        return _identity_grid(input_grid, "laplacian", asdict(self))

    def describe(self) -> dict[str, Any]:
        return {"type": type(self).__name__, "parameters": asdict(self)}


@dataclass(frozen=True, slots=True)
class Canny:
    """Configured spherical Canny edge detector."""

    threshold_low: float
    threshold_high: float
    gaussian_ksize: int = 5
    gaussian_sigma: float = 1.0
    gradient_operator: str = "sobel"
    angular_step_deg: float | None = None
    l2_gradient: bool = True
    backend: str = "auto"

    spherical_operation = True
    canonical_erp_output = True
    preserves_shadow_state = True
    interpolation = "nearest"

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return spherical_canny(
            image,
            self.threshold_low,
            self.threshold_high,
            gaussian_ksize=self.gaussian_ksize,
            gaussian_sigma=self.gaussian_sigma,
            gradient_operator=self.gradient_operator,
            angular_step_deg=self.angular_step_deg,
            l2_gradient=self.l2_gradient,
            backend=self.backend,
        )

    def transform_mask(self, mask: Any) -> np.ndarray:
        source = _boolean_mask(mask)
        if source.all():
            return source.copy()
        return np.zeros_like(source)

    def transform_grid(self, input_grid: Any) -> Any:
        return _identity_grid(input_grid, "canny", asdict(self))

    def describe(self) -> dict[str, Any]:
        return {"type": type(self).__name__, "parameters": asdict(self)}


@dataclass(frozen=True, slots=True)
class GaussianBlur:
    """Configured local-tangent Gaussian blur."""

    ksize: int = 5
    sigma: float = 1.0
    angular_step_deg: float | None = None
    backend: str = "auto"

    spherical_operation = True
    canonical_erp_output = True
    preserves_shadow_state = True

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return spherical_gaussian_blur(
            image,
            self.ksize,
            self.sigma,
            angular_step_deg=self.angular_step_deg,
            backend=self.backend,
        )

    def transform_mask(self, mask: Any) -> np.ndarray:
        source = _boolean_mask(mask)
        kernel = np.ones((self.ksize, self.ksize), dtype=np.float64)
        covered = spherical_filter2d(
            source.astype(np.float64),
            kernel,
            angular_step_deg=self.angular_step_deg,
            backend="numpy",
        )
        return covered >= float(kernel.size) - 1e-9

    def transform_grid(self, input_grid: Any) -> Any:
        return input_grid.with_transform(
            operation="gaussian_blur",
            output_shape_hw=input_grid.shape_hw,
            domain=input_grid.domain,
            alignment=input_grid.alignment,
            first_center_px_yx=input_grid.first_center_px_yx,
            pixel_stride_yx=input_grid.pixel_stride_yx,
            parameters={
                "ksize": self.ksize,
                "sigma": self.sigma,
                "angular_step_deg": self.angular_step_deg,
                "backend": self.backend,
            },
        )

    def apply_with_masks(
        self,
        values: Any,
        support: Any,
        validity: Any,
        *,
        invalid_policy: str,
        min_valid_weight: float | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        source = np.asarray(values)
        source_support = _boolean_mask(support)
        source_validity = _boolean_mask(validity)
        if invalid_policy == "propagate":
            return (
                self(source),
                self.transform_mask(source_support),
                self.transform_mask(source_validity),
            )
        if min_valid_weight is None:
            raise ValueError("renormalize requires min_valid_weight")
        usable = source_support & source_validity
        numerator = self(_masked_values(source, usable))
        valid_weight = self(usable.astype(np.float64))
        support_weight = self(source_support.astype(np.float64))
        return _finalize_renormalized(
            numerator,
            valid_weight,
            support_weight,
            min_valid_weight=min_valid_weight,
        )

    def describe(self) -> dict[str, Any]:
        return {"type": type(self).__name__, "parameters": asdict(self)}


@dataclass(frozen=True, slots=True)
class Resize:
    """Configured canonical ERP resampling at output pixel-centre rays."""

    output_shape_hw: tuple[int, int]
    interpolation: str = "bilinear"

    spherical_operation = True
    canonical_erp_output = True
    preserves_shadow_state = True

    def __call__(self, image: np.ndarray) -> np.ndarray:
        source = np.asarray(image)
        discrete = source.dtype == np.bool_ or np.issubdtype(source.dtype, np.integer)
        working = source.astype(np.uint8) if source.dtype == np.bool_ else source
        result = spherical_resize(
            working,
            self.output_shape_hw,
            interpolation=self.interpolation,
        )
        if discrete and self.interpolation == "nearest":
            return result.astype(source.dtype, copy=False)
        return result

    def transform_mask(self, mask: Any) -> np.ndarray:
        source = _boolean_mask(mask)
        sampled = spherical_resize(
            source.astype(np.uint8),
            self.output_shape_hw,
            interpolation=self.interpolation,
        )
        threshold = 0.5 if self.interpolation == "nearest" else 1.0 - 1e-12
        return sampled >= threshold

    def transform_grid(self, input_grid: Any) -> Any:
        if not input_grid.is_canonical_erp:
            raise ValueError(
                "Resize requires a canonical ERP input grid; a derived lattice "
                "must be explicitly regridded with its coordinate transform"
            )
        return input_grid.with_transform(
            operation="resize",
            output_shape_hw=self.output_shape_hw,
            domain="erp",
            alignment="erp-pixel-centres",
            erp_shape_hw=self.output_shape_hw,
            first_center_px_yx=(0.5, 0.5),
            pixel_stride_yx=(1.0, 1.0),
            parameters={
                "output_shape_hw": self.output_shape_hw,
                "interpolation": self.interpolation,
            },
            longitude_periodic=True,
        )

    def apply_with_masks(
        self,
        values: Any,
        support: Any,
        validity: Any,
        *,
        invalid_policy: str,
        min_valid_weight: float | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        source = np.asarray(values)
        source_support = _boolean_mask(support)
        source_validity = _boolean_mask(validity)
        if invalid_policy == "propagate":
            return (
                self(source),
                self.transform_mask(source_support),
                self.transform_mask(source_validity),
            )
        if self.interpolation != "bilinear":
            raise ValueError("renormalize is defined only for bilinear Resize")
        if min_valid_weight is None:
            raise ValueError("renormalize requires min_valid_weight")
        usable = source_support & source_validity
        numerator = spherical_resize(
            _masked_values(source, usable),
            self.output_shape_hw,
            interpolation="bilinear",
        )
        valid_weight = spherical_resize(
            usable.astype(np.float64),
            self.output_shape_hw,
            interpolation="bilinear",
        )
        support_weight = spherical_resize(
            source_support.astype(np.float64),
            self.output_shape_hw,
            interpolation="bilinear",
        )
        return _finalize_renormalized(
            numerator,
            valid_weight,
            support_weight,
            min_valid_weight=min_valid_weight,
        )

    def describe(self) -> dict[str, Any]:
        return {"type": type(self).__name__, "parameters": asdict(self)}


@dataclass(frozen=True, slots=True)
class Rotate:
    """Configured active rotation in PanorAi's canonical Cartesian frame."""

    rotation: Any
    output_shape_hw: tuple[int, int] | None = None
    interpolation: str = "bilinear"

    spherical_operation = True
    canonical_erp_output = True
    preserves_shadow_state = False

    def __post_init__(self) -> None:
        matrix = np.asarray(self.rotation, dtype=np.float64)
        object.__setattr__(self, "rotation", matrix.copy())

    def __call__(self, image: np.ndarray) -> np.ndarray:
        source = np.asarray(image)
        discrete = source.dtype == np.bool_ or np.issubdtype(source.dtype, np.integer)
        working = source.astype(np.uint8) if source.dtype == np.bool_ else source
        result = spherical_rotate(
            working,
            self.rotation,
            output_shape_hw=self.output_shape_hw,
            interpolation=self.interpolation,
        )
        if discrete and self.interpolation == "nearest":
            return result.astype(source.dtype, copy=False)
        return result

    def transform_mask(self, mask: Any) -> np.ndarray:
        source = _boolean_mask(mask)
        sampled = spherical_rotate(
            source.astype(np.uint8),
            self.rotation,
            output_shape_hw=self.output_shape_hw,
            interpolation=self.interpolation,
        )
        threshold = 0.5 if self.interpolation == "nearest" else 1.0 - 1e-12
        return sampled >= threshold

    def transform_grid(self, input_grid: Any) -> Any:
        if not input_grid.is_canonical_erp:
            raise ValueError(
                "Rotate requires a canonical ERP input grid; a derived lattice "
                "cannot be interpreted as canonical ERP samples"
            )
        output_shape = self.output_shape_hw or input_grid.shape_hw
        return input_grid.with_transform(
            operation="rotate",
            output_shape_hw=output_shape,
            domain="erp",
            alignment="erp-pixel-centres",
            erp_shape_hw=output_shape,
            first_center_px_yx=(0.5, 0.5),
            pixel_stride_yx=(1.0, 1.0),
            parameters={
                "rotation": tuple(
                    tuple(float(item) for item in row) for row in self.rotation
                ),
                "output_shape_hw": self.output_shape_hw,
                "interpolation": self.interpolation,
            },
            longitude_periodic=True,
        )

    def apply_with_masks(
        self,
        values: Any,
        support: Any,
        validity: Any,
        *,
        invalid_policy: str,
        min_valid_weight: float | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        source = np.asarray(values)
        source_support = _boolean_mask(support)
        source_validity = _boolean_mask(validity)
        if invalid_policy == "propagate":
            return (
                self(source),
                self.transform_mask(source_support),
                self.transform_mask(source_validity),
            )
        if self.interpolation != "bilinear":
            raise ValueError("renormalize is defined only for bilinear Rotate")
        if min_valid_weight is None:
            raise ValueError("renormalize requires min_valid_weight")
        usable = source_support & source_validity
        numerator = spherical_rotate(
            _masked_values(source, usable),
            self.rotation,
            output_shape_hw=self.output_shape_hw,
            interpolation="bilinear",
        )
        valid_weight = spherical_rotate(
            usable.astype(np.float64),
            self.rotation,
            output_shape_hw=self.output_shape_hw,
            interpolation="bilinear",
        )
        support_weight = spherical_rotate(
            source_support.astype(np.float64),
            self.rotation,
            output_shape_hw=self.output_shape_hw,
            interpolation="bilinear",
        )
        return _finalize_renormalized(
            numerator,
            valid_weight,
            support_weight,
            min_valid_weight=min_valid_weight,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "parameters": {
                "rotation": self.rotation.tolist(),
                "output_shape_hw": self.output_shape_hw,
                "interpolation": self.interpolation,
            },
        }


__all__ = [
    "AreaAverage",
    "AreaHistogram",
    "BilateralFilter",
    "BoxBlur",
    "Canny",
    "Filter2D",
    "GaussianBlur",
    "Gradient",
    "Laplacian",
    "MedianBlur",
    "Resize",
    "Resample",
    "Rotate",
    "Sobel",
]
