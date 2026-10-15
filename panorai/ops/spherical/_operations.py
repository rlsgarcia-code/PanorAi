"""Configured sphere-native operations for use inside ``OnSphere``."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from panorai.image_processing import (
    spherical_filter2d,
    spherical_gaussian_blur,
    spherical_resize,
    spherical_rotate,
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


__all__ = ["GaussianBlur", "Resize", "Rotate"]
