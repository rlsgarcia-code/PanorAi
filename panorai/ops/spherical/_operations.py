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
