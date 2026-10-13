"""NumPy model-to-container bridge for composed spherical operations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from panorai.core.numpy import Module
from panorai.data import EquirectangularImage, RasterFieldResult

from ._models import (
    SphericalInputSpec,
    SphericalOutputSpec,
    SphericalOutputTree,
    materialize_output_tree,
    validate_output_specs,
)


def _all_true(mask: np.ndarray) -> bool:
    return bool(np.asarray(mask, dtype=bool).all())


class SphericalImageModule(Module):
    """NumPy module that can also execute against an ERP container."""

    def __init__(
        self,
        *,
        input_spec: SphericalInputSpec,
        output_specs: Mapping[str, SphericalOutputSpec],
        support_policy: str = "require_full",
    ) -> None:
        if not isinstance(input_spec, SphericalInputSpec):
            raise TypeError("input_spec must be a SphericalInputSpec")
        if support_policy not in {"require_full", "explicit"}:
            raise ValueError("support_policy must be 'require_full' or 'explicit'")
        self.input_spec = input_spec
        self.output_specs = validate_output_specs(output_specs)
        self.support_policy = support_policy

    def _prepare(self, panorama: EquirectangularImage) -> np.ndarray:
        if not isinstance(panorama, EquirectangularImage):
            raise TypeError("panorama must be an EquirectangularImage")
        source = panorama.field(self.input_spec.field)
        if not isinstance(source, np.ndarray):
            raise TypeError("the NumPy workflow requires a NumPy input field")
        if self.support_policy == "require_full" and (
            not _all_true(panorama.support(self.input_spec.field))
            or not _all_true(panorama.validity(self.input_spec.field))
        ):
            raise ValueError(
                "predict() requires complete support and validity; use "
                "support_policy='explicit' and return RasterFieldResult leaves"
            )
        if source.ndim == 2:
            chw = source[None]
        elif source.ndim == 3:
            chw = np.moveaxis(source, -1, 0)
        else:
            raise ValueError("NumPy ERP fields must use HW or HWC layout")
        values = chw.astype(np.float32, copy=False)
        if self.input_spec.grayscale and values.shape[0] != 1:
            if values.shape[0] < 3:
                raise ValueError("grayscale conversion requires one or at least three channels")
            values = (
                values[0] * 0.299 + values[1] * 0.587 + values[2] * 0.114
            )[None]
        values = values * float(self.input_spec.scale)
        if self.input_spec.mean:
            mean = np.asarray(self.input_spec.mean, dtype=values.dtype)[:, None, None]
            std = np.asarray(self.input_spec.std, dtype=values.dtype)[:, None, None]
            if len(mean) != values.shape[0]:
                raise ValueError("normalization channel count does not match input")
            values = (values - mean) / std
        return values[None]

    def predict(self, panorama: EquirectangularImage) -> SphericalOutputTree:
        raw = self(self._prepare(panorama))

        def convert(value: Any):
            validity = support = None
            if isinstance(value, RasterFieldResult):
                value, validity, support = value.data, value.validity, value.support
            array = np.asarray(value)
            if array.ndim != 4 or array.shape[0] != 1:
                raise ValueError("NumPy output leaves must use single-batch NCHW")
            data = np.moveaxis(array[0], 0, -1)
            if data.shape[-1] == 1:
                data = data[..., 0]
            for name, mask in (("validity", validity), ("support", support)):
                if mask is None:
                    continue
                mask_array = np.asarray(mask)
                if mask_array.ndim == 4 and mask_array.shape[0] == 1:
                    if mask_array.shape[1] != 1:
                        raise ValueError(f"{name} must have one mask channel")
                    mask_array = mask_array[0, 0]
                elif mask_array.ndim == 3 and mask_array.shape[0] == 1:
                    mask_array = mask_array[0]
                if mask_array.shape != data.shape[:2]:
                    raise ValueError(f"{name} must match the output spatial shape")
                if name == "validity":
                    validity = mask_array
                else:
                    support = mask_array
            return data, validity, support

        return materialize_output_tree(
            raw,
            output_specs=self.output_specs,
            convert_leaf=convert,
        )


__all__ = ["SphericalImageModule"]
