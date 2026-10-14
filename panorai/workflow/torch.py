"""Optional Torch model-to-container bridge for spherical modules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from panorai.data import EquirectangularImage, RasterFieldResult

from ._models import (
    SphericalInputSpec,
    SphericalOutputSpec,
    SphericalOutputTree,
    materialize_output_tree,
    validate_output_specs,
)


class SphericalImageModule(nn.Module):
    """``nn.Module`` that can also execute against an ERP container."""

    def __init__(
        self,
        *,
        input_spec: SphericalInputSpec,
        output_specs: Mapping[str, SphericalOutputSpec],
        support_policy: str = "require_full",
    ) -> None:
        super().__init__()
        if not isinstance(input_spec, SphericalInputSpec):
            raise TypeError("input_spec must be a SphericalInputSpec")
        if support_policy not in {"require_full", "explicit"}:
            raise ValueError("support_policy must be 'require_full' or 'explicit'")
        self.input_spec = input_spec
        self.output_specs = validate_output_specs(output_specs)
        self.support_policy = support_policy

    def _prepare(self, panorama: EquirectangularImage) -> torch.Tensor:
        if not isinstance(panorama, EquirectangularImage):
            raise TypeError("panorama must be an EquirectangularImage")
        source = panorama.field(self.input_spec.field)
        if not isinstance(source, torch.Tensor):
            raise TypeError("the Torch workflow requires a Torch input field")
        if self.support_policy == "require_full" and (
            not bool(panorama.support(self.input_spec.field).all())
            or not bool(panorama.validity(self.input_spec.field).all())
        ):
            raise ValueError(
                "predict() requires complete support and validity; use "
                "support_policy='explicit' and return RasterFieldResult leaves"
            )
        if source.ndim == 2:
            values = source[None, None]
        elif source.ndim == 3:
            values = source[None]
        elif source.ndim == 4:
            values = source
        else:
            raise ValueError("Torch ERP fields must use HW, CHW, or NCHW")
        if not values.is_floating_point():
            values = values.to(dtype=torch.float32)
        if self.input_spec.grayscale and values.shape[1] != 1:
            if values.shape[1] < 3:
                raise ValueError("grayscale conversion requires one or at least three channels")
            values = (
                values[:, 0] * 0.299
                + values[:, 1] * 0.587
                + values[:, 2] * 0.114
            )[:, None]
        values = values * float(self.input_spec.scale)
        if self.input_spec.mean:
            mean = values.new_tensor(self.input_spec.mean)[None, :, None, None]
            std = values.new_tensor(self.input_spec.std)[None, :, None, None]
            if mean.shape[1] != values.shape[1]:
                raise ValueError("normalization channel count does not match input")
            values = (values - mean) / std
        return values

    def predict(self, panorama: EquirectangularImage) -> SphericalOutputTree:
        raw = self(self._prepare(panorama))

        def convert(value: Any):
            validity = support = None
            if isinstance(value, RasterFieldResult):
                value, validity, support = value.data, value.validity, value.support
            if not isinstance(value, torch.Tensor) or value.ndim != 4:
                raise ValueError("Torch output leaves must use NCHW tensors")
            return value, validity, support

        return materialize_output_tree(
            raw,
            output_specs=self.output_specs,
            convert_leaf=convert,
        )


__all__ = ["SphericalImageModule"]
