"""Reusable immutable projectors backed by the canonical functional API."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, cast

from ._contracts import CubemapSpec, GnomonicSpec, ProjectionResult
from ._engine import (
    cubemap_to_equirectangular,
    equirectangular_to_cubemap,
    equirectangular_to_gnomonic,
    gnomonic_to_equirectangular,
)
from ._typing import (
    ArrayT,
    Interpolation,
    InvalidPolicy,
    ShapeHW,
    _validate_interpolation,
    _validate_invalid_policy,
)


def _projector_options(
    interpolation: Interpolation,
    invalid_policy: InvalidPolicy,
    min_valid_weight: float | None,
) -> tuple[Interpolation, InvalidPolicy, float | None]:
    interpolation = _validate_interpolation(interpolation)
    invalid_policy = _validate_invalid_policy(invalid_policy)
    if invalid_policy == "propagate":
        if min_valid_weight is not None:
            raise ValueError(
                "min_valid_weight is only valid with invalid_policy='renormalize'"
            )
        return interpolation, invalid_policy, None
    if interpolation != "bilinear":
        raise ValueError("invalid_policy='renormalize' requires bilinear interpolation")
    if min_valid_weight is None:
        raise ValueError(
            "min_valid_weight is required for invalid_policy='renormalize'"
        )
    if isinstance(min_valid_weight, bool) or not isinstance(min_valid_weight, Real):
        raise TypeError("min_valid_weight must be a finite real number")
    threshold = float(min_valid_weight)
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("min_valid_weight must be finite and in the interval (0, 1]")
    return interpolation, invalid_policy, threshold


def _result_payload(
    value: ArrayT | ProjectionResult[ArrayT],
    validity_mask: ArrayT | None,
) -> tuple[ArrayT, ArrayT | None]:
    if not isinstance(value, ProjectionResult):
        return value, validity_mask
    if validity_mask is not None and value.validity_mask is not None:
        raise ValueError(
            "validity_mask was supplied both explicitly and by ProjectionResult"
        )
    return cast(ArrayT, value.data), cast(
        ArrayT | None,
        value.validity_mask if validity_mask is None else validity_mask,
    )


@dataclass(frozen=True, slots=True)
class GnomonicProjector:
    """Configured ERP/gnomonic projector with no mutable runtime state."""

    spec: GnomonicSpec
    interpolation: Interpolation = "bilinear"
    fill_value: Any | None = None
    invalid_policy: InvalidPolicy = "propagate"
    min_valid_weight: float | None = None

    def __post_init__(self) -> None:
        interpolation, invalid_policy, threshold = _projector_options(
            self.interpolation, self.invalid_policy, self.min_valid_weight
        )
        object.__setattr__(self, "interpolation", interpolation)
        object.__setattr__(self, "invalid_policy", invalid_policy)
        object.__setattr__(self, "min_valid_weight", threshold)

    def project(
        self,
        erp: ArrayT,
        *,
        validity_mask: ArrayT | None = None,
        return_source_pixels: bool = False,
    ) -> ProjectionResult[ArrayT]:
        return equirectangular_to_gnomonic(
            erp,
            self.spec,
            interpolation=self.interpolation,
            invalid_policy=self.invalid_policy,
            validity_mask=validity_mask,
            min_valid_weight=self.min_valid_weight,
            return_source_pixels=return_source_pixels,
        )

    def back_project(
        self,
        face: ArrayT | ProjectionResult[ArrayT],
        output_shape_hw: ShapeHW,
        *,
        validity_mask: ArrayT | None = None,
    ) -> ProjectionResult[ArrayT]:
        data, mask = _result_payload(face, validity_mask)
        return gnomonic_to_equirectangular(
            data,
            self.spec,
            output_shape_hw,
            interpolation=self.interpolation,
            fill_value=self.fill_value,
            invalid_policy=self.invalid_policy,
            validity_mask=mask,
            min_valid_weight=self.min_valid_weight,
        )


@dataclass(frozen=True, slots=True)
class CubemapProjector:
    """Configured ERP/cubemap projector with canonical face ordering."""

    spec: CubemapSpec
    interpolation: Interpolation = "bilinear"
    fill_value: Any | None = None
    invalid_policy: InvalidPolicy = "propagate"
    min_valid_weight: float | None = None

    def __post_init__(self) -> None:
        interpolation, invalid_policy, threshold = _projector_options(
            self.interpolation, self.invalid_policy, self.min_valid_weight
        )
        object.__setattr__(self, "interpolation", interpolation)
        object.__setattr__(self, "invalid_policy", invalid_policy)
        object.__setattr__(self, "min_valid_weight", threshold)

    def project(
        self, erp: ArrayT, *, validity_mask: ArrayT | None = None
    ) -> dict[str, ProjectionResult[ArrayT]]:
        return equirectangular_to_cubemap(
            erp,
            self.spec.face_shape_hw,
            interpolation=self.interpolation,
            invalid_policy=self.invalid_policy,
            validity_mask=validity_mask,
            min_valid_weight=self.min_valid_weight,
        )

    def back_project(
        self,
        faces: Mapping[str, ArrayT | ProjectionResult[ArrayT]],
        output_shape_hw: ShapeHW,
        *,
        validity_masks: Mapping[str, ArrayT] | None = None,
    ) -> ProjectionResult[ArrayT]:
        data: dict[str, ArrayT] = {}
        masks: dict[str, ArrayT] = {}
        for name, value in faces.items():
            explicit = None if validity_masks is None else validity_masks.get(name)
            face_data, face_mask = _result_payload(value, explicit)
            data[name] = face_data
            if face_mask is not None:
                masks[name] = face_mask
        result = cubemap_to_equirectangular(
            data,
            output_shape_hw,
            interpolation=self.interpolation,
            invalid_policy=self.invalid_policy,
            validity_masks=masks or None,
            min_valid_weight=self.min_valid_weight,
        )
        if self.fill_value is None:
            return result
        # A complete cubemap covers the sphere, so ``fill_value`` is retained
        # as configuration symmetry and has no pixels to replace.
        return result
