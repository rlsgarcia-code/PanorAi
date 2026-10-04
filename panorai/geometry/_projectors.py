"""Reusable immutable projectors backed by the canonical functional API."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from numbers import Real
from threading import RLock
from typing import Any, Callable, cast

from ._contracts import CubemapSpec, GnomonicSpec, ProjectionResult
from ._engine import (
    _cubemap_forward_plan,
    _cubemap_full_back_plan,
    _cubemap_selective_back_plan,
    _cubemap_to_equirectangular_with_plan,
    _equirectangular_to_cubemap_with_plan,
    _equirectangular_to_gnomonic_with_plan,
    _gnomonic_back_plan,
    _gnomonic_forward_plan,
    _gnomonic_to_equirectangular_with_plan,
    _is_torch,
    _require_array,
    _use_selective_cubemap_plan,
    _validate_cubemap_inputs,
    _validate_shape,
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


class _PlanCache:
    """Small per-projector thread-safe LRU for immutable geometry plans."""

    __slots__ = ("_capacity", "_entries", "_lock")

    def __init__(self, capacity: int = 4) -> None:
        self._capacity = capacity
        self._entries: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._lock = RLock()

    def get_or_create(self, key: tuple[Any, ...], builder: Callable[[], Any]) -> Any:
        with self._lock:
            try:
                value = self._entries.pop(key)
            except KeyError:
                value = builder()
                if len(self._entries) >= self._capacity:
                    self._entries.popitem(last=False)
            self._entries[key] = value
            return value

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def keys(self) -> tuple[tuple[Any, ...], ...]:
        with self._lock:
            return tuple(self._entries)


def _array_signature(value: Any) -> tuple[str, str, str]:
    if _is_torch(value):
        return "torch", str(value.dtype), str(value.device)
    return "numpy", value.dtype.str, "cpu"


@dataclass(frozen=True, slots=True)
class GnomonicProjector:
    """Publicly immutable ERP/gnomonic projector with private reusable plans."""

    spec: GnomonicSpec
    interpolation: Interpolation = "bilinear"
    fill_value: Any | None = None
    invalid_policy: InvalidPolicy = "propagate"
    min_valid_weight: float | None = None
    _plans: _PlanCache = field(
        default_factory=_PlanCache,
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )

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
        _require_array(erp, "image", image=True)
        source_shape = erp.shape[-2:] if _is_torch(erp) else erp.shape[:2]
        key = (
            "erp_to_gnomonic",
            self.spec,
            tuple(source_shape),
            *_array_signature(erp),
        )
        plan = self._plans.get_or_create(
            key, lambda: _gnomonic_forward_plan(self.spec, source_shape, erp)
        )
        return _equirectangular_to_gnomonic_with_plan(
            erp,
            self.spec,
            interpolation=self.interpolation,
            invalid_policy=self.invalid_policy,
            validity_mask=validity_mask,
            min_valid_weight=self.min_valid_weight,
            return_source_pixels=return_source_pixels,
            plan=plan,
        )

    def back_project(
        self,
        face: ArrayT | ProjectionResult[ArrayT],
        output_shape_hw: ShapeHW,
        *,
        validity_mask: ArrayT | None = None,
    ) -> ProjectionResult[ArrayT]:
        data, mask = _result_payload(face, validity_mask)
        _require_array(data, "image", image=True)
        output_shape_hw = _validate_shape(output_shape_hw, "output_shape_hw")
        face_shape = data.shape[-2:] if _is_torch(data) else data.shape[:2]
        key = (
            "gnomonic_to_erp",
            self.spec,
            output_shape_hw,
            tuple(face_shape),
            *_array_signature(data),
        )
        plan = self._plans.get_or_create(
            key,
            lambda: _gnomonic_back_plan(self.spec, output_shape_hw, face_shape, data),
        )
        return _gnomonic_to_equirectangular_with_plan(
            data,
            self.spec,
            output_shape_hw,
            interpolation=self.interpolation,
            fill_value=self.fill_value,
            invalid_policy=self.invalid_policy,
            validity_mask=mask,
            min_valid_weight=self.min_valid_weight,
            plan=plan,
        )


@dataclass(frozen=True, slots=True)
class CubemapProjector:
    """Configured ERP/cubemap projector with canonical face ordering."""

    spec: CubemapSpec
    interpolation: Interpolation = "bilinear"
    fill_value: Any | None = None
    invalid_policy: InvalidPolicy = "propagate"
    min_valid_weight: float | None = None
    _plans: _PlanCache = field(
        default_factory=_PlanCache,
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )

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
        _require_array(erp, "image", image=True)
        source_shape = erp.shape[-2:] if _is_torch(erp) else erp.shape[:2]
        key = (
            "erp_to_cubemap",
            self.spec,
            tuple(source_shape),
            *_array_signature(erp),
        )
        plan = self._plans.get_or_create(
            key,
            lambda: _cubemap_forward_plan(self.spec.face_shape_hw, source_shape, erp),
        )
        return _equirectangular_to_cubemap_with_plan(
            erp,
            self.spec.face_shape_hw,
            interpolation=self.interpolation,
            invalid_policy=self.invalid_policy,
            validity_mask=validity_mask,
            min_valid_weight=self.min_valid_weight,
            plan=plan,
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
        output_shape_hw = _validate_shape(output_shape_hw, "output_shape_hw")
        _, output_shape_hw, first, _, face_shape = _validate_cubemap_inputs(
            data, output_shape_hw, self.interpolation, masks or None
        )
        key = (
            "cubemap_to_erp",
            self.spec,
            output_shape_hw,
            tuple(face_shape),
            *_array_signature(first),
        )
        plan = self._plans.get_or_create(
            key,
            lambda: (
                _cubemap_selective_back_plan(output_shape_hw, face_shape, first)
                if _use_selective_cubemap_plan(first)
                else _cubemap_full_back_plan(output_shape_hw, face_shape, first)
            ),
        )
        result = _cubemap_to_equirectangular_with_plan(
            data,
            output_shape_hw,
            interpolation=self.interpolation,
            invalid_policy=self.invalid_policy,
            validity_masks=masks or None,
            min_valid_weight=self.min_valid_weight,
            plan=plan,
        )
        if self.fill_value is None:
            return result
        # A complete cubemap covers the sphere, so ``fill_value`` is retained
        # as configuration symmetry and has no pixels to replace.
        return result
