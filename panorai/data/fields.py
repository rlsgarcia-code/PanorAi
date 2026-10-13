"""Stable typed raster-field contracts for the object workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


FieldKind = Literal["continuous", "categorical"]


@dataclass(frozen=True, slots=True)
class RasterFieldSpec:
    """Semantics required to transport and fuse one ERP-aligned raster field.

    This first contract covers scalar or channel-wise invariant fields.  Values
    whose components change with the local basis (for example optical flow or
    surface normals) require a future frame-aware transport contract.
    """

    kind: FieldKind
    units: str | None = None
    interpolation: str | None = None
    default_blend: str | None = None
    component_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"continuous", "categorical"}:
            raise ValueError("kind must be 'continuous' or 'categorical'")
        interpolation = self.interpolation or (
            "nearest" if self.kind == "categorical" else "bilinear"
        )
        default_blend = self.default_blend or (
            "closest" if self.kind == "categorical" else "average"
        )
        if interpolation not in {"nearest", "bilinear"}:
            raise ValueError("interpolation must be 'nearest' or 'bilinear'")
        if default_blend not in {"average", "gaussian", "closest"}:
            raise ValueError(
                "default_blend must be 'average', 'gaussian', or 'closest'"
            )
        if self.kind == "categorical" and interpolation != "nearest":
            raise ValueError("categorical fields require nearest interpolation")
        if self.kind == "continuous" and interpolation != "bilinear":
            raise ValueError("continuous fields require bilinear interpolation")
        if self.kind == "categorical" and default_blend != "closest":
            raise ValueError("categorical fields require closest blending")
        if self.units is not None and not str(self.units).strip():
            raise ValueError("units must be non-empty when provided")
        components = tuple(str(item).strip() for item in self.component_ids)
        if any(not item for item in components):
            raise ValueError("component_ids must be non-empty")
        if len(set(components)) != len(components):
            raise ValueError("component_ids must be unique")
        object.__setattr__(self, "interpolation", interpolation)
        object.__setattr__(self, "default_blend", default_blend)
        object.__setattr__(self, "component_ids", components)

    @classmethod
    def continuous(
        cls,
        *,
        units: str | None = None,
        interpolation: str = "bilinear",
        default_blend: str = "average",
        component_ids: tuple[str, ...] = (),
    ) -> "RasterFieldSpec":
        return cls(
            "continuous", units, interpolation, default_blend, component_ids
        )

    @classmethod
    def categorical(
        cls, *, component_ids: tuple[str, ...] = ()
    ) -> "RasterFieldSpec":
        return cls("categorical", None, "nearest", "closest", component_ids)


@dataclass(frozen=True, slots=True)
class RasterFieldResult:
    """Array result with optional explicit data validity and support."""

    data: Any
    validity: Any | None = None
    support: Any | None = None


def legacy_field_spec(name: str, units: str | None = None) -> RasterFieldSpec:
    if name == "labels":
        return RasterFieldSpec.categorical()
    return RasterFieldSpec.continuous(units=units)
