"""Typed raster values carried by the Experimental pipeline.

The classes in this module describe data and provenance only.  Projection and
reconstruction continue to be performed by the Stable object workflow and the
canonical :mod:`panorai.geometry` implementation.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


FIELD_KINDS = (
    "image",
    "depth",
    "labels",
    "mask",
    "logits",
    "features",
    "confidence",
)


def _output_kinds() -> tuple[str, ...]:
    from ._observations import NON_RASTER_KINDS

    return (*FIELD_KINDS, *NON_RASTER_KINDS)


_DISCRETE_KINDS = frozenset({"labels", "mask"})
_FLOAT_KINDS = frozenset({"depth", "logits", "features", "confidence"})


def _clone(value: Any) -> Any:
    from panorai.data._workflow import clone_array

    return clone_array(value)


def _normalize_name(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _inferred_kind(name: str) -> str:
    if name in FIELD_KINDS:
        return name
    # Named enhanced images and other user-defined continuous rasters use the
    # image sampling contract unless the caller declares a stronger semantic.
    return "image"


def _normalize_kind(kind: str | None, *, name: str) -> str:
    resolved = _inferred_kind(name) if kind is None else kind
    if resolved not in FIELD_KINDS:
        raise ValueError(f"kind must be one of {FIELD_KINDS}; got {resolved!r}")
    return resolved


def _internal_kind(kind: str) -> str:
    if kind in _DISCRETE_KINDS:
        return "labels"
    if kind == "depth":
        return "depth"
    return "image"


def _validate_units(kind: str, units: str | None) -> str | None:
    if units is not None and (not isinstance(units, str) or not units.strip()):
        raise ValueError("units must be a non-empty string or None")
    if kind == "depth" and units is None:
        raise ValueError(
            "depth means radial range and requires explicit units, for example "
            "Output(model, kind='depth', units='m')"
        )
    if kind != "depth" and units is not None:
        raise ValueError("units are currently valid only for radial-range depth")
    return units


def _validate_invalid_policy(kind: str, invalid_policy: str) -> str:
    if invalid_policy not in {"propagate", "renormalize"}:
        raise ValueError("invalid_policy must be 'propagate' or 'renormalize'")
    if invalid_policy == "renormalize" and kind != "depth":
        raise ValueError(
            "validity renormalization is currently defined only for floating "
            "radial-range depth; other fields use 'propagate'"
        )
    return invalid_policy


def _spec_signature(spec: Any) -> tuple[Any, ...]:
    return (
        float(spec.center_lat_deg),
        float(spec.center_lon_deg),
        float(spec.hfov_deg),
        float(spec.vfov_deg),
        float(spec.roll_deg),
        tuple(int(item) for item in spec.output_shape_hw),
    )


@dataclass(frozen=True, slots=True)
class RasterGrid:
    """Explicit alignment identity for one ERP or ordered view raster set."""

    domain: str
    shape_hw: tuple[int, int]
    erp_shape_hw: tuple[int, int]
    alignment: str
    view_ids: tuple[str, ...] = ()
    view_specs: tuple[tuple[Any, ...], ...] = ()

    def __post_init__(self) -> None:
        if self.domain not in {"erp", "views", "derived"}:
            raise ValueError("RasterGrid domain must be 'erp', 'views', or 'derived'")
        if any(int(item) <= 0 for item in (*self.shape_hw, *self.erp_shape_hw)):
            raise ValueError("RasterGrid shapes must be positive")
        if self.domain == "views":
            if not self.view_ids or len(self.view_ids) != len(self.view_specs):
                raise ValueError("view grids require equally sized view_ids/view_specs")
        elif self.view_ids or self.view_specs:
            raise ValueError("only view grids may contain view identities")

    def describe(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "shape_hw": self.shape_hw,
            "erp_shape_hw": self.erp_shape_hw,
            "alignment": self.alignment,
            "view_ids": list(self.view_ids),
            "view_specs": [list(item) for item in self.view_specs],
        }


def _values_for_domain(value: Any, domain: str) -> tuple[Any, ...]:
    if domain == "views":
        if not isinstance(value, tuple) or not value:
            raise TypeError("view RasterField values must be a non-empty tuple")
        return value
    if isinstance(value, tuple):
        raise TypeError("ERP/derived RasterField values must be one array or tensor")
    return (value,)


def _validate_field_arrays(
    values: tuple[Any, ...],
    supports: tuple[Any, ...],
    validities: tuple[Any, ...],
    *,
    kind: str,
    grid: RasterGrid,
) -> None:
    from panorai.data._workflow import (
        backend,
        batch_size,
        device,
        is_floating,
        is_integer_or_bool,
        spatial_shape,
        validate_finite_where_valid,
        validate_mask,
    )

    expected_count = len(grid.view_ids) if grid.domain == "views" else 1
    if not (len(values) == len(supports) == len(validities) == expected_count):
        raise ValueError("field values, support, validity, and grid counts must agree")
    reference: tuple[Any, ...] | None = None
    for index, (value, support, validity) in enumerate(
        zip(values, supports, validities, strict=True)
    ):
        if spatial_shape(value) != grid.shape_hw:
            raise ValueError(
                f"field value {index} has spatial shape {spatial_shape(value)}; "
                f"grid declares {grid.shape_hw}"
            )
        current = (backend(value), device(value), batch_size(value))
        if reference is None:
            reference = current
        elif current != reference:
            raise ValueError("all field rasters must share backend, device, and batch")
        validate_mask(value, support, name="support")
        checked_validity = validate_mask(value, validity, name="validity")
        if kind == "mask":
            if str(value.dtype).replace("torch.", "") != "bool":
                raise TypeError("mask fields must use boolean dtype")
        elif kind == "labels" and not is_integer_or_bool(value):
            raise TypeError("label fields must use integer or boolean dtype")
        elif kind in _FLOAT_KINDS and not is_floating(value):
            raise TypeError(f"{kind} fields must use a floating-point dtype")
        validate_finite_where_valid(value, checked_validity, name=kind)


@dataclass(frozen=True, slots=True)
class RasterField:
    """A raster with explicit semantics, support, validity, and grid identity.

    ``values`` is one array/tensor for ERP or derived fields and a tuple in the
    exact face order for view fields.  Support and validity remain independent.
    """

    name: str
    values: Any
    kind: str
    support: Any
    validity: Any
    grid: RasterGrid
    units: str | None = None
    invalid_policy: str = "propagate"

    interface = "panorai-spherical-raster-field/v1-experimental"
    stability = "experimental"

    def __post_init__(self) -> None:
        name = _normalize_name(self.name, label="field name")
        kind = _normalize_kind(self.kind, name=name)
        units = _validate_units(kind, self.units)
        _validate_invalid_policy(kind, self.invalid_policy)
        values = _values_for_domain(self.values, self.grid.domain)
        supports = _values_for_domain(self.support, self.grid.domain)
        validities = _values_for_domain(self.validity, self.grid.domain)
        _validate_field_arrays(values, supports, validities, kind=kind, grid=self.grid)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "units", units)

    @property
    def value(self) -> Any:
        """Return the single ERP/derived value; view fields use ``values``."""

        if self.grid.domain == "views":
            raise AttributeError("view fields contain an ordered values tuple")
        return self.values

    @property
    def interpolation(self) -> str:
        return "nearest" if self.kind in _DISCRETE_KINDS else "bilinear"

    def describe(self) -> dict[str, Any]:
        from panorai.data._workflow import array_description

        values = _values_for_domain(self.values, self.grid.domain)
        return {
            "interface": self.interface,
            "stability": self.stability,
            "name": self.name,
            "kind": self.kind,
            "units": self.units,
            "invalid_policy": self.invalid_policy,
            "interpolation": self.interpolation,
            "grid": self.grid.describe(),
            "value": array_description(values[0]),
            "count": len(values),
        }


@dataclass(frozen=True, slots=True)
class Output:
    """Declare the typed raster or non-raster semantics of an operation."""

    operation: Any
    kind: str | None = None
    units: str | None = None
    invalid_policy: str = "propagate"

    def __post_init__(self) -> None:
        if not callable(self.operation):
            raise TypeError("Output operation must be callable")
        if self.kind is not None and self.kind not in _output_kinds():
            raise ValueError(
                f"kind must be one of {_output_kinds()}; got {self.kind!r}"
            )
        if self.invalid_policy not in {"propagate", "renormalize"}:
            raise ValueError("invalid_policy must be 'propagate' or 'renormalize'")

    def resolve(self, name: str) -> tuple[str, str | None]:
        from ._observations import NON_RASTER_KINDS

        resolved = _inferred_kind(name) if self.kind is None else self.kind
        if resolved not in _output_kinds():
            raise ValueError(f"kind must be one of {_output_kinds()}; got {resolved!r}")
        if name in _output_kinds() and resolved != name:
            raise ValueError(
                f"reserved output name {name!r} requires kind={name!r}; "
                f"got {resolved!r}"
            )
        if resolved in NON_RASTER_KINDS:
            if self.units is not None:
                raise ValueError("units are valid only for radial-range depth")
            if self.invalid_policy != "propagate":
                raise ValueError(
                    "non-raster outputs require invalid_policy='propagate'"
                )
            return resolved, None
        _validate_invalid_policy(resolved, self.invalid_policy)
        return resolved, _validate_units(resolved, self.units)

    def __call__(self, value: Any) -> Any:
        return self.operation(value)


def field_from_container(value: Any, name: str) -> RasterField:
    """Extract a typed field from a workflow ERP or ordered face set."""

    from panorai.data._workflow import clone_array, spatial_shape

    name = _normalize_name(name, label="field name")
    if callable(getattr(value, "_workflow_data", None)):
        metadata = value._require_workflow()
        data = value._workflow_data()
        if name not in metadata:
            raise KeyError(f"unknown raster field {name!r}")
        item = metadata[name]
        kind = item.get("field_kind", _inferred_kind(name))
        shape = spatial_shape(data[name])
        grid = RasterGrid(
            domain="erp",
            shape_hw=shape,
            erp_shape_hw=shape,
            alignment="erp-pixel-centres",
        )
        return RasterField(
            name=name,
            values=clone_array(data[name]),
            kind=kind,
            units=item.get("units"),
            support=clone_array(value._workflow_support[name]),
            validity=clone_array(item["validity"]),
            grid=grid,
            invalid_policy=item.get("invalid_policy", "propagate"),
        )
    if callable(getattr(value, "_require_workflow", None)) and hasattr(value, "_faces"):
        workflow = value._require_workflow()
        metadata = value[0]._workflow_metadata
        if name not in metadata:
            raise KeyError(f"unknown raster field {name!r}")
        item = metadata[name]
        values = tuple(clone_array(face._workflow_data()[name]) for face in value)
        supports = tuple(clone_array(face._workflow_support[name]) for face in value)
        validities = tuple(
            clone_array(face._workflow_metadata[name]["validity"]) for face in value
        )
        grid = RasterGrid(
            domain="views",
            shape_hw=spatial_shape(values[0]),
            erp_shape_hw=tuple(workflow["erp_shape"]),
            alignment="gnomonic-pixel-centres",
            view_ids=tuple(str(face.face_id) for face in value),
            view_specs=tuple(_spec_signature(face.spec) for face in value),
        )
        return RasterField(
            name=name,
            values=values,
            kind=item.get("field_kind", _inferred_kind(name)),
            units=item.get("units"),
            support=supports,
            validity=validities,
            grid=grid,
            invalid_policy=item.get("invalid_policy", "propagate"),
        )
    raise TypeError("Select requires a workflow ERP, face set, or ResultBundle")


def _field_metadata(field: RasterField, validity: Any) -> dict[str, Any]:
    from panorai.data.fields import RasterFieldSpec

    field_spec = (
        RasterFieldSpec.categorical()
        if field.kind in _DISCRETE_KINDS
        else RasterFieldSpec.continuous(units=field.units)
    )
    return {
        "kind": _internal_kind(field.kind),
        "field_kind": field.kind,
        "units": field.units,
        "validity": _clone(validity),
        "interpolation": field.interpolation,
        "invalid_policy": field.invalid_policy,
        "field_spec": field_spec,
    }


def _same_execution_grid(target: Any, field: RasterField) -> None:
    metadata = target._require_workflow()
    if hasattr(target, "_faces"):
        existing_name = next(iter(target[0]._workflow_metadata))
    else:
        existing_name = next(iter(metadata))
    target_grid = field_from_container(target, existing_name).grid
    if target_grid != field.grid:
        raise ValueError(
            "cannot attach a raster field from an incompatible grid; "
            f"target={target_grid.describe()}, field={field.grid.describe()}"
        )


def attach_raster_field(
    target: Any,
    name: str,
    field: RasterField,
    *,
    replace: bool = False,
) -> Any:
    """Return a workflow container with one compatible typed field attached."""

    from panorai.data._workflow import (
        backend,
        batch_size,
        clone_array,
        copy_metadata,
        device,
    )

    name = _normalize_name(name, label="attached field name")
    if not isinstance(field, RasterField):
        raise TypeError("attached values must be RasterField objects")
    _same_execution_grid(target, field)

    if hasattr(target, "_faces"):
        existing = target[0]._workflow_metadata
        if name in existing and not replace:
            raise ValueError(f"raster field {name!r} already exists; pass replace=True")
        from panorai.data.gnomonic_image import GnomonicFace
        from panorai.data.gnomonic_imageset import GnomonicFaceSet

        values = _values_for_domain(field.values, "views")
        supports = _values_for_domain(field.support, "views")
        validities = _values_for_domain(field.validity, "views")
        new_faces = []
        for face, value, support, validity in zip(
            target, values, supports, validities, strict=True
        ):
            source = next(iter(face._workflow_data().values()))
            if (backend(value), device(value), batch_size(value)) != (
                backend(source),
                device(source),
                batch_size(source),
            ):
                raise ValueError(
                    "attached fields must preserve target backend, device, and batch"
                )
            data = {
                key: clone_array(item) for key, item in face._workflow_data().items()
            }
            metadata = copy_metadata(face._workflow_metadata)
            field_support = {
                key: clone_array(item) for key, item in face._workflow_support.items()
            }
            data[name] = clone_array(value)
            metadata[name] = _field_metadata(field, validity)
            field_support[name] = clone_array(support)
            new_face = GnomonicFace(
                data,
                face.lat,
                face.lon,
                face.fov,
                hfov_deg=face.hfov_deg,
                vfov_deg=face.vfov_deg,
                roll_deg=face.roll_deg,
                face_id=face.face_id,
            )
            new_face.spec = face.spec
            new_face._erp_shape_hw = face._erp_shape_hw
            new_face._workflow_metadata = metadata
            new_face._workflow_support = field_support
            new_face.support_mask = clone_array(next(iter(field_support.values())))
            new_faces.append(new_face)
        result = GnomonicFaceSet(new_faces, channel_name=target.channel_name)
        result._workflow = dict(target._workflow)
        result._workflow["source_metadata"] = copy_metadata(
            new_faces[0]._workflow_metadata
        )
        return result

    if callable(getattr(target, "_workflow_data", None)):
        existing = target._require_workflow()
        if name in existing and not replace:
            raise ValueError(f"raster field {name!r} already exists; pass replace=True")
        value = field.value
        source = next(iter(target._workflow_data().values()))
        if (backend(value), device(value), batch_size(value)) != (
            backend(source),
            device(source),
            batch_size(source),
        ):
            raise ValueError(
                "attached fields must preserve target backend, device, and batch"
            )
        from panorai.data.equirectangular_image import EquirectangularImage

        data = {key: clone_array(item) for key, item in target._workflow_data().items()}
        metadata = copy_metadata(target._workflow_metadata)
        supports = {
            key: clone_array(item) for key, item in target._workflow_support.items()
        }
        data[name] = clone_array(value)
        metadata[name] = _field_metadata(field, field.validity)
        supports[name] = clone_array(field.support)
        result = EquirectangularImage(
            data,
            shadow_angle=target.shadow_angle,
            lat=target.lat,
            lon=target.lon,
            shadow_padded=target.shadow_padded,
        )
        result._workflow_metadata = metadata
        result._workflow_support = supports
        result.support_mask = clone_array(next(iter(supports.values())))
        result.sampler = target.sampler
        result.projection = target.projection
        return result

    raise TypeError("Attach target must be a workflow ERP or face set")


class ResultBundle(Mapping[str, Any]):
    """Immutable named results produced by logical pipeline branches."""

    interface = "panorai-spherical-result-bundle/v1-experimental"
    stability = "experimental"

    def __init__(
        self,
        values: Mapping[str, Any],
        *,
        descriptions: Mapping[str, Any] | None = None,
    ) -> None:
        if not values:
            raise ValueError("ResultBundle cannot be empty")
        normalized = {
            _normalize_name(name, label="result name"): value
            for name, value in values.items()
        }
        if len(normalized) != len(values):
            raise ValueError("ResultBundle names must be unique")
        self._values = MappingProxyType(normalized)
        self._descriptions = MappingProxyType(dict(descriptions or {}))

    def __getitem__(self, name: str) -> Any:
        return self._values[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def describe(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "stability": self.stability,
            "branches": [
                {
                    "name": name,
                    "output_type": type(value).__name__,
                    "execution": self._descriptions.get(name),
                }
                for name, value in self._values.items()
            ],
        }


__all__ = [
    "FIELD_KINDS",
    "Output",
    "RasterField",
    "RasterGrid",
    "ResultBundle",
]
