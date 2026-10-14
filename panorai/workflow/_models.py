"""Backend-neutral contracts for executing composed models on ERP containers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from panorai.data import EquirectangularImage, RasterFieldResult, RasterFieldSpec


SPHERICAL_MODEL_WORKFLOW_INTERFACE = "panorai-spherical-model-workflow/v1"


@dataclass(frozen=True, slots=True)
class SphericalInputSpec:
    """Declare how one ERP field becomes a model input tensor."""

    field: str = "image"
    grayscale: bool = False
    scale: float = 1.0
    mean: tuple[float, ...] = ()
    std: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.field).strip():
            raise ValueError("field must be non-empty")
        if not isinstance(self.grayscale, bool):
            raise TypeError("grayscale must be boolean")
        if not isinstance(self.scale, (int, float)) or isinstance(self.scale, bool):
            raise TypeError("scale must be a real number")
        if self.mean and self.std and len(self.mean) != len(self.std):
            raise ValueError("mean and std must have the same number of channels")
        if any(float(value) == 0.0 for value in self.std):
            raise ValueError("std entries must be non-zero")


@dataclass(frozen=True, slots=True)
class SphericalOutputSpec:
    """Declare the field semantics of one model-output leaf."""

    field_spec: RasterFieldSpec
    field: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.field_spec, RasterFieldSpec):
            raise TypeError("field_spec must be a RasterFieldSpec")
        if self.field is not None and not str(self.field).strip():
            raise ValueError("field must be non-empty when provided")

    @classmethod
    def continuous(
        cls,
        *,
        field: str | None = None,
        units: str | None = None,
        component_ids: tuple[str, ...] = (),
    ) -> "SphericalOutputSpec":
        return cls(
            RasterFieldSpec.continuous(units=units, component_ids=component_ids),
            field,
        )

    @classmethod
    def categorical(
        cls,
        *,
        field: str | None = None,
        component_ids: tuple[str, ...] = (),
    ) -> "SphericalOutputSpec":
        return cls(
            RasterFieldSpec.categorical(component_ids=component_ids),
            field,
        )


class SphericalOutputTree(Mapping[str, EquirectangularImage]):
    """Immutable mapping from output paths to ERP containers."""

    def __init__(self, outputs: Mapping[str, EquirectangularImage]) -> None:
        if not outputs:
            raise ValueError("an output tree cannot be empty")
        self._outputs = MappingProxyType(dict(outputs))

    def __getitem__(self, key: str) -> EquirectangularImage:
        return self._outputs[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._outputs)

    def __len__(self) -> int:
        return len(self._outputs)


def flatten_outputs(value: Any, *, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        flattened: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key).strip()
            if not name:
                raise ValueError("model output keys must be non-empty")
            path = name if not prefix else f"{prefix}.{name}"
            flattened.update(flatten_outputs(item, prefix=path))
        return flattened
    if not prefix:
        return {"output": value}
    return {prefix: value}


def validate_output_specs(
    output_specs: Mapping[str, SphericalOutputSpec],
) -> MappingProxyType[str, SphericalOutputSpec]:
    result = dict(output_specs)
    if not result:
        raise ValueError("output_specs cannot be empty")
    if any(not isinstance(value, SphericalOutputSpec) for value in result.values()):
        raise TypeError("every output spec must be a SphericalOutputSpec")
    return MappingProxyType(result)


def materialize_output_tree(
    raw_output: Any,
    *,
    output_specs: Mapping[str, SphericalOutputSpec],
    convert_leaf,
) -> SphericalOutputTree:
    leaves = flatten_outputs(raw_output)
    if set(leaves) != set(output_specs):
        raise ValueError(
            "model outputs and output_specs differ: "
            f"outputs={sorted(leaves)}, specs={sorted(output_specs)}"
        )
    containers: dict[str, EquirectangularImage] = {}
    for path, value in leaves.items():
        spec = output_specs[path]
        data, validity, support = convert_leaf(value)
        if isinstance(data, RasterFieldResult):
            raise TypeError("convert_leaf must unpack RasterFieldResult")
        field = spec.field or path.rsplit(".", 1)[-1]
        containers[path] = EquirectangularImage.from_field(
            field,
            data,
            spec=spec.field_spec,
            valid=validity,
            support=support,
        )
    return SphericalOutputTree(containers)


__all__ = [
    "SPHERICAL_MODEL_WORKFLOW_INTERFACE",
    "SphericalInputSpec",
    "SphericalOutputSpec",
    "SphericalOutputTree",
]
