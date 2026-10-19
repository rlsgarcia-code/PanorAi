"""Linear stages that adapt the existing stable object workflow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import sys
from types import MappingProxyType
from typing import Any

import numpy as np

from ._results import StageResult
from ._fields import (
    FIELD_KINDS,
    Output,
    RasterField,
    ResultBundle,
    attach_raster_field,
    field_from_container,
)
from ._observations import (
    ClassificationResult,
    EmbeddingResult,
    NON_RASTER_KINDS,
    PerViewResults,
    SphericalObservations,
    ViewResult,
    coerce_view_observations,
)


def _is_torch_tensor(value: Any) -> bool:
    module = type(value).__module__
    return module == "torch" or module.startswith("torch.")


def _clone(value: Any) -> Any:
    return value.clone() if _is_torch_tensor(value) else np.asarray(value).copy()


def _spatial_shape(value: Any) -> tuple[int, int]:
    if _is_torch_tensor(value):
        return int(value.shape[-2]), int(value.shape[-1])
    return int(value.shape[0]), int(value.shape[1])


def _is_torch_module(value: Any) -> bool:
    torch = sys.modules.get("torch")
    return bool(torch is not None and isinstance(value, torch.nn.Module))


def _has_methods(value: Any, *names: str) -> bool:
    return all(callable(getattr(value, name, None)) for name in names)


def _erp_input(value: Any, name: str) -> tuple[Any, RasterField]:
    if not _has_methods(value, "_require_workflow", "validity", "views") or not hasattr(
        value, "image"
    ):
        raise TypeError("OnSphere requires an EquirectangularImage input")
    metadata = value._require_workflow()
    if name not in metadata:
        raise KeyError(
            f"unknown OnSphere input field {name!r}; choose one of {list(metadata)}"
        )
    return value, field_from_container(value, name)


def _wrap_image_result(
    source: Any,
    result: Any,
    *,
    support: Any,
    validity: Any,
    preserves_shadow_state: bool,
) -> Any:
    source_shape = _spatial_shape(source.image)
    result_shape = _spatial_shape(result)
    if _is_torch_tensor(source.image) != _is_torch_tensor(result):
        raise TypeError("OnSphere must preserve the input backend")
    if preserves_shadow_state:
        shadow_angle = source.shadow_angle
        shadow_padded = source.shadow_padded
    else:
        shadow_angle = 0.0
        shadow_padded = False
    wrapped = type(source)(
        result,
        shadow_angle=shadow_angle,
        lat=source.lat,
        lon=source.lon,
        support_mask=support,
        valid=validity,
        shadow_padded=shadow_padded,
    )
    if source_shape == result_shape:
        wrapped.sampler = source.sampler
        wrapped.projection = source.projection
    return wrapped


@dataclass(frozen=True, slots=True)
class OnSphere:
    """Execute a declared sphere-aware operation directly on an ERP.

    Registered NumPy operations from :mod:`panorai.ops.spherical` and
    validated Torch modules are accepted. Conventional spatial Torch layers
    fail before their ``forward`` method executes.
    """

    operation: Any
    output: str = "image"
    input: str = "image"

    input_kind = "erp"

    def __post_init__(self) -> None:
        if not isinstance(self.output, str) or not self.output.strip():
            raise ValueError("output must be a non-empty string")
        if not isinstance(self.input, str) or not self.input.strip():
            raise ValueError("input must be a non-empty string")

    @property
    def output_kind(self) -> str:
        if isinstance(self.operation, Output):
            if self.operation.kind in NON_RASTER_KINDS:
                return "result"
            if self.operation.kind is None and self.output in NON_RASTER_KINDS:
                return "result"
            return "erp"
        return "erp" if self.output == "image" else "result"

    def execute(self, value: Any) -> StageResult:
        panorama, input_field = _erp_input(value, self.input)
        source = input_field.value
        input_shape = _spatial_shape(source)
        output_grid = None
        declaration = self.operation if isinstance(self.operation, Output) else None
        operation = (
            self.operation.operation
            if isinstance(self.operation, Output)
            else self.operation
        )
        if declaration is not None:
            output_kind, output_units = declaration.resolve(self.output)
        else:
            output_kind = output_units = None

        invalid_policy = (
            declaration.invalid_policy if declaration is not None else "propagate"
        )
        min_valid_weight = (
            declaration.min_valid_weight if declaration is not None else None
        )

        if getattr(operation, "spherical_operation", False):
            if _is_torch_tensor(source):
                raise TypeError(
                    "registered functional spherical operations currently use NumPy; "
                    "use panorai.ops.spherical.nn for Torch tensors"
                )
            if (
                input_field.kind in {"labels", "mask"}
                and getattr(operation, "interpolation", None) != "nearest"
            ):
                raise ValueError(
                    "labels and masks require a sphere operation with "
                    "interpolation='nearest'"
                )
            apply_with_masks = getattr(operation, "apply_with_masks", None)
            if not callable(apply_with_masks):
                if invalid_policy != "propagate":
                    raise ValueError(
                        f"{type(operation).__name__} does not define validity "
                        "renormalization"
                    )
                result = operation(source)
                support = operation.transform_mask(input_field.support)
                validity = operation.transform_mask(input_field.validity)
            else:
                result, support, validity = apply_with_masks(
                    source,
                    input_field.support,
                    input_field.validity,
                    invalid_policy=invalid_policy,
                    min_valid_weight=min_valid_weight,
                )
            transform_grid = getattr(operation, "transform_grid", None)
            grid_transformer = transform_grid
            operation_description = operation.describe()
            preserves_shadow = bool(getattr(operation, "preserves_shadow_state", True))
            validation = None
        elif _is_torch_module(operation):
            from panorai.ops.spherical import nn as spherical_nn

            if input_field.kind in {"labels", "mask"}:
                raise TypeError(
                    "Torch OnSphere modules do not accept discrete labels or masks"
                )
            validation_report = spherical_nn.validate(
                operation, input_shape=input_shape
            )
            if not _is_torch_tensor(source):
                raise TypeError(
                    "a Torch OnSphere module requires a Torch-backed panorama"
                )
            if source.ndim == 3:
                model_input = source.unsqueeze(0)
                result, support, validity = spherical_nn.execute_with_masks(
                    operation,
                    model_input,
                    input_field.support,
                    input_field.validity,
                )
                if not _is_torch_tensor(result) or result.ndim < 1:
                    raise TypeError("a spherical Torch module must return one tensor")
                result = result.squeeze(0)
            elif source.ndim == 4:
                result, support, validity = spherical_nn.execute_with_masks(
                    operation,
                    source,
                    input_field.support,
                    input_field.validity,
                )
            else:
                raise ValueError(
                    "Torch OnSphere modules require CHW or NCHW image layout"
                )
            if not _is_torch_tensor(result):
                raise TypeError("a spherical Torch module must return one tensor")
            operation_description = {
                "type": type(operation).__name__,
                "validation": validation_report.to_dict(),
            }
            validation = validation_report.to_dict()
            if invalid_policy != "propagate":
                raise ValueError(
                    "general learned Torch modules support only strict validity "
                    "propagation; arbitrary learned weights cannot be renormalized"
                )

            def grid_transformer(grid: Any) -> Any:
                return spherical_nn.transform_grid(operation, grid)

            preserves_shadow = True
        else:
            raise TypeError(
                "OnSphere accepts a registered panorai.ops.spherical operation "
                "or a validated Torch module"
            )

        non_raster = declaration is not None and output_kind in NON_RASTER_KINDS
        output_shape = None
        if not non_raster and (
            (_is_torch_tensor(result) and getattr(result, "ndim", 0) >= 3)
            or (isinstance(result, np.ndarray) and getattr(result, "ndim", 0) >= 2)
        ):
            output_shape = _spatial_shape(result)
        if output_shape is not None:
            if not callable(grid_transformer):
                raise TypeError(
                    f"registered spherical operation {type(operation).__name__} "
                    "must implement transform_grid(input_grid)"
                )
            output_grid = grid_transformer(input_field.grid)
        if output_shape is not None and output_grid.shape_hw != output_shape:
            raise ValueError(
                f"{type(operation).__name__}.transform_grid declared shape "
                f"{output_grid.shape_hw}, but execution returned {output_shape}"
            )
        grid = output_grid.describe() if output_grid is not None else None
        details = {
            "domain": "sphere",
            "input": self.input,
            "output": self.output,
            "operation": operation_description,
            "grid": grid,
        }
        if validation is not None:
            details["validation"] = validation

        if declaration is not None:
            if output_kind in NON_RASTER_KINDS:
                details["result_kind"] = output_kind
                if output_kind == "classification":
                    result = ClassificationResult(result)
                elif output_kind == "embedding":
                    result = EmbeddingResult(result)
                return StageResult(result, details)
            if output_shape is None:
                raise ValueError(
                    "typed raster OnSphere output must have a spatial grid"
                )
            field = RasterField(
                name=self.output,
                values=result,
                kind=output_kind,
                units=output_units,
                support=support,
                validity=validity,
                grid=output_grid,
                invalid_policy=declaration.invalid_policy,
            )
            details["field"] = field.describe()
            return StageResult(field, details, grid=field.grid)

        if self.output != "image":
            if self.output not in FIELD_KINDS:
                return StageResult(result, details)
            if self.output == "depth":
                raise ValueError(
                    "OnSphere output='depth' requires Output(..., kind='depth', "
                    "units=...)"
                )
            if output_shape is None:
                raise ValueError(
                    "typed raster OnSphere output must have a spatial grid"
                )
            field = RasterField(
                name=self.output,
                values=result,
                kind=self.output,
                support=support,
                validity=validity,
                grid=output_grid,
            )
            details["field"] = field.describe()
            return StageResult(field, details, grid=field.grid)
        if self.input != "image":
            raise ValueError(
                "plain OnSphere output='image' requires input='image'; wrap the "
                "operation in Output(...) to declare another typed field"
            )
        if output_shape != input_shape and not output_grid.is_canonical_erp:
            raise ValueError(
                "a shape-changing Torch model produces a derived lattice; set "
                "output to a non-image name or finish with a spherical Resize"
            )
        if tuple(panorama._require_workflow()) == ("image",):
            wrapped = _wrap_image_result(
                panorama,
                result,
                support=support,
                validity=validity,
                preserves_shadow_state=preserves_shadow,
            )
        else:
            if output_shape != input_shape:
                raise ValueError(
                    "shape-changing OnSphere cannot update one field of a multimodal "
                    "ERP; select one field or resize all aligned fields"
                )
            field = RasterField(
                name="image",
                values=result,
                kind="image",
                units=None,
                support=support,
                validity=validity,
                grid=input_field.grid,
            )
            wrapped = attach_raster_field(panorama, "image", field, replace=True)
        return StageResult(wrapped, details)


@dataclass(frozen=True, slots=True)
class SampleViews:
    """Create canonical gnomonic views through ``EquirectangularImage.views``."""

    layout: str = "cube"
    size: Any = None
    fov: Any = 90.0
    count: int | None = None
    subdivisions: int = 0
    rotations: tuple[Any, ...] = ()
    depth_policy: str = "propagate"
    min_valid_weight: float | None = None
    projector: Any = None

    input_kind = "erp"
    output_kind = "views"

    def execute(self, value: Any) -> StageResult:
        if not _has_methods(value, "_require_workflow", "views"):
            raise TypeError("SampleViews requires an EquirectangularImage input")
        faces = value.views(
            self.layout,
            size=self.size,
            fov=self.fov,
            count=self.count,
            subdivisions=self.subdivisions,
            rotations=self.rotations,
            depth_policy=self.depth_policy,
            min_valid_weight=self.min_valid_weight,
            projector=self.projector,
        )
        return StageResult(
            faces,
            {
                "domain": "projection",
                "transition": "erp-to-gnomonic",
                "workflow": faces.describe(),
            },
        )


@dataclass(frozen=True, slots=True)
class PerView:
    """Apply one planar callable using the Stable ``GnomonicFaceSet.map``."""

    operation: Any
    input: str | None = None
    output: str | None = None
    units: str | None = None
    replace: bool = False

    input_kind = "views"

    @property
    def output_kind(self) -> str:
        if isinstance(self.operation, Output):
            if self.operation.kind in NON_RASTER_KINDS:
                return "view_results"
        if self.output in NON_RASTER_KINDS:
            return "view_results"
        return "views"

    @classmethod
    def parallel(
        cls,
        *,
        input: str | None = None,
        replace: bool = False,
        batch: int | str | None = None,
        max_batch_elements: int = 16_777_216,
        **operations: Any,
    ) -> "PerViewParallel":
        """Build a logical multi-head stage over the same original views.

        Branches execute sequentially in declaration order.  Each branch sees
        the selected field from the input face set, rather than the output of a
        preceding branch.
        """

        return PerViewParallel(
            operations,
            input=input,
            replace=replace,
            batch=batch,
            max_batch_elements=max_batch_elements,
        )

    def execute(self, value: Any) -> StageResult:
        if not _has_methods(value, "_require_workflow", "map", "describe"):
            raise TypeError("PerView requires a GnomonicFaceSet input")
        output_name = self.output
        if self.output_kind == "view_results":
            return _map_non_raster_view_results(
                value,
                self.operation,
                input_name=self.input,
                output_name=output_name,
                units=self.units,
            )
        if isinstance(self.operation, Output) or (
            output_name is not None and output_name not in {"image", "depth", "labels"}
        ):
            resolved_name = output_name
            if resolved_name is None:
                if (
                    isinstance(self.operation, Output)
                    and self.operation.kind is not None
                ):
                    resolved_name = self.operation.kind
                else:
                    raise ValueError(
                        "typed PerView output requires an explicit output name"
                    )
            result, field = _map_typed_view_field(
                value,
                self.operation,
                input_name=self.input,
                output_name=resolved_name,
                units=self.units,
                replace=self.replace,
            )
            return StageResult(
                result,
                {
                    "domain": "views",
                    "operation": type(
                        self.operation.operation
                        if isinstance(self.operation, Output)
                        else self.operation
                    ).__name__,
                    "input": self.input,
                    "output": resolved_name,
                    "field": field.describe(),
                    "workflow": result.describe(),
                },
            )
        result = value.map(
            self.operation,
            input=self.input,
            output=self.output,
            units=self.units,
            replace=self.replace,
        )
        return StageResult(
            result,
            {
                "domain": "views",
                "operation": type(self.operation).__name__,
                "input": self.input,
                "output": self.output,
                "workflow": result.describe(),
            },
        )


def _map_non_raster_view_results(
    value: Any,
    operation: Any,
    *,
    input_name: str | None,
    output_name: str | None,
    units: str | None,
) -> StageResult:
    selected = _selected_input(value, input_name)
    if isinstance(operation, Output):
        name = output_name or operation.kind
        if name is None:
            raise ValueError("non-raster Output requires a kind or output name")
        kind, resolved_units = operation.resolve(name)
        callable_operation = operation.operation
    else:
        if output_name not in NON_RASTER_KINDS:
            raise ValueError(
                f"non-raster PerView output must be one of {NON_RASTER_KINDS}"
            )
        kind = output_name
        resolved_units = units
        callable_operation = operation
    if kind not in NON_RASTER_KINDS:
        raise ValueError(f"non-raster PerView kind must be one of {NON_RASTER_KINDS}")
    if resolved_units is not None or units is not None:
        raise ValueError("units are valid only for radial-range depth")
    items = []
    for face in value:
        result = callable_operation(face._workflow_data()[selected])
        items.append(
            ViewResult(
                face_id=str(face.face_id),
                spec=face.spec,
                value=result,
                support=_clone(face._workflow_support[selected]),
                validity=_clone(face._workflow_metadata[selected]["validity"]),
            )
        )
    common = {
        "items": items,
        "source_field": selected,
        "erp_shape_hw": tuple(value._require_workflow()["erp_shape"]),
    }
    if kind == "classification":
        results = ClassificationResult(**common)
    elif kind == "embedding":
        results = EmbeddingResult(**common)
    else:
        results = PerViewResults(kind, **common)
    return StageResult(
        results,
        {
            "domain": "views",
            "operation": type(callable_operation).__name__,
            "input": selected,
            "output": output_name or kind,
            "result": results.describe(),
        },
    )


def _selected_input(value: Any, requested: str | None) -> str:
    metadata = value[0]._workflow_metadata
    available = list(metadata)
    if requested is None:
        if len(available) != 1:
            raise ValueError(
                "input is required when mapping a multimodal face set; "
                f"choose one of {available}"
            )
        return available[0]
    if requested not in metadata:
        raise KeyError(f"unknown input raster field {requested!r}")
    return requested


def _batch_size_for_operation(
    operation: Any,
    source: Any,
    requested: int | str | None,
    max_batch_elements: int,
) -> int | None:
    if requested is None:
        return None
    if isinstance(requested, bool):
        raise TypeError("batch must be None, a positive integer, or 'auto'")
    if isinstance(requested, int):
        if requested <= 0:
            raise ValueError("batch integers must be positive")
        return requested
    if requested != "auto":
        raise ValueError("batch must be None, a positive integer, or 'auto'")
    if not (_is_torch_module(operation) or getattr(operation, "supports_batch", False)):
        raise TypeError(
            "batch='auto' requires a Torch module or a callable declaring "
            "supports_batch=True"
        )
    element_count = (
        int(source.numel()) if _is_torch_tensor(source) else int(source.size)
    )
    return max(1, max_batch_elements // max(1, element_count))


def _stack_batch(values: list[Any]) -> Any:
    if _is_torch_tensor(values[0]):
        import torch

        if not all(_is_torch_tensor(item) for item in values):
            raise TypeError("batched per-view inputs must share one backend")
        return torch.stack(values, dim=0)
    if any(_is_torch_tensor(item) for item in values):
        raise TypeError("batched per-view inputs must share one backend")
    return np.stack(values, axis=0)


def _split_batch_value(value: Any, count: int, *, output_name: str) -> list[Any]:
    if isinstance(value, (list, tuple)):
        if len(value) != count:
            raise ValueError(
                f"batched output {output_name!r} returned {len(value)} items for "
                f"a chunk of {count}"
            )
        return list(value)
    if _is_torch_tensor(value) or isinstance(value, np.ndarray):
        if getattr(value, "ndim", 0) == 0 or int(value.shape[0]) != count:
            raise ValueError(
                f"batched output {output_name!r} must have leading dimension "
                f"{count}; got {getattr(value, 'shape', None)}"
            )
        return [value[index] for index in range(count)]
    raise TypeError(
        f"batched output {output_name!r} must be an array/tensor with a batch "
        "dimension or a sequence with one item per view"
    )


def _execute_across_views(
    value: Any,
    operation: Any,
    *,
    selected: str,
    output_name: str,
    batch: int | str | None,
    max_batch_elements: int,
) -> tuple[list[Any], int | None]:
    sources = [face._workflow_data()[selected] for face in value]
    chunk_size = _batch_size_for_operation(
        operation,
        sources[0],
        batch,
        max_batch_elements,
    )
    if chunk_size is None:
        return [operation(source) for source in sources], None
    results: list[Any] = []
    for offset in range(0, len(sources), chunk_size):
        chunk = sources[offset : offset + chunk_size]
        raw = operation(_stack_batch(chunk))
        if isinstance(raw, tuple) and len(raw) == 2:
            data, validity = raw
            split_data = _split_batch_value(data, len(chunk), output_name=output_name)
            split_validity = _split_batch_value(
                validity,
                len(chunk),
                output_name=f"{output_name}.validity",
            )
            results.extend(zip(split_data, split_validity, strict=True))
        else:
            results.extend(_split_batch_value(raw, len(chunk), output_name=output_name))
    return results, chunk_size


def _resolved_output(
    operation: Any,
    output_name: str,
    units: str | None,
) -> tuple[Any, str, str | None, str]:
    if isinstance(operation, Output):
        if (
            units is not None
            and operation.units is not None
            and units != operation.units
        ):
            raise ValueError("PerView units conflict with Output units")
        declaration = Output(
            operation.operation,
            kind=operation.kind,
            units=operation.units if operation.units is not None else units,
            invalid_policy=operation.invalid_policy,
        )
        kind, declared_units = declaration.resolve(output_name)
        return operation.operation, kind, declared_units, operation.invalid_policy
    declaration = Output(
        operation,
        kind=(
            output_name
            if output_name
            in {
                "depth",
                "labels",
                "mask",
                "logits",
                "features",
                "confidence",
                "image",
                *NON_RASTER_KINDS,
            }
            else None
        ),
        units=units,
    )
    kind, declared_units = declaration.resolve(output_name)
    return operation, kind, declared_units, declaration.invalid_policy


def _map_typed_view_field(
    value: Any,
    operation: Any,
    *,
    input_name: str | None,
    output_name: str,
    units: str | None,
    replace: bool,
    batch: int | str | None = None,
    max_batch_elements: int = 16_777_216,
) -> tuple[Any, RasterField]:
    from panorai.data._workflow import (
        backend,
        batch_size,
        clone_array,
        device,
        spatial_shape,
        support_for_value,
        validate_mask,
    )

    if not _has_methods(value, "_require_workflow", "describe") or not hasattr(
        value, "_faces"
    ):
        raise TypeError("PerView requires a GnomonicFaceSet input")
    selected = _selected_input(value, input_name)
    callable_operation, kind, resolved_units, invalid_policy = _resolved_output(
        operation, output_name, units
    )
    workflow = value._require_workflow()
    if kind == "depth" and invalid_policy != workflow["depth_policy"]:
        raise ValueError(
            "depth Output invalid_policy must match SampleViews depth_policy; "
            f"got {invalid_policy!r} and {workflow['depth_policy']!r}"
        )
    values = []
    supports = []
    validities = []
    raw_results, _ = _execute_across_views(
        value,
        callable_operation,
        selected=selected,
        output_name=output_name,
        batch=batch,
        max_batch_elements=max_batch_elements,
    )
    for face, raw_result in zip(value, raw_results, strict=True):
        source = face._workflow_data()[selected]
        if isinstance(raw_result, tuple):
            if len(raw_result) != 2:
                raise ValueError(
                    "model tuple results must contain exactly (array, validity_mask)"
                )
            result, validity = raw_result
        else:
            result, validity = raw_result, None
        expected = (
            backend(source),
            device(source),
            batch_size(source),
            spatial_shape(source),
        )
        actual = (
            backend(result),
            device(result),
            batch_size(result),
            spatial_shape(result),
        )
        if actual != expected:
            raise ValueError(
                "typed PerView outputs must preserve backend, device, batch, and "
                f"the declared gnomonic grid; expected {expected}, got {actual}"
            )
        support = support_for_value(face._workflow_support[selected], result)
        valid = (
            clone_array(face._workflow_metadata[selected]["validity"] & support)
            if validity is None
            else validate_mask(result, validity)
        )
        values.append(clone_array(result))
        supports.append(clone_array(support))
        validities.append(clone_array(valid & support))
    input_field = field_from_container(value, selected)
    field = RasterField(
        name=output_name,
        values=tuple(values),
        kind=kind,
        units=resolved_units,
        support=tuple(supports),
        validity=tuple(validities),
        grid=input_field.grid,
        invalid_policy=invalid_policy,
    )
    attached = attach_raster_field(
        value,
        output_name,
        field,
        replace=replace or output_name == selected,
    )
    return attached, field


@dataclass(frozen=True, slots=True, init=False)
class PerViewParallel:
    """Internal stage returned by :meth:`PerView.parallel`."""

    operations: Mapping[str, Any]
    input: str | None
    replace: bool
    batch: int | str | None
    max_batch_elements: int

    input_kind = "views"
    output_kind = "bundle"
    stage_name = "per_view"

    def __init__(
        self,
        operations: Mapping[str, Any],
        *,
        input: str | None = None,
        replace: bool = False,
        batch: int | str | None = None,
        max_batch_elements: int = 16_777_216,
    ) -> None:
        if not operations:
            raise ValueError("PerView.parallel requires at least one named operation")
        if not isinstance(replace, bool):
            raise TypeError("replace must be boolean")
        if isinstance(max_batch_elements, bool) or not isinstance(
            max_batch_elements, int
        ):
            raise TypeError("max_batch_elements must be a positive integer")
        if max_batch_elements <= 0:
            raise ValueError("max_batch_elements must be positive")
        if batch is not None and (
            (isinstance(batch, bool))
            or (not isinstance(batch, int) and batch != "auto")
            or (isinstance(batch, int) and batch <= 0)
        ):
            raise ValueError("batch must be None, a positive integer, or 'auto'")
        normalized: dict[str, Any] = {}
        for name, operation in operations.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("parallel output names must be non-empty strings")
            callable_operation = (
                operation.operation if isinstance(operation, Output) else operation
            )
            if not callable(callable_operation):
                raise TypeError("PerView.parallel operations must be callable")
            normalized[name] = operation
        object.__setattr__(self, "operations", MappingProxyType(normalized))
        object.__setattr__(self, "input", input)
        object.__setattr__(self, "replace", replace)
        object.__setattr__(self, "batch", batch)
        object.__setattr__(self, "max_batch_elements", max_batch_elements)

    def execute(self, value: Any) -> StageResult:
        base = value
        result = value
        branch_outputs: dict[str, Any] = {}
        descriptions: dict[str, Any] = {}
        for name, operation in self.operations.items():
            callable_operation, kind, _, _ = _resolved_output(operation, name, None)
            if kind in NON_RASTER_KINDS:
                selected = _selected_input(base, self.input)
                raw_results, chunk_size = _execute_across_views(
                    base,
                    callable_operation,
                    selected=selected,
                    output_name=name,
                    batch=self.batch,
                    max_batch_elements=self.max_batch_elements,
                )
                items = tuple(
                    ViewResult(
                        face_id=str(face.face_id),
                        spec=face.spec,
                        value=raw,
                        support=_clone(face._workflow_support[selected]),
                        validity=_clone(face._workflow_metadata[selected]["validity"]),
                    )
                    for face, raw in zip(base, raw_results, strict=True)
                )
                common = {
                    "items": items,
                    "source_field": selected,
                    "erp_shape_hw": tuple(base._require_workflow()["erp_shape"]),
                }
                if kind == "classification":
                    branch = ClassificationResult(**common)
                elif kind == "embedding":
                    branch = EmbeddingResult(**common)
                else:
                    branch = PerViewResults(kind, **common)
                branch_outputs[name] = branch
                descriptions[name] = {
                    "kind": kind,
                    "batch_size": chunk_size,
                    "result": branch.describe(),
                }
            else:
                _, field = _map_typed_view_field(
                    base,
                    operation,
                    input_name=self.input,
                    output_name=name,
                    units=None,
                    replace=self.replace,
                    batch=self.batch,
                    max_batch_elements=self.max_batch_elements,
                )
                result = attach_raster_field(
                    result,
                    name,
                    field,
                    replace=self.replace or name == self.input,
                )
                branch_outputs[name] = field
                descriptions[name] = field.describe()
        bundle = ResultBundle(
            {"views": result, **branch_outputs},
            descriptions=descriptions,
        )
        return StageResult(
            bundle,
            {
                "domain": "views",
                "execution": "sequential-logical-branches",
                "input": self.input,
                "batch": self.batch,
                "max_batch_elements": self.max_batch_elements,
                "outputs": descriptions,
                "workflow": result.describe(),
            },
        )


@dataclass(frozen=True, slots=True, init=False)
class ReconstructERP:
    """Reconstruct selected view fields with the Stable workflow implementation."""

    blend: Any = None
    modalities: Any = "all"
    eq_shape: tuple[int, int] | None = None
    field_blends: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )

    input_kind = ("views", "bundle")
    output_kind = "any"

    def __init__(
        self,
        blend: Any = None,
        modalities: Any = "all",
        eq_shape: tuple[int, int] | None = None,
        **field_blends: Any,
    ) -> None:
        if field_blends:
            if modalities != "all":
                raise ValueError(
                    "direct field blends already select modalities; do not also "
                    "pass modalities"
                )
            if blend is None:
                resolved_blend = dict(field_blends)
            elif isinstance(blend, Mapping):
                overlap = set(blend) & set(field_blends)
                if overlap:
                    raise ValueError(
                        f"duplicate blend declarations for {sorted(overlap)}"
                    )
                resolved_blend = {**dict(blend), **field_blends}
            else:
                raise ValueError(
                    "a scalar blend cannot be combined with direct field blends"
                )
            resolved_modalities: Any = tuple(resolved_blend)
        else:
            resolved_blend = blend
            resolved_modalities = modalities
        if isinstance(resolved_blend, Mapping):
            resolved_blend = MappingProxyType(dict(resolved_blend))
        object.__setattr__(self, "blend", resolved_blend)
        object.__setattr__(self, "modalities", resolved_modalities)
        object.__setattr__(self, "eq_shape", eq_shape)
        object.__setattr__(self, "field_blends", MappingProxyType(dict(field_blends)))

    def execute(self, value: Any) -> StageResult:
        bundle = value if isinstance(value, ResultBundle) else None
        views = value["views"] if bundle is not None and "views" in value else value
        if not _has_methods(views, "_require_workflow", "reconstruct"):
            raise TypeError("ReconstructERP requires a GnomonicFaceSet input")
        result = views.reconstruct(
            self.eq_shape,
            blend=self.blend,
            modalities=self.modalities,
        )
        output = (
            ResultBundle({**dict(bundle), "erp": result})
            if bundle is not None
            else result
        )
        return StageResult(
            output,
            {
                "domain": "projection",
                "transition": "gnomonic-to-erp",
                "blend": self.blend,
                "modalities": self.modalities,
                "output_shape_hw": _spatial_shape(
                    next(iter(result._workflow_data().values()))
                ),
            },
        )


def _mask_at_pixels(mask: Any, pixels_xy: np.ndarray) -> np.ndarray:
    array = mask
    if _is_torch_tensor(array):
        array = array.detach().cpu().numpy()
    array = np.asarray(array, dtype=bool)
    if array.ndim != 2:
        raise ValueError(
            "observation lifting currently requires one unbatched 2D support or "
            "validity mask per view"
        )
    height, width = array.shape[-2:]
    finite = np.isfinite(pixels_xy).all(axis=1)
    indices = np.zeros_like(pixels_xy, dtype=np.int64)
    indices[finite] = np.floor(pixels_xy[finite] + 0.5).astype(np.int64)
    inside = (
        finite
        & (indices[:, 0] >= 0)
        & (indices[:, 0] < width)
        & (indices[:, 1] >= 0)
        & (indices[:, 1] < height)
    )
    result = np.zeros(len(pixels_xy), dtype=bool)
    if inside.any():
        result[inside] = array[indices[inside, 1], indices[inside, 0]]
    return result


@dataclass(frozen=True, slots=True)
class LiftToBearings:
    """Lift detector-neutral view geometry through canonical gnomonic rays."""

    require_footprint_valid: bool = True

    input_kind = "view_results"
    output_kind = "observations"

    def __post_init__(self) -> None:
        if not isinstance(self.require_footprint_valid, bool):
            raise TypeError("require_footprint_valid must be boolean")

    def execute(self, value: Any) -> StageResult:
        if not isinstance(value, PerViewResults):
            raise TypeError("LiftToBearings requires PerViewResults input")
        if value.kind not in {"detections", "keypoints"}:
            raise TypeError(
                f"LiftToBearings accepts detections or keypoints; got {value.kind!r}"
            )
        from panorai.geometry import gnomonic_pixels_to_rays

        bearings = []
        footprints = []
        scores = []
        supports = []
        data_validities = []
        valid = []
        face_ids = []
        source_indices = []
        source_pixels = []
        labels = []
        features = []
        feature_shape = None
        feature_dtype = None
        metadata = []
        geometry = None
        has_labels = None
        has_features = None
        for item in value:
            observations = coerce_view_observations(item.value)
            if geometry is None:
                geometry = observations.geometry
                has_labels = observations.labels is not None
                has_features = observations.features is not None
            elif observations.geometry != geometry:
                raise ValueError("all views must return the same observation geometry")
            elif (observations.labels is not None) != has_labels:
                raise ValueError("labels must be present consistently across views")
            elif (observations.features is not None) != has_features:
                raise ValueError("features must be present consistently across views")
            if observations.features is not None:
                current_shape = observations.features.shape[1:]
                if feature_shape is None:
                    feature_shape = current_shape
                    feature_dtype = observations.features.dtype
                elif current_shape != feature_shape:
                    raise ValueError(
                        "observation feature shapes must match across views"
                    )
                elif observations.features.dtype != feature_dtype:
                    raise TypeError(
                        "observation feature dtypes must match across views"
                    )
            for local_index, pixels in enumerate(observations.pixels_xy):
                projected = gnomonic_pixels_to_rays(pixels, item.spec)
                footprint = np.asarray(projected.rays_xyz, dtype=np.float64)
                centre_pixels = np.mean(pixels, axis=0, keepdims=True)
                centre_projection = gnomonic_pixels_to_rays(centre_pixels, item.spec)
                centre = np.asarray(centre_projection.rays_xyz[0], dtype=np.float64)
                point_support = _mask_at_pixels(item.support, pixels)
                point_validity = _mask_at_pixels(item.validity, pixels)
                centre_support = _mask_at_pixels(item.support, centre_pixels)
                centre_validity = _mask_at_pixels(item.validity, centre_pixels)
                geometry_valid = bool(np.asarray(projected.valid).all())
                centre_valid = bool(np.asarray(centre_projection.valid)[0])
                support_valid = (
                    bool(point_support.all())
                    if self.require_footprint_valid
                    else bool(centre_support[0])
                )
                data_valid = (
                    bool(point_validity.all())
                    if self.require_footprint_valid
                    else bool(centre_validity[0])
                )
                has_support = bool(geometry_valid and centre_valid and support_valid)
                is_valid = bool(
                    observations.valid[local_index] and has_support and data_valid
                )
                bearings.append(centre)
                footprints.append(footprint)
                scores.append(observations.scores[local_index])
                supports.append(has_support)
                data_validities.append(data_valid)
                valid.append(is_valid)
                face_ids.append(item.face_id)
                source_indices.append(local_index)
                source_pixels.append(pixels)
                metadata.append(observations.metadata[local_index])
                if has_labels:
                    labels.append(observations.labels[local_index])
                if has_features:
                    features.append(observations.features[local_index])
        observation_count = len(bearings)
        result = SphericalObservations(
            np.asarray(bearings, dtype=np.float64).reshape(observation_count, 3),
            footprints_xyz=tuple(footprints),
            scores=np.asarray(scores, dtype=np.float64),
            support=np.asarray(supports, dtype=bool),
            data_validity=np.asarray(data_validities, dtype=bool),
            valid=np.asarray(valid, dtype=bool),
            source_face_ids=np.asarray(face_ids, dtype=object),
            source_indices=np.asarray(source_indices, dtype=np.int64),
            source_pixels_xy=tuple(source_pixels),
            geometry=geometry or "points",
            labels=np.asarray(labels) if has_labels else None,
            features=(
                np.asarray(features)
                if features
                else (
                    np.empty((0, *feature_shape), dtype=feature_dtype)
                    if has_features
                    else None
                )
            ),
            metadata=tuple(metadata),
        )
        return StageResult(
            result,
            {
                "domain": "geometry",
                "transition": "gnomonic-pixels-to-panorama-bearings",
                "source": value.describe(),
                "observations": result.describe(),
                "require_footprint_valid": self.require_footprint_valid,
            },
        )


def _aggregate_view_values(value: Any, reducer: str) -> Any:
    if not isinstance(value, (ClassificationResult, EmbeddingResult)):
        raise TypeError(
            "string reducers accept ClassificationResult or EmbeddingResult; "
            "lift detections/keypoints before spherical aggregation"
        )
    if not value.is_per_view:
        raise ValueError("AggregateViews requires a per-view typed result")
    values = value.values
    first = values[0]
    if _is_torch_tensor(first):
        import torch

        if not all(_is_torch_tensor(item) for item in values):
            raise TypeError("all aggregated view results must share one backend")
        stacked = torch.stack(values)
        if reducer == "mean":
            return stacked.mean(dim=0)
        if reducer == "sum":
            return stacked.sum(dim=0)
        if reducer == "max":
            return stacked.max(dim=0).values
        if reducer == "vote":
            if stacked.numel() != len(values):
                raise ValueError("vote requires one scalar class identifier per view")
            return torch.mode(stacked.reshape(-1)).values
    elif any(_is_torch_tensor(item) for item in values):
        raise TypeError("all aggregated view results must share one backend")
    else:
        stacked = np.stack([np.asarray(item) for item in values])
        if reducer == "mean":
            return stacked.mean(axis=0)
        if reducer == "sum":
            return stacked.sum(axis=0)
        if reducer == "max":
            return stacked.max(axis=0)
        if reducer == "vote":
            if stacked.size != len(values):
                raise ValueError("vote requires one scalar class identifier per view")
            labels, counts = np.unique(stacked.reshape(-1), return_counts=True)
            return labels[np.argmax(counts)]
    raise ValueError("reducer must be 'mean', 'sum', 'max', or 'vote'")


@dataclass(frozen=True, slots=True)
class AggregateViews:
    """Aggregate per-view global results or spherical observations."""

    reducer: Any = "mean"

    input_kind = "any"
    output_kind = "result"

    def __post_init__(self) -> None:
        if not isinstance(self.reducer, str) and not callable(self.reducer):
            raise TypeError("AggregateViews reducer must be a name or callable")

    def execute(self, value: Any) -> StageResult:
        if isinstance(self.reducer, str):
            result_value = _aggregate_view_values(value, self.reducer)
            result_type = (
                ClassificationResult
                if isinstance(value, ClassificationResult)
                else EmbeddingResult
            )
            result = result_type(result_value)
            reducer_name = self.reducer
        else:
            result = self.reducer(value)
            reducer_name = type(self.reducer).__name__
        return StageResult(
            result,
            {
                "domain": "aggregation",
                "reducer": reducer_name,
                "input_type": type(value).__name__,
                "output_type": type(result).__name__,
                "input_count": len(value) if hasattr(value, "__len__") else None,
                "output_count": len(result) if hasattr(result, "__len__") else None,
            },
        )


@dataclass(frozen=True, slots=True)
class SphericalWatershed:
    """Partition an ERP from named gradient and marker fields."""

    gradient: str = "features"
    markers: str = "labels"
    output: str = "labels"
    connectivity: int = 8

    input_kind = "erp"
    output_kind = "result"
    interface = "panorai-spherical-region-analysis/v1-experimental"
    stability = "experimental"

    def __post_init__(self) -> None:
        for name in ("gradient", "markers", "output"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if isinstance(self.connectivity, bool) or self.connectivity not in {4, 8}:
            raise ValueError("connectivity must be 4 or 8")

    def execute(self, value: Any) -> StageResult:
        panorama, gradient = _erp_input(value, self.gradient)
        _, markers = _erp_input(panorama, self.markers)
        if gradient.grid != markers.grid:
            raise ValueError("gradient and marker fields must share one RasterGrid")
        if _is_torch_tensor(gradient.values) or _is_torch_tensor(markers.values):
            raise TypeError("SphericalWatershed is NumPy-canonical")
        support = np.asarray(gradient.support) & np.asarray(markers.support)
        validity = np.asarray(gradient.validity) & np.asarray(markers.validity)
        from panorai.ops.spherical import watershed

        labels = watershed(
            gradient.values,
            markers.values,
            support=support,
            validity=validity,
            connectivity=self.connectivity,
        )
        output_grid = gradient.grid.with_transform(
            operation="watershed",
            output_shape_hw=gradient.grid.shape_hw,
            domain=gradient.grid.domain,
            alignment=gradient.grid.alignment,
            first_center_px_yx=gradient.grid.first_center_px_yx,
            pixel_stride_yx=gradient.grid.pixel_stride_yx,
            parameters={
                "gradient": self.gradient,
                "markers": self.markers,
                "connectivity": self.connectivity,
            },
        )
        field = RasterField(
            name=self.output,
            values=labels,
            kind="labels",
            support=support,
            validity=validity,
            grid=output_grid,
        )
        return StageResult(
            field,
            {
                "domain": "sphere",
                "operation": "watershed",
                "gradient": self.gradient,
                "markers": self.markers,
                "output": self.output,
                "connectivity": self.connectivity,
                "grid": output_grid.describe(),
            },
            grid=output_grid,
        )


@dataclass(frozen=True, slots=True)
class AnalyzeRegions:
    """Convert a named ERP label field into typed region evidence."""

    input: str = "labels"
    include_boundaries: bool = True

    input_kind = "result"
    output_kind = "result"
    interface = "panorai-spherical-region-analysis/v1-experimental"
    stability = "experimental"

    def __post_init__(self) -> None:
        if not isinstance(self.input, str) or not self.input.strip():
            raise ValueError("input must be a non-empty string")
        if not isinstance(self.include_boundaries, bool):
            raise TypeError("include_boundaries must be boolean")

    def execute(self, value: Any) -> StageResult:
        if isinstance(value, RasterField):
            field = value
            if field.name != self.input:
                raise KeyError(
                    f"AnalyzeRegions expected field {self.input!r}, "
                    f"received {field.name!r}"
                )
        else:
            _, field = _erp_input(value, self.input)
        if field.kind != "labels":
            raise TypeError("AnalyzeRegions requires a labels field")
        if _is_torch_tensor(field.values):
            raise TypeError("AnalyzeRegions is NumPy-canonical")
        from panorai.data import RegionAnalysisResult
        from panorai.ops.spherical import (
            extract_boundaries,
            measure_regions,
            region_adjacency_graph,
        )

        measurements = measure_regions(
            field.values,
            support=field.support,
            validity=field.validity,
        )
        adjacency = region_adjacency_graph(
            field.values,
            support=field.support,
            validity=field.validity,
        )
        boundaries = (
            extract_boundaries(
                field.values,
                support=field.support,
                validity=field.validity,
            )
            if self.include_boundaries
            else None
        )
        result = RegionAnalysisResult(measurements, adjacency, boundaries)
        return StageResult(
            result,
            {
                "domain": "sphere",
                "operation": "analyze_regions",
                "input": self.input,
                "region_count": len(measurements.regions),
                "include_boundaries": self.include_boundaries,
                "grid": field.grid.describe(),
            },
        )
