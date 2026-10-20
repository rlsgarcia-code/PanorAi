"""Generic execution of symbol-level plans derived from public PanorAi APIs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import fields, is_dataclass
from enum import Enum
import inspect
from pathlib import Path
import types
from typing import Annotated, Any, Union, get_args, get_origin, get_type_hints

import numpy as np
from PIL import Image

from .discovery import resolve_component
from .rendering import render_result


def _unwrap(annotation: Any) -> Any:
    if get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    origin = get_origin(annotation)
    if origin in {Union, types.UnionType}:
        values = [item for item in get_args(annotation) if item is not type(None)]
        if len(values) == 1:
            return _unwrap(values[0])
    return annotation


def _dataclass_type(annotation: Any) -> type[Any] | None:
    annotation = _unwrap(annotation)
    if inspect.isclass(annotation) and is_dataclass(annotation):
        return annotation
    origin = get_origin(annotation)
    if origin in {Union, types.UnionType}:
        for item in get_args(annotation):
            candidate = _dataclass_type(item)
            if candidate is not None:
                return candidate
    return None


def _coerce(value: Any, annotation: Any) -> Any:
    annotation = _unwrap(annotation)
    if value is None:
        return None
    dataclass_type = _dataclass_type(annotation)
    if dataclass_type is not None and isinstance(value, dict):
        hints = get_type_hints(dataclass_type, include_extras=True)
        known = {item.name for item in fields(dataclass_type) if item.init}
        unknown = set(value) - known
        if unknown:
            raise ValueError(
                f"unknown {dataclass_type.__name__} fields: {', '.join(sorted(unknown))}"
            )
        return dataclass_type(
            **{
                name: _coerce(item, hints.get(name, Any))
                for name, item in value.items()
            }
        )
    if inspect.isclass(annotation) and issubclass(annotation, Enum):
        return annotation(value)
    if annotation is np.ndarray:
        return np.asarray(value)
    if annotation in {bool, int, float, str}:
        return annotation(value)
    origin = get_origin(annotation)
    if origin in {Union, types.UnionType}:
        for candidate in get_args(annotation):
            candidate_origin = get_origin(candidate)
            if isinstance(value, list) and candidate_origin is tuple:
                return _coerce(value, candidate)
            if (
                isinstance(value, dict)
                and inspect.isclass(candidate)
                and is_dataclass(candidate)
            ):
                return _coerce(value, candidate)
            if candidate in {bool, int, float, str} and isinstance(value, candidate):
                return value
    if origin in {list, tuple, Sequence, Iterable} and isinstance(value, (list, tuple)):
        arguments = get_args(annotation)
        item_type = arguments[0] if arguments else Any
        items = [_coerce(item, item_type) for item in value]
        return tuple(items) if origin is tuple else items
    return value


def _coerce_input(value: Any, annotation: Any, descriptor: dict[str, Any]) -> Any:
    result = _coerce(value, annotation)
    if descriptor.get("container") == "equirectangular-image" and isinstance(
        result, np.ndarray
    ):
        from panorai.data import EquirectangularImage

        return EquirectangularImage(result)
    return result


def _read_upload(item: dict[str, Any], kind: str) -> Any:
    path = Path(item["path"])
    if kind in {"erp-image", "image-list", "image-map", "mask"}:
        with Image.open(path) as image:
            image.load()
            if kind == "mask":
                return np.asarray(image.convert("L")) > 0
            return np.asarray(image.convert("RGB"))
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    return path.read_bytes()


def _nested_descriptor(descriptor: dict[str, Any], path: list[str]) -> dict[str, Any]:
    current = descriptor
    for name in path:
        fields_by_name = current.get("fields")
        if not isinstance(fields_by_name, dict) or name not in fields_by_name:
            raise ValueError(f"unknown nested upload field {name!r}")
        current = fields_by_name[name]
    return current


def _set_nested(target: dict[str, Any], path: list[str], value: Any) -> None:
    cursor = target
    for name in path[:-1]:
        nested = cursor.get(name)
        if not isinstance(nested, dict):
            nested = {}
            cursor[name] = nested
        cursor = nested
    cursor[path[-1]] = value


def _append_nested(target: dict[str, Any], path: list[str], value: Any) -> None:
    cursor = target
    for name in path[:-1]:
        nested = cursor.get(name)
        if not isinstance(nested, dict):
            nested = {}
            cursor[name] = nested
        cursor = nested
    current = cursor.get(path[-1])
    if current is None:
        cursor[path[-1]] = [value]
    elif isinstance(current, list):
        current.append(value)
    else:
        raise ValueError(f"input {'.'.join(path)!r} mixes files and scalar values")


def _bind_inputs(
    target: Any,
    descriptors: list[dict[str, Any]],
    uploads: list[dict[str, Any]],
    values: dict[str, Any],
) -> tuple[list[Any], dict[str, Any]]:
    signature = inspect.signature(target)
    try:
        hints = get_type_hints(target, include_extras=True)
    except (NameError, TypeError):
        hints = {}
    descriptor_by_name = {item["name"]: item for item in descriptors}
    resolved_values = deepcopy(values)
    for item in uploads:
        path = str(item.get("field")).split(".")
        descriptor = descriptor_by_name[path[0]]
        leaf = (
            descriptor
            if descriptor["kind"] == "image-map" and len(path) == 2
            else _nested_descriptor(descriptor, path[1:])
        )
        uploaded = _read_upload(item, leaf["kind"])
        if leaf["kind"] == "image-list":
            _append_nested(resolved_values, path, uploaded)
        else:
            _set_nested(resolved_values, path, uploaded)
    positional: list[Any] = []
    keyword: dict[str, Any] = {}
    for parameter in signature.parameters.values():
        if parameter.name in {"self", "cls"} or parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        descriptor = descriptor_by_name[parameter.name]
        if parameter.name in resolved_values:
            value = _coerce_input(
                resolved_values[parameter.name],
                hints.get(parameter.name, parameter.annotation),
                descriptor,
            )
        elif parameter.default is not inspect.Parameter.empty:
            continue
        else:
            raise ValueError(f"missing required input {parameter.name!r}")
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            positional.append(value)
        else:
            keyword[parameter.name] = value
    return positional, keyword


def _construct(owner: type[Any], configuration: dict[str, Any]) -> Any:
    signature = inspect.signature(owner)
    try:
        hints = get_type_hints(owner, include_extras=True)
    except (NameError, TypeError):
        hints = {}
    known = {
        parameter.name: parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    }
    unknown = set(configuration) - set(known)
    if unknown:
        raise ValueError(f"unknown configuration fields: {', '.join(sorted(unknown))}")
    resolved = {
        name: _coerce(value, hints.get(name, known[name].annotation))
        for name, value in configuration.items()
    }
    return owner(**resolved)


def execute_component(
    run: dict[str, Any], directory: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    """Execute one catalog-derived component and persist semantic artifacts."""

    descriptor, public_value = resolve_component(run["component_id"])
    if descriptor["status"] != "ready":
        raise ValueError(descriptor["diagnostic"] or "component is not executable")

    configuration = run.get("configuration", {})
    values = run.get("input_values", {})
    if not isinstance(configuration, dict) or not isinstance(values, dict):
        raise TypeError("configuration and input_values must be objects")

    kind = descriptor["execution_kind"]
    if inspect.isclass(public_value):
        instance = _construct(public_value, configuration)
        target = (
            instance.run
            if kind == "application"
            else instance.execute
            if kind == "stage"
            else instance.forward
            if kind == "torch-layer"
            else instance
        )
    else:
        target = public_value
    positional, keyword = _bind_inputs(
        target,
        descriptor["input_schema"],
        run.get("inputs", []),
        values,
    )
    result = target(*positional, **keyword)
    artifacts, metrics = render_result(result, directory)
    metrics.update(
        {
            "component_id": descriptor["id"],
            "execution_kind": kind,
            "public_api": descriptor["public_api"],
        }
    )
    return artifacts, metrics, []


__all__ = ["execute_component"]
