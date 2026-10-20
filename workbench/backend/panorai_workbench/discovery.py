"""Derive Workbench components from the installed PanorAi public contract.

The browser never supplies an import path.  Only canonical imports and public
symbols read from :func:`panorai.catalog.public_catalog` are resolved here.
Bindings are inferred from concrete public signatures; no scientific adapter
is duplicated in the Workbench.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from functools import lru_cache
from importlib import import_module, util
import inspect
from numbers import Real
import re
import types
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints

import numpy as np

from panorai.catalog import PublicCatalog, public_catalog


EXECUTABLE_ROLES = frozenset(
    {
        "application",
        "geometry",
        "operation",
        "stage",
        "estimator-policy",
        "torch-layer",
    }
)
_INSPECT_ONLY_PREFIXES = ("acquire_", "download_", "load_", "prefetch_")
_IMAGE_NAMES = frozenset(
    {"image", "images", "panorama", "panoramas", "reference", "source", "values"}
)
_OPAQUE_INPUT_NAMES = frozenset(
    {
        "backend",
        "callback",
        "detector",
        "family",
        "heuristic",
        "matcher",
        "model",
        "operation",
        "predictor",
        "proposal",
        "session",
    }
)
_UPSTREAM_SEMANTICS = frozenset(
    {
        "panorai:feature-set",
        "panorai:cam-evidence",
        "panorai:mask-proposals",
        "panorai:pipeline-value",
        "panorai:projection-result",
        "panorai:raster-field",
        "panorai:region-proposals",
        "panorai:resolution-observations",
        "panorai:result-bundle",
        "panorai:stereo-result",
        "panorai:spherical-observations",
        "panorai:view-results",
        "panorai:views",
        "panorai:workflow-erp",
    }
)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {
            item.name: _jsonable(getattr(value, item.name)) for item in fields(value)
        }
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return repr(value)


def _unwrap(annotation: Any) -> tuple[Any, tuple[Any, ...], bool]:
    metadata: tuple[Any, ...] = ()
    if get_origin(annotation) is Annotated:
        annotation, *extra = get_args(annotation)
        metadata = tuple(extra)
    origin = get_origin(annotation)
    optional = False
    if origin in {Union, types.UnionType}:
        values = tuple(item for item in get_args(annotation) if item is not type(None))
        optional = len(values) != len(get_args(annotation))
        if len(values) == 1:
            annotation, nested, _ = _unwrap(values[0])
            metadata += nested
    return annotation, metadata, optional


def _type_name(annotation: Any) -> str:
    return getattr(annotation, "__name__", str(annotation).replace("typing.", ""))


def _dataclass_schema(owner: type[Any], *, depth: int = 0) -> dict[str, Any]:
    if depth > 3:
        return {"kind": "json", "type": owner.__name__}
    result: dict[str, Any] = {}
    try:
        hints = get_type_hints(owner, include_extras=True)
    except (NameError, TypeError):
        hints = {}
    for item in fields(owner):
        if not item.init:
            continue
        default: Any = None
        required = item.default is MISSING and item.default_factory is MISSING
        if item.default is not MISSING:
            default = _jsonable(item.default)
        elif item.default_factory is not MISSING:
            default = _jsonable(item.default_factory())
        result[item.name] = _value_schema(
            hints.get(item.name, item.type),
            item.name,
            required=required,
            default=default,
            depth=depth + 1,
            input_position=False,
        )
    return {"kind": "object", "type": owner.__name__, "fields": result}


def _value_schema(
    annotation: Any,
    name: str,
    *,
    required: bool,
    default: Any,
    depth: int = 0,
    input_position: bool,
) -> dict[str, Any]:
    annotation, metadata, optional = _unwrap(annotation)
    semantic = next(
        (
            item
            for item in metadata
            if isinstance(item, str) and item.startswith("panorai:")
        ),
        None,
    )
    required = required and not optional
    base = {
        "name": name,
        "required": required,
        "default": default,
        "annotation": _type_name(annotation),
    }
    if semantic:
        base["semantic"] = semantic

    if semantic == "panorai:image-sequence":
        return {**base, "kind": "image-list"}
    if semantic in _UPSTREAM_SEMANTICS:
        return {**base, "kind": "json", "upstream": True}

    origin = get_origin(annotation)
    if origin is Literal:
        choices = [_jsonable(item) for item in get_args(annotation)]
        return {**base, "kind": "select", "choices": choices}
    if inspect.isclass(annotation) and issubclass(annotation, Enum):
        return {
            **base,
            "kind": "select",
            "choices": [_jsonable(item.value) for item in annotation],
        }
    if annotation is bool:
        return {**base, "kind": "boolean"}
    if annotation is int:
        return {**base, "kind": "integer"}
    if annotation is float:
        return {**base, "kind": "number"}
    if annotation is str:
        return {**base, "kind": "text"}
    if origin in {Union, types.UnionType} and all(
        _json_annotation_supported(item) for item in get_args(annotation)
    ):
        return {**base, "kind": "json", "bindable": True}
    if inspect.isclass(annotation) and is_dataclass(annotation):
        return {**base, **_dataclass_schema(annotation, depth=depth)}

    collection_origins = {list, tuple, Sequence, Iterable}
    if origin in {Union, types.UnionType} and all(
        get_origin(item) in collection_origins for item in get_args(annotation)
    ):
        return {**base, "kind": "array"}

    names = {_type_name(annotation)}
    type_variable_bound = getattr(annotation, "__bound__", None)
    if type_variable_bound is not None:
        names.add(_type_name(type_variable_bound))
    if origin in {Union, types.UnionType}:
        names.update(_type_name(item) for item in get_args(annotation))
    normalized = name.lower()
    semantic_text = semantic or ""
    if "EquirectangularImage" in names or "erp" in semantic_text:
        result = {**base, "kind": "erp-image"}
        if semantic == "panorai:erp-container":
            result["container"] = "equirectangular-image"
        return result
    type_variable_contract = str(type_variable_bound) if type_variable_bound else ""
    array_contract = (
        annotation is np.ndarray
        or any("ndarray" in item or "TensorLike" in item for item in names)
        or "numpy.ndarray" in type_variable_contract
        or "TensorLike" in type_variable_contract
    )
    if array_contract:
        if "mask" in normalized or "mask" in semantic_text:
            return {**base, "kind": "mask"}
        if "rotation" in normalized or "rotation" in semantic_text:
            return {**base, "kind": "matrix", "shape": [3, 3]}
        if "translation" in normalized or "translation" in semantic_text:
            return {**base, "kind": "vector", "shape": [3]}
        if input_position and normalized in _IMAGE_NAMES:
            return {**base, "kind": "erp-image"}
        return {**base, "kind": "array"}
    if origin in collection_origins:
        arguments = get_args(annotation)
        return {
            **base,
            "kind": "array",
            "items": _type_name(arguments[0]) if arguments else "Any",
        }
    if origin in {dict, Mapping} and normalized in {"faces", "images"}:
        return {**base, "kind": "image-map"}
    if annotation in {Any, inspect.Signature.empty}:
        return {**base, "kind": "unsupported", "reason": "annotation is Any"}
    if input_position and normalized in _IMAGE_NAMES:
        return {**base, "kind": "unsupported", "reason": "image input is not typed"}
    return {**base, "kind": "json", "type": _type_name(annotation)}


def _json_annotation_supported(annotation: Any) -> bool:
    annotation, _, optional = _unwrap(annotation)
    if optional:
        return _json_annotation_supported(annotation)
    if annotation in {type(None), bool, int, float, str}:
        return True
    if inspect.isclass(annotation):
        if is_dataclass(annotation) or issubclass(annotation, (Enum, Real)):
            return True
        return False
    origin = get_origin(annotation)
    if origin in {Union, types.UnionType, list, tuple, Sequence, Iterable}:
        return all(_json_annotation_supported(item) for item in get_args(annotation))
    return False


def _signature_schema(callable_value: Any) -> tuple[list[dict[str, Any]], str | None]:
    signature = inspect.signature(callable_value)
    try:
        hints = get_type_hints(callable_value, include_extras=True)
    except (NameError, TypeError):
        hints = {}
    parameters: list[dict[str, Any]] = []
    issues: list[str] = []
    for parameter in signature.parameters.values():
        if parameter.name in {"self", "cls"}:
            continue
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            issues.append(f"{parameter.name} is variadic")
            continue
        required = parameter.default is inspect.Parameter.empty
        default = None if required else _jsonable(parameter.default)
        descriptor = _value_schema(
            hints.get(parameter.name, parameter.annotation),
            parameter.name,
            required=required,
            default=default,
            input_position=True,
        )
        parameters.append(descriptor)
        if (
            required
            and descriptor["kind"] in {"unsupported", "json"}
            and descriptor.get("bindable") is not True
        ):
            reason = descriptor.get("reason", "input is not automatically bindable")
            issues.append(f"{parameter.name}: {reason}")

    output = _value_schema(
        hints.get("return", signature.return_annotation),
        "result",
        required=True,
        default=None,
        input_position=False,
    )
    if output["kind"] == "unsupported":
        issues.append("return annotation is Any")
    return parameters + [{"output": output}], "; ".join(issues) or None


def _configuration_schema(owner: type[Any]) -> tuple[list[dict[str, Any]], str | None]:
    try:
        signature = inspect.signature(owner)
        hint_owner = owner if is_dataclass(owner) else owner.__init__
        hints = get_type_hints(hint_owner, include_extras=True)
    except (TypeError, NameError):
        return [], "constructor signature cannot be resolved"
    result: list[dict[str, Any]] = []
    issues: list[str] = []
    for parameter in signature.parameters.values():
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        required = parameter.default is inspect.Parameter.empty
        default = None if required else _jsonable(parameter.default)
        descriptor = _value_schema(
            hints.get(parameter.name, parameter.annotation),
            parameter.name,
            required=required,
            default=default,
            input_position=False,
        )
        result.append(descriptor)
        if required and descriptor["kind"] in {"unsupported", "json"}:
            issues.append(f"constructor {parameter.name} is not automatically bindable")
    return result, "; ".join(issues) or None


def _opaque_dependency_reason(
    inputs: list[dict[str, Any]], configuration: list[dict[str, Any]]
) -> str | None:
    opaque = [
        item["name"]
        for item in (*inputs, *configuration)
        if item.get("required")
        and item.get("name") in _OPAQUE_INPUT_NAMES
        and item.get("kind") in {"unsupported", "json"}
    ]
    if not opaque:
        return None
    return "requires opaque external dependency: " + ", ".join(opaque)


def _upstream_dependency_reason(inputs: list[dict[str, Any]]) -> str | None:
    upstream = [
        item["name"]
        for item in inputs
        if item.get("required") and item.get("upstream") is True
    ]
    if not upstream:
        return None
    return "requires upstream typed result: " + ", ".join(upstream)


def _resolve_public(surface: Any, symbol_name: str) -> tuple[Any | None, str | None]:
    module_name = surface.canonical_import
    if module_name is None:
        return None, "surface has no canonical public import"
    if not module_name.startswith("panorai.") or any(
        part.startswith("_") for part in module_name.split(".")
    ):
        return None, "canonical import is not a public PanorAi namespace"
    if "." in symbol_name or symbol_name.startswith("_"):
        return None, "delegated or private symbols are inspection-only"
    if util.find_spec(module_name) is None:
        return None, f"module {module_name!r} is unavailable"
    try:
        module = import_module(module_name)
        if symbol_name not in getattr(module, "__all__", ()):
            return None, "symbol is not exported through __all__"
        return getattr(module, symbol_name), None
    except (AttributeError, ImportError, ModuleNotFoundError, OSError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _missing_requirement(surface: Any) -> str | None:
    for requirement in surface.requires:
        module_name = str(requirement).replace("-", "_")
        try:
            available = util.find_spec(module_name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            available = False
        if not available:
            return str(requirement)
    return None


def _kind(value: Any, roles: tuple[str, ...]) -> tuple[str, Any | None]:
    try:
        from panorai.applications import Application
    except ImportError:
        Application = None  # type: ignore[assignment,misc]
    if (
        Application is not None
        and inspect.isclass(value)
        and issubclass(value, Application)
    ):
        if inspect.isabstract(value):
            return "auxiliary", None
        return "application", value.run
    if "stage" in roles and inspect.isclass(value) and hasattr(value, "execute"):
        if inspect.isabstract(value) or getattr(value, "_is_protocol", False):
            return "auxiliary", None
        return "stage", value.execute
    if "operation" in roles or "geometry" in roles:
        if inspect.isfunction(value):
            return "operation", value
        if inspect.isclass(value) and "__call__" in value.__dict__:
            if inspect.isabstract(value):
                return "auxiliary", None
            return "operation", value.__call__
    if "torch-layer" in roles and inspect.isclass(value):
        try:
            import torch.nn as nn

            if issubclass(value, nn.Module) and not inspect.isabstract(value):
                return "torch-layer", value.forward
        except (ImportError, TypeError):
            return "auxiliary", None
    if "estimator-policy" in roles and inspect.isfunction(value):
        return "estimator", value
    return "auxiliary", None


def _component_id(surface_id: str, symbol: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", symbol.lower()).strip("-")
    return f"auto--{surface_id}--{slug}"


def discover_components(catalog: PublicCatalog | None = None) -> list[dict[str, Any]]:
    """Return symbol-level descriptors derived from the installed catalog."""

    installed = catalog or public_catalog()
    components: list[dict[str, Any]] = []
    for surface in installed.iter_surfaces():
        executable_surface = bool(EXECUTABLE_ROLES.intersection(surface.roles))
        missing_requirement = (
            _missing_requirement(surface) if executable_surface else None
        )
        lazy_optional_backend = "torch-layer" in surface.roles
        for symbol_name in surface.public_symbols:
            value: Any | None = None
            diagnostic: str | None = None
            if not executable_surface:
                diagnostic = "public auxiliary/configuration/result symbol"
            elif symbol_name.lower().startswith(_INSPECT_ONLY_PREFIXES):
                diagnostic = (
                    "external acquisition, loading, and download entry points are "
                    "inspect-only"
                )
            elif missing_requirement:
                diagnostic = (
                    f"optional dependency {missing_requirement!r} is unavailable"
                )
            elif lazy_optional_backend:
                diagnostic = (
                    "optional Torch components are inspect-only until explicitly loaded"
                )
            else:
                value, diagnostic = _resolve_public(surface, symbol_name)
            execution_kind = "auxiliary"
            target = None
            if value is not None and executable_surface:
                execution_kind, target = _kind(value, surface.roles)

            configuration: list[dict[str, Any]] = []
            inputs: list[dict[str, Any]] = []
            output: dict[str, Any] | None = None
            bind_issue: str | None = None
            if target is not None:
                signature_items, signature_issue = _signature_schema(target)
                output = signature_items[-1]["output"]
                inputs = signature_items[:-1]
                if inspect.isclass(value):
                    configuration, configuration_issue = _configuration_schema(value)
                else:
                    configuration_issue = None
                bind_issue = (
                    "; ".join(
                        item for item in (signature_issue, configuration_issue) if item
                    )
                    or None
                )
            inspect_reason = _opaque_dependency_reason(inputs, configuration)
            inspect_reason = inspect_reason or _upstream_dependency_reason(inputs)

            if symbol_name.lower().startswith(_INSPECT_ONLY_PREFIXES):
                status = "inspect"
            elif missing_requirement:
                status = "unavailable"
            elif lazy_optional_backend:
                status = "inspect"
            elif execution_kind == "auxiliary":
                status = "inspect"
                diagnostic = (
                    diagnostic or "public auxiliary/configuration/result symbol"
                )
            elif inspect_reason:
                status = "inspect"
                diagnostic = inspect_reason
            elif diagnostic:
                status = "unavailable"
            elif bind_issue:
                status = "blocked"
                diagnostic = bind_issue
            else:
                status = "ready"

            component_id = _component_id(surface.id, symbol_name)
            public_api = (
                f"{surface.canonical_import}.{symbol_name}"
                if surface.canonical_import
                else symbol_name
            )
            doc = inspect.getdoc(value) if value is not None else None
            components.append(
                {
                    "id": component_id,
                    "surface_id": surface.id,
                    "symbol": symbol_name,
                    "title": symbol_name,
                    "summary": (doc or surface.as_dict().get("summary") or "").split(
                        "\n", 1
                    )[0],
                    "public_api": public_api,
                    "execution_kind": execution_kind,
                    "status": status,
                    "diagnostic": diagnostic,
                    "tier": surface.tier,
                    "roles": list(surface.roles),
                    "requires": list(surface.requires),
                    "configuration_schema": configuration,
                    "input_schema": inputs,
                    "output_schema": output,
                    "route": f"/runner.html?component={component_id}",
                }
            )
    return components


@lru_cache(maxsize=1)
def component_index() -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in discover_components()}


def resolve_component(component_id: str) -> tuple[dict[str, Any], Any]:
    descriptor = component_index().get(component_id)
    if descriptor is None:
        raise KeyError(component_id)
    surface = public_catalog().surface(descriptor["surface_id"])
    value, diagnostic = _resolve_public(surface, descriptor["symbol"])
    if value is None:
        raise ImportError(diagnostic or descriptor["public_api"])
    return descriptor, value


__all__ = [
    "component_index",
    "discover_components",
    "resolve_component",
]
