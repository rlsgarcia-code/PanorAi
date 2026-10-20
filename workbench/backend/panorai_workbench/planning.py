"""Validation and safe value routing for catalog-derived execution plans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
import re
from typing import Any

from .discovery import component_index


_STEP_ID = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_UPSTREAM_DIAGNOSTIC = "requires upstream typed result:"
_TYPE_SEMANTICS = {
    "CAMEvidenceFrame": "panorai:cam-evidence",
    "PerViewResults": "panorai:view-results",
    "ProjectionResult": "panorai:projection-result",
    "ResolutionObservations": "panorai:resolution-observations",
    "ResultBundle": "panorai:result-bundle",
    "SAMMaskProposalBatch": "panorai:mask-proposals",
    "SphericalFeatureSet": "panorai:feature-set",
    "SphericalObservations": "panorai:spherical-observations",
    "SphericalStereoResult": "panorai:stereo-result",
}


def _field_names(descriptor: dict[str, Any], key: str) -> set[str]:
    return {str(item["name"]) for item in descriptor[key]}


def _schema_at_path(schema: dict[str, Any], path: list[str]) -> dict[str, Any]:
    current = schema
    for part in path:
        if part.startswith("_"):
            raise ValueError("binding paths cannot access private fields")
        children = current.get("fields")
        if not isinstance(children, dict) or part not in children:
            raise ValueError(f"unknown public result field {part!r}")
        current = children[part]
    return current


def _semantic(schema: dict[str, Any]) -> str | None:
    declared = schema.get("semantic")
    if isinstance(declared, str):
        return declared
    annotation = str(schema.get("annotation", "")).split("[", 1)[0]
    type_name = str(schema.get("type", annotation)).split("[", 1)[0]
    return _TYPE_SEMANTICS.get(type_name) or _TYPE_SEMANTICS.get(annotation)


def _compatible(source: dict[str, Any], target: dict[str, Any]) -> bool:
    source_semantic = _semantic(source)
    target_semantic = _semantic(target)
    if target_semantic == "panorai:pipeline-value":
        return True
    if (
        source_semantic == "panorai:labels"
        and target_semantic == "panorai:categorical-raster"
    ):
        return True
    if target_semantic is not None:
        return source_semantic == target_semantic
    if source_semantic is not None:
        return source_semantic == target_semantic
    if source.get("annotation") == target.get("annotation"):
        return True
    source_kind = source.get("kind")
    target_kind = target.get("kind")
    return source_kind == target_kind and source_kind in {
        "array",
        "boolean",
        "integer",
        "mask",
        "matrix",
        "number",
        "object",
        "text",
        "vector",
    }


def _validate_binding(
    *,
    step_id: str,
    field: str,
    binding: Any,
    target: dict[str, Any],
    outputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(binding, dict):
        raise ValueError(f"binding {step_id}.{field} must be an object")
    source_step = binding.get("step")
    path = binding.get("path", [])
    if not isinstance(source_step, str) or source_step not in outputs:
        raise ValueError(f"binding {step_id}.{field} must reference a prior plan step")
    if not isinstance(path, list) or not all(isinstance(item, str) for item in path):
        raise ValueError(f"binding {step_id}.{field} path must be a string array")
    if len(path) > 8:
        raise ValueError(f"binding {step_id}.{field} path is too deep")
    source = _schema_at_path(outputs[source_step], path)
    if not _compatible(source, target):
        source_name = source.get("semantic") or source.get("annotation")
        target_name = target.get("semantic") or target.get("annotation")
        raise ValueError(
            f"binding {step_id}.{field} is incompatible: "
            f"{source_name!r} cannot supply {target_name!r}"
        )
    return {"step": source_step, "path": path}


def _validate_step_fields(
    *,
    step_id: str,
    descriptor: dict[str, Any],
    configuration: dict[str, Any],
    input_values: dict[str, Any],
    bindings: dict[str, Any],
    upload_roots: set[str],
    outputs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    configuration_names = _field_names(descriptor, "configuration_schema")
    input_names = _field_names(descriptor, "input_schema")
    unknown_configuration = set(configuration) - configuration_names
    unknown_inputs = (set(input_values) | set(bindings) | upload_roots) - input_names
    if unknown_configuration:
        raise ValueError(
            f"step {step_id} has unknown configuration fields: "
            + ", ".join(sorted(unknown_configuration))
        )
    if unknown_inputs:
        raise ValueError(
            f"step {step_id} has unknown input fields: "
            + ", ".join(sorted(unknown_inputs))
        )
    duplicates = (set(input_values) & set(bindings)) | (upload_roots & set(bindings))
    if duplicates:
        raise ValueError(
            f"step {step_id} supplies inputs more than once: "
            + ", ".join(sorted(duplicates))
        )

    normalized: dict[str, dict[str, Any]] = {}
    for field, binding in bindings.items():
        target = next(
            item for item in descriptor["input_schema"] if item["name"] == field
        )
        normalized[field] = _validate_binding(
            step_id=step_id,
            field=field,
            binding=binding,
            target=target,
            outputs=outputs,
        )
    for item in descriptor["input_schema"]:
        name = item["name"]
        supplied = name in input_values or name in normalized or name in upload_roots
        if item.get("required") and item.get("upstream") and name not in normalized:
            raise ValueError(
                f"step {step_id} input {name!r} requires a prior typed result"
            )
        if item.get("required") and not supplied:
            raise ValueError(f"step {step_id} is missing required input {name!r}")
    return normalized


def validate_plan(steps: Any, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a canonical ordered plan after allow-list and type validation."""

    if not isinstance(steps, list) or not 1 <= len(steps) <= 32:
        raise ValueError("steps must be an array containing between 1 and 32 steps")
    index = component_index()
    outputs: dict[str, dict[str, Any]] = {}
    normalized_steps: list[dict[str, Any]] = []
    upload_fields = [str(item.get("field", "")) for item in inputs]

    for position, raw_step in enumerate(steps):
        if not isinstance(raw_step, dict):
            raise ValueError(f"step {position} must be an object")
        step_id = raw_step.get("id")
        component_id = raw_step.get("component_id")
        if not isinstance(step_id, str) or _STEP_ID.fullmatch(step_id) is None:
            raise ValueError(f"step {position} has an invalid id")
        if step_id in outputs:
            raise ValueError(f"duplicate step id {step_id!r}")
        if not isinstance(component_id, str) or component_id not in index:
            raise ValueError(f"unknown component {component_id!r}")
        descriptor = index[component_id]
        plan_only = descriptor["status"] == "inspect" and str(
            descriptor.get("diagnostic", "")
        ).startswith(_UPSTREAM_DIAGNOSTIC)
        if descriptor["status"] != "ready" and not plan_only:
            raise ValueError(descriptor["diagnostic"] or "component is not executable")

        configuration = raw_step.get("configuration", {})
        input_values = raw_step.get("input_values", {})
        bindings = raw_step.get("bindings", {})
        if not all(
            isinstance(item, dict) for item in (configuration, input_values, bindings)
        ):
            raise ValueError(
                f"step {step_id} configuration, input_values, and bindings "
                "must be objects"
            )
        prefix = f"{step_id}."
        scoped_uploads = [
            field[len(prefix) :] for field in upload_fields if field.startswith(prefix)
        ]
        stray = [
            field
            for field in upload_fields
            if "." not in field or field.split(".", 1)[0] == step_id
            if not field.startswith(prefix)
        ]
        if stray:
            raise ValueError("plan upload fields must use <step>.<input> names")
        upload_roots = {field.split(".", 1)[0] for field in scoped_uploads}
        normalized_bindings = _validate_step_fields(
            step_id=step_id,
            descriptor=descriptor,
            configuration=configuration,
            input_values=input_values,
            bindings=bindings,
            upload_roots=upload_roots,
            outputs=outputs,
        )
        normalized_steps.append(
            {
                "id": step_id,
                "component_id": component_id,
                "configuration": configuration,
                "input_values": input_values,
                "bindings": normalized_bindings,
            }
        )
        outputs[step_id] = descriptor["output_schema"]

    known_step_ids = set(outputs)
    for field in upload_fields:
        if "." not in field or field.split(".", 1)[0] not in known_step_ids:
            raise ValueError(f"upload field {field!r} does not identify a plan step")
    return normalized_steps


def resolve_result_path(value: Any, path: list[str]) -> Any:
    """Read only declared mapping/dataclass/sequence values from a prior result."""

    current = value
    for part in path:
        if part.startswith("_"):
            raise ValueError("binding paths cannot access private fields")
        if isinstance(current, Mapping):
            if part not in current:
                raise ValueError(f"result has no mapping field {part!r}")
            current = current[part]
        elif is_dataclass(current):
            public_fields = {item.name for item in fields(current)}
            if part not in public_fields:
                raise ValueError(f"result has no dataclass field {part!r}")
            current = getattr(current, part)
        elif isinstance(current, Sequence) and not isinstance(
            current, (str, bytes, bytearray)
        ):
            if not part.isdigit() or int(part) >= len(current):
                raise ValueError(f"result has no sequence index {part!r}")
            current = current[int(part)]
        else:
            raise ValueError(f"result path cannot traverse {type(current).__name__}")
    return current


__all__ = ["resolve_result_path", "validate_plan"]
