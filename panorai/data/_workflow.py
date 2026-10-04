"""Private machinery for the stable ergonomic workflow.

This module deliberately composes :mod:`panorai.geometry`; it does not own a
second projection convention.  Torch is discovered from the values passed by
the caller and is never imported on the NumPy path.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
import math
from numbers import Integral, Real
from typing import Any

import numpy as np

WORKFLOW_CONTRACT = "geometry-v1"
WORKFLOW_INTERFACE = "panorai-object-workflow/v1"
WORKFLOW_STABILITY = "stable"
MODALITIES = ("image", "depth", "labels")


def is_torch(value: Any) -> bool:
    module = type(value).__module__
    return module == "torch" or module.startswith("torch.")


def torch_module():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Torch support is optional; install PanorAi with `pip install panorai[torch]`."
        ) from exc
    return torch


def clone_array(value: Any) -> Any:
    return value.clone() if is_torch(value) else np.asarray(value).copy()


def backend(value: Any) -> str:
    if is_torch(value):
        return "torch"
    if isinstance(value, np.ndarray):
        return "numpy"
    raise TypeError("data must be a numpy.ndarray or torch.Tensor")


def device(value: Any) -> str | None:
    return str(value.device) if is_torch(value) else None


def spatial_shape(value: Any) -> tuple[int, int]:
    kind = backend(value)
    if kind == "numpy":
        if value.ndim not in {2, 3}:
            raise ValueError("NumPy modalities must use HW or HWC layout")
        return int(value.shape[0]), int(value.shape[1])
    if value.ndim not in {2, 3, 4}:
        raise ValueError("Torch modalities must use HW, CHW, or NCHW layout")
    return int(value.shape[-2]), int(value.shape[-1])


def batch_size(value: Any) -> int | None:
    spatial_shape(value)
    return int(value.shape[0]) if is_torch(value) and value.ndim == 4 else None


def layout(value: Any) -> str:
    if is_torch(value):
        return {2: "HW", 3: "CHW", 4: "NCHW"}[value.ndim]
    return {2: "HW", 3: "HWC"}[value.ndim]


def dtype_name(value: Any) -> str:
    return str(value.dtype).replace("torch.", "")


def is_floating(value: Any) -> bool:
    return (
        bool(value.dtype.is_floating_point)
        if is_torch(value)
        else bool(np.issubdtype(value.dtype, np.floating))
    )


def is_integer_or_bool(value: Any) -> bool:
    if is_torch(value):
        torch = torch_module()
        return value.dtype == torch.bool or not (
            value.dtype.is_floating_point or value.dtype.is_complex
        )
    return bool(
        np.issubdtype(value.dtype, np.integer) or np.issubdtype(value.dtype, np.bool_)
    )


def expected_mask_shape(value: Any) -> tuple[int, ...]:
    h, w = spatial_shape(value)
    batch = batch_size(value)
    return (h, w) if batch is None else (batch, h, w)


def ones_mask(value: Any) -> Any:
    shape = expected_mask_shape(value)
    if is_torch(value):
        return torch_module().ones(
            shape, dtype=torch_module().bool, device=value.device
        )
    return np.ones(shape, dtype=bool)


def validate_mask(value: Any, mask: Any, *, name: str = "valid") -> Any:
    if backend(mask) != backend(value):
        raise TypeError(f"{name} and data must use the same backend")
    if is_torch(mask):
        torch = torch_module()
        if mask.dtype != torch.bool:
            raise TypeError(f"{name} must have boolean dtype")
        if mask.device != value.device:
            raise ValueError(f"{name} and data must use the same Torch device")
    elif mask.dtype != np.bool_:
        raise TypeError(f"{name} must have boolean dtype")
    expected = expected_mask_shape(value)
    if tuple(mask.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}; got {tuple(mask.shape)}")
    return clone_array(mask)


def validate_support_mask(value: Any, mask: Any) -> Any:
    """Validate source support without interpreting it as data validity."""

    if is_torch(value):
        if not is_torch(mask):
            raise TypeError("support_mask and data must use the same backend")
        torch = torch_module()
        if mask.dtype != torch.bool:
            raise TypeError("support_mask must have boolean dtype")
        if mask.device != value.device:
            raise ValueError("support_mask and data must use the same Torch device")
    else:
        mask = np.asarray(mask, dtype=bool)
    h, w = spatial_shape(value)
    allowed = {(h, w)}
    if batch_size(value) is not None:
        allowed.add(expected_mask_shape(value))
    if tuple(mask.shape) not in allowed:
        choices = " or ".join(str(shape) for shape in sorted(allowed))
        raise ValueError(
            f"support_mask must have shape {choices}; got {tuple(mask.shape)}"
        )
    return clone_array(mask)


def expand_mask(mask: Any, value: Any) -> Any:
    if is_torch(value):
        if value.ndim == 3:
            return mask.unsqueeze(0)
        if value.ndim == 4:
            return mask.unsqueeze(1)
        return mask
    return mask[..., None] if value.ndim == 3 else mask


def finite_spatial(value: Any) -> Any:
    if is_torch(value):
        finite = torch_module().isfinite(value)
        if value.ndim == 3:
            return finite.all(dim=0)
        if value.ndim == 4:
            return finite.all(dim=1)
        return finite
    finite = np.isfinite(value)
    return finite.all(axis=-1) if value.ndim == 3 else finite


def any_true(value: Any) -> bool:
    return bool(value.any().item()) if is_torch(value) else bool(np.any(value))


def validate_finite_where_valid(value: Any, valid: Any, *, name: str) -> None:
    bad = valid & ~finite_spatial(value)
    if any_true(bad):
        raise ValueError(f"{name} contains non-finite values marked as valid")


def validate_modality(value: Any, kind: str, *, name: str) -> None:
    backend(value)
    spatial_shape(value)
    if kind == "depth" and not is_floating(value):
        raise TypeError("depth must use a floating-point dtype")
    if kind == "labels" and not is_integer_or_bool(value):
        raise TypeError("labels must use an integer or boolean dtype")
    if kind not in MODALITIES:
        raise ValueError(f"unsupported modality {kind!r}; choose one of {MODALITIES}")


def validate_bundle(data: Mapping[str, Any]) -> None:
    if not data:
        raise ValueError("a workflow bundle cannot be empty")
    first_name = next(iter(data))
    first = data[first_name]
    reference = (backend(first), device(first), batch_size(first), spatial_shape(first))
    for name, value in data.items():
        current = (
            backend(value),
            device(value),
            batch_size(value),
            spatial_shape(value),
        )
        if current != reference:
            raise ValueError(
                "all modalities must share backend, device, batch, and spatial shape; "
                f"{first_name!r} has {reference} but {name!r} has {current}"
            )


def copy_metadata(
    metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    copied: dict[str, dict[str, Any]] = {}
    for name, item in metadata.items():
        copied[name] = dict(item)
        copied[name]["validity"] = clone_array(item["validity"])
    return copied


def build_primary_metadata(
    data: Any, valid: Any | None = None
) -> dict[str, dict[str, Any]]:
    validate_modality(data, "image", name="image")
    valid = ones_mask(data) if valid is None else validate_mask(data, valid)
    return {
        "image": {
            "kind": "image",
            "units": None,
            "validity": valid,
            "interpolation": "bilinear",
        }
    }


def add_modality(
    data: Mapping[str, Any],
    metadata: Mapping[str, Mapping[str, Any]],
    name: str,
    value: Any,
    *,
    valid: Any | None,
    units: str | None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if name in data:
        raise ValueError(f"modality {name!r} already exists")
    validate_modality(value, name, name=name)
    proposed = {key: clone_array(array) for key, array in data.items()}
    proposed[name] = clone_array(value)
    validate_bundle(proposed)
    mask = ones_mask(value) if valid is None else validate_mask(value, valid)
    validate_finite_where_valid(value, mask, name=name)
    result_meta = copy_metadata(metadata)
    result_meta[name] = {
        "kind": name,
        "units": units if name == "depth" else None,
        "validity": mask,
        "interpolation": "nearest" if name == "labels" else "bilinear",
    }
    return proposed, result_meta


def normalize_size(size: Any, source_shape: tuple[int, int]) -> tuple[int, int]:
    if size is None:
        return max(1, math.ceil(source_shape[0] / 2)), max(
            1, math.ceil(source_shape[1] / 4)
        )
    if isinstance(size, bool):
        raise TypeError("size values must be integers")
    if isinstance(size, Integral):
        dimensions = (int(size), int(size))
    else:
        try:
            dimensions = tuple(size)
        except TypeError as exc:
            raise TypeError("size must be an integer or (height, width)") from exc
        if len(dimensions) != 2:
            raise ValueError("size must contain exactly (height, width)")
        if any(
            isinstance(item, bool) or not isinstance(item, Integral)
            for item in dimensions
        ):
            raise TypeError("size values must be integers")
        dimensions = int(dimensions[0]), int(dimensions[1])
    if dimensions[0] <= 0 or dimensions[1] <= 0:
        raise ValueError("size values must be positive")
    return dimensions


def normalize_fov(fov: Any) -> tuple[float, float]:
    if isinstance(fov, bool):
        raise TypeError("fov must be a real number or (horizontal, vertical)")
    if isinstance(fov, Real):
        values = (float(fov), float(fov))
    else:
        try:
            raw = tuple(fov)
        except TypeError as exc:
            raise TypeError(
                "fov must be a real number or (horizontal, vertical)"
            ) from exc
        if len(raw) != 2:
            raise ValueError("fov must contain exactly (horizontal, vertical)")
        if any(isinstance(item, bool) or not isinstance(item, Real) for item in raw):
            raise TypeError("fov values must be real numbers")
        values = float(raw[0]), float(raw[1])
    if any(not math.isfinite(item) or not 0.0 < item < 180.0 for item in values):
        raise ValueError("fov values must be finite and between 0 and 180 degrees")
    return values


def _icosahedron_points(subdivisions: int) -> list[tuple[float, float]]:
    """Return deterministic unique icosphere vertices in legacy sampler order."""

    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.asarray(
        [
            (-1, phi, 0),
            (1, phi, 0),
            (-1, -phi, 0),
            (1, -phi, 0),
            (0, -1, phi),
            (0, 1, phi),
            (0, -1, -phi),
            (0, 1, -phi),
            (phi, 0, -1),
            (phi, 0, 1),
            (-phi, 0, -1),
            (-phi, 0, 1),
        ],
        dtype=np.float64,
    )
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    faces = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (5, 4, 9),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]
    verts = [row for row in vertices]
    for _ in range(subdivisions):
        midpoint_cache: dict[tuple[int, int], int] = {}
        next_faces = []

        def midpoint(left: int, right: int) -> int:
            key = tuple(sorted((left, right)))
            if key not in midpoint_cache:
                value = (verts[left] + verts[right]) / 2.0
                value /= np.linalg.norm(value)
                midpoint_cache[key] = len(verts)
                verts.append(value)
            return midpoint_cache[key]

        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            next_faces.extend(((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)))
        faces = next_faces
    points = []
    for x, y, z in verts:
        points.append(
            (float(np.degrees(np.arcsin(z))), float(np.degrees(np.arctan2(y, x))))
        )
    return points


def resolve_tangent_points(
    layout_name: Any,
    *,
    count: int | None,
    subdivisions: int,
    rotations: Iterable[tuple[float, float]],
) -> tuple[str, list[tuple[float, float]], list[str]]:
    if isinstance(layout_name, str):
        if layout_name not in {"cube", "fibonacci", "icosahedron", "spiral"}:
            raise ValueError(
                "layout must be 'cube', 'fibonacci', 'icosahedron', 'spiral', "
                "or a sampler object"
            )
        from panorai.samplers.default_samplers import (
            CubeSampler,
            FibonacciSampler,
            SpiralSampler,
        )

        if count is not None and (
            isinstance(count, bool) or not isinstance(count, Integral)
        ):
            raise TypeError("count must be an integer")
        if count is not None and count <= 0:
            raise ValueError("count must be positive")
        if layout_name in {"cube", "icosahedron"} and count is not None:
            raise ValueError(f"count is not valid for layout={layout_name!r}")
        if layout_name != "icosahedron" and subdivisions != 0:
            raise ValueError("subdivisions is only valid for layout='icosahedron'")
        if layout_name == "cube":
            sampler = CubeSampler()
            order = ["front", "right", "back", "left", "up", "down"]
        elif layout_name == "fibonacci":
            sampler = FibonacciSampler(n_points=20 if count is None else int(count))
            order = [
                f"fibonacci-{index:03d}"
                for index in range(20 if count is None else int(count))
            ]
        elif layout_name == "spiral":
            sampler = SpiralSampler(n_points=20 if count is None else int(count))
            order = [
                f"spiral-{index:03d}"
                for index in range(20 if count is None else int(count))
            ]
        else:
            if isinstance(subdivisions, bool) or not isinstance(subdivisions, Integral):
                raise TypeError("subdivisions must be an integer")
            if subdivisions < 0:
                raise ValueError("subdivisions must be non-negative")
            sampler = None
            order = []
        points = (
            _icosahedron_points(int(subdivisions))
            if layout_name == "icosahedron"
            else [(float(lat), float(lon)) for lat, lon in sampler.get_tangent_points()]
        )
        if not order:
            order = [f"icosahedron-{index:03d}" for index in range(len(points))]
        resolved_name = layout_name
    else:
        if count is not None or subdivisions != 0:
            raise ValueError("count and subdivisions are configured by sampler objects")
        if not hasattr(layout_name, "get_tangent_points"):
            raise TypeError("layout objects must provide get_tangent_points()")
        points = [
            (float(lat), float(lon)) for lat, lon in layout_name.get_tangent_points()
        ]
        resolved_name = type(layout_name).__name__
        order = [f"view-{index:03d}" for index in range(len(points))]
    if not points:
        raise ValueError("the selected layout produced no views")

    rotation_values = list(rotations)
    unrotated_points = list(points)
    for rotation in rotation_values:
        if len(rotation) != 2:
            raise ValueError("each rotation must contain (latitude, longitude)")
        dlat, dlon = float(rotation[0]), float(rotation[1])
        start = len(points)
        rotated = []
        for lat, lon in unrotated_points:
            new_lat, new_lon = lat + dlat, lon + dlon
            if new_lat > 90.0:
                new_lat, new_lon = 180.0 - new_lat, new_lon + 180.0
            elif new_lat < -90.0:
                new_lat, new_lon = -180.0 - new_lat, new_lon + 180.0
            rotated.append((new_lat, (new_lon + 180.0) % 360.0 - 180.0))
        points.extend(rotated)
        order.extend(f"rotation-{start + index:03d}" for index in range(len(rotated)))
    return resolved_name, points, order


def as_bilinear_input(value: Any) -> Any:
    if is_floating(value):
        return value
    if is_torch(value):
        return value.to(dtype=torch_module().float32)
    return value.astype(np.float32)


def cast_mask_float(mask: Any, like: Any) -> Any:
    if is_torch(mask):
        return mask.to(dtype=like.dtype)
    return mask.astype(like.dtype, copy=False)


def logical_and(left: Any, right: Any) -> Any:
    return left & right


def support_for_value(support: Any, value: Any) -> Any:
    if batch_size(value) is None:
        return support
    if tuple(support.shape) == expected_mask_shape(value):
        return support
    if is_torch(value):
        return support.unsqueeze(0).expand(value.shape[0], -1, -1)
    raise AssertionError("NumPy has no batched image layout")


def masked_invalid_to_nan(value: Any, valid: Any) -> Any:
    expanded = expand_mask(valid, value)
    if is_torch(value):
        return torch_module().where(
            expanded, value, torch_module().full_like(value, float("nan"))
        )
    return np.where(expanded, value, np.nan).astype(value.dtype, copy=False)


def configured_projector(
    template: Any,
    spec: Any,
    *,
    interpolation: str,
    invalid_policy: str = "propagate",
    min_valid_weight: float | None = None,
):
    from panorai.geometry import GnomonicProjector

    if template is None:
        return GnomonicProjector(
            spec,
            interpolation=interpolation,
            invalid_policy=invalid_policy,
            min_valid_weight=min_valid_weight,
        )
    if not isinstance(template, GnomonicProjector):
        raise TypeError(
            "projector objects must be panorai.geometry.GnomonicProjector instances"
        )
    return replace(
        template,
        spec=spec,
        interpolation=interpolation,
        invalid_policy=invalid_policy,
        min_valid_weight=min_valid_weight,
    )


def _strict_validity(projector: Any, valid: Any, like: Any, *, back_shape=None) -> Any:
    mask_float = cast_mask_float(valid, like)
    mask_projector = replace(
        projector,
        interpolation="bilinear",
        fill_value=None,
        invalid_policy="propagate",
        min_valid_weight=None,
    )
    result = (
        mask_projector.project(mask_float)
        if back_shape is None
        else mask_projector.back_project(mask_float, back_shape)
    )
    tolerance = 32.0 * np.finfo(np.float32).eps
    return result.data >= (1.0 - tolerance)


def project_modality(
    value: Any,
    metadata: Mapping[str, Any],
    spec: Any,
    *,
    depth_policy: str,
    min_valid_weight: float | None,
    projector_template: Any = None,
    source_support: Any = None,
) -> tuple[Any, Any, Any]:
    kind = metadata["kind"]
    support_source = ones_mask(value) if source_support is None else source_support
    effective_validity = metadata["validity"] & support_source
    validate_finite_where_valid(value, effective_validity, name=kind)
    working = value if kind == "labels" else as_bilinear_input(value)
    valid = effective_validity
    support_projector = configured_projector(
        projector_template, spec, interpolation="nearest"
    )
    projected_source_support = support_projector.project(support_source).data
    if kind == "labels":
        projector = configured_projector(
            projector_template, spec, interpolation="nearest"
        )
        result = projector.project(working)
        mask_result = projector.project(valid)
        output_valid = mask_result.data & support_for_value(
            result.support_mask, result.data
        )
    elif depth_policy == "renormalize" and kind == "depth":
        result = configured_projector(
            projector_template,
            spec,
            interpolation="bilinear",
            invalid_policy="renormalize",
            min_valid_weight=min_valid_weight,
        ).project(working, validity_mask=valid)
        output_valid = result.validity_mask
    else:
        strict_input = masked_invalid_to_nan(working, valid)
        projector = configured_projector(
            projector_template, spec, interpolation="bilinear"
        )
        result = projector.project(strict_input)
        output_valid = _strict_validity(projector, valid, working)
        output_valid = output_valid & finite_spatial(result.data)
    support = support_for_value(result.support_mask, result.data)
    projected_source_support = support_for_value(projected_source_support, result.data)
    support = support & projected_source_support
    return result.data, support, output_valid & support


def project_modality_batch(
    value: Any,
    metadata: Mapping[str, Any],
    specs: list[Any],
    *,
    depth_policy: str,
    min_valid_weight: float | None,
    projector_template: Any = None,
    source_support: Any = None,
    plan: Any = None,
) -> tuple[tuple[Any, Any, Any], ...]:
    """Project one modality into N views through one reusable forward plan."""

    if (
        is_torch(value)
        or projector_template is not None
        or (depth_policy == "renormalize" and metadata["kind"] == "depth")
    ):
        return tuple(
            project_modality(
                value,
                metadata,
                spec,
                depth_policy=depth_policy,
                min_valid_weight=min_valid_weight,
                projector_template=projector_template,
                source_support=source_support,
            )
            for spec in specs
        )

    from panorai.geometry._engine import (
        _gnomonic_batch_forward_plan,
        _gnomonic_batch_from_equirectangular,
    )

    kind = metadata["kind"]
    support_source = ones_mask(value) if source_support is None else source_support
    effective_validity = metadata["validity"] & support_source
    validate_finite_where_valid(value, effective_validity, name=kind)
    working = value if kind == "labels" else as_bilinear_input(value)
    interpolation = "nearest" if kind == "labels" else "bilinear"
    if plan is None:
        plan = _gnomonic_batch_forward_plan(specs, spatial_shape(value), working)

    sampled_support = None
    if not bool(np.all(support_source)):
        sampled_support = _gnomonic_batch_from_equirectangular(
            cast_mask_float(support_source, working),
            plan,
            interpolation="nearest",
        )
    if kind == "labels":
        projected = _gnomonic_batch_from_equirectangular(
            working, plan, interpolation="nearest"
        )
        sampled_validity = _gnomonic_batch_from_equirectangular(
            effective_validity, plan, interpolation="nearest"
        )
        validity = tuple(item.astype(bool, copy=False) for item in sampled_validity)
    else:
        projected = _gnomonic_batch_from_equirectangular(
            masked_invalid_to_nan(working, effective_validity),
            plan,
            interpolation=interpolation,
        )
        if bool(np.all(effective_validity)):
            validity = tuple(finite_spatial(item) for item in projected)
        else:
            sampled_validity = _gnomonic_batch_from_equirectangular(
                cast_mask_float(effective_validity, working),
                plan,
                interpolation="bilinear",
            )
            tolerance = 32.0 * np.finfo(np.float32).eps
            validity = tuple(
                (mask >= (1.0 - tolerance)) & finite_spatial(item)
                for item, mask in zip(projected, sampled_validity, strict=True)
            )

    results = []
    for index, (spec, item, valid) in enumerate(
        zip(specs, projected, validity, strict=True)
    ):
        support = (
            np.ones(spec.output_shape_hw, dtype=bool)
            if sampled_support is None
            else sampled_support[index].astype(bool, copy=False)
        )
        results.append((item, support, valid & support))
    return tuple(results)


def back_project_modality(
    value: Any,
    valid: Any,
    spec: Any,
    output_shape: tuple[int, int],
    *,
    kind: str,
    depth_policy: str,
    min_valid_weight: float | None,
    projector_template: Any = None,
) -> tuple[Any, Any, Any]:
    if kind == "labels":
        projector = configured_projector(
            projector_template, spec, interpolation="nearest"
        )
        result = projector.back_project(value, output_shape)
        valid_result = projector.back_project(valid, output_shape)
        validity = valid_result.data
    elif depth_policy == "renormalize" and kind == "depth":
        projector = configured_projector(
            projector_template,
            spec,
            interpolation="bilinear",
            invalid_policy="renormalize",
            min_valid_weight=min_valid_weight,
        )
        result = projector.back_project(value, output_shape, validity_mask=valid)
        validity = result.validity_mask
    else:
        working = as_bilinear_input(value)
        projector = configured_projector(
            projector_template, spec, interpolation="bilinear"
        )
        result = projector.back_project(
            masked_invalid_to_nan(working, valid), output_shape
        )
        validity = _strict_validity(projector, valid, working, back_shape=output_shape)
        validity = validity & finite_spatial(result.data)
    support = support_for_value(result.support_mask, result.data)
    return result.data, support, support & validity


@dataclass(frozen=True, slots=True)
class _SparseWorkflowContribution:
    flat_indices: Any
    data: Any
    valid: Any
    center_score: Any


def _compact_finite(value: Any, source: Any) -> Any:
    if is_torch(value):
        finite = torch_module().isfinite(value)
        if source.ndim == 3:
            return finite.all(dim=0)
        if source.ndim == 4:
            return finite.all(dim=1)
        return finite
    finite = np.isfinite(value)
    return finite.all(axis=-1) if source.ndim == 3 else finite


def back_project_modality_sparse(
    values: list[Any],
    valid_masks: list[Any],
    plan: Any,
    *,
    kind: str,
    depth_policy: str,
    min_valid_weight: float | None,
) -> tuple[_SparseWorkflowContribution, ...]:
    """Back-project one modality through an already batched selective plan."""

    from panorai.geometry._engine import _gnomonic_batch_to_sparse

    if kind == "labels":
        sampled_values = _gnomonic_batch_to_sparse(
            values, plan, interpolation="nearest"
        )
        sampled_validity = _gnomonic_batch_to_sparse(
            [cast_mask_float(mask, value) for value, mask in zip(values, valid_masks)],
            plan,
            interpolation="nearest",
        )
        validity = [
            (
                item.data.to(dtype=torch_module().bool)
                if is_torch(item.data)
                else item.data.astype(bool, copy=False)
            )
            for item in sampled_validity
        ]
    elif depth_policy == "renormalize" and kind == "depth":
        sampled_values = _gnomonic_batch_to_sparse(
            values,
            plan,
            interpolation="bilinear",
            invalid_policy="renormalize",
            validity_masks=valid_masks,
            min_valid_weight=min_valid_weight,
        )
        validity = [item.validity_mask for item in sampled_values]
    else:
        working = [as_bilinear_input(value) for value in values]
        strict_inputs = [
            masked_invalid_to_nan(value, valid)
            for value, valid in zip(working, valid_masks)
        ]
        sampled_values = _gnomonic_batch_to_sparse(
            strict_inputs, plan, interpolation="bilinear"
        )
        sampled_validity = _gnomonic_batch_to_sparse(
            [cast_mask_float(mask, value) for value, mask in zip(working, valid_masks)],
            plan,
            interpolation="bilinear",
        )
        tolerance = 32.0 * np.finfo(np.float32).eps
        validity = [
            mask_item.data >= (1.0 - tolerance) for mask_item in sampled_validity
        ]
        validity = [
            valid & _compact_finite(item.data, source)
            for valid, item, source in zip(validity, sampled_values, working)
        ]
    return tuple(
        _SparseWorkflowContribution(
            item.flat_indices,
            item.data,
            valid,
            item.center_score,
        )
        for item, valid in zip(sampled_values, validity, strict=True)
    )


def native_gaussian_reconstruct(
    values: list[Any],
    valid_masks: list[Any],
    plan: Any,
    *,
    kind: str,
    depth_policy: str,
) -> tuple[Any, Any] | None:
    """Use the fused native Gaussian path when its strict NumPy ABI applies."""

    if (
        kind == "labels"
        or depth_policy != "propagate"
        or not values
        or is_torch(values[0])
    ):
        return None
    working = [as_bilinear_input(value) for value in values]
    strict_inputs = [
        value if bool(np.all(mask)) else masked_invalid_to_nan(value, mask)
        for value, mask in zip(working, valid_masks, strict=True)
    ]
    from panorai.geometry import _native as native_geometry

    if not native_geometry.supports_native_gnomonic_gaussian(
        strict_inputs, valid_masks
    ):
        return None
    native_plans = tuple(
        (
            face.flat_indices,
            face.map_x,
            face.map_y,
            face.center_score,
        )
        for face in plan.faces
    )
    return native_geometry.native_gnomonic_gaussian_to_equirectangular(
        strict_inputs,
        valid_masks,
        native_plans,
        plan.output_shape_hw,
    )


def sparse_support_union(plan: Any, like: Any) -> Any:
    """Materialize only the final union support mask for a sparse batch plan."""

    pixel_count = plan.output_shape_hw[0] * plan.output_shape_hw[1]
    if is_torch(like):
        torch = torch_module()
        support = torch.zeros(pixel_count, dtype=torch.bool, device=like.device)
        for face in plan.faces:
            support[face.flat_indices] = True
        support = support.reshape(plan.output_shape_hw)
        if batch_size(like) is not None:
            support = support.unsqueeze(0).expand(like.shape[0], -1, -1)
        return support
    support = np.zeros(pixel_count, dtype=bool)
    for face in plan.faces:
        support[face.flat_indices] = True
    return support.reshape(plan.output_shape_hw)


def _sparse_output_buffers(like: Any, pixel_count: int, *, floating: bool):
    if is_torch(like):
        torch = torch_module()
        dtype = (
            torch.float32
            if floating and not like.dtype.is_floating_point
            else like.dtype
        )
        shape = tuple(like.shape[:-2]) + (pixel_count,)
        output = torch.zeros(shape, dtype=dtype, device=like.device)
        mask_shape = (
            (like.shape[0], pixel_count)
            if batch_size(like) is not None
            else (pixel_count,)
        )
        return output, torch.zeros(mask_shape, dtype=dtype, device=like.device)
    dtype = (
        np.float32
        if floating and not np.issubdtype(like.dtype, np.floating)
        else like.dtype
    )
    trailing = tuple(like.shape[2:])
    return (
        np.zeros((pixel_count, *trailing), dtype=dtype),
        np.zeros(pixel_count, dtype=dtype),
    )


def _compact_expand(mask: Any, data: Any, like: Any) -> Any:
    if is_torch(data):
        if like.ndim == 3:
            return mask.unsqueeze(0)
        if like.ndim == 4:
            return mask.unsqueeze(1)
        return mask
    return mask[..., None] if like.ndim == 3 else mask


def _reshape_sparse_output(value: Any, like: Any, shape: tuple[int, int]) -> Any:
    if is_torch(value):
        return value.reshape((*like.shape[:-2], *shape))
    return value.reshape((*shape, *like.shape[2:]))


def blend_sparse_reprojected(
    contributions: tuple[_SparseWorkflowContribution, ...],
    like: Any,
    shape: tuple[int, int],
    method: str,
) -> tuple[Any, Any]:
    """Fuse compact N-view contributions without N full ERP intermediates."""

    if method not in {"average", "closest", "gaussian"}:
        raise ValueError("sparse blending supports average, closest, or gaussian")
    pixel_count = shape[0] * shape[1]
    if method in {"average", "gaussian"}:
        total, weight_sum = _sparse_output_buffers(like, pixel_count, floating=True)
        for item in contributions:
            data = item.data
            valid = item.valid
            if method == "gaussian":
                if is_torch(data):
                    weight = (
                        torch_module()
                        .exp(6.0 * (item.center_score - 1.0))
                        .to(dtype=data.dtype)
                    )
                else:
                    weight = np.exp(6.0 * (item.center_score - 1.0)).astype(
                        data.dtype, copy=False
                    )
                if batch_size(like) is not None:
                    weight = weight.unsqueeze(0).expand(like.shape[0], -1)
                effective = weight * cast_mask_float(valid, data)
            else:
                effective = cast_mask_float(valid, data)
            weighted = _where(
                _compact_expand(valid, data, like),
                data,
                _zeros_like(data),
            ) * _compact_expand(effective, data, like)
            if is_torch(data):
                total = total.index_add(-1, item.flat_indices, weighted)
                weight_sum = weight_sum.index_add(-1, item.flat_indices, effective)
            else:
                total[item.flat_indices] += weighted
                weight_sum[item.flat_indices] += effective
        valid = weight_sum > 0
        safe_weight = _where(valid, weight_sum, weight_sum + 1)
        result = total / _compact_expand(safe_weight, total, like)
        if is_torch(result):
            fill = torch_module().full_like(result, float("nan"))
        else:
            fill = np.full_like(result, np.nan)
        result = _where(_compact_expand(valid, result, like), result, fill)
        return _reshape_sparse_output(result, like, shape), valid.reshape(
            (*valid.shape[:-1], *shape)
        )

    output, _ = _sparse_output_buffers(like, pixel_count, floating=False)
    if is_torch(like):
        torch = torch_module()
        mask_shape = (
            (like.shape[0], pixel_count)
            if batch_size(like) is not None
            else (pixel_count,)
        )
        best = torch.full(
            mask_shape, -torch.inf, dtype=torch.float32, device=like.device
        )
        supported = torch.zeros(mask_shape, dtype=torch.bool, device=like.device)
    else:
        best = np.full(pixel_count, -np.inf, dtype=np.float64)
        supported = np.zeros(pixel_count, dtype=bool)
    for item in contributions:
        score = item.center_score
        if is_torch(score):
            score = score.to(dtype=torch_module().float32)
        if batch_size(like) is not None:
            score = score.unsqueeze(0).expand(like.shape[0], -1)
        current = best[..., item.flat_indices]
        choose = item.valid & (score > current)
        current_output = (
            output[..., item.flat_indices]
            if is_torch(output)
            else output[item.flat_indices]
        )
        replacement = _where(
            _compact_expand(choose, item.data, like),
            item.data,
            current_output,
        )
        if is_torch(output):
            output = output.index_copy(-1, item.flat_indices, replacement)
            best = best.index_copy(
                -1, item.flat_indices, _where(choose, score, current)
            )
            supported = supported.index_copy(
                -1,
                item.flat_indices,
                supported[..., item.flat_indices] | item.valid,
            )
        else:
            output[item.flat_indices] = replacement
            best[item.flat_indices] = np.where(choose, score, current)
            supported[item.flat_indices] |= item.valid
    return _reshape_sparse_output(output, like, shape), supported.reshape(
        (*supported.shape[:-1], *shape)
    )


def union_masks(masks: list[Any]) -> Any:
    stacked = _stack(masks)
    if is_torch(stacked):
        return stacked.any(dim=0)
    return np.any(stacked, axis=0)


def validate_model_result(
    result: Any,
    source: Any,
    *,
    output: str,
    support: Any,
) -> tuple[Any, Any]:
    if isinstance(result, tuple):
        if len(result) != 2:
            raise ValueError(
                "model tuple results must contain exactly (array, validity_mask)"
            )
        value, valid = result
    else:
        value, valid = result, None
    validate_modality(value, output, name=output)
    expected = (
        backend(source),
        device(source),
        batch_size(source),
        spatial_shape(source),
    )
    actual = (backend(value), device(value), batch_size(value), spatial_shape(value))
    if actual != expected:
        raise ValueError(
            "model output must preserve backend, device, batch, and spatial shape; "
            f"expected {expected}, got {actual}"
        )
    geometric_support = support_for_value(support, value)
    mask = (
        clone_array(geometric_support) if valid is None else validate_mask(value, valid)
    )
    validate_finite_where_valid(
        value,
        mask & geometric_support,
        name="model output",
    )
    return clone_array(value), mask


def array_description(value: Any) -> dict[str, Any]:
    return {
        "backend": backend(value),
        "device": device(value),
        "dtype": dtype_name(value),
        "layout": layout(value),
        "shape": tuple(int(item) for item in value.shape),
    }


def _erp_center_scores(specs: list[Any], shape: tuple[int, int]) -> np.ndarray:
    from panorai.geometry import erp_pixels_to_rays

    h, w = shape
    y, x = np.indices((h, w), dtype=np.float64)
    rays = erp_pixels_to_rays(np.stack((x, y), axis=-1), shape)
    scores = []
    for spec in specs:
        lat = np.deg2rad(spec.center_lat_deg)
        lon = np.deg2rad(spec.center_lon_deg)
        center = np.asarray(
            (np.sin(lon) * np.cos(lat), np.sin(lat), np.cos(lon) * np.cos(lat))
        )
        scores.append(rays @ center)
    return np.stack(scores, axis=0)


def _zeros_like(value: Any, *, floating: bool = False) -> Any:
    if is_torch(value):
        torch = torch_module()
        dtype = (
            torch.float32
            if floating and not value.dtype.is_floating_point
            else value.dtype
        )
        return torch.zeros_like(value, dtype=dtype)
    dtype = (
        np.float32
        if floating and not np.issubdtype(value.dtype, np.floating)
        else value.dtype
    )
    return np.zeros_like(value, dtype=dtype)


def _where(mask: Any, left: Any, right: Any) -> Any:
    return (
        torch_module().where(mask, left, right)
        if is_torch(left)
        else np.where(mask, left, right)
    )


def _stack(values: list[Any], axis: int = 0) -> Any:
    return (
        torch_module().stack(values, dim=axis)
        if is_torch(values[0])
        else np.stack(values, axis=axis)
    )


def _sum(value: Any, axis: int) -> Any:
    return value.sum(dim=axis) if is_torch(value) else value.sum(axis=axis)


def _argmax(value: Any, axis: int) -> Any:
    return value.argmax(dim=axis) if is_torch(value) else value.argmax(axis=axis)


def blend_average(values: list[Any], masks: list[Any]) -> tuple[Any, Any]:
    floating_values = [as_bilinear_input(value) for value in values]
    weighted = [
        _where(expand_mask(mask, value), value, _zeros_like(value))
        for value, mask in zip(floating_values, masks)
    ]
    total = _sum(_stack(weighted), 0)
    count = _sum(
        _stack([cast_mask_float(mask, floating_values[0]) for mask in masks]), 0
    )
    valid = count > 0
    denominator = _where(valid, count, count + 1)
    result = total / expand_mask(denominator, total)
    fill = (
        torch_module().full_like(result, float("nan"))
        if is_torch(result)
        else np.full_like(result, np.nan)
    )
    return _where(expand_mask(valid, result), result, fill), valid


def blend_weighted(
    values: list[Any], masks: list[Any], weights: np.ndarray
) -> tuple[Any, Any]:
    floating_values = [as_bilinear_input(value) for value in values]
    backend_weights: list[Any] = []
    for index, value in enumerate(floating_values):
        if is_torch(value):
            weight = torch_module().as_tensor(
                weights[index], dtype=value.dtype, device=value.device
            )
            if batch_size(value) is not None:
                weight = weight.unsqueeze(0).expand(value.shape[0], -1, -1)
        else:
            weight = weights[index].astype(value.dtype, copy=False)
        backend_weights.append(weight)
    effective = [
        weight * cast_mask_float(mask, value)
        for weight, mask, value in zip(backend_weights, masks, floating_values)
    ]
    total = _sum(
        _stack(
            [
                _where(expand_mask(mask, value), value, _zeros_like(value))
                * expand_mask(weight, value)
                for value, mask, weight in zip(floating_values, masks, backend_weights)
            ]
        ),
        0,
    )
    weight_sum = _sum(_stack(effective), 0)
    valid = weight_sum > 0
    denominator = _where(valid, weight_sum, weight_sum + 1)
    result = total / expand_mask(denominator, total)
    fill = (
        torch_module().full_like(result, float("nan"))
        if is_torch(result)
        else np.full_like(result, np.nan)
    )
    return _where(expand_mask(valid, result), result, fill), valid


def blend_closest(
    values: list[Any], masks: list[Any], specs: list[Any], shape: tuple[int, int]
) -> tuple[Any, Any]:
    scores_np = _erp_center_scores(specs, shape)
    first = values[0]
    if is_torch(first):
        torch = torch_module()
        scores = torch.as_tensor(scores_np, dtype=torch.float32, device=first.device)
        if batch_size(first) is not None:
            scores = scores.unsqueeze(0).expand(first.shape[0], -1, -1, -1)
            valid_stack = _stack(masks, 1)
            face_axis = 1
        else:
            valid_stack = _stack(masks, 0)
            face_axis = 0
        scores = torch.where(valid_stack, scores, torch.full_like(scores, -torch.inf))
    else:
        valid_stack = _stack(masks, 0)
        face_axis = 0
        scores = np.where(valid_stack, scores_np, -np.inf)
    selected = _argmax(scores, face_axis)
    support = (
        _sum(valid_stack.to(dtype=torch_module().int32), face_axis) > 0
        if is_torch(first)
        else np.any(valid_stack, axis=face_axis)
    )
    output = _zeros_like(first)
    for index, (value, mask) in enumerate(zip(values, masks)):
        choice = (selected == index) & mask & support
        output = _where(expand_mask(choice, value), value, output)
    return output, support


def blend_reprojected(
    values: list[Any],
    masks: list[Any],
    specs: list[Any],
    shape: tuple[int, int],
    method: Any,
) -> tuple[Any, Any]:
    if method == "average":
        return blend_average(values, masks)
    if method == "closest":
        return blend_closest(values, masks, specs, shape)
    if method == "gaussian":
        scores = _erp_center_scores(specs, shape)
        weights = np.exp(6.0 * (scores - 1.0))
        return blend_weighted(values, masks, weights)
    if isinstance(method, str):
        raise ValueError(
            "blend must be 'average', 'closest', 'gaussian', or a blender object"
        )
    if is_torch(values[0]):
        raise TypeError(
            "custom blender objects are only supported for NumPy workflow data"
        )
    if not hasattr(method, "blend"):
        raise TypeError("custom blender objects must provide blend(images, masks)")
    result = method.blend(values, masks, return_mask=True)
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("custom workflow blenders must return (array, validity_mask)")
    output, valid = result
    validate_mask(output, valid, name="blender validity")
    return output, valid


def resolved_blends(
    modality_names: list[str], metadata: Mapping[str, Mapping[str, Any]], blend: Any
) -> dict[str, Any]:
    defaults = {
        name: "closest" if metadata[name]["kind"] == "labels" else "average"
        for name in modality_names
    }
    if blend is None:
        return defaults
    if isinstance(blend, Mapping):
        unknown = set(blend) - set(modality_names)
        if unknown:
            raise KeyError(
                f"blend overrides reference unknown modalities: {sorted(unknown)}"
            )
        defaults.update(blend)
        return defaults
    if len(modality_names) != 1:
        raise ValueError("bundles require blend overrides as a mapping by modality")
    defaults[modality_names[0]] = blend
    return defaults
