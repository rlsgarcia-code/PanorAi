"""Translate preserved v1 experiment forms into catalog-derived executions.

This module owns only the finite compatibility vocabulary of the old UI.  It
does not implement PanorAi algorithms: every target is resolved from the
installed public catalog and executed by the generic component/plan engine.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from PIL import Image

from .discovery import component_index
from .planning import validate_plan


MIGRATED_LEGACY_IDS = frozenset(
    {
        "classical-stereo",
        "on-sphere",
        "projected-view-route",
        "region-analysis-pipeline",
        "filters-edges",
        "morphology",
        "region-ops",
        "dog-op",
        "geometry-contract",
    }
)


def _component(surface_id: str, symbol: str) -> str:
    matches = [
        descriptor["id"]
        for descriptor in component_index().values()
        if descriptor["surface_id"] == surface_id
        and descriptor["symbol"] == symbol
        and descriptor["status"] == "ready"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"legacy migration requires one ready {surface_id}.{symbol}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _uploads(
    inputs: list[dict[str, Any]],
    fields: tuple[str, ...],
    *,
    transform: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if len(inputs) < len(fields):
        raise ValueError(f"migration requires {len(fields)} uploaded image(s)")
    result = []
    for item, field in zip(inputs, fields, strict=False):
        migrated = deepcopy(item)
        migrated["field"] = field
        if transform is not None:
            migrated["workbench_transform"] = deepcopy(transform)
        result.append(migrated)
    return result


def _single_or_pair_plan(
    *,
    component_id: str,
    configuration: dict[str, Any],
    inputs: list[dict[str, Any]],
    params: dict[str, Any],
    field: str,
    transform: dict[str, Any] | None = None,
    input_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    values = input_values or {}
    target = str(params.get("_target", "a"))
    selected = (
        [(0, "a"), (1, "b")]
        if target == "both" and len(inputs) > 1
        else [(1, "b")]
        if target == "b" and len(inputs) > 1
        else [(0, "a")]
    )
    if len(selected) == 1:
        index, _ = selected[0]
        return {
            "component_id": component_id,
            "configuration": configuration,
            "input_values": deepcopy(values),
            "execution_inputs": _uploads(
                [inputs[index]], (field,), transform=transform
            ),
        }
    steps = []
    execution_inputs = []
    for index, step_id in selected:
        steps.append(
            {
                "id": step_id,
                "component_id": component_id,
                "configuration": deepcopy(configuration),
                "input_values": deepcopy(values),
                "bindings": {},
            }
        )
        scoped = _uploads([inputs[index]], (f"{step_id}.{field}",), transform=transform)
        execution_inputs.extend(scoped)
    return {
        "component_id": component_id,
        "plan": {"steps": validate_plan(steps, execution_inputs), "retain": ["a"]},
        "configuration": configuration,
        "input_values": {},
        "execution_inputs": execution_inputs,
    }


def _filter_execution(
    experiment_id: str, params: dict[str, Any], inputs: list[dict[str, Any]]
) -> dict[str, Any]:
    operation = str(params.get("operation", "gaussian"))
    targets = {
        "gaussian": ("spherical-ops-core", "GaussianBlur"),
        "box": ("spherical-ops-core", "BoxBlur"),
        "median": ("spherical-ops-core", "MedianBlur"),
        "bilateral": ("spherical-ops-core", "BilateralFilter"),
        "sobel": ("spherical-ops-core", "Sobel"),
        "gradient": ("spherical-ops-core", "Gradient"),
        "laplacian": ("spherical-ops-core", "Laplacian"),
        "canny": ("spherical-algorithms-general", "Canny"),
    }
    if experiment_id == "on-sphere" and operation not in {
        "gaussian",
        "sobel",
        "laplacian",
        "canny",
    }:
        raise ValueError(f"unsupported OnSphere operation {operation!r}")
    surface, symbol = targets[operation]
    ksize = int(params.get("ksize", 5))
    sigma = float(params.get("sigma", 1.0))
    if operation == "gaussian":
        configuration = {"ksize": ksize, "sigma": sigma}
    elif operation in {"box", "median"}:
        configuration = {"ksize": ksize}
    elif operation == "bilateral":
        configuration = {
            "ksize": ksize,
            "sigma_color": max(sigma, 0.01),
            "sigma_space": max(sigma, 0.01),
        }
    elif operation == "canny":
        configuration = {
            "threshold_low": float(params.get("threshold_low", 0.1)),
            "threshold_high": float(params.get("threshold_high", 0.3)),
            "gaussian_ksize": ksize,
            "gaussian_sigma": sigma,
        }
    else:
        configuration = {}
    return _single_or_pair_plan(
        component_id=_component(surface, symbol),
        configuration=configuration,
        inputs=inputs,
        params=params,
        field="image",
        transform={"kind": "rgb-float01"},
    )


def _mask_execution(
    experiment_id: str, params: dict[str, Any], inputs: list[dict[str, Any]]
) -> dict[str, Any]:
    operation = str(params.get("operation", "dilate"))
    threshold = float(params.get("threshold", 0.5))
    transform = {"kind": "grayscale-threshold", "threshold": threshold}
    if experiment_id == "morphology":
        if operation == "fill-holes":
            component_id = _component("spherical-region-analysis-ops", "fill_holes")
            field = "mask"
            values = {"max_hole_solid_angle_sr": 0.01}
        else:
            component_id = _component("spherical-ops-morphology", operation)
            field = "values"
            values = {"footprint": float(params.get("radius_deg", 1.0))}
    else:
        targets = {
            "connected-components": ("connected_components", "mask", {}),
            "measure": ("connected_components", "mask", {}),
            "boundaries": ("region_boundaries", "values", {}),
            "distance": ("geodesic_distance_transform", "mask", {}),
        }
        symbol, field, values = targets[operation]
        component_id = _component("spherical-region-analysis-ops", symbol)
    execution = _single_or_pair_plan(
        component_id=component_id,
        configuration={},
        inputs=inputs,
        params=params,
        field=field,
        transform=transform,
        input_values=values,
    )
    return execution


def _support(image: np.ndarray, policy: str) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = image.shape[:2]
    if policy == "full-sphere":
        return np.ones((height, width), dtype=bool), {
            "policy": policy,
            "detected_polar_voids": False,
            "valid_fraction": 1.0,
        }
    if policy != "auto-polar-voids":
        raise ValueError(f"unsupported input support policy {policy!r}")
    black = np.all(np.asarray(image)[..., :3] == 0, axis=-1)
    polar_border = (
        float(np.mean(black[0])) >= 0.98 and float(np.mean(black[-1])) >= 0.98
    )
    support = np.ones((height, width), dtype=bool)
    if polar_border:
        for x in range(width):
            observed = np.flatnonzero(~black[:, x])
            if observed.size == 0:
                support[:, x] = False
                continue
            support[: int(observed[0]), x] = False
            support[int(observed[-1]) + 1 :, x] = False
    invalid_fraction = 1.0 - float(np.mean(support))
    detected = bool(polar_border and invalid_fraction >= 0.005)
    if not detected:
        support.fill(True)
        invalid_fraction = 0.0
    return support, {
        "policy": policy,
        "detected_polar_voids": detected,
        "valid_fraction": 1.0 - invalid_fraction,
        "criterion": "exact-black runs connected to both polar borders",
    }


def decode_migrated_upload(item: dict[str, Any]) -> Any:
    """Apply a server-authored compatibility transform to one image upload."""

    transform = item.get("workbench_transform")
    if not isinstance(transform, dict):
        raise ValueError("migrated upload is missing its transform contract")
    with Image.open(item["path"]) as opened:
        opened.load()
        image = np.asarray(opened.convert("RGB"))
    kind = transform.get("kind")
    if kind == "rgb-float01":
        return image.astype(np.float32) / 255.0
    if kind == "grayscale-threshold":
        rgb = image.astype(np.float32) / 255.0
        gray = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
        return gray >= float(transform["threshold"])
    if kind not in {"working-rgb", "working-support"}:
        raise ValueError(f"unsupported migrated upload transform {kind!r}")
    support, _ = _support(image, str(transform["support_policy"]))
    requested = str(transform["working_width"])
    if requested == "original" or int(requested) >= image.shape[1]:
        return image if kind == "working-rgb" else support
    from panorai.ops import spherical as ops

    target_width = int(requested)
    target_height = max(1, round(image.shape[0] * target_width / image.shape[1]))
    target = (target_height, target_width)
    if kind == "working-support":
        return ops.Resize(target, interpolation="nearest")(support).astype(bool)
    return ops.Resize(target)(image.astype(np.float32) / 255.0)


def _classical_stereo(
    params: dict[str, Any], inputs: list[dict[str, Any]]
) -> dict[str, Any]:
    symbols = {
        "bm": "StereoBMDepth",
        "sgbm": "StereoSGBMDepth",
        "ad-census": "ADCensusStereoDepth",
        "census-mgm": "CensusMGMStereoDepth",
        "elas": "ELASStereoDepth",
        "patch-match": "PatchMatchStereoDepth",
    }
    symbol = symbols[str(params.get("algorithm", "sgbm"))]
    return {
        "component_id": _component("spherical-applications", symbol),
        "configuration": {
            "config": {"max_disparity": int(params.get("max_disparity", 64))}
        },
        "input_values": {},
        "execution_inputs": _uploads(inputs, ("reference", "source")),
    }


def _geometry(params: dict[str, Any], inputs: list[dict[str, Any]]) -> dict[str, Any]:
    if not inputs or not isinstance(inputs[0].get("image_shape_hw"), list):
        raise ValueError("Geometry v1 requires one decoded ERP")
    height, width = (int(value) for value in inputs[0]["image_shape_hw"])
    points = params.get("points_xy") or [[width / 2 - 0.5, height / 2 - 0.5]]
    return {
        "component_id": _component("canonical-geometry", "erp_pixels_to_rays"),
        "configuration": {},
        "input_values": {"pixels_xy": points, "image_shape_hw": [height, width]},
        "execution_inputs": [],
    }


def _dog(params: dict[str, Any], inputs: list[dict[str, Any]]) -> dict[str, Any]:
    if not inputs:
        raise ValueError("DoG requires one ERP")
    base = inputs[0]
    common = {
        "working_width": str(params.get("working_width", "1024")),
        "support_policy": str(params.get("input_support", "auto-polar-voids")),
    }
    panorama = _uploads(
        [base],
        ("panorama",),
        transform={"kind": "working-rgb", **common},
    )[0]
    validity = _uploads(
        [base],
        ("validity_mask",),
        transform={"kind": "working-support", **common},
    )[0]
    source_shape = base.get("image_shape_hw")
    return {
        "component_id": _component("spherical-ops-dog-detector", "DoGDetector"),
        "configuration": {
            "config": {
                "max_keypoints": int(params.get("max_features", 1000)),
                "contrast_threshold": float(params.get("contrast_threshold", 0.012)),
            }
        },
        "input_values": {"panorama_id": base.get("name", "workbench")},
        "execution_inputs": [panorama, validity],
        "render_context": {"source_shape_hw": source_shape},
    }


def _projected_route(
    params: dict[str, Any], inputs: list[dict[str, Any]]
) -> dict[str, Any]:
    if not inputs:
        raise ValueError("Projected route requires one ERP")
    size = int(params.get("face_size", 512))
    height, width = (int(value) for value in inputs[0]["image_shape_hw"])
    steps = [
        {
            "id": "sample",
            "component_id": _component("spherical-pipeline-core", "SampleViews"),
            "configuration": {
                "layout": "fibonacci",
                "count": 6,
                "size": size,
                "fov": float(params.get("hfov_deg", 90.0)),
            },
            "input_values": {},
            "bindings": {},
        },
        {
            "id": "reconstruct",
            "component_id": next(
                descriptor["id"]
                for descriptor in component_index().values()
                if descriptor["surface_id"] == "spherical-pipeline-core"
                and descriptor["symbol"] == "ReconstructERP"
            ),
            "configuration": {
                "blend": {"image": "average"},
                "eq_shape": [height, width],
            },
            "input_values": {},
            "bindings": {"value": {"step": "sample", "path": ["value"]}},
        },
    ]
    execution_inputs = _uploads(
        [inputs[0]],
        ("sample.value",),
        transform={"kind": "rgb-float01"},
    )
    retain = ["sample"] if params.get("retain", "views") == "views" else []
    return {
        "component_id": steps[-1]["component_id"],
        "plan": {"steps": validate_plan(steps, execution_inputs), "retain": retain},
        "configuration": steps[-1]["configuration"],
        "input_values": {},
        "execution_inputs": execution_inputs,
    }


def _region_analysis(
    params: dict[str, Any], inputs: list[dict[str, Any]]
) -> dict[str, Any]:
    transform = {
        "kind": "grayscale-threshold",
        "threshold": float(params.get("threshold", 0.5)),
    }
    steps = [
        {
            "id": "labels",
            "component_id": _component(
                "spherical-region-analysis-ops", "connected_components"
            ),
            "configuration": {},
            "input_values": {},
            "bindings": {},
        },
        {
            "id": "boundaries",
            "component_id": _component(
                "spherical-region-analysis-ops", "region_boundaries"
            ),
            "configuration": {},
            "input_values": {},
            "bindings": {"values": {"step": "labels", "path": []}},
        },
        {
            "id": "measurements",
            "component_id": _component(
                "spherical-region-analysis-ops", "measure_regions"
            ),
            "configuration": {},
            "input_values": {},
            "bindings": {"labels": {"step": "labels", "path": []}},
        },
        {
            "id": "adjacency",
            "component_id": _component(
                "spherical-region-analysis-ops", "region_adjacency_graph"
            ),
            "configuration": {},
            "input_values": {},
            "bindings": {"labels": {"step": "labels", "path": []}},
        },
    ]
    execution_inputs = _uploads(inputs, ("labels.mask",), transform=transform)
    return {
        "component_id": steps[-1]["component_id"],
        "plan": {
            "steps": validate_plan(steps, execution_inputs),
            "retain": ["labels", "boundaries", "measurements"],
        },
        "configuration": {},
        "input_values": {},
        "execution_inputs": execution_inputs,
    }


def migrate_legacy_execution(
    experiment_id: str,
    params: dict[str, Any],
    inputs: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return generic execution fields for one migrated compatibility alias."""

    if experiment_id not in MIGRATED_LEGACY_IDS:
        return None
    if experiment_id in {"filters-edges", "on-sphere"}:
        return _filter_execution(experiment_id, params, inputs)
    if experiment_id in {"morphology", "region-ops"}:
        return _mask_execution(experiment_id, params, inputs)
    if experiment_id == "classical-stereo":
        return _classical_stereo(params, inputs)
    if experiment_id == "geometry-contract":
        return _geometry(params, inputs)
    if experiment_id == "dog-op":
        return _dog(params, inputs)
    if experiment_id == "projected-view-route":
        return _projected_route(params, inputs)
    if experiment_id == "region-analysis-pipeline":
        return _region_analysis(params, inputs)
    raise AssertionError(experiment_id)


__all__ = [
    "MIGRATED_LEGACY_IDS",
    "decode_migrated_upload",
    "migrate_legacy_execution",
]
