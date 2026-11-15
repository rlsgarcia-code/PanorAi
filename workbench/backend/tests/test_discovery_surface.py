from __future__ import annotations

from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from panorai_workbench.api import app
from panorai_workbench.catalog import experiment_catalog
from panorai_workbench.discovery_surface import (
    SCHEMA,
    discovery_catalog,
    recommendations,
    validate_context,
)


client = TestClient(app)


def _goal(payload: dict, goal_id: str) -> dict:
    return next(item for item in payload["recommendations"] if item["id"] == goal_id)


def test_default_discovery_is_goal_oriented_and_deterministic() -> None:
    discovery = discovery_catalog()
    assert discovery["schema"] == SCHEMA
    assert discovery["default_sample_id"] == "poly-haven-studio-cc0"
    assert discovery["default_context"]["inputs"][0]["kind"] == "erp"
    assert discovery["default_context"]["inputs"][0]["role"] == "a"
    assert discovery["default_context"]["inputs"][0]["semantic"] == "panorai:rgb-erp"

    first = recommendations(discovery["default_context"])
    second = recommendations(deepcopy(discovery["default_context"]))
    assert first == second
    assert _goal(first, "features")["recommended"]["id"] == "spherical-dog"
    assert _goal(first, "regions")["recommended"]["id"] == "spherical-slic"
    assert _goal(first, "align-pair")["state"] == "needs-input"
    assert _goal(first, "align-pair")["recommended"]["missing"] == ["erp-b"]
    assert _goal(first, "analyze-results")["state"] == "needs-input"
    histogram_recipes = [
        item
        for item in [
            _goal(first, "analyze-results")["recommended"],
            *_goal(first, "analyze-results")["alternatives"],
        ]
        if "histogram" in item["title"].casefold()
    ]
    assert histogram_recipes
    assert all(item["state"] != "ready" for item in histogram_recipes)
    assert all("scalar-raster" in item["missing"] for item in histogram_recipes)


def test_scalar_run_artifact_unlocks_scalar_histograms_without_treating_rgb_as_scalar() -> (
    None
):
    discovery = discovery_catalog()
    context = deepcopy(discovery["default_context"])
    context["inputs"].append(
        {
            "id": "run-42-range",
            "kind": "run-artifact",
            "role": "auxiliary",
            "source": "history",
            "name": "range radial",
            "semantic": "panorai:scalar-raster",
            "shape_hw": [512, 1024],
        }
    )

    result = recommendations(context)
    analysis = _goal(result, "analyze-results")
    assert analysis["state"] == "ready"
    assert "scalar-raster" in result["capabilities"]
    assert "histogram" in analysis["recommended"]["title"].casefold()
    assert analysis["recommended"]["missing"] == []


def test_pair_and_pose_unlock_pair_and_stereo_without_image_inspection() -> None:
    context = {
        "inputs": [
            {"id": "a", "kind": "erp", "role": "a", "source": "upload"},
            {"id": "b", "kind": "erp", "role": "b", "source": "upload"},
            {"id": "pose", "kind": "pose", "role": "auxiliary", "source": "user"},
        ],
        "facts": {},
    }
    payload = recommendations(context)
    assert _goal(payload, "align-pair")["state"] == "ready"
    assert _goal(payload, "depth-range")["state"] == "ready"
    assert _goal(payload, "depth-range")["recommended"]["missing"] == []


def test_height_unlocks_metric_planar_recipe_but_not_stereo() -> None:
    discovery = discovery_catalog()
    context = deepcopy(discovery["default_context"])
    context["facts"] = {"camera_height_m": 1.62}
    payload = recommendations(context)
    planes = _goal(payload, "planes")
    known_height = next(
        item
        for item in [planes["recommended"], *planes["alternatives"]]
        if item["id"] == "known-height-planar"
    )
    assert known_height["state"] == "ready"
    assert _goal(payload, "depth-range")["state"] == "needs-input"


def test_optional_sam_remains_visible_with_explicit_diagnostic() -> None:
    result = recommendations(discovery_catalog()["default_context"])
    regions = _goal(result, "regions")
    sam = next(item for item in regions["alternatives"] if item["id"] == "cam-sam")

    assert sam["state"] == "unavailable"
    assert "deep-learning-sam" in sam["diagnostic"]
    assert "baixado automaticamente" in sam["diagnostic"]


def test_new_typed_component_is_discoverable_without_a_recipe_adapter() -> None:
    catalog = experiment_catalog()
    synthetic = {
        "id": "auto--future--spherical-edge-probe",
        "surface_id": "future-surface",
        "symbol": "SphericalEdgeProbe",
        "title": "SphericalEdgeProbe",
        "summary": "Detect spherical edges.",
        "public_api": "panorai.future.SphericalEdgeProbe",
        "execution_kind": "operation",
        "status": "ready",
        "diagnostic": None,
        "tier": "experimental",
        "roles": ["operation"],
        "requires": [],
        "configuration_schema": [],
        "input_schema": [
            {
                "name": "image",
                "required": True,
                "default": None,
                "annotation": "EquirectangularImage",
                "kind": "erp-image",
            }
        ],
        "output_schema": {
            "name": "result",
            "required": True,
            "default": None,
            "annotation": "ndarray",
            "kind": "array",
            "semantic": "panorai:edges",
        },
        "route": "/runner.html?component=auto--future--spherical-edge-probe",
    }
    catalog["components"] = [*catalog["components"], synthetic]
    projection = discovery_catalog(catalog)
    classification = next(
        item
        for item in projection["component_discovery"]
        if item["component_id"] == synthetic["id"]
    )
    assert classification == {
        "component_id": synthetic["id"],
        "goal_ids": ["filter-inspect"],
        "classification": "discoverable",
        "reason": None,
    }
    assert any(
        recipe["source"] == "automatic"
        and recipe["steps"][0]["component_id"] == synthetic["id"]
        for recipe in projection["recipes"]
    )


def test_every_ready_component_is_discoverable_or_explicitly_technical_only() -> None:
    catalog = experiment_catalog()
    projection = discovery_catalog(catalog)
    ready_ids = {
        item["id"] for item in catalog["components"] if item["status"] == "ready"
    }
    classified = {item["component_id"] for item in projection["component_discovery"]}
    assert classified == ready_ids
    assert all(
        item["classification"] in {"discoverable", "technical-only"}
        for item in projection["component_discovery"]
    )
    assert all(
        item["reason"]
        for item in projection["component_discovery"]
        if item["classification"] == "technical-only"
    )


def test_context_is_closed_and_never_accepts_imports_or_image_bytes() -> None:
    with pytest.raises(ValueError, match="unknown discovery context"):
        validate_context({"import": "subprocess.run"})
    with pytest.raises(ValueError, match="unsupported metadata"):
        validate_context(
            {
                "inputs": [
                    {
                        "kind": "erp",
                        "role": "a",
                        "source": "upload",
                        "bytes": "not-accepted",
                    }
                ]
            }
        )
    with pytest.raises(ValueError, match="PanorAi semantic ID"):
        validate_context(
            {
                "inputs": [
                    {
                        "kind": "run-artifact",
                        "role": "auxiliary",
                        "source": "history",
                        "semantic": "numpy.ndarray",
                    }
                ]
            }
        )


def test_discovery_endpoints_keep_catalog_separate() -> None:
    discovery = client.get("/api/v1/discovery")
    assert discovery.status_code == 200
    assert discovery.json()["schema"] == SCHEMA
    catalog = client.get("/api/v1/catalog").json()
    assert catalog["schema"] == "panorai-workbench-catalog/v2"
    assert "recommendations" not in catalog

    response = client.post(
        "/api/v1/discovery/recommendations",
        json=discovery.json()["default_context"],
    )
    assert response.status_code == 200
    assert response.json()["schema"] == SCHEMA
    invalid = client.post(
        "/api/v1/discovery/recommendations",
        json={"component_id": "subprocess.run"},
    )
    assert invalid.status_code == 422
