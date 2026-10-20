from fastapi.testclient import TestClient
from io import BytesIO
import json
import time

import numpy as np
from PIL import Image

from panorai_workbench.api import app


client = TestClient(app)


def test_health_identifies_local_workbench() -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "panorai-workbench/v0",
    }


def test_semantic_runner_fails_closed_until_implemented() -> None:
    response = client.get("/api/v1/semantics/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is False
    assert payload["status"] == "not-implemented"


def test_academy_single_view_routes_manual_points_through_panorai() -> None:
    status = client.get("/api/v1/academy/single-view/status")
    assert status.status_code == 200
    assert status.json()["case"] == "single-view-geometry"

    response = client.post(
        "/api/v1/academy/single-view/rays",
        json={"image_shape_hw": [4, 8], "points_xy": [[3.5, 1.5]]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["execution"]["public_api"] == ("panorai.geometry.erp_pixels_to_rays")
    assert payload["rays_xyz"] == [[0.0, 0.0, 1.0]]


def test_runtime_identifies_exact_panorai_installation() -> None:
    response = client.get("/api/v1/runtime")
    assert response.status_code == 200
    payload = response.json()
    assert payload["schema"] == "panorai-workbench-runtime/v1"
    assert payload["panorai"]["version"]
    assert payload["panorai"]["module_path"].endswith("panorai/__init__.py")
    assert payload["panorai"]["installation_type"] in {
        "editable",
        "installed-distribution",
        "source-checkout",
        "vcs-install",
    }
    assert payload["devices"]["cpu"] is True


def test_catalog_exposes_installed_surfaces_and_local_experiments() -> None:
    response = client.get("/api/v1/experiments")
    assert response.status_code == 200
    payload = response.json()
    assert payload["schema"] == "panorai-workbench-catalog/v2"
    assert payload["catalog"]["interface"] == "panorai-catalog/v1"
    assert payload["surfaces"]
    assert any(item["status"] == "ready" for item in payload["components"])
    by_id = {component["id"]: component for component in payload["experiments"]}
    assert by_id["relative-pose"]["surface_id"] == "spherical-relative-pose"
    assert (
        by_id["relative-pose"]["public_api"]
        == "panorai.estimators.estimate_relative_pose"
    )
    assert by_id["relative-pose"]["workbench_support"] == "execute"
    assert by_id["on-sphere"]["public_api"] == "panorai.pipeline.OnSphere"
    assert by_id["filters-edges"]["public_api"] == (
        "panorai.ops.spherical.GaussianBlur"
    )
    assert by_id["spherical-slic"]["surface_id"] == "spherical-applications-slic"
    assert by_id["spherical-slic"]["public_api"] == (
        "panorai.applications.SphericalSLIC"
    )
    assert by_id["spherical-slic"]["workbench_support"] == "execute"
    assert by_id["spherical-dog-features"]["public_api"] == (
        "panorai.applications.SphericalDoGFeatures"
    )
    assert by_id["spherical-stereo"]["public_api"] == (
        "panorai.applications.SphericalStereoDepth"
    )
    assert by_id["gnomonic-projection"]["surface_id"] == "canonical-geometry"


def test_v2_runner_rejects_arbitrary_import_paths() -> None:
    response = client.post(
        "/api/v2/runs",
        data={
            "plan": json.dumps(
                {
                    "component_id": "subprocess.run",
                    "configuration": {},
                    "input_values": {},
                    "file_fields": [],
                }
            )
        },
    )
    assert response.status_code == 422
    assert "unknown component" in response.json()["detail"]


def test_v2_runner_accepts_dynamic_named_uploads() -> None:
    catalog = client.get("/api/v1/catalog").json()
    gaussian = next(
        item
        for item in catalog["components"]
        if item["surface_id"] == "spherical-ops-core"
        and item["symbol"] == "GaussianBlur"
    )
    stream = BytesIO()
    Image.fromarray(np.full((8, 16, 3), 127, dtype=np.uint8)).save(stream, format="PNG")
    response = client.post(
        "/api/v2/runs",
        data={
            "plan": json.dumps(
                {
                    "component_id": gaussian["id"],
                    "configuration": {"ksize": 3},
                    "input_values": {},
                    "file_fields": ["image"],
                }
            )
        },
        files={"files": ("erp.png", stream.getvalue(), "image/png")},
    )
    assert response.status_code == 200, response.text
    run = response.json()
    for _ in range(100):
        if run["status"] not in {"created", "running"}:
            break
        time.sleep(0.02)
        run = client.get(f"/api/v1/runs/{run['id']}").json()
    assert run["status"] == "completed", run.get("error")
    assert run["component_id"] == gaussian["id"]
    assert run["inputs"][0]["field"] == "image"


def test_catalog_alias_and_workspace_routes_share_the_same_contract() -> None:
    catalog = client.get("/api/v1/catalog")
    alias = client.get("/api/v1/experiments")
    single = client.get("/api/v1/workspaces/single")
    two = client.get("/api/v1/workspaces/two")
    assert catalog.status_code == alias.status_code == 200
    assert catalog.json()["surfaces"] == alias.json()["surfaces"]
    assert catalog.json()["experiments"] == alias.json()["experiments"]
    assert single.status_code == two.status_code == 200
    assert single.json()["workspace"] == "single"
    assert two.json()["workspace"] == "two"
    assert "experiments" in single.json() and "surfaces" not in single.json()
    assert client.get("/api/v1/workspaces/multi").status_code == 404
