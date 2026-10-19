from fastapi.testclient import TestClient

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
