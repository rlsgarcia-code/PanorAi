from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import math
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from panorai.calibration import PinholeIntrinsics, pinhole_pixels_to_rays

from panorai.reconstruction import (
    RegisteredSphericalScene,
    SphericalDenseReconstructionResult,
    SphericalPairwisePoseEdge,
    SurfaceMesh,
)
import panorai_streetview.api as api_module
from panorai_streetview.api import create_app
from panorai_streetview.dense_preview import _reference_validity
import panorai_streetview.global_map as global_map_module
import panorai_streetview.store as store_module
from panorai_streetview.models import (
    DenseAnnotation,
    FloorCalibrationRequest,
    ManualCorrespondence,
    MatchReview,
)
from panorai_streetview.setup import scene_setup


def png(width: int = 16, height: int = 8) -> bytes:
    output = BytesIO()
    Image.new("RGB", (width, height), (24, 48, 72)).save(output, format="PNG")
    return output.getvalue()


def client(tmp_path) -> TestClient:
    return TestClient(create_app(tmp_path / "data"))


def textured_png(shift: int = 0) -> bytes:
    rng = np.random.default_rng(228)
    image = rng.integers(0, 256, size=(128, 256, 3), dtype=np.uint8)
    image[24:72, 38:98] = (220, 180, 40)
    image[60:110, 156:218] = (35, 200, 170)
    image = np.roll(image, shift, axis=1)
    output = BytesIO()
    Image.fromarray(image).save(output, format="PNG")
    return output.getvalue()


def perspective_oracle_png(yaw_deg: float) -> bytes:
    shape = (240, 320)
    angle = math.radians(yaw_deg)
    rotation = np.asarray(
        (
            (math.cos(angle), 0.0, math.sin(angle)),
            (0.0, 1.0, 0.0),
            (-math.sin(angle), 0.0, math.cos(angle)),
        )
    )
    intrinsics = PinholeIntrinsics.from_horizontal_fov(shape, 92.0)
    rows, columns = np.indices(shape, dtype=np.float64)
    rays = pinhole_pixels_to_rays(np.stack((columns, rows), axis=-1), intrinsics)
    world = rays @ rotation.T
    flat = world.reshape(-1, 3)
    random = np.random.default_rng(4207)
    centres = random.normal(size=(180, 3))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    colours = random.uniform(0.15, 1.0, size=(180, 3))
    weights = np.exp(np.clip((flat @ centres.T - 1.0) / 0.0009, -80.0, 0.0))
    blobs = weights @ colours / np.maximum(weights.sum(axis=1, keepdims=True), 0.35)
    longitude = np.arctan2(flat[:, 0], flat[:, 2])
    latitude = np.arcsin(np.clip(flat[:, 1], -1.0, 1.0))
    waves = np.stack(
        (
            np.sin(17.0 * longitude + 7.0 * latitude),
            np.sin(23.0 * longitude - 11.0 * latitude + 0.7),
            np.sin(13.0 * longitude + 19.0 * latitude + 1.4),
        ),
        axis=1,
    )
    image = np.rint(np.clip(42.0 + 185.0 * blobs + 28.0 * waves, 0, 255)).astype(
        np.uint8
    )
    image = image.reshape(shape + (3,))
    output = BytesIO()
    Image.fromarray(image).save(output, format="PNG")
    return output.getvalue()


def create_pair_project(
    api: TestClient, workflow: str = "spherical_unregistered"
) -> dict:
    response = api.post(
        "/api/v1/projects",
        data={
            "name": "Evidence pair",
            "workflow": workflow,
            "metadata": "{}",
            "pairs": json.dumps(
                [
                    {
                        "id": "pair-1",
                        "reference_filename": "a.png",
                        "source_filename": "b.png",
                    }
                ]
            ),
        },
        files=[
            ("images", ("a.png", textured_png(), "image/png")),
            ("images", ("b.png", textured_png(8), "image/png")),
        ],
    )
    assert response.status_code == 200
    return response.json()


def create_three_view_project(api: TestClient) -> dict:
    response = api.post(
        "/api/v1/projects",
        data={
            "name": "Metric dense scene",
            "workflow": "spherical_unregistered",
            "metadata": "{}",
            "pairs": "[]",
        },
        files=[
            ("images", ("a.png", textured_png(), "image/png")),
            ("images", ("b.png", textured_png(4), "image/png")),
            ("images", ("c.png", textured_png(8), "image/png")),
        ],
    )
    assert response.status_code == 200, response.text
    return response.json()


def store_metric_map(
    api: TestClient,
    project: dict,
    *,
    scale="metric-baseline-prior",
    image_count: int | None = None,
):
    mapped_images = project["images"][:image_count]
    centers = np.linspace(-0.2, 0.2, len(mapped_images))
    record = {
        "schema": "panorai-streetview-global-map/v1",
        "project_id": project["id"],
        "revision": 1,
        "created_at": "2026-10-19T23:38:00+00:00",
        "stale": False,
        "success": True,
        "failure_reasons": [],
        "scale": scale,
        "reference_image_id": mapped_images[1]["id"],
        "input_edge_revisions": {},
        "range_prior_candidates": [],
        "range_priors_applied": 0,
        "range_prior_min_count": 3,
        "edges": [],
        "skipped_pairs": [],
        "poses": [
            {
                "image_id": image["id"],
                "rotation_world_to_panorama": np.eye(3).tolist(),
                "center_world": [centers[index], 0.0, 0.0],
                "display_xy": [centers[index], 0.0],
                "active_track_count": 12,
                "max_reprojection_deg": 0.1,
            }
            for index, image in enumerate(mapped_images)
        ],
        "points": {"total_count": 0, "displayed_count": 0, "display_xy": []},
        "track_count": 12,
        "diagnostics": {},
        "recommended_edit": {
            "kind": "accepted",
            "pair_id": None,
            "title": "accepted",
            "detail": "accepted",
        },
    }
    api.app.state.global_map._write(record)
    return record


def create_project(
    api: TestClient, *, workflow: str = "spherical_unregistered", metadata=None
):
    return api.post(
        "/api/v1/projects",
        data={
            "name": "Boiler route",
            "workflow": workflow,
            "metadata": json.dumps(metadata or {}),
        },
        files=[("images", ("pano.png", png(), "image/png"))],
    )


def test_create_reload_and_plan_all_four_workflows(tmp_path) -> None:
    api = client(tmp_path)
    for workflow in (
        "spherical_registered",
        "spherical_unregistered",
        "perspective_calibrated",
        "perspective_uncalibrated",
    ):
        response = create_project(api, workflow=workflow)
        assert response.status_code == 200
        project = response.json()
        assert project["workflow"] == workflow
        assert len(project["images"]) == 1
        plan = api.get(f"/api/v1/projects/{project['id']}/build-plan")
        assert plan.status_code == 200
        assert plan.json()["workflow"] == workflow

    listed = api.get("/api/v1/projects").json()["projects"]
    assert len(listed) == 4


def test_line_and_vanishing_surfaces_are_not_exposed(tmp_path) -> None:
    api = client(tmp_path)
    project = create_project(api).json()
    image_id = project["images"][0]["id"]
    response = api.post(
        f"/api/v1/projects/{project['id']}/constraints",
        json={
            "kind": "line_segment",
            "image_id": image_id,
            "group_id": "vertical-1",
            "axis_hint": "z",
            "points_xy": [[2.5, 1.5], [2.5, 7.0]],
            "state": "accepted",
        },
    )
    assert response.status_code == 404
    store = api.app.state.store
    project_path = store._project_path(project["id"])
    legacy_payload = json.loads(project_path.read_text(encoding="utf-8"))
    legacy_payload["constraints"] = [
        {
            "id": "legacy-line",
            "kind": "line_segment",
            "image_id": image_id,
            "group_id": "vertical-1",
            "axis_hint": "z",
            "points_xy": [[2.5, 1.5], [2.5, 7.0]],
            "state": "accepted",
            "created_at": "2026-10-19T00:00:00+00:00",
        }
    ]
    project_path.write_text(json.dumps(legacy_payload), encoding="utf-8")
    stored = api.get(f"/api/v1/projects/{project['id']}").json()
    assert stored["id"] == project["id"]
    assert "constraints" not in stored


def test_floor_calibration_is_revisioned_and_survives_backend_restart(tmp_path) -> None:
    data_root = tmp_path / "data"
    api = TestClient(create_app(data_root))
    response = api.post(
        "/api/v1/projects",
        data={
            "name": "Single-view metric room",
            "workflow": "spherical_unregistered",
            "metadata": "{}",
        },
        files=[("images", ("room.png", png(360, 180), "image/png"))],
    )
    assert response.status_code == 200, response.text
    project = response.json()
    image = project["images"][0]

    def pixel(ray):
        value = np.asarray(ray, dtype=np.float64)
        value /= np.linalg.norm(value)
        longitude = math.atan2(value[0], value[2])
        latitude = math.asin(value[1])
        return [
            ((longitude + math.pi) / (2 * math.pi)) * image["width"] - 0.5,
            ((math.pi / 2 - latitude) / math.pi) * image["height"] - 0.5,
        ]

    camera_height = 1.6
    payload = {
        "image_id": image["id"],
        "camera_height_m": camera_height,
        "floor_hint_xy": pixel((0, -camera_height, 3)),
        "lines": [
            {
                "family": "floor_a",
                "start_xy": pixel((-3, -camera_height, 2)),
                "end_xy": pixel((3, -camera_height, 2)),
            },
            {
                "family": "floor_a",
                "start_xy": pixel((-3, -camera_height, 5)),
                "end_xy": pixel((3, -camera_height, 5)),
            },
            {
                "family": "floor_b",
                "start_xy": pixel((2, -camera_height, 1)),
                "end_xy": pixel((2, -camera_height, 6)),
            },
            {
                "family": "floor_b",
                "start_xy": pixel((-2, -camera_height, 1)),
                "end_xy": pixel((-2, -camera_height, 6)),
            },
        ],
    }
    outside_payload = json.loads(json.dumps(payload))
    outside_payload["lines"][0]["end_xy"] = [image["width"], 10.0]
    outside_calibration = api.post(
        f"/api/v1/projects/{project['id']}/metrology/calibrate",
        json=outside_payload,
    )
    assert outside_calibration.status_code == 422
    assert outside_calibration.json()["detail"] == (
        "metrology line 1 end must lie inside the source image"
    )
    first = api.post(
        f"/api/v1/projects/{project['id']}/metrology/calibrate", json=payload
    )
    assert first.status_code == 200, first.text
    assert first.json()["revision"] == 1
    assert first.json()["engine"]["public_api"] == (
        "panorai.experimental.single_view_metrology"
    )
    second = api.post(
        f"/api/v1/projects/{project['id']}/metrology/calibrate", json=payload
    )
    assert second.status_code == 200, second.text
    assert second.json()["revision"] == 2

    restarted = TestClient(create_app(data_root))
    loaded = restarted.get(f"/api/v1/projects/{project['id']}/metrology/{image['id']}")
    assert loaded.status_code == 200
    assert loaded.json()["revision"] == 2
    history = restarted.get(
        f"/api/v1/projects/{project['id']}/metrology/{image['id']}/history"
    ).json()
    assert [item["revision"] for item in history["revisions"]] == [1, 2]

    floor = restarted.post(
        f"/api/v1/projects/{project['id']}/metrology/floor-point",
        json={
            "image_id": image["id"],
            "calibration_revision": 2,
            "point_xy": pixel((2, -camera_height, 4)),
        },
    )
    assert floor.status_code == 200, floor.text
    assert floor.json()["calibration_revision"] == 2
    assert floor.json()["horizontal_distance_m"] == pytest.approx(math.sqrt(20))
    stale_floor = restarted.post(
        f"/api/v1/projects/{project['id']}/metrology/floor-point",
        json={
            "image_id": image["id"],
            "calibration_revision": 1,
            "point_xy": pixel((2, -camera_height, 4)),
        },
    )
    assert stale_floor.status_code == 409
    assert stale_floor.json()["detail"] == (
        "floor calibration revision changed; reload measurements"
    )
    outside_floor = restarted.post(
        f"/api/v1/projects/{project['id']}/metrology/floor-point",
        json={
            "image_id": image["id"],
            "calibration_revision": 2,
            "point_xy": [-0.5, 20.0],
        },
    )
    assert outside_floor.status_code == 422
    assert outside_floor.json()["detail"] == (
        "floor point must lie inside the source image"
    )
    vertical = restarted.post(
        f"/api/v1/projects/{project['id']}/metrology/vertical",
        json={
            "image_id": image["id"],
            "calibration_revision": 2,
            "base_xy": pixel((2, -camera_height, 4)),
            "top_xy": pixel((2, 1.0 - camera_height, 4)),
        },
    )
    assert vertical.status_code == 200, vertical.text
    assert vertical.json()["calibration_revision"] == 2
    assert vertical.json()["height_m"] == pytest.approx(1.0)
    stale_vertical = restarted.post(
        f"/api/v1/projects/{project['id']}/metrology/vertical",
        json={
            "image_id": image["id"],
            "calibration_revision": 1,
            "base_xy": pixel((2, -camera_height, 4)),
            "top_xy": pixel((2, 1.0 - camera_height, 4)),
        },
    )
    assert stale_vertical.status_code == 409
    outside_vertical = restarted.post(
        f"/api/v1/projects/{project['id']}/metrology/vertical",
        json={
            "image_id": image["id"],
            "calibration_revision": 2,
            "base_xy": pixel((2, -camera_height, 4)),
            "top_xy": [20.0, image["height"]],
        },
    )
    assert outside_vertical.status_code == 422
    assert outside_vertical.json()["detail"] == (
        "vertical top must lie inside the source image"
    )
    capabilities = api.get("/api/v1/capabilities").json()["engines"]
    assert "line_fit" not in capabilities
    assert "vanishing_directions" not in capabilities
    plan = api.get(f"/api/v1/projects/{project['id']}/build-plan").json()
    stage_ids = {stage["id"] for stage in plan["stages"]}
    assert "manual-orientation" not in stage_ids
    assert "manual-calibration-evidence" not in stage_ids


def test_floor_calibration_revisions_are_serialized_per_image(tmp_path) -> None:
    api = client(tmp_path)
    project = create_project(api).json()
    image = project["images"][0]

    def pixel(ray):
        value = np.asarray(ray, dtype=np.float64)
        value /= np.linalg.norm(value)
        longitude = math.atan2(value[0], value[2])
        latitude = math.asin(value[1])
        return (
            ((longitude + math.pi) / (2 * math.pi)) * image["width"] - 0.5,
            ((math.pi / 2 - latitude) / math.pi) * image["height"] - 0.5,
        )

    height = 1.6
    request = FloorCalibrationRequest(
        image_id=image["id"],
        camera_height_m=height,
        floor_hint_xy=pixel((0, -height, 3)),
        lines=[
            {
                "family": "floor_a",
                "start_xy": pixel((-3, -height, 2)),
                "end_xy": pixel((3, -height, 2)),
            },
            {
                "family": "floor_a",
                "start_xy": pixel((-3, -height, 5)),
                "end_xy": pixel((3, -height, 5)),
            },
            {
                "family": "floor_b",
                "start_xy": pixel((2, -height, 1)),
                "end_xy": pixel((2, -height, 6)),
            },
            {
                "family": "floor_b",
                "start_xy": pixel((-2, -height, 1)),
                "end_xy": pixel((-2, -height, 6)),
            },
        ],
    )
    service = api.app.state.metrology
    original_write = service._write
    first_writing = Event()
    second_writing = Event()
    release_first = Event()
    counter_lock = Lock()
    write_count = 0

    def delayed_write(record):
        nonlocal write_count
        with counter_lock:
            write_count += 1
            sequence = write_count
        if sequence == 1:
            first_writing.set()
            assert release_first.wait(timeout=2)
        elif sequence == 2:
            second_writing.set()
        original_write(record)

    service._write = delayed_write
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(service.calibrate, project["id"], request)
        assert first_writing.wait(timeout=2)
        second = executor.submit(service.calibrate, project["id"], request)
        assert not second_writing.wait(timeout=0.1)
        release_first.set()
        results = [first.result(timeout=2), second.result(timeout=2)]

    assert sorted(item["revision"] for item in results) == [1, 2]
    assert [
        item["revision"]
        for item in service.history(project["id"], image["id"])["revisions"]
    ] == [1, 2]


def test_single_view_metrology_rejects_perspective_workflows(tmp_path) -> None:
    api = client(tmp_path)
    project = create_project(api, workflow="perspective_uncalibrated").json()
    image_id = project["images"][0]["id"]
    response = api.post(
        f"/api/v1/projects/{project['id']}/metrology/calibrate",
        json={
            "image_id": image_id,
            "camera_height_m": 1.6,
            "floor_hint_xy": [4.0, 6.0],
            "lines": [
                {"family": "floor_a", "start_xy": [0, 1], "end_xy": [4, 1]},
                {"family": "floor_a", "start_xy": [0, 2], "end_xy": [4, 2]},
                {"family": "floor_b", "start_xy": [1, 0], "end_xy": [1, 4]},
                {"family": "floor_b", "start_xy": [2, 0], "end_xy": [2, 4]},
            ],
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "single-view metrology requires a spherical ERP workflow"
    )


def test_pair_registration_rejects_perspective_workflows_before_features(
    tmp_path,
) -> None:
    api = client(tmp_path)
    project = create_pair_project(api, "perspective_uncalibrated")

    response = api.post(
        f"/api/v1/projects/{project['id']}/pairs/pair-1/registration",
        json={"region_mode": "all"},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "pair registration requires a spherical ERP workflow"
    )


def test_global_map_binds_edge_evidence_to_the_same_registration_snapshot(
    tmp_path, monkeypatch
) -> None:
    api = client(tmp_path)
    project = create_pair_project(api)
    service = api.app.state.global_map
    registration = api.app.state.registration
    snapshot = {
        "revision": 1,
        "pose": {
            "num_inliers": 17,
            "quality": {"raw_quality_score": 0.91},
        },
    }
    later = {
        "revision": 2,
        "pose": {
            "num_inliers": 9,
            "quality": {"raw_quality_score": 0.42},
        },
    }
    snapshot_edge = SimpleNamespace(
        pair=tuple(item["id"] for item in project["images"]),
        pose=SimpleNamespace(num_inliers=17),
    )
    captured_edges = []

    class FakeOptions:
        range_prior_min_count = 3

        def to_dict(self):
            return {"range_prior_min_count": self.range_prior_min_count}

    class FakeMapper:
        options = FakeOptions()

        def reconstruct(self, *, edges, **_kwargs):
            captured_edges.extend(edges)
            return SimpleNamespace(
                interface="test-global-map/v1",
                options=self.options,
                success=False,
                failure_reasons=("test-stop",),
                scale="unresolved",
                reference_panorama_id=None,
                poses=(),
                points_xyz=np.empty((0, 3)),
                tracks=(),
                admitted_edge_indices=tuple(range(len(edges))),
                diagnostics=SimpleNamespace(to_dict=lambda: {}),
            )

    monkeypatch.setattr(global_map_module, "SphericalGlobalMapper", FakeMapper)
    monkeypatch.setattr(
        registration,
        "mapper_edge_snapshot",
        lambda project_id, pair_id: (snapshot_edge, snapshot),
    )
    monkeypatch.setattr(
        registration,
        "mapper_edge",
        lambda project_id, pair_id: "edge-from-revision-1",
    )
    monkeypatch.setattr(registration, "load", lambda project_id, pair_id: later)

    record = service.run(project["id"])

    assert captured_edges == [snapshot_edge]
    assert record["input_edge_revisions"] == {"pair-1": 1}
    assert record["edges"][0]["num_inliers"] == 17


def test_global_map_ignores_disconnected_revisions_and_range_priors(
    tmp_path, monkeypatch
) -> None:
    api = client(tmp_path)
    response = api.post(
        "/api/v1/projects",
        data={
            "name": "Disconnected components",
            "workflow": "spherical_unregistered",
            "metadata": "{}",
            "pairs": json.dumps(
                [
                    {
                        "id": "pair-ab",
                        "reference_filename": "a.png",
                        "source_filename": "b.png",
                    },
                    {
                        "id": "pair-bc",
                        "reference_filename": "b.png",
                        "source_filename": "c.png",
                    },
                    {
                        "id": "pair-ac",
                        "reference_filename": "a.png",
                        "source_filename": "c.png",
                    },
                    {
                        "id": "pair-de",
                        "reference_filename": "d.png",
                        "source_filename": "e.png",
                        "baseline_m": 2.5,
                    },
                    {
                        "id": "pair-cd",
                        "reference_filename": "c.png",
                        "source_filename": "d.png",
                    },
                ]
            ),
        },
        files=[
            ("images", (f"{name}.png", textured_png(index * 3), "image/png"))
            for index, name in enumerate("abcde")
        ],
    )
    assert response.status_code == 200, response.text
    project = response.json()
    stored = api.app.state.store.load(project["id"])
    pairs = {item.id: item for item in stored.pairs}
    revisions = {pair_id: 1 for pair_id in pairs}
    available_pairs = {"pair-ab", "pair-bc", "pair-ac", "pair-de"}
    captured = {}

    def registration_record(pair_id):
        if pair_id not in available_pairs:
            raise KeyError(pair_id)
        return {
            "revision": revisions[pair_id],
            "pending_match_reviews": [],
            "pose": {
                "accepted": True,
                "num_inliers": 24 if pair_id != "pair-de" else 12,
                "quality": {"raw_quality_score": 0.9},
            },
        }

    def edge_snapshot(_project_id, pair_id):
        pair = pairs[pair_id]
        record = registration_record(pair_id)
        edge = SimpleNamespace(
            pair=(pair.reference_image_id, pair.source_image_id),
            pose=SimpleNamespace(num_inliers=record["pose"]["num_inliers"]),
        )
        return edge, record

    disconnected = pairs["pair-de"]
    filtered = pairs["pair-ac"]
    main_component = {
        pairs["pair-ab"].reference_image_id,
        pairs["pair-ab"].source_image_id,
        pairs["pair-bc"].source_image_id,
    }
    range_records = [
        {
            "annotation_id": f"range-{index}",
            "panorama_id": disconnected.reference_image_id,
            "feature_index": 100 + index,
            "range_m": 2.0 + index,
            "binding_pair_id": "pair-de",
        }
        for index in range(3)
    ]
    range_records.extend(
        {
            "annotation_id": f"filtered-range-{index}",
            "panorama_id": filtered.reference_image_id,
            "feature_index": 200 + index,
            "range_m": 4.0 + index,
            "binding_pair_id": "pair-ac",
        }
        for index in range(3)
    )

    class FakeOptions:
        range_prior_min_count = 3

        def to_dict(self):
            return {"range_prior_min_count": self.range_prior_min_count}

    class FakeMapper:
        options = FakeOptions()

        def reconstruct(self, *, edges, panorama_ids, range_priors, **_kwargs):
            captured["pairs"] = [edge.pair for edge in edges]
            captured["panorama_ids"] = set(panorama_ids)
            captured["range_priors"] = range_priors
            image_ids = sorted(main_component)
            return SimpleNamespace(
                interface="test-global-map/v1",
                options=self.options,
                success=True,
                failure_reasons=(),
                scale="arbitrary",
                reference_panorama_id=image_ids[0],
                poses=tuple(
                    SimpleNamespace(
                        panorama_id=image_id,
                        center=np.asarray([index, 0.0, 0.0]),
                        R=np.eye(3),
                    )
                    for index, image_id in enumerate(image_ids)
                ),
                points_xyz=np.empty((0, 3)),
                tracks=(SimpleNamespace(),),
                admitted_edge_indices=(0, 1),
                diagnostics=SimpleNamespace(to_dict=lambda: {}),
            )

    registration = api.app.state.registration
    monkeypatch.setattr(global_map_module, "SphericalGlobalMapper", FakeMapper)
    monkeypatch.setattr(registration, "mapper_edge_snapshot", edge_snapshot)
    monkeypatch.setattr(
        registration,
        "load",
        lambda _project_id, pair_id: registration_record(pair_id),
    )
    monkeypatch.setattr(
        registration,
        "validated_range_prior_records",
        lambda _project: range_records,
    )

    record = api.app.state.global_map.run(project["id"])

    assert captured["panorama_ids"] == {item["id"] for item in project["images"]}
    assert len(captured["pairs"]) == 4
    assert captured["range_priors"] == ()
    assert record["component_image_ids"] == sorted(main_component)
    assert record["input_edge_revisions"] == {"pair-ab": 1, "pair-bc": 1}
    assert record["component_edge_revisions"] == {
        "pair-ab": 1,
        "pair-bc": 1,
        "pair-ac": 1,
    }
    assert record["range_prior_candidates"] == []
    skipped = {item["pair_id"]: item["reason"] for item in record["skipped_pairs"]}
    assert skipped == {
        "pair-cd": "'pair-cd'",
        "pair-de": "outside-selected-map-component",
    }
    setup = scene_setup(stored, registration, record)
    assert setup["summary"]["bound_range_prior_count"] == 0
    assert setup["summary"]["metric_scale_available"] is False
    assert (
        "0/3 âncoras radiais"
        in {item["id"]: item for item in setup["steps"]}["scale"]["detail"]
    )

    revisions["pair-de"] = 2
    assert api.app.state.global_map.load(project["id"])["stale"] is False
    eligible_start = len(range_records)
    admitted = pairs["pair-ab"]
    for index in range(2):
        range_records.append(
            {
                "annotation_id": f"admitted-range-{index}",
                "panorama_id": admitted.reference_image_id,
                "feature_index": 300 + index,
                "range_m": 6.0 + index,
                "binding_pair_id": "pair-ab",
            }
        )
        assert api.app.state.global_map.load(project["id"])["stale"] is False
    range_records.append(
        {
            "annotation_id": "admitted-range-2",
            "panorama_id": admitted.reference_image_id,
            "feature_index": 302,
            "range_m": 8.0,
            "binding_pair_id": "pair-ab",
        }
    )
    assert api.app.state.global_map.load(project["id"])["stale"] is True
    del range_records[eligible_start:]
    assert api.app.state.global_map.load(project["id"])["stale"] is False
    available_pairs.add("pair-cd")
    assert api.app.state.global_map.load(project["id"])["stale"] is True


def test_global_map_runs_are_serialized_per_project(tmp_path, monkeypatch) -> None:
    api = client(tmp_path)
    project = create_pair_project(api)
    service = api.app.state.global_map
    first_writing = Event()
    second_writing = Event()
    release_first = Event()
    write_lock = Lock()
    write_count = 0
    original_write = service._write

    class FakeOptions:
        range_prior_min_count = 3

        def to_dict(self):
            return {"range_prior_min_count": self.range_prior_min_count}

    class FakeMapper:
        options = FakeOptions()

        def reconstruct(self, *, edges, **_kwargs):
            return SimpleNamespace(
                interface="test-global-map/v1",
                options=self.options,
                success=False,
                failure_reasons=("test-stop",),
                scale="unresolved",
                reference_panorama_id=None,
                poses=(),
                points_xyz=np.empty((0, 3)),
                tracks=(),
                admitted_edge_indices=tuple(range(len(edges))),
                diagnostics=SimpleNamespace(to_dict=lambda: {}),
            )

    def delayed_write(record):
        nonlocal write_count
        with write_lock:
            write_count += 1
            sequence = write_count
        if sequence == 1:
            first_writing.set()
            assert release_first.wait(timeout=2)
        elif sequence == 2:
            second_writing.set()
        original_write(record)

    monkeypatch.setattr(global_map_module, "SphericalGlobalMapper", FakeMapper)
    monkeypatch.setattr(service, "_write", delayed_write)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(service.run, project["id"])
        assert first_writing.wait(timeout=2)
        second = executor.submit(service.run, project["id"])
        try:
            assert not second_writing.wait(timeout=0.1)
        finally:
            release_first.set()
        results = [first.result(timeout=3), second.result(timeout=3)]

    assert sorted(item["revision"] for item in results) == [1, 2]


def test_dense_preview_binds_pose_and_provenance_to_one_registration_snapshot(
    tmp_path, monkeypatch
) -> None:
    api = client(tmp_path)
    project = create_pair_project(api)
    service = api.app.state.dense_preview
    registration = api.app.state.registration
    edge = SimpleNamespace(
        pose=SimpleNamespace(R=np.eye(3), t=np.asarray([1.0, 0.0, 0.0]))
    )
    snapshot = {
        "revision": 7,
        "updated_at": "2026-10-20T16:20:00+00:00",
    }

    class FakeDenseApplication:
        route = "sphere-native"

        def describe(self):
            return {"application": "fake-dense-preview/v1"}

        def run(self, reference, source, *, pose, reference_validity):
            shape = reference.shape[:2]
            return SimpleNamespace(
                validity=np.ones(shape, dtype=bool),
                confidence=np.ones(shape, dtype=np.float64),
                radial_range=np.full(shape, 2.0, dtype=np.float64),
                pose=pose,
                diagnostics={"pipeline_stages": ("test-snapshot",)},
            )

    monkeypatch.setattr(
        registration,
        "mapper_edge_snapshot",
        lambda project_id, pair_id: (edge, snapshot),
    )
    monkeypatch.setattr(
        registration,
        "load",
        lambda project_id, pair_id: snapshot,
    )
    monkeypatch.setattr(
        registration,
        "mapper_edge",
        lambda project_id, pair_id: pytest.fail(
            "dense preview must not split edge and provenance reads"
        ),
    )
    monkeypatch.setattr(service, "application", FakeDenseApplication())
    original_write = service._write
    first_writing = Event()
    second_writing = Event()
    release_first = Event()
    write_lock = Lock()
    write_count = 0
    replace_targets = []
    original_replace = Path.replace

    def observed_replace(source, target):
        replace_targets.append(Path(target).name)
        return original_replace(source, target)

    def delayed_write(record, visualization):
        nonlocal write_count
        with write_lock:
            write_count += 1
            sequence = write_count
        if sequence == 1:
            first_writing.set()
            assert release_first.wait(timeout=2)
        elif sequence == 2:
            second_writing.set()
        original_write(record, visualization)

    monkeypatch.setattr(service, "_write", delayed_write)
    monkeypatch.setattr(Path, "replace", observed_replace)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(service.run, project["id"], "pair-1")
        assert first_writing.wait(timeout=2)
        second = executor.submit(service.run, project["id"], "pair-1")
        try:
            assert not second_writing.wait(timeout=0.1)
        finally:
            release_first.set()
        records = [first.result(timeout=3), second.result(timeout=3)]

    assert sorted(item["revision"] for item in records) == [1, 2]
    assert all(item["registration_revision"] == 7 for item in records)
    assert all(
        item["registration_updated_at"] == snapshot["updated_at"] for item in records
    )
    assert replace_targets == [
        "pair-1.revision-0001.png",
        "pair-1.json",
        "pair-1.revision-0002.png",
        "pair-1.json",
    ]
    assert service.image_path(project["id"], "pair-1").name == (
        "pair-1.revision-0002.png"
    )
    assert (
        api.get(
            f"/api/v1/projects/{project['id']}/pairs/pair-1/"
            "dense-preview/image?revision=1"
        ).status_code
        == 409
    )
    assert (
        api.get(
            f"/api/v1/projects/{project['id']}/pairs/pair-1/"
            "dense-preview/image?revision=2"
        ).status_code
        == 200
    )


def test_registered_scene_manifest_preserves_supplied_pose(tmp_path) -> None:
    api = client(tmp_path)
    metadata = {
        filename: {
            "pose": {
                "rotation_world_to_panorama": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "center_world": [float(index), 2.0, 3.0],
            }
        }
        for index, filename in enumerate(("pano-a.png", "pano-b.png"), 1)
    }
    response = api.post(
        "/api/v1/projects",
        data={
            "name": "Registered pair",
            "workflow": "spherical_registered",
            "metadata": json.dumps(metadata),
            "pairs": json.dumps(
                [
                    {
                        "id": "pair-1",
                        "reference_filename": "pano-a.png",
                        "source_filename": "pano-b.png",
                        "baseline_m": 1.0,
                        "minimum_directional_overlap": 0.75,
                    }
                ]
            ),
        },
        files=[
            ("images", ("pano-a.png", png(), "image/png")),
            ("images", ("pano-b.png", png(width=18), "image/png")),
        ],
    )
    assert response.status_code == 200
    project = response.json()
    plan = api.get(f"/api/v1/projects/{project['id']}/build-plan").json()
    assert plan["can_build_now"] is True

    scene = api.post(f"/api/v1/projects/{project['id']}/scene-manifest")
    assert scene.status_code == 200
    assert scene.json()["nodes"][0]["center_world"] == [1.0, 2.0, 3.0]
    assert scene.json()["edges"][0]["baseline_m"] == 1.0
    image = api.get(scene.json()["nodes"][0]["image_url"])
    assert image.status_code == 200


def test_invalid_upload_and_wrong_scene_route_fail_closed(tmp_path) -> None:
    api = client(tmp_path)
    bad = api.post(
        "/api/v1/projects",
        data={"name": "bad", "workflow": "spherical_unregistered"},
        files=[("images", ("bad.png", b"not-an-image", "image/png"))],
    )
    assert bad.status_code == 422

    project = create_project(api).json()
    scene = api.post(f"/api/v1/projects/{project['id']}/scene-manifest")
    assert scene.status_code == 409


def test_ingest_derives_safe_object_type_from_decoded_image(tmp_path) -> None:
    api = client(tmp_path)
    payload = png() + b"<script>alert('not executable')</script>"
    response = api.post(
        "/api/v1/projects",
        data={
            "name": "Decoded type",
            "workflow": "spherical_unregistered",
            "metadata": "{}",
            "pairs": "[]",
        },
        files=[("images", ("attack.html", payload, "text/html"))],
    )

    assert response.status_code == 200, response.text
    image = response.json()["images"][0]
    assert image["object_name"].endswith(".png")
    assert image["media_type"] == "image/png"
    served = api.get(f"/api/v1/projects/{response.json()['id']}/images/{image['id']}")
    assert served.status_code == 200
    assert served.headers["content-type"] == "image/png"
    assert served.headers["x-content-type-options"] == "nosniff"


def test_project_uploads_enforce_count_file_and_aggregate_limits(
    tmp_path, monkeypatch
) -> None:
    payload = png()
    api = client(tmp_path)

    monkeypatch.setattr(api_module, "MAX_PROJECT_IMAGES", 1)
    too_many = api.post(
        "/api/v1/projects",
        data={"name": "too many", "workflow": "spherical_unregistered"},
        files=[
            ("images", ("a.png", payload, "image/png")),
            ("images", ("b.png", payload, "image/png")),
        ],
    )
    assert too_many.status_code == 422
    assert "at most 1 images" in too_many.json()["detail"]

    monkeypatch.setattr(api_module, "MAX_PROJECT_IMAGES", 64)
    monkeypatch.setattr(api_module, "MAX_IMAGE_BYTES", len(payload) - 1)
    too_large = api.post(
        "/api/v1/projects",
        data={"name": "too large", "workflow": "spherical_unregistered"},
        files=[("images", ("large.png", payload, "image/png"))],
    )
    assert too_large.status_code == 422
    assert "exceeds the 160 MiB limit" in too_large.json()["detail"]

    monkeypatch.setattr(api_module, "MAX_IMAGE_BYTES", len(payload) + 1)
    monkeypatch.setattr(api_module, "MAX_PROJECT_UPLOAD_BYTES", len(payload) + 1)
    aggregate = api.post(
        "/api/v1/projects",
        data={"name": "aggregate", "workflow": "spherical_unregistered"},
        files=[
            ("images", ("a.png", payload, "image/png")),
            ("images", ("b.png", payload, "image/png")),
        ],
    )
    assert aggregate.status_code == 422
    assert "aggregate limit" in aggregate.json()["detail"]


def test_ingest_rejects_decoded_dimensions_before_loading_pixels(
    tmp_path, monkeypatch
) -> None:
    class OversizedImage:
        format = "PNG"

        def __init__(self, size):
            self.size = size
            self.loaded = False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def load(self):
            self.loaded = True

    store = store_module.ProjectStore(tmp_path / "data")
    for size in (
        (store_module.MAX_DECODED_IMAGE_WIDTH + 1, 1),
        (8_193, 8_193),
    ):
        decoded = OversizedImage(size)
        monkeypatch.setattr(store_module.Image, "open", lambda _stream: decoded)
        with pytest.raises(ValueError, match="exceeds decoded image limits"):
            store._ingest(0, ("large.png", "image/png", b"payload"), {})
        assert decoded.loaded is False
    assert list((tmp_path / "data" / "objects").iterdir()) == []


def test_project_intake_rejects_duplicate_unordered_pairs(tmp_path) -> None:
    api = client(tmp_path)
    response = api.post(
        "/api/v1/projects",
        data={
            "name": "Duplicate pair",
            "workflow": "spherical_unregistered",
            "metadata": "{}",
            "pairs": json.dumps(
                [
                    {
                        "id": "forward",
                        "reference_filename": "a.png",
                        "source_filename": "b.png",
                    },
                    {
                        "id": "reverse",
                        "reference_filename": "b.png",
                        "source_filename": "a.png",
                    },
                ]
            ),
        },
        files=[
            ("images", ("a.png", png(), "image/png")),
            ("images", ("b.png", png(width=18), "image/png")),
        ],
    )

    assert response.status_code == 422
    assert "duplicate unordered image pair" in response.json()["detail"]
    assert list((tmp_path / "data" / "objects").iterdir()) == []

    duplicate_ids = api.post(
        "/api/v1/projects",
        data={
            "name": "Duplicate IDs",
            "workflow": "spherical_unregistered",
            "metadata": "{}",
            "pairs": json.dumps(
                [
                    {
                        "id": "same",
                        "reference_filename": "a.png",
                        "source_filename": "b.png",
                    },
                    {
                        "id": "same",
                        "reference_filename": "a.png",
                        "source_filename": "c.png",
                    },
                ]
            ),
        },
        files=[
            ("images", ("a.png", png(), "image/png")),
            ("images", ("b.png", png(width=18), "image/png")),
            ("images", ("c.png", png(width=20), "image/png")),
        ],
    )
    assert duplicate_ids.status_code == 422
    assert "pair IDs must be unique" in duplicate_ids.json()["detail"]
    assert list((tmp_path / "data" / "objects").iterdir()) == []


def test_metric_inputs_reject_non_finite_positive_values(tmp_path) -> None:
    api = client(tmp_path)
    baseline = api.post(
        "/api/v1/projects",
        data={
            "name": "Infinite baseline",
            "workflow": "spherical_unregistered",
            "metadata": "{}",
            "pairs": (
                '[{"id":"pair-1","reference_filename":"a.png",'
                '"source_filename":"b.png","baseline_m":1e400}]'
            ),
        },
        files=[
            ("images", ("a.png", png(), "image/png")),
            ("images", ("b.png", png(width=18), "image/png")),
        ],
    )
    assert baseline.status_code == 422
    assert "finite_number" in baseline.text
    assert list((tmp_path / "data" / "objects").iterdir()) == []

    project = create_project(api).json()
    image_id = project["images"][0]["id"]
    anchor = api.post(
        f"/api/v1/projects/{project['id']}/dense-annotations",
        content=(
            '{"kind":"range_anchor","image_id":"'
            + image_id
            + '","label":"infinite","points_xy":[[1,1]],'
            '"radial_range":1e400,"state":"accepted"}'
        ),
        headers={"content-type": "application/json"},
    )
    assert anchor.status_code == 422
    assert "finite_number" in anchor.text
    reloaded = api.get(f"/api/v1/projects/{project['id']}")
    assert reloaded.status_code == 200
    assert reloaded.json()["dense_annotations"] == []


def test_range_anchor_rejects_an_image_outside_the_project(tmp_path) -> None:
    api = client(tmp_path)
    project = create_pair_project(api)

    response = api.post(
        f"/api/v1/projects/{project['id']}/dense-annotations",
        json={
            "kind": "range_anchor",
            "image_id": "unknown-image",
            "label": "invalid anchor",
            "points_xy": [[1.0, 1.0]],
            "radial_range": 2.0,
            "state": "accepted",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "dense annotation image_id does not belong to project"
    )
    reloaded = api.get(f"/api/v1/projects/{project['id']}")
    assert reloaded.status_code == 200
    assert reloaded.json()["dense_annotations"] == []


def test_intake_free_form_json_rejects_non_finite_values_without_persistence(
    tmp_path,
) -> None:
    api = client(tmp_path)
    metadata = api.post(
        "/api/v1/projects",
        data={
            "name": "Infinite metadata",
            "workflow": "perspective_uncalibrated",
            "metadata": '{"a.png":{"horizontal_fov_deg":1e400}}',
            "pairs": "[]",
        },
        files=[("images", ("a.png", png(), "image/png"))],
    )
    assert metadata.status_code == 422
    assert (
        "metadata.horizontal_fov_deg must contain only finite numbers"
        in (metadata.json()["detail"])
    )
    assert list((tmp_path / "data" / "objects").iterdir()) == []

    provenance = api.post(
        "/api/v1/projects",
        data={
            "name": "Infinite provenance",
            "workflow": "spherical_unregistered",
            "metadata": "{}",
            "pairs": (
                '[{"id":"pair-1","reference_filename":"a.png",'
                '"source_filename":"b.png","provenance":{"score":1e400}}]'
            ),
        },
        files=[
            ("images", ("a.png", png(), "image/png")),
            ("images", ("b.png", png(width=18), "image/png")),
        ],
    )
    assert provenance.status_code == 422
    assert (
        "provenance.score must contain only finite numbers"
        in provenance.json()["detail"]
    )
    assert list((tmp_path / "data" / "objects").iterdir()) == []
    assert api.get("/api/v1/projects").json()["projects"] == []


def test_unregistered_project_rejects_supplied_pose_metadata(tmp_path) -> None:
    api = client(tmp_path)
    response = create_project(
        api,
        workflow="spherical_unregistered",
        metadata={
            "pano.png": {
                "pose": {
                    "rotation_world_to_panorama": [
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                    ],
                    "center_world": [0, 0, 0],
                }
            }
        },
    )

    assert response.status_code == 422
    assert "must not contain supplied poses" in response.json()["detail"]


def test_registered_scene_rejects_a_reflection_pose(tmp_path) -> None:
    api = client(tmp_path)
    metadata = {
        "pano.png": {
            "pose": {
                "rotation_world_to_panorama": [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "center_world": [0, 0, 0],
            }
        }
    }
    project = create_project(
        api, workflow="spherical_registered", metadata=metadata
    ).json()

    plan = api.get(f"/api/v1/projects/{project['id']}/build-plan").json()
    assert plan["can_build_now"] is False
    scene = api.post(f"/api/v1/projects/{project['id']}/scene-manifest")
    assert scene.status_code == 409
    assert "invalid rigid pose" in scene.json()["detail"]


def test_registered_scene_rejects_numeric_pose_overflow_without_500(tmp_path) -> None:
    api = client(tmp_path)
    metadata = {
        "pano.png": {
            "pose": {
                "rotation_world_to_panorama": [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ],
                "center_world": [int("9" * 400), 0, 0],
            }
        }
    }
    response = create_project(api, workflow="spherical_registered", metadata=metadata)
    assert response.status_code == 200, response.text
    project = response.json()

    plan = api.get(f"/api/v1/projects/{project['id']}/build-plan")
    assert plan.status_code == 200
    assert plan.json()["can_build_now"] is False
    scene = api.post(f"/api/v1/projects/{project['id']}/scene-manifest")
    assert scene.status_code == 409
    assert "invalid rigid pose" in scene.json()["detail"]


def test_registered_project_requires_an_explicit_pair_graph(tmp_path) -> None:
    api = client(tmp_path)
    metadata = {
        "pano.png": {
            "pose": {
                "rotation_world_to_panorama": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "center_world": [0, 0, 0],
            }
        }
    }
    project = create_project(
        api, workflow="spherical_registered", metadata=metadata
    ).json()

    plan = api.get(f"/api/v1/projects/{project['id']}/build-plan").json()
    assert plan["can_build_now"] is False
    assert any(stage["id"] == "registered-pairs" for stage in plan["stages"])
    scene = api.post(f"/api/v1/projects/{project['id']}/scene-manifest")
    assert scene.status_code == 409
    assert "explicit pair graph" in scene.json()["detail"]


def test_human_pair_and_dense_evidence_are_distinct_and_persisted(tmp_path) -> None:
    api = client(tmp_path)
    project = create_pair_project(api)
    reference_id, source_id = [item["id"] for item in project["images"]]

    for image_id, bbox in (
        (reference_id, [20, 20, 100, 90]),
        (source_id, [28, 20, 108, 90]),
    ):
        response = api.post(
            f"/api/v1/projects/{project['id']}/object-regions",
            json={
                "object_id": "object-a",
                "label": "Object A",
                "image_id": image_id,
                "bbox_xyxy": bbox,
                "state": "accepted",
            },
        )
        assert response.status_code == 200

    manual = api.post(
        f"/api/v1/projects/{project['id']}/manual-correspondences",
        json={
            "pair_id": "pair-1",
            "reference_xy": [52.5, 44.5],
            "source_xy": [60.5, 44.5],
            "state": "accepted",
            "created_at": "2099-01-01T00:00:00+00:00",
        },
    )
    assert manual.status_code == 200
    dense = api.post(
        f"/api/v1/projects/{project['id']}/dense-annotations",
        json={
            "kind": "range_anchor",
            "image_id": reference_id,
            "label": "wall",
            "points_xy": [[80.5, 64.5]],
            "radial_range": 2.4,
            "state": "accepted",
        },
    )
    assert dense.status_code == 200
    forged = api.post(
        f"/api/v1/projects/{project['id']}/dense-annotations",
        json={
            "kind": "range_anchor",
            "image_id": reference_id,
            "label": "untrusted client binding",
            "points_xy": [[160.5, 96.5]],
            "radial_range": 7.5,
            "feature_index": 42,
            "binding_pair_id": "pair-1",
            "binding_distance_px": 0.0,
            "engine_effect": "range_prior",
            "state": "accepted",
        },
    )
    assert forged.status_code == 200

    stored = api.get(f"/api/v1/projects/{project['id']}").json()
    assert len(stored["object_regions"]) == 2
    assert stored["manual_correspondences"][0]["pair_id"] == "pair-1"
    assert stored["manual_correspondences"][0]["created_at"] != (
        "2099-01-01T00:00:00+00:00"
    )
    assert stored["dense_annotations"][0]["radial_range"] == 2.4
    assert stored["dense_annotations"][0]["feature_index"] is None
    assert stored["dense_annotations"][0]["engine_effect"] == "staged"
    untrusted = stored["dense_annotations"][1]
    assert untrusted["feature_index"] is None
    assert untrusted["binding_pair_id"] is None
    assert untrusted["binding_distance_px"] is None
    assert untrusted["engine_effect"] == "staged"
    assert "pose" not in stored["images"][0]["metadata"]


def test_project_manifest_updates_are_serialized_per_project(tmp_path) -> None:
    api = client(tmp_path)
    project = create_pair_project(api)
    store = api.app.state.store
    image_id = project["images"][0]["id"]
    original_load = store.load
    first_loaded = Event()
    release_first = Event()
    second_loaded = Event()
    counter_lock = Lock()
    load_count = 0

    def delayed_load(project_id):
        nonlocal load_count
        loaded = original_load(project_id)
        with counter_lock:
            load_count += 1
            sequence = load_count
        if sequence == 1:
            first_loaded.set()
            assert release_first.wait(timeout=2)
        elif sequence == 2:
            second_loaded.set()
        return loaded

    store.load = delayed_load
    annotations = [
        DenseAnnotation(
            kind="range_anchor",
            image_id=image_id,
            label=f"concurrent-{index}",
            points_xy=[(40.5 + index, 30.5)],
            radial_range=2.0 + index,
        )
        for index in range(2)
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            store.add_dense_annotation, project["id"], annotations[0]
        )
        assert first_loaded.wait(timeout=2)
        second = executor.submit(
            store.add_dense_annotation, project["id"], annotations[1]
        )
        assert not second_loaded.wait(timeout=0.1)
        release_first.set()
        first.result(timeout=2)
        second.result(timeout=2)

    stored = original_load(project["id"])
    assert [item.label for item in stored.dense_annotations] == [
        "concurrent-0",
        "concurrent-1",
    ]


def test_pair_match_reviews_are_serialized_per_registration(tmp_path) -> None:
    api = client(tmp_path)
    project = create_pair_project(api)
    registration = api.app.state.registration
    registration._write(
        {
            "project_id": project["id"],
            "pair_id": "pair-1",
            "revision": 1,
            "updated_at": "2026-10-20T00:00:00+00:00",
            "matches": [
                {"review_state": "unreviewed"},
                {"review_state": "unreviewed"},
            ],
            "pending_match_reviews": [],
        },
        snapshot=False,
    )
    original_load = registration.load
    first_loaded = Event()
    release_first = Event()
    second_loaded = Event()
    counter_lock = Lock()
    load_count = 0

    def delayed_load(project_id, pair_id):
        nonlocal load_count
        loaded = original_load(project_id, pair_id)
        with counter_lock:
            load_count += 1
            sequence = load_count
        if sequence == 1:
            first_loaded.set()
            assert release_first.wait(timeout=2)
        elif sequence == 2:
            second_loaded.set()
        return loaded

    registration.load = delayed_load
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            registration.review_match,
            project["id"],
            "pair-1",
            0,
            MatchReview(state="accepted", revision=1),
        )
        assert first_loaded.wait(timeout=2)
        second = executor.submit(
            registration.review_match,
            project["id"],
            "pair-1",
            1,
            MatchReview(state="rejected", revision=1),
        )
        assert not second_loaded.wait(timeout=0.1)
        release_first.set()
        first.result(timeout=2)
        second.result(timeout=2)

    stored = original_load(project["id"], "pair-1")
    assert [item["review_state"] for item in stored["matches"]] == [
        "accepted",
        "rejected",
    ]
    assert stored["pending_match_reviews"] == [0, 1]
    with pytest.raises(
        ValueError, match="registration revision changed; reload matches"
    ):
        registration.review_match(
            project["id"],
            "pair-1",
            0,
            MatchReview(state="rejected", revision=2),
        )


def test_pair_match_review_waits_for_project_wide_operations(
    tmp_path, monkeypatch
) -> None:
    api = client(tmp_path)
    project = create_pair_project(api)
    registration = api.app.state.registration
    registration._write(
        {
            "project_id": project["id"],
            "pair_id": "pair-1",
            "revision": 1,
            "updated_at": "2026-10-20T00:00:00+00:00",
            "matches": [{"review_state": "unreviewed"}],
            "pending_match_reviews": [],
        },
        snapshot=False,
    )
    entered_review = Event()
    original_review = registration._review_match

    def observed_review(project_id, pair_id, match_index, review):
        entered_review.set()
        return original_review(project_id, pair_id, match_index, review)

    monkeypatch.setattr(registration, "_review_match", observed_review)
    project_lock = api.app.state.store.project_lock(project["id"])
    with ThreadPoolExecutor(max_workers=1) as executor:
        project_lock.acquire()
        try:
            future = executor.submit(
                registration.review_match,
                project["id"],
                "pair-1",
                0,
                MatchReview(state="accepted", revision=1),
            )
            assert not entered_review.wait(timeout=0.1)
        finally:
            project_lock.release()
        result = future.result(timeout=2)

    assert entered_review.is_set()
    assert result["matches"][0]["review_state"] == "accepted"


def test_explicit_match_review_overrides_matcher_validity(tmp_path) -> None:
    api = client(tmp_path)
    project = create_pair_project(api)
    registration = api.app.state.registration
    stored_project = api.app.state.store.load(project["id"])
    captured = {}

    class CapturingPoseApplication:
        @staticmethod
        def describe():
            return {"application": "capturing-test-double"}

        @staticmethod
        def estimate_correspondences(correspondences):
            captured["valid"] = correspondences.valid.copy()
            captured["weights"] = correspondences.weights.copy()
            return None

    registration.pose_application = CapturingPoseApplication()
    record = {
        "detections": {"reference": [], "source": []},
        "matches": [
            {
                "index": 0,
                "feature_index_reference": 10,
                "feature_index_source": 20,
                "reference_xy": [20.0, 30.0],
                "source_xy": [22.0, 30.0],
                "bearing_reference": [0.0, 0.0, 1.0],
                "bearing_source": [0.01, 0.0, 0.99995],
                "auto_valid": False,
                "review_state": "accepted",
            },
            {
                "index": 1,
                "feature_index_reference": 11,
                "feature_index_source": 21,
                "reference_xy": [80.0, 50.0],
                "source_xy": [82.0, 50.0],
                "bearing_reference": [0.0, 0.1, 0.994987],
                "bearing_source": [0.01, 0.1, 0.994937],
                "auto_valid": True,
                "review_state": "rejected",
            },
        ],
    }

    registration._estimate(record, stored_project, stored_project.pairs[0], "all")

    assert captured["valid"].tolist() == [True, False]
    assert captured["weights"].tolist() == [3.0, 1.0]
    assert record["active_correspondence_count"] == 1


def test_registration_holds_project_evidence_stable_through_estimation(
    tmp_path, monkeypatch
) -> None:
    api = client(tmp_path)
    project = create_pair_project(api)
    registration = api.app.state.registration
    store = api.app.state.store
    estimating = Event()
    release_estimation = Event()
    evidence_mutation_entered = Event()
    original_add = store._add_manual_correspondence

    def delayed_estimate(record, project_record, pair, region_mode):
        estimating.set()
        assert release_estimation.wait(timeout=2)

    def observed_add(project_id, correspondence):
        evidence_mutation_entered.set()
        return original_add(project_id, correspondence)

    monkeypatch.setattr(registration, "_estimate", delayed_estimate)
    monkeypatch.setattr(store, "_add_manual_correspondence", observed_add)
    correspondence = ManualCorrespondence(
        pair_id="pair-1",
        reference_xy=(32.0, 32.0),
        source_xy=(40.0, 32.0),
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        registration_run = executor.submit(
            registration.run,
            project["id"],
            "pair-1",
            SimpleNamespace(region_mode="all"),
        )
        assert estimating.wait(timeout=10)
        evidence_edit = executor.submit(
            store.add_manual_correspondence,
            project["id"],
            correspondence,
        )
        try:
            assert not evidence_mutation_entered.wait(timeout=0.1)
        finally:
            release_estimation.set()
        registration_run.result(timeout=5)
        evidence_edit.result(timeout=5)

    stored = store.load(project["id"])
    assert len(stored.manual_correspondences) == 1
    assert stored.manual_correspondences[0].pair_id == "pair-1"


def test_manual_correspondence_ids_are_always_server_allocated(tmp_path) -> None:
    api = client(tmp_path)
    project = create_pair_project(api)
    store = api.app.state.store

    for offset in (0.0, 8.0):
        store.add_manual_correspondence(
            project["id"],
            ManualCorrespondence(
                id="client-controlled-duplicate",
                pair_id="pair-1",
                reference_xy=(32.0 + offset, 32.0),
                source_xy=(40.0 + offset, 32.0),
            ),
        )

    stored = store.load(project["id"])
    identifiers = [item.id for item in stored.manual_correspondences]
    assert len(set(identifiers)) == 2
    assert all(identifier.startswith("manual_match_") for identifier in identifiers)


def test_project_intake_rejects_storage_unsafe_pair_ids(tmp_path) -> None:
    api = client(tmp_path)
    response = api.post(
        "/api/v1/projects",
        data={
            "name": "Unsafe pair id",
            "workflow": "spherical_unregistered",
            "metadata": "{}",
            "pairs": json.dumps(
                [
                    {
                        "id": "../pair one",
                        "reference_filename": "a.png",
                        "source_filename": "b.png",
                    }
                ]
            ),
        },
        files=[
            ("images", ("a.png", textured_png(), "image/png")),
            ("images", ("b.png", textured_png(8), "image/png")),
        ],
    )

    assert response.status_code == 422
    assert "string_pattern_mismatch" in response.json()["detail"]


def test_object_region_staleness_requires_matching_pair_regions(tmp_path) -> None:
    api = client(tmp_path)
    response = api.post(
        "/api/v1/projects",
        data={
            "name": "Three image object graph",
            "workflow": "spherical_unregistered",
            "metadata": "{}",
            "pairs": json.dumps(
                [
                    {
                        "id": "pair-ab",
                        "reference_filename": "a.png",
                        "source_filename": "b.png",
                    },
                    {
                        "id": "pair-ac",
                        "reference_filename": "a.png",
                        "source_filename": "c.png",
                    },
                ]
            ),
        },
        files=[
            ("images", ("a.png", textured_png(), "image/png")),
            ("images", ("b.png", textured_png(4), "image/png")),
            ("images", ("c.png", textured_png(8), "image/png")),
        ],
    )
    assert response.status_code == 200, response.text
    project = response.json()
    image_ids = {item["filename"]: item["id"] for item in project["images"]}
    registration = api.app.state.registration
    stored = api.app.state.store.load(project["id"])
    pairs = {item.id: item for item in stored.pairs}
    prior_registrations = {}
    for pair in pairs.values():
        record = {
            "project_id": project["id"],
            "pair_id": pair.id,
            "revision": 1,
            "updated_at": "2026-01-01T00:00:00+00:00",
            "pending_match_reviews": [],
            "evidence_signature": registration.evidence_signature(stored, pair),
            "pose": {
                "accepted": True,
                "num_inliers": 12,
                "quality": {"raw_quality_score": 0.9},
            },
        }
        registration._write(record, snapshot=False)
        prior_registrations[pair.id] = record

    first = api.post(
        f"/api/v1/projects/{project['id']}/object-regions",
        json={
            "object_id": "chair",
            "label": "Chair",
            "image_id": image_ids["a.png"],
            "bbox_xyxy": [10, 10, 80, 80],
            "state": "accepted",
        },
    )
    assert first.status_code == 200, first.text
    stored = api.app.state.store.load(project["id"])
    pairs = {item.id: item for item in stored.pairs}
    assert not registration.evidence_is_stale(
        stored, pairs["pair-ab"], prior_registrations["pair-ab"]
    )
    assert not registration.evidence_is_stale(
        stored, pairs["pair-ac"], prior_registrations["pair-ac"]
    )

    second = api.post(
        f"/api/v1/projects/{project['id']}/object-regions",
        json={
            "object_id": "chair",
            "label": "Chair",
            "image_id": image_ids["b.png"],
            "bbox_xyxy": [12, 12, 82, 82],
            "state": "accepted",
        },
    )
    assert second.status_code == 200, second.text
    stored = api.app.state.store.load(project["id"])
    pairs = {item.id: item for item in stored.pairs}
    assert registration.evidence_is_stale(
        stored, pairs["pair-ab"], prior_registrations["pair-ab"]
    )
    assert not registration.evidence_is_stale(
        stored, pairs["pair-ac"], prior_registrations["pair-ac"]
    )
    setup = scene_setup(stored, registration)
    statuses = {item["pair_id"]: item["status"] for item in setup["pairs"]}
    assert statuses == {"pair-ab": "stale", "pair-ac": "accepted"}

    consumed_record = {
        **prior_registrations["pair-ab"],
        "evidence_signature": registration.evidence_signature(stored, pairs["pair-ab"]),
    }
    withdrawn = api.post(
        f"/api/v1/projects/{project['id']}/object-regions",
        json={
            "object_id": "chair",
            "label": "Chair withdrawn",
            "image_id": image_ids["b.png"],
            "bbox_xyxy": [12, 12, 82, 82],
            "state": "rejected",
        },
    )
    assert withdrawn.status_code == 200, withdrawn.text
    stored = api.app.state.store.load(project["id"])
    pairs = {item.id: item for item in stored.pairs}
    assert registration.evidence_is_stale(stored, pairs["pair-ab"], consumed_record)


def test_real_public_feature_pipeline_produces_reviewable_pair_record(tmp_path) -> None:
    api = client(tmp_path)
    project = create_pair_project(api)
    reference_id, source_id = [item["id"] for item in project["images"]]
    for image_id in (reference_id, source_id):
        region = api.post(
            f"/api/v1/projects/{project['id']}/object-regions",
            json={
                "object_id": "whole-pair-test",
                "label": "Whole pair test region",
                "image_id": image_id,
                "bbox_xyxy": [0, 0, 256, 128],
                "state": "accepted",
            },
        )
        assert region.status_code == 200
    response = api.post(
        f"/api/v1/projects/{project['id']}/pairs/pair-1/registration",
        json={"region_mode": "prioritize"},
    )
    assert response.status_code == 200, response.text
    record = response.json()
    assert record["schema"] == "panorai-streetview-pair-registration/v1"
    assert record["feature_counts"]["reference"] > 0
    assert record["feature_counts"]["source"] > 0
    assert record["detection_payload"] == {
        "bounded": True,
        "limit_per_image": 900,
    }
    assert record["engine"]["relative_pose"]["application"] == (
        "panorai.applications.SphericalRelativePose"
    )
    assert record["engine"]["relative_pose"]["profile_mode"] == "reference"
    assert record["region_mode"] == "prioritize"
    assert record["object_summaries"][0]["match_count"] == len(record["matches"])
    assert record["recommended_edit"]["tool"]
    assert record["pose"] is not None
    if record["pose"] is not None:
        assert {gate["id"] for gate in record["quality_gates"]} >= {
            "coverage",
            "residual",
            "parallax",
            "rotation-stability",
            "translation-stability",
            "model",
        }
        assert record["pose"]["diagnostics"]["interface"] == (
            "panorai-spherical-relative-pose/v1"
        )
        if record["pose"]["accepted"]:
            edge = api.app.state.registration.mapper_edge(project["id"], "pair-1")
            assert isinstance(edge, SphericalPairwisePoseEdge)
            assert edge.panorama_id_a == reference_id
            assert edge.panorama_id_b == source_id
            assert edge.pose.inlier_mask.shape == edge.matches.valid.shape
            inlier = next(item for item in record["matches"] if item["inlier"])
            anchored = api.post(
                f"/api/v1/projects/{project['id']}/dense-annotations",
                json={
                    "kind": "range_anchor",
                    "image_id": reference_id,
                    "label": "measured corner",
                    "points_xy": [inlier["reference_xy"]],
                    "radial_range": 3.25,
                    "state": "accepted",
                },
            )
            assert anchored.status_code == 200, anchored.text
            annotation = anchored.json()["dense_annotations"][-1]
            assert annotation["engine_effect"] == "range_prior"
            assert annotation["feature_index"] == inlier["feature_index_reference"]
            assert annotation["binding_pair_id"] == "pair-1"
            assert annotation["binding_distance_px"] == 0.0
            duplicate = api.post(
                f"/api/v1/projects/{project['id']}/dense-annotations",
                json={
                    "kind": "range_anchor",
                    "image_id": reference_id,
                    "label": "duplicate measurement",
                    "points_xy": [inlier["reference_xy"]],
                    "radial_range": 3.25,
                    "state": "accepted",
                },
            )
            assert duplicate.status_code == 422
            assert "already exists" in duplicate.json()["detail"]

            excluded = api.post(
                f"/api/v1/projects/{project['id']}/dense-annotations",
                json={
                    "kind": "exclude_region",
                    "image_id": reference_id,
                    "label": "unreliable left quarter",
                    "points_xy": [[0, 0], [64, 127.9]],
                    "state": "accepted",
                },
            )
            assert excluded.status_code == 200, excluded.text
            preview_response = api.post(
                f"/api/v1/projects/{project['id']}/pairs/pair-1/dense-preview"
            )
            assert preview_response.status_code == 200, preview_response.text
            preview = preview_response.json()
            assert preview["schema"] == "panorai-streetview-dense-preview/v1"
            assert preview["scale_status"] == "scale-free"
            assert preview["units"] == "baseline-units"
            assert preview["input_mask"]["semantics"] == ("explicit-reference-validity")
            assert preview["input_mask"]["eligible_fraction"] == 0.75
            assert len(preview["input_mask"]["exclude_annotation_ids"]) == 1
            assert preview["valid_outside_input_count"] == 0
            assert preview["valid_pixel_count"] > 0
            preview_image = api.get(
                f"/api/v1/projects/{project['id']}/pairs/pair-1/dense-preview/image"
            )
            assert preview_image.status_code == 200
            assert preview_image.headers["content-type"] == "image/png"
            rejected_mask = api.post(
                f"/api/v1/projects/{project['id']}/dense-annotations",
                json={
                    "kind": "exclude_region",
                    "image_id": source_id,
                    "label": "rejected source mask",
                    "points_xy": [[0, 0], [32, 32]],
                    "state": "rejected",
                },
            )
            assert rejected_mask.status_code == 200, rejected_mask.text
            reloaded_preview = api.get(
                f"/api/v1/projects/{project['id']}/pairs/pair-1/dense-preview"
            )
            assert reloaded_preview.status_code == 200
            assert reloaded_preview.json()["stale"] is False

    history = api.get(
        f"/api/v1/projects/{project['id']}/pairs/pair-1/registration/history"
    )
    assert history.status_code == 200
    assert [item["revision"] for item in history.json()["revisions"]] == [1]

    setup = api.get(f"/api/v1/projects/{project['id']}/setup")
    assert setup.status_code == 200
    setup_payload = setup.json()
    assert setup_payload["summary"]["registered_pair_count"] == 1
    assert setup_payload["profiles"]["relative_pose"]["application"] == (
        "panorai.applications.SphericalRelativePose"
    )
    assert setup_payload["profiles"]["pair_dense"]["scale"].startswith(
        "metric with supplied baseline"
    )
    if record.get("pose") and record["pose"]["accepted"]:
        evidence = api.post(
            f"/api/v1/projects/{project['id']}/manual-correspondences",
            json={
                "pair_id": "pair-1",
                "reference_xy": [20.0, 30.0],
                "source_xy": [28.0, 30.0],
                "state": "accepted",
            },
        )
        assert evidence.status_code == 200, evidence.text
        stale_preview = api.get(
            f"/api/v1/projects/{project['id']}/pairs/pair-1/dense-preview"
        )
        assert stale_preview.status_code == 200
        assert stale_preview.json()["stale"] is True
    if record["matches"]:
        reviewed = api.patch(
            f"/api/v1/projects/{project['id']}/pairs/pair-1/registration/matches/0",
            json={"state": "rejected", "revision": record["revision"]},
        )
        assert reviewed.status_code == 200
        assert reviewed.json()["matches"][0]["review_state"] == "rejected"
        loaded = api.get(f"/api/v1/projects/{project['id']}/pairs/pair-1/registration")
        assert loaded.status_code == 200
        assert loaded.json()["matches"][0]["review_state"] == "rejected"
        assert loaded.json()["pending_match_reviews"] == [0]
        history_after_review = api.get(
            f"/api/v1/projects/{project['id']}/pairs/pair-1/registration/history"
        ).json()
        assert history_after_review["revisions"][0]["rejected_reviews"] == 0


def test_scene_setup_starts_from_unknown_pose_and_keeps_metric_dense_blocked(
    tmp_path,
) -> None:
    api = client(tmp_path)
    project = create_pair_project(api)

    response = api.get(f"/api/v1/projects/{project['id']}/setup")

    assert response.status_code == 200
    setup = response.json()
    assert setup["summary"] == {
        "image_count": 2,
        "pair_count": 1,
        "registered_pair_count": 0,
        "accepted_pair_count": 0,
        "accepted_component_size": 1,
        "metric_scale_available": False,
        "metric_scale_sources": [],
        "bound_range_prior_count": 0,
        "range_prior_min_count": 3,
        "global_map_eligible": False,
        "global_map_status": "not-run",
        "dense_scene_eligible": False,
        "dense_scene_blockers": [
            "current-global-map-required",
            "metric-scale-required",
            "three-registered-views-required",
        ],
    }
    assert setup["pairs"][0]["status"] == "not-run"
    assert setup["recommended_action"]["kind"] == "not-run"
    steps = {item["id"]: item for item in setup["steps"]}
    assert steps["relative-pose"]["status"] == "action"
    assert steps["global-map"]["status"] == "blocked"
    assert steps["dense-pair"]["status"] == "blocked"
    assert steps["dense-scene"]["status"] == "blocked"
    assert "SphericalRangePrior" in steps["scale"]["detail"]


def test_range_prior_requires_a_current_accepted_inlier_binding(tmp_path) -> None:
    api = client(tmp_path)
    project = create_pair_project(api)
    registration = api.app.state.registration
    match = {
        "feature_index_reference": 17,
        "feature_index_source": 29,
        "reference_xy": [40.0, 32.0],
        "source_xy": [48.0, 32.0],
        "inlier": True,
        "review_state": "accepted",
    }
    record = {
        "project_id": project["id"],
        "pair_id": "pair-1",
        "revision": 1,
        "updated_at": "2026-10-20T12:00:00+00:00",
        "region_mode": "all",
        "pending_match_reviews": [],
        "matches": [match],
        "pose": {
            "accepted": True,
            "num_inliers": 1,
            "quality": {"raw_quality_score": 1.0},
        },
    }
    registration._write(record)
    anchor = api.post(
        f"/api/v1/projects/{project['id']}/dense-annotations",
        json={
            "kind": "range_anchor",
            "image_id": project["images"][0]["id"],
            "label": "measured feature",
            "points_xy": [match["reference_xy"]],
            "radial_range": 4.0,
        },
    )
    assert anchor.status_code == 200, anchor.text
    stored = api.app.state.store.load(project["id"])
    assert len(registration.validated_range_prior_records(stored)) == 1
    assert scene_setup(stored, registration)["summary"]["bound_range_prior_count"] == 1

    record["revision"] = 2
    record["matches"][0]["inlier"] = False
    registration._write(record)
    stored = api.app.state.store.load(project["id"])
    assert registration.validated_range_prior_records(stored) == []
    assert scene_setup(stored, registration)["summary"]["bound_range_prior_count"] == 0


def test_scene_setup_routes_arbitrary_map_to_scale_and_accepts_metric_range_map(
    tmp_path,
) -> None:
    api = client(tmp_path)
    project = create_pair_project(api)
    stored = api.app.state.store.load(project["id"])
    current_map = {
        "success": True,
        "stale": False,
        "revision": 1,
        "track_count": 12,
        "poses": [{"image_id": f"camera-{index}"} for index in range(3)],
        "scale": "arbitrary",
        "range_priors_applied": 0,
        "recommended_edit": {"detail": "map accepted"},
    }

    arbitrary = scene_setup(
        stored,
        api.app.state.registration,
        current_map,
    )

    assert arbitrary["summary"]["metric_scale_available"] is False
    assert arbitrary["summary"]["dense_scene_eligible"] is False
    assert arbitrary["summary"]["dense_scene_blockers"] == ["metric-scale-required"]
    assert arbitrary["recommended_action"]["kind"] == "scale"
    assert "3 medida(s)" in arbitrary["recommended_action"]["detail"]

    metric = scene_setup(
        stored,
        api.app.state.registration,
        {
            **current_map,
            "scale": "metric-range-prior",
            "range_priors_applied": 3,
        },
    )

    assert metric["summary"]["metric_scale_available"] is True
    assert metric["summary"]["metric_scale_sources"] == ["range-prior"]
    assert metric["summary"]["dense_scene_eligible"] is True
    assert metric["summary"]["dense_scene_blockers"] == []
    assert metric["recommended_action"]["kind"] == "dense-scene"
    assert {item["id"]: item for item in metric["steps"]}["dense-scene"][
        "status"
    ] == "ready"


def test_global_map_failure_is_persisted_without_partial_geometry(tmp_path) -> None:
    api = client(tmp_path)
    project = create_pair_project(api)

    response = api.post(f"/api/v1/projects/{project['id']}/global-map")

    assert response.status_code == 200, response.text
    record = response.json()
    assert record["schema"] == "panorai-streetview-global-map/v1"
    assert record["success"] is False
    assert record["failure_reasons"] == ["no-admitted-pairwise-edges"]
    assert record["poses"] == []
    assert record["points"] == {
        "total_count": 0,
        "displayed_count": 0,
        "display_xy": [],
    }
    assert record["track_count"] == 0
    assert record["recommended_edit"]["kind"] == "add-edge"
    assert record["skipped_pairs"][0]["pair_id"] == "pair-1"

    loaded = api.get(f"/api/v1/projects/{project['id']}/global-map")
    assert loaded.status_code == 200
    assert loaded.json()["revision"] == record["revision"]
    assert loaded.json()["stale"] is False

    setup = api.get(f"/api/v1/projects/{project['id']}/setup").json()
    assert setup["summary"]["global_map_status"] == "failed"


def test_dense_include_exclude_regions_become_explicit_boolean_validity(
    tmp_path,
) -> None:
    api = client(tmp_path)
    project = create_pair_project(api)
    reference = project["images"][0]
    source = project["images"][1]
    for kind, points in (
        ("include_region", [[0, 0], [128, 127.9]]),
        ("exclude_region", [[0, 0], [64, 127.9]]),
    ):
        response = api.post(
            f"/api/v1/projects/{project['id']}/dense-annotations",
            json={
                "kind": kind,
                "image_id": reference["id"],
                "label": kind,
                "points_xy": points,
                "state": "accepted",
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["dense_annotations"][-1]["engine_effect"] == (
            "pair_validity"
        )
    for kind, points in (
        ("include_region", [[128, 0], [255.9, 127.9]]),
        ("exclude_region", [[192, 0], [255.9, 127.9]]),
    ):
        response = api.post(
            f"/api/v1/projects/{project['id']}/dense-annotations",
            json={
                "kind": kind,
                "image_id": source["id"],
                "label": f"source-{kind}",
                "points_xy": points,
                "state": "accepted",
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["dense_annotations"][-1]["engine_effect"] == (
            "pair_validity"
        )

    stored = api.app.state.store.load(project["id"])
    validity, details = _reference_validity(stored, reference["id"], (128, 256))
    source_validity, source_details = _reference_validity(
        stored, source["id"], (128, 256)
    )

    assert validity.dtype == np.bool_
    assert validity.shape == (128, 256)
    assert details["semantics"] == "explicit-reference-validity"
    assert details["eligible_fraction"] == 0.25
    assert not validity[:, :64].any()
    assert validity[:, 64:128].all()
    assert not validity[:, 128:].any()
    assert source_validity.dtype == np.bool_
    assert source_validity.shape == (128, 256)
    assert source_details["semantics"] == "explicit-reference-validity"
    assert source_details["eligible_fraction"] == 0.25
    assert not source_validity[:, :128].any()
    assert source_validity[:, 128:192].all()
    assert not source_validity[:, 192:].any()
    assert set(details["include_annotation_ids"]).isdisjoint(
        source_details["include_annotation_ids"]
    )


def test_dense_scene_builds_public_masked_scene_persists_mesh_and_becomes_stale(
    tmp_path,
    monkeypatch,
) -> None:
    api = client(tmp_path)
    project = create_three_view_project(api)
    store_metric_map(api, project)
    first_image = project["images"][0]
    included = api.post(
        f"/api/v1/projects/{project['id']}/dense-annotations",
        json={
            "kind": "include_region",
            "image_id": first_image["id"],
            "label": "trusted half",
            "points_xy": [[0, 0], [128, 127.9]],
            "state": "accepted",
        },
    )
    assert included.status_code == 200, included.text
    captured = []

    class CapturingDenseApplication:
        def describe(self):
            return {
                "application": "panorai.applications.SphericalDenseReconstruction",
                "route": "test-public-result-capture",
            }

        def run(self, scene):
            captured.append(scene)
            surface = SurfaceMesh(
                vertices_xyz=np.asarray(
                    ((0.0, 0.0, 2.0), (1.0, 0.0, 2.0), (0.0, 1.0, 2.0))
                ),
                faces=np.asarray(((0, 1, 2),), dtype=np.int32),
                normals_xyz=np.asarray(((0.0, 0.0, 1.0),) * 3),
                vertex_weight=np.ones(3, dtype=np.float32),
                vertex_view_count=np.full(3, 3, dtype=np.uint16),
                colors_rgb=np.asarray(
                    ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                    dtype=np.float32,
                ),
                color_validity=np.ones(3, dtype=bool),
                color_contribution_count=np.full(3, 3, dtype=np.uint16),
                units="m",
                frame=scene.frame,
            )
            accepted = len(captured) == 1
            return SphericalDenseReconstructionResult(
                status="accepted" if accepted else "rejected",
                reasons=() if accepted else ("synthetic-quality-rejection",),
                diagnostics={"captured": True},
                surface=surface,
            )

    api.app.state.dense_scene.application_factory = lambda scene, profile: (
        CapturingDenseApplication()
    )
    replace_targets = []
    original_replace = Path.replace

    def observed_replace(source, target):
        replace_targets.append(Path(target).name)
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", observed_replace)
    response = api.post(
        f"/api/v1/projects/{project['id']}/dense-scene",
        json={"profile": "interactive"},
    )

    assert response.status_code == 200, response.text
    record = response.json()
    assert record["schema"] == "panorai-streetview-dense-scene/v1"
    assert record["accepted"] is True, record["reasons"]
    assert record["surface"]["vertex_count"] == 3
    assert record["surface"]["face_count"] == 1
    assert record["surface"]["file_available"] is True
    assert record["surface"]["file_name"] == (f"{project['id']}.revision-0001.ply")
    assert replace_targets[:2] == [
        f"{project['id']}.revision-0001.ply",
        f"{project['id']}.json",
    ]
    assert len(captured) == 1
    scene = captured[0]
    assert isinstance(scene, RegisteredSphericalScene)
    assert scene.units == "m"
    assert len(scene.views) == 3
    assert scene.views[0].validity.dtype == np.bool_
    assert scene.views[0].validity.mean() == 0.5
    assert scene.views[1].validity.all()
    assert record["view_masks"][0]["eligible_fraction"] == 0.5
    mesh = api.get(f"/api/v1/projects/{project['id']}/dense-scene/mesh")
    assert mesh.status_code == 200
    assert mesh.content.startswith(b"ply\nformat ascii 1.0\n")
    assert b"element vertex 3\n" in mesh.content
    assert b"element face 1\n" in mesh.content
    assert (
        api.get(
            f"/api/v1/projects/{project['id']}/dense-scene/mesh?revision=1"
        ).status_code
        == 200
    )

    rejected_response = api.post(
        f"/api/v1/projects/{project['id']}/dense-scene",
        json={"profile": "interactive"},
    )
    assert rejected_response.status_code == 200, rejected_response.text
    rejected = rejected_response.json()
    assert rejected["accepted"] is False
    assert rejected["surface"]["file_available"] is False
    assert rejected["surface"]["vertex_count"] == 0
    assert rejected["surface"]["face_count"] == 0
    assert (
        api.get(
            f"/api/v1/projects/{project['id']}/dense-scene/mesh?revision=1"
        ).status_code
        == 409
    )
    assert (
        api.get(f"/api/v1/projects/{project['id']}/dense-scene/mesh").status_code == 404
    )

    rejected_mask = api.post(
        f"/api/v1/projects/{project['id']}/dense-annotations",
        json={
            "kind": "exclude_region",
            "image_id": project["images"][1]["id"],
            "label": "rejected edit",
            "points_xy": [[0, 0], [32, 32]],
            "state": "rejected",
        },
    )
    assert rejected_mask.status_code == 200, rejected_mask.text
    loaded_after_rejected = api.get(f"/api/v1/projects/{project['id']}/dense-scene")
    assert loaded_after_rejected.status_code == 200
    assert loaded_after_rejected.json()["stale"] is False

    excluded = api.post(
        f"/api/v1/projects/{project['id']}/dense-annotations",
        json={
            "kind": "exclude_region",
            "image_id": project["images"][1]["id"],
            "label": "new edit",
            "points_xy": [[0, 0], [32, 32]],
            "state": "accepted",
        },
    )
    assert excluded.status_code == 200, excluded.text
    loaded = api.get(f"/api/v1/projects/{project['id']}/dense-scene")
    assert loaded.status_code == 200
    assert loaded.json()["stale"] is True


def test_dense_scene_refuses_arbitrary_scale_before_engine_execution(tmp_path) -> None:
    api = client(tmp_path)
    project = create_three_view_project(api)
    store_metric_map(api, project, scale="arbitrary")

    response = api.post(
        f"/api/v1/projects/{project['id']}/dense-scene",
        json={"profile": "interactive"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "dense scene requires metric global-map scale"


def test_dense_scene_ignores_mask_edits_outside_mapped_component(tmp_path) -> None:
    api = client(tmp_path)
    response = api.post(
        "/api/v1/projects",
        data={
            "name": "Disconnected dense view",
            "workflow": "spherical_unregistered",
            "metadata": "{}",
            "pairs": "[]",
        },
        files=[
            ("images", ("a.png", textured_png(), "image/png")),
            ("images", ("b.png", textured_png(4), "image/png")),
            ("images", ("c.png", textured_png(8), "image/png")),
            ("images", ("outside.png", textured_png(12), "image/png")),
        ],
    )
    assert response.status_code == 200, response.text
    project = response.json()
    store_metric_map(api, project, image_count=3)

    class RejectedDenseApplication:
        def describe(self):
            return {"application": "test-component-signature/v1"}

        def run(self, scene):
            assert len(scene.views) == 3
            return SphericalDenseReconstructionResult(
                status="rejected",
                reasons=("synthetic-test",),
                diagnostics={},
                surface=None,
            )

    api.app.state.dense_scene.application_factory = lambda scene, profile: (
        RejectedDenseApplication()
    )
    built = api.post(
        f"/api/v1/projects/{project['id']}/dense-scene",
        json={"profile": "interactive"},
    )
    assert built.status_code == 200, built.text
    assert built.json()["stale"] is False

    outside = api.post(
        f"/api/v1/projects/{project['id']}/dense-annotations",
        json={
            "kind": "exclude_region",
            "image_id": project["images"][3]["id"],
            "label": "outside mapped component",
            "points_xy": [[0, 0], [32, 32]],
            "state": "accepted",
        },
    )
    assert outside.status_code == 200, outside.text
    assert (
        api.get(f"/api/v1/projects/{project['id']}/dense-scene").json()["stale"]
        is False
    )

    mapped = api.post(
        f"/api/v1/projects/{project['id']}/dense-annotations",
        json={
            "kind": "exclude_region",
            "image_id": project["images"][0]["id"],
            "label": "mapped component",
            "points_xy": [[0, 0], [32, 32]],
            "state": "accepted",
        },
    )
    assert mapped.status_code == 200, mapped.text
    assert (
        api.get(f"/api/v1/projects/{project['id']}/dense-scene").json()["stale"] is True
    )


def test_dense_scene_runs_are_serialized_per_project(tmp_path, monkeypatch) -> None:
    api = client(tmp_path)
    project = create_three_view_project(api)
    store_metric_map(api, project)
    service = api.app.state.dense_scene
    first_writing = Event()
    second_writing = Event()
    release_first = Event()
    write_lock = Lock()
    write_count = 0
    original_write = service._write

    class RejectedDenseApplication:
        def describe(self):
            return {"application": "test-rejected-dense/v1"}

        def run(self, scene):
            return SphericalDenseReconstructionResult(
                status="rejected",
                reasons=("test-serialization",),
                diagnostics={},
                surface=None,
            )

    def delayed_write(record, result):
        nonlocal write_count
        with write_lock:
            write_count += 1
            sequence = write_count
        if sequence == 1:
            first_writing.set()
            assert release_first.wait(timeout=2)
        elif sequence == 2:
            second_writing.set()
        original_write(record, result)

    monkeypatch.setattr(
        service,
        "application_factory",
        lambda scene, profile: RejectedDenseApplication(),
    )
    monkeypatch.setattr(service, "_write", delayed_write)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(service.run, project["id"])
        assert first_writing.wait(timeout=2)
        second = executor.submit(service.run, project["id"])
        try:
            assert not second_writing.wait(timeout=0.1)
        finally:
            release_first.set()
        results = [first.result(timeout=4), second.result(timeout=4)]

    assert sorted(item["revision"] for item in results) == [1, 2]
    assert service.load(project["id"])["revision"] == 2


def test_dense_scene_interactive_profile_runs_public_application(tmp_path) -> None:
    api = client(tmp_path)
    project = create_three_view_project(api)
    store_metric_map(api, project)

    response = api.post(
        f"/api/v1/projects/{project['id']}/dense-scene",
        json={"profile": "interactive"},
    )

    assert response.status_code == 200, response.text
    record = response.json()
    assert record["profile"] == "interactive"
    assert record["engine"]["application"] == (
        "panorai.applications.SphericalDenseReconstruction"
    )
    assert record["engine"]["route"] == (
        "multireference-full-sphere-mvs -> dense-tsdf -> marching-cubes"
    )
    assert record["scene"]["view_count"] == 3
    assert len(record["range_maps"]) == 3
    assert record["status"] in {"accepted", "rejected"}
    stages = record["diagnostics"]["pipeline_stages"]
    assert stages == [
        "build_registered_reference_sets",
        "estimate_multi_view_ranges",
        "fuse_tsdf",
        "extract_surface",
        "colorize_surface",
        "validate_dense_reconstruction",
    ]
    mesh = api.get(f"/api/v1/projects/{project['id']}/dense-scene/mesh")
    assert mesh.status_code == (200 if record["accepted"] else 404)


def test_perspective_group_runs_public_stitcher_with_supplied_fov(tmp_path) -> None:
    api = client(tmp_path)
    response = api.post(
        "/api/v1/projects",
        data={
            "name": "Calibrated rotating capture",
            "workflow": "perspective_calibrated",
            "metadata": json.dumps(
                {
                    "left.png": {"horizontal_fov_deg": 92.0},
                    "right.png": {"intrinsics": {"horizontal_fov_deg": 92.0}},
                }
            ),
        },
        files=[
            ("images", ("left.png", perspective_oracle_png(-16), "image/png")),
            ("images", ("right.png", perspective_oracle_png(16), "image/png")),
        ],
    )
    assert response.status_code == 200, response.text
    project = response.json()
    plan = api.get(f"/api/v1/projects/{project['id']}/build-plan").json()
    assert plan["can_build_now"] is True
    assert plan["stages"][2]["engine"] == "panorai.applications.PanoramaStitcher"

    stitched = api.post(f"/api/v1/projects/{project['id']}/stitching")

    assert stitched.status_code == 200, stitched.text
    record = stitched.json()
    assert record["schema"] == "panorai-streetview-stitching/v2"
    assert record["accepted_group_count"] == 1
    group = record["groups"][0]
    assert group["accepted"] is True, group["reasons"]
    assert record["model"] == "central-rotation-only"
    assert record["engine"]["application"] == "panorai.applications.PanoramaStitcher"
    assert group["diagnostics"]["estimated_fov_count"] == 0
    assert group["diagnostics"]["accepted_pair_count"] >= 1
    assert group["output"]["valid_fraction"] > 0.05
    assert all(item["fov_source"] == "supplied" for item in group["sources"])
    panorama = api.get(f"/api/v1/projects/{project['id']}/stitching/image")
    assert panorama.status_code == 200
    with Image.open(BytesIO(panorama.content)) as output:
        assert output.size == (512, 256)


def test_uncalibrated_group_estimates_fov_and_rejection_stays_auditable(
    tmp_path,
) -> None:
    api = client(tmp_path)
    response = api.post(
        "/api/v1/projects",
        data={
            "name": "Unknown camera capture",
            "workflow": "perspective_uncalibrated",
            "metadata": "{}",
        },
        files=[
            ("images", ("a.png", perspective_oracle_png(-16), "image/png")),
            ("images", ("b.png", perspective_oracle_png(16), "image/png")),
        ],
    )
    project = response.json()
    stitched = api.post(f"/api/v1/projects/{project['id']}/stitching")
    assert stitched.status_code == 200, stitched.text
    accepted = stitched.json()["groups"][0]
    assert accepted["accepted"] is True, accepted["reasons"]
    assert accepted["diagnostics"]["estimated_fov_count"] == 2
    assert all(item["fov_source"] == "estimated" for item in accepted["sources"])
    assert all(
        30.0 < item["horizontal_fov_deg"] < 140.0 for item in accepted["intrinsics"]
    )

    rejected_response = api.post(
        "/api/v1/projects",
        data={
            "name": "Disconnected capture",
            "workflow": "perspective_uncalibrated",
            "metadata": "{}",
        },
        files=[
            ("images", ("blank-a.png", png(160, 120), "image/png")),
            ("images", ("blank-b.png", png(160, 120), "image/png")),
        ],
    )
    rejected_project = rejected_response.json()
    rejected = api.post(f"/api/v1/projects/{rejected_project['id']}/stitching")
    assert rejected.status_code == 200, rejected.text
    rejection_record = rejected.json()
    rejection = rejection_record["groups"][0]
    assert rejection["accepted"] is False
    assert rejection_record["status"] == "rejected"
    assert rejection["reasons"]
    assert (
        api.get(
            f"/api/v1/projects/{rejected_project['id']}/stitching/image"
        ).status_code
        == 404
    )


def test_uncalibrated_camera_metadata_overflow_rejects_group_without_500(
    tmp_path,
) -> None:
    api = client(tmp_path)
    huge_focal_length = int("9" * 400)
    response = api.post(
        "/api/v1/projects",
        data={
            "name": "Overflow camera",
            "workflow": "perspective_uncalibrated",
            "metadata": json.dumps(
                {
                    "a.png": {"intrinsics": {"fx": huge_focal_length}},
                    "b.png": {},
                }
            ),
        },
        files=[
            ("images", ("a.png", perspective_oracle_png(-16), "image/png")),
            ("images", ("b.png", perspective_oracle_png(16), "image/png")),
        ],
    )
    assert response.status_code == 200, response.text
    project = response.json()
    plan = api.get(f"/api/v1/projects/{project['id']}/build-plan")
    assert plan.status_code == 200
    assert plan.json()["can_build_now"] is False

    stitched = api.post(f"/api/v1/projects/{project['id']}/stitching")
    assert stitched.status_code == 200, stitched.text
    group = stitched.json()["groups"][0]
    assert group["accepted"] is False
    assert "too large to convert to float" in group["reasons"][0]


def test_multiple_capture_groups_promote_only_accepted_erps_without_poses(
    tmp_path,
    monkeypatch,
) -> None:
    api = client(tmp_path)
    metadata = {
        "a-left.png": {"capture_group": "hall-a"},
        "a-right.png": {"capture_group": "hall-a"},
        "b-left.png": {"capture_group": "hall-b"},
        "b-right.png": {"capture_group": "hall-b"},
        "bad-left.png": {
            "capture_group": "bad",
            "horizontal_fov_deg": "invalid",
        },
        "bad-right.png": {"capture_group": "bad"},
    }
    response = api.post(
        "/api/v1/projects",
        data={
            "name": "x" * 120,
            "workflow": "perspective_uncalibrated",
            "metadata": json.dumps(metadata),
        },
        files=[
            ("images", ("a-left.png", perspective_oracle_png(-16), "image/png")),
            ("images", ("a-right.png", perspective_oracle_png(16), "image/png")),
            ("images", ("b-left.png", perspective_oracle_png(-20), "image/png")),
            ("images", ("b-right.png", perspective_oracle_png(12), "image/png")),
            ("images", ("bad-left.png", png(160, 120), "image/png")),
            ("images", ("bad-right.png", png(160, 120), "image/png")),
        ],
    )
    assert response.status_code == 200, response.text
    project = response.json()

    stitched = api.post(f"/api/v1/projects/{project['id']}/stitching")

    assert stitched.status_code == 200, stitched.text
    record = stitched.json()
    assert record["schema"] == "panorai-streetview-stitching/v2"
    assert record["status"] == "partial"
    assert record["accepted_group_count"] == 2
    assert record["rejected_group_count"] == 1
    assert record["ready_for_promotion"] is True
    groups = {item["group_id"]: item for item in record["groups"]}
    assert groups["hall-a"]["accepted"] is True
    assert groups["hall-b"]["accepted"] is True
    assert groups["bad"]["accepted"] is False
    assert "could not convert string to float" in groups["bad"]["reasons"][0]
    assert (
        api.get(
            f"/api/v1/projects/{project['id']}/stitching/groups/hall-a/image"
        ).status_code
        == 200
    )
    assert (
        api.get(
            f"/api/v1/projects/{project['id']}/stitching/groups/bad/image"
        ).status_code
        == 404
    )

    service = api.app.state.stitching
    original_create = service.store.create
    original_write = service._write
    rerun_writing = Event()
    promotion_creating = Event()
    release_rerun = Event()
    create_lock = Lock()
    create_count = 0
    replace_targets = []
    original_replace = Path.replace

    def observed_create(*args, **kwargs):
        nonlocal create_count
        with create_lock:
            create_count += 1
        promotion_creating.set()
        return original_create(*args, **kwargs)

    def delayed_write(rerun_record, results):
        rerun_writing.set()
        assert release_rerun.wait(timeout=2)
        original_write(rerun_record, results)

    def observed_replace(source, target):
        replace_targets.append(Path(target).name)
        return original_replace(source, target)

    monkeypatch.setattr(service.store, "create", observed_create)
    monkeypatch.setattr(service, "_write", delayed_write)
    monkeypatch.setattr(Path, "replace", observed_replace)
    with ThreadPoolExecutor(max_workers=3) as executor:
        rerun = executor.submit(service.run, project["id"])
        assert rerun_writing.wait(timeout=3)
        first = executor.submit(service.promote, project["id"])
        second = executor.submit(service.promote, project["id"])
        try:
            assert not promotion_creating.wait(timeout=0.1)
        finally:
            release_rerun.set()
        rerun_record = rerun.result(timeout=5)
        promoted_record = first.result(timeout=5)
        repeated_record = second.result(timeout=5)

    promoted = promoted_record.model_dump(mode="json", by_alias=True)
    assert rerun_record["revision"] == 2
    assert replace_targets[:3] == [
        f"{project['id']}.hall-a.revision-0002.png",
        f"{project['id']}.hall-b.revision-0002.png",
        f"{project['id']}.json",
    ]
    rerun_groups = {item["group_id"]: item for item in rerun_record["groups"]}
    assert rerun_groups["hall-a"]["output"]["file_name"] == (
        f"{project['id']}.hall-a.revision-0002.png"
    )
    assert (
        api.get(
            f"/api/v1/projects/{project['id']}/stitching/groups/hall-a/image?revision=1"
        ).status_code
        == 409
    )
    assert (
        api.get(
            f"/api/v1/projects/{project['id']}/stitching/groups/hall-a/image?revision=2"
        ).status_code
        == 200
    )
    assert repeated_record.id == promoted_record.id
    assert create_count == 1
    assert promoted["name"] == f"{'x' * 113} · ERPs"
    assert len(promoted["name"]) == 120
    assert promoted["workflow"] == "spherical_unregistered"
    assert len(promoted["images"]) == 2
    assert len(promoted["pairs"]) == 1
    assert promoted["pairs"][0]["relation"] == "overlap_candidate"
    assert promoted["pairs"][0]["provenance"]["pose_status"] == "unknown"
    assert {
        item["metadata"]["source_capture_group"] for item in promoted["images"]
    } == {"hall-a", "hall-b"}
    assert all("pose" not in item["metadata"] for item in promoted["images"])
    assert all(
        item["metadata"]["pose_status"] == "unknown" for item in promoted["images"]
    )
    setup = api.get(f"/api/v1/projects/{promoted['id']}/setup").json()
    assert setup["summary"]["registered_pair_count"] == 0
    assert setup["pairs"][0]["status"] == "not-run"
