from io import BytesIO
import json
import zipfile

import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from panorai.reconstruction.planar import encode_planar_mask, decode_planar_mask
from panorai_workbench import api
from panorai_workbench.planar import PlanarSceneStore
from panorai_workbench.runs import RunStore


def _fixture():
    shape = (32, 64)
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    mask = (xx > 22) & (xx < 40) & (yy > 8) & (yy < 24)
    views, tracks = [], []
    for i in range(2):
        views.append(
            {
                "view_id": str(i),
                "image_key": f"view-{i}.png",
                "rotation": np.eye(3).tolist(),
                "center_world": [i * 0.6, 0, 0],
                "pose_convention": "world_to_camera",
                "support": "full",
            }
        )
    for j, (x, y) in enumerate(
        [(x, y) for y in np.linspace(-0.8, 0.8, 4) for x in np.linspace(-0.8, 0.8, 4)]
    ):
        observations = []
        for i in range(2):
            vector = np.array([x - i * 0.6, y, 4.0])
            vector /= np.linalg.norm(vector)
            pixel = [
                (np.arctan2(vector[0], vector[2]) + np.pi) * w / (2 * np.pi) - 0.5,
                (np.pi / 2 - np.arcsin(vector[1])) * h / np.pi - 0.5,
            ]
            observations.append({"view_id": str(i), "pixel_xy": pixel})
        tracks.append(
            {"plane_id": "wall", "track_id": str(j), "observations": observations}
        )
    document = {
        "schema": "panorai-planar-scene/v1",
        "geometry_convention": "panorai-geometry-v1",
        "units": "m",
        "views": views,
        "regions": [
            {
                "plane_id": "wall",
                "view_id": str(i),
                "mask": encode_planar_mask(mask),
                "state": "accepted",
            }
            for i in range(2)
        ],
        "correspondences": tracks,
    }
    stream = BytesIO()
    Image.fromarray(np.full((*shape, 3), 120, np.uint8)).save(stream, format="PNG")
    return document, stream.getvalue()


def test_scene_revisions_are_immutable_and_runs_export_real_native_geometry(tmp_path):
    document, payload = _fixture()
    store = RunStore(tmp_path)
    inputs = [store.ingest(payload, v["image_key"]) for v in document["views"]]
    scenes = PlanarSceneStore(store.project)
    first = scenes.save(document, inputs)
    document["regions"][0]["state"] = "draft"
    second = scenes.save(document, inputs, scene_id=first["scene_id"])
    assert first["revision"] != second["revision"]
    saved = scenes.load(first["scene_id"], first["revision"])
    assert saved["document"]["regions"][0]["state"] == "accepted"
    run = store.create(
        experiment_id="human-guided-planar",
        workspace="multi",
        parameters={"matching": "manual"},
        inputs=inputs,
    )
    run["planar_scene"] = saved["document"]
    store.save(run)
    completed = store.execute(run["id"])
    assert completed["status"] == "completed", completed.get("error")
    assert completed["metrics"]["accepted_planes"] == 1
    path = store.artifact_path(run["id"], "planar-export")
    with zipfile.ZipFile(path) as archive:
        assert {
            "scene.json",
            "result.json",
            "manifest.json",
            "view-0000.npz",
            "view-0000.ply",
        } <= set(archive.namelist())
        result = json.loads(archive.read("result.json"))
        assert result["planes"][0]["offset_m"] == pytest.approx(4, abs=1e-6)
        with np.load(
            BytesIO(archive.read("view-0000.npz")), allow_pickle=False
        ) as maps:
            assert maps["range"].shape == (32, 64)
            assert maps["validity"].any()
            assert maps["source"][maps["validity"]].min() == 2


def test_scene_api_import_revision_and_reopen(tmp_path, monkeypatch):
    store = RunStore(tmp_path)
    monkeypatch.setattr(api, "STORE", store)
    document, payload = _fixture()
    with TestClient(api.app) as client:
        response = client.post(
            "/api/v1/planar/scenes",
            data={"document": json.dumps(document)},
            files=[
                ("images", (v["image_key"], payload, "image/png"))
                for v in document["views"]
            ],
        )
        assert response.status_code == 200, response.text
        scene = response.json()
        base = f"/api/v1/planar/scenes/{scene['scene_id']}/{scene['revision']}"
        assert client.get(base).json()["document"]["revision"] == scene["revision"]
        assert client.get(base + "/images/0").content == payload
        assert client.get(base + "/images/-1").status_code == 404
        document["views"][0]["center_world"] = [0.01, 0, 0]
        revised = client.post(base + "/revisions", json=document)
        assert revised.status_code == 200
        assert revised.json()["revision"] != scene["revision"]
        assert client.get(base).json()["document"]["views"][0]["center_world"] == [
            0,
            0,
            0,
        ]


def test_segmentation_run_exports_draft_alternatives_without_accepting(tmp_path):
    document, payload = _fixture()
    r = document["regions"][0]
    r["state"] = "draft"
    r["interpretation"] = "seed"
    r["roi"] = r["mask"]
    mask = np.zeros((32, 64), bool)
    mask[15:18, 29:33] = True
    r["positive"] = encode_planar_mask(mask)
    r["mask"] = r["positive"]
    store = RunStore(tmp_path)
    inputs = [store.ingest(payload, v["image_key"]) for v in document["views"]]
    scene = PlanarSceneStore(store.project).save(document, inputs)
    run = store.create(
        experiment_id="human-guided-planar",
        workspace="multi",
        parameters={"operation": "segment", "face_size": 128},
        inputs=inputs,
    )
    run["planar_scene"] = scene["document"]
    store.save(run)
    result = store.execute(run["id"])
    assert result["status"] == "completed", result.get("error")
    summary = json.loads(store.artifact_path(run["id"], "planar-result").read_text())
    assert summary["regions"][0]["state"] == "draft"
    assert summary["segmentation_alternatives"]


def test_identifiers_cannot_escape_store(tmp_path):
    store = PlanarSceneStore(tmp_path)
    with pytest.raises(ValueError):
        store.load("../outside", "f" * 64)
    with pytest.raises(ValueError):
        store.load("scene_" + "a" * 32, "../outside")


def test_segment_current_region_only_preserves_unrelated_seed_in_saved_scene(tmp_path):
    document, payload = _fixture()
    first = document["regions"][0]
    first.update(state="draft", interpretation="seed", roi=first["mask"])
    positive = np.zeros((32, 64), bool)
    positive[15:18, 29:33] = True
    first["positive"] = encode_planar_mask(positive)
    first["mask"] = first["positive"]
    # This other draft deliberately has no ROI. Processing it would fail.
    document["regions"][1].update(state="draft", interpretation="seed")
    store = RunStore(tmp_path)
    inputs = [store.ingest(payload, v["image_key"]) for v in document["views"]]
    saved = PlanarSceneStore(store.project).save(document, inputs)
    run = store.create(
        experiment_id="human-guided-planar",
        workspace="multi",
        parameters={"operation": "segment", "face_size": 128, "region_index": 0},
        inputs=inputs,
    )
    run["planar_scene"] = saved["document"]
    store.save(run)
    completed = store.execute(run["id"])
    assert completed["status"] == "completed", completed.get("error")
    summary = json.loads(store.artifact_path(run["id"], "planar-result").read_text())
    assert len(summary["regions"]) == 1
    assert summary["regions"][0]["view_id"] == "0"
    assert summary["regions"][0]["state"] == "draft"
    assert completed["planar_scene"]["regions"][1]["interpretation"] == "seed"
    with zipfile.ZipFile(store.artifact_path(run["id"], "planar-export")) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["segmentation_target"] == {"plane_id": "wall", "view_id": "0"}


@pytest.mark.parametrize("method", ["watershed", "classifier"])
def test_seeded_methods_execute_public_pipeline_and_export_reproducible_history(
    tmp_path, method
):
    document, _ = _fixture()
    shape = (32, 64)
    image = np.full((*shape, 3), (30, 50, 150), np.uint8)
    image[11:22, 27:38] = (30, 150, 50)
    stream = BytesIO()
    Image.fromarray(image).save(stream, format="PNG")
    first = document["regions"][0]
    first.update(state="draft", interpretation="seed", roi=first["mask"])
    positive, negative = np.zeros(shape, bool), np.zeros(shape, bool)
    positive[15:18, 30:33] = True
    negative[15:18, 24:26] = True
    first.update(
        mask=encode_planar_mask(positive),
        positive=encode_planar_mask(positive),
        negative=encode_planar_mask(negative),
    )
    store = RunStore(tmp_path)
    inputs = [
        store.ingest(stream.getvalue(), v["image_key"]) for v in document["views"]
    ]
    saved = PlanarSceneStore(store.project).save(document, inputs)
    run = store.create(
        experiment_id="human-guided-planar",
        workspace="multi",
        parameters={
            "operation": "segment",
            "segmentation": method,
            "face_size": 128,
            "region_index": 0,
        },
        inputs=inputs,
    )
    run["planar_scene"] = saved["document"]
    store.save(run)
    completed = store.execute(run["id"])
    assert completed["status"] == "completed", completed.get("error")
    summary = json.loads(store.artifact_path(run["id"], "planar-result").read_text())
    region = summary["regions"][0]
    assert region["state"] == "draft" and region["source"] == method
    assert region["history"][-1]["algorithm"]
    mask = decode_planar_mask(region["mask"])
    assert mask[positive].all() and not mask[negative].any()
    assert not mask[~decode_planar_mask(first["roi"])].any()
    with zipfile.ZipFile(store.artifact_path(run["id"], "planar-export")) as archive:
        exported = json.loads(archive.read("result.json"))
        assert exported["regions"][0]["history"] == region["history"]
