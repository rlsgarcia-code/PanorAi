from __future__ import annotations

import json

import numpy as np
from PIL import Image

from panorai_streetview.datasets import import_stanford_selection
from panorai_streetview.store import ProjectStore


def _fixture(tmp_path):
    root = tmp_path / "stanford"
    rgb = root / "rgb" / "area_3"
    poses = root / "poses" / "area_3"
    rgb.mkdir(parents=True)
    poses.mkdir(parents=True)
    views = ["reference", "source-a", "source-b", "source-c"]
    for index, view in enumerate(views):
        Image.new("RGB", (32, 16), (index * 40, 80, 120)).save(rgb / f"{view}.png")
        matrix = np.column_stack((np.eye(3), -np.array([index, 0, 0], dtype=float)))
        np.savetxt(poses / f"{view}.txt", matrix.reshape(1, 12))
    manifest = tmp_path / "selection.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "panorai-public-three-source-stereo-selection/v1",
                "sets": [
                    {
                        "dataset_id": "stanford2d3d",
                        "set_id": "scene-a",
                        "group": "area_3",
                        "reference": "reference",
                        "sources": [
                            {
                                "pair_id": f"pair-{index}",
                                "source": source,
                                "baseline_m": float(index),
                                "minimum_directional_cloud_overlap": 0.9 - index * 0.1,
                            }
                            for index, source in enumerate(views[1:], 1)
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return root, manifest


def test_imports_four_registered_views_and_three_explicit_pairs(tmp_path) -> None:
    dataset, manifest = _fixture(tmp_path)
    store = ProjectStore(tmp_path / "local")

    project = import_stanford_selection(
        store=store,
        dataset_root=dataset,
        selection_manifest=manifest,
        set_id="scene-a",
        input_contract="registered-ground-truth",
    )
    scene = store.scene_manifest(project.id)

    assert project.workflow == "spherical_registered"
    assert len(project.images) == 4
    assert len({item.sha256 for item in project.images}) == 4
    assert len(project.pairs) == 3
    assert len(scene["nodes"]) == 4
    assert len(scene["edges"]) == 3
    assert scene["navigation"] == "explicit registered overlap pairs"


def test_default_import_has_no_poses_or_ground_truth_pair_metrics(tmp_path) -> None:
    dataset, manifest = _fixture(tmp_path)
    for path in (dataset / "poses" / "area_3").glob("*.txt"):
        path.unlink()
    store = ProjectStore(tmp_path / "local")

    project = import_stanford_selection(
        store=store,
        dataset_root=dataset,
        selection_manifest=manifest,
        set_id="scene-a",
    )

    assert project.workflow == "spherical_unregistered"
    assert len(project.images) == 4
    assert all("pose" not in image.metadata for image in project.images)
    assert {image.metadata["dataset"]["role"] for image in project.images} == {"input"}
    assert len(project.pairs) == 6
    assert all(pair.relation == "overlap_candidate" for pair in project.pairs)
    assert all(pair.baseline_m is None for pair in project.pairs)
    assert all(pair.minimum_directional_overlap is None for pair in project.pairs)
    assert {pair.provenance["candidate_policy"] for pair in project.pairs} == {
        "complete-four-view-graph"
    }


def test_import_fails_closed_when_a_selected_view_is_missing(tmp_path) -> None:
    dataset, manifest = _fixture(tmp_path)
    (dataset / "rgb" / "area_3" / "source-b.png").unlink()

    try:
        import_stanford_selection(
            store=ProjectStore(tmp_path / "local"),
            dataset_root=dataset,
            selection_manifest=manifest,
            set_id="scene-a",
        )
    except ValueError as exc:
        assert "missing Stanford RGB view" in str(exc)
    else:
        raise AssertionError("missing selected media must fail closed")
