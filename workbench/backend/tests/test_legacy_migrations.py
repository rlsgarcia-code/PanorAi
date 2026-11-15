from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from panorai_workbench.catalog import experiment_catalog
from panorai_workbench.discovery import component_index
from panorai_workbench.migrations import (
    MIGRATED_LEGACY_IDS,
    decode_migrated_upload,
)
from panorai_workbench.runs import RunStore


def _image() -> np.ndarray:
    y, x = np.indices((64, 128))
    base = ((x * 17 + y * 29 + ((x // 4 + y // 4) % 2) * 97) % 256).astype(np.uint8)
    return np.stack((base, np.roll(base, 2, 1), np.roll(base, 3, 0)), axis=-1)


def _png(image: np.ndarray) -> bytes:
    stream = BytesIO()
    Image.fromarray(image).save(stream, format="PNG")
    return stream.getvalue()


def _inputs(store: RunStore) -> tuple[dict, dict]:
    first_image = _image()
    second_image = np.roll(first_image, 2, axis=1)
    result = []
    for name, image in (("a.png", first_image), ("b.png", second_image)):
        item = store.ingest(_png(image), name)
        item["image_shape_hw"] = [64, 128]
        result.append(item)
    return result[0], result[1]


@pytest.mark.parametrize(
    ("experiment_id", "workspace", "parameters", "count"),
    [
        ("geometry-contract", "single", {}, 1),
        ("filters-edges", "single", {"operation": "gaussian", "ksize": 5}, 1),
        ("on-sphere", "single", {"operation": "sobel"}, 1),
        (
            "morphology",
            "single",
            {"operation": "dilate", "radius_deg": 2.0},
            1,
        ),
        ("region-ops", "single", {"operation": "connected-components"}, 1),
        (
            "classical-stereo",
            "two",
            {"algorithm": "sgbm", "max_disparity": 16},
            2,
        ),
        (
            "dog-op",
            "single",
            {"working_width": "original", "max_features": 10},
            1,
        ),
        (
            "projected-view-route",
            "single",
            {"face_size": 64, "retain": "views"},
            1,
        ),
        ("region-analysis-pipeline", "single", {}, 1),
    ],
)
def test_legacy_aliases_execute_through_catalog_derived_engine(
    tmp_path: Path,
    experiment_id: str,
    workspace: str,
    parameters: dict,
    count: int,
) -> None:
    store = RunStore(tmp_path / experiment_id)
    inputs = list(_inputs(store)[:count])

    created = store.create(
        experiment_id=experiment_id,
        workspace=workspace,
        parameters=parameters,
        inputs=inputs,
    )
    assert created["schema"] == "panorai-workbench-run/v5"
    assert created["migration"] == {
        "engine": "catalog-derived",
        "component_id": created["component_id"],
        "preserved_alias": experiment_id,
    }
    assert created["adapter_id"] == created["experiment_id"] == experiment_id
    assert created["component_id"] in component_index()

    completed = store.execute(created["id"])
    assert completed["status"] == "completed", completed.get("error")
    assert completed["artifacts"]
    assert completed["metrics"]["component_id"] in component_index()
    if experiment_id == "projected-view-route":
        labels = {item["label"] for item in completed["artifacts"]}
        assert any("retained.sample" in label for label in labels)
        assert any("result.result.value.image" in label for label in labels)


def test_every_ready_compatibility_alias_has_an_automatic_route() -> None:
    experiments = experiment_catalog()["experiments"]
    ready = {item["id"] for item in experiments if item["adapter_status"] == "ready"}
    automatic = {
        item["id"] for item in experiments if item["execution_path"] == "automatic"
    }

    assert MIGRATED_LEGACY_IDS <= automatic
    assert ready == automatic


def test_automatic_component_ids_do_not_collapse_type_and_function_symbols() -> None:
    descriptors = [
        item
        for item in component_index().values()
        if item["surface_id"] == "spherical-ops-core"
        and item["symbol"]
        in {"Gradient", "gradient", "Laplacian", "laplacian", "Sobel", "sobel"}
    ]

    assert {item["symbol"] for item in descriptors} == {
        "Gradient",
        "gradient",
        "Laplacian",
        "laplacian",
        "Sobel",
        "sobel",
    }
    assert len({item["id"] for item in descriptors}) == len(descriptors)
    assert all(item["status"] == "ready" for item in descriptors)


def test_saved_v2_alias_is_translated_at_execution_time(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    first, _ = _inputs(store)
    created = store.create(
        experiment_id="filters-edges",
        workspace="single",
        parameters={"operation": "laplacian"},
        inputs=[first],
    )
    saved = store.load(created["id"])
    for name in (
        "component_id",
        "configuration",
        "input_values",
        "execution_inputs",
        "migration",
    ):
        saved.pop(name, None)
    saved["schema"] = "panorai-workbench-run/v2"
    store.save(saved)

    completed = store.execute(saved["id"])

    assert completed["status"] == "completed", completed.get("error")
    assert completed["metrics"]["public_api"] == "panorai.ops.spherical.Laplacian"


def test_dog_public_adapter_exposes_typed_config_and_result() -> None:
    descriptors = [
        item
        for item in component_index().values()
        if item["surface_id"] == "spherical-ops-dog-detector"
        and item["symbol"] == "DoGDetector"
    ]
    assert len(descriptors) == 1
    descriptor = descriptors[0]
    assert descriptor["status"] == "ready"
    assert descriptor["configuration_schema"][0]["type"] == (
        "SphericalDoGDetectorConfig"
    )
    assert descriptor["output_schema"]["type"] == "SphericalKeypointSet"


def test_migrated_support_transform_preserves_internal_black_and_uses_nearest(
    tmp_path: Path,
) -> None:
    image = np.full((32, 64, 3), 127, dtype=np.uint8)
    image[:4] = 0
    image[-4:] = 0
    image[16, 20:30] = 0
    path = tmp_path / "polar-voids.png"
    path.write_bytes(_png(image))
    item = {
        "path": str(path),
        "workbench_transform": {
            "kind": "working-support",
            "working_width": "32",
            "support_policy": "auto-polar-voids",
        },
    }

    support = decode_migrated_upload(item)

    assert support.dtype == np.bool_
    assert support.shape == (16, 32)
    assert not support[:2].any()
    assert not support[-2:].any()
    assert support[8, 10:15].all()


def test_migrated_full_erp_never_receives_a_polar_mask(tmp_path: Path) -> None:
    image = _image()
    path = tmp_path / "full.png"
    path.write_bytes(_png(image))
    item = {
        "path": str(path),
        "workbench_transform": {
            "kind": "working-support",
            "working_width": "64",
            "support_policy": "auto-polar-voids",
        },
    }

    support = decode_migrated_upload(item)

    assert support.shape == (32, 64)
    assert support.all()
