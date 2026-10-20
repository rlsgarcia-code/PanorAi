from __future__ import annotations

from dataclasses import dataclass
import inspect
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from panorai.core.numpy import Module
from panorai.data import EquirectangularImage
from panorai.pipeline import SampleViews
from panorai_workbench.discovery import discover_components
from panorai_workbench.executor import _bind_inputs
from panorai_workbench.rendering import render_result
from panorai_workbench.runs import RunStore


def _component(symbol: str, surface_id: str | None = None) -> dict:
    return next(
        item
        for item in discover_components()
        if item["symbol"] == symbol
        and (surface_id is None or item["surface_id"] == surface_id)
    )


def _png(array: np.ndarray) -> bytes:
    stream = BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return stream.getvalue()


@dataclass(frozen=True)
class _NestedImageRequest:
    image: np.ndarray
    threshold: int


def _accept_nested_request(request: _NestedImageRequest) -> np.ndarray:
    return request.image


def _count_nested_requests(requests: list[_NestedImageRequest]) -> int:
    return len(requests)


def test_nested_dataclass_uploads_are_bound_by_typed_field_path(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nested.png"
    path.write_bytes(_png(np.zeros((4, 8, 3), dtype=np.uint8)))
    descriptor = {
        "name": "request",
        "kind": "object",
        "fields": {
            "image": {"name": "image", "kind": "erp-image"},
            "threshold": {"name": "threshold", "kind": "integer"},
        },
    }
    positional, keyword = _bind_inputs(
        _accept_nested_request,
        [descriptor],
        [{"field": "request.image", "path": str(path)}],
        {"request": {"threshold": 3}},
    )
    assert positional == []
    assert isinstance(keyword["request"], _NestedImageRequest)
    assert keyword["request"].image.shape == (4, 8, 3)
    assert keyword["request"].threshold == 3


def test_json_sequences_are_coerced_to_typed_dataclasses() -> None:
    descriptor = {"name": "requests", "kind": "array"}
    _, keyword = _bind_inputs(
        _count_nested_requests,
        [descriptor],
        [],
        {
            "requests": [
                {"image": [[0, 1], [2, 3]], "threshold": 2},
                {"image": [[4, 5], [6, 7]], "threshold": 3},
            ]
        },
    )
    assert [item.threshold for item in keyword["requests"]] == [2, 3]
    assert all(isinstance(item, _NestedImageRequest) for item in keyword["requests"])


def test_public_application_is_discovered_from_types_without_adapter() -> None:
    slic = _component("SphericalSLIC", "spherical-applications-slic")
    assert slic["status"] == "ready"
    assert slic["execution_kind"] == "application"
    assert slic["input_schema"][0]["kind"] == "erp-image"
    assert {item["name"] for item in slic["configuration_schema"]} >= {
        "n_segments",
        "compactness",
    }
    assert slic["route"].startswith("/runner.html?component=auto--")


def test_stitching_application_derives_a_multi_image_upload() -> None:
    stitching = _component("PanoramaStitcher", "panorama-stitching-application")
    assert stitching["status"] == "ready", stitching["diagnostic"]
    assert stitching["input_schema"][0]["kind"] == "image-list"


def test_multi_image_uploads_bind_as_one_public_input(tmp_path: Path) -> None:
    paths = []
    for index in range(2):
        path = tmp_path / f"view-{index}.png"
        path.write_bytes(_png(np.full((4, 6, 3), index * 127, dtype=np.uint8)))
        paths.append(path)
    descriptor = {"name": "images", "kind": "image-list"}

    def accept_images(images: list[np.ndarray]) -> int:
        return len(images)

    positional, keyword = _bind_inputs(
        accept_images,
        [descriptor],
        [{"field": "images", "path": str(path)} for path in paths],
        {},
    )
    assert positional == []
    assert len(keyword["images"]) == 2
    assert all(item.shape == (4, 6, 3) for item in keyword["images"])


def test_named_image_map_uploads_bind_by_filename_key(tmp_path: Path) -> None:
    paths = []
    for name in ("front", "back"):
        path = tmp_path / f"{name}.png"
        path.write_bytes(_png(np.zeros((4, 4, 3), dtype=np.uint8)))
        paths.append((name, path))
    descriptor = {"name": "faces", "kind": "image-map"}

    def accept_faces(faces: dict[str, np.ndarray]) -> int:
        return len(faces)

    _, keyword = _bind_inputs(
        accept_faces,
        [descriptor],
        [{"field": f"faces.{name}", "path": str(path)} for name, path in paths],
        {},
    )
    assert set(keyword["faces"]) == {"front", "back"}
    assert all(item.shape == (4, 4, 3) for item in keyword["faces"].values())


def test_opaque_external_model_dependencies_are_inspect_only() -> None:
    mask_rcnn = _component("SphericalMaskRCNNWorkflow", "spherical-applications")
    assert mask_rcnn["status"] == "inspect"
    assert "opaque external dependency" in mask_rcnn["diagnostic"]


def test_typed_upstream_results_are_inspect_only_until_plan_chaining() -> None:
    feature_mask = _component("gnomonic_feature_mask", "spherical-features-core")
    assert feature_mask["status"] == "inspect"
    assert feature_mask["diagnostic"] == (
        "requires upstream typed result: projection_result"
    )
    assert feature_mask["input_schema"][0]["upstream"] is True

    nms = _component("SphericalNMS", "spherical-ops-core")
    assert nms["status"] == "inspect"
    assert nms["diagnostic"] == "requires upstream typed result: observations"


def test_typed_collections_get_generic_runner_controls() -> None:
    cubemap = _component("cubemap_to_equirectangular", "canonical-geometry")
    assert cubemap["status"] == "ready", cubemap["diagnostic"]
    assert cubemap["input_schema"][0]["kind"] == "image-map"

    bundle_adjustment = _component("refine_metric_spherical_landmarks")
    assert bundle_adjustment["status"] == "ready", bundle_adjustment["diagnostic"]
    assert bundle_adjustment["input_schema"][0]["kind"] == "array"


def test_standalone_sample_views_stage_wraps_uploaded_erp(tmp_path: Path) -> None:
    sample = _component("SampleViews", "spherical-pipeline-core")
    assert sample["status"] == "ready", sample["diagnostic"]
    assert sample["input_schema"][0]["container"] == "equirectangular-image"
    path = tmp_path / "erp.png"
    path.write_bytes(_png(np.zeros((8, 16, 3), dtype=np.uint8)))
    _, keyword = _bind_inputs(
        SampleViews(size=4).execute,
        sample["input_schema"],
        [{"field": "value", "path": str(path)}],
        {},
    )
    assert isinstance(keyword["value"], EquirectangularImage)


def test_pipeline_components_are_ready_or_deliberately_inspect_only() -> None:
    pipeline_components = [
        item
        for item in discover_components()
        if item["public_api"].startswith("panorai.pipeline.")
    ]
    assert pipeline_components
    assert not [
        (item["symbol"], item["diagnostic"])
        for item in pipeline_components
        if item["status"] == "blocked"
    ]


def test_abstract_numpy_module_is_inspect_only() -> None:
    assert inspect.isabstract(Module)
    module = _component("Module", "spherical-core-numpy")
    assert module["status"] == "inspect"
    assert module["execution_kind"] == "auxiliary"


def test_auxiliary_and_optional_torch_symbols_remain_inspect_only() -> None:
    components = discover_components()
    assert any(
        item["status"] == "inspect" and item["execution_kind"] == "auxiliary"
        for item in components
    )
    assert all(
        item["status"] == "inspect"
        for item in components
        if "torch-layer" in item["roles"]
    )
    assert all(
        item["status"] == "inspect"
        for item in components
        if item["symbol"]
        .lower()
        .startswith(("acquire_", "download_", "load_", "prefetch_"))
    )


def test_public_gnomonic_function_is_an_automatic_typed_operation() -> None:
    gnomonic = _component("equirectangular_to_gnomonic", "canonical-geometry")
    assert gnomonic["status"] == "ready", gnomonic["diagnostic"]
    assert gnomonic["execution_kind"] == "operation"
    assert gnomonic["input_schema"][0]["kind"] == "erp-image"
    spec = next(item for item in gnomonic["input_schema"] if item["name"] == "spec")
    assert spec["kind"] == "object"


def test_generic_run_executes_public_operation_and_persists_exact_output(
    tmp_path: Path,
) -> None:
    gaussian = _component("GaussianBlur", "spherical-ops-core")
    store = RunStore(tmp_path)
    image = np.zeros((12, 24, 3), dtype=np.uint8)
    image[:, 8:16] = 255
    stored = store.ingest(_png(image), "erp.png")
    stored["field"] = "image"
    run = store.create_component(
        component_id=gaussian["id"],
        configuration={"ksize": 3, "sigma": 1.0},
        input_values={},
        inputs=[stored],
    )
    completed = store.execute(run["id"])
    assert completed["status"] == "completed", completed.get("error")
    assert completed["metrics"]["public_api"] == "panorai.ops.spherical.GaussianBlur"
    exact = next(
        item
        for item in completed["artifacts"]
        if item["mime_type"] == "application/x-npy"
    )
    values = np.load(store.artifact_path(run["id"], exact["id"]), allow_pickle=False)
    assert values.shape == image.shape


def test_generic_run_executes_public_gnomonic_projection(tmp_path: Path) -> None:
    gnomonic = _component("equirectangular_to_gnomonic", "canonical-geometry")
    store = RunStore(tmp_path)
    image = np.zeros((16, 32, 3), dtype=np.uint8)
    image[:, 8:24] = (32, 160, 224)
    stored = store.ingest(_png(image), "erp.png")
    stored["field"] = "image"
    run = store.create_component(
        component_id=gnomonic["id"],
        configuration={},
        input_values={
            "interpolation": "nearest",
            "spec": {
                "center_lat_deg": 0.0,
                "center_lon_deg": 0.0,
                "hfov_deg": 90.0,
                "vfov_deg": 60.0,
                "output_shape_hw": [8, 12],
            },
        },
        inputs=[stored],
    )
    completed = store.execute(run["id"])
    assert completed["status"] == "completed", completed.get("error")
    exact = [
        item
        for item in completed["artifacts"]
        if item["mime_type"] == "application/x-npy"
    ]
    assert any(item["type"] == "image" for item in exact)
    assert any(item["type"] == "mask" for item in exact)


def test_typed_plan_routes_public_projection_result_into_feature_mask(
    tmp_path: Path,
) -> None:
    projection = _component("equirectangular_to_gnomonic", "canonical-geometry")
    feature_mask = _component("gnomonic_feature_mask", "spherical-features-core")
    assert feature_mask["status"] == "inspect"
    assert feature_mask["diagnostic"].startswith("requires upstream typed result:")

    store = RunStore(tmp_path)
    image = np.full((16, 32, 3), 127, dtype=np.uint8)
    stored = store.ingest(_png(image), "erp.png")
    stored["field"] = "projection.image"
    run = store.create_plan(
        steps=[
            {
                "id": "projection",
                "component_id": projection["id"],
                "configuration": {},
                "input_values": {
                    "interpolation": "nearest",
                    "spec": {
                        "center_lat_deg": 0.0,
                        "center_lon_deg": 0.0,
                        "hfov_deg": 90.0,
                        "vfov_deg": 60.0,
                        "output_shape_hw": [8, 12],
                    },
                },
            },
            {
                "id": "feature-mask",
                "component_id": feature_mask["id"],
                "configuration": {},
                "input_values": {"edge_margin_px": 1},
                "bindings": {"projection_result": {"step": "projection"}},
            },
        ],
        inputs=[stored],
    )
    completed = store.execute(run["id"])
    assert completed["schema"] == "panorai-workbench-run/v4"
    assert completed["status"] == "completed", completed.get("error")
    assert completed["metrics"]["plan_step_count"] == 2
    assert [item["public_api"] for item in completed["metrics"]["plan_steps"]] == [
        "panorai.geometry.equirectangular_to_gnomonic",
        "panorai.features.gnomonic_feature_mask",
    ]
    exact = next(
        item
        for item in completed["artifacts"]
        if item["type"] == "mask" and item["mime_type"] == "application/x-npy"
    )
    mask = np.load(store.artifact_path(run["id"], exact["id"]), allow_pickle=False)
    assert mask.shape == (8, 12)
    assert exact["interpolation"] == "nearest"
    assert not mask[[0, -1], :].any()
    assert not mask[:, [0, -1]].any()
    assert mask[2:-2, 2:-2].all()


def test_plan_rejects_unbound_or_incompatible_upstream_inputs(tmp_path: Path) -> None:
    projection = _component("equirectangular_to_gnomonic", "canonical-geometry")
    feature_mask = _component("gnomonic_feature_mask", "spherical-features-core")
    store = RunStore(tmp_path)
    base_step = {
        "id": "projection",
        "component_id": projection["id"],
        "configuration": {},
        "input_values": {
            "image": np.zeros((4, 8, 3), dtype=np.uint8).tolist(),
            "spec": {"output_shape_hw": [4, 4]},
        },
    }
    mask_step = {
        "id": "mask",
        "component_id": feature_mask["id"],
        "configuration": {},
        "input_values": {"edge_margin_px": 0},
    }
    try:
        store.create_plan(steps=[base_step, mask_step], inputs=[])
    except ValueError as exc:
        assert "requires a prior typed result" in str(exc)
    else:
        raise AssertionError("unbound upstream input was accepted")

    incompatible = {
        **mask_step,
        "bindings": {"projection_result": {"step": "projection", "path": ["data"]}},
    }
    try:
        store.create_plan(steps=[base_step, incompatible], inputs=[])
    except ValueError as exc:
        assert "unknown public result field" in str(exc)
    else:
        raise AssertionError("undeclared result traversal was accepted")


def test_generic_runner_executes_slic_and_dog_applications(tmp_path: Path) -> None:
    y, x = np.indices((32, 64))
    base = ((x * 17 + y * 29 + ((x // 4 + y // 4) % 2) * 97) % 256).astype(np.uint8)
    image = np.stack((base, np.roll(base, 2, 1), np.roll(base, 3, 0)), axis=-1)
    store = RunStore(tmp_path)
    stored = store.ingest(_png(image), "textured-erp.png")
    stored["field"] = "panorama"

    slic = _component("SphericalSLIC", "spherical-applications-slic")
    slic_run = store.create_component(
        component_id=slic["id"],
        configuration={"n_segments": 12, "max_num_iter": 3},
        input_values={},
        inputs=[stored],
    )
    slic_run = store.execute(slic_run["id"])
    assert slic_run["status"] == "completed", slic_run.get("error")
    labels = next(
        item
        for item in slic_run["artifacts"]
        if item["type"] == "labels" and item["mime_type"] == "application/x-npy"
    )
    assert labels["interpolation"] == "nearest"

    dog = _component("SphericalDoGFeatures", "spherical-applications")
    dog_run = store.create_component(
        component_id=dog["id"],
        configuration={
            "config": {
                "octaves": 1,
                "levels_per_octave": 2,
                "max_features": 12,
                "patch_size": 24,
                "convolution_backend": "numpy",
            }
        },
        input_values={"panorama_id": "automatic-runner"},
        inputs=[stored],
    )
    dog_run = store.execute(dog_run["id"])
    assert dog_run["status"] == "completed", dog_run.get("error")
    assert dog_run["metrics"]["result_type"].endswith("SphericalFeatureSet")
    assert any(item["label"].endswith("erp-points") for item in dog_run["artifacts"])


def test_generic_runner_executes_typed_spherical_stereo(tmp_path: Path) -> None:
    y, x = np.indices((8, 16))
    base = ((x * 19 + y * 31) % 256).astype(np.uint8)
    reference = np.stack((base, np.roll(base, 1, 0), np.roll(base, 2, 1)), axis=-1)
    source = np.roll(reference, 1, axis=1)
    store = RunStore(tmp_path)
    inputs = []
    for field, image in (("reference", reference), ("source", source)):
        stored = store.ingest(_png(image), f"{field}.png")
        stored["field"] = field
        inputs.append(stored)
    stereo = _component("SphericalStereoDepth", "spherical-applications")
    run = store.create_component(
        component_id=stereo["id"],
        configuration={
            "options": {
                "min_range": 0.5,
                "max_range": 2.0,
                "num_hypotheses": 3,
                "window_size": 3,
                "bidirectional_consistency": False,
                "filter_backend": "numpy",
            }
        },
        input_values={
            "rotation_source_from_reference": np.eye(3).tolist(),
            "translation_source_from_reference": [0.05, 0.0, 0.0],
        },
        inputs=inputs,
    )
    completed = store.execute(run["id"])
    assert completed["status"] == "completed", completed.get("error")
    assert any(
        item["type"] == "mask" and item["interpolation"] == "nearest"
        for item in completed["artifacts"]
    )


@dataclass(frozen=True)
class _TypedRasterResult:
    labels: np.ndarray
    support: np.ndarray
    validity: np.ndarray


def test_result_router_uses_nearest_for_discrete_outputs_and_keeps_masks_separate(
    tmp_path: Path,
) -> None:
    result = _TypedRasterResult(
        labels=np.array([[0, 1], [2, 3]], dtype=np.int32),
        support=np.array([[True, True], [False, True]]),
        validity=np.array([[True, False], [True, True]]),
    )
    artifacts, _ = render_result(result, tmp_path)
    exact = [item for item in artifacts if item["mime_type"] == "application/x-npy"]
    assert {item["type"] for item in exact} == {"labels", "mask"}
    assert all(item["interpolation"] == "nearest" for item in exact)
    support = next(item for item in exact if item["label"].endswith("support"))
    validity = next(item for item in exact if item["label"].endswith("validity"))
    assert support["id"] != validity["id"]


def test_unknown_browser_component_id_cannot_be_resolved(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    try:
        store.create_component(
            component_id="os.system",
            configuration={},
            input_values={},
            inputs=[],
        )
    except ValueError as exc:
        assert "unknown component" in str(exc)
    else:
        raise AssertionError("arbitrary browser import was accepted")
