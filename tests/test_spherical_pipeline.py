from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import panorai as pa
from panorai.ops import spherical as S
from panorai.pipeline import PipelineRun

ROOT = Path(__file__).resolve().parents[1]


def _assert_same_panorama(left, right, modalities=("image",)) -> None:
    for name in modalities:
        np.testing.assert_array_equal(
            left._workflow_data()[name], right._workflow_data()[name]
        )
        np.testing.assert_array_equal(left.validity(name), right.validity(name))
        np.testing.assert_array_equal(
            left._workflow_support[name], right._workflow_support[name]
        )


def test_root_and_canonical_pipeline_exports_are_additive() -> None:
    from panorai import (
        AggregateViews,
        Attach,
        CAMGuidedSphericalInstances,
        CAMRegionProposals,
        LiftToBearings,
        OnSphere,
        Output,
        Parallel,
        PerView,
        Pipeline,
        RasterField,
        ReconstructERP,
        ResultBundle,
        SampleViews,
        SAMMaskRefinement,
        Select,
        SphericalObservations,
        SphericalCAM,
        SphericalInstanceFusion,
        Stage,
        StageResult,
    )
    from panorai.pipeline import ClassificationResult, EmbeddingResult, StageRecord

    assert Pipeline is pa.Pipeline
    assert AggregateViews is pa.AggregateViews
    assert Attach is pa.Attach
    assert CAMGuidedSphericalInstances is pa.CAMGuidedSphericalInstances
    assert CAMRegionProposals is pa.CAMRegionProposals
    assert LiftToBearings is pa.LiftToBearings
    assert OnSphere is pa.OnSphere
    assert Output is pa.Output
    assert Parallel is pa.Parallel
    assert PerView is pa.PerView
    assert RasterField is pa.RasterField
    assert ReconstructERP is pa.ReconstructERP
    assert ResultBundle is pa.ResultBundle
    assert SampleViews is pa.SampleViews
    assert SAMMaskRefinement is pa.SAMMaskRefinement
    assert Select is pa.Select
    assert SphericalObservations is pa.SphericalObservations
    assert SphericalCAM is pa.SphericalCAM
    assert SphericalInstanceFusion is pa.SphericalInstanceFusion
    assert Stage is pa.Stage
    assert StageResult is pa.StageResult
    assert ClassificationResult is pa.ClassificationResult
    assert EmbeddingResult is pa.EmbeddingResult
    assert StageRecord.__name__ == "StageRecord"


def test_detection_lift_matches_canonical_geometry_and_feature_nms() -> None:
    from panorai.features import deduplicate_spherical_keypoints
    from panorai.geometry import gnomonic_pixels_to_rays

    panorama = pa.EquirectangularImage(np.ones((18, 36, 3), dtype=np.float32))
    point = np.asarray(((3.0, 3.0),), dtype=np.float64)

    def detector(_face):
        return {
            "points_xy": point,
            "scores": np.asarray((0.75,), dtype=np.float64),
            "labels": np.asarray((4,), dtype=np.int64),
            "features": np.asarray(((1.0, 2.0),), dtype=np.float32),
        }

    faces = panorama.views("cube", size=7, rotations=((0.0, 0.0),))
    expected = np.stack(
        [gnomonic_pixels_to_rays(point, face.spec).rays_xyz[0] for face in faces]
    )
    pipeline = pa.Pipeline(
        pa.SampleViews("cube", size=7, rotations=((0.0, 0.0),)),
        pa.PerView(pa.Output(detector, kind="detections")),
        pa.LiftToBearings(),
        pa.AggregateViews(S.SphericalNMS(angle_deg=0.01)),
    )
    run = pipeline.run(panorama, retain=["per_view", "lift_to_bearings"])
    lifted = run["lift_to_bearings"]

    assert len(faces) == 12
    assert len(run["per_view"]) == 12
    assert isinstance(lifted, pa.SphericalObservations)
    np.testing.assert_allclose(lifted.bearings_xyz, expected, atol=1e-15)
    assert lifted.valid.all()
    assert lifted.features.shape == (12, 2)
    deduplicated = deduplicate_spherical_keypoints(
        lifted.bearings_xyz,
        lifted.scores,
        angular_threshold_rad=np.deg2rad(0.01),
    )
    np.testing.assert_array_equal(
        run.output.bearings_xyz,
        lifted.bearings_xyz[deduplicated.selected_indices],
    )
    assert len(run.output) == 6
    assert set(run.output.source_face_ids).issubset(set(lifted.source_face_ids))
    assert run.records[-1].details["reducer"] == "SphericalNMS"


@pytest.mark.parametrize(
    ("payload", "geometry", "vertex_count"),
    [
        ({"boxes_xyxy": np.asarray(((1, 1, 5, 4),))}, "boxes", 4),
        (
            {"polygons_xy": (np.asarray(((1, 1), (5, 1), (4, 5), (2, 4))),)},
            "polygons",
            4,
        ),
    ],
)
def test_box_and_polygon_footprints_use_canonical_gnomonic_rays(
    payload, geometry, vertex_count
) -> None:
    from panorai.geometry import gnomonic_pixels_to_rays
    from panorai.pipeline import PerViewResults, ViewResult

    panorama = pa.EquirectangularImage(np.ones((12, 24, 3), dtype=np.float32))
    face = panorama.views("cube", size=7)[0]
    source = face._workflow_data()["image"]
    view_results = PerViewResults(
        "detections",
        (
            ViewResult(
                face.face_id,
                face.spec,
                payload,
                face._workflow_support["image"],
                face.validity("image"),
            ),
        ),
        source_field="image",
        erp_shape_hw=(12, 24),
    )
    lifted = pa.Pipeline(pa.LiftToBearings())(view_results)

    assert lifted.geometry == geometry
    assert lifted.footprints_xyz[0].shape == (vertex_count, 3)
    np.testing.assert_allclose(
        lifted.footprints_xyz[0],
        gnomonic_pixels_to_rays(lifted.source_pixels_xy[0], face.spec).rays_xyz,
        atol=1e-15,
    )
    assert source.shape[:2] == (7, 7)


def test_lifting_keeps_geometric_support_and_data_validity_explicit() -> None:
    from panorai.pipeline import PerViewResults, ViewResult

    panorama = pa.EquirectangularImage(np.ones((12, 24, 3), dtype=np.float32))
    face = panorama.views("cube", size=7)[0]
    support = face._workflow_support["image"].copy()
    validity = face.validity("image")
    support[3, 3] = False
    results = PerViewResults(
        "keypoints",
        (
            ViewResult(
                face.face_id,
                face.spec,
                np.asarray(((3.0, 3.0),)),
                support,
                validity,
            ),
        ),
        source_field="image",
        erp_shape_hw=(12, 24),
    )

    lifted = pa.Pipeline(pa.LiftToBearings())(results)
    assert not lifted.valid[0]
    assert not lifted.support[0]
    assert lifted.data_validity[0]
    assert np.isfinite(lifted.bearings_xyz[0]).all()


def test_spherical_nms_is_label_aware_by_default() -> None:
    bearings = np.asarray(((0, 0, 1), (0, 0, 1)), dtype=np.float64)
    observations = pa.SphericalObservations(
        bearings,
        footprints_xyz=(bearings[:1], bearings[1:]),
        scores=np.asarray((0.9, 0.8)),
        support=np.asarray((True, True)),
        data_validity=np.asarray((True, True)),
        valid=np.asarray((True, True)),
        source_face_ids=np.asarray(("a", "b"), dtype=object),
        source_indices=np.asarray((0, 0)),
        source_pixels_xy=(np.asarray(((1, 1),)), np.asarray(((1, 1),))),
        geometry="points",
        labels=np.asarray((1, 2)),
    )

    assert len(S.SphericalNMS(angle_deg=1.0)(observations)) == 2
    selected = S.SphericalNMS(angle_deg=1.0, class_aware=False)(observations)
    assert len(selected) == 1
    assert selected.scores[0] == pytest.approx(0.9)


@pytest.mark.parametrize("kind", ["classification", "embedding"])
def test_global_view_results_aggregate_by_direct_numpy_oracle(kind) -> None:
    panorama = pa.EquirectangularImage(
        np.linspace(0, 1, 12 * 24 * 3, dtype=np.float32).reshape(12, 24, 3)
    )
    faces = panorama.views("cube", size=7)
    expected_values = tuple(
        np.asarray((face.image.mean(), face.image.std()), dtype=np.float32)
        for face in faces
    )
    expected = np.stack(expected_values).mean(axis=0)

    pipeline = pa.Pipeline(
        pa.SampleViews("cube", size=7),
        pa.PerView(
            pa.Output(
                lambda face: np.asarray((face.mean(), face.std()), dtype=np.float32),
                kind=kind,
            )
        ),
        pa.AggregateViews("mean"),
    )
    actual = pipeline(panorama)
    expected_type = (
        pa.ClassificationResult if kind == "classification" else pa.EmbeddingResult
    )
    assert isinstance(actual, expected_type)
    np.testing.assert_allclose(actual.value, expected, rtol=1e-7, atol=1e-7)


def test_torch_view_aggregation_preserves_device_and_autograd() -> None:
    torch = pytest.importorskip("torch")
    values = torch.linspace(0, 1, 3 * 8 * 16, dtype=torch.float64).reshape(3, 8, 16)
    values.requires_grad_()
    panorama = pa.EquirectangularImage(values)
    result = pa.Pipeline(
        pa.SampleViews("cube", size=5),
        pa.PerView(
            pa.Output(
                lambda face: face.mean(dim=(-2, -1)),
                kind="embedding",
            )
        ),
        pa.AggregateViews("mean"),
    )(panorama)

    assert isinstance(result, pa.EmbeddingResult)
    assert result.value.shape == (3,)
    assert result.value.device == values.device
    result.value.sum().backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()


def test_on_sphere_can_return_a_declared_global_classification() -> None:
    class SphereClassifier:
        spherical_operation = True
        preserves_shadow_state = True

        def __call__(self, image):
            return np.asarray((image.mean(), image.max()), dtype=np.float32)

        @staticmethod
        def transform_mask(mask):
            return mask.copy()

        @staticmethod
        def describe():
            return {"type": "SphereClassifier", "domain": "sphere"}

    panorama = pa.EquirectangularImage(np.ones((8, 16, 3), dtype=np.float32))
    pipeline = pa.Pipeline(
        pa.OnSphere(
            pa.Output(SphereClassifier(), kind="classification"),
            output="classification",
        )
    )
    result = pipeline(panorama)
    assert isinstance(result, pa.ClassificationResult)
    np.testing.assert_array_equal(
        result.value, np.asarray((1.0, 1.0), dtype=np.float32)
    )
    assert pipeline.output_kind == "result"


def test_mixed_per_view_bundle_preserves_non_raster_results() -> None:
    from panorai.pipeline import PerViewResults

    detector = pa.Output(lambda _face: {"scores": [1.0]}, kind="detections")
    panorama = pa.EquirectangularImage(np.ones((8, 16, 3), dtype=np.float32))
    pipeline = pa.Pipeline(
        pa.SampleViews("cube", size=5),
        pa.PerView.parallel(
            labels=lambda face: np.zeros(face.shape[:2], dtype=np.int16),
            detections=detector,
        ),
        pa.ReconstructERP(labels="closest"),
    )
    bundle = pipeline(panorama)
    assert isinstance(bundle, pa.ResultBundle)
    assert isinstance(bundle["detections"], PerViewResults)
    assert bundle["erp"].labels.shape == (8, 16)
    with pytest.raises(ValueError, match="exactly one"):
        pa.Pipeline(
            pa.SampleViews("cube", size=5),
            pa.PerView(detector),
            pa.LiftToBearings(),
        )(panorama)


def test_registered_pipeline_config_round_trip_is_deterministic() -> None:
    rotation = np.asarray(
        ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)),
        dtype=np.float64,
    )
    pipeline = pa.Pipeline(
        pa.Parallel(
            original=None,
            transformed=pa.Pipeline(
                pa.OnSphere(S.Rotate(rotation, interpolation="nearest")),
                pa.OnSphere(S.Resize((6, 12))),
            ),
        ),
        pa.Select("transformed"),
    )
    config = pipeline.to_config()
    restored = pa.Pipeline.from_config(json.loads(json.dumps(config)))
    assert restored.to_config() == config
    assert restored.describe() == pipeline.describe()

    values = np.arange(8 * 16 * 3, dtype=np.float32).reshape(8, 16, 3)
    panorama = pa.EquirectangularImage(values)
    _assert_same_panorama(restored(panorama), pipeline(panorama))


def test_registered_stage_inventory_round_trips_without_dynamic_imports() -> None:
    from panorai.pipeline import registered_pipeline_components

    expected = {
        "panorai.applications.AssessRelativePoseCapture",
        "panorai.applications.RelativePoseCapturePolicy",
        "panorai.estimators.EstimateRelativePose",
        "panorai.estimators.RelativePoseAcceptancePolicy",
        "panorai.estimators.RelativePoseOptions",
        "panorai.estimators.SpatiallyWeightedFivePointSampler",
        "panorai.estimators.UniformFivePointSampler",
        "panorai.features.ExtractSphericalFeaturePair",
        "panorai.features.FaceSetSpec",
        "panorai.features.FeatureExtractorConfig",
        "panorai.features.FeatureMatcherConfig",
        "panorai.features.MatchSphericalFeatures",
        "panorai.features.SphericalDoGDetectorConfig",
        "panorai.features.SphericalFeaturePipelineConfig",
        "panorai.features.ToBearingCorrespondences",
        "panorai.ops.spherical.AreaAverage",
        "panorai.ops.spherical.AreaHistogram",
        "panorai.ops.spherical.BilateralFilter",
        "panorai.ops.spherical.BoxBlur",
        "panorai.ops.spherical.Canny",
        "panorai.ops.spherical.Closing",
        "panorai.ops.spherical.ConnectedComponents",
        "panorai.ops.spherical.Dilate",
        "panorai.ops.spherical.DoGDetector",
        "panorai.ops.spherical.Erode",
        "panorai.ops.spherical.FillHoles",
        "panorai.ops.spherical.Filter2D",
        "panorai.ops.spherical.GaussianBlur",
        "panorai.ops.spherical.GaussianPyramid",
        "panorai.ops.spherical.GeodesicDisk",
        "panorai.ops.spherical.GeodesicDistanceTransform",
        "panorai.ops.spherical.Gradient",
        "panorai.ops.spherical.Laplacian",
        "panorai.ops.spherical.LaplacianPyramid",
        "panorai.ops.spherical.MedianBlur",
        "panorai.ops.spherical.Opening",
        "panorai.ops.spherical.Resample",
        "panorai.ops.spherical.Resize",
        "panorai.ops.spherical.RegionBoundaries",
        "panorai.ops.spherical.Rotate",
        "panorai.ops.spherical.SLIC",
        "panorai.ops.spherical.Sobel",
        "panorai.ops.spherical.SphericalNMS",
        "panorai.ops.spherical.Skeletonize",
        "panorai.pipeline.AggregateViews",
        "panorai.pipeline.AnalyzeRegions",
        "panorai.pipeline.Attach",
        "panorai.pipeline.LiftToBearings",
        "panorai.pipeline.OnSphere",
        "panorai.pipeline.Output",
        "panorai.pipeline.Parallel",
        "panorai.pipeline.PerView",
        "panorai.pipeline.PerViewParallel",
        "panorai.pipeline.ReconstructERP",
        "panorai.pipeline.SampleViews",
        "panorai.pipeline.Select",
        "panorai.pipeline.SphericalWatershed",
        "panorai.stereo.classical.ADCensusCost",
        "panorai.stereo.classical.AbsoluteDifferenceCost",
        "panorai.stereo.classical.BirchfieldTomasiCost",
        "panorai.stereo.classical.BoxCostAggregation",
        "panorai.stereo.classical.BuildStereoCost",
        "panorai.stereo.classical.CensusCost",
        "panorai.stereo.classical.CrossBasedCostAggregation",
        "panorai.stereo.classical.ELASSupportPrior",
        "panorai.stereo.classical.MoreGlobalMatching",
        "panorai.stereo.classical.OpenCVStereoMatcher",
        "panorai.stereo.classical.PatchMatchStereo",
        "panorai.stereo.classical.RefineStereoDisparity",
        "panorai.stereo.classical.SelectStereoDisparity",
        "panorai.stereo.classical.SemiGlobalMatching",
        "panorai.stereo.classical.TransformStereoCost",
        "panorai.stitching.BlendStitchingPanorama",
        "panorai.stitching.CalibrateAndRegister",
        "panorai.stitching.CompensateExposure",
        "panorai.stitching.DetectStitchingFeatures",
        "panorai.stitching.DiscoverOverlaps",
        "panorai.stitching.StitchingOptions",
        "panorai.stitching.WarpRegisteredImages",
    }
    assert set(registered_pipeline_components()) == expected

    declarations = (
        pa.Pipeline(
            pa.SampleViews("fibonacci", count=4, size=5),
            pa.PerView(S.Resize((5, 5))),
            pa.ReconstructERP(image="closest"),
        ),
        pa.Pipeline(
            pa.LiftToBearings(),
            pa.AggregateViews(S.SphericalNMS(angle_deg=2.0)),
        ),
        pa.Pipeline(pa.Attach(target="base", depth="prediction.depth")),
        pa.Pipeline(
            pa.PerView.parallel(
                mask=pa.Output(S.Resize((5, 5), interpolation="nearest"), kind="mask")
            )
        ),
        pa.Pipeline(
            pa.OnSphere(
                pa.Output(S.GaussianBlur(3, 0.7), kind="image"),
                output="image",
            )
        ),
    )
    for declaration in declarations:
        restored = pa.Pipeline.from_config(declaration.to_config())
        assert restored.to_config() == declaration.to_config()


def test_unregistered_callables_and_unknown_components_fail_with_paths() -> None:
    from panorai.pipeline import PipelineSerializationError

    pipeline = pa.Pipeline(
        pa.SampleViews("cube", size=5),
        pa.PerView(lambda face: face),
    )
    with pytest.raises(
        PipelineSerializationError,
        match=r"steps\.per_view\.operation: unregistered component",
    ):
        pipeline.to_config()

    config = pa.Pipeline(pa.OnSphere(S.Resize((4, 8)))).to_config()
    config["steps"][0]["stage"]["id"] = "untrusted.module.ExecuteMe"
    with pytest.raises(PipelineSerializationError, match="unknown or unregistered"):
        pa.Pipeline.from_config(config)

    non_finite = pa.Pipeline(pa.OnSphere(S.GaussianBlur(3, 0.8))).to_config()
    non_finite["steps"][0]["stage"]["parameters"]["operation"]["parameters"][
        "sigma"
    ] = float("nan")
    with pytest.raises(PipelineSerializationError, match="non-finite"):
        pa.Pipeline.from_config(non_finite)


def test_pipeline_archive_verifies_checksum_manifest_and_overwrite_policy(
    tmp_path: Path,
) -> None:
    from panorai.pipeline import PipelineSerializationError

    pipeline = pa.Pipeline(
        pa.OnSphere(S.GaussianBlur(3, 0.8)),
        pa.OnSphere(S.Resize((4, 8))),
    )
    archive = pipeline.save(tmp_path / "registered-pipeline")
    assert sorted(item.name for item in archive.iterdir()) == [
        "manifest.json",
        "pipeline.json",
    ]
    manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "panorai-pipeline-archive/v1"
    assert manifest["registered_components"] == sorted(
        manifest["registered_components"]
    )
    restored = pa.Pipeline.load(archive)
    assert restored.to_config() == pipeline.to_config()

    with pytest.raises(FileExistsError, match="already exists"):
        pipeline.save(archive)

    payload = (archive / "pipeline.json").read_text(encoding="utf-8")
    (archive / "pipeline.json").write_text(payload + " ", encoding="utf-8")
    with pytest.raises(PipelineSerializationError, match="checksum mismatch"):
        pa.Pipeline.load(archive)

    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "keep.txt").write_text("user data", encoding="utf-8")
    with pytest.raises(PipelineSerializationError, match="unrelated files"):
        pipeline.save(unrelated, overwrite=True)
    assert (unrelated / "keep.txt").read_text(encoding="utf-8") == "user data"


def test_content_addressed_cache_covers_values_support_validity_and_parameters(
    tmp_path: Path,
) -> None:
    height, width = 8, 16
    image = np.ones((height, width, 3), dtype=np.float32)
    depth = np.zeros((height, width), dtype=np.float32)
    support = np.ones((height, width), dtype=bool)
    validity = np.ones((height, width), dtype=bool)
    support[1, 2] = False
    validity[3, 5] = False
    panorama = pa.EquirectangularImage(
        image, support_mask=support, valid=validity
    ).with_depth(depth, units="m")
    cache = tmp_path / "cache"
    pipeline = pa.Pipeline(
        pa.OnSphere(S.GaussianBlur(3, 0.8)),
        memory=cache,
    )

    cold = pipeline.run(panorama)
    warm = pipeline.run(panorama)
    assert cold.manifest()["cache"]["hit"] is False
    assert warm.manifest()["cache"]["hit"] is True
    assert cold.manifest()["cache"]["key"] == warm.manifest()["cache"]["key"]
    assert warm.records[0].stage == "ContentAddressedCache"
    _assert_same_panorama(cold.output, warm.output, modalities=("image", "depth"))
    assert warm.output._workflow_metadata["image"].keys() == (
        cold.output._workflow_metadata["image"].keys()
    )
    json.dumps(warm.manifest(), allow_nan=False)
    assert not list(cache.rglob("*.pkl"))
    assert not list(cache.rglob("*.pickle"))

    variants = []
    changed_value = image.copy()
    changed_value[0, 0, 0] = 2.0
    variants.append(
        pa.EquirectangularImage(
            changed_value, support_mask=support, valid=validity
        ).with_depth(depth, units="m")
    )
    changed_support = support.copy()
    changed_support[0, 1] = False
    variants.append(
        pa.EquirectangularImage(
            image, support_mask=changed_support, valid=validity
        ).with_depth(depth, units="m")
    )
    changed_validity = validity.copy()
    changed_validity[0, 1] = False
    variants.append(
        pa.EquirectangularImage(
            image, support_mask=support, valid=changed_validity
        ).with_depth(depth, units="m")
    )
    keys = {cold.manifest()["cache"]["key"]}
    for variant in variants:
        keys.add(pipeline.run(variant).manifest()["cache"]["key"])
    parameter_variant = pipeline.with_params(
        on_sphere__operation=S.GaussianBlur(5, 1.1)
    )
    keys.add(parameter_variant.run(panorama).manifest()["cache"]["key"])
    assert len(keys) == 5

    layout_pipeline = pa.Pipeline(
        pa.Parallel(value=None),
        pa.Select("value"),
        memory=tmp_path / "layout-cache",
    )
    c_layout = np.arange(24, dtype=np.float32).reshape(4, 6)
    f_layout = np.asfortranarray(c_layout)
    assert (
        layout_pipeline.run(c_layout).manifest()["cache"]["key"]
        != layout_pipeline.run(f_layout).manifest()["cache"]["key"]
    )

    retained = pipeline.run(panorama, retain="on_sphere")
    assert retained.manifest()["cache"]["bypassed"] is True
    assert retained.records[0].stage == "OnSphere"


def test_cache_corruption_and_unsupported_torch_fail_explicitly(tmp_path: Path) -> None:
    from panorai.pipeline import PipelineCacheError

    panorama = pa.EquirectangularImage(np.ones((8, 16, 3), dtype=np.float32))
    pipeline = pa.Pipeline(pa.OnSphere(S.GaussianBlur(3, 0.8)), memory=tmp_path)
    pipeline(panorama)
    payload = next(tmp_path.rglob("result.npz"))
    payload.write_bytes(payload.read_bytes() + b"corrupt")
    with pytest.raises(PipelineCacheError, match="checksum mismatch"):
        pipeline(panorama)

    torch = pytest.importorskip("torch")
    torch_panorama = pa.EquirectangularImage(
        torch.ones((3, 8, 16), dtype=torch.float32, requires_grad=True)
    )
    torch_pipeline = pa.Pipeline(
        pa.OnSphere(S.GaussianBlur(3, 0.8)),
        memory=tmp_path / "torch-cache",
    )
    with pytest.raises(PipelineCacheError, match="disconnect autograd"):
        torch_pipeline(torch_panorama)


def test_numpy_import_path_does_not_load_torch() -> None:
    source = """
import json
import sys
import numpy as np
import panorai as pa
from panorai.ops import spherical as S
from panorai.ops.spherical import *  # noqa: F403 - import-isolation contract

before = 'torch' in sys.modules
pano = pa.EquirectangularImage(np.ones((6, 12, 3), dtype=np.float32))
result = pa.Pipeline(pa.OnSphere(S.Resize((4, 8))))(pano)
print(json.dumps({
    'before': before,
    'after': 'torch' in sys.modules,
    'shape': result.image.shape,
}))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report == {"before": False, "after": False, "shape": [4, 8, 3]}


def test_pipeline_matches_expanded_numpy_workflow_with_support_and_validity() -> None:
    height, width = 12, 24
    rows, columns = np.indices((height, width))
    rgb = np.stack(
        (
            rows / height,
            columns / width,
            np.zeros((height, width)),
        ),
        axis=-1,
    ).astype(np.float32)
    support = np.ones((height, width), dtype=bool)
    support[-2:, 4:9] = False
    valid = np.ones((height, width), dtype=bool)
    valid[2:4, 11:14] = False
    panorama = pa.EquirectangularImage(rgb, support_mask=support, valid=valid)

    def model(face):
        validity = np.isfinite(face).all(axis=-1)
        return face * np.float32(0.5), validity

    expected = (
        panorama.views("fibonacci", count=10, size=6, fov=110)
        .map(model)
        .reconstruct(blend="gaussian")
    )
    pipeline = pa.Pipeline(
        pa.SampleViews("fibonacci", count=10, size=6, fov=110),
        pa.PerView(model),
        pa.ReconstructERP(blend="gaussian"),
    )
    run = pipeline.run(panorama, retain=["sample_views", "per_view"])

    assert isinstance(run, PipelineRun)
    assert len(run["sample_views"]) == 10
    assert len(run["per_view"]) == 10
    _assert_same_panorama(run.output, expected)
    assert run.output.image.dtype == rgb.dtype
    assert run.describe()["steps"][0]["details"]["workflow"]["contract"] == (
        "geometry-v1"
    )


def test_pipeline_matches_process_views_without_changing_stable_methods() -> None:
    panorama = pa.EquirectangularImage(
        np.linspace(0.0, 1.0, 8 * 16 * 3, dtype=np.float64).reshape(8, 16, 3)
    )

    def model(face):
        return face + 0.25

    expected = panorama.process_views(
        model,
        layout="cube",
        size=5,
        blend="average",
    )
    actual = pa.Pipeline(
        pa.SampleViews("cube", size=5),
        pa.PerView(model),
        pa.ReconstructERP(blend="average"),
    )(panorama)
    _assert_same_panorama(actual, expected)


def test_pipeline_names_params_retention_and_transition_validation() -> None:
    pipeline = pa.Pipeline(
        ("blur", pa.OnSphere(S.GaussianBlur(3, 0.8))),
        pa.OnSphere(S.Resize((4, 8))),
    )
    params = pipeline.get_params()
    assert params["blur__operation"].sigma == 0.8
    assert params["on_sphere__output"] == "image"
    variant = pipeline.with_params(
        blur__operation=S.GaussianBlur(5, 1.2),
        on_sphere__operation=S.Resize((6, 12)),
    )
    assert variant.get_params()["blur__operation"].ksize == 5
    assert variant.get_params()["on_sphere__operation"].output_shape_hw == (6, 12)
    assert pipeline.get_params()["on_sphere__operation"].output_shape_hw == (4, 8)
    assert list(pipeline.named_steps) == ["blur", "on_sphere"]

    with pytest.raises(ValueError, match="incompatible pipeline transition"):
        pa.Pipeline(pa.SampleViews(), pa.OnSphere(S.Resize((4, 8))))
    with pytest.raises(KeyError, match="unknown retained"):
        pipeline.run(
            pa.EquirectangularImage(np.ones((8, 16, 3), dtype=np.float32)),
            retain=["missing"],
        )


def test_public_stage_protocol_accepts_external_and_nested_stages() -> None:
    class Add:
        input_kind = "number"
        output_kind = "number"

        def __init__(self, amount):
            self.amount = amount

        def execute(self, value):
            return pa.StageResult(
                value + self.amount,
                {"amount": self.amount, "implementation": "external"},
            )

    inner = pa.Pipeline(Add(2), Add(3))
    outer = pa.Pipeline(Add(1), inner, Add(4))
    run = outer.run(10, retain="pipeline")

    assert pa.Stage.__name__ == "Stage"
    assert outer(10) == 20
    assert run["pipeline"] == 16
    assert run.records[1].details["domain"] == "composition"
    with pytest.raises(TypeError, match="must return StageResult"):
        pa.Pipeline(
            type(
                "Bad",
                (),
                {
                    "input_kind": "any",
                    "output_kind": "any",
                    "execute": lambda self, value: value,
                },
            )()
        )(1)


def test_mixed_per_view_parallel_batches_deterministically() -> None:
    class BatchedFeatures:
        supports_batch = True

        def __init__(self):
            self.chunks = []

        def __call__(self, values):
            self.chunks.append(int(values.shape[0]))
            return values * np.float32(2.0)

    class BatchedEmbeddings:
        supports_batch = True

        def __init__(self):
            self.chunks = []

        def __call__(self, values):
            self.chunks.append(int(values.shape[0]))
            return values.mean(axis=(1, 2, 3), keepdims=False)[:, None]

    features = BatchedFeatures()
    embeddings = BatchedEmbeddings()
    panorama = pa.EquirectangularImage(
        np.linspace(0, 1, 8 * 16 * 3, dtype=np.float32).reshape(8, 16, 3)
    )
    pipeline = pa.Pipeline(
        pa.SampleViews("cube", size=5),
        pa.PerView.parallel(
            batch=4,
            features=pa.Output(features, kind="features"),
            embedding=pa.Output(embeddings, kind="embedding"),
            detections=pa.Output(
                lambda batch: [
                    {"points_xy": np.asarray(((2.0, 2.0),))}
                    for _ in range(batch.shape[0])
                ],
                kind="detections",
            ),
        ),
        pa.ReconstructERP(features="gaussian"),
    )
    bundle = pipeline(panorama)

    assert features.chunks == [4, 2]
    assert embeddings.chunks == [4, 2]
    assert tuple(bundle["erp"]._workflow_data()["features"].shape) == (8, 16, 3)
    assert isinstance(bundle["embedding"], pa.EmbeddingResult)
    assert len(bundle["embedding"]) == 6
    assert bundle["embedding"].values[0].shape == (1,)
    assert len(bundle["detections"]) == 6

    auto_features = BatchedFeatures()
    auto_bundle = pa.Pipeline(
        pa.SampleViews("cube", size=5),
        pa.PerView.parallel(
            batch="auto",
            max_batch_elements=2 * 5 * 5 * 3,
            features=pa.Output(auto_features, kind="features"),
        ),
    )(panorama)
    assert auto_features.chunks == [2, 2, 2]
    assert isinstance(auto_bundle["features"], pa.RasterField)

    with pytest.raises(TypeError, match="supports_batch"):
        pa.Pipeline(
            pa.SampleViews("cube", size=5),
            pa.PerView.parallel(
                batch="auto",
                features=pa.Output(lambda value: value, kind="features"),
            ),
        )(panorama)


def test_configured_operations_match_existing_public_functions() -> None:
    rng = np.random.default_rng(812)
    image = rng.normal(size=(8, 16, 3)).astype(np.float32)
    blur = S.GaussianBlur(ksize=3, sigma=0.75, backend="numpy")
    resize = S.Resize((6, 12))
    rotation = np.asarray(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    rotate = S.Rotate(rotation)

    np.testing.assert_array_equal(
        blur(image), S.gaussian_blur(image, 3, 0.75, backend="numpy")
    )
    np.testing.assert_array_equal(resize(image), S.resize(image, (6, 12)))
    np.testing.assert_array_equal(rotate(image), S.rotate(image, rotation))


def test_on_sphere_transforms_support_and_validity_separately() -> None:
    image = np.ones((8, 16, 3), dtype=np.float32)
    support = np.ones((8, 16), dtype=bool)
    valid = np.ones((8, 16), dtype=bool)
    support[3, 5] = False
    valid[5, 11] = False
    panorama = pa.EquirectangularImage(image, support_mask=support, valid=valid)
    result = pa.Pipeline(pa.OnSphere(S.Resize((4, 8))))(panorama)

    expected_support = (
        S.resize(support.astype(np.uint8), (4, 8), interpolation="bilinear")
        >= 1.0 - 1e-12
    )
    expected_valid = (
        S.resize(valid.astype(np.uint8), (4, 8), interpolation="bilinear")
        >= 1.0 - 1e-12
    )
    np.testing.assert_array_equal(result._workflow_support["image"], expected_support)
    np.testing.assert_array_equal(result.validity("image"), expected_valid)
    assert not np.array_equal(expected_support, expected_valid)

    blurred = pa.Pipeline(
        pa.OnSphere(S.GaussianBlur(ksize=3, sigma=1.0, backend="numpy"))
    )(panorama)
    assert np.count_nonzero(~blurred._workflow_support["image"]) > 1
    assert np.count_nonzero(~blurred.validity("image")) > 1


def test_gaussian_depth_renormalization_preserves_valid_zero_and_masks_nan() -> None:
    depth = np.zeros((7, 14), dtype=np.float64)
    depth[3, 6] = np.nan
    validity = np.ones(depth.shape, dtype=bool)
    validity[3, 6] = False
    panorama = pa.EquirectangularImage(
        np.zeros((*depth.shape, 3), dtype=np.float64)
    ).with_depth(depth, valid=validity, units="m")

    field = pa.Pipeline(
        pa.OnSphere(
            pa.Output(
                S.GaussianBlur(3, 1.0, backend="numpy"),
                kind="depth",
                units="m",
                invalid_policy="renormalize",
                min_valid_weight=0.5,
            ),
            input="depth",
            output="depth",
        )
    )(panorama)

    assert isinstance(field, pa.RasterField)
    assert field.invalid_policy == "renormalize"
    assert field.support.all()
    assert field.validity.all()
    np.testing.assert_array_equal(field.values, np.zeros_like(depth))
    assert field.grid.is_canonical_erp
    assert field.grid.transforms[-1].operation == "gaussian_blur"


def test_output_renormalization_requires_explicit_valid_weight_threshold() -> None:
    with pytest.raises(ValueError, match="min_valid_weight is required"):
        pa.Output(
            S.GaussianBlur(3, 1.0),
            kind="depth",
            units="m",
            invalid_policy="renormalize",
        )
    with pytest.raises(ValueError, match="only valid"):
        pa.Output(
            S.GaussianBlur(3, 1.0),
            kind="depth",
            units="m",
            min_valid_weight=0.5,
        )


def test_on_sphere_preserves_multimodal_fields_and_accepts_typed_outputs() -> None:
    rgb = np.ones((6, 12, 3), dtype=np.float32)
    depth = np.linspace(0.0, 1.0, 6 * 12, dtype=np.float32).reshape(6, 12)
    labels = np.arange(6 * 12, dtype=np.int16).reshape(6, 12)
    panorama = (
        pa.EquirectangularImage(rgb).with_depth(depth, units="m").with_labels(labels)
    )

    blurred = pa.Pipeline(pa.OnSphere(S.GaussianBlur(3, 0.8)))(panorama)
    np.testing.assert_array_equal(blurred.depth, panorama.depth)
    np.testing.assert_array_equal(blurred.labels, panorama.labels)

    typed = pa.Pipeline(
        pa.OnSphere(
            pa.Output(S.Resize((6, 12)), kind="depth", units="m"),
            input="depth",
            output="filtered_depth",
        )
    )(panorama)
    np.testing.assert_array_equal(typed.values, S.resize(depth, (6, 12)))
    np.testing.assert_array_equal(typed.validity, panorama.validity("depth"))
    assert typed.kind == "depth"
    assert typed.units == "m"

    nearest_labels = pa.Pipeline(
        pa.OnSphere(
            pa.Output(S.Resize((6, 12), interpolation="nearest"), kind="labels"),
            input="labels",
            output="labels",
        )
    )(panorama)
    np.testing.assert_array_equal(nearest_labels.values, labels)
    with pytest.raises(ValueError, match="labels and masks require"):
        pa.Pipeline(
            pa.OnSphere(
                pa.Output(S.GaussianBlur(3, 0.8), kind="labels"),
                input="labels",
                output="labels",
            )
        )(panorama)

    features = pa.Pipeline(
        pa.OnSphere(
            pa.Output(S.Resize((4, 8)), kind="features"),
            output="features",
        )
    )(panorama)
    assert isinstance(features, pa.RasterField)
    assert features.grid.domain == "erp"
    assert features.grid.shape_hw == (4, 8)


def test_output_declaration_enforces_raster_semantics() -> None:
    with pytest.raises(ValueError, match="requires explicit units"):
        pa.Output(lambda value: value, kind="depth").resolve("depth")
    with pytest.raises(ValueError, match="only for radial-range depth"):
        pa.Output(lambda value: value, kind="logits", units="m").resolve("logits")
    with pytest.raises(ValueError, match="validity renormalization"):
        from panorai.pipeline import RasterField, RasterGrid

        RasterField(
            name="mask",
            values=np.ones((2, 4), dtype=bool),
            kind="mask",
            support=np.ones((2, 4), dtype=bool),
            validity=np.ones((2, 4), dtype=bool),
            grid=RasterGrid(
                domain="erp",
                shape_hw=(2, 4),
                erp_shape_hw=(2, 4),
                alignment="erp-pixel-centres",
            ),
            invalid_policy="renormalize",
        )


def test_per_view_parallel_typed_fields_reconstruct_with_stable_oracles() -> None:
    height, width = 10, 20
    rows, columns = np.indices((height, width))
    image = np.stack(
        (rows / height, columns / width, (rows + columns) / (height + width)),
        axis=-1,
    ).astype(np.float32)
    source_support = np.ones((height, width), dtype=bool)
    source_support[2:4, 8:11] = False
    panorama = pa.EquirectangularImage(image, support_mask=source_support)

    def labels_model(face):
        return (
            np.argmax(face, axis=-1).astype(np.int16),
            np.isfinite(face).all(axis=-1),
        )

    def mask_model(face):
        return face[..., 0] >= face[..., 1], np.isfinite(face).all(axis=-1)

    def depth_model(face):
        valid = np.isfinite(face).all(axis=-1)
        valid[0, 0] = False
        return np.zeros(face.shape[:2], dtype=np.float32), valid

    def logits_model(face):
        return face * np.float32(2.0), np.isfinite(face).all(axis=-1)

    views = panorama.views("cube", size=6)
    expected_labels = views.map(labels_model, output="labels").reconstruct(
        modalities="labels", blend="closest"
    )
    expected_mask = views.map(mask_model, output="labels").reconstruct(
        modalities="labels", blend="closest"
    )
    expected_depth = views.map(depth_model, output="depth", units="m").reconstruct(
        modalities="depth", blend="gaussian"
    )
    expected_logits = views.map(logits_model).reconstruct(
        modalities="image", blend="gaussian"
    )

    pipeline = pa.Pipeline(
        pa.SampleViews("cube", size=6),
        pa.PerView.parallel(
            labels=labels_model,
            mask=mask_model,
            depth=pa.Output(depth_model, kind="depth", units="m"),
            logits=pa.Output(logits_model, kind="logits"),
        ),
        pa.ReconstructERP(
            labels="closest",
            mask="closest",
            depth="gaussian",
            logits="gaussian",
        ),
    )
    run = pipeline.run(
        panorama,
        retain=["sample_views", "per_view.views.depth", "reconstruct_erp.erp.depth"],
    )
    result = run.output["erp"]

    assert list(result._workflow_data()) == ["labels", "mask", "depth", "logits"]
    assert len(run["sample_views"]) == 6
    assert isinstance(run["per_view.views.depth"], pa.RasterField)
    assert run["per_view.views.depth"].grid.domain == "views"
    assert isinstance(run["reconstruct_erp.erp.depth"], pa.RasterField)
    np.testing.assert_array_equal(result.labels, expected_labels.labels)
    np.testing.assert_array_equal(result._workflow_data()["mask"], expected_mask.labels)
    np.testing.assert_array_equal(
        result.validity("mask"), expected_mask.validity("labels")
    )
    np.testing.assert_array_equal(
        result._workflow_support["mask"], expected_mask._workflow_support["labels"]
    )
    np.testing.assert_allclose(
        result.depth, expected_depth.depth, rtol=0.0, atol=0.0, equal_nan=True
    )
    np.testing.assert_array_equal(
        result.validity("depth"), expected_depth.validity("depth")
    )
    np.testing.assert_allclose(
        result._workflow_data()["logits"],
        expected_logits.image,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    assert result._workflow_metadata["labels"]["interpolation"] == "nearest"
    assert result._workflow_metadata["mask"]["interpolation"] == "nearest"
    assert result._workflow_metadata["depth"]["units"] == "m"
    assert result._workflow_metadata["depth"]["field_kind"] == "depth"
    assert np.all(result.depth[result.validity("depth")] == 0.0)


def test_per_view_parallel_branches_share_original_input_not_prior_outputs() -> None:
    image = np.arange(6 * 12 * 3, dtype=np.float32).reshape(6, 12, 3)
    panorama = pa.EquirectangularImage(image)
    views = panorama.views("cube", size=4)
    mapped = (
        pa.PerView.parallel(
            first=pa.Output(lambda face: np.zeros_like(face), kind="features"),
            second=pa.Output(lambda face: face.copy(), kind="features"),
        )
        .execute(views)
        .value["views"]
    )
    for source, output in zip(views, mapped, strict=True):
        np.testing.assert_array_equal(output._workflow_data()["first"], 0.0)
        np.testing.assert_array_equal(output._workflow_data()["second"], source.image)


def test_typed_output_without_mask_inherits_input_validity_not_pixel_value() -> None:
    image = np.zeros((8, 16, 3), dtype=np.float32)
    valid = np.ones((8, 16), dtype=bool)
    valid[3:5, 7:10] = False
    views = pa.EquirectangularImage(image, valid=valid).views("cube", size=6)
    mapped = (
        pa.PerView.parallel(mask=lambda face: np.zeros(face.shape[:2], dtype=bool))
        .execute(views)
        .value["views"]
    )
    for source, output in zip(views, mapped, strict=True):
        np.testing.assert_array_equal(output.validity("mask"), source.validity("image"))
        assert not output._workflow_data()["mask"].any()
        assert output.validity("mask").any()


def test_parallel_select_attach_and_grid_mismatch_are_explicit() -> None:
    panorama = pa.EquirectangularImage(np.ones((8, 16, 3), dtype=np.float32))
    depth_branch = pa.Pipeline(
        pa.SampleViews("cube", size=5),
        pa.PerView(
            pa.Output(
                lambda face: face[..., 0] + np.float32(2.0),
                kind="depth",
                units="m",
            ),
            output="depth",
        ),
        pa.ReconstructERP(depth="gaussian"),
        pa.Select("depth"),
    )
    branching = pa.Pipeline(
        pa.Parallel(base=None, prediction=depth_branch),
        pa.Attach(target="base", depth="prediction"),
    )
    attached = branching(panorama)
    assert attached is not panorama
    assert panorama.depth is None
    assert attached.depth.shape == (8, 16)
    assert attached._workflow_metadata["depth"]["units"] == "m"

    bundle = pa.Pipeline(pa.Parallel(base=None, prediction=depth_branch))(panorama)
    assert isinstance(bundle, pa.ResultBundle)
    assert list(bundle) == ["base", "prediction"]
    assert isinstance(bundle["prediction"], pa.RasterField)
    assert bundle.describe()["branches"][0]["execution"] == {"kind": "identity"}

    selected = pa.Pipeline(
        pa.Parallel(base=None, prediction=depth_branch),
        pa.Select("prediction"),
    )(panorama)
    assert isinstance(selected, pa.RasterField)
    assert selected.kind == "depth"

    smaller = pa.Pipeline(pa.OnSphere(S.Resize((4, 8))), pa.Select("image"))
    incompatible = pa.Pipeline(
        pa.Parallel(base=None, smaller=smaller),
        pa.Attach(target="base", enhanced="smaller"),
    )
    with pytest.raises(ValueError, match="incompatible grid"):
        incompatible(panorama)


def test_single_typed_per_view_output_and_parameter_updates() -> None:
    panorama = pa.EquirectangularImage(np.ones((6, 12, 3), dtype=np.float32))
    pipeline = pa.Pipeline(
        pa.SampleViews("cube", size=4),
        pa.PerView(
            pa.Output(lambda face: face + 3.0, kind="logits"),
            output="logits",
        ),
        pa.ReconstructERP(logits="average"),
    )
    result = pipeline(panorama)
    assert result.image is None
    assert result._workflow_data()["logits"].shape == (6, 12, 3)
    assert pipeline.get_params()["reconstruct_erp__field_blends"] == {
        "logits": "average"
    }
    variant = pipeline.with_params(reconstruct_erp__blend={"logits": "gaussian"})
    assert variant.get_params()["reconstruct_erp__blend"] == {"logits": "gaussian"}


def test_reconstruct_erp_uses_a_python_311_compatible_default_factory() -> None:
    field_blends = next(
        item
        for item in dataclasses.fields(pa.ReconstructERP)
        if item.name == "field_blends"
    )

    assert field_blends.default is dataclasses.MISSING
    assert field_blends.default_factory is not dataclasses.MISSING
    assert dict(field_blends.default_factory()) == {}


def test_torch_view_pipeline_equivalence_and_device_layout() -> None:
    torch = pytest.importorskip("torch")
    values = torch.linspace(0.0, 1.0, 3 * 8 * 16, dtype=torch.float64).reshape(3, 8, 16)
    valid = torch.ones((8, 16), dtype=torch.bool)
    valid[1:3, 5:8] = False
    panorama = pa.EquirectangularImage(values, valid=valid)

    def model(face):
        validity = torch.isfinite(face).all(dim=0)
        return face * 2.0, validity

    expected = panorama.views("cube", size=5).map(model).reconstruct(blend="average")
    actual = pa.Pipeline(
        pa.SampleViews("cube", size=5),
        pa.PerView(model),
        pa.ReconstructERP(blend="average"),
    )(panorama)
    torch.testing.assert_close(
        actual.image, expected.image, rtol=0.0, atol=0.0, equal_nan=True
    )
    assert actual.image.dtype == values.dtype
    assert actual.image.device == values.device
    assert tuple(actual.image.shape) == (3, 8, 16)
    assert torch.equal(actual.validity("image"), expected.validity("image"))
    assert torch.equal(
        actual._workflow_support["image"], expected._workflow_support["image"]
    )


def test_torch_spherical_sequential_preserves_autograd_and_tracks_grid() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn
    from panorai.ops.spherical import nn as SNN

    values = torch.randn(3, 8, 16, dtype=torch.float64, requires_grad=True)
    model = nn.Sequential(
        SNN.Conv2d(3, 4, 3, padding=1, dtype=torch.float64),
        nn.BatchNorm2d(4, dtype=torch.float64),
        nn.ReLU(),
    )
    pipeline = pa.Pipeline(pa.OnSphere(model))
    run = pipeline.run(pa.EquirectangularImage(values))
    assert tuple(run.output.image.shape) == (4, 8, 16)
    assert run.output.image.dtype == values.dtype
    loss = run.output.image.square().mean()
    loss.backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()
    assert model[0].weight.grad is not None
    grid = run.records[0].details["grid"]
    assert grid["shape_hw"] == (8, 16)
    assert grid["erp_shape_hw"] == (8, 16)
    assert grid["alignment"] == "erp-pixel-centres"
    assert grid["first_center_px_yx"] == (0.5, 0.5)
    assert grid["pixel_stride_yx"] == (1.0, 1.0)
    assert grid["longitude_periodic"] is True
    assert grid["transforms"][-1]["operation"] == "conv2d"

    reduced = pa.Pipeline(pa.OnSphere(SNN.MaxPool2d(2), output="features")).run(
        pa.EquirectangularImage(values.detach())
    )
    assert isinstance(reduced.output, pa.RasterField)
    assert tuple(reduced.output.values.shape) == (3, 4, 8)
    assert reduced.output.grid.domain == "derived"
    assert reduced.records[0].details["grid"]["alignment"] == (
        "derived-spherical-lattice"
    )


def test_strided_torch_lattice_tracks_anchor_stride_and_strict_masks() -> None:
    torch = pytest.importorskip("torch")
    from panorai.ops.spherical import nn as SNN

    values = torch.ones((1, 8, 16), dtype=torch.float64, requires_grad=True)
    support = torch.ones((8, 16), dtype=torch.bool)
    validity = torch.ones((8, 16), dtype=torch.bool)
    support[0, -1] = False
    validity[4, 7] = False
    panorama = pa.EquirectangularImage(values, support_mask=support, valid=validity)
    field = pa.Pipeline(
        pa.OnSphere(
            pa.Output(SNN.MaxPool2d(3, stride=2, padding=1), kind="features"),
            output="features",
        )
    )(panorama)

    assert field.grid.domain == "derived"
    assert field.grid.shape_hw == (4, 8)
    assert field.grid.erp_shape_hw == (8, 16)
    assert field.grid.first_center_px_yx == (0.5, 0.5)
    assert field.grid.pixel_stride_yx == (2.0, 2.0)
    assert field.grid.angular_stride_rad == pytest.approx((np.pi / 4, np.pi / 4))
    assert field.grid.longitude_periodic is True
    assert field.grid.transforms[-1].operation == "max_pool2d"
    assert tuple(field.support.shape) == (4, 8)
    assert tuple(field.validity.shape) == (4, 8)
    assert not field.support[0].all()
    assert not field.support[0, 1]
    assert not field.support[0, -2]
    assert not field.validity[2, 3]
    field.values.sum().backward()
    assert values.grad is not None


def test_max_pool_ignore_invalid_is_explicit_and_seam_aware() -> None:
    torch = pytest.importorskip("torch")
    from panorai.ops.spherical import nn as SNN

    values = torch.ones((1, 6, 12), dtype=torch.float64)
    values[0, 3, -1] = 100.0
    validity = torch.ones((6, 12), dtype=torch.bool)
    validity[3, -1] = False
    panorama = pa.EquirectangularImage(values, valid=validity)

    strict = pa.Pipeline(
        pa.OnSphere(
            pa.Output(SNN.MaxPool2d(3, stride=1, padding=1), kind="features"),
            output="features",
        )
    )(panorama)
    ignored = pa.Pipeline(
        pa.OnSphere(
            pa.Output(
                SNN.MaxPool2d(3, stride=1, padding=1, invalid_policy="ignore-invalid"),
                kind="features",
            ),
            output="features",
        )
    )(panorama)

    assert strict.values[0, 3, 0] > 1.0
    assert not strict.validity[3, 0]
    assert ignored.values[0, 3, 0] == 1.0
    assert ignored.validity[3, 0]
    assert ignored.support[3, 0]
    with pytest.raises(ValueError, match="ignore-invalid"):
        SNN.MaxPool2d(3, invalid_policy="renormalize")


def test_conv_transpose_restores_the_original_grid_identity() -> None:
    pytest.importorskip("torch")
    from torch import nn
    from panorai.ops.spherical import nn as SNN
    from panorai.pipeline import RasterGrid

    source_grid = RasterGrid(
        domain="erp",
        shape_hw=(8, 16),
        erp_shape_hw=(8, 16),
        alignment="erp-pixel-centres",
    )
    down = SNN.Conv2d(1, 1, 3, stride=2, padding=1)
    down_grid = down.transform_grid(source_grid)
    up = SNN.ConvTranspose2d(1, 1, 3, stride=2, padding=1, output_padding=1)
    restored = up.transform_grid(down_grid)

    assert down_grid.shape_hw == (4, 8)
    assert down_grid.domain == "derived"
    assert restored.shape_hw == source_grid.shape_hw
    assert restored.domain == "erp"
    assert restored.is_canonical_erp
    assert [item.operation for item in restored.transforms] == [
        "conv2d",
        "conv_transpose2d",
    ]
    model = nn.Sequential(down, up)
    report = SNN.validate(model)
    assert len(report.layers) == 2


def test_torch_typed_view_field_preserves_autograd_support_and_device() -> None:
    torch = pytest.importorskip("torch")
    source = torch.linspace(
        0.0, 1.0, 3 * 8 * 16, dtype=torch.float64, requires_grad=True
    )
    values = source.reshape(3, 8, 16)
    panorama = pa.EquirectangularImage(values)

    pipeline = pa.Pipeline(
        pa.SampleViews("cube", size=5),
        pa.PerView.parallel(
            logits=pa.Output(lambda face: face.square(), kind="logits"),
            mask=lambda face: face[0] > 0.5,
        ),
        pa.ReconstructERP(logits="gaussian", mask="closest"),
    )
    result = pipeline(panorama)["erp"]
    logits = result._workflow_data()["logits"]
    mask = result._workflow_data()["mask"]
    assert logits.dtype == values.dtype
    assert logits.device == values.device
    assert mask.dtype == torch.bool
    assert mask.device == values.device
    assert torch.equal(result.validity("mask"), result._workflow_support["mask"])
    logits[result.validity("logits").unsqueeze(0).expand_as(logits)].mean().backward()
    assert source.grad is not None
    assert torch.isfinite(source.grad).all()


def test_torch_per_view_batch_preserves_order_device_dtype_and_autograd() -> None:
    torch = pytest.importorskip("torch")

    class BatchedSquare:
        supports_batch = True

        def __call__(self, value):
            return value.square()

    source = torch.linspace(
        0.0,
        1.0,
        3 * 8 * 16,
        dtype=torch.float64,
        requires_grad=True,
    )
    panorama = pa.EquirectangularImage(source.reshape(3, 8, 16))
    bundle = pa.Pipeline(
        pa.SampleViews("cube", size=5),
        pa.PerView.parallel(
            batch=2,
            logits=pa.Output(BatchedSquare(), kind="logits"),
        ),
        pa.ReconstructERP(logits="gaussian"),
    )(panorama)
    logits = bundle["erp"]._workflow_data()["logits"]

    assert logits.dtype == source.dtype
    assert logits.device == source.device
    assert tuple(logits.shape) == (3, 8, 16)
    logits[
        bundle["erp"].validity("logits").unsqueeze(0).expand_as(logits)
    ].mean().backward()
    assert source.grad is not None
    assert torch.isfinite(source.grad).all()


def test_torch_planar_spatial_layer_is_rejected_before_forward() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn
    from panorai.ops.spherical import nn as SNN

    class TrackingConv(nn.Conv2d):
        called = False

        def forward(self, value):
            self.called = True
            return super().forward(value)

    layer = TrackingConv(3, 4, 3, padding=1)
    panorama = pa.EquirectangularImage(torch.ones(3, 8, 16))
    with pytest.raises(SNN.PlanarSpatialOperationError, match="SampleViews/PerView"):
        pa.Pipeline(pa.OnSphere(layer))(panorama)
    assert layer.called is False

    pointwise = nn.Conv2d(3, 4, 1)
    result = pa.Pipeline(pa.OnSphere(pointwise))(panorama)
    assert tuple(result.image.shape) == (4, 8, 16)

    class HiddenSpatialCall(nn.Module):
        def __init__(self):
            super().__init__()
            self.activation = nn.ReLU()

        def forward(self, value):
            return nn.functional.interpolate(self.activation(value), scale_factor=0.5)

    with pytest.raises(SNN.PlanarSpatialOperationError, match="custom forward"):
        pa.Pipeline(pa.OnSphere(HiddenSpatialCall()))(panorama)

    partial_validity = torch.ones((8, 16), dtype=torch.bool)
    partial_validity[2, 3] = False
    masked = pa.EquirectangularImage(torch.ones(3, 8, 16), valid=partial_validity)
    propagated = pa.Pipeline(pa.OnSphere(SNN.Conv2d(3, 4, 3, padding=1)))(masked)
    assert propagated._workflow_support["image"].all()
    assert not propagated.validity("image")[2, 3]
    assert np.count_nonzero(~propagated.validity("image").cpu().numpy()) > 1


def test_torch_direct_constructor_adapter_and_port_report() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn
    from panorai.ops.spherical import nn as SNN

    planar = nn.Conv2d(3, 5, 3, padding=1)
    adapted = SNN.Conv2d.from_planar(planar)
    assert adapted.weight is planar.weight
    report = SNN.validate(adapted)
    assert report.layers[0].role == "spherical-spatial"

    source = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU())
    converted, port_report = SNN.from_planar(source)
    assert isinstance(converted[0], SNN.SphericalConv2d)
    assert isinstance(source[0], nn.Conv2d)
    assert len(port_report.layers) == 1
    output = converted(torch.ones(1, 3, 6, 12))
    assert tuple(output.shape) == (1, 4, 6, 12)


def _one_peak_cam(_rgb: np.ndarray) -> np.ndarray:
    scores = np.zeros((1, 8, 16), dtype=np.float32)
    scores[0, 4, 8] = 1.0
    return scores


def test_cam_region_proposals_are_seam_aware_search_regions() -> None:
    panorama = pa.EquirectangularImage(np.zeros((16, 32, 3), dtype=np.uint8))

    def seam_cam(_rgb):
        scores = np.zeros((1, 8, 16), dtype=np.float32)
        scores[0, 4, 0] = 1.0
        scores[0, 4, 15] = 0.9
        return scores

    pipeline = pa.Pipeline(
        pa.SphericalCAM(seam_cam, concept_ids=("pipe",)),
        pa.CAMRegionProposals(
            maximum_per_concept=4,
            minimum_separation_degrees=30.0,
            relative_threshold=0.5,
            fov_degrees=60.0,
            face_size=32,
        ),
    )
    proposals = pipeline(panorama)

    assert len(proposals.proposals) == 1
    proposal = proposals.proposals[0]
    assert proposal.concept_id == "pipe"
    assert proposal.spec.hfov_deg == pytest.approx(60.0)
    assert proposal.spec.output_shape_hw == (32, 32)
    assert proposal.seed.source == "semantic"
    assert proposals.parameters["minimum_separation_degrees"] == 30.0


def test_spherical_cam_array_output_requires_explicit_concepts() -> None:
    panorama = pa.EquirectangularImage(np.zeros((8, 16, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="concept_ids"):
        pa.Pipeline(pa.SphericalCAM(_one_peak_cam))(panorama)


def test_spherical_cam_discards_scores_outside_explicit_input_support() -> None:
    support = np.ones((8, 16), dtype=bool)
    support[:2, :3] = False
    panorama = pa.EquirectangularImage(
        np.zeros((8, 16, 3), dtype=np.uint8), support_mask=support
    )

    def unmasked_cam(_rgb: np.ndarray) -> np.ndarray:
        return np.ones((1, 4, 8), dtype=np.float32)

    frame = pa.Pipeline(pa.SphericalCAM(unmasked_cam, concept_ids=("pipe",)))(panorama)

    expected_support = np.ones((4, 8), dtype=bool)
    expected_support[0, 0] = False
    np.testing.assert_array_equal(frame.evidence.support, expected_support)
    assert np.all(frame.evidence.scores[:, ~expected_support] == 0)
    assert np.all(frame.evidence.scores[:, expected_support] == 1)


@pytest.mark.parametrize("result_kind", ("mapping", "tuple", "typed"))
def test_spherical_cam_never_expands_input_support(result_kind: str) -> None:
    from panorai.experimental.deep_learning.segmentation import DenseSemanticEvidence

    support = np.ones((4, 8), dtype=bool)
    support[0, :2] = False
    panorama = pa.EquirectangularImage(
        np.zeros((4, 8, 3), dtype=np.uint8), support_mask=support
    )
    scores = np.ones((1, 4, 8), dtype=np.float32)
    declared_support = np.ones((4, 8), dtype=bool)

    def cam(_rgb: np.ndarray):
        if result_kind == "mapping":
            return {
                "concept_ids": ("pipe",),
                "scores": scores,
                "support": declared_support,
            }
        if result_kind == "tuple":
            return scores, declared_support
        return DenseSemanticEvidence(("pipe",), scores, declared_support)

    frame = pa.Pipeline(pa.SphericalCAM(cam, concept_ids=("pipe",)))(panorama)

    np.testing.assert_array_equal(frame.evidence.support, support)
    assert np.all(frame.evidence.scores[:, ~support] == 0)
