from __future__ import annotations

from dataclasses import replace
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from panorai.estimators import RelativePoseOptions, estimate_relative_pose
from panorai.features import (
    FaceSetSpec,
    FeatureProvenance,
    MatchProvenance,
    SphericalFeature,
    SphericalFeatureMatches,
    SphericalFeatureSet,
)
from panorai.geometry import GnomonicSpec, rays_to_erp_pixels
from panorai.object_localization import (
    ObjectLocalizationConfig,
    ObjectLocalizationPipeline,
    PairObjectLocalizationInput,
    SemanticQuery,
    SemanticRegionObservation,
    object_hypothesis_id,
    semantic_region_from_map,
)

ROOT = Path(__file__).resolve().parents[1]


def _rotation_exp(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    x, y, z = vector
    skew = np.asarray(((0, -z, y), (z, 0, -x), (-y, x, 0)), dtype=np.float64)
    return (
        np.eye(3)
        + math.sin(angle) / angle * skew
        + (1.0 - math.cos(angle)) / angle**2 * (skew @ skew)
    )


def _feature_set(view_id: str, bearings: np.ndarray) -> SphericalFeatureSet:
    pixels = rays_to_erp_pixels(bearings, (128, 256)).pixels_xy
    spec = GnomonicSpec(output_shape_hw=(32, 32))
    features = [
        SphericalFeature(
            feature_id=f"{view_id}:{index}",
            panorama_id=view_id,
            face_id="synthetic",
            pixel_xy=np.asarray((16.0, 16.0)),
            source_erp_xy=np.asarray(pixel),
            bearing_xyz=np.asarray(bearing),
            response=1.0,
            scale=1.0,
            angle_deg=0.0,
            octave=0,
            descriptor_index=index,
            valid=True,
            projection_spec=spec,
            provenance=FeatureProvenance(
                interface="test-known-se3/v1",
                source_panorama_checksum=view_id,
                projection_backend="analytic",
                projection_backend_version="1",
                face_id="synthetic",
            ),
        )
        for index, (bearing, pixel) in enumerate(zip(bearings, pixels, strict=True))
    ]
    return SphericalFeatureSet(
        panorama_id=view_id,
        features=features,
        descriptors=np.eye(len(features), dtype=np.float32),
        descriptor_type="synthetic-float32",
        descriptor_metric="l2",
        extractor_name="analytic",
        extractor_config={},
        backend_name="analytic",
        backend_version="1",
        face_set_spec=FaceSetSpec(sampler="custom", shape_hw=(32, 32)),
        projection_backend="analytic",
        projection_backend_version="1",
        panorama_checksum=view_id,
    )


def _region(
    view_id: str,
    region_id: str,
    class_id: int,
    class_name: str,
    indices: np.ndarray,
    bearings: np.ndarray,
) -> SemanticRegionObservation:
    centroid = np.sum(bearings[indices], axis=0)
    centroid /= np.linalg.norm(centroid)
    return SemanticRegionObservation(
        region_id=region_id,
        view_id=view_id,
        class_id=class_id,
        class_name=class_name,
        semantic_score=0.9,
        feature_indices=indices,
        membership_weights=np.ones(len(indices)),
        centroid_bearing=centroid,
    )


def _scene() -> tuple[PairObjectLocalizationInput, np.ndarray]:
    rng = np.random.default_rng(20261009)
    object_centers = np.asarray(((-1.2, 0.2, 5.5), (1.3, -0.3, 7.0)))
    points = np.concatenate(
        [center + rng.normal(scale=0.12, size=(12, 3)) for center in object_centers]
    )
    rotation = _rotation_exp(np.asarray((0.03, -0.08, 0.02)))
    translation = np.asarray((0.9, 0.15, 0.25), dtype=np.float64)
    translation /= np.linalg.norm(translation)
    points_b = points @ rotation.T + translation
    bearings_a = points / np.linalg.norm(points, axis=1, keepdims=True)
    bearings_b = points_b / np.linalg.norm(points_b, axis=1, keepdims=True)
    features_a = _feature_set("view-a", bearings_a)
    features_b = _feature_set("view-b", bearings_b)
    count = len(points)
    matches = SphericalFeatureMatches(
        panorama_id_a="view-a",
        panorama_id_b="view-b",
        feature_indices_a=np.arange(count),
        feature_indices_b=np.arange(count),
        bearings_a=bearings_a,
        bearings_b=bearings_b,
        descriptor_distances=np.zeros(count),
        ratio_scores=np.zeros(count),
        mutual=np.ones(count, dtype=bool),
        valid=np.ones(count, dtype=bool),
        matcher_name="analytic",
        matcher_config={},
        backend_name="analytic",
        backend_version="1",
        provenance=MatchProvenance(
            interface="test-known-se3/v1",
            source_checksums=("view-a", "view-b"),
            face_pairs=(("synthetic", "synthetic"),),
            face_pair_groups=((("synthetic", "synthetic"),),),
            deduplicated=True,
        ),
        keypoint_responses=np.ones((count, 2)),
        face_ids_a=np.full(count, "synthetic", dtype=object),
        face_ids_b=np.full(count, "synthetic", dtype=object),
    )
    pose = estimate_relative_pose(
        bearings_a,
        bearings_b,
        options=RelativePoseOptions(
            max_angular_error_deg=0.1,
            min_inlier_ratio=0.2,
            min_num_trials=8,
            max_num_trials=80,
            min_inliers=10,
            local_optimization_steps=2,
            minimal_solver_starts=16,
            random_seed=19,
        ),
    )
    assert pose is not None
    indices_0 = np.arange(0, 12)
    indices_1 = np.arange(12, 24)
    return (
        PairObjectLocalizationInput(
            view_id_a="view-a",
            view_id_b="view-b",
            query=SemanticQuery(text="find chairs and tables", class_ids=(7, 11)),
            regions_a=(
                _region("view-a", "chair-a", 7, "chair", indices_0, bearings_a),
                _region("view-a", "table-a", 11, "table", indices_1, bearings_a),
            ),
            regions_b=(
                _region("view-b", "chair-b", 7, "chair", indices_0, bearings_b),
                _region("view-b", "table-b", 11, "table", indices_1, bearings_b),
            ),
            features_a=features_a,
            features_b=features_b,
            matches=matches,
            pose=pose,
            translation_scale=1.0,
        ),
        object_centers,
    )


def test_pipeline_assigns_one_id_per_object_and_localizes_known_scene() -> None:
    evidence, expected_centers = _scene()
    pipeline = ObjectLocalizationPipeline(
        ObjectLocalizationConfig(require_accepted_pose=False)
    )

    first = pipeline.estimate(evidence)
    second = pipeline.estimate(evidence)

    assert len(first.hypotheses) == 2
    assert [item.hypothesis_id for item in first.hypotheses] == [
        item.hypothesis_id for item in second.hypotheses
    ]
    assert len({item.hypothesis_id for item in first.hypotheses}) == 2
    assert {item.class_name for item in first.hypotheses} == {"chair", "table"}
    by_class = {item.class_name: item for item in first.hypotheses}
    assert by_class["chair"].region_ids == ("chair-a", "chair-b")
    assert by_class["table"].region_ids == ("table-a", "table-b")
    for hypothesis, expected in zip(
        (by_class["chair"], by_class["table"]), expected_centers, strict=True
    ):
        assert hypothesis.identity_state == "confirmed"
        assert hypothesis.location.state == "localized"
        assert hypothesis.location.mode == "metric_3d"
        assert hypothesis.location.units == "m"
        assert hypothesis.location.triangulated_count == 12
        assert np.linalg.norm(hypothesis.location.position_xyz - expected) < 0.25


def test_object_id_is_independent_of_pair_direction() -> None:
    forward = object_hypothesis_id("a", "region-a", "b", "region-b")
    reverse = object_hypothesis_id("b", "region-b", "a", "region-a")
    assert forward == reverse
    assert object_hypothesis_id("a:b", "c", "d", "e") != object_hypothesis_id(
        "a", "b:c", "d", "e"
    )


def test_text_query_limits_the_returned_classes() -> None:
    evidence, _ = _scene()
    chair_query = SemanticQuery(text="find the chair", class_ids=(7,))

    result = ObjectLocalizationPipeline(
        ObjectLocalizationConfig(require_accepted_pose=False)
    ).estimate(replace(evidence, query=chair_query, translation_scale=None))

    assert result.query == chair_query
    assert [item.class_name for item in result.hypotheses] == ["chair"]
    assert result.hypotheses[0].location.mode == "scale_free_3d"
    assert result.hypotheses[0].location.units == "normalized_baseline"


def test_ambiguous_regions_are_not_promoted_to_object_hypotheses() -> None:
    evidence, _ = _scene()
    original = evidence.regions_b[0]
    duplicate = replace(original, region_id="chair-b-duplicate")
    ambiguous_evidence = replace(
        evidence,
        regions_b=(original, duplicate, evidence.regions_b[1]),
    )

    result = ObjectLocalizationPipeline(
        ObjectLocalizationConfig(require_accepted_pose=False, ambiguity_margin=0.01)
    ).estimate(ambiguous_evidence)

    chair_states = {
        item.state for item in result.associations if item.class_name == "chair"
    }
    assert "ambiguous" in chair_states
    assert all(item.class_name != "chair" for item in result.hypotheses)
    assert {item.class_name for item in result.hypotheses} == {"table"}


def test_rejected_pose_fails_closed() -> None:
    evidence, _ = _scene()
    rejected_quality = replace(
        evidence.pose.quality_report,
        accepted=False,
        rejection_reasons=("test-rejection",),
    )
    rejected = replace(evidence.pose, quality_report=rejected_quality)

    result = ObjectLocalizationPipeline().estimate(replace(evidence, pose=rejected))

    assert result.hypotheses == ()
    assert result.failure_reasons == ("relative-pose-not-accepted",)
    assert all(item.state == "rejected" for item in result.associations)


def test_low_parallax_keeps_a_bearing_only_location() -> None:
    evidence, _ = _scene()
    result = ObjectLocalizationPipeline(
        ObjectLocalizationConfig(
            require_accepted_pose=False,
            min_triangulation_angle_deg=89.0,
        )
    ).estimate(evidence)

    assert len(result.hypotheses) == 2
    assert all(item.location.state == "view_only" for item in result.hypotheses)
    assert all(item.location.mode == "bearing_only" for item in result.hypotheses)
    assert all(item.location.position_xyz is None for item in result.hypotheses)
    assert all(item.location.bearing_xyz is not None for item in result.hypotheses)


def test_semantic_map_indexes_existing_features_without_copying_descriptors() -> None:
    evidence, _ = _scene()
    score_map = np.ones((16, 32), dtype=np.float32)
    descriptors_before = evidence.features_a.descriptors.copy()

    region = semantic_region_from_map(
        score_map,
        erp_shape_hw=(128, 256),
        features=evidence.features_a,
        region_id="whole-view",
        class_id=1,
        class_name="object",
        threshold=0.5,
        source_id="cam:whole-view",
    )

    assert np.array_equal(region.feature_indices, np.arange(24))
    assert np.array_equal(region.membership_weights, np.ones(24))
    assert region.source_id == "cam:whole-view"
    assert np.array_equal(evidence.features_a.descriptors, descriptors_before)


def test_semantic_map_wraps_fractional_feature_coordinates_across_erp_seam() -> None:
    evidence, _ = _scene()
    original = evidence.features_a.features[0]
    wrapped_feature = replace(
        original,
        source_erp_xy=np.asarray((255.8, 64.0)),
        descriptor_index=0,
    )
    seam_features = replace(
        evidence.features_a,
        features=[wrapped_feature],
        descriptors=evidence.features_a.descriptors[:1].copy(),
    )
    score_map = np.zeros((16, 32), dtype=np.float32)
    score_map[8, 0] = 1.0

    region = semantic_region_from_map(
        score_map,
        erp_shape_hw=(128, 256),
        features=seam_features,
        region_id="seam-object",
        class_id=1,
        class_name="object",
    )

    assert np.array_equal(region.feature_indices, np.asarray((0,)))
    assert np.array_equal(region.membership_weights, np.asarray((1.0,)))


def test_import_does_not_load_optional_vision_or_tensor_backends() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import panorai.object_localization; "
            "assert 'torch' not in sys.modules; assert 'cv2' not in sys.modules",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
