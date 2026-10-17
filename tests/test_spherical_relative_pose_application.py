from __future__ import annotations

from hashlib import sha256
import json
import math

import numpy as np
import pytest

from panorai import EquirectangularImage
from panorai.applications import (
    Application,
    ApplicationConfigurationError,
    AssessRelativePoseCapture,
    RelativePoseCaptureError,
    RelativePoseCaptureInput,
    RelativePoseCaptureMetadata,
    RelativePoseCapturePolicy,
    RelativePoseProfile,
    SphericalRelativePose,
)
from panorai.estimators import (
    RelativePoseAcceptancePolicy,
    RelativePoseOptions,
    RelativePoseResult,
    SphericalRelativePoseEstimator,
    UniformFivePointSampler,
)
from panorai.features import (
    ExtractSphericalFeaturePair,
    SphericalBearingCorrespondences,
    SphericalFeaturePipeline,
    SphericalImagePair,
)
from panorai.pipeline import Parallel, Pipeline
from panorai.pipeline._cache import (
    _CACHE_MISS,
    cache_identity,
    load_cached_result,
    store_cached_result,
)
from panorai.pipeline._cache_typed import decode_typed, encode_typed


def _rotation_exp(vector: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(vector)
    x, y, z = vector
    skew = np.asarray(((0, -z, y), (z, 0, -x), (-y, x, 0)), dtype=np.float64)
    return (
        np.eye(3)
        + math.sin(angle) / angle * skew
        + (1.0 - math.cos(angle)) / angle**2 * (skew @ skew)
    )


def _known_pose_correspondences() -> tuple[
    SphericalBearingCorrespondences, np.ndarray, np.ndarray
]:
    """Independent central-camera oracle from known 3D points and SE(3)."""

    rng = np.random.default_rng(20261016)
    rotation = _rotation_exp(np.asarray((0.08, -0.13, 0.04)))
    translation = np.asarray((0.8, 0.1, 0.25), dtype=np.float64)
    translation /= np.linalg.norm(translation)
    points_reference = rng.uniform((-3.0, -2.0, 3.0), (3.0, 2.0, 12.0), (28, 3))
    points_source = points_reference @ rotation.T + translation
    bearings_reference = points_reference / np.linalg.norm(
        points_reference, axis=1, keepdims=True
    )
    bearings_source = points_source / np.linalg.norm(
        points_source, axis=1, keepdims=True
    )
    count = len(points_reference)
    return (
        SphericalBearingCorrespondences(
            bearings_reference,
            bearings_source,
            np.ones(count, dtype=np.float32),
            np.ones(count, dtype=bool),
        ),
        rotation,
        translation,
    )


def _fast_options() -> RelativePoseOptions:
    return RelativePoseOptions(
        max_angular_error_deg=0.15,
        min_num_trials=8,
        max_num_trials=24,
        min_inliers=8,
        local_optimization_steps=1,
        minimal_solver_starts=8,
        random_seed=7,
        stability_trials=0,
        model_competition_trials=8,
        nonminimal_refit_max_steps=0,
    )


def _permissive_policy() -> RelativePoseAcceptancePolicy:
    return RelativePoseAcceptancePolicy(
        min_inliers=5,
        min_inlier_ratio=0.0,
        min_occupied_cells=1,
        min_coverage_entropy=0.0,
        max_median_residual_deg=1.0,
        min_median_parallax_deg=0.0,
        min_cheirality_ratio=0.0,
        min_translation_orientation_margin=0.0,
        require_essential_preferred=False,
        require_stability=False,
    )


def _textured_panorama(height: int = 96, width: int = 192) -> np.ndarray:
    y, x = np.indices((height, width))
    rng = np.random.default_rng(20261016)
    noise = rng.integers(0, 80, size=(height, width), dtype=np.uint8)
    checker = (((x // 7) + (y // 5)) % 2) * 150
    base = np.clip(noise.astype(np.int16) + checker, 0, 255).astype(np.uint8)
    return np.stack(
        (
            base,
            np.roll(base, 3, axis=1),
            np.bitwise_xor(base, ((x * 13 + y * 7) % 256).astype(np.uint8)),
        ),
        axis=-1,
    )


def test_default_application_is_the_frozen_reference_composition() -> None:
    application = SphericalRelativePose()
    pipeline = application.as_pipeline()
    description = application.describe()

    assert application.resolved_frontend_config == (
        SphericalFeaturePipeline.for_relative_pose().config
    )
    assert application == SphericalRelativePose.reference()
    assert description["profile_mode"] == "reference"
    assert description["profile_id"] == (
        "panorai-relative-pose-reference-v1-experimental"
    )
    assert description["frontend_mode"] == "reference"
    assert description["translation"] == "unit-direction-only"
    assert description["options"]["hypothesis_ranking"] == "msac-first"
    assert description["options"]["nonminimal_refit_max_steps"] == 100
    assert description["capture_contract"] == "not assessed"
    assert list(pipeline.named_steps) == [
        "extract_features",
        "match_features",
        "bearing_correspondences",
        "estimate_pose",
    ]
    assert isinstance(
        pipeline.named_steps["extract_features"], ExtractSphericalFeaturePair
    )
    assert pipeline.named_steps["extract_features"].frontend_config is None
    assert pipeline.input_kind == "spherical_image_pair"
    assert pipeline.output_kind == "relative_pose"
    assert application.to_config()["parameters"]["frontend"] == {
        "mode": "reference",
        "config": None,
    }


def test_legacy_gold_standard_frontend_config_remains_readable() -> None:
    config = SphericalRelativePose.reference().to_config()
    config["parameters"]["frontend"]["mode"] = "gold-standard"

    restored = SphericalRelativePose.from_config(config)

    assert restored == SphericalRelativePose.reference()
    assert restored.to_config()["parameters"]["frontend"]["mode"] == "reference"


def test_reference_profile_names_the_complete_computation_not_only_frontend() -> None:
    profile = RelativePoseProfile.evidence_backed_v1()
    application = SphericalRelativePose.from_profile(profile)

    assert application.resolved_profile == profile
    assert profile.resolved_frontend_config == (
        SphericalFeaturePipeline.for_relative_pose().config
    )
    assert profile.options.hypothesis_ranking == "msac-first"
    assert profile.options.nonminimal_refit_max_steps == 100
    assert profile.options.pose_refinement_method == "joint"
    assert profile.options.translation_orientation_method == "parallax-weighted"
    assert profile.sampler.name == "spatially-weighted-five-point-v1"
    assert profile.to_dict()["frontend_mode"] == "reference"


def test_versioned_reference_and_capture_profiles_have_canonical_payloads() -> None:
    def digest(value) -> str:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return sha256(payload).hexdigest()

    assert digest(RelativePoseProfile.evidence_backed_v1().to_dict()) == (
        "2e92782d2b8a5c9ed282eeb033bdd31ef20c36ff65f5edc641037c00ce46bca6"
    )
    assert digest(RelativePoseCapturePolicy.scanner_assisted_reference().to_dict()) == (
        "10118092a156938403255f1287e10a60ebbdc30bae248951928910ac20b50ea2"
    )
    assert digest(RelativePoseCapturePolicy.hilti_fisheye_development().to_dict()) == (
        "d1029bf4cc640da05ff78a6dfc72fcf578529855cafb8432cb329aea3a1267e2"
    )


def test_named_profile_identity_survives_config_and_invalidates_on_change() -> None:
    reference = RelativePoseProfile.evidence_backed_v1()
    named = RelativePoseProfile(
        profile_id="organization-relative-pose-v1",
        frontend_config=reference.frontend_config,
        options=reference.options,
        sampler=reference.sampler,
        quality_policy=reference.quality_policy,
        evidence_scope="organization-owned validation corpus v1",
    )
    application = SphericalRelativePose.from_profile(named)

    assert application.resolved_profile == named
    assert application.describe()["profile_mode"] == "named"
    assert application.describe()["profile_id"] == named.profile_id
    assert application.to_config()["parameters"]["profile"] == {
        "profile_id": named.profile_id,
        "evidence_scope": named.evidence_scope,
    }

    restored = SphericalRelativePose.from_config(application.to_config())
    assert restored.resolved_profile == named
    assert restored.describe()["profile_id"] == named.profile_id

    capture_variant = application.with_params(capture_mode="advisory")
    assert capture_variant.resolved_profile == named
    no_op = application.with_params(
        options__max_num_trials=application.options.max_num_trials
    )
    assert no_op.resolved_profile == named
    changed = application.with_params(options__max_num_trials=999)
    assert changed.resolved_profile is None
    assert changed.to_config()["parameters"]["profile"] is None


def test_reserved_reference_profile_id_rejects_tampered_config() -> None:
    config = SphericalRelativePose.reference().to_config()
    config["parameters"]["options"]["max_num_trials"] = 999
    config["parameters"]["profile"]["evidence_scope"] = "tampered"

    with pytest.raises(
        ApplicationConfigurationError,
        match="reserved profile_id.*canonical v1 payload",
    ):
        SphericalRelativePose.from_config(config)


def test_reserved_capture_policy_id_rejects_tampered_config() -> None:
    application = SphericalRelativePose.reference(
        capture_policy=RelativePoseCapturePolicy.scanner_assisted_reference()
    )
    config = application.to_config()
    config["parameters"]["capture_policy"]["minimum_registered_overlap"] = 0.01

    with pytest.raises(
        ApplicationConfigurationError,
        match="reserved profile_id.*canonical payload",
    ):
        SphericalRelativePose.from_config(config)


def test_reference_frontend_stage_records_use_reference_terminology() -> None:
    image = EquirectangularImage(_textured_panorama(24, 48))
    run = Pipeline(ExtractSphericalFeaturePair()).run(
        SphericalImagePair(image, image, "reference", "source")
    )

    assert run.records[0].details["frontend_mode"] == "reference"


def test_scanner_capture_policy_has_exact_overlap_boundaries() -> None:
    policy = RelativePoseCapturePolicy.scanner_assisted_reference()

    missing = policy.assess(RelativePoseCaptureMetadata())
    below = policy.assess(
        RelativePoseCaptureMetadata(registered_cloud_overlap_min=0.499999)
    )
    boundary = policy.assess(
        RelativePoseCaptureMetadata(registered_cloud_overlap_min=0.50)
    )
    preferred = policy.assess(
        RelativePoseCaptureMetadata(registered_cloud_overlap_min=0.70)
    )

    assert missing.decision == "insufficient-metadata"
    assert missing.reasons == ("registered-cloud-overlap-unavailable",)
    assert below.decision == "ineligible"
    assert below.reasons == ("registered-cloud-overlap-below-minimum",)
    assert boundary.eligible
    assert not boundary.preferred
    assert boundary.warnings == ("registered-cloud-overlap-below-preferred",)
    assert preferred.eligible and preferred.preferred
    assert policy.metric_baseline_interval_m is None


def test_hilti_development_policy_keeps_target_and_maximum_distinct() -> None:
    policy = RelativePoseCapturePolicy.hilti_fisheye_development()
    target = RelativePoseCaptureMetadata(
        capture_rate_hz=5.0,
        step_translation_m=0.4,
        step_rotation_deg=15.0,
    )
    warning = RelativePoseCaptureMetadata(
        capture_rate_hz=5.0,
        step_translation_m=0.5,
        step_rotation_deg=20.0,
    )
    failure = RelativePoseCaptureMetadata(
        capture_rate_hz=4.9,
        step_translation_m=0.61,
        step_rotation_deg=31.0,
    )

    assert policy.assess(target).preferred
    assert policy.assess(warning).decision == "eligible"
    assert policy.assess(warning).warnings == (
        "step-translation-above-target",
        "step-rotation-above-target",
    )
    assert policy.assess(failure).reasons == (
        "capture-rate-below-minimum",
        "step-translation-above-maximum",
        "step-rotation-above-maximum",
    )


def test_capture_stage_enforces_or_reports_without_estimating() -> None:
    image = EquirectangularImage(_textured_panorama(24, 48))
    pair = SphericalImagePair(image, image, "reference", "source")
    policy = RelativePoseCapturePolicy.scanner_assisted_reference()
    rejected = RelativePoseCaptureInput(
        pair,
        RelativePoseCaptureMetadata(registered_cloud_overlap_min=0.49),
    )

    with pytest.raises(RelativePoseCaptureError) as error:
        Pipeline(AssessRelativePoseCapture(policy))(rejected)
    assert error.value.assessment.decision == "ineligible"

    run = Pipeline(AssessRelativePoseCapture(policy, mode="advisory")).run(rejected)
    assert run.output is pair
    assert run.records[0].details["assessment"]["decision"] == "ineligible"


def test_capture_aware_application_and_pipeline_configs_round_trip() -> None:
    policy = RelativePoseCapturePolicy.scanner_assisted_reference()
    application = SphericalRelativePose.reference(capture_policy=policy)
    pipeline = application.as_pipeline()

    assert list(pipeline.named_steps)[0] == "assess_capture"
    assert pipeline.input_kind == "relative_pose_capture_input"
    assert application.describe()["capture_contract"] == "pre-estimation"
    assert SphericalRelativePose.from_config(application.to_config()) == application
    assert (
        Pipeline.from_config(pipeline.to_config()).to_config() == pipeline.to_config()
    )

    assessment = application.assess_capture(
        RelativePoseCaptureMetadata(registered_cloud_overlap_min=0.8)
    )
    assert assessment.eligible


def test_capture_metadata_cannot_be_silently_ignored() -> None:
    image = _textured_panorama(24, 48)
    metadata = RelativePoseCaptureMetadata(registered_cloud_overlap_min=0.8)

    with pytest.raises(ValueError, match="requires an explicit capture_policy"):
        SphericalRelativePose().run(image, image, capture_metadata=metadata)

    with pytest.raises(ValueError, match="capture_metadata is required"):
        SphericalRelativePose.reference(
            capture_policy=RelativePoseCapturePolicy.scanner_assisted_reference()
        ).run(image, image)


def test_custom_frontend_and_safe_configs_are_explicit_and_round_trip(tmp_path) -> None:
    custom = SphericalFeaturePipeline.from_preset(
        "orb-hamming",
        face_sampler="cube",
        face_shape_hw=96,
        face_fov_deg=100.0,
        edge_margin_px=2,
        max_features=300,
    ).config
    application = SphericalRelativePose(
        frontend_config=custom,
        options=_fast_options(),
        quality_policy=_permissive_policy(),
    )

    assert application.describe()["frontend_mode"] == "custom"
    assert application.resolved_frontend_config == custom
    assert type(application.with_params(options__max_num_trials=32)) is (
        SphericalRelativePose
    )
    assert (
        application.with_params(options__max_num_trials=32).options.max_num_trials == 32
    )
    assert SphericalRelativePose.from_config(application.to_config()) == application
    assert Pipeline.from_config(application.as_pipeline().to_config()).to_config() == (
        application.as_pipeline().to_config()
    )
    destination = application.save(tmp_path / "relative-pose.json")
    assert Application.load(destination) == application
    json.loads(destination.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("stability",), "bogus", "stability must be 'stable'"),
        (
            ("face_set", "effective_fov_deg"),
            [1.0, 1.0],
            "effective_fov_deg does not match",
        ),
    ],
)
def test_custom_frontend_config_rejects_tampered_derived_fields(
    path, value, message
) -> None:
    custom = SphericalFeaturePipeline.from_preset(
        "orb-hamming",
        face_sampler="cube",
        face_shape_hw=96,
        face_fov_deg=100.0,
        edge_margin_px=2,
        max_features=300,
    ).config
    payload = SphericalRelativePose(frontend_config=custom).to_config()
    frontend = payload["parameters"]["frontend"]["config"]
    target = frontend
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ApplicationConfigurationError, match=message):
        SphericalRelativePose.from_config(payload)


def test_unregistered_sampler_executes_in_memory_but_cannot_be_persisted() -> None:
    class CustomSampler:
        name = "custom-test"
        supports_uniform_trial_bound = True

        def __init__(self) -> None:
            self._delegate = UniformFivePointSampler()

        def prepare(self, *args, **kwargs):
            return self._delegate.prepare(*args, **kwargs)

        def describe(self):
            return {"name": self.name}

    application = SphericalRelativePose(sampler=CustomSampler())
    correspondences, _, _ = _known_pose_correspondences()

    assert application.estimate_correspondences(correspondences) is not None
    with pytest.raises(ApplicationConfigurationError, match="not registered"):
        application.to_config()


def test_application_estimator_stage_matches_direct_estimator_and_known_pose() -> None:
    correspondences, expected_rotation, expected_translation = (
        _known_pose_correspondences()
    )
    application = SphericalRelativePose(
        options=_fast_options(), quality_policy=_permissive_policy()
    )

    composed = application.estimate_correspondences(correspondences)
    direct = SphericalRelativePoseEstimator(
        application.options,
        sampler=application.sampler,
        quality_policy=application.quality_policy,
    ).estimate(correspondences)

    assert composed is not None and direct is not None
    np.testing.assert_allclose(composed.R, direct.R, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(composed.t, direct.t, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(composed.R, expected_rotation, atol=1e-5)
    np.testing.assert_allclose(composed.t, expected_translation, atol=1e-5)
    assert np.linalg.norm(composed.t) == pytest.approx(1.0)
    assert composed.describe()["pose_convention"] == "panorama-2-from-panorama-1"


def test_typed_cache_round_trips_relative_pose_result() -> None:
    correspondences, _, _ = _known_pose_correspondences()
    result = SphericalRelativePose(
        options=_fast_options(), quality_policy=_permissive_policy()
    ).estimate_correspondences(correspondences)
    assert result is not None

    arrays, state = encode_typed(result)
    restored = decode_typed(
        {name: np.array(value, copy=True) for name, value in arrays.items()}, state
    )

    assert isinstance(restored, RelativePoseResult)
    np.testing.assert_array_equal(restored.R, result.R)
    np.testing.assert_array_equal(restored.t, result.t)
    np.testing.assert_array_equal(restored.inlier_mask, result.inlier_mask)
    assert restored.quality_report == result.quality_report
    assert restored.sampling_diagnostics == result.sampling_diagnostics
    assert restored.options == result.options


def test_content_cache_distinguishes_cached_none_from_cache_miss(tmp_path) -> None:
    pipeline = Pipeline(Parallel(value=None), memory=tmp_path)
    value = np.arange(8, dtype=np.float32)
    identity = cache_identity(pipeline, value)

    assert load_cached_result(tmp_path, identity, value) is _CACHE_MISS
    store_cached_result(tmp_path, identity, None, value)

    assert load_cached_result(tmp_path, identity, value) is None


def test_pair_cache_identity_preserves_order_content_and_ids() -> None:
    pipeline = SphericalRelativePose().as_pipeline()
    reference = EquirectangularImage(_textured_panorama(24, 48))
    source = EquirectangularImage(np.roll(reference.image, 1, axis=1))
    forward = SphericalImagePair(reference, source, "reference", "source")
    reverse = SphericalImagePair(source, reference, "source", "reference")
    renamed = SphericalImagePair(reference, source, "reference-v2", "source")

    forward_identity = cache_identity(pipeline, forward)

    assert forward_identity["input_kind"] == "spherical_image_pair"
    assert cache_identity(pipeline, reverse)["key"] != forward_identity["key"]
    assert cache_identity(pipeline, renamed)["key"] != forward_identity["key"]


def test_capture_cache_identity_includes_independent_metadata() -> None:
    policy = RelativePoseCapturePolicy.scanner_assisted_reference()
    pipeline = Pipeline(AssessRelativePoseCapture(policy, mode="advisory"))
    image = EquirectangularImage(_textured_panorama(24, 48))
    pair = SphericalImagePair(image, image, "reference", "source")
    first = RelativePoseCaptureInput(
        pair,
        RelativePoseCaptureMetadata(registered_cloud_overlap_min=0.50),
    )
    second = RelativePoseCaptureInput(
        pair,
        RelativePoseCaptureMetadata(registered_cloud_overlap_min=0.70),
    )

    first_identity = cache_identity(pipeline, first)
    second_identity = cache_identity(pipeline, second)
    assert first_identity["input_kind"] == "relative_pose_capture_input"
    assert first_identity["key"] != second_identity["key"]


def test_real_opencv_components_interoperate_through_the_explicit_pipeline() -> None:
    pytest.importorskip("cv2")
    panorama = _textured_panorama()
    source = np.roll(panorama, 2, axis=1)
    frontend = SphericalFeaturePipeline.from_preset(
        "orb-hamming",
        face_sampler="cube",
        face_shape_hw=96,
        face_fov_deg=100.0,
        edge_margin_px=2,
        max_features=300,
    ).config
    application = SphericalRelativePose(
        frontend_config=frontend,
        options=_fast_options(),
        quality_policy=_permissive_policy(),
        capture_policy=RelativePoseCapturePolicy.scanner_assisted_reference(),
    )
    pair = SphericalImagePair(
        EquirectangularImage(panorama),
        EquirectangularImage(source),
        "reference",
        "source",
    )

    capture_input = RelativePoseCaptureInput(
        pair,
        RelativePoseCaptureMetadata(registered_cloud_overlap_min=0.70),
    )
    run = application.as_pipeline().run(
        capture_input,
        retain=("extract_features", "match_features", "bearing_correspondences"),
    )
    direct = application(
        panorama,
        source,
        reference_id="reference",
        source_id="source",
        capture_metadata=RelativePoseCaptureMetadata(registered_cloud_overlap_min=0.70),
    )

    assert len(run["extract_features"].reference) > 0
    assert len(run["extract_features"].source) > 0
    assert len(run["match_features"]) >= 5
    assert run["bearing_correspondences"].valid.sum() >= 5
    assert isinstance(run.output, RelativePoseResult)
    assert isinstance(direct, RelativePoseResult)
    np.testing.assert_allclose(direct.R, run.output.R, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(direct.t, run.output.t, rtol=0.0, atol=0.0)
    assert run.output.describe()["translation"] == "unit-direction-only"


def test_pair_contract_and_validity_overrides_are_explicit() -> None:
    image = _textured_panorama(24, 48)
    with pytest.raises(TypeError, match="reference must be an EquirectangularImage"):
        SphericalImagePair(image, EquirectangularImage(image))
    with pytest.raises(ValueError, match="already carries explicit validity"):
        SphericalRelativePose().run(
            EquirectangularImage(image),
            EquirectangularImage(image),
            reference_validity=np.ones(image.shape[:2], dtype=bool),
        )
