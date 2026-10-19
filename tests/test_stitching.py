from __future__ import annotations

import math

import numpy as np
import pytest

from panorai.applications import Application, PanoramaStitcher
from panorai.calibration import PinholeIntrinsics, pinhole_pixels_to_rays
from panorai.pipeline import Pipeline
from panorai.stitching import (
    BlendStitchingPanorama,
    CalibrateAndRegister,
    CompensateExposure,
    DetectStitchingFeatures,
    DiscoverOverlaps,
    PanoramaStitchingResult,
    RegisteredImageSet,
    StitchingInput,
    StitchingOptions,
    StitchingRegistrationError,
    WarpedImageSet,
    WarpRegisteredImages,
)


def _yaw(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.asarray(
        ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)),
        dtype=np.float64,
    )


def _texture(world_rays: np.ndarray) -> np.ndarray:
    """Independent deterministic continuous texture on unit directions."""

    flat = np.asarray(world_rays, dtype=np.float64).reshape(-1, 3)
    random = np.random.default_rng(4207)
    centres = random.normal(size=(180, 3))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    colours = random.uniform(0.15, 1.0, size=(180, 3))
    dots = flat @ centres.T
    weights = np.exp(np.clip((dots - 1.0) / 0.0009, -80.0, 0.0))
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
    values = 42.0 + 185.0 * blobs + 28.0 * waves
    return (
        np.rint(np.clip(values, 0.0, 255.0))
        .astype(np.uint8)
        .reshape(world_rays.shape[:-1] + (3,))
    )


def _render_view(
    camera_to_world: np.ndarray,
    *,
    shape_hw: tuple[int, int] = (240, 320),
    horizontal_fov_deg: float = 92.0,
) -> np.ndarray:
    intrinsics = PinholeIntrinsics.from_horizontal_fov(shape_hw, horizontal_fov_deg)
    rows, columns = np.indices(shape_hw, dtype=np.float64)
    pixels = np.stack((columns, rows), axis=-1)
    camera_rays = pinhole_pixels_to_rays(pixels, intrinsics)
    world_rays = camera_rays @ camera_to_world.T
    return _texture(world_rays)


def _options(**changes) -> StitchingOptions:
    values = {
        "feature_method": "sift",
        "max_features": 2500,
        "ratio_test": 0.8,
        "min_pair_matches": 12,
        "min_pair_inliers": 8,
        "max_transfer_error_px": 4.0,
        "max_angular_error_deg": 1.5,
        "output_shape_hw": (128, 256),
        "random_seed": 17,
    }
    values.update(changes)
    return StitchingOptions(**values)


def test_stitching_options_reject_an_impossible_homography_consensus() -> None:
    with pytest.raises(ValueError, match="min_pair_inliers must be at least 4"):
        StitchingOptions(min_pair_inliers=3)


def test_stitching_application_and_public_stages_compose_on_known_oracle() -> None:
    rotations = (_yaw(-32.0), _yaw(0.0), _yaw(32.0))
    images = tuple(_render_view(rotation) for rotation in rotations)
    options = _options()
    stitching_input = StitchingInput.from_arrays(
        images,
        image_ids=("left", "centre", "right"),
        horizontal_fov_deg=92.0,
    )
    application = PanoramaStitcher(options)

    direct = application(stitching_input)
    composed = application.as_pipeline()(stitching_input)

    assert isinstance(direct, PanoramaStitchingResult)
    assert isinstance(composed, PanoramaStitchingResult)
    np.testing.assert_array_equal(direct.panorama.data, composed.panorama.data)
    np.testing.assert_array_equal(
        direct.panorama.support_mask, composed.panorama.support_mask
    )
    assert direct.diagnostics.input_image_count == 3
    assert direct.diagnostics.accepted_pair_count >= 2
    assert all(count > 20 for count in direct.diagnostics.feature_counts)
    assert direct.diagnostics.estimated_fov_count == 0
    assert direct.diagnostics.coverage > 0.15
    assert direct.diagnostics.overlap_fraction > 0.02
    assert direct.registered.rotation_final_cost <= (
        direct.registered.rotation_initial_cost + 1e-12
    )
    for edge in direct.registered.pairwise:
        expected = rotations[edge.index_b].T @ rotations[edge.index_a]
        delta = edge.rotation_b_from_a.T @ expected
        angle_deg = math.degrees(
            math.acos(float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)))
        )
        assert angle_deg < 1.0
    assert type(application.with_params(options__blend="average")) is PanoramaStitcher
    assert [
        type(item).__name__ for item in application.pipeline.named_steps.values()
    ] == [
        "DetectStitchingFeatures",
        "DiscoverOverlaps",
        "CalibrateAndRegister",
        "WarpRegisteredImages",
        "CompensateExposure",
        "BlendStitchingPanorama",
    ]


def test_missing_fov_is_estimated_without_image_metadata() -> None:
    images = (_render_view(_yaw(-16.0)), _render_view(_yaw(16.0)))
    result = PanoramaStitcher(_options())(images, image_ids=("a", "b"))

    estimated = tuple(item.horizontal_fov_deg for item in result.registered.intrinsics)
    assert result.diagnostics.estimated_fov_count == 2
    assert all(abs(item - 92.0) < 8.0 for item in estimated)
    assert all(edge.calibration.observable for edge in result.registered.pairwise)


def test_stitching_recovers_pairwise_geometry_after_input_shuffle() -> None:
    expected_by_id = {
        "left": _yaw(-32.0),
        "centre": _yaw(0.0),
        "right": _yaw(32.0),
    }
    image_ids = ("right", "left", "centre")
    images = tuple(_render_view(expected_by_id[image_id]) for image_id in image_ids)

    result = PanoramaStitcher(_options())(
        images,
        image_ids=image_ids,
        horizontal_fov_deg=92.0,
    )

    assert result.diagnostics.accepted_pair_count >= 2
    for edge in result.registered.pairwise:
        rotation_a = expected_by_id[image_ids[edge.index_a]]
        rotation_b = expected_by_id[image_ids[edge.index_b]]
        expected = rotation_b.T @ rotation_a
        delta = edge.rotation_b_from_a.T @ expected
        angle_deg = math.degrees(
            math.acos(float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)))
        )
        assert angle_deg < 1.0


def test_exposure_compensation_recovers_global_gain_from_valid_overlap() -> None:
    shape = (8, 8)
    source = StitchingInput.from_arrays(
        (
            np.full(shape + (3,), 100.0),
            np.full(shape + (3,), 50.0),
        ),
        horizontal_fov_deg=90.0,
    )
    intrinsics = PinholeIntrinsics.from_horizontal_fov(shape, 90.0)
    registered = RegisteredImageSet(
        source=source,
        intrinsics=(intrinsics, intrinsics),
        camera_to_world=(np.eye(3), np.eye(3)),
        pairwise=(),
        feature_counts=(0, 0),
        reference_index=0,
        rotation_initial_cost=0.0,
        rotation_final_cost=0.0,
        rejected_pairs=(),
    )
    valid = np.ones(shape, dtype=bool)
    warped = WarpedImageSet(
        registered=registered,
        values=tuple(image.values for image in source.images),
        support=(valid, valid),
        validity=(valid, valid),
        weights=(valid.astype(float), valid.astype(float)),
        gains=(1.0, 1.0),
    )

    compensated = CompensateExposure(_options()).execute(warped).value

    assert compensated.gains == pytest.approx((1.0, 2.0), abs=1e-12)
    np.testing.assert_allclose(compensated.values[0], compensated.values[1])


def test_stitching_rejects_disconnected_textureless_images() -> None:
    blank = np.zeros((120, 160, 3), dtype=np.uint8)
    pipeline = Pipeline(
        DetectStitchingFeatures(_options()),
        DiscoverOverlaps(_options()),
        CalibrateAndRegister(_options()),
    )

    with pytest.raises(StitchingRegistrationError, match="no image pair"):
        pipeline(StitchingInput.from_arrays((blank, blank), horizontal_fov_deg=90.0))


def test_warp_and_blend_keep_black_valid_and_invalidity_separate_from_support() -> None:
    shape = (80, 120)
    black = np.zeros(shape + (3,), dtype=np.uint8)
    support = np.ones(shape, dtype=bool)
    validity = np.ones(shape, dtype=bool)
    validity[:, 50:70] = False
    source = StitchingInput.from_arrays(
        (black, black),
        support_masks=(support, support),
        validity_masks=(validity, validity),
        horizontal_fov_deg=90.0,
    )
    intrinsics = PinholeIntrinsics.from_horizontal_fov(shape, 90.0)
    registered = RegisteredImageSet(
        source=source,
        intrinsics=(intrinsics, intrinsics),
        camera_to_world=(np.eye(3), np.eye(3)),
        pairwise=(),
        feature_counts=(0, 0),
        reference_index=0,
        rotation_initial_cost=0.0,
        rotation_final_cost=0.0,
        rejected_pairs=(),
    )
    options = _options(output_shape_hw=(80, 160), exposure_compensation="none")
    warped = WarpRegisteredImages(options).execute(registered).value
    result = BlendStitchingPanorama(options).execute(warped).value

    result_validity = result.panorama.validity("image")
    assert np.any(result.panorama.support_mask & result_validity)
    assert np.all(result.panorama.data[result_validity] == 0)
    assert np.any(result.panorama.support_mask & ~result_validity)
    assert np.all(result.support_count[~result.panorama.support_mask] == 0)


def test_stitcher_config_round_trip_uses_explicit_registry(tmp_path) -> None:
    application = PanoramaStitcher(_options(output_shape_hw=(96, 192), blend="average"))
    path = application.save(tmp_path / "stitching.json")

    restored = Application.load(path)

    assert restored == application
    assert restored.to_config() == application.to_config()
    assert restored.describe()["model"] == "central rotation-only"

    pipeline = application.as_pipeline()
    restored_pipeline = Pipeline.from_config(pipeline.to_config())
    assert restored_pipeline.to_config() == pipeline.to_config()
    archive = pipeline.save(tmp_path / "pipeline")
    assert Pipeline.load(archive).to_config() == pipeline.to_config()
