from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from panorai.features import (  # noqa: E402
    OpenCVTangentDescriptor,
    OpenCVTangentDescriptorConfig,
    SphericalDoGDetector,
    SphericalDoGDetectorConfig,
    SphericalDoGSIFTConfig,
    SphericalDoGSIFTExtractor,
    TangentPatchProvider,
    TangentPatchRequest,
)


def _textured_panorama(shape: tuple[int, int] = (64, 128)) -> np.ndarray:
    height, width = shape
    yy, xx = np.indices(shape)
    values = (
        np.sin(xx * 0.41)
        + np.sin(yy * 0.57)
        + np.sin((xx + yy) * 0.19)
        + 0.4 * np.cos((2 * xx - yy) * 0.11)
    )
    return np.clip(values * 35.0 + 128.0, 0.0, 255.0).astype(np.uint8)


def _keypoints(image: np.ndarray, *, maximum: int = 30):
    return SphericalDoGDetector(
        SphericalDoGDetectorConfig(
            octaves=2,
            max_keypoints=maximum,
            contrast_threshold=0.003,
            convolution_backend="numpy",
        )
    ).detect(image, panorama_id="synthetic")


def test_request_keeps_context_separate_from_sampling_density() -> None:
    request = TangentPatchRequest(
        output_shape_hw=(40, 64),
        radius_in_scales=8.0,
        minimum_valid_fraction=0.75,
    )
    description = request.to_dict()

    assert description["interface"] == "panorai-tangent-patches/v1"
    assert description["output_shape_hw"] == (40, 64)
    assert description["radius_in_scales"] == 8.0
    assert description["fov_rule"].startswith("2 * radius_in_scales")

    with pytest.raises(ValueError, match="minimum_fov_deg"):
        TangentPatchRequest(minimum_fov_deg=20.0, maximum_fov_deg=10.0)
    with pytest.raises(ValueError, match="min_valid_weight"):
        TangentPatchRequest(min_valid_weight=0.5)


def test_provider_preserves_keypoint_order_geometry_and_validity() -> None:
    image = _textured_panorama()
    keypoints = _keypoints(image)
    request = TangentPatchRequest(
        output_shape_hw=(40, 64),
        radius_in_scales=6.0,
        minimum_valid_fraction=0.95,
    )

    patches = TangentPatchProvider().materialize(image, keypoints, request)

    assert patches.interface == "panorai-tangent-patches/v1"
    assert patches.keypoint_interface == "panorai-spherical-dog-detector/v1"
    assert len(patches) == len(keypoints)
    assert patches.valid_count > 0
    for index, patch in enumerate(patches.patches):
        assert patch.geometry.keypoint_index == index
        assert patch.image.shape == request.output_shape_hw
        assert patch.support_mask.shape == request.output_shape_hw
        assert patch.validity_mask.shape == request.output_shape_hw
        assert patch.geometry.K.shape == (3, 3)
        assert patch.geometry.R_panorama_from_patch.shape == (3, 3)
        assert np.allclose(
            patch.geometry.center_bearing_xyz,
            keypoints.keypoints[index].bearing_xyz,
        )
        assert np.allclose(
            patch.geometry.source_erp_xy,
            keypoints.keypoints[index].source_erp_xy,
        )
        assert 0.0 <= patch.valid_fraction <= 1.0
        assert patch.source_panorama_checksum == patches.source_checksum


def test_provider_rejects_wrong_source_and_requires_explicit_rolls() -> None:
    image = _textured_panorama()
    keypoints = _keypoints(image, maximum=12)
    changed = image.copy()
    changed[0, 0] ^= np.uint8(1)

    with pytest.raises(ValueError, match="not detected on this panorama"):
        TangentPatchProvider().materialize(changed, keypoints, TangentPatchRequest())

    request = TangentPatchRequest(orientation_policy="per-keypoint-roll")
    with pytest.raises(ValueError, match="rolls_deg is required"):
        TangentPatchProvider().materialize(image, keypoints, request)
    rolls = np.linspace(0.0, 30.0, len(keypoints))
    patches = TangentPatchProvider().materialize(
        image, keypoints, request, rolls_deg=rolls
    )
    observed = np.asarray(
        [item.geometry.projection_spec.roll_deg for item in patches.patches]
    )
    assert np.allclose(observed, rolls)


def test_provider_keeps_support_and_observation_validity_distinct() -> None:
    image = _textured_panorama()
    keypoints = _keypoints(image, maximum=10)
    no_observations = np.zeros(image.shape, dtype=bool)
    patches = TangentPatchProvider().materialize(
        image,
        keypoints,
        TangentPatchRequest(
            invalid_policy="renormalize",
            min_valid_weight=0.5,
            minimum_valid_fraction=0.1,
        ),
        validity_mask=no_observations,
    )

    assert len(patches) == len(keypoints)
    assert patches.valid_count == 0
    assert all(np.any(item.support_mask) for item in patches.patches)
    assert all(not np.any(item.validity_mask) for item in patches.patches)


def test_colour_patches_are_consumed_without_detector_coupling() -> None:
    gray = _textured_panorama()
    colour = np.stack((gray, np.roll(gray, 3, axis=1), 255 - gray), axis=-1)
    keypoints = _keypoints(colour, maximum=15)
    patches = TangentPatchProvider().materialize(
        colour, keypoints, TangentPatchRequest(output_shape_hw=(64, 64))
    )
    described = OpenCVTangentDescriptor(
        OpenCVTangentDescriptorConfig(method="sift")
    ).describe(patches)

    assert len(described) > 0
    assert patches.patches[0].image.shape == (64, 64, 3)
    assert described.descriptors.shape[1] == 128


def test_same_patches_are_reusable_by_sift_and_orb() -> None:
    image = _textured_panorama((96, 192))
    keypoints = _keypoints(image, maximum=40)
    patches = TangentPatchProvider().materialize(
        image,
        keypoints,
        TangentPatchRequest(
            output_shape_hw=(64, 64),
            radius_in_scales=6.0,
            minimum_valid_fraction=0.95,
        ),
    )
    responses = np.asarray([item.response for item in keypoints.keypoints])

    sift = OpenCVTangentDescriptor(
        OpenCVTangentDescriptorConfig(method="sift")
    ).describe(patches, responses=responses)
    orb = OpenCVTangentDescriptor(
        OpenCVTangentDescriptorConfig(
            method="orb",
            algorithm_parameters=(("edgeThreshold", 5), ("patchSize", 31)),
        )
    ).describe(patches, responses=responses)

    assert len(sift) > 0
    assert len(orb) > 0
    assert sift.patch_interface == orb.patch_interface == patches.interface
    assert sift.descriptors.shape[1] == 128
    assert sift.descriptors.dtype == np.float32
    assert sift.descriptor_metric == "l2"
    assert orb.descriptors.shape[1] == 32
    assert orb.descriptors.dtype == np.uint8
    assert orb.descriptor_metric == "hamming"


def test_legacy_spherical_sift_is_a_client_of_the_generic_contract() -> None:
    image = _textured_panorama()
    config = SphericalDoGSIFTConfig(
        octaves=2,
        max_features=30,
        contrast_threshold=0.003,
        convolution_backend="numpy",
    )
    extractor = SphericalDoGSIFTExtractor(config)
    keypoints = extractor.detect(image, panorama_id="synthetic")
    legacy = extractor.describe_keypoints(image, keypoints)
    patches = TangentPatchProvider().materialize(
        image,
        keypoints,
        TangentPatchRequest(
            output_shape_hw=(config.patch_size, config.patch_size),
            radius_in_scales=config.descriptor_radius_sigmas,
            minimum_fov_deg=1.0,
            maximum_fov_deg=120.0,
            minimum_valid_fraction=config.minimum_valid_fraction,
        ),
    )
    direct = OpenCVTangentDescriptor(
        OpenCVTangentDescriptorConfig(
            method="sift",
            keypoint_diameter_in_scales=(
                2.0
                * config.descriptor_radius_sigmas
                * config.descriptor_keypoint_size_fraction
            ),
            orientation_bins=config.orientation_bins,
        )
    ).describe(
        patches,
        responses=np.asarray([item.response for item in keypoints.keypoints]),
    )

    assert legacy.extractor_config["patch_interface"] == patches.interface
    assert len(legacy) == len(direct)
    assert np.array_equal(legacy.descriptors, direct.descriptors)
    assert np.allclose(
        np.asarray([item.angle_deg for item in legacy.features]),
        direct.angles_deg,
    )
    assert np.allclose(
        legacy.bearings,
        keypoints.bearings[direct.patch_indices],
    )
