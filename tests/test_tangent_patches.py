from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("cv2")

from panorai.features import (  # noqa: E402
    OpenCVTangentDescriptor,
    OpenCVTangentDescriptorConfig,
    OpenCVTangentDescriptorV2,
    OpenCVTangentDescriptorV2Config,
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
    assert patches.keypoint_interface == "panorai-spherical-dog-detector/v2"
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


def test_descriptor_v2_explicit_neutral_profile_matches_fixed_zero_v1() -> None:
    image = _textured_panorama((96, 192))
    keypoints = _keypoints(image, maximum=25)
    patches = TangentPatchProvider().materialize(
        image,
        keypoints,
        TangentPatchRequest(output_shape_hw=(48, 48), minimum_valid_fraction=0.95),
    )
    responses = np.asarray([item.response for item in keypoints.keypoints])
    v1 = OpenCVTangentDescriptor(
        OpenCVTangentDescriptorConfig(
            method="sift",
            keypoint_diameter_in_scales=1.5,
            orientation_policy="fixed-zero",
        )
    ).describe(patches, responses=responses)
    v2 = OpenCVTangentDescriptorV2(
        OpenCVTangentDescriptorV2Config(
            keypoint_diameter_in_scales=1.5,
            photometric_normalization="none",
            minimum_descriptor_valid_fraction=0.0,
            root_sift=False,
        )
    ).describe(patches, responses=responses)

    assert v2.interface == "panorai-tangent-opencv-descriptor/v2"
    np.testing.assert_array_equal(v2.patch_indices, v1.patch_indices)
    np.testing.assert_array_equal(v2.physical_keypoint_ids, v1.patch_indices)
    np.testing.assert_array_equal(v2.descriptors, v1.descriptors)
    np.testing.assert_array_equal(v2.angles_deg, v1.angles_deg)
    np.testing.assert_array_equal(v2.scale_multipliers, 1.0)
    np.testing.assert_array_equal(v2.orientation_ranks, 0)


@pytest.mark.parametrize(
    "normalization",
    ("local-standardization", "robust-percentile-2-98"),
)
def test_descriptor_v2_photometric_normalization_is_deterministic(
    normalization: str,
) -> None:
    image = _textured_panorama((96, 192))
    keypoints = _keypoints(image, maximum=20)
    patches = TangentPatchProvider().materialize(
        image, keypoints, TangentPatchRequest(minimum_valid_fraction=0.95)
    )
    config = OpenCVTangentDescriptorV2Config(
        photometric_normalization=normalization,
        root_sift=True,
    )
    adapter = OpenCVTangentDescriptorV2(config)
    first = adapter.describe(patches)
    second = adapter.describe(patches)

    assert len(first) > 0
    np.testing.assert_array_equal(first.descriptors, second.descriptors)
    assert np.isfinite(first.descriptors).all()
    assert np.all(first.descriptor_valid_fractions >= 0.0)
    assert np.all(first.descriptor_valid_fractions <= 1.0)


def test_descriptor_v2_preserves_physical_identity_across_scale_hypotheses() -> None:
    image = _textured_panorama((96, 192))
    keypoints = _keypoints(image, maximum=18)
    patches = TangentPatchProvider().materialize(
        image, keypoints, TangentPatchRequest(minimum_valid_fraction=0.95)
    )
    result = OpenCVTangentDescriptorV2(
        OpenCVTangentDescriptorV2Config(
            scale_multipliers=(0.8, 1.0, 1.25),
            orientation_policy="fixed-zero",
        )
    ).describe(patches)

    assert len(result) > 0
    for physical_id in np.unique(result.physical_keypoint_ids):
        selected = result.physical_keypoint_ids == physical_id
        np.testing.assert_allclose(result.scale_multipliers[selected], (0.8, 1.0, 1.25))
        np.testing.assert_array_equal(result.orientation_ranks[selected], 0)
        assert np.unique(result.patch_indices[selected]).size == 1


def test_descriptor_v2_rejects_invalid_effective_support() -> None:
    image = _textured_panorama((96, 192))
    keypoints = _keypoints(image, maximum=15)
    patches = TangentPatchProvider().materialize(
        image, keypoints, TangentPatchRequest(minimum_valid_fraction=0.95)
    )
    first = patches.patches[0]
    invalid_first = replace(
        first,
        validity_mask=np.zeros_like(first.validity_mask),
        valid=True,
    )
    changed = replace(patches, patches=(invalid_first, *patches.patches[1:]))
    result = OpenCVTangentDescriptorV2(
        OpenCVTangentDescriptorV2Config(minimum_descriptor_valid_fraction=1.0)
    ).describe(changed)

    assert 0 not in result.patch_indices
    assert np.all(result.descriptor_valid_fractions == 1.0)


def test_descriptor_v2_multi_peak_orientation_is_bounded_and_auditable() -> None:
    image = _textured_panorama((96, 192))
    keypoints = _keypoints(image, maximum=16)
    patches = TangentPatchProvider().materialize(
        image, keypoints, TangentPatchRequest(minimum_valid_fraction=0.95)
    )
    result = OpenCVTangentDescriptorV2(
        OpenCVTangentDescriptorV2Config(
            orientation_policy="multi-peak-gradient",
            orientation_peak_ratio=0.6,
            max_orientations=2,
        )
    ).describe(patches)

    assert len(result) > 0
    assert np.all(result.orientation_ranks >= 0)
    assert np.all(result.orientation_ranks < 2)
    assert np.all(result.orientation_confidences >= 0.6)
    assert np.all(result.orientation_confidences <= 1.0)
    counts = np.unique(result.physical_keypoint_ids, return_counts=True)[1]
    assert np.all(counts <= 2)


def test_parallel_provider_is_exact_and_ordered() -> None:
    image = _textured_panorama((96, 192))
    keypoints = _keypoints(image, maximum=40)
    request = TangentPatchRequest(
        output_shape_hw=(48, 48),
        radius_in_scales=6.0,
        minimum_valid_fraction=0.95,
    )

    serial = TangentPatchProvider(max_workers=1).materialize(image, keypoints, request)
    parallel = TangentPatchProvider(max_workers=4).materialize(
        image, keypoints, request
    )

    assert len(serial) == len(parallel)
    for left, right in zip(serial.patches, parallel.patches, strict=True):
        assert left.geometry.patch_id == right.geometry.patch_id
        assert left.valid_fraction == right.valid_fraction
        assert left.valid == right.valid
        np.testing.assert_array_equal(left.image, right.image)
        np.testing.assert_array_equal(left.support_mask, right.support_mask)
        np.testing.assert_array_equal(left.validity_mask, right.validity_mask)


def test_canonical_spherical_sift_is_a_client_of_descriptor_v2() -> None:
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
    direct = OpenCVTangentDescriptorV2(config.descriptor_config).describe(
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
        keypoints.bearings[direct.physical_keypoint_ids],
    )
