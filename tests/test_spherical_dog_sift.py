from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("cv2")

from panorai.features import (  # noqa: E402
    SphericalCoarseDoGDetector,
    SphericalCoarseDoGDetectorConfig,
    SphericalDoGSIFTConfig,
    SphericalDoGSIFTExtractor,
    SphericalDoGSIFTPipeline,
)
from panorai.geometry import erp_pixels_to_rays  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _textured_panorama(shape: tuple[int, int] = (64, 128)) -> np.ndarray:
    height, width = shape
    yy, xx = np.indices(shape)
    values = np.sin(xx * 0.41) + np.sin(yy * 0.57) + np.sin((xx + yy) * 0.19)
    return np.clip(values * 35.0 + 128.0, 0.0, 255.0).astype(np.uint8)


def _config(**changes: object) -> SphericalDoGSIFTConfig:
    values: dict[str, object] = {
        "octaves": 2,
        "max_features": 80,
        "contrast_threshold": 0.003,
        "convolution_backend": "numpy",
    }
    values.update(changes)
    return SphericalDoGSIFTConfig(**values)  # type: ignore[arg-type]


def test_configuration_is_explicitly_experimental_and_validated() -> None:
    description = _config().to_dict()

    assert description["interface"] == "panorai-spherical-dog-sift/v2"
    assert description["stability"] == "experimental"
    assert description["descriptor"] == "opencv-sift"
    assert description["descriptor_adapter_interface"] == (
        "panorai-tangent-opencv-descriptor/v2"
    )
    assert description["descriptor_config"]["keypoint_diameter_in_scales"] == 1.25
    assert description["descriptor_config"]["orientation_policy"] == "fixed-zero"
    assert description["descriptor_config"]["photometric_normalization"] == (
        "local-standardization"
    )
    assert description["descriptor_config"]["root_sift"] is True
    assert description["scale_units"] == "degrees"

    with pytest.raises(ValueError, match="patch_size must be even"):
        _config(patch_size=47)
    with pytest.raises(ValueError, match="convolution_backend"):
        _config(convolution_backend="unknown")


def test_spherical_detector_produces_aligned_sift_descriptors() -> None:
    result = SphericalDoGSIFTExtractor(_config()).extract(
        _textured_panorama(), panorama_id="synthetic"
    )

    assert 20 <= len(result) <= 80
    assert result.descriptors.shape == (len(result), 128)
    assert result.descriptors.dtype == np.float32
    assert result.descriptor_type == "root-sift-float32"
    assert result.interface == "panorai-spherical-dog-sift/v2"
    assert result.stability == "experimental"
    assert np.allclose(np.linalg.norm(result.bearings, axis=1), 1.0)
    assert np.isfinite(result.source_erp_xy).all()
    assert len(set(result.face_ids)) == len(result)
    assert all(
        item.provenance.interface == "panorai-spherical-dog-sift/v2"
        for item in result.features
    )


def test_descriptor_can_consume_public_descriptor_free_keypoints() -> None:
    image = _textured_panorama()
    extractor = SphericalDoGSIFTExtractor(_config())

    keypoints = extractor.detect(image, panorama_id="synthetic")
    result = extractor.describe_keypoints(image, keypoints)

    assert keypoints.interface == "panorai-spherical-dog-detector/v2"
    assert 0 < len(result) <= len(keypoints)
    assert result.panorama_id == keypoints.panorama_id
    assert result.descriptors.shape == (len(result), 128)
    assert result.extractor_config["detector_interface"] == keypoints.interface


def test_descriptor_rejects_keypoints_from_another_panorama() -> None:
    image = _textured_panorama()
    extractor = SphericalDoGSIFTExtractor(_config())
    keypoints = extractor.detect(image)
    changed = image.copy()
    changed[0, 0] ^= np.uint8(1)

    with pytest.raises(ValueError, match="not detected on this panorama"):
        extractor.describe_keypoints(changed, keypoints)


def test_descriptor_consumes_promoted_coarse_keypoints_on_source_panorama() -> None:
    image = _textured_panorama((128, 256))
    keypoints = SphericalCoarseDoGDetector(
        SphericalCoarseDoGDetectorConfig(
            proposal_height=64,
            octaves=2,
            max_keypoints=60,
            contrast_threshold=0.003,
            convolution_backend="numpy",
        )
    ).detect(image, panorama_id="coarse-synthetic")

    result = SphericalDoGSIFTExtractor(_config(max_features=60)).describe_keypoints(
        image, keypoints
    )

    assert 0 < len(result) <= len(keypoints)
    assert result.panorama_id == keypoints.panorama_id
    assert result.descriptors.shape == (len(result), 128)


def test_longitude_roll_is_equivariant_through_matching() -> None:
    image = _textured_panorama()
    shift = 9
    pipeline = SphericalDoGSIFTPipeline(_config(max_features=100))
    left = pipeline.extract(image, panorama_id="left")
    right = pipeline.extract(np.roll(image, shift, axis=1), panorama_id="right")
    matches = pipeline.match(left, right)

    assert len(matches) >= 40
    assert matches.interface == "panorai-spherical-dog-sift/v2"
    assert matches.stability == "experimental"
    horizontal_shift = (
        right.source_erp_xy[matches.feature_indices_b, 0]
        - left.source_erp_xy[matches.feature_indices_a, 0]
    ) % image.shape[1]
    circular_error = np.minimum(
        np.abs(horizontal_shift - shift),
        image.shape[1] - np.abs(horizontal_shift - shift),
    )
    assert np.mean(circular_error < 1.5) >= 0.9


def test_high_latitude_blob_crossing_seam_has_a_finite_detection() -> None:
    height, width = 64, 128
    yy, xx = np.indices((height, width))
    rays = erp_pixels_to_rays(
        np.stack((xx, yy), axis=-1).astype(np.float64), (height, width)
    )
    longitude = np.deg2rad(179.0)
    latitude = np.deg2rad(74.0)
    centre = np.asarray(
        (
            np.sin(longitude) * np.cos(latitude),
            np.sin(latitude),
            np.cos(longitude) * np.cos(latitude),
        )
    )
    angular_distance = np.arccos(np.clip(rays @ centre, -1.0, 1.0))
    image = np.clip(
        40.0 + 210.0 * np.exp(-0.5 * (angular_distance / np.deg2rad(5.0)) ** 2),
        0.0,
        255.0,
    ).astype(np.uint8)
    result = SphericalDoGSIFTExtractor(
        _config(
            max_features=30,
            base_sigma_px=1.0,
            contrast_threshold=0.0005,
            edge_threshold=20.0,
        )
    ).extract(image)

    assert len(result) > 0
    assert np.isfinite(result.descriptors).all()
    distance_deg = np.rad2deg(np.arccos(np.clip(result.bearings @ centre, -1.0, 1.0)))
    assert distance_deg.min() < 1.0
    closest_x = result.source_erp_xy[int(np.argmin(distance_deg)), 0]
    assert closest_x < 2.0 or closest_x > width - 2.0


def test_validity_mask_rejects_keypoints_outside_valid_support() -> None:
    image = _textured_panorama()
    mask = np.zeros(image.shape, dtype=bool)
    mask[:, : image.shape[1] // 2] = True
    result = SphericalDoGSIFTExtractor(
        _config(max_features=50, minimum_valid_fraction=0.8)
    ).extract(image, validity_mask=mask)

    assert len(result) > 0
    pixels = np.rint(result.source_erp_xy).astype(np.int64)
    assert mask[pixels[:, 1], pixels[:, 0]].all()


def test_panorama_validity_and_support_are_combined_with_caller_mask() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import numpy as np
from panorai.data import EquirectangularImage
from panorai.features import SphericalDoGSIFTConfig, SphericalDoGSIFTExtractor

height, width = 64, 128
yy, xx = np.indices((height, width))
values = np.sin(xx * 0.41) + np.sin(yy * 0.57) + np.sin((xx + yy) * 0.19)
image = np.clip(values * 35.0 + 128.0, 0.0, 255.0).astype(np.uint8)
valid = np.zeros(image.shape, dtype=bool)
valid[:, :96] = True
support = np.zeros(image.shape, dtype=bool)
support[:, 24:] = True
caller = np.zeros(image.shape, dtype=bool)
caller[8:-8, :] = True
panorama = EquirectangularImage(image, valid=valid, support_mask=support)
config = SphericalDoGSIFTConfig(
    octaves=2,
    max_features=50,
    contrast_threshold=0.003,
    minimum_valid_fraction=0.6,
    convolution_backend="numpy",
)
result = SphericalDoGSIFTExtractor(config).extract(panorama, validity_mask=caller)
assert len(result) > 0
pixels = np.rint(result.source_erp_xy).astype(np.int64)
assert (valid & support & caller)[pixels[:, 1], pixels[:, 0]].all()
""",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
