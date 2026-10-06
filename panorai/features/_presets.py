"""Versioned spherical feature presets with no silent OpenCV defaults."""

from __future__ import annotations

from dataclasses import replace

from ._config import (
    FaceSetSpec,
    FeatureExtractorConfig,
    FeatureMatcherConfig,
    SphericalFeaturePipelineConfig,
)


PRESET_VERSION = 1
MINIMUM_OPENCV_VERSION = "4.9.0"
RELATIVE_POSE_REFERENCE_PROFILE = "relative-pose-reference"


def preset_components(
    name: str,
) -> tuple[FeatureExtractorConfig, FeatureMatcherConfig]:
    normalized = name.strip().lower()
    if normalized == "sift-flann":
        return (
            FeatureExtractorConfig(
                method="sift",
                parameters={
                    "nOctaveLayers": 3,
                    "sigma": 1.6,
                    "enable_precise_upscale": False,
                },
            ),
            FeatureMatcherConfig(
                method="flann",
                ratio_test=0.75,
                cross_check=False,
                parameters={"trees": 5, "checks": 50},
            ),
        )
    if normalized == "sift-bf":
        return (
            FeatureExtractorConfig(
                method="sift",
                parameters={
                    "nOctaveLayers": 3,
                    "sigma": 1.6,
                    "enable_precise_upscale": False,
                },
            ),
            FeatureMatcherConfig(method="bf", ratio_test=0.75, cross_check=False),
        )
    if normalized == "orb-hamming":
        return (
            FeatureExtractorConfig(
                method="orb",
                parameters={
                    "scaleFactor": 1.2,
                    "nlevels": 8,
                    "edgeThreshold": 31,
                    "firstLevel": 0,
                    "WTA_K": 2,
                    "scoreType": 0,
                    "patchSize": 31,
                    "fastThreshold": 20,
                },
            ),
            FeatureMatcherConfig(method="bf", ratio_test=0.8, cross_check=False),
        )
    if normalized == "akaze-hamming":
        return (
            FeatureExtractorConfig(
                method="akaze",
                akaze_descriptor_type="binary",
                parameters={
                    "descriptor_size": 0,
                    "descriptor_channels": 3,
                    "threshold": 0.001,
                    "nOctaves": 4,
                    "nOctaveLayers": 4,
                    "diffusivity": 1,
                    "max_points": -1,
                },
            ),
            FeatureMatcherConfig(method="bf", ratio_test=0.8, cross_check=False),
        )
    raise ValueError(
        "unknown preset; choose 'sift-flann', 'sift-bf', "
        "'orb-hamming', or 'akaze-hamming'"
    )


def available_presets() -> tuple[str, ...]:
    return ("sift-flann", "sift-bf", "orb-hamming", "akaze-hamming")


def relative_pose_reference_config() -> SphericalFeaturePipelineConfig:
    """Return the frozen feature/matching profile validated for relative pose."""

    extractor, matcher = preset_components("sift-flann")
    extractor = replace(
        extractor,
        max_features=4096,
        edge_margin_px=16,
        validity_margin_px=0,
        validity_scale_margin=1.5,
        deduplicate_overlaps=True,
        angular_dedup_threshold_deg=0.15,
    )
    matcher = replace(
        matcher,
        ratio_test=0.72,
        cross_check=False,
        max_distance=None,
        deduplicate_matches=True,
        angular_dedup_threshold_deg=0.15,
    )
    return SphericalFeaturePipelineConfig(
        extractor=extractor,
        matcher=matcher,
        face_set=FaceSetSpec(
            sampler="cube",
            shape_hw=(1024, 1024),
            fov_deg=(95.0, 95.0),
        ),
        preset_name=RELATIVE_POSE_REFERENCE_PROFILE,
        preset_version=PRESET_VERSION,
        minimum_opencv_version=MINIMUM_OPENCV_VERSION,
    )
