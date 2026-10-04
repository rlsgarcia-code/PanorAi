"""Versioned spherical feature presets with no silent OpenCV defaults."""

from __future__ import annotations

from ._config import FeatureExtractorConfig, FeatureMatcherConfig


PRESET_VERSION = 1
MINIMUM_OPENCV_VERSION = "4.9.0"


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
