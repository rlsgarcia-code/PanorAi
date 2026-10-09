"""The explicit optimized two-image spherical frontend route."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from panorai.features import (
    FaceSetSpec,
    FeatureMatcher,
    FeatureMatcherConfig,
    FeatureProvenance,
    OpenCVTangentDescriptorV2,
    OpenCVTangentDescriptorV2Config,
    SphericalDoGDetector,
    SphericalDoGDetectorConfig,
    SphericalFeature,
    SphericalFeatureMatches,
    SphericalFeatureSet,
    TangentPatchProvider,
    TangentPatchRequest,
)

from ._contract import CALIBRATED_FRONTEND_ID


@dataclass(frozen=True, slots=True)
class FrontendTimings:
    detection_seconds: float
    patches_seconds: float
    descriptor_seconds: float
    matching_seconds: float
    total_seconds: float


@dataclass(frozen=True, slots=True)
class FrontendResult:
    matches: SphericalFeatureMatches
    keypoint_counts: tuple[int, int]
    valid_fractions: tuple[float, float]
    timings: FrontendTimings
    configuration: dict[str, Any]


class OptimizedSphericalFrontend:
    """PanorAi 3.5-calibrated batch-2 native spherical DoG + RootSIFT route."""

    def __init__(self) -> None:
        self.detector_config = SphericalDoGDetectorConfig(
            octaves=3,
            levels_per_octave=3,
            base_sigma_px=1.6,
            contrast_threshold=0.012,
            edge_threshold=10.0,
            refinement_max_iterations=5,
            max_keypoints=4096,
            minimum_valid_support_fraction=0.99,
            angular_dedup_threshold_deg=0.12,
            scale_dedup_log2=0.5,
            selection_policy="equal-area-round-robin",
            selection_grid_shape=(12, 24),
            convolution_backend="native",
        )
        self.patch_request = TangentPatchRequest(
            output_shape_hw=(48, 48),
            radius_in_scales=6.0,
            minimum_fov_deg=1.0,
            maximum_fov_deg=120.0,
            interpolation="bilinear",
            invalid_policy="propagate",
            minimum_valid_fraction=0.99,
            orientation_policy="upright",
        )
        self.descriptor_config = OpenCVTangentDescriptorV2Config(
            method="sift",
            keypoint_diameter_in_scales=1.25,
            scale_multipliers=(1.0,),
            orientation_policy="fixed-zero",
            photometric_normalization="local-standardization",
            minimum_descriptor_valid_fraction=0.99,
            root_sift=True,
        )
        self.matcher_config = FeatureMatcherConfig(
            method="flann",
            ratio_test=0.72,
            cross_check=False,
            deduplicate_matches=True,
            angular_dedup_threshold_deg=0.15,
        )
        self._detector = SphericalDoGDetector(self.detector_config)
        self._patch_provider = TangentPatchProvider(max_workers=4)
        self._descriptor = OpenCVTangentDescriptorV2(self.descriptor_config)
        self._matcher = FeatureMatcher(self.matcher_config)

    @property
    def configuration(self) -> dict[str, Any]:
        return {
            "detector": self.detector_config.to_dict(),
            "patches": asdict(self.patch_request),
            "descriptor": self.descriptor_config.to_dict(),
            "matcher": self.matcher_config.to_dict(),
            "batch_size": 2,
            "patch_workers": 4,
        }

    @property
    def calibration_id(self) -> str:
        """Identity of the frozen probability calibration for this route."""

        return CALIBRATED_FRONTEND_ID

    def extract_and_match(
        self,
        panorama_a: np.ndarray,
        panorama_b: np.ndarray,
        *,
        validity_a: np.ndarray,
        validity_b: np.ndarray,
        panorama_ids: tuple[str, str] = ("a", "b"),
    ) -> FrontendResult:
        """Process exactly two equal-resolution EQR images via the batch route."""

        first = np.asarray(panorama_a)
        second = np.asarray(panorama_b)
        mask_a = np.asarray(validity_a)
        mask_b = np.asarray(validity_b)
        if first.shape != second.shape or first.ndim not in (2, 3):
            raise ValueError("panoramas must have the same HxW or HxWxC shape")
        if first.shape[:2] != mask_a.shape or mask_a.shape != mask_b.shape:
            raise ValueError("validity masks must match the panorama HxW shape")
        if mask_a.dtype != np.bool_ or mask_b.dtype != np.bool_:
            raise TypeError("validity masks must be explicit boolean arrays")

        total_started = perf_counter()
        started = perf_counter()
        keypoints_a, keypoints_b = self._detector.detect_batch(
            (first, second),
            panorama_ids=panorama_ids,
            validity_masks=(mask_a, mask_b),
        )
        detection_seconds = perf_counter() - started

        started = perf_counter()
        patches_a = self._patch_provider.materialize(
            first, keypoints_a, self.patch_request, validity_mask=mask_a
        )
        patches_b = self._patch_provider.materialize(
            second, keypoints_b, self.patch_request, validity_mask=mask_b
        )
        patches_seconds = perf_counter() - started

        started = perf_counter()
        described_a = self._descriptor.describe(
            patches_a,
            responses=np.asarray(
                [keypoint.response for keypoint in keypoints_a.keypoints],
                dtype=np.float64,
            ),
        )
        described_b = self._descriptor.describe(
            patches_b,
            responses=np.asarray(
                [keypoint.response for keypoint in keypoints_b.keypoints],
                dtype=np.float64,
            ),
        )
        features_a = _materialize_feature_set(
            keypoints_a, patches_a, described_a, extractor_config=self.configuration
        )
        features_b = _materialize_feature_set(
            keypoints_b, patches_b, described_b, extractor_config=self.configuration
        )
        descriptor_seconds = perf_counter() - started

        started = perf_counter()
        matches = self._matcher.match(features_a, features_b)
        matches.provenance = replace(
            matches.provenance,
            calibration_id=self.calibration_id,
        )
        matching_seconds = perf_counter() - started
        return FrontendResult(
            matches=matches,
            keypoint_counts=(len(keypoints_a.keypoints), len(keypoints_b.keypoints)),
            valid_fractions=(float(mask_a.mean()), float(mask_b.mean())),
            timings=FrontendTimings(
                detection_seconds=detection_seconds,
                patches_seconds=patches_seconds,
                descriptor_seconds=descriptor_seconds,
                matching_seconds=matching_seconds,
                total_seconds=perf_counter() - total_started,
            ),
            configuration=self.configuration,
        )


def _materialize_feature_set(
    keypoints: Any,
    patches: Any,
    described: Any,
    *,
    extractor_config: dict[str, Any],
) -> SphericalFeatureSet:
    interface = "panorai-optimized-public-pair-features/v1"
    features: list[SphericalFeature] = []
    descriptors: list[np.ndarray] = []
    for described_index, keypoint_index in enumerate(
        described.physical_keypoint_ids.tolist()
    ):
        keypoint = keypoints.keypoints[keypoint_index]
        patch = patches.patches[int(described.patch_indices[described_index])]
        output_index = len(features)
        face_id = f"tangent-keypoint-{output_index:06d}"
        provenance = FeatureProvenance(
            interface=interface,
            source_panorama_checksum=keypoints.source_checksum,
            projection_backend="panorai.geometry",
            projection_backend_version=keypoints.projection_backend_version,
            face_id=face_id,
            generating_commit=keypoints.generating_commit,
            selection_reason="spherical-dog-batch-then-parallel-tangent-rootsift",
        )
        centre = (patches.request.output_shape_hw[0] - 1.0) / 2.0
        features.append(
            SphericalFeature(
                feature_id=f"{keypoints.panorama_id}:feature-{output_index:06d}",
                panorama_id=keypoints.panorama_id,
                face_id=face_id,
                pixel_xy=np.asarray((centre, centre), dtype=np.float64),
                source_erp_xy=np.asarray(keypoint.source_erp_xy).copy(),
                bearing_xyz=np.asarray(keypoint.bearing_xyz).copy(),
                response=float(keypoint.response),
                scale=float(keypoint.scale_deg),
                angle_deg=float(described.angles_deg[described_index]),
                octave=int(keypoint.octave),
                descriptor_index=output_index,
                valid=True,
                projection_spec=patch.geometry.projection_spec,
                provenance=provenance,
            )
        )
        descriptors.append(np.asarray(described.descriptors[described_index]))
    descriptor_array = (
        np.stack(descriptors).astype(np.float32, copy=False)
        if descriptors
        else np.empty((0, described.descriptors.shape[1]), dtype=np.float32)
    )
    return SphericalFeatureSet(
        panorama_id=keypoints.panorama_id,
        features=features,
        descriptors=descriptor_array,
        descriptor_type=described.descriptor_type,
        descriptor_metric=described.descriptor_metric,
        extractor_name=described.extractor_name,
        extractor_config=extractor_config,
        backend_name="opencv",
        backend_version=cv2.__version__,
        face_set_spec=FaceSetSpec(
            sampler="per-keypoint-tangent",
            shape_hw=patches.request.output_shape_hw,
            fov_deg=(1.0, 1.0),
        ),
        projection_backend="panorai.geometry",
        projection_backend_version=keypoints.projection_backend_version,
        panorama_checksum=keypoints.source_checksum,
        generating_commit=keypoints.generating_commit,
        interface=interface,
        stability="experimental",
    )
