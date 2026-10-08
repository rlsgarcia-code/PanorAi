"""Experimental spherical DoG detector with tangent-patch SIFT descriptors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from typing import Any

import numpy as np

from panorai.geometry import erp_pixels_to_rays
from panorai.image_processing import spherical_gaussian_blur, spherical_resize
from panorai.image_processing._sampling import row_chunks, sample_tangent

from ._config import FaceSetSpec, FeatureMatcherConfig
from ._extractor import (
    _array_checksum,
    _as_panorama,
    _panorai_commit,
    _panorai_version,
)
from ._matcher import FeatureMatcher
from ._models import FeatureProvenance, SphericalFeature, SphericalFeatureSet
from ._tangent_patches import (
    OpenCVTangentDescriptor,
    OpenCVTangentDescriptorConfig,
    TangentPatchProvider,
    TangentPatchRequest,
)
from .backends.opencv import OpenCVFeatureBackend

SPHERICAL_DOG_SIFT_INTERFACE = "panorai-spherical-dog-sift/v1"


def _positive_integer(value: int, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_real(
    value: float,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


@dataclass(frozen=True, slots=True)
class SphericalDoGSIFTConfig:
    """Configuration for direct spherical DoG detection and SIFT description.

    ``base_sigma_px`` is measured at each octave's ERP vertical resolution;
    returned feature scales are converted to degrees. ``contrast_threshold``
    is an absolute DoG amplitude after grayscale normalization to ``[0, 1]``.
    """

    octaves: int = 3
    levels_per_octave: int = 3
    base_sigma_px: float = 1.6
    contrast_threshold: float = 0.012
    edge_threshold: float = 10.0
    max_features: int = 1000
    patch_size: int = 48
    descriptor_radius_sigmas: float = 6.0
    descriptor_keypoint_size_fraction: float = 0.25
    orientation_bins: int = 36
    minimum_valid_fraction: float = 1.0
    minimum_valid_support_fraction: float = 0.99
    angular_dedup_threshold_deg: float = 0.12
    scale_dedup_log2: float = 0.5
    selection_policy: str = "equal-area-round-robin"
    selection_grid_shape: tuple[int, int] = (12, 24)
    root_sift: bool = False
    convolution_backend: str = "auto"

    def __post_init__(self) -> None:
        object.__setattr__(self, "octaves", _positive_integer(self.octaves, "octaves"))
        object.__setattr__(
            self,
            "levels_per_octave",
            _positive_integer(self.levels_per_octave, "levels_per_octave", minimum=2),
        )
        object.__setattr__(
            self,
            "base_sigma_px",
            _finite_real(self.base_sigma_px, "base_sigma_px", minimum=0.5),
        )
        object.__setattr__(
            self,
            "contrast_threshold",
            _finite_real(self.contrast_threshold, "contrast_threshold", minimum=0.0),
        )
        object.__setattr__(
            self,
            "edge_threshold",
            _finite_real(self.edge_threshold, "edge_threshold", minimum=1.0),
        )
        object.__setattr__(
            self,
            "max_features",
            _positive_integer(self.max_features, "max_features"),
        )
        patch_size = _positive_integer(self.patch_size, "patch_size", minimum=24)
        if patch_size % 2:
            raise ValueError(
                "patch_size must be even so its ray lies at the patch centre"
            )
        object.__setattr__(self, "patch_size", patch_size)
        object.__setattr__(
            self,
            "descriptor_radius_sigmas",
            _finite_real(
                self.descriptor_radius_sigmas,
                "descriptor_radius_sigmas",
                minimum=2.0,
            ),
        )
        object.__setattr__(
            self,
            "descriptor_keypoint_size_fraction",
            _finite_real(
                self.descriptor_keypoint_size_fraction,
                "descriptor_keypoint_size_fraction",
                minimum=0.05,
                maximum=0.75,
            ),
        )
        object.__setattr__(
            self,
            "orientation_bins",
            _positive_integer(self.orientation_bins, "orientation_bins", minimum=8),
        )
        object.__setattr__(
            self,
            "minimum_valid_fraction",
            _finite_real(
                self.minimum_valid_fraction,
                "minimum_valid_fraction",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "minimum_valid_support_fraction",
            _finite_real(
                self.minimum_valid_support_fraction,
                "minimum_valid_support_fraction",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "angular_dedup_threshold_deg",
            _finite_real(
                self.angular_dedup_threshold_deg,
                "angular_dedup_threshold_deg",
                minimum=0.0,
                maximum=180.0,
            ),
        )
        object.__setattr__(
            self,
            "scale_dedup_log2",
            _finite_real(self.scale_dedup_log2, "scale_dedup_log2", minimum=0.0),
        )
        policy = str(self.selection_policy).strip().lower()
        if policy not in {"response", "equal-area-round-robin"}:
            raise ValueError(
                "selection_policy must be 'response' or 'equal-area-round-robin'"
            )
        object.__setattr__(self, "selection_policy", policy)
        if (
            not isinstance(self.selection_grid_shape, tuple)
            or len(self.selection_grid_shape) != 2
        ):
            raise TypeError("selection_grid_shape must be a two-integer tuple")
        object.__setattr__(
            self,
            "selection_grid_shape",
            tuple(
                _positive_integer(item, "selection_grid_shape item")
                for item in self.selection_grid_shape
            ),
        )
        if not isinstance(self.root_sift, bool):
            raise TypeError("root_sift must be a boolean")
        if self.convolution_backend not in {"auto", "numpy", "native"}:
            raise ValueError("convolution_backend must be 'auto', 'numpy', or 'native'")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            {
                "interface": SPHERICAL_DOG_SIFT_INTERFACE,
                "stability": "experimental",
                "scale_units": "degrees",
                "descriptor": "opencv-sift",
                "detector_interface": "panorai-spherical-dog-detector/v2",
                "patch_interface": "panorai-tangent-patches/v1",
                "descriptor_adapter_interface": (
                    "panorai-tangent-opencv-descriptor/v1"
                ),
            }
        )
        return result


@dataclass(frozen=True, slots=True)
class _Candidate:
    source_xy: np.ndarray
    bearing: np.ndarray
    response: float
    sigma_deg: float
    octave: int
    level: int


class SphericalDoGSIFTExtractor:
    """Detect DoG extrema on the sphere and describe local tangent patches."""

    def __init__(
        self,
        config: SphericalDoGSIFTConfig | None = None,
        *,
        backend: OpenCVFeatureBackend | None = None,
        minimum_opencv_version: str = "4.9.0",
    ) -> None:
        self.config = config or SphericalDoGSIFTConfig()
        self.backend = backend or OpenCVFeatureBackend()
        self.minimum_opencv_version = minimum_opencv_version

    def extract(
        self,
        panorama: Any,
        *,
        panorama_id: str | None = None,
        validity_mask: np.ndarray | None = None,
    ) -> SphericalFeatureSet:
        """Detect keypoints with the public detector and describe them with SIFT."""

        keypoints = self.detect(
            panorama,
            panorama_id=panorama_id,
            validity_mask=validity_mask,
        )
        return self.describe_keypoints(
            panorama,
            keypoints,
            validity_mask=validity_mask,
        )

    def detect(
        self,
        panorama: Any,
        *,
        panorama_id: str | None = None,
        validity_mask: np.ndarray | None = None,
    ):
        """Return descriptor-free keypoints from the corrected spherical detector."""

        # Local import avoids a module-initialization cycle: the detector reuses
        # the tangent-neighbour DoG extrema kernel defined in this module.
        from ._spherical_detector import (
            SphericalDoGDetector,
            SphericalDoGDetectorConfig,
        )

        detector_config = SphericalDoGDetectorConfig(
            octaves=self.config.octaves,
            levels_per_octave=self.config.levels_per_octave,
            base_sigma_px=self.config.base_sigma_px,
            contrast_threshold=self.config.contrast_threshold,
            edge_threshold=self.config.edge_threshold,
            max_keypoints=self.config.max_features,
            minimum_valid_support_fraction=(self.config.minimum_valid_support_fraction),
            angular_dedup_threshold_deg=(self.config.angular_dedup_threshold_deg),
            scale_dedup_log2=self.config.scale_dedup_log2,
            selection_policy=self.config.selection_policy,
            selection_grid_shape=self.config.selection_grid_shape,
            convolution_backend=self.config.convolution_backend,
        )
        return SphericalDoGDetector(detector_config).detect(
            panorama,
            panorama_id=panorama_id,
            validity_mask=validity_mask,
        )

    def describe_keypoints(
        self,
        panorama: Any,
        keypoints: Any,
        *,
        validity_mask: np.ndarray | None = None,
    ) -> SphericalFeatureSet:
        """Describe a :class:`SphericalKeypointSet` on tangent SIFT patches."""

        self.backend.require_version(self.minimum_opencv_version)
        wrapped = _as_panorama(panorama)
        if not isinstance(wrapped.image, np.ndarray):
            raise TypeError("spherical DoG-SIFT currently requires a NumPy panorama")
        image = wrapped.image
        if image.ndim not in {2, 3} or image.shape[0] < 8 or image.shape[1] < 16:
            raise ValueError("panorama must use HW/HWC layout and be at least 8x16")
        checksum = _array_checksum(image)
        if tuple(keypoints.source_shape_hw) != tuple(image.shape[:2]):
            raise ValueError("keypoints and panorama must have the same source shape")
        if keypoints.source_checksum != checksum:
            raise ValueError("keypoints were not detected on this panorama")
        panorama_id = keypoints.panorama_id
        candidates = [
            _Candidate(
                source_xy=item.source_erp_xy,
                bearing=item.bearing_xyz,
                response=item.response,
                sigma_deg=item.scale_deg,
                octave=item.octave,
                level=item.level,
            )
            for item in keypoints.keypoints
        ]
        patch_request = TangentPatchRequest(
            output_shape_hw=(self.config.patch_size, self.config.patch_size),
            radius_in_scales=self.config.descriptor_radius_sigmas,
            minimum_fov_deg=1.0,
            maximum_fov_deg=120.0,
            interpolation="bilinear",
            invalid_policy="propagate",
            minimum_valid_fraction=self.config.minimum_valid_fraction,
            orientation_policy="upright",
        )
        patch_set = TangentPatchProvider().materialize(
            panorama,
            keypoints,
            patch_request,
            validity_mask=validity_mask,
        )
        descriptor_config = OpenCVTangentDescriptorConfig(
            method="sift",
            keypoint_diameter_in_scales=(
                2.0
                * self.config.descriptor_radius_sigmas
                * self.config.descriptor_keypoint_size_fraction
            ),
            orientation_policy="dominant-gradient",
            orientation_bins=self.config.orientation_bins,
            root_sift=self.config.root_sift,
            minimum_opencv_version=self.minimum_opencv_version,
        )
        described_set = OpenCVTangentDescriptor(
            descriptor_config, backend=self.backend
        ).describe(
            patch_set,
            responses=np.asarray(
                [item.response for item in candidates], dtype=np.float64
            ),
        )
        features: list[SphericalFeature] = []
        descriptors: list[np.ndarray] = []
        for described_index, candidate_index in enumerate(
            described_set.patch_indices.tolist()
        ):
            candidate = candidates[candidate_index]
            patch = patch_set.patches[candidate_index]
            descriptor = described_set.descriptors[described_index]
            angle_deg = float(described_set.angles_deg[described_index])
            spec = patch.geometry.projection_spec
            output_index = len(features)
            face_id = f"tangent-keypoint-{output_index:06d}"
            provenance = FeatureProvenance(
                interface=SPHERICAL_DOG_SIFT_INTERFACE,
                source_panorama_checksum=checksum,
                projection_backend="panorai.geometry",
                projection_backend_version=_panorai_version(),
                face_id=face_id,
                generating_commit=_panorai_commit(),
                selection_reason=(
                    f"spherical-dog-{self.config.selection_policy}-then-tangent-sift"
                ),
            )
            centre = (self.config.patch_size - 1.0) / 2.0
            features.append(
                SphericalFeature(
                    feature_id=f"{panorama_id}:feature-{output_index:06d}",
                    panorama_id=panorama_id,
                    face_id=face_id,
                    pixel_xy=np.asarray((centre, centre), dtype=np.float64),
                    source_erp_xy=candidate.source_xy.copy(),
                    bearing_xyz=candidate.bearing.copy(),
                    response=candidate.response,
                    scale=candidate.sigma_deg,
                    angle_deg=angle_deg,
                    octave=candidate.octave,
                    descriptor_index=output_index,
                    valid=True,
                    projection_spec=spec,
                    provenance=provenance,
                )
            )
            descriptors.append(descriptor)
            if len(features) >= self.config.max_features:
                break
        descriptor_array = (
            np.stack(descriptors).astype(np.float32, copy=False)
            if descriptors
            else np.empty((0, described_set.descriptors.shape[1]), dtype=np.float32)
        )
        if self.config.root_sift:
            descriptor_type = "root-sift-float32"
            extractor_name = "spherical-dog+root-sift"
        else:
            descriptor_type = "spherical-dog-sift-float32"
            extractor_name = "spherical-dog+opencv-sift"
        base_fov = _patch_fov_deg(
            180.0 / image.shape[0] * self.config.base_sigma_px, self.config
        )
        return SphericalFeatureSet(
            panorama_id=panorama_id,
            features=features,
            descriptors=descriptor_array,
            descriptor_type=descriptor_type,
            descriptor_metric="l2",
            extractor_name=extractor_name,
            extractor_config=self.config.to_dict(),
            backend_name=self.backend.name,
            backend_version=self.backend.version,
            face_set_spec=FaceSetSpec(
                sampler="per-keypoint-tangent",
                shape_hw=(self.config.patch_size, self.config.patch_size),
                fov_deg=(base_fov, base_fov),
            ),
            projection_backend="panorai.geometry",
            projection_backend_version=_panorai_version(),
            panorama_checksum=checksum,
            generating_commit=_panorai_commit(),
            interface=SPHERICAL_DOG_SIFT_INTERFACE,
            stability="experimental",
        )

    def _detect(self, gray: np.ndarray, validity: np.ndarray) -> list[_Candidate]:
        original_shape = gray.shape
        candidates: list[_Candidate] = []
        octave_image = gray
        for octave in range(self.config.octaves):
            if min(octave_image.shape) < 8:
                break
            gaussians = _gaussian_levels(octave_image, self.config)
            dogs = [
                gaussians[index + 1].astype(np.float64)
                - gaussians[index].astype(np.float64)
                for index in range(len(gaussians) - 1)
            ]
            step = math.pi / octave_image.shape[0]
            for level in range(1, len(dogs) - 1):
                extrema = _dog_extrema_mask(
                    dogs[level - 1],
                    dogs[level],
                    dogs[level + 1],
                    step,
                    self.config.contrast_threshold,
                    self.config.edge_threshold,
                )
                yy, xx = np.nonzero(extrema)
                if not len(xx):
                    continue
                source_x = (xx + 0.5) * original_shape[1] / octave_image.shape[1] - 0.5
                source_y = (yy + 0.5) * original_shape[0] / octave_image.shape[0] - 0.5
                valid_x = np.floor(source_x + 0.5).astype(np.int64) % original_shape[1]
                valid_y = np.clip(
                    np.floor(source_y + 0.5).astype(np.int64), 0, original_shape[0] - 1
                )
                keep = validity[valid_y, valid_x]
                if not keep.any():
                    continue
                source_pixels = np.stack((source_x[keep], source_y[keep]), axis=1)
                bearings = erp_pixels_to_rays(source_pixels, original_shape)
                sigma_px = self.config.base_sigma_px * 2.0 ** (
                    (level + 1) / self.config.levels_per_octave
                )
                sigma_deg = 180.0 / octave_image.shape[0] * sigma_px
                responses = np.abs(dogs[level][yy[keep], xx[keep]])
                for source_xy, bearing, response in zip(
                    source_pixels, bearings, responses, strict=True
                ):
                    candidates.append(
                        _Candidate(
                            source_xy=np.asarray(source_xy, dtype=np.float64),
                            bearing=np.asarray(bearing, dtype=np.float64),
                            response=float(response),
                            sigma_deg=float(sigma_deg),
                            octave=octave,
                            level=level,
                        )
                    )
            if octave + 1 < self.config.octaves:
                next_shape = (
                    max(2, (octave_image.shape[0] + 1) // 2),
                    max(2, (octave_image.shape[1] + 1) // 2),
                )
                octave_image = spherical_resize(
                    gaussians[self.config.levels_per_octave], next_shape
                ).astype(np.float32, copy=False)
        return candidates


class SphericalDoGSIFTPipeline:
    """Experimental direct spherical detector plus existing L2 matcher."""

    def __init__(
        self,
        detector_config: SphericalDoGSIFTConfig | None = None,
        matcher_config: FeatureMatcherConfig | None = None,
        *,
        backend: OpenCVFeatureBackend | None = None,
    ) -> None:
        self.detector_config = detector_config or SphericalDoGSIFTConfig()
        self.matcher_config = matcher_config or FeatureMatcherConfig(method="flann")
        self.backend = backend or OpenCVFeatureBackend()
        self.extractor = SphericalDoGSIFTExtractor(
            self.detector_config, backend=self.backend
        )
        self.matcher = FeatureMatcher(self.matcher_config, backend=self.backend)

    def extract(self, panorama: Any, **kwargs: Any) -> SphericalFeatureSet:
        return self.extractor.extract(panorama, **kwargs)

    def match(self, features_a: SphericalFeatureSet, features_b: SphericalFeatureSet):
        return self.matcher.match(features_a, features_b)

    def extract_and_match(
        self,
        panorama_a: Any,
        panorama_b: Any,
        *,
        panorama_id_a: str | None = None,
        panorama_id_b: str | None = None,
        validity_mask_a: np.ndarray | None = None,
        validity_mask_b: np.ndarray | None = None,
    ):
        features_a = self.extract(
            panorama_a,
            panorama_id=panorama_id_a,
            validity_mask=validity_mask_a,
        )
        features_b = self.extract(
            panorama_b,
            panorama_id=panorama_id_b,
            validity_mask=validity_mask_b,
        )
        return self.match(features_a, features_b)

    def describe(self) -> dict[str, Any]:
        return {
            "interface": SPHERICAL_DOG_SIFT_INTERFACE,
            "stability": "experimental",
            "detector": self.detector_config.to_dict(),
            "matcher": self.matcher_config.to_dict(),
            "backend": {"name": self.backend.name, "version": self.backend.version},
        }


def _gaussian_levels(
    image: np.ndarray, config: SphericalDoGSIFTConfig
) -> list[np.ndarray]:
    result = []
    for level in range(config.levels_per_octave + 3):
        sigma = config.base_sigma_px * 2.0 ** (level / config.levels_per_octave)
        radius = max(1, int(math.ceil(3.0 * sigma)))
        result.append(
            spherical_gaussian_blur(
                image,
                ksize=2 * radius + 1,
                sigma=sigma,
                backend=config.convolution_backend,
            )
        )
    return result


def _dog_extrema_mask(
    previous: np.ndarray,
    current: np.ndarray,
    following: np.ndarray,
    step: float,
    contrast_threshold: float,
    edge_threshold: float,
) -> np.ndarray:
    height, width = current.shape
    result = np.zeros(current.shape, dtype=bool)
    edge_limit = (edge_threshold + 1.0) ** 2 / edge_threshold
    for rows in row_chunks(height, width):
        center = current[rows]
        maxima = center >= contrast_threshold
        minima = center <= -contrast_threshold
        neighbours: dict[tuple[int, int], np.ndarray] = {}
        for scale_index, scale in enumerate((previous, current, following)):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if scale_index == 1 and dx == 0 and dy == 0:
                        continue
                    sample = sample_tangent(scale, rows, dx * step, -dy * step)
                    maxima &= center > sample
                    minima &= center < sample
                    if scale_index == 1:
                        neighbours[(dx, dy)] = sample
        dxx = neighbours[(1, 0)] + neighbours[(-1, 0)] - 2.0 * center
        dyy = neighbours[(0, 1)] + neighbours[(0, -1)] - 2.0 * center
        dxy = (
            neighbours[(1, 1)]
            - neighbours[(-1, 1)]
            - neighbours[(1, -1)]
            + neighbours[(-1, -1)]
        ) / 4.0
        determinant = dxx * dyy - dxy * dxy
        trace = dxx + dyy
        not_edge = (determinant > np.finfo(np.float64).eps) & (
            trace * trace < edge_limit * determinant
        )
        result[rows] = (maxima | minima) & not_edge
    return result


def _deduplicate_candidates(
    candidates: list[_Candidate], config: SphericalDoGSIFTConfig
) -> list[_Candidate]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            -item.response,
            item.octave,
            item.level,
            float(item.source_xy[1]),
            float(item.source_xy[0]),
        ),
    )
    threshold_cosine = math.cos(math.radians(config.angular_dedup_threshold_deg))
    selected: list[_Candidate] = []
    for candidate in ordered:
        duplicate = any(
            float(candidate.bearing @ other.bearing) >= threshold_cosine - 1e-15
            and abs(math.log2(candidate.sigma_deg / other.sigma_deg))
            <= config.scale_dedup_log2
            for other in selected
        )
        if not duplicate:
            selected.append(candidate)
        if len(selected) >= config.max_features * 2:
            break
    return selected


def _patch_fov_deg(sigma_deg: float, config: SphericalDoGSIFTConfig) -> float:
    return min(120.0, max(1.0, 2.0 * config.descriptor_radius_sigmas * sigma_deg))
