"""Experimental descriptor-agnostic keypoint detection on the sphere."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import math
from numbers import Integral, Real
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter  # type: ignore[import-untyped]
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

from panorai.geometry import erp_pixels_to_rays, rays_to_erp_pixels
from panorai.image_processing import spherical_gaussian_blur, spherical_resize
from panorai.image_processing._sampling import sample_rays

from ._extractor import (
    _array_checksum,
    _as_panorama,
    _opencv_image,
    _panorai_commit,
    _panorai_version,
)
from ._spherical_dog import _dog_extrema_mask

SPHERICAL_DOG_DETECTOR_INTERFACE = "panorai-spherical-dog-detector/v1"
SPHERICAL_COARSE_DOG_DETECTOR_INTERFACE = "panorai-spherical-coarse-dog-detector/v1"


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
class SphericalDoGDetectorConfig:
    """Configuration for descriptor-independent spherical DoG detection."""

    octaves: int = 3
    levels_per_octave: int = 3
    base_sigma_px: float = 1.6
    contrast_threshold: float = 0.012
    edge_threshold: float = 10.0
    max_keypoints: int = 1000
    minimum_valid_support_fraction: float = 0.99
    angular_dedup_threshold_deg: float = 0.12
    scale_dedup_log2: float = 0.5
    selection_policy: str = "equal-area-round-robin"
    selection_grid_shape: tuple[int, int] = (12, 24)
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
            "max_keypoints",
            _positive_integer(self.max_keypoints, "max_keypoints"),
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
        grid = tuple(
            _positive_integer(item, "selection_grid_shape item")
            for item in self.selection_grid_shape
        )
        object.__setattr__(self, "selection_grid_shape", grid)
        backend = str(self.convolution_backend).strip().lower()
        if backend not in {"auto", "numpy", "native"}:
            raise ValueError("convolution_backend must be 'auto', 'numpy', or 'native'")
        object.__setattr__(self, "convolution_backend", backend)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            {
                "interface": SPHERICAL_DOG_DETECTOR_INTERFACE,
                "stability": "experimental",
                "scale_units": "degrees",
                "descriptor": None,
            }
        )
        return result


@dataclass(frozen=True, slots=True)
class SphericalCoarseDoGDetectorConfig(SphericalDoGDetectorConfig):
    """CPU proposal detector evaluated on a reduced spherical raster.

    The output bearings, pixels, checksum, and shape are promoted back to the
    source panorama.  Only the proposal scale space is reduced; descriptor
    extraction can therefore continue on the original panorama without a
    geometry adapter or a second back-projection contract.
    """

    octaves: int = 3
    levels_per_octave: int = 3
    base_sigma_px: float = 0.5
    contrast_threshold: float = 0.012
    edge_threshold: float = 10.0
    max_keypoints: int = 4096
    minimum_valid_support_fraction: float = 0.97
    angular_dedup_threshold_deg: float = 0.12
    scale_dedup_log2: float = 0.5
    selection_policy: str = "equal-area-round-robin"
    selection_grid_shape: tuple[int, int] = (12, 24)
    convolution_backend: str = "auto"
    proposal_height: int = 512
    proposal_resampling: str = "gaussian"
    proposal_prefilter_sigma_px: float = 2.0
    proposal_prefilter_intermediate_height: int = 1024
    fine_verification: str = "none"
    fine_candidate_multiplier: float = 2.0
    fine_patch_size: int = 15
    fine_samples_per_sigma: float = 2.0
    fine_search_radius_samples: int = 2

    def __post_init__(self) -> None:
        super(SphericalCoarseDoGDetectorConfig, self).__post_init__()
        object.__setattr__(
            self,
            "proposal_height",
            _positive_integer(self.proposal_height, "proposal_height", minimum=64),
        )
        resampling = str(self.proposal_resampling).strip().lower()
        if resampling not in {
            "area",
            "bilinear",
            "gaussian",
            "spherical-gaussian",
        }:
            raise ValueError(
                "proposal_resampling must be 'area', 'bilinear', 'gaussian', "
                "or 'spherical-gaussian'"
            )
        object.__setattr__(self, "proposal_resampling", resampling)
        object.__setattr__(
            self,
            "proposal_prefilter_sigma_px",
            _finite_real(
                self.proposal_prefilter_sigma_px,
                "proposal_prefilter_sigma_px",
                minimum=0.5,
            ),
        )
        object.__setattr__(
            self,
            "proposal_prefilter_intermediate_height",
            _positive_integer(
                self.proposal_prefilter_intermediate_height,
                "proposal_prefilter_intermediate_height",
                minimum=64,
            ),
        )
        verification = str(self.fine_verification).strip().lower()
        if verification not in {"none", "tangent-dog"}:
            raise ValueError("fine_verification must be 'none' or 'tangent-dog'")
        object.__setattr__(self, "fine_verification", verification)
        object.__setattr__(
            self,
            "fine_candidate_multiplier",
            _finite_real(
                self.fine_candidate_multiplier,
                "fine_candidate_multiplier",
                minimum=1.0,
            ),
        )
        patch_size = _positive_integer(
            self.fine_patch_size, "fine_patch_size", minimum=9
        )
        if patch_size % 2 == 0:
            raise ValueError("fine_patch_size must be odd")
        object.__setattr__(self, "fine_patch_size", patch_size)
        object.__setattr__(
            self,
            "fine_samples_per_sigma",
            _finite_real(
                self.fine_samples_per_sigma,
                "fine_samples_per_sigma",
                minimum=1.0,
            ),
        )
        search_radius = _positive_integer(
            self.fine_search_radius_samples,
            "fine_search_radius_samples",
            minimum=1,
        )
        if 2 * search_radius + 5 > patch_size:
            raise ValueError(
                "fine_patch_size is too small for fine_search_radius_samples"
            )
        object.__setattr__(self, "fine_search_radius_samples", search_radius)

    def proposal_config(self) -> SphericalDoGDetectorConfig:
        base_names = {item.name for item in fields(SphericalDoGDetectorConfig)}
        values = {name: getattr(self, name) for name in base_names}
        if self.fine_verification != "none":
            values["max_keypoints"] = int(
                math.ceil(self.max_keypoints * self.fine_candidate_multiplier)
            )
        return SphericalDoGDetectorConfig(**values)

    def to_dict(self) -> dict[str, Any]:
        result = super(SphericalCoarseDoGDetectorConfig, self).to_dict()
        result.update(
            {
                "interface": SPHERICAL_COARSE_DOG_DETECTOR_INTERFACE,
                "proposal_shape_policy": "(proposal_height, 2 * proposal_height)",
                "promotion": "bearing-preserving-to-source-erp",
                "proposal_resampling": self.proposal_resampling,
                "area_weighting": "source-pixel solid angle (cos(latitude))",
                "proposal_prefilter_sigma_px": self.proposal_prefilter_sigma_px,
                "proposal_prefilter_intermediate_height": (
                    self.proposal_prefilter_intermediate_height
                ),
                "gaussian_boundary": "longitude-wrap/pole-reflect-half-turn",
                "gaussian_support_policy": "support-normalized",
                "fine_verification": (
                    None if self.fine_verification == "none" else self.fine_verification
                ),
                "fine_candidate_multiplier": self.fine_candidate_multiplier,
                "fine_patch_size": self.fine_patch_size,
                "fine_samples_per_sigma": self.fine_samples_per_sigma,
                "fine_search_radius_samples": self.fine_search_radius_samples,
            }
        )
        return result


@dataclass(frozen=True, slots=True)
class SphericalKeypoint:
    """One descriptor-independent spherical keypoint."""

    source_erp_xy: np.ndarray
    bearing_xyz: np.ndarray
    response: float
    scale_deg: float
    octave: int
    level: int
    valid_support_fraction: float

    def __post_init__(self) -> None:
        pixel = np.asarray(self.source_erp_xy, dtype=np.float64)
        bearing = np.asarray(self.bearing_xyz, dtype=np.float64)
        if pixel.shape != (2,) or not np.isfinite(pixel).all():
            raise ValueError("source_erp_xy must be a finite length-2 vector")
        if bearing.shape != (3,) or not np.isfinite(bearing).all():
            raise ValueError("bearing_xyz must be a finite length-3 vector")
        norm = float(np.linalg.norm(bearing))
        if norm <= 0.0:
            raise ValueError("bearing_xyz must be non-zero")
        object.__setattr__(self, "source_erp_xy", pixel)
        object.__setattr__(self, "bearing_xyz", bearing / norm)
        object.__setattr__(
            self, "response", _finite_real(self.response, "response", minimum=0.0)
        )
        object.__setattr__(
            self,
            "scale_deg",
            _finite_real(
                self.scale_deg, "scale_deg", minimum=float(np.finfo(float).tiny)
            ),
        )
        object.__setattr__(
            self, "octave", _positive_integer(self.octave + 1, "octave+1") - 1
        )
        object.__setattr__(
            self, "level", _positive_integer(self.level + 1, "level+1") - 1
        )
        object.__setattr__(
            self,
            "valid_support_fraction",
            _finite_real(
                self.valid_support_fraction,
                "valid_support_fraction",
                minimum=0.0,
                maximum=1.0,
            ),
        )


@dataclass(frozen=True, slots=True)
class SphericalOctaveDetectionDiagnostics:
    """Candidate counts for one octave before and after validity filtering."""

    octave: int
    shape_hw: tuple[int, int]
    raw_extrema_count: int
    valid_support_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SphericalDetectionDiagnostics:
    """Stage counts for descriptor-free spherical detection."""

    raw_extrema_count: int
    valid_support_count: int
    unique_after_deduplication: int
    output_after_budget: int
    octaves: tuple[SphericalOctaveDetectionDiagnostics, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "octaves": [item.to_dict() for item in self.octaves],
        }


@dataclass(frozen=True, slots=True)
class SphericalKeypointSet:
    """Descriptor-free spherical detections with explicit provenance."""

    panorama_id: str
    keypoints: tuple[SphericalKeypoint, ...]
    source_shape_hw: tuple[int, int]
    source_checksum: str
    config: SphericalDoGDetectorConfig | SphericalCoarseDoGDetectorConfig
    generating_commit: str | None
    projection_backend_version: str
    diagnostics: SphericalDetectionDiagnostics
    interface: str = SPHERICAL_DOG_DETECTOR_INTERFACE
    stability: str = "experimental"

    def __len__(self) -> int:
        return len(self.keypoints)

    @property
    def bearings(self) -> np.ndarray:
        return (
            np.stack([item.bearing_xyz for item in self.keypoints])
            if self.keypoints
            else np.empty((0, 3), dtype=np.float64)
        )

    @property
    def source_erp_xy(self) -> np.ndarray:
        return (
            np.stack([item.source_erp_xy for item in self.keypoints])
            if self.keypoints
            else np.empty((0, 2), dtype=np.float64)
        )

    @property
    def responses(self) -> np.ndarray:
        return np.asarray([item.response for item in self.keypoints], dtype=np.float64)

    @property
    def scales_deg(self) -> np.ndarray:
        return np.asarray([item.scale_deg for item in self.keypoints], dtype=np.float64)

    @property
    def valid_support_fractions(self) -> np.ndarray:
        return np.asarray(
            [item.valid_support_fraction for item in self.keypoints], dtype=np.float64
        )

    @property
    def octaves(self) -> np.ndarray:
        return np.asarray([item.octave for item in self.keypoints], dtype=np.int32)

    @property
    def levels(self) -> np.ndarray:
        return np.asarray([item.level for item in self.keypoints], dtype=np.int32)

    def describe(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "stability": self.stability,
            "panorama_id": self.panorama_id,
            "keypoint_count": len(self),
            "source_shape_hw": self.source_shape_hw,
            "source_checksum": self.source_checksum,
            "config": self.config.to_dict(),
            "generating_commit": self.generating_commit,
            "projection_backend_version": self.projection_backend_version,
            "diagnostics": self.diagnostics.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class _Candidate:
    source_xy: np.ndarray
    bearing: np.ndarray
    response: float
    sigma_deg: float
    octave: int
    level: int
    valid_support_fraction: float


class SphericalDoGDetector:
    """Detect keypoints in a spherical DoG scale space without descriptors."""

    def __init__(self, config: SphericalDoGDetectorConfig | None = None) -> None:
        self.config = config or SphericalDoGDetectorConfig()

    def detect(
        self,
        panorama: Any,
        *,
        panorama_id: str | None = None,
        validity_mask: np.ndarray | None = None,
    ) -> SphericalKeypointSet:
        wrapped = _as_panorama(panorama)
        if not isinstance(wrapped.image, np.ndarray):
            raise TypeError("SphericalDoGDetector currently requires a NumPy panorama")
        image = wrapped.image
        if image.ndim not in {2, 3} or image.shape[0] < 8 or image.shape[1] < 16:
            raise ValueError("panorama must use HW/HWC layout and be at least 8x16")
        validity = np.asarray(wrapped.validity("image"), dtype=bool)
        if wrapped.support_mask is not None:
            validity &= np.asarray(wrapped.support_mask, dtype=bool)
        if validity_mask is not None:
            if (
                not isinstance(validity_mask, np.ndarray)
                or validity_mask.dtype != np.bool_
            ):
                raise TypeError("validity_mask must be a boolean NumPy array")
            if validity_mask.shape != image.shape[:2]:
                raise ValueError(
                    f"validity_mask must have shape {image.shape[:2]}; got {validity_mask.shape}"
                )
            validity &= validity_mask
        gray = _opencv_image(image, validity).astype(np.float32) / 255.0
        checksum = _array_checksum(image)
        resolved_id = panorama_id or f"panorama-{checksum[:16]}"
        candidates, octave_diagnostics = self._detect(gray, validity)
        selected, unique_count = _select_candidates(candidates, self.config)
        keypoints = tuple(
            SphericalKeypoint(
                source_erp_xy=item.source_xy,
                bearing_xyz=item.bearing,
                response=item.response,
                scale_deg=item.sigma_deg,
                octave=item.octave,
                level=item.level,
                valid_support_fraction=item.valid_support_fraction,
            )
            for item in selected
        )
        return SphericalKeypointSet(
            panorama_id=resolved_id,
            keypoints=keypoints,
            source_shape_hw=tuple(image.shape[:2]),
            source_checksum=checksum,
            config=self.config,
            generating_commit=_panorai_commit(),
            projection_backend_version=_panorai_version(),
            diagnostics=SphericalDetectionDiagnostics(
                raw_extrema_count=sum(
                    item.raw_extrema_count for item in octave_diagnostics
                ),
                valid_support_count=len(candidates),
                unique_after_deduplication=unique_count,
                output_after_budget=len(selected),
                octaves=tuple(octave_diagnostics),
            ),
        )

    def _detect(
        self, gray: np.ndarray, validity: np.ndarray
    ) -> tuple[list[_Candidate], list[SphericalOctaveDetectionDiagnostics]]:
        original_shape = gray.shape
        candidates: list[_Candidate] = []
        diagnostics: list[SphericalOctaveDetectionDiagnostics] = []
        octave_image = gray
        octave_validity = validity
        octave_support: np.ndarray = validity.astype(np.float32)
        for octave in range(self.config.octaves):
            if min(octave_image.shape) < 8:
                break
            gaussians, supports = _gaussian_levels_with_support(
                octave_image,
                octave_support,
                self.config,
                base_preblurred=octave > 0,
            )
            dogs = [
                gaussians[index + 1].astype(np.float64)
                - gaussians[index].astype(np.float64)
                for index in range(len(gaussians) - 1)
            ]
            step = math.pi / octave_image.shape[0]
            octave_raw_count = 0
            octave_valid_count = 0
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
                octave_raw_count += int(len(xx))
                if not len(xx):
                    continue
                support = supports[level + 2][yy, xx]
                keep = octave_validity[yy, xx] & (
                    support >= self.config.minimum_valid_support_fraction
                )
                if not keep.any():
                    continue
                xx = xx[keep]
                yy = yy[keep]
                support = support[keep]
                octave_valid_count += int(len(xx))
                source_x = (xx + 0.5) * original_shape[1] / octave_image.shape[1] - 0.5
                source_y = (yy + 0.5) * original_shape[0] / octave_image.shape[0] - 0.5
                source_pixels = np.stack((source_x, source_y), axis=1)
                bearings = erp_pixels_to_rays(source_pixels, original_shape)
                sigma_px = self.config.base_sigma_px * 2.0 ** (
                    (level + 1) / self.config.levels_per_octave
                )
                sigma_deg = 180.0 / octave_image.shape[0] * sigma_px
                responses = np.abs(dogs[level][yy, xx])
                for source_xy, bearing, response, valid_fraction in zip(
                    source_pixels, bearings, responses, support, strict=True
                ):
                    candidates.append(
                        _Candidate(
                            source_xy=np.asarray(source_xy, dtype=np.float64),
                            bearing=np.asarray(bearing, dtype=np.float64),
                            response=float(response),
                            sigma_deg=float(sigma_deg),
                            octave=octave,
                            level=level,
                            valid_support_fraction=float(valid_fraction),
                        )
                    )
            diagnostics.append(
                SphericalOctaveDetectionDiagnostics(
                    octave=octave,
                    shape_hw=tuple(octave_image.shape),
                    raw_extrema_count=octave_raw_count,
                    valid_support_count=octave_valid_count,
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
                octave_validity = (
                    spherical_resize(
                        octave_validity.astype(np.float32),
                        next_shape,
                        interpolation="nearest",
                    )
                    >= 0.5
                )
                octave_support = np.clip(
                    spherical_resize(
                        supports[self.config.levels_per_octave], next_shape
                    ).astype(np.float32, copy=False),
                    0.0,
                    1.0,
                )
        return candidates, diagnostics


class SphericalCoarseDoGDetector:
    """Generate spherical DoG proposals cheaply and promote them to the source ERP.

    This deliberately exposes the coarse proposal stage as a separate
    experimental detector.  It does not claim parity with the dense detector;
    callers can compare the quality/runtime trade-off without changing the
    dense detector's defaults.
    """

    def __init__(self, config: SphericalCoarseDoGDetectorConfig | None = None) -> None:
        self.config = config or SphericalCoarseDoGDetectorConfig()

    def detect(
        self,
        panorama: Any,
        *,
        panorama_id: str | None = None,
        validity_mask: np.ndarray | None = None,
    ) -> SphericalKeypointSet:
        from panorai.data import EquirectangularImage

        wrapped = _as_panorama(panorama)
        if not isinstance(wrapped.image, np.ndarray):
            raise TypeError("SphericalCoarseDoGDetector currently requires NumPy")
        image = wrapped.image
        if image.ndim not in {2, 3} or image.shape[0] < 8 or image.shape[1] < 16:
            raise ValueError("panorama must use HW/HWC layout and be at least 8x16")
        if self.config.proposal_height > image.shape[0]:
            raise ValueError("proposal_height cannot exceed the source height")

        validity = np.asarray(wrapped.validity("image"), dtype=bool)
        if wrapped.support_mask is not None:
            validity &= np.asarray(wrapped.support_mask, dtype=bool)
        if validity_mask is not None:
            if (
                not isinstance(validity_mask, np.ndarray)
                or validity_mask.dtype != np.bool_
            ):
                raise TypeError("validity_mask must be a boolean NumPy array")
            if validity_mask.shape != image.shape[:2]:
                raise ValueError(
                    f"validity_mask must have shape {image.shape[:2]}; got {validity_mask.shape}"
                )
            validity &= validity_mask

        proposal_shape = (self.config.proposal_height, 2 * self.config.proposal_height)
        if self.config.proposal_resampling == "area":
            proposal_image, proposal_validity = _solid_angle_area_downsample(
                image, validity, proposal_shape
            )
        elif self.config.proposal_resampling == "gaussian":
            proposal_image, proposal_validity = _gaussian_antialiased_downsample(
                image,
                validity,
                proposal_shape,
                sigma_px=self.config.proposal_prefilter_sigma_px,
            )
        elif self.config.proposal_resampling == "spherical-gaussian":
            proposal_image, proposal_validity = (
                _spherical_gaussian_antialiased_downsample(
                    image,
                    validity,
                    proposal_shape,
                    sigma_px=self.config.proposal_prefilter_sigma_px,
                    intermediate_height=(
                        self.config.proposal_prefilter_intermediate_height
                    ),
                    backend=self.config.convolution_backend,
                )
            )
        else:
            proposal_image = spherical_resize(image, proposal_shape)
            if np.issubdtype(image.dtype, np.integer):
                limits = np.iinfo(image.dtype)
                proposal_image = np.clip(
                    np.rint(proposal_image), limits.min, limits.max
                ).astype(image.dtype)
            else:
                proposal_image = proposal_image.astype(image.dtype, copy=False)
            proposal_validity = spherical_resize(
                validity.astype(np.uint8), proposal_shape, interpolation="nearest"
            ).astype(bool)
        shadow_angle = float(getattr(wrapped, "shadow_angle", 0.0))
        proposal = EquirectangularImage(
            proposal_image,
            shadow_angle=shadow_angle,
            shadow_padded=shadow_angle > 0.0,
            support_mask=proposal_validity,
        )
        detected = SphericalDoGDetector(self.config.proposal_config()).detect(
            proposal,
            panorama_id=panorama_id,
        )
        if self.config.fine_verification == "tangent-dog":
            detected = _fine_verify_tangent_dog(
                image,
                validity,
                detected,
                self.config,
            )
        source_pixels = rays_to_erp_pixels(
            detected.bearings, tuple(image.shape[:2])
        ).pixels_xy
        promoted = tuple(
            replace(keypoint, source_erp_xy=np.asarray(source_xy, dtype=np.float64))
            for keypoint, source_xy in zip(
                detected.keypoints, source_pixels, strict=True
            )
        )
        return replace(
            detected,
            keypoints=promoted,
            source_shape_hw=tuple(image.shape[:2]),
            source_checksum=_array_checksum(image),
            config=self.config,
            interface=SPHERICAL_COARSE_DOG_DETECTOR_INTERFACE,
        )


def _fine_verify_tangent_dog(
    image: np.ndarray,
    validity: np.ndarray,
    detected: SphericalKeypointSet,
    config: SphericalCoarseDoGDetectorConfig,
) -> SphericalKeypointSet:
    """Re-rank and locally refine coarse proposals on native-resolution samples.

    A compact tangent grid is evaluated for all candidates in vectorized chunks.
    The grid spacing follows each proposal's angular sigma, so this stage never
    builds a second dense source-resolution pyramid or one projector per point.
    """

    if not detected.keypoints:
        return detected
    gray = _opencv_image(image, validity).astype(np.float32) / 255.0
    bearings = detected.bearings
    scales_rad = np.radians(detected.scales_deg)
    sample_count = config.fine_patch_size
    radius = config.fine_search_radius_samples
    coordinates: np.ndarray = (
        np.arange(sample_count, dtype=np.float64) - sample_count // 2
    )
    grid_x, grid_y = np.meshgrid(coordinates, coordinates, indexing="xy")

    up = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    east = np.cross(np.broadcast_to(up, bearings.shape), bearings)
    east_norm = np.linalg.norm(east, axis=1, keepdims=True)
    polar = east_norm[:, 0] < 1e-12
    east[~polar] /= east_norm[~polar]
    east[polar] = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    north = np.cross(bearings, east)

    verified: list[_Candidate] = []
    scale_step = 2.0 ** (1.0 / config.levels_per_octave)
    sigma_samples = config.fine_samples_per_sigma
    gaussian_sigmas = (
        sigma_samples / scale_step,
        sigma_samples,
        sigma_samples * scale_step,
        sigma_samples * scale_step * scale_step,
    )
    center = sample_count // 2
    chunk_size = 1024
    for start in range(0, len(detected), chunk_size):
        stop = min(len(detected), start + chunk_size)
        step = (scales_rad[start:stop] / config.fine_samples_per_sigma)[:, None, None]
        east_offset = grid_x[None, ...] * step
        north_offset = -grid_y[None, ...] * step
        rho = np.hypot(east_offset, north_offset)
        sinc = np.ones_like(rho)
        np.divide(np.sin(rho), rho, out=sinc, where=rho != 0.0)
        tangent = (
            east_offset[..., None] * east[start:stop, None, None, :]
            + north_offset[..., None] * north[start:stop, None, None, :]
        )
        rays = (
            np.cos(rho)[..., None] * bearings[start:stop, None, None, :]
            + sinc[..., None] * tangent
        )
        patches = sample_rays(gray, rays, interpolation="bilinear").astype(
            np.float32, copy=False
        )
        sampled_validity = sample_rays(validity, rays, interpolation="nearest").astype(
            bool, copy=False
        )
        valid_fraction = sampled_validity.mean(axis=(1, 2))

        smoothed = [
            gaussian_filter(
                patches,
                sigma=(0.0, sigma, sigma),
                mode=("nearest", "reflect", "reflect"),
            )
            for sigma in gaussian_sigmas
        ]
        dogs = np.stack(
            [smoothed[index + 1] - smoothed[index] for index in range(3)],
            axis=1,
        )
        local = dogs[
            :,
            :,
            center - radius : center + radius + 1,
            center - radius : center + radius + 1,
        ]
        flattened = np.abs(local).reshape(stop - start, -1)
        best = np.argmax(flattened, axis=1)
        scale_index, remainder = np.divmod(best, (2 * radius + 1) ** 2)
        row_index, column_index = np.divmod(remainder, 2 * radius + 1)
        shift_y = row_index.astype(np.int64) - radius
        shift_x = column_index.astype(np.int64) - radius
        absolute_y = center + shift_y
        absolute_x = center + shift_x
        batch_index = np.arange(stop - start)
        response = np.abs(dogs[batch_index, scale_index, absolute_y, absolute_x])

        dxx = (
            dogs[batch_index, scale_index, absolute_y, absolute_x + 1]
            + dogs[batch_index, scale_index, absolute_y, absolute_x - 1]
            - 2.0 * dogs[batch_index, scale_index, absolute_y, absolute_x]
        )
        dyy = (
            dogs[batch_index, scale_index, absolute_y + 1, absolute_x]
            + dogs[batch_index, scale_index, absolute_y - 1, absolute_x]
            - 2.0 * dogs[batch_index, scale_index, absolute_y, absolute_x]
        )
        dxy = 0.25 * (
            dogs[batch_index, scale_index, absolute_y + 1, absolute_x + 1]
            - dogs[batch_index, scale_index, absolute_y + 1, absolute_x - 1]
            - dogs[batch_index, scale_index, absolute_y - 1, absolute_x + 1]
            + dogs[batch_index, scale_index, absolute_y - 1, absolute_x - 1]
        )
        determinant = dxx * dyy - dxy * dxy
        trace = dxx + dyy
        edge_limit = (config.edge_threshold + 1.0) ** 2 / config.edge_threshold
        not_edge = (determinant > 0.0) & (trace * trace < edge_limit * determinant)
        keep = (
            not_edge
            & (valid_fraction >= config.minimum_valid_support_fraction)
            & np.isfinite(response)
            & (response > 0.0)
        )

        refined_step = step[:, 0, 0]
        refined_east = shift_x * refined_step
        refined_north = -shift_y * refined_step
        refined_rho = np.hypot(refined_east, refined_north)
        refined_sinc = np.ones_like(refined_rho)
        np.divide(
            np.sin(refined_rho),
            refined_rho,
            out=refined_sinc,
            where=refined_rho != 0.0,
        )
        refined_bearings = np.cos(refined_rho)[:, None] * bearings[
            start:stop
        ] + refined_sinc[:, None] * (
            refined_east[:, None] * east[start:stop]
            + refined_north[:, None] * north[start:stop]
        )
        refined_bearings /= np.linalg.norm(refined_bearings, axis=1, keepdims=True)
        refined_pixels = rays_to_erp_pixels(
            refined_bearings, tuple(image.shape[:2])
        ).pixels_xy
        refined_scales = detected.scales_deg[start:stop] * np.power(
            scale_step, scale_index.astype(np.float64) - 1.0
        )
        for local_index in np.flatnonzero(keep):
            source = detected.keypoints[start + int(local_index)]
            verified.append(
                _Candidate(
                    source_xy=np.asarray(refined_pixels[local_index], dtype=np.float64),
                    bearing=np.asarray(refined_bearings[local_index], dtype=np.float64),
                    response=float(response[local_index]),
                    sigma_deg=float(refined_scales[local_index]),
                    octave=source.octave,
                    level=source.level,
                    valid_support_fraction=float(valid_fraction[local_index]),
                )
            )

    selected, unique_count = _select_candidates(verified, config)
    keypoints = tuple(
        SphericalKeypoint(
            source_erp_xy=item.source_xy,
            bearing_xyz=item.bearing,
            response=item.response,
            scale_deg=item.sigma_deg,
            octave=item.octave,
            level=item.level,
            valid_support_fraction=item.valid_support_fraction,
        )
        for item in selected
    )
    return replace(
        detected,
        keypoints=keypoints,
        diagnostics=replace(
            detected.diagnostics,
            unique_after_deduplication=unique_count,
            output_after_budget=len(keypoints),
        ),
    )


def _solid_angle_area_downsample(
    image: np.ndarray,
    validity: np.ndarray,
    output_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate aligned ERP source footprints with solid-angle row weights."""

    source_height, source_width = image.shape[:2]
    output_height, output_width = output_shape
    if source_height % output_height or source_width % output_width:
        raise ValueError(
            "area proposal resampling requires integer source/output ratios"
        )
    factor_y = source_height // output_height
    factor_x = source_width // output_width
    latitude = (
        np.pi / 2.0
        - (np.arange(source_height, dtype=np.float64) + 0.5) / source_height * np.pi
    )
    weights = np.cos(latitude)[:, None] * validity.astype(np.float64)
    blocked_weights = weights.reshape(output_height, factor_y, output_width, factor_x)
    denominator = blocked_weights.sum(axis=(1, 3))
    blocked_image: np.ndarray
    if image.ndim == 3:
        blocked_image = image.astype(np.float64).reshape(
            output_height,
            factor_y,
            output_width,
            factor_x,
            image.shape[2],
        )
        numerator = (blocked_image * blocked_weights[..., None]).sum(axis=(1, 3))
        downsampled = np.zeros_like(numerator)
        np.divide(
            numerator,
            np.maximum(denominator[..., None], np.finfo(np.float64).eps),
            out=downsampled,
        )
    else:
        blocked_image = image.astype(np.float64).reshape(
            output_height, factor_y, output_width, factor_x
        )
        numerator = (blocked_image * blocked_weights).sum(axis=(1, 3))
        downsampled = np.zeros_like(numerator)
        np.divide(
            numerator,
            np.maximum(denominator, np.finfo(np.float64).eps),
            out=downsampled,
        )
    full_weight = (
        np.cos(latitude)[:, None]
        .repeat(source_width, axis=1)
        .reshape(output_height, factor_y, output_width, factor_x)
        .sum(axis=(1, 3))
    )
    output_validity = denominator >= full_weight * (1.0 - 1e-6)
    if np.issubdtype(image.dtype, np.integer):
        limits = np.iinfo(image.dtype)
        downsampled = np.clip(np.rint(downsampled), limits.min, limits.max).astype(
            image.dtype
        )
    else:
        downsampled = downsampled.astype(image.dtype, copy=False)
    return downsampled, output_validity


def _spherical_pad(array: np.ndarray, radius: int) -> np.ndarray:
    """Pad an ERP with longitude wrap and pole reflection plus a half turn."""

    if radius <= 0:
        return array
    if array.shape[1] % 2:
        raise ValueError("spherical Gaussian prefilter requires an even ERP width")
    if radius >= array.shape[0]:
        raise ValueError("Gaussian prefilter radius must be smaller than ERP height")
    half_turn = array.shape[1] // 2
    north = np.roll(array[:radius][::-1], half_turn, axis=1)
    south = np.roll(array[-radius:][::-1], half_turn, axis=1)
    vertical = np.concatenate((north, array, south), axis=0)
    return np.concatenate(
        (vertical[:, -radius:], vertical, vertical[:, :radius]), axis=1
    )


def _gaussian_antialiased_downsample(
    image: np.ndarray,
    validity: np.ndarray,
    output_shape: tuple[int, int],
    *,
    sigma_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Low-pass an ERP with normalized support before decimation.

    The Gaussian operates in source ERP pixel coordinates because its purpose is
    to suppress frequencies above the reduced raster's Nyquist limit.  Boundary
    samples still obey spherical ERP topology.  Invalid samples never leak into
    RGB: image and support are filtered separately and then normalized.
    """

    import cv2

    source_height, source_width = image.shape[:2]
    output_height, output_width = output_shape
    if source_height % output_height or source_width % output_width:
        raise ValueError(
            "Gaussian proposal resampling requires integer source/output ratios"
        )
    sigma = _finite_real(sigma_px, "sigma_px", minimum=0.5)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    kernel_size = 2 * radius + 1
    support: np.ndarray = validity.astype(np.float32)
    source: np.ndarray = image.astype(np.float32)
    weighted = source * support[..., None] if source.ndim == 3 else source * support
    padded_support = _spherical_pad(support, radius)
    padded_weighted = _spherical_pad(weighted, radius)
    blurred_support = cv2.GaussianBlur(
        padded_support,
        (kernel_size, kernel_size),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_CONSTANT,
    )[radius:-radius, radius:-radius]
    blurred_weighted = cv2.GaussianBlur(
        padded_weighted,
        (kernel_size, kernel_size),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_CONSTANT,
    )[radius:-radius, radius:-radius]
    denominator = np.maximum(blurred_support, np.finfo(np.float32).eps)
    filtered = (
        blurred_weighted / denominator[..., None]
        if source.ndim == 3
        else blurred_weighted / denominator
    )
    downsampled = spherical_resize(filtered, output_shape, interpolation="bilinear")
    factor_y = source_height // output_height
    factor_x = source_width // output_width
    output_validity = np.asarray(
        validity.reshape(output_height, factor_y, output_width, factor_x).all(
            axis=(1, 3)
        ),
        dtype=bool,
    )
    if np.issubdtype(image.dtype, np.integer):
        limits = np.iinfo(image.dtype)
        downsampled = np.clip(np.rint(downsampled), limits.min, limits.max).astype(
            image.dtype
        )
    else:
        downsampled = downsampled.astype(image.dtype, copy=False)
    return downsampled, output_validity


def _spherical_gaussian_antialiased_downsample(
    image: np.ndarray,
    validity: np.ndarray,
    output_shape: tuple[int, int],
    *,
    sigma_px: float,
    intermediate_height: int,
    backend: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Prefilter on local spherical tangent planes at an intermediate scale."""

    source_height, source_width = image.shape[:2]
    output_height, output_width = output_shape
    middle_height = _positive_integer(
        intermediate_height, "intermediate_height", minimum=output_height
    )
    middle_width = 2 * middle_height
    if middle_height > source_height:
        raise ValueError("intermediate_height cannot exceed source height")
    if (
        source_height % middle_height
        or source_width % middle_width
        or middle_height % output_height
        or middle_width % output_width
    ):
        raise ValueError(
            "spherical Gaussian proposal resampling requires integer ratios"
        )
    source_sigma = _finite_real(sigma_px, "sigma_px", minimum=0.5)
    source_to_middle = source_height / middle_height
    middle_sigma = source_sigma / source_to_middle
    radius = max(1, int(math.ceil(3.0 * middle_sigma)))
    kernel_size = 2 * radius + 1

    support: np.ndarray = validity.astype(np.float32)
    source: np.ndarray = image.astype(np.float32)
    weighted = source * support[..., None] if source.ndim == 3 else source * support
    middle_support = spherical_resize(
        support, (middle_height, middle_width), interpolation="bilinear"
    ).astype(np.float32, copy=False)
    middle_weighted = spherical_resize(
        weighted, (middle_height, middle_width), interpolation="bilinear"
    ).astype(np.float32, copy=False)
    middle_image = (
        middle_weighted / np.maximum(middle_support[..., None], 1e-6)
        if source.ndim == 3
        else middle_weighted / np.maximum(middle_support, 1e-6)
    )
    filtered_support = spherical_gaussian_blur(
        middle_support,
        ksize=kernel_size,
        sigma=middle_sigma,
        backend=backend,
    ).astype(np.float32, copy=False)
    filtered_weighted = spherical_gaussian_blur(
        (
            middle_image * middle_support[..., None]
            if source.ndim == 3
            else middle_image * middle_support
        ),
        ksize=kernel_size,
        sigma=middle_sigma,
        backend=backend,
    ).astype(np.float32, copy=False)
    normalized = (
        filtered_weighted / np.maximum(filtered_support[..., None], 1e-6)
        if source.ndim == 3
        else filtered_weighted / np.maximum(filtered_support, 1e-6)
    )
    downsampled = spherical_resize(normalized, output_shape, interpolation="bilinear")
    factor_y = source_height // output_height
    factor_x = source_width // output_width
    output_validity = np.asarray(
        validity.reshape(output_height, factor_y, output_width, factor_x).all(
            axis=(1, 3)
        ),
        dtype=bool,
    )
    if np.issubdtype(image.dtype, np.integer):
        limits = np.iinfo(image.dtype)
        downsampled = np.clip(np.rint(downsampled), limits.min, limits.max).astype(
            image.dtype
        )
    else:
        downsampled = downsampled.astype(image.dtype, copy=False)
    return downsampled, output_validity


def _gaussian_levels_with_support(
    image: np.ndarray,
    initial_support: np.ndarray,
    config: SphericalDoGDetectorConfig,
    *,
    base_preblurred: bool,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    current_image: np.ndarray = image.astype(np.float32, copy=False)
    current_support = np.asarray(initial_support, dtype=np.float32)
    if current_support.shape != current_image.shape:
        raise ValueError("initial_support and image must have the same shape")
    gaussians: list[np.ndarray] = [current_image] if base_preblurred else []
    supports: list[np.ndarray] = [current_support] if base_preblurred else []
    epsilon = np.finfo(np.float32).eps
    previous_sigma = config.base_sigma_px if base_preblurred else 0.0
    start_level = 1 if base_preblurred else 0
    for level in range(start_level, config.levels_per_octave + 3):
        target_sigma = config.base_sigma_px * 2.0 ** (level / config.levels_per_octave)
        incremental_sigma = math.sqrt(
            max(0.0, target_sigma * target_sigma - previous_sigma * previous_sigma)
        )
        radius = max(1, int(math.ceil(3.0 * incremental_sigma)))
        size = 2 * radius + 1
        next_support = spherical_gaussian_blur(
            current_support,
            ksize=size,
            sigma=incremental_sigma,
            backend=config.convolution_backend,
        ).astype(np.float32, copy=False)
        numerator = spherical_gaussian_blur(
            current_image * current_support,
            ksize=size,
            sigma=incremental_sigma,
            backend=config.convolution_backend,
        ).astype(np.float32, copy=False)
        normalized = np.zeros_like(numerator, dtype=np.float32)
        np.divide(numerator, np.maximum(next_support, epsilon), out=normalized)
        gaussians.append(normalized)
        supports.append(np.clip(next_support, 0.0, 1.0))
        current_image = normalized
        current_support = supports[-1]
        previous_sigma = target_sigma
    return gaussians, supports


def _select_candidates(
    candidates: list[_Candidate], config: SphericalDoGDetectorConfig
) -> tuple[list[_Candidate], int]:
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
    if not ordered:
        return [], 0
    angular_threshold = math.radians(config.angular_dedup_threshold_deg)
    chord_threshold = 2.0 * math.sin(angular_threshold * 0.5)
    bearings = np.stack([item.bearing for item in ordered])
    log_scales = np.log2(np.asarray([item.sigma_deg for item in ordered]))
    neighbors = cKDTree(bearings)
    suppressed: np.ndarray = np.zeros(len(ordered), dtype=bool)
    deduplicated: list[_Candidate] = []
    for index, candidate in enumerate(ordered):
        if suppressed[index]:
            continue
        deduplicated.append(candidate)
        nearby = neighbors.query_ball_point(candidate.bearing, chord_threshold + 1e-15)
        for neighbor_index in nearby:
            if neighbor_index <= index:
                continue
            if (
                abs(log_scales[neighbor_index] - log_scales[index])
                <= config.scale_dedup_log2
            ):
                suppressed[neighbor_index] = True
    if config.selection_policy == "response":
        return deduplicated[: config.max_keypoints], len(deduplicated)
    rows, columns = config.selection_grid_shape
    cells: dict[tuple[int, int], list[_Candidate]] = {}
    for candidate in deduplicated:
        y = float(np.clip(candidate.bearing[1], -1.0, 1.0))
        longitude = math.atan2(float(candidate.bearing[0]), float(candidate.bearing[2]))
        row = min(rows - 1, int((y + 1.0) * 0.5 * rows))
        column = min(
            columns - 1, int((longitude + math.pi) / (2.0 * math.pi) * columns)
        )
        cells.setdefault((row, column), []).append(candidate)
    ordered_cells = sorted(cells)
    selected: list[_Candidate] = []
    depth = 0
    while len(selected) < config.max_keypoints:
        added = False
        for cell in ordered_cells:
            values = cells[cell]
            if depth < len(values):
                selected.append(values[depth])
                added = True
                if len(selected) >= config.max_keypoints:
                    break
        if not added:
            break
        depth += 1
    return selected, len(deduplicated)
