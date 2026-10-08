"""Experimental descriptor-agnostic keypoint detection on the sphere."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import math
from numbers import Integral, Real
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from panorai.geometry import erp_pixels_to_rays, rays_to_erp_pixels
from panorai.image_processing import spherical_gaussian_blur, spherical_resize
from panorai.image_processing._native import (
    native_spherical_extrema3d,
    supports_native_extrema,
)
from panorai.image_processing._sampling import (
    row_chunks,
    sample_rays,
    sample_rays_multi,
    sample_tangent,
)

from ._extractor import (
    _array_checksum,
    _as_panorama,
    _opencv_image,
    _panorai_commit,
    _panorai_version,
)

SPHERICAL_DOG_DETECTOR_INTERFACE = "panorai-spherical-dog-detector/v2"
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
    refinement_max_iterations: int = 5
    refinement_maximum_offset: float = 1.5
    refinement_maximum_hessian_condition: float = 1.0e6
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
            "refinement_max_iterations",
            _positive_integer(
                self.refinement_max_iterations, "refinement_max_iterations"
            ),
        )
        object.__setattr__(
            self,
            "refinement_maximum_offset",
            _finite_real(
                self.refinement_maximum_offset,
                "refinement_maximum_offset",
                minimum=0.5,
            ),
        )
        object.__setattr__(
            self,
            "refinement_maximum_hessian_condition",
            _finite_real(
                self.refinement_maximum_hessian_condition,
                "refinement_maximum_hessian_condition",
                minimum=1.0,
            ),
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
                "localization": "second-order-taylor-tangent-east-north-scale-level",
                "contrast_policy": "interpolated-dog-after-refinement",
                "edge_policy": "refined-spatial-hessian",
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
    """Common descriptor-independent spherical keypoint fields.

    The direct DoG detector returns :class:`SphericalRefinedKeypoint` objects.
    This base remains public so descriptor and patch consumers can depend only
    on location, bearing, response, angular scale, and support.
    """

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
            _finite_real(self.scale_deg, "scale_deg", minimum=np.finfo(float).tiny),
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
class SphericalRefinedKeypoint(SphericalKeypoint):
    """A DoG extremum localized continuously in tangent position and scale.

    ``tangent_offset_rad`` is ``[east, north]`` from the final discrete
    scale-space sample. ``refined_level`` is continuous within ``octave``.
    The signed interpolated DoG value is retained separately from the
    non-negative response used for ranking.
    """

    refined_level: float
    tangent_offset_rad: np.ndarray
    interpolated_dog_response: float
    edge_score: float
    hessian_condition: float
    localization_iterations: int

    def __post_init__(self) -> None:
        super(SphericalRefinedKeypoint, self).__post_init__()
        offset = np.asarray(self.tangent_offset_rad, dtype=np.float64)
        if offset.shape != (2,) or not np.isfinite(offset).all():
            raise ValueError("tangent_offset_rad must be a finite length-2 vector")
        object.__setattr__(self, "tangent_offset_rad", offset)
        object.__setattr__(
            self,
            "refined_level",
            _finite_real(self.refined_level, "refined_level", minimum=0.0),
        )
        object.__setattr__(
            self,
            "interpolated_dog_response",
            _finite_real(self.interpolated_dog_response, "interpolated_dog_response"),
        )
        object.__setattr__(
            self,
            "edge_score",
            _finite_real(self.edge_score, "edge_score", minimum=0.0),
        )
        object.__setattr__(
            self,
            "hessian_condition",
            _finite_real(self.hessian_condition, "hessian_condition", minimum=1.0),
        )
        object.__setattr__(
            self,
            "localization_iterations",
            _positive_integer(self.localization_iterations, "localization_iterations"),
        )


@dataclass(frozen=True, slots=True)
class SphericalOctaveDetectionDiagnostics:
    """Candidate counts through localization and validity filtering."""

    octave: int
    shape_hw: tuple[int, int]
    raw_extrema_count: int
    refinement_converged_count: int
    interpolated_contrast_count: int
    refined_edge_count: int
    valid_support_count: int
    rejected_nonconverged_count: int
    rejected_ill_conditioned_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SphericalDetectionDiagnostics:
    """Stage counts for descriptor-free spherical detection."""

    raw_extrema_count: int
    refinement_converged_count: int
    interpolated_contrast_count: int
    refined_edge_count: int
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
    keypoints: tuple[SphericalRefinedKeypoint, ...]
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

    @property
    def refined_levels(self) -> np.ndarray:
        return np.asarray(
            [item.refined_level for item in self.keypoints], dtype=np.float64
        )

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
    refined_level: float
    tangent_offset_rad: np.ndarray
    interpolated_dog_response: float
    edge_score: float
    hessian_condition: float
    localization_iterations: int


@dataclass(frozen=True, slots=True)
class _RefinementResult:
    candidate: _Candidate | None
    rejection_reason: str | None


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
            SphericalRefinedKeypoint(
                source_erp_xy=item.source_xy,
                bearing_xyz=item.bearing,
                response=item.response,
                scale_deg=item.sigma_deg,
                octave=item.octave,
                level=item.level,
                valid_support_fraction=item.valid_support_fraction,
                refined_level=item.refined_level,
                tangent_offset_rad=item.tangent_offset_rad,
                interpolated_dog_response=item.interpolated_dog_response,
                edge_score=item.edge_score,
                hessian_condition=item.hessian_condition,
                localization_iterations=item.localization_iterations,
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
                refinement_converged_count=sum(
                    item.refinement_converged_count for item in octave_diagnostics
                ),
                interpolated_contrast_count=sum(
                    item.interpolated_contrast_count for item in octave_diagnostics
                ),
                refined_edge_count=sum(
                    item.refined_edge_count for item in octave_diagnostics
                ),
                valid_support_count=len(candidates),
                unique_after_deduplication=unique_count,
                output_after_budget=len(selected),
                octaves=tuple(octave_diagnostics),
            ),
        )

    def detect_batch(
        self,
        panoramas: tuple[Any, ...],
        *,
        panorama_ids: tuple[str | None, ...] | None = None,
        validity_masks: tuple[np.ndarray | None, ...] | None = None,
    ) -> tuple[SphericalKeypointSet, ...]:
        """Detect same-shape panoramas with shared spherical pyramid traversals."""

        items = tuple(panoramas)
        if not items:
            raise ValueError("panoramas must not be empty")
        ids = panorama_ids or (None,) * len(items)
        masks = validity_masks or (None,) * len(items)
        if len(ids) != len(items) or len(masks) != len(items):
            raise ValueError("panorama_ids and validity_masks must match panoramas")
        images: list[np.ndarray] = []
        grays: list[np.ndarray] = []
        validities: list[np.ndarray] = []
        checksums: list[str] = []
        resolved_ids: list[str] = []
        for panorama, panorama_id, validity_mask in zip(items, ids, masks, strict=True):
            wrapped = _as_panorama(panorama)
            if not isinstance(wrapped.image, np.ndarray):
                raise TypeError(
                    "SphericalDoGDetector currently requires NumPy panoramas"
                )
            image = wrapped.image
            if image.ndim not in {2, 3} or image.shape[0] < 8 or image.shape[1] < 16:
                raise ValueError(
                    "panoramas must use HW/HWC layout and be at least 8x16"
                )
            validity = np.asarray(wrapped.validity("image"), dtype=bool)
            if wrapped.support_mask is not None:
                validity &= np.asarray(wrapped.support_mask, dtype=bool)
            if validity_mask is not None:
                if (
                    not isinstance(validity_mask, np.ndarray)
                    or validity_mask.dtype != np.bool_
                    or validity_mask.shape != image.shape[:2]
                ):
                    raise ValueError(
                        "each validity mask must be boolean and match its panorama"
                    )
                validity &= validity_mask
            checksum = _array_checksum(image)
            images.append(image)
            grays.append(_opencv_image(image, validity).astype(np.float32) / 255.0)
            validities.append(validity)
            checksums.append(checksum)
            resolved_ids.append(panorama_id or f"panorama-{checksum[:16]}")
        if any(image.shape[:2] != images[0].shape[:2] for image in images[1:]):
            raise ValueError("batched panoramas must have the same spatial shape")

        detections = self._detect_batch(tuple(grays), tuple(validities))
        outputs: list[SphericalKeypointSet] = []
        for image, checksum, resolved_id, (candidates, octave_diagnostics) in zip(
            images, checksums, resolved_ids, detections, strict=True
        ):
            selected, unique_count = _select_candidates(candidates, self.config)
            keypoints = tuple(
                SphericalRefinedKeypoint(
                    source_erp_xy=item.source_xy,
                    bearing_xyz=item.bearing,
                    response=item.response,
                    scale_deg=item.sigma_deg,
                    octave=item.octave,
                    level=item.level,
                    valid_support_fraction=item.valid_support_fraction,
                    refined_level=item.refined_level,
                    tangent_offset_rad=item.tangent_offset_rad,
                    interpolated_dog_response=item.interpolated_dog_response,
                    edge_score=item.edge_score,
                    hessian_condition=item.hessian_condition,
                    localization_iterations=item.localization_iterations,
                )
                for item in selected
            )
            outputs.append(
                SphericalKeypointSet(
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
                        refinement_converged_count=sum(
                            item.refinement_converged_count
                            for item in octave_diagnostics
                        ),
                        interpolated_contrast_count=sum(
                            item.interpolated_contrast_count
                            for item in octave_diagnostics
                        ),
                        refined_edge_count=sum(
                            item.refined_edge_count for item in octave_diagnostics
                        ),
                        valid_support_count=len(candidates),
                        unique_after_deduplication=unique_count,
                        output_after_budget=len(selected),
                        octaves=tuple(octave_diagnostics),
                    ),
                )
            )
        return tuple(outputs)

    def _detect(
        self, gray: np.ndarray, validity: np.ndarray
    ) -> tuple[list[_Candidate], list[SphericalOctaveDetectionDiagnostics]]:
        original_shape = gray.shape
        candidates: list[_Candidate] = []
        diagnostics: list[SphericalOctaveDetectionDiagnostics] = []
        octave_image = gray
        octave_validity = validity
        octave_support = validity.astype(np.float32)
        for octave in range(self.config.octaves):
            if min(octave_image.shape) < 8:
                break
            gaussians, supports = _gaussian_levels_with_support(
                octave_image,
                octave_support,
                self.config,
                base_preblurred=octave > 0,
            )
            octave_candidates, octave_diagnostics = _detect_octave_candidates(
                gaussians=gaussians,
                supports=supports,
                validity=octave_validity,
                octave=octave,
                original_shape=original_shape,
                config=self.config,
            )
            candidates.extend(octave_candidates)
            diagnostics.append(octave_diagnostics)
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

    def _detect_batch(
        self,
        grays: tuple[np.ndarray, ...],
        validities: tuple[np.ndarray, ...],
    ) -> tuple[tuple[list[_Candidate], list[SphericalOctaveDetectionDiagnostics]], ...]:
        if not grays or len(grays) != len(validities):
            raise ValueError("grays and validities must have the same positive length")
        original_shape = grays[0].shape
        if any(gray.shape != original_shape for gray in grays[1:]):
            raise ValueError("batched grayscale panoramas must have the same shape")
        octave_images = [gray for gray in grays]
        octave_validities = [validity for validity in validities]
        octave_supports = [validity.astype(np.float32) for validity in validities]
        candidate_sets: list[list[_Candidate]] = [[] for _ in grays]
        diagnostic_sets: list[list[SphericalOctaveDetectionDiagnostics]] = [
            [] for _ in grays
        ]
        for octave in range(self.config.octaves):
            if min(octave_images[0].shape) < 8:
                break
            gaussian_sets, support_sets = _gaussian_levels_with_support_batch(
                tuple(octave_images),
                tuple(octave_supports),
                self.config,
                base_preblurred=octave > 0,
            )
            for index in range(len(grays)):
                candidates, diagnostics = _detect_octave_candidates(
                    gaussians=gaussian_sets[index],
                    supports=support_sets[index],
                    validity=octave_validities[index],
                    octave=octave,
                    original_shape=original_shape,
                    config=self.config,
                )
                candidate_sets[index].extend(candidates)
                diagnostic_sets[index].append(diagnostics)
            if octave + 1 < self.config.octaves:
                next_shape = (
                    max(2, (octave_images[0].shape[0] + 1) // 2),
                    max(2, (octave_images[0].shape[1] + 1) // 2),
                )
                for index in range(len(grays)):
                    octave_images[index] = spherical_resize(
                        gaussian_sets[index][self.config.levels_per_octave],
                        next_shape,
                    ).astype(np.float32, copy=False)
                    octave_validities[index] = (
                        spherical_resize(
                            octave_validities[index].astype(np.float32),
                            next_shape,
                            interpolation="nearest",
                        )
                        >= 0.5
                    )
                    octave_supports[index] = np.clip(
                        spherical_resize(
                            support_sets[index][self.config.levels_per_octave],
                            next_shape,
                        ).astype(np.float32, copy=False),
                        0.0,
                        1.0,
                    )
        return tuple(
            (candidate_sets[index], diagnostic_sets[index])
            for index in range(len(grays))
        )


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
    """Re-rank and locally refine coarse proposals on native-resolution samples."""

    if not detected.keypoints:
        return detected
    gray = _opencv_image(image, validity).astype(np.float32) / 255.0
    bearings = detected.bearings
    scales_rad = np.radians(detected.scales_deg)
    sample_count = config.fine_patch_size
    radius = config.fine_search_radius_samples
    coordinates = np.arange(sample_count, dtype=np.float64) - sample_count // 2
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
        signed_response = dogs[batch_index, scale_index, absolute_y, absolute_x]
        response = np.abs(signed_response)

        dxx = (
            dogs[batch_index, scale_index, absolute_y, absolute_x + 1]
            + dogs[batch_index, scale_index, absolute_y, absolute_x - 1]
            - 2.0 * signed_response
        )
        dyy = (
            dogs[batch_index, scale_index, absolute_y + 1, absolute_x]
            + dogs[batch_index, scale_index, absolute_y - 1, absolute_x]
            - 2.0 * signed_response
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
        edge_score = np.full_like(trace, np.inf, dtype=np.float64)
        np.divide(trace * trace, determinant, out=edge_score, where=determinant > 0.0)
        not_edge = (determinant > 0.0) & (edge_score < edge_limit)
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
            spatial_hessian = np.asarray(
                (
                    (dxx[local_index], dxy[local_index]),
                    (dxy[local_index], dyy[local_index]),
                ),
                dtype=np.float64,
            )
            condition = float(np.linalg.cond(spatial_hessian))
            verified.append(
                _Candidate(
                    source_xy=np.asarray(refined_pixels[local_index], dtype=np.float64),
                    bearing=np.asarray(refined_bearings[local_index], dtype=np.float64),
                    response=float(response[local_index]),
                    sigma_deg=float(refined_scales[local_index]),
                    octave=source.octave,
                    level=source.level,
                    valid_support_fraction=float(valid_fraction[local_index]),
                    refined_level=float(
                        source.refined_level + scale_index[local_index] - 1.0
                    ),
                    tangent_offset_rad=np.asarray(
                        (refined_east[local_index], refined_north[local_index]),
                        dtype=np.float64,
                    ),
                    interpolated_dog_response=float(signed_response[local_index]),
                    edge_score=float(edge_score[local_index]),
                    hessian_condition=condition,
                    localization_iterations=source.localization_iterations,
                )
            )

    selected, unique_count = _select_candidates(verified, config)
    keypoints = tuple(
        SphericalRefinedKeypoint(
            source_erp_xy=item.source_xy,
            bearing_xyz=item.bearing,
            response=item.response,
            scale_deg=item.sigma_deg,
            octave=item.octave,
            level=item.level,
            valid_support_fraction=item.valid_support_fraction,
            refined_level=item.refined_level,
            tangent_offset_rad=item.tangent_offset_rad,
            interpolated_dog_response=item.interpolated_dog_response,
            edge_score=item.edge_score,
            hessian_condition=item.hessian_condition,
            localization_iterations=item.localization_iterations,
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
    support = validity.astype(np.float32)
    source = image.astype(np.float32)
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
    output_validity = validity.reshape(
        output_height, factor_y, output_width, factor_x
    ).all(axis=(1, 3))
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

    support = validity.astype(np.float32)
    source = image.astype(np.float32)
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
    output_validity = validity.reshape(
        output_height, factor_y, output_width, factor_x
    ).all(axis=(1, 3))
    if np.issubdtype(image.dtype, np.integer):
        limits = np.iinfo(image.dtype)
        downsampled = np.clip(np.rint(downsampled), limits.min, limits.max).astype(
            image.dtype
        )
    else:
        downsampled = downsampled.astype(image.dtype, copy=False)
    return downsampled, output_validity


def _dog_extrema_candidates_mask(
    previous: np.ndarray,
    current: np.ndarray,
    following: np.ndarray,
    step_rad: float,
    preliminary_contrast_threshold: float,
) -> np.ndarray:
    """Return strict 3-D extrema before continuous localization.

    Spatial neighbours are sampled on the tangent plane, so the 26-sample
    comparison has the same angular meaning at every latitude and crosses the
    ERP seam without a special case.  Contrast and edge rejection are applied
    again after Taylor refinement; this threshold is deliberately permissive.
    """

    if supports_native_extrema(previous, current, following):
        return native_spherical_extrema3d(
            previous,
            current,
            following,
            step_rad,
            preliminary_contrast_threshold,
        )
    return _dog_extrema_candidates_mask_numpy(
        previous,
        current,
        following,
        step_rad,
        preliminary_contrast_threshold,
    )


def _dog_extrema_candidates_mask_numpy(
    previous: np.ndarray,
    current: np.ndarray,
    following: np.ndarray,
    step_rad: float,
    preliminary_contrast_threshold: float,
) -> np.ndarray:
    """Reference implementation for parity and extension-free installs."""

    height, width = current.shape
    result = np.zeros(current.shape, dtype=bool)
    for rows in row_chunks(height, width):
        center = current[rows]
        maxima = center >= preliminary_contrast_threshold
        minima = center <= -preliminary_contrast_threshold
        for scale_index, scale in enumerate((previous, current, following)):
            for north_index in (-1, 0, 1):
                for east_index in (-1, 0, 1):
                    if scale_index == 1 and east_index == 0 and north_index == 0:
                        continue
                    neighbour = sample_tangent(
                        scale,
                        rows,
                        east_index * step_rad,
                        north_index * step_rad,
                    )
                    maxima &= center > neighbour
                    minima &= center < neighbour
        result[rows] = maxima | minima
    return result


def _tangent_offset_rays(
    bearing: np.ndarray,
    east_offset_rad: float | np.ndarray,
    north_offset_rad: float | np.ndarray,
) -> np.ndarray:
    """Move a canonical ray by exponential-map tangent offsets."""

    center = np.asarray(bearing, dtype=np.float64)
    center /= np.linalg.norm(center)
    longitude = math.atan2(float(center[0]), float(center[2]))
    east = np.asarray(
        (math.cos(longitude), 0.0, -math.sin(longitude)), dtype=np.float64
    )
    north = np.cross(center, east)
    north /= np.linalg.norm(north)
    east_offset = np.asarray(east_offset_rad, dtype=np.float64)
    north_offset = np.asarray(north_offset_rad, dtype=np.float64)
    radius = np.hypot(east_offset, north_offset)
    sinc = np.ones_like(radius, dtype=np.float64)
    np.divide(np.sin(radius), radius, out=sinc, where=radius != 0.0)
    tangent = east_offset[..., None] * east + north_offset[..., None] * north
    rays = np.cos(radius)[..., None] * center + sinc[..., None] * tangent
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def _sample_dog_cube(
    dogs: list[np.ndarray],
    level: int,
    bearing: np.ndarray,
    step_rad: float,
) -> np.ndarray:
    """Sample a 3x3x3 DoG neighbourhood in scale, north, east order."""

    offsets = np.asarray((-step_rad, 0.0, step_rad), dtype=np.float64)
    east_grid, north_grid = np.meshgrid(offsets, offsets)
    rays = _tangent_offset_rays(bearing, east_grid, north_grid)
    return np.stack(
        sample_rays_multi(
            tuple(dogs[index] for index in (level - 1, level, level + 1)),
            rays,
            interpolation="bilinear",
        ),
        axis=0,
    ).astype(np.float64, copy=False)


def _sample_initial_dog_cubes(
    dogs: list[np.ndarray],
    level: int,
    bearings: np.ndarray,
    step_rad: float,
) -> np.ndarray:
    """Batch the first 3x3x3 sample while preserving scalar ray construction."""

    centers = np.asarray(bearings, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("bearings must have shape (N, 3)")
    if len(centers) == 0:
        return np.empty((0, 3, 3, 3), dtype=np.float64)
    offsets = np.asarray((-step_rad, 0.0, step_rad), dtype=np.float64)
    east_grid, north_grid = np.meshgrid(offsets, offsets)
    rays = np.stack(
        [_tangent_offset_rays(bearing, east_grid, north_grid) for bearing in centers],
        axis=0,
    )
    return np.stack(
        sample_rays_multi(
            tuple(dogs[index] for index in (level - 1, level, level + 1)),
            rays,
            interpolation="bilinear",
        ),
        axis=1,
    ).astype(np.float64, copy=False)


def _quadratic_derivatives(
    cube: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the gradient and Hessian in east, north, scale coordinates."""

    samples = np.asarray(cube, dtype=np.float64)
    if samples.shape != (3, 3, 3):
        raise ValueError("cube must have shape (3, 3, 3)")
    center = samples[1, 1, 1]
    gradient = np.asarray(
        (
            0.5 * (samples[1, 1, 2] - samples[1, 1, 0]),
            0.5 * (samples[1, 2, 1] - samples[1, 0, 1]),
            0.5 * (samples[2, 1, 1] - samples[0, 1, 1]),
        ),
        dtype=np.float64,
    )
    hessian = np.asarray(
        (
            (
                samples[1, 1, 2] + samples[1, 1, 0] - 2.0 * center,
                0.25
                * (
                    samples[1, 2, 2]
                    - samples[1, 2, 0]
                    - samples[1, 0, 2]
                    + samples[1, 0, 0]
                ),
                0.25
                * (
                    samples[2, 1, 2]
                    - samples[2, 1, 0]
                    - samples[0, 1, 2]
                    + samples[0, 1, 0]
                ),
            ),
            (
                0.0,
                samples[1, 2, 1] + samples[1, 0, 1] - 2.0 * center,
                0.25
                * (
                    samples[2, 2, 1]
                    - samples[2, 0, 1]
                    - samples[0, 2, 1]
                    + samples[0, 0, 1]
                ),
            ),
            (
                0.0,
                0.0,
                samples[2, 1, 1] + samples[0, 1, 1] - 2.0 * center,
            ),
        ),
        dtype=np.float64,
    )
    hessian[1, 0] = hessian[0, 1]
    hessian[2, 0] = hessian[0, 2]
    hessian[2, 1] = hessian[1, 2]
    return gradient, hessian


def _solve_quadratic_offset(
    gradient: np.ndarray,
    hessian: np.ndarray,
    maximum_condition: float,
) -> tuple[np.ndarray | None, float]:
    """Solve ``H delta = -g`` and reject an unstable Hessian."""

    condition = float(np.linalg.cond(hessian))
    if not math.isfinite(condition) or condition > maximum_condition:
        return None, condition
    try:
        offset = np.linalg.solve(hessian, -gradient)
    except np.linalg.LinAlgError:
        return None, condition
    if not np.isfinite(offset).all():
        return None, condition
    return offset, condition


def _refine_dog_extremum(
    *,
    dogs: list[np.ndarray],
    supports: list[np.ndarray],
    validity: np.ndarray,
    initial_xy: tuple[int, int],
    initial_level: int,
    octave: int,
    original_shape: tuple[int, int],
    step_rad: float,
    config: SphericalDoGDetectorConfig,
    initial_bearing: np.ndarray | None = None,
    initial_cube: np.ndarray | None = None,
) -> _RefinementResult:
    """Refine one DoG extremum by the SIFT second-order Taylor model."""

    x_index, y_index = initial_xy
    octave_shape = dogs[0].shape
    if initial_bearing is None:
        bearing = erp_pixels_to_rays(
            np.asarray(((x_index, y_index),), dtype=np.float64), octave_shape
        )[0]
    else:
        bearing = np.asarray(initial_bearing, dtype=np.float64).copy()
    level = initial_level
    final_cube: np.ndarray | None = None
    final_gradient: np.ndarray | None = None
    final_hessian: np.ndarray | None = None
    final_offset: np.ndarray | None = None
    final_condition = math.inf
    iterations = 0

    for iterations in range(1, config.refinement_max_iterations + 1):
        if iterations == 1 and initial_cube is not None:
            cube = np.asarray(initial_cube, dtype=np.float64)
        else:
            cube = _sample_dog_cube(dogs, level, bearing, step_rad)
        gradient, hessian = _quadratic_derivatives(cube)
        offset, condition = _solve_quadratic_offset(
            gradient,
            hessian,
            config.refinement_maximum_hessian_condition,
        )
        if offset is None:
            return _RefinementResult(None, "ill-conditioned-hessian")
        if float(np.max(np.abs(offset))) > config.refinement_maximum_offset:
            return _RefinementResult(None, "unstable-offset")
        if float(np.max(np.abs(offset))) <= 0.5:
            final_cube = cube
            final_gradient = gradient
            final_hessian = hessian
            final_offset = offset
            final_condition = condition
            break

        scale_shift = int(np.sign(offset[2])) if abs(offset[2]) > 0.5 else 0
        next_level = level + scale_shift
        if next_level < 1 or next_level > len(dogs) - 2:
            return _RefinementResult(None, "nonconverged")
        east_shift = step_rad * int(np.sign(offset[0])) if abs(offset[0]) > 0.5 else 0.0
        north_shift = (
            step_rad * int(np.sign(offset[1])) if abs(offset[1]) > 0.5 else 0.0
        )
        if east_shift or north_shift:
            bearing = _tangent_offset_rays(bearing, east_shift, north_shift)
        level = next_level
    else:
        return _RefinementResult(None, "nonconverged")

    if (
        final_cube is None
        or final_gradient is None
        or final_hessian is None
        or final_offset is None
    ):
        raise RuntimeError("refinement converged without Taylor coefficients")

    interpolated_response = float(
        final_cube[1, 1, 1] + 0.5 * final_gradient @ final_offset
    )
    if abs(interpolated_response) < config.contrast_threshold:
        return _RefinementResult(None, "low-interpolated-contrast")

    spatial_hessian = final_hessian[:2, :2]
    determinant = float(np.linalg.det(spatial_hessian))
    if determinant <= np.finfo(np.float64).eps:
        return _RefinementResult(None, "edge-response")
    trace = float(np.trace(spatial_hessian))
    edge_score = trace * trace / determinant
    edge_limit = (config.edge_threshold + 1.0) ** 2 / config.edge_threshold
    if not math.isfinite(edge_score) or edge_score >= edge_limit:
        return _RefinementResult(None, "edge-response")

    tangent_offset = final_offset[:2] * step_rad
    refined_bearing = _tangent_offset_rays(
        bearing, tangent_offset[0], tangent_offset[1]
    )
    octave_xy = rays_to_erp_pixels(
        np.asarray((refined_bearing,), dtype=np.float64), octave_shape
    ).pixels_xy[0]
    valid_x = int(math.floor(float(octave_xy[0]) + 0.5)) % octave_shape[1]
    valid_y = int(
        np.clip(math.floor(float(octave_xy[1]) + 0.5), 0, octave_shape[0] - 1)
    )
    support_fraction = float(
        sample_rays(supports[level + 2], refined_bearing, interpolation="bilinear")
    )
    if (
        not validity[valid_y, valid_x]
        or support_fraction < config.minimum_valid_support_fraction
    ):
        return _RefinementResult(None, "invalid-support")

    refined_level = float(level + final_offset[2])
    sigma_px = config.base_sigma_px * 2.0 ** (
        (refined_level + 1.0) / config.levels_per_octave
    )
    source_xy = rays_to_erp_pixels(
        np.asarray((refined_bearing,), dtype=np.float64), original_shape
    ).pixels_xy[0]
    return _RefinementResult(
        _Candidate(
            source_xy=source_xy,
            bearing=refined_bearing,
            response=abs(interpolated_response),
            sigma_deg=180.0 / octave_shape[0] * sigma_px,
            octave=octave,
            level=level,
            valid_support_fraction=support_fraction,
            refined_level=refined_level,
            tangent_offset_rad=np.asarray(tangent_offset, dtype=np.float64),
            interpolated_dog_response=interpolated_response,
            edge_score=edge_score,
            hessian_condition=final_condition,
            localization_iterations=iterations,
        ),
        None,
    )


def _gaussian_levels_with_support(
    image: np.ndarray,
    initial_support: np.ndarray,
    config: SphericalDoGDetectorConfig,
    *,
    base_preblurred: bool,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    current_image = image.astype(np.float32, copy=False)
    current_support = np.asarray(initial_support, dtype=np.float32)
    if current_support.shape != current_image.shape:
        raise ValueError("initial_support and image must have the same shape")
    gaussians: list[np.ndarray] = [current_image] if base_preblurred else []
    supports: list[np.ndarray] = [current_support] if base_preblurred else []
    epsilon = np.finfo(np.float32).eps
    full_support = bool(np.all(current_support == 1.0))
    previous_sigma = config.base_sigma_px if base_preblurred else 0.0
    start_level = 1 if base_preblurred else 0
    for level in range(start_level, config.levels_per_octave + 3):
        target_sigma = config.base_sigma_px * 2.0 ** (level / config.levels_per_octave)
        incremental_sigma = math.sqrt(
            max(0.0, target_sigma * target_sigma - previous_sigma * previous_sigma)
        )
        radius = max(1, int(math.ceil(3.0 * incremental_sigma)))
        size = 2 * radius + 1
        if full_support:
            normalized = spherical_gaussian_blur(
                current_image,
                ksize=size,
                sigma=incremental_sigma,
                backend=config.convolution_backend,
            ).astype(np.float32, copy=False)
            next_support = current_support
        else:
            filtered = spherical_gaussian_blur(
                np.stack((current_support, current_image * current_support), axis=-1),
                ksize=size,
                sigma=incremental_sigma,
                backend=config.convolution_backend,
            ).astype(np.float32, copy=False)
            next_support = filtered[..., 0]
            numerator = filtered[..., 1]
            normalized = np.zeros_like(numerator, dtype=np.float32)
            np.divide(numerator, np.maximum(next_support, epsilon), out=normalized)
        gaussians.append(normalized)
        supports.append(np.clip(next_support, 0.0, 1.0))
        current_image = normalized
        current_support = supports[-1]
        previous_sigma = target_sigma
    return gaussians, supports


def _gaussian_levels_with_support_batch(
    images: tuple[np.ndarray, ...],
    initial_supports: tuple[np.ndarray, ...],
    config: SphericalDoGDetectorConfig,
    *,
    base_preblurred: bool,
) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]]]:
    """Build same-shape pyramids while sharing each spherical filter traversal."""

    if not images or len(images) != len(initial_supports):
        raise ValueError(
            "images and initial_supports must have the same positive length"
        )
    shape = images[0].shape
    if any(image.shape != shape for image in images[1:]):
        raise ValueError("all images must have the same shape")
    if any(support.shape != shape for support in initial_supports):
        raise ValueError("all supports must match the image shape")
    current_images = [image.astype(np.float32, copy=False) for image in images]
    current_supports = [
        np.asarray(support, dtype=np.float32) for support in initial_supports
    ]
    gaussian_sets = [[image] if base_preblurred else [] for image in current_images]
    support_sets = [
        [support] if base_preblurred else [] for support in current_supports
    ]
    full_support = [bool(np.all(support == 1.0)) for support in current_supports]
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
        next_images: list[np.ndarray | None] = [None] * len(images)
        next_supports: list[np.ndarray | None] = [None] * len(images)

        full_indices = [index for index, value in enumerate(full_support) if value]
        if full_indices:
            payload = np.stack(
                [current_images[index] for index in full_indices], axis=-1
            )
            filtered = spherical_gaussian_blur(
                payload,
                ksize=size,
                sigma=incremental_sigma,
                backend=config.convolution_backend,
            ).astype(np.float32, copy=False)
            for channel, index in enumerate(full_indices):
                next_images[index] = filtered[..., channel]
                next_supports[index] = current_supports[index]

        masked_indices = [
            index for index, value in enumerate(full_support) if not value
        ]
        if masked_indices:
            payload = np.stack(
                [
                    channel
                    for index in masked_indices
                    for channel in (
                        current_supports[index],
                        current_images[index] * current_supports[index],
                    )
                ],
                axis=-1,
            )
            filtered = spherical_gaussian_blur(
                payload,
                ksize=size,
                sigma=incremental_sigma,
                backend=config.convolution_backend,
            ).astype(np.float32, copy=False)
            for pair_channel, index in enumerate(masked_indices):
                next_support = filtered[..., 2 * pair_channel]
                numerator = filtered[..., 2 * pair_channel + 1]
                normalized = np.zeros_like(numerator, dtype=np.float32)
                np.divide(numerator, np.maximum(next_support, epsilon), out=normalized)
                next_images[index] = normalized
                next_supports[index] = np.clip(next_support, 0.0, 1.0)

        for index in range(len(images)):
            image = next_images[index]
            support = next_supports[index]
            if image is None or support is None:
                raise RuntimeError("batched Gaussian level was not materialized")
            gaussian_sets[index].append(image)
            support_sets[index].append(support)
            current_images[index] = image
            current_supports[index] = support
        previous_sigma = target_sigma
    return gaussian_sets, support_sets


def _detect_octave_candidates(
    *,
    gaussians: list[np.ndarray],
    supports: list[np.ndarray],
    validity: np.ndarray,
    octave: int,
    original_shape: tuple[int, int],
    config: SphericalDoGDetectorConfig,
) -> tuple[list[_Candidate], SphericalOctaveDetectionDiagnostics]:
    """Detect and refine candidates from one already materialized octave."""

    dogs = [
        gaussians[index + 1].astype(np.float64) - gaussians[index].astype(np.float64)
        for index in range(len(gaussians) - 1)
    ]
    step = math.pi / gaussians[0].shape[0]
    candidates: list[_Candidate] = []
    raw_count = 0
    refined_count = 0
    contrast_count = 0
    edge_count = 0
    valid_count = 0
    nonconverged_count = 0
    ill_conditioned_count = 0
    preliminary_contrast = 0.5 * config.contrast_threshold / config.levels_per_octave
    for level in range(1, len(dogs) - 1):
        extrema = _dog_extrema_candidates_mask(
            dogs[level - 1],
            dogs[level],
            dogs[level + 1],
            step,
            preliminary_contrast,
        )
        yy, xx = np.nonzero(extrema)
        raw_count += int(len(xx))
        initial_bearings = (
            erp_pixels_to_rays(
                np.column_stack((xx, yy)).astype(np.float64, copy=False),
                dogs[0].shape,
            )
            if len(xx)
            else np.empty((0, 3), dtype=np.float64)
        )
        initial_cubes = _sample_initial_dog_cubes(dogs, level, initial_bearings, step)
        for candidate_index, (x_index, y_index) in enumerate(zip(xx, yy, strict=True)):
            result = _refine_dog_extremum(
                dogs=dogs,
                supports=supports,
                validity=validity,
                initial_xy=(int(x_index), int(y_index)),
                initial_level=level,
                octave=octave,
                original_shape=original_shape,
                step_rad=step,
                config=config,
                initial_bearing=initial_bearings[candidate_index],
                initial_cube=initial_cubes[candidate_index],
            )
            reason = result.rejection_reason
            if reason in {"nonconverged", "unstable-offset"}:
                nonconverged_count += 1
                continue
            if reason == "ill-conditioned-hessian":
                ill_conditioned_count += 1
                continue
            refined_count += 1
            if reason == "low-interpolated-contrast":
                continue
            contrast_count += 1
            if reason == "edge-response":
                continue
            edge_count += 1
            if reason == "invalid-support":
                continue
            if result.candidate is None:
                raise RuntimeError("refinement accepted without a candidate")
            candidates.append(result.candidate)
            valid_count += 1
    return candidates, SphericalOctaveDetectionDiagnostics(
        octave=octave,
        shape_hw=tuple(gaussians[0].shape),
        raw_extrema_count=raw_count,
        refinement_converged_count=refined_count,
        interpolated_contrast_count=contrast_count,
        refined_edge_count=edge_count,
        valid_support_count=valid_count,
        rejected_nonconverged_count=nonconverged_count,
        rejected_ill_conditioned_count=ill_conditioned_count,
    )


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
    suppressed = np.zeros(len(ordered), dtype=bool)
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
