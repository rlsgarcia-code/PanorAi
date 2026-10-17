"""Full-sphere multibaseline stereo on an overlapping cube-oriented atlas.

The cube atlas is an execution strategy, not a different camera model.  Each
face stores canonical reference-camera bearings.  A radial hypothesis is
therefore transformed directly into every registered source camera and sampled
from that source ERP.  Multiple non-collinear baselines make directions that
are degenerate for one pair observable from another pair.

Sampling coverage, stereo observability, and accepted numerical validity are
reported separately.  The estimator never fills a direction that lacks enough
source evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping
import math

import cv2
import numpy as np

from panorai.data import EquirectangularImage
from panorai.geometry import (
    GnomonicProjector,
    GnomonicSpec,
    gnomonic_pixels_to_rays,
    rays_to_erp_pixels,
)

from .._dense import (
    SphericalStereoOptions,
    _aggregate_cost_volume,
    _best_two_costs,
    _feature_cost,
    _matching_features,
    _prepare_image,
    _refine_inverse_range,
    _sample_wrapped_nearest,
    _seam_safe_remap,
    _validate_pose,
)
from ..classical._operations import _census, _hamming, _mgm_one
from ._fusion import compose_fusion_confidence
from ._types import PanoramicStereoPose


_INTERFACE = "panorai-full-sphere-multibaseline-stereo/v1-experimental"
# Reviewed defaults preserve the angular support represented by the historical
# 256-row ERP feature grid and 128-pixel, 100-degree reference face.  They are
# physical angular spacings, not a request to preserve either raster size.
_DEFAULT_ERP_FEATURE_TAP_DEG = 0.703125
_DEFAULT_FACE_COST_TAP_DEG = 1.0667900072919114
_FACE_CENTRES = (
    ("front", 0.0, 0.0),
    ("right", 0.0, 90.0),
    ("back", 0.0, -180.0),
    ("left", 0.0, -90.0),
    ("up", 90.0, 0.0),
    ("down", -90.0, 0.0),
)
_ACCEPTANCE_ORDER = (
    "reference_support",
    "interior_hypothesis",
    "finite_range",
    "texture",
    "confidence",
    "matching_cost",
    "source_evidence",
    "range_consensus",
    "parallax",
    "accepted",
)


def _readonly(value: Any, dtype: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {result.shape}")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class OverlappingCubemapPlanner:
    """Plan six cube-oriented gnomonic tiles with deliberate edge overlap."""

    face_size: int = 128
    face_fov_deg: float = 100.0

    def __post_init__(self) -> None:
        if isinstance(self.face_size, bool) or not isinstance(
            self.face_size, (int, np.integer)
        ):
            raise TypeError("face_size must be an integer")
        if self.face_size < 8:
            raise ValueError("face_size must be at least 8")
        if not math.isfinite(self.face_fov_deg) or not 90.0 < self.face_fov_deg < 180.0:
            raise ValueError("face_fov_deg must be finite and between 90 and 180")

    def plan(self) -> tuple[tuple[str, GnomonicSpec], ...]:
        """Return canonical front/right/back/left/up/down tile specifications."""

        shape = (int(self.face_size), int(self.face_size))
        return tuple(
            (
                face_id,
                GnomonicSpec(
                    center_lat_deg=latitude,
                    center_lon_deg=longitude,
                    hfov_deg=float(self.face_fov_deg),
                    vfov_deg=float(self.face_fov_deg),
                    output_shape_hw=shape,
                ),
            )
            for face_id, latitude, longitude in _FACE_CENTRES
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class StereoSourceView:
    """One source view and its directed relation to a set reference.

    This object is not independently "registered". Its pose has meaning only
    inside :class:`RegisteredPanoramaSet` and uses
    ``X_source = R_source_from_reference @ X_reference + t``. Translation
    scale defines the output radial-range unit.
    """

    source_id: str
    panorama: Any
    pose: PanoramicStereoPose
    validity: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id must be a non-empty string")


@dataclass(frozen=True, slots=True, kw_only=True)
class RegisteredPanoramaSet:
    """A reference panorama and every source registered to that reference.

    Registration is a relation among views, so the aggregate—not an
    individual image—is the public stereo input. Every source pose maps from
    the reference frame named by ``reference_id`` into that source frame.
    """

    reference_id: str
    reference: Any
    sources: tuple[StereoSourceView, ...]
    reference_validity: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reference_id, str) or not self.reference_id.strip():
            raise ValueError("reference_id must be a non-empty string")
        sources = tuple(self.sources)
        if not sources:
            raise ValueError("at least one stereo source view is required")
        if any(not isinstance(source, StereoSourceView) for source in sources):
            raise TypeError("sources must contain only StereoSourceView objects")
        source_ids = tuple(source.source_id for source in sources)
        if self.reference_id in source_ids:
            raise ValueError("reference_id must differ from every source_id")
        if len(set(source_ids)) != len(source_ids):
            raise ValueError(
                "source_id values must be unique within the registered set"
            )
        units = {source.pose.units for source in sources}
        scale_status = {source.pose.scale_status for source in sources}
        if len(units) != 1 or len(scale_status) != 1:
            raise ValueError(
                "all source poses must use the same units and scale status"
            )
        object.__setattr__(self, "sources", sources)


def _default_stereo_options() -> SphericalStereoOptions:
    return SphericalStereoOptions(
        pole_margin_fraction=0.0,
        bidirectional_consistency=False,
        pyramid_levels=1,
    )


@dataclass(frozen=True, slots=True)
class FullSphereStereoOptions:
    """Configuration for cube-atlas multibaseline plane sweep."""

    planner: OverlappingCubemapPlanner = field(
        default_factory=OverlappingCubemapPlanner
    )
    stereo: SphericalStereoOptions = field(default_factory=_default_stereo_options)
    min_parallax_deg: float = 0.25
    min_source_evidence: int = 1
    max_sources_per_hypothesis: int | None = None
    source_aggregation: str = "median"
    matching_cost: str = "ad-census"
    local_aggregation: str = "box"
    regularizer: str = "sgm4"
    census_radius: int = 2
    ad_weight: float = 0.25
    census_weight: float = 0.75
    ad_truncation: float = 0.20
    erp_feature_tap_deg: float | None = _DEFAULT_ERP_FEATURE_TAP_DEG
    face_cost_tap_deg: float | None = _DEFAULT_FACE_COST_TAP_DEG
    range_consensus_tolerance_steps: int = 2
    min_range_consensus_sources: int = 0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.min_parallax_deg)
            or not 0.0 <= self.min_parallax_deg < 90.0
        ):
            raise ValueError("min_parallax_deg must be finite and in [0, 90)")
        if isinstance(self.min_source_evidence, bool) or not isinstance(
            self.min_source_evidence, (int, np.integer)
        ):
            raise TypeError("min_source_evidence must be an integer")
        if self.min_source_evidence < 1:
            raise ValueError("min_source_evidence must be at least 1")
        maximum = self.max_sources_per_hypothesis
        if maximum is not None and (
            isinstance(maximum, bool)
            or not isinstance(maximum, (int, np.integer))
            or maximum < 1
        ):
            raise ValueError(
                "max_sources_per_hypothesis must be None or a positive integer"
            )
        if self.source_aggregation not in {"median", "weighted-mean", "best"}:
            raise ValueError(
                "source_aggregation must be 'median', 'weighted-mean', or 'best'"
            )
        if self.matching_cost not in {
            "intensity-gradient",
            "ad-census",
            "multiscale-ad-census",
        }:
            raise ValueError(
                "matching_cost must be 'intensity-gradient', 'ad-census', "
                "or 'multiscale-ad-census'"
            )
        if self.local_aggregation not in {"none", "box"}:
            raise ValueError("local_aggregation must be 'none' or 'box'")
        if self.regularizer not in {"none", "sgm4", "tmgm"}:
            raise ValueError("regularizer must be 'none', 'sgm4', or 'tmgm'")
        if (
            isinstance(self.census_radius, bool)
            or not isinstance(self.census_radius, (int, np.integer))
            or not 1 <= self.census_radius <= 3
        ):
            raise ValueError("census_radius must be an integer in [1, 3]")
        if not math.isclose(self.ad_weight + self.census_weight, 1.0, abs_tol=1e-12):
            raise ValueError("ad_weight and census_weight must sum to 1")
        if self.ad_weight < 0.0 or self.census_weight < 0.0:
            raise ValueError("matching-cost weights must be non-negative")
        if not math.isfinite(self.ad_truncation) or not 0.0 < self.ad_truncation <= 1.0:
            raise ValueError("ad_truncation must be finite and in (0, 1]")
        for name, value in (
            ("erp_feature_tap_deg", self.erp_feature_tap_deg),
            ("face_cost_tap_deg", self.face_cost_tap_deg),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be None or a finite positive value")
        if (
            isinstance(self.range_consensus_tolerance_steps, bool)
            or not isinstance(self.range_consensus_tolerance_steps, (int, np.integer))
            or self.range_consensus_tolerance_steps < 0
        ):
            raise ValueError(
                "range_consensus_tolerance_steps must be a non-negative integer"
            )
        if (
            isinstance(self.min_range_consensus_sources, bool)
            or not isinstance(self.min_range_consensus_sources, (int, np.integer))
            or self.min_range_consensus_sources < 0
        ):
            raise ValueError(
                "min_range_consensus_sources must be a non-negative integer"
            )
        if self.stereo.bidirectional_consistency:
            raise ValueError(
                "multibaseline cube stereo does not use pairwise bidirectional consistency; "
                "set stereo.bidirectional_consistency=False"
            )
        if self.stereo.pyramid_levels != 1:
            raise ValueError(
                "multibaseline cube stereo currently requires pyramid_levels=1"
            )


@dataclass(frozen=True, slots=True)
class FullSphereStereoResult:
    """Fused reference-ERP radial range and explicit evidence masks.

    ``diagnostics['face_acceptance_cumulative_counts']`` reports the exact
    cumulative predicates used by the Experimental face estimator. Counts are
    face pixels, so overlapping faces are intentionally counted separately;
    they are not unique ERP directions or evaluation metrics.
    """

    radial_range: np.ndarray
    validity: np.ndarray
    confidence: np.ndarray
    disagreement: np.ndarray
    sampling_support: np.ndarray
    observable: np.ndarray
    max_parallax_deg: np.ndarray
    source_count: np.ndarray
    source_consensus_count: np.ndarray
    contribution_count: np.ndarray
    source_ids: tuple[str, ...]
    diagnostics: Mapping[str, Any]
    interface: str = _INTERFACE
    matching_confidence: np.ndarray | None = None
    consistency_confidence: np.ndarray | None = None

    def __post_init__(self) -> None:
        radial = np.asarray(self.radial_range)
        if radial.ndim != 2:
            raise ValueError("radial_range must have shape HW")
        shape = radial.shape
        radial = _readonly(radial, np.float32, "radial_range", shape)
        validity = _readonly(self.validity, np.bool_, "validity", shape)
        confidence = _readonly(self.confidence, np.float32, "confidence", shape)
        matching_confidence = _readonly(
            confidence
            if self.matching_confidence is None
            else self.matching_confidence,
            np.float32,
            "matching_confidence",
            shape,
        )
        consistency_confidence = _readonly(
            (
                np.where(validity, 1.0, 0.0)
                if self.consistency_confidence is None
                else self.consistency_confidence
            ),
            np.float32,
            "consistency_confidence",
            shape,
        )
        disagreement = _readonly(self.disagreement, np.float32, "disagreement", shape)
        sampling = _readonly(self.sampling_support, np.bool_, "sampling_support", shape)
        observable = _readonly(self.observable, np.bool_, "observable", shape)
        parallax = _readonly(
            self.max_parallax_deg, np.float32, "max_parallax_deg", shape
        )
        source_count = _readonly(self.source_count, np.uint16, "source_count", shape)
        source_consensus_count = _readonly(
            self.source_consensus_count,
            np.uint16,
            "source_consensus_count",
            shape,
        )
        contribution_count = _readonly(
            self.contribution_count, np.uint16, "contribution_count", shape
        )
        if np.any(validity & ~np.isfinite(radial)):
            raise ValueError("valid radial-range samples must be finite")
        if np.any(~validity & ~np.isnan(radial)):
            raise ValueError("invalid radial-range samples must be NaN")
        if np.any(validity & ~observable):
            raise ValueError("a valid range sample must be observable")
        if np.any(observable & ~sampling):
            raise ValueError("an observable direction must be covered by the atlas")
        for name, value in (
            ("confidence", confidence),
            ("matching_confidence", matching_confidence),
            ("consistency_confidence", consistency_confidence),
        ):
            if np.any(~np.isfinite(value)) or np.any((value < 0.0) | (value > 1.0)):
                raise ValueError(f"{name} must be finite and in [0, 1]")
            if np.any(value[~validity] != 0.0):
                raise ValueError(f"invalid {name} samples must be zero")
        if not np.allclose(
            confidence,
            matching_confidence * consistency_confidence,
            rtol=1e-6,
            atol=1e-7,
        ):
            raise ValueError(
                "confidence must equal matching_confidence * consistency_confidence"
            )
        object.__setattr__(self, "radial_range", radial)
        object.__setattr__(self, "validity", validity)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "matching_confidence", matching_confidence)
        object.__setattr__(self, "consistency_confidence", consistency_confidence)
        object.__setattr__(self, "disagreement", disagreement)
        object.__setattr__(self, "sampling_support", sampling)
        object.__setattr__(self, "observable", observable)
        object.__setattr__(self, "max_parallax_deg", parallax)
        object.__setattr__(self, "source_count", source_count)
        object.__setattr__(self, "source_consensus_count", source_consensus_count)
        object.__setattr__(self, "contribution_count", contribution_count)
        object.__setattr__(self, "source_ids", tuple(self.source_ids))
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics))
        )

    @property
    def range(self) -> np.ndarray:
        """Compatibility-style alias for radial range."""

        return self.radial_range


@dataclass(frozen=True, slots=True)
class _PreparedSource:
    source_id: str
    gray: np.ndarray
    feature: np.ndarray
    validity: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    centre_reference: np.ndarray


@dataclass(frozen=True, slots=True)
class _FaceEstimate:
    radial_range: np.ndarray
    validity: np.ndarray
    confidence: np.ndarray
    max_parallax_deg: np.ndarray
    source_count: np.ndarray
    source_consensus_count: np.ndarray
    acceptance_cumulative_counts: Mapping[str, int]


class FullSphereMultiViewStereo:
    """Estimate dense radial range with an overlapping cubemap and many baselines."""

    def __init__(self, options: FullSphereStereoOptions | None = None) -> None:
        self.options = options or FullSphereStereoOptions()

    def estimate(
        self,
        registered: RegisteredPanoramaSet,
    ) -> FullSphereStereoResult:
        """Estimate range for one explicit reference-centred registered set."""

        if not isinstance(registered, RegisteredPanoramaSet):
            raise TypeError("registered must be a RegisteredPanoramaSet")
        sources = registered.sources
        if self.options.min_range_consensus_sources > len(sources):
            raise ValueError(
                "min_range_consensus_sources cannot exceed the registered source count"
            )
        source_ids = tuple(item.source_id for item in sources)
        reference_image, reference_mask = _panorama_payload(
            registered.reference,
            registered.reference_validity,
            f"reference {registered.reference_id!r}",
        )
        reference_gray = _prepare_image(reference_image, "reference")
        reference_feature, reference_texture = _matching_features(
            reference_gray,
            self.options.stereo.window_size,
            self.options.stereo.filter_backend,
            angular_step_deg=self.options.erp_feature_tap_deg,
        )
        prepared = tuple(self._prepare_source(item) for item in sources)
        units = {item.pose.units for item in sources}
        scale = {item.pose.scale_status for item in sources}
        shape = reference_gray.shape
        sum_weight = np.zeros(shape, dtype=np.float64)
        sum_taper = np.zeros(shape, dtype=np.float64)
        sum_range = np.zeros(shape, dtype=np.float64)
        sum_squared = np.zeros(shape, dtype=np.float64)
        contribution_count = np.zeros(shape, dtype=np.uint16)
        sampling_count = np.zeros(shape, dtype=np.uint8)
        source_count = np.zeros(shape, dtype=np.uint16)
        source_consensus_count = np.zeros(shape, dtype=np.uint16)
        max_parallax = np.zeros(shape, dtype=np.float32)
        face_valid_fractions: dict[str, float] = {}
        face_acceptance_counts: dict[str, dict[str, int]] = {}

        for face_id, spec in self.options.planner.plan():
            projector = GnomonicProjector(
                spec,
                interpolation="bilinear",
                invalid_policy="renormalize",
                min_valid_weight=0.5,
            )
            reference_face = projector.project(
                reference_feature, validity_mask=reference_mask
            )
            texture_face = projector.project(
                reference_texture, validity_mask=reference_mask
            )
            guidance_face = projector.project(
                reference_gray, validity_mask=reference_mask
            )
            support = np.array(reference_face.support_mask, dtype=bool, copy=True)
            if reference_face.validity_mask is not None:
                support &= np.asarray(reference_face.validity_mask, dtype=bool)
            if texture_face.validity_mask is not None:
                support &= np.asarray(texture_face.validity_mask, dtype=bool)
            if guidance_face.validity_mask is not None:
                support &= np.asarray(guidance_face.validity_mask, dtype=bool)
            face = self._estimate_face(
                spec,
                np.asarray(reference_face.data, dtype=np.float32),
                np.asarray(texture_face.data, dtype=np.float32),
                np.asarray(guidance_face.data, dtype=np.float32),
                support,
                prepared,
            )
            face_valid_fractions[face_id] = float(face.validity.mean())
            face_acceptance_counts[face_id] = dict(face.acceptance_cumulative_counts)
            self._accumulate_face(
                spec,
                face,
                shape,
                sum_weight,
                sum_taper,
                sum_range,
                sum_squared,
                contribution_count,
                sampling_count,
                source_count,
                source_consensus_count,
                max_parallax,
            )

        sampling_support = sampling_count > 0
        observable = (
            sampling_support
            & (source_count >= self.options.min_source_evidence)
            & (max_parallax >= self.options.min_parallax_deg)
        )
        validity = (
            (sum_weight > 1e-8)
            & observable
            & reference_mask
            & (source_consensus_count >= self.options.min_range_consensus_sources)
        )
        radial = np.full(shape, np.nan, dtype=np.float32)
        radial[validity] = (sum_range[validity] / sum_weight[validity]).astype(
            np.float32
        )
        variance = np.zeros(shape, dtype=np.float64)
        variance[validity] = np.maximum(
            0.0,
            sum_squared[validity] / sum_weight[validity]
            - radial[validity].astype(np.float64) ** 2,
        )
        disagreement = np.full(shape, np.nan, dtype=np.float32)
        disagreement[validity] = np.sqrt(variance[validity]).astype(np.float32)
        matching_confidence, consistency_confidence, confidence = (
            compose_fusion_confidence(
                sum_weight,
                sum_taper,
                radial,
                disagreement,
                validity,
            )
        )
        return FullSphereStereoResult(
            radial,
            validity,
            confidence,
            disagreement,
            sampling_support,
            observable,
            max_parallax,
            source_count,
            source_consensus_count,
            contribution_count,
            source_ids,
            {
                "route": "overlapping-cubemap-multibaseline",
                "reference_id": registered.reference_id,
                "quantity": "radial_range",
                "range_units": next(iter(units)),
                "scale_status": next(iter(scale)),
                "feature_cost": {
                    "matching_cost": self.options.matching_cost,
                    **(
                        {
                            "intensity_weight": self.options.stereo.intensity_weight,
                            "gradient_weight": self.options.stereo.gradient_weight,
                            "source_gradient_basis": "transported-to-reference",
                        }
                        if self.options.matching_cost == "intensity-gradient"
                        else {
                            "census_radius": self.options.census_radius,
                            "ad_weight": self.options.ad_weight,
                            "census_weight": self.options.census_weight,
                            "ad_truncation": self.options.ad_truncation,
                            **(
                                {"census_tap_scales": (1, 2)}
                                if self.options.matching_cost == "multiscale-ad-census"
                                else {}
                            ),
                        }
                    ),
                },
                "cost_aggregation": {
                    "local": self.options.local_aggregation,
                    "regularizer": self.options.regularizer,
                    "window_size": self.options.stereo.window_size,
                },
                "angular_neighbourhood": {
                    "erp_feature_tap_deg": self.options.erp_feature_tap_deg,
                    "face_cost_tap_deg": self.options.face_cost_tap_deg,
                    "face_tap_stride_by_id": {
                        face_id: _face_tap_stride(spec, self.options.face_cost_tap_deg)
                        for face_id, spec in self.options.planner.plan()
                    },
                },
                "face_count": len(self.options.planner.plan()),
                "source_count": len(sources),
                "range_consensus_tolerance_steps": self.options.range_consensus_tolerance_steps,
                "min_range_consensus_sources": self.options.min_range_consensus_sources,
                "sampling_coverage_fraction": float(sampling_support.mean()),
                "observable_fraction": float(observable.mean()),
                "valid_fraction": float(validity.mean()),
                "face_valid_fractions": face_valid_fractions,
                "face_acceptance_order": _ACCEPTANCE_ORDER,
                "face_acceptance_cumulative_counts": face_acceptance_counts,
                "face_acceptance_cumulative_totals": {
                    name: sum(
                        counts[name] for counts in face_acceptance_counts.values()
                    )
                    for name in _ACCEPTANCE_ORDER
                },
                "face_acceptance_semantics": (
                    "cumulative face-pixel counts in predicate order; overlapping "
                    "faces are counted separately and these are not ERP evaluation metrics"
                ),
                "streaming": "one reference face and one hypothesis source stack at a time",
                "confidence": (
                    "taper-normalized matching margin multiplied by "
                    "exp(-range_disagreement/radial_range)"
                ),
            },
            matching_confidence=matching_confidence,
            consistency_confidence=consistency_confidence,
        )

    def _prepare_source(self, source: StereoSourceView) -> _PreparedSource:
        image, validity = _panorama_payload(
            source.panorama, source.validity, f"source {source.source_id!r}"
        )
        gray = _prepare_image(image, f"source {source.source_id!r}")
        feature, _ = _matching_features(
            gray,
            self.options.stereo.window_size,
            self.options.stereo.filter_backend,
            angular_step_deg=self.options.erp_feature_tap_deg,
        )
        rotation, translation = _validate_pose(
            source.pose.rotation_source_from_reference,
            source.pose.translation_source_from_reference,
        )
        centre = -rotation.T @ translation
        return _PreparedSource(
            source.source_id,
            gray,
            feature,
            validity,
            rotation,
            translation,
            centre,
        )

    def _estimate_face(
        self,
        spec: GnomonicSpec,
        reference_feature: np.ndarray,
        reference_texture: np.ndarray,
        guidance: np.ndarray,
        reference_support: np.ndarray,
        sources: tuple[_PreparedSource, ...],
    ) -> _FaceEstimate:
        settings = self.options.stereo
        height, width = spec.output_shape_hw
        yy, xx = np.indices((height, width), dtype=np.float64)
        pixels = np.stack((xx, yy), axis=-1)
        rays = np.asarray(
            gnomonic_pixels_to_rays(pixels, spec).rays_xyz, dtype=np.float32
        )
        inverse_ranges = np.linspace(
            1.0 / settings.max_range,
            1.0 / settings.min_range,
            settings.num_hypotheses,
            dtype=np.float32,
        )
        volume_hwd = np.empty(
            (height, width, settings.num_hypotheses), dtype=np.float32
        )
        volume_valid_hwd = np.zeros_like(volume_hwd, dtype=bool)
        tap_stride = _face_tap_stride(spec, self.options.face_cost_tap_deg)
        reference_census = (
            _census(
                guidance,
                self.options.census_radius,
                tap_stride=tap_stride,
            )
            if self.options.matching_cost in {"ad-census", "multiscale-ad-census"}
            else None
        )
        reference_census_secondary = (
            _census(
                guidance,
                self.options.census_radius,
                tap_stride=2 * tap_stride,
            )
            if self.options.matching_cost == "multiscale-ad-census"
            else None
        )
        source_best_cost = np.full(
            (len(sources), height, width), np.inf, dtype=np.float32
        )
        source_best_index = np.full((len(sources), height, width), -1, dtype=np.int32)
        for hypothesis_index, inverse_range in enumerate(inverse_ranges):
            points = rays / float(inverse_range)
            costs, parallaxes, validities = self._source_evidence(
                reference_feature,
                guidance,
                reference_census,
                reference_census_secondary,
                rays,
                points,
                sources,
                tap_stride,
            )
            for source_index in range(len(sources)):
                improved = validities[source_index] & (
                    costs[source_index] < source_best_cost[source_index]
                )
                source_best_cost[source_index][improved] = costs[source_index][improved]
                source_best_index[source_index][improved] = hypothesis_index
            selected = _top_source_mask(
                parallaxes,
                validities,
                self.options.max_sources_per_hypothesis,
            )
            weights = np.where(selected, np.sin(np.deg2rad(parallaxes)) ** 2, 0.0)
            count = selected.sum(axis=0)
            combined = _combine_source_costs(
                costs,
                weights,
                selected,
                self.options.source_aggregation,
            )
            combined_valid = count >= self.options.min_source_evidence
            volume_hwd[..., hypothesis_index] = np.where(combined_valid, combined, 1.0)
            volume_valid_hwd[..., hypothesis_index] = combined_valid

        volume_hwd, volume_valid_hwd = _local_cost_aggregation(
            volume_hwd,
            volume_valid_hwd,
            settings.window_size,
            self.options.local_aggregation,
            tap_stride=tap_stride,
        )
        aggregated = _regularize_cost_volume(
            volume_hwd,
            volume_valid_hwd,
            guidance,
            settings,
            self.options.regularizer,
        )
        indices, best, second_best = _best_two_costs(aggregated)
        confidence = np.clip(
            (second_best - best) / np.maximum(second_best, np.finfo(np.float32).eps),
            0.0,
            1.0,
        ).astype(np.float32)
        refined_inverse = _refine_inverse_range(aggregated, indices, inverse_ranges)
        radial = np.reciprocal(refined_inverse).astype(np.float32)
        points = rays * radial[..., None]
        _, parallaxes, validities = self._source_evidence(
            reference_feature,
            guidance,
            reference_census,
            reference_census_secondary,
            rays,
            points,
            sources,
            tap_stride,
        )
        selected = _top_source_mask(
            parallaxes,
            validities,
            self.options.max_sources_per_hypothesis,
        )
        source_count = selected.sum(axis=0).astype(np.uint16)
        consensus = (
            selected
            & (source_best_index >= 0)
            & (
                np.abs(source_best_index - indices[None, ...])
                <= self.options.range_consensus_tolerance_steps
            )
        )
        source_consensus_count = consensus.sum(axis=0).astype(np.uint16)
        max_parallax = np.max(np.where(validities, parallaxes, 0.0), axis=0).astype(
            np.float32
        )
        cumulative = np.asarray(reference_support, dtype=bool).copy()
        acceptance_counts = {"reference_support": int(cumulative.sum())}
        cumulative &= (indices > 0) & (indices < settings.num_hypotheses - 1)
        acceptance_counts["interior_hypothesis"] = int(cumulative.sum())
        cumulative &= np.isfinite(radial)
        acceptance_counts["finite_range"] = int(cumulative.sum())
        cumulative &= reference_texture >= settings.min_texture_std
        acceptance_counts["texture"] = int(cumulative.sum())
        cumulative &= confidence >= settings.min_confidence
        acceptance_counts["confidence"] = int(cumulative.sum())
        cumulative &= best <= settings.max_matching_cost
        acceptance_counts["matching_cost"] = int(cumulative.sum())
        cumulative &= source_count >= self.options.min_source_evidence
        acceptance_counts["source_evidence"] = int(cumulative.sum())
        cumulative &= source_consensus_count >= self.options.min_range_consensus_sources
        acceptance_counts["range_consensus"] = int(cumulative.sum())
        cumulative &= max_parallax >= self.options.min_parallax_deg
        acceptance_counts["parallax"] = int(cumulative.sum())
        validity = cumulative
        acceptance_counts["accepted"] = int(validity.sum())
        return _FaceEstimate(
            np.where(validity, radial, np.nan).astype(np.float32),
            validity,
            confidence,
            max_parallax,
            source_count,
            source_consensus_count,
            MappingProxyType(acceptance_counts),
        )

    def _source_evidence(
        self,
        reference_feature: np.ndarray,
        reference_gray: np.ndarray,
        reference_census: tuple[np.ndarray, np.ndarray] | None,
        reference_census_secondary: tuple[np.ndarray, np.ndarray] | None,
        rays: np.ndarray,
        points: np.ndarray,
        sources: tuple[_PreparedSource, ...],
        tap_stride: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        costs = []
        parallaxes = []
        validities = []
        for source in sources:
            points_source = points @ source.rotation.T + source.translation
            projected = rays_to_erp_pixels(points_source, source.gray.shape)
            sampled_validity = _sample_wrapped_nearest(
                source.validity.astype(np.float32), projected.pixels_xy
            )
            direction_from_source = points - source.centre_reference
            direction_from_source /= np.maximum(
                np.linalg.norm(direction_from_source, axis=-1, keepdims=True), 1e-12
            )
            cosine = np.clip(
                np.einsum("...i,...i->...", rays, direction_from_source), -1.0, 1.0
            )
            parallax = np.degrees(np.arccos(cosine)).astype(np.float32)
            geometric_support = np.asarray(projected.valid, dtype=bool)
            valid = (
                geometric_support
                & np.isfinite(sampled_validity)
                & (sampled_validity >= 0.5)
                & (parallax >= self.options.min_parallax_deg)
            )
            if self.options.matching_cost in {
                "ad-census",
                "multiscale-ad-census",
            }:
                assert reference_census is not None
                geometric_valid = valid
                warped_gray, remap_support = _seam_safe_remap(
                    source.gray, projected.pixels_xy
                )
                geometric_valid = geometric_valid & remap_support
                cost, cost_valid = _ad_census_face_cost(
                    reference_gray,
                    np.asarray(warped_gray[..., 0], dtype=np.float32),
                    geometric_valid,
                    reference_census,
                    self.options,
                    tap_stride=tap_stride,
                )
                valid = cost_valid
                if self.options.matching_cost == "multiscale-ad-census":
                    assert reference_census_secondary is not None
                    secondary_cost, secondary_valid = _ad_census_face_cost(
                        reference_gray,
                        np.asarray(warped_gray[..., 0], dtype=np.float32),
                        geometric_valid,
                        reference_census_secondary,
                        self.options,
                        tap_stride=2 * tap_stride,
                    )
                    valid &= secondary_valid
                    cost = 0.5 * (cost + secondary_cost)
            else:
                warped, remap_support = _seam_safe_remap(
                    source.feature, projected.pixels_xy
                )
                valid &= remap_support
                aligned = _align_source_features(
                    np.asarray(warped, dtype=np.float32),
                    points_source,
                    rays,
                    source.rotation,
                )
                cost = _feature_cost(reference_feature, aligned, self.options.stereo)
            costs.append(cost)
            parallaxes.append(parallax)
            validities.append(valid)
        return (
            np.stack(costs, axis=0),
            np.stack(parallaxes, axis=0),
            np.stack(validities, axis=0),
        )

    @staticmethod
    def _accumulate_face(
        spec: GnomonicSpec,
        face: _FaceEstimate,
        output_shape: tuple[int, int],
        sum_weight: np.ndarray,
        sum_taper: np.ndarray,
        sum_range: np.ndarray,
        sum_squared: np.ndarray,
        contribution_count: np.ndarray,
        sampling_count: np.ndarray,
        source_count: np.ndarray,
        source_consensus_count: np.ndarray,
        max_parallax: np.ndarray,
    ) -> None:
        projector = GnomonicProjector(
            spec,
            interpolation="bilinear",
            invalid_policy="renormalize",
            min_valid_weight=0.5,
        )
        nearest = GnomonicProjector(spec, interpolation="nearest")
        ones = np.ones(spec.output_shape_hw, dtype=np.float32)
        sampling = nearest.back_project(ones, output_shape)
        sampling_count[np.asarray(sampling.support_mask, dtype=bool)] += 1
        taper = _edge_taper(spec.output_shape_hw)
        face_weight = np.where(face.validity, face.confidence * taper, 0.0).astype(
            np.float32
        )
        finite_range_storage = np.where(face.validity, face.radial_range, 0.0).astype(
            np.float32
        )
        range_erp = projector.back_project(
            finite_range_storage, output_shape, validity_mask=face.validity
        )
        weight_erp = projector.back_project(
            face_weight, output_shape, validity_mask=face.validity
        )
        taper_erp = projector.back_project(
            np.where(face.validity, taper, 0.0).astype(np.float32),
            output_shape,
            validity_mask=face.validity,
        )
        supported = np.array(range_erp.support_mask, dtype=bool, copy=True)
        if range_erp.validity_mask is not None:
            supported &= np.asarray(range_erp.validity_mask, dtype=bool)
        radial = np.asarray(range_erp.data, dtype=np.float64)
        weight = np.asarray(weight_erp.data, dtype=np.float64)
        taper_weight = np.asarray(taper_erp.data, dtype=np.float64)
        supported &= np.asarray(taper_erp.support_mask, dtype=bool)
        supported &= (
            np.isfinite(radial)
            & np.isfinite(weight)
            & np.isfinite(taper_weight)
            & (weight > 0.0)
            & (taper_weight > 0.0)
        )
        sum_weight[supported] += weight[supported]
        sum_taper[supported] += taper_weight[supported]
        sum_range[supported] += weight[supported] * radial[supported]
        sum_squared[supported] += weight[supported] * radial[supported] ** 2
        contribution_count[supported] = np.minimum(
            np.iinfo(np.uint16).max,
            contribution_count[supported].astype(np.uint32) + 1,
        ).astype(np.uint16)
        validity_erp = nearest.back_project(
            face.validity.astype(np.uint8), output_shape
        )
        parallax_erp = nearest.back_project(face.max_parallax_deg, output_shape)
        count_erp = nearest.back_project(face.source_count, output_shape)
        consensus_erp = nearest.back_project(face.source_consensus_count, output_shape)
        parallax_support = np.array(parallax_erp.support_mask, dtype=bool, copy=True)
        count_support = np.array(count_erp.support_mask, dtype=bool, copy=True)
        consensus_support = np.array(consensus_erp.support_mask, dtype=bool, copy=True)
        valid_nearest = np.asarray(validity_erp.support_mask, dtype=bool) & (
            np.asarray(validity_erp.data, dtype=np.uint8) > 0
        )
        consensus_support &= valid_nearest
        max_parallax[parallax_support] = np.maximum(
            max_parallax[parallax_support],
            np.asarray(parallax_erp.data, dtype=np.float32)[parallax_support],
        )
        source_count[count_support] = np.maximum(
            source_count[count_support],
            np.asarray(count_erp.data, dtype=np.uint16)[count_support],
        )
        source_consensus_count[consensus_support] = np.maximum(
            source_consensus_count[consensus_support],
            np.asarray(consensus_erp.data, dtype=np.uint16)[consensus_support],
        )


def _erp_tangent_basis(rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return unit east/north vectors for canonical PanorAi bearings."""

    bearings = np.asarray(rays, dtype=np.float64)
    horizontal = np.hypot(bearings[..., 0], bearings[..., 2])
    safe = np.maximum(horizontal, 1e-12)
    east = np.stack(
        (
            bearings[..., 2] / safe,
            np.zeros_like(horizontal),
            -bearings[..., 0] / safe,
        ),
        axis=-1,
    )
    north = np.cross(bearings, east)
    east = np.where((horizontal > 1e-12)[..., None], east, (1.0, 0.0, 0.0))
    pole_north = np.stack(
        (
            np.zeros_like(horizontal),
            np.zeros_like(horizontal),
            -np.sign(bearings[..., 1]),
        ),
        axis=-1,
    )
    north = np.where((horizontal > 1e-12)[..., None], north, pole_north)
    return east.astype(np.float32), north.astype(np.float32)


def _align_source_features(
    warped_source: np.ndarray,
    points_source: np.ndarray,
    reference_rays: np.ndarray,
    rotation_source_from_reference: np.ndarray,
) -> np.ndarray:
    """Transport source tangent-gradient channels into the reference basis."""

    source_bearings = points_source / np.maximum(
        np.linalg.norm(points_source, axis=-1, keepdims=True), 1e-12
    )
    source_east, source_north = _erp_tangent_basis(source_bearings)
    gradient_source = (
        warped_source[..., 1, None] * source_east
        + warped_source[..., 2, None] * source_north
    )
    gradient_reference = gradient_source @ np.asarray(
        rotation_source_from_reference, dtype=np.float32
    )
    reference_east, reference_north = _erp_tangent_basis(reference_rays)
    return np.stack(
        (
            warped_source[..., 0],
            np.einsum("...i,...i->...", gradient_reference, reference_east),
            np.einsum("...i,...i->...", gradient_reference, reference_north),
        ),
        axis=-1,
    ).astype(np.float32)


def _top_source_mask(
    parallaxes: np.ndarray,
    validities: np.ndarray,
    maximum: int | None,
) -> np.ndarray:
    score = np.where(validities, parallaxes, -np.inf)
    if maximum is None or maximum >= score.shape[0]:
        return validities.copy()
    order = np.argsort(-score, axis=0, kind="stable")
    rank = np.empty_like(order)
    source_indices = np.arange(score.shape[0], dtype=order.dtype)[:, None, None]
    np.put_along_axis(rank, order, source_indices, axis=0)
    return validities & (rank < maximum)


def _combine_source_costs(
    costs: np.ndarray,
    weights: np.ndarray,
    selected: np.ndarray,
    policy: str,
) -> np.ndarray:
    count = selected.sum(axis=0)
    if policy == "weighted-mean":
        weight_sum = weights.sum(axis=0)
        return np.divide(
            (weights * costs).sum(axis=0),
            weight_sum,
            out=np.ones(costs.shape[1:], dtype=np.float32),
            where=weight_sum > 0.0,
        )
    ordered = np.sort(np.where(selected, costs, np.inf), axis=0)
    if policy == "best":
        result = ordered[0]
    else:
        # The lower median keeps the one-source and two-source cases defined
        # without manufacturing evidence from an invalid source. With three or
        # more views it rejects one-sided occlusion and appearance outliers.
        median_index = np.maximum((count - 1) // 2, 0)
        result = np.take_along_axis(ordered, median_index[None, ...], axis=0)[0]
    return np.where(count > 0, result, 1.0).astype(np.float32)


def _ad_census_face_cost(
    reference: np.ndarray,
    warped: np.ndarray,
    warped_validity: np.ndarray,
    reference_census: tuple[np.ndarray, np.ndarray],
    options: FullSphereStereoOptions,
    *,
    tap_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the shared AD-Census unary cost on one gnomonic face."""

    reference_descriptor, reference_descriptor_valid = reference_census
    warped_descriptor, warped_descriptor_valid = _census(
        warped,
        options.census_radius,
        tap_stride=tap_stride,
    )
    size = 2 * options.census_radius + 1
    footprint_size = 2 * options.census_radius * tap_stride + 1
    footprint = np.zeros((footprint_size, footprint_size), dtype=np.uint8)
    footprint[::tap_stride, ::tap_stride] = 1
    neighbourhood_valid = cv2.erode(
        warped_validity.astype(np.uint8),
        footprint,
        iterations=1,
    ).astype(bool)
    valid = (
        warped_validity
        & neighbourhood_valid
        & reference_descriptor_valid
        & warped_descriptor_valid
    )
    bits = size * size - 1
    census = _hamming(reference_descriptor, warped_descriptor) / float(bits)
    ad = (
        np.minimum(np.abs(reference - warped), options.ad_truncation)
        / options.ad_truncation
    )
    cost = options.ad_weight * ad + options.census_weight * census
    return cost.astype(np.float32), valid


def _local_cost_aggregation(
    volume_hwd: np.ndarray,
    validity_hwd: np.ndarray,
    window_size: int,
    method: str,
    *,
    tap_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply exactly one valid-normalized local aggregation stage."""

    if (
        isinstance(tap_stride, bool)
        or not isinstance(tap_stride, (int, np.integer))
        or tap_stride < 1
    ):
        raise ValueError("tap_stride must be a positive integer")

    if method == "none" or window_size == 1:
        return (
            np.where(validity_hwd, volume_hwd, 1.0).astype(np.float32),
            validity_hwd.copy(),
        )
    weighted = np.where(validity_hwd, volume_hwd, 0.0).astype(np.float32)
    if tap_stride == 1:
        kernel = (window_size, window_size)
        sums = cv2.boxFilter(
            weighted,
            -1,
            kernel,
            normalize=False,
            borderType=cv2.BORDER_REFLECT101,
        )
        counts = cv2.boxFilter(
            validity_hwd.astype(np.float32),
            -1,
            kernel,
            normalize=False,
            borderType=cv2.BORDER_REFLECT101,
        )
    else:
        footprint_size = (window_size - 1) * tap_stride + 1
        footprint = np.zeros((footprint_size, footprint_size), dtype=np.float32)
        footprint[::tap_stride, ::tap_stride] = 1.0
        sums = cv2.filter2D(
            weighted,
            -1,
            footprint,
            borderType=cv2.BORDER_REFLECT101,
        )
        counts = cv2.filter2D(
            validity_hwd.astype(np.float32),
            -1,
            footprint,
            borderType=cv2.BORDER_REFLECT101,
        )
    if sums.ndim == 2:
        sums = sums[..., None]
        counts = counts[..., None]
    aggregated_valid = counts > 0.5 * window_size**2
    aggregated = np.where(
        aggregated_valid,
        sums / np.maximum(counts, 1.0),
        1.0,
    ).astype(np.float32)
    return aggregated, aggregated_valid


def _face_center_pixel_step_deg(spec: GnomonicSpec) -> float:
    """Return mean horizontal/vertical centre-ray spacing in degrees."""

    height, width = spec.output_shape_hw
    centre = np.asarray(((width - 1.0) / 2.0, (height - 1.0) / 2.0))
    pixels = np.stack(
        (
            centre,
            centre + np.asarray((1.0, 0.0)),
            centre + np.asarray((0.0, 1.0)),
        )
    )
    rays = np.asarray(gnomonic_pixels_to_rays(pixels, spec).rays_xyz)
    horizontal = math.degrees(
        math.acos(float(np.clip(np.dot(rays[0], rays[1]), -1.0, 1.0)))
    )
    vertical = math.degrees(
        math.acos(float(np.clip(np.dot(rays[0], rays[2]), -1.0, 1.0)))
    )
    return 0.5 * (horizontal + vertical)


def _face_tap_stride(
    spec: GnomonicSpec,
    angular_step_deg: float | None,
) -> int:
    """Map a requested angular tap step to a deterministic face-pixel stride."""

    if angular_step_deg is None:
        return 1
    pixel_step = _face_center_pixel_step_deg(spec)
    return max(1, int(math.floor(float(angular_step_deg) / pixel_step + 0.5)))


def _regularize_cost_volume(
    volume_hwd: np.ndarray,
    validity_hwd: np.ndarray,
    guidance: np.ndarray,
    settings: SphericalStereoOptions,
    method: str,
) -> np.ndarray:
    """Regularize one HWD inverse-range volume and return DHW storage."""

    finite = np.where(validity_hwd, volume_hwd, 1.0).astype(np.float32)
    if method == "none":
        return np.moveaxis(finite, -1, 0)
    if method == "sgm4":
        return _aggregate_cost_volume(np.moveaxis(finite, -1, 0), guidance, settings)
    return np.moveaxis(
        _mgm_one(
            finite,
            guidance,
            settings.smoothness_penalty_small,
            settings.smoothness_penalty_large,
            0.08,
            "tmgm",
        ),
        -1,
        0,
    )


def _edge_taper(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    return np.clip((1.0 - xx * xx) * (1.0 - yy * yy), 0.0, 1.0)


def _panorama_payload(
    value: Any,
    explicit_validity: np.ndarray | None,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(value, EquirectangularImage):
        image = np.asarray(value.image)
        validity = np.asarray(value.validity("image"), dtype=bool).copy()
        validity &= np.asarray(value.support("image"), dtype=bool)
    else:
        image = np.asarray(value)
        if image.ndim not in {2, 3}:
            raise ValueError(f"{name} must use HW or HWC layout")
        validity = np.ones(image.shape[:2], dtype=bool)
    if image.shape[1] != 2 * image.shape[0]:
        raise ValueError(f"{name} must be a canonical 2:1 ERP")
    if explicit_validity is not None:
        provided = np.asarray(explicit_validity)
        if provided.dtype != np.bool_ or provided.shape != image.shape[:2]:
            raise ValueError(f"{name} validity must be a boolean HW mask")
        validity &= provided
    finite = (
        np.isfinite(image) if image.ndim == 2 else np.all(np.isfinite(image), axis=-1)
    )
    if not np.all(finite[validity]):
        raise ValueError(f"valid {name} samples must be finite")
    return image, validity


__all__ = [
    "FullSphereMultiViewStereo",
    "FullSphereStereoOptions",
    "FullSphereStereoResult",
    "OverlappingCubemapPlanner",
    "RegisteredPanoramaSet",
    "StereoSourceView",
]
