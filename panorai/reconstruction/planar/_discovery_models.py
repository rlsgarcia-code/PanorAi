"""Typed contracts for spherical line and planar-orientation discovery.

The Experimental application reports image evidence and orientation hypotheses.
It deliberately cannot publish a metric plane offset from a single panorama.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from panorai.data import EquirectangularImage

from ._models import PlaneEstimate, PlanarRegion, readonly


PLANAR_DISCOVERY_INTERFACE = "panorai-spherical-planar-discovery/v1-experimental"


def _unit(value: Any, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    length = float(np.linalg.norm(vector))
    if length < 1e-12:
        raise ValueError(f"{name} must be non-degenerate")
    return readonly(vector / length)


def _string_tuple(value: Any, name: str, *, minimum: int = 0) -> tuple[str, ...]:
    result = tuple(str(item) for item in value)
    if len(result) < minimum or any(not item for item in result):
        raise ValueError(f"{name} requires at least {minimum} non-empty IDs")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} cannot contain duplicate IDs")
    return result


@dataclass(frozen=True)
class SphericalPlanarDiscoveryOptions:
    """Bounded CPU policy for deterministic full-sphere discovery."""

    atlas_face_size: int = 384
    atlas_hfov_deg: float = 100.0
    canny_low: float = 50.0
    canny_high: float = 140.0
    local_contrast_clip_limit: float = 0.0
    minimum_segment_px: float = 24.0
    minimum_edge_support: float = 0.45
    minimum_line_length_deg: float = 2.0
    merge_normal_deg: float = 1.25
    merge_midpoint_deg: float = 4.0
    vanishing_residual_deg: float = 2.5
    suggestion_residual_deg: float = 5.0
    minimum_bundle_lines: int = 3
    minimum_bundle_support_deg: float = 8.0
    maximum_lines: int = 480
    maximum_intersections: int = 12000
    maximum_bundles: int = 12
    maximum_suggestions_per_bundle: int = 24
    maximum_proposals: int = 24

    def __post_init__(self) -> None:
        for name in (
            "atlas_face_size",
            "minimum_bundle_lines",
            "maximum_lines",
            "maximum_intersections",
            "maximum_bundles",
            "maximum_suggestions_per_bundle",
            "maximum_proposals",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 128 <= self.atlas_face_size <= 1024:
            raise ValueError("atlas_face_size must lie in [128, 1024]")
        if not 60.0 <= self.atlas_hfov_deg < 150.0:
            raise ValueError("atlas_hfov_deg must lie in [60, 150)")
        for name in (
            "canny_low",
            "canny_high",
            "minimum_segment_px",
            "minimum_line_length_deg",
            "merge_normal_deg",
            "merge_midpoint_deg",
            "vanishing_residual_deg",
            "suggestion_residual_deg",
            "minimum_bundle_support_deg",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.canny_low >= self.canny_high:
            raise ValueError("canny_low must be below canny_high")
        if (
            not np.isfinite(self.local_contrast_clip_limit)
            or not 0.0 <= self.local_contrast_clip_limit <= 8.0
        ):
            raise ValueError("local_contrast_clip_limit must lie in [0, 8]")
        if not 0 < self.minimum_edge_support <= 1:
            raise ValueError("minimum_edge_support must lie in (0, 1]")
        if self.minimum_bundle_lines < 2:
            raise ValueError("minimum_bundle_lines must be at least two")
        if self.suggestion_residual_deg < self.vanishing_residual_deg:
            raise ValueError(
                "suggestion_residual_deg must not be stricter than vanishing_residual_deg"
            )


@dataclass(frozen=True)
class SphericalLineSeed:
    """One reviewed manual great-circle seed on the native ERP grid."""

    line_id: str
    erp_endpoints_xy: np.ndarray
    state: str = "accepted"

    def __post_init__(self) -> None:
        endpoints = np.asarray(self.erp_endpoints_xy, dtype=np.float64)
        if (
            not self.line_id
            or endpoints.shape != (2, 2)
            or not np.isfinite(endpoints).all()
        ):
            raise ValueError("line seed requires an ID and finite (2, 2) ERP endpoints")
        if self.state not in {"accepted", "rejected"}:
            raise ValueError("manual line seed state must be accepted or rejected")
        object.__setattr__(self, "erp_endpoints_xy", readonly(endpoints))


@dataclass(frozen=True)
class PlanarDiscoveryRequest:
    """One immutable discovery/review revision for an ERP."""

    panorama: EquirectangularImage
    line_decisions: Mapping[str, str] = field(default_factory=dict)
    bundle_memberships: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    bundle_line_decisions: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    plane_bundle_groups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    manual_line_seeds: tuple[SphericalLineSeed, ...] = ()
    regions: tuple[PlanarRegion, ...] = ()
    plane_line_groups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    revision: str = "unversioned"
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.panorama, EquirectangularImage):
            raise TypeError("panorama must be an EquirectangularImage")
        image = np.asarray(self.panorama.image)
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError("panorama image must be native HWC RGB uint8")
        support = (
            np.ones(image.shape[:2], dtype=bool)
            if self.panorama.support_mask is None
            else np.asarray(self.panorama.support_mask)
        )
        validity = np.asarray(self.panorama.validity("image"))
        if support.dtype != np.bool_ or validity.dtype != np.bool_:
            raise TypeError("support and image validity must be boolean")
        if support.shape != image.shape[:2] or validity.shape != image.shape[:2]:
            raise ValueError("support and validity must use the native ERP grid")
        object.__setattr__(
            self,
            "panorama",
            EquirectangularImage(
                readonly(image),
                support_mask=readonly(support),
                valid=readonly(validity),
            ),
        )
        decisions = {str(key): str(value) for key, value in self.line_decisions.items()}
        if any(value not in {"accepted", "rejected"} for value in decisions.values()):
            raise ValueError("line decisions must be accepted or rejected")
        memberships = {
            str(key): _string_tuple(value, "bundle membership", minimum=2)
            for key, value in self.bundle_memberships.items()
        }
        bundle_line_decisions = {
            str(bundle_id): {
                str(line_id): str(decision) for line_id, decision in decisions.items()
            }
            for bundle_id, decisions in self.bundle_line_decisions.items()
        }
        if any(
            decision not in {"rejected"}
            for decisions in bundle_line_decisions.values()
            for decision in decisions.values()
        ):
            raise ValueError(
                "bundle line decisions currently support only explicit rejection"
            )
        groups = {
            str(key): _string_tuple(value, "plane bundle group", minimum=2)
            for key, value in self.plane_bundle_groups.items()
        }
        line_groups = {
            str(key): _string_tuple(value, "plane line group", minimum=1)
            for key, value in self.plane_line_groups.items()
        }
        seeds = tuple(self.manual_line_seeds)
        if len({seed.line_id for seed in seeds}) != len(seeds):
            raise ValueError("manual line seed IDs must be unique")
        regions = tuple(self.regions)
        if len({region.plane_id for region in regions}) != len(regions):
            raise ValueError("discovery regions must use unique plane IDs")
        if any(not key for key in (*memberships, *groups, *line_groups)):
            raise ValueError("bundle and plane IDs must be non-empty")
        if any(
            not bundle_id or any(not line_id for line_id in decisions)
            for bundle_id, decisions in bundle_line_decisions.items()
        ):
            raise ValueError("bundle and line decision IDs must be non-empty")
        if not self.revision:
            raise ValueError("revision is required")
        object.__setattr__(self, "line_decisions", MappingProxyType(decisions))
        object.__setattr__(self, "bundle_memberships", MappingProxyType(memberships))
        object.__setattr__(
            self,
            "bundle_line_decisions",
            MappingProxyType(
                {
                    bundle_id: MappingProxyType(decisions)
                    for bundle_id, decisions in bundle_line_decisions.items()
                }
            ),
        )
        object.__setattr__(self, "plane_bundle_groups", MappingProxyType(groups))
        object.__setattr__(self, "manual_line_seeds", seeds)
        object.__setattr__(self, "regions", regions)
        object.__setattr__(self, "plane_line_groups", MappingProxyType(line_groups))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


@dataclass(frozen=True)
class SphericalLineObservation:
    """One fitted finite great-circle arc reported on the original ERP grid."""

    line_id: str
    erp_endpoints_xy: np.ndarray
    great_circle_normal_camera: np.ndarray
    angular_length_deg: float
    support_pixels: int
    residual_px: float
    edge_support: float
    score: float
    chart_ids: tuple[str, ...]
    state: str = "detected"
    source: str = "automatic"
    region_support: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        endpoints = np.asarray(self.erp_endpoints_xy, dtype=np.float64)
        if endpoints.shape != (2, 2) or not np.isfinite(endpoints).all():
            raise ValueError("erp_endpoints_xy must be a finite (2, 2) array")
        if not self.line_id:
            raise ValueError("line_id is required")
        if self.state not in {"detected", "accepted", "rejected"}:
            raise ValueError("invalid line state")
        if self.source not in {"automatic", "human"}:
            raise ValueError("line source must be automatic or human")
        for name in ("angular_length_deg", "residual_px", "edge_support", "score"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 <= self.edge_support <= 1 or self.support_pixels < 0:
            raise ValueError("invalid edge support")
        object.__setattr__(self, "erp_endpoints_xy", readonly(endpoints))
        object.__setattr__(
            self,
            "great_circle_normal_camera",
            _unit(self.great_circle_normal_camera, "great_circle_normal_camera"),
        )
        object.__setattr__(
            self, "chart_ids", _string_tuple(self.chart_ids, "chart_ids", minimum=1)
        )
        support = {str(key): float(value) for key, value in self.region_support.items()}
        if any(
            not np.isfinite(value) or not 0 <= value <= 1 for value in support.values()
        ):
            raise ValueError("region support values must lie in [0, 1]")
        object.__setattr__(self, "region_support", MappingProxyType(support))


@dataclass(frozen=True)
class VanishingLineSuggestion:
    """One review-only line compatible with an estimated vanishing direction."""

    line_id: str
    residual_deg: float
    confidence: float
    state: str = "suggested"
    assigned_bundle_id: str | None = None

    def __post_init__(self) -> None:
        if not self.line_id:
            raise ValueError("suggested line ID is required")
        if self.state not in {"suggested", "rejected"}:
            raise ValueError("suggested line state must be suggested or rejected")
        if not np.isfinite(self.residual_deg) or self.residual_deg < 0:
            raise ValueError("suggested line residual must be finite and nonnegative")
        if not np.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("suggested line confidence must lie in [0, 1]")
        if self.assigned_bundle_id is not None and not self.assigned_bundle_id:
            raise ValueError("assigned bundle ID must be non-empty when supplied")


@dataclass(frozen=True)
class VanishingBundle:
    """A reviewed or automatic family of image lines sharing one 3D direction."""

    bundle_id: str
    direction_camera: np.ndarray
    line_ids: tuple[str, ...]
    residual_deg: float
    support_deg: float
    confidence: float
    state: str = "draft"
    source: str = "automatic"
    suggestions: tuple[VanishingLineSuggestion, ...] = ()

    def __post_init__(self) -> None:
        if not self.bundle_id:
            raise ValueError("bundle_id is required")
        if self.state not in {"draft", "accepted", "rejected", "needs-review"}:
            raise ValueError("invalid bundle state")
        if self.source not in {"automatic", "human"}:
            raise ValueError("bundle source must be automatic or human")
        for name in ("residual_deg", "support_deg", "confidence"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.confidence > 1:
            raise ValueError("confidence must not exceed one")
        object.__setattr__(
            self, "direction_camera", _unit(self.direction_camera, "direction_camera")
        )
        object.__setattr__(
            self, "line_ids", _string_tuple(self.line_ids, "line_ids", minimum=2)
        )
        suggestions = tuple(self.suggestions)
        if len({item.line_id for item in suggestions}) != len(suggestions):
            raise ValueError("suggested line IDs must be unique within a bundle")
        if set(self.line_ids) & {item.line_id for item in suggestions}:
            raise ValueError("bundle members cannot also be suggestions")
        object.__setattr__(self, "suggestions", suggestions)


@dataclass(frozen=True)
class PlanarOrientationProposal:
    """Orientation-only plane proposal; metric offset is always unavailable."""

    proposal_id: str
    estimate: PlaneEstimate
    normal_camera: np.ndarray
    bundle_ids: tuple[str, ...]
    line_ids: tuple[str, ...]
    roi_erp_xy: np.ndarray
    confidence: float
    state: str = "draft"
    reason: str | None = None
    source: str = "automatic"
    plane_id: str | None = None

    def __post_init__(self) -> None:
        if not self.proposal_id or self.estimate.offset_m is not None:
            raise ValueError("orientation proposals require an ID and offset_m=None")
        if self.state not in {"draft", "accepted", "rejected", "needs-review"}:
            raise ValueError("invalid proposal state")
        if self.source not in {"automatic", "human"}:
            raise ValueError("proposal source must be automatic or human")
        if not np.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must lie in [0, 1]")
        roi = np.asarray(self.roi_erp_xy, dtype=np.float64)
        if roi.ndim != 2 or roi.shape[1:] != (2,) or not np.isfinite(roi).all():
            raise ValueError("roi_erp_xy must be a finite Nx2 point array")
        object.__setattr__(
            self, "bundle_ids", _string_tuple(self.bundle_ids, "bundle_ids", minimum=2)
        )
        object.__setattr__(
            self, "line_ids", _string_tuple(self.line_ids, "line_ids", minimum=2)
        )
        object.__setattr__(
            self, "normal_camera", _unit(self.normal_camera, "normal_camera")
        )
        object.__setattr__(self, "roi_erp_xy", readonly(roi))
        if self.plane_id is not None and not self.plane_id:
            raise ValueError("plane_id must be non-empty when supplied")


@dataclass(frozen=True)
class SphericalPlanarDiscoveryResult:
    """Exact line evidence, vanishing bundles and orientation-only proposals."""

    request: PlanarDiscoveryRequest
    lines: tuple[SphericalLineObservation, ...]
    bundles: tuple[VanishingBundle, ...]
    proposals: tuple[PlanarOrientationProposal, ...]
    edge_preview_erp_xy: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float64)
    )
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    state: str = "needs-review"

    @property
    def interface(self) -> str:
        return PLANAR_DISCOVERY_INTERFACE

    def __post_init__(self) -> None:
        if self.state not in {"empty", "needs-review", "reviewed"}:
            raise ValueError("invalid discovery state")
        for name in ("lines", "bundles", "proposals"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        edges = np.asarray(self.edge_preview_erp_xy, dtype=np.float64)
        if edges.ndim != 2 or edges.shape[1:] != (2,) or not np.isfinite(edges).all():
            raise ValueError("edge_preview_erp_xy must be a finite Nx2 point array")
        object.__setattr__(self, "edge_preview_erp_xy", readonly(edges))
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics))
        )
        if len({item.line_id for item in self.lines}) != len(self.lines):
            raise ValueError("line IDs must be unique")
        if len({item.bundle_id for item in self.bundles}) != len(self.bundles):
            raise ValueError("bundle IDs must be unique")
        if len({item.proposal_id for item in self.proposals}) != len(self.proposals):
            raise ValueError("proposal IDs must be unique")
