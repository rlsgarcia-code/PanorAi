"""Multiview association and consensus refitting for spherical FOV anchors.

An anchor is a gnomonic field-of-view proposal attached to one registered
panorama.  Anchors are not merged by taking unions of ERP rectangles: each
view keeps its own angular extent.  Cross-view association uses auditable
pair cues, deterministic complete-link clustering, and optional registered
depth transfer.  Transient raster masks are used only while refitting a
per-view consensus region and are never retained by the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
from itertools import combinations
import math
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np

from panorai.geometry import (
    GnomonicSpec,
    erp_pixels_to_rays,
    gnomonic_rotation,
    rays_to_gnomonic_pixels,
)
from panorai.reconstruction import SphericalCameraPose

from ._multiview import (
    HypothesisGnomonicRegion,
    RegisteredBearingObservation,
    RegisteredBearingTrack,
    SphericalObjectHypothesisConfig,
    triangulate_registered_bearing_track,
)
from ._region_consistency import (
    RegisteredRadialDepth,
    SymmetricRegionTransfer,
    spherical_region_mask,
    spherical_solid_angle_weights,
    symmetric_spherical_region_transfer,
)


SPHERICAL_ANCHOR_FUSION_INTERFACE = "panorai-spherical-anchor-fusion/v1"


def _unit_interval(value: float, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must lie within [0,1]")
    return numeric


def _sorted_unique(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    normalized = tuple(str(value).strip() for value in values)
    if any(not value for value in normalized):
        raise ValueError(f"{name} must contain non-empty values")
    if tuple(sorted(set(normalized))) != normalized:
        raise ValueError(f"{name} must be sorted and unique")
    return normalized


def _stable_id(prefix: str, *parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(repr(part).encode("utf-8"))
        digest.update(b"\0")
    return f"{prefix}-{digest.hexdigest()[:20]}"


@dataclass(frozen=True, slots=True)
class SphericalFOVAnchor:
    """One local object-region proposal expressed as a gnomonic FOV."""

    anchor_id: str
    view_id: str
    concept_id: str
    vocabulary: str
    region: HypothesisGnomonicRegion
    score: float
    track_ids: tuple[str, ...] = ()
    source_hypothesis_ids: tuple[str, ...] = ()
    provenance: Mapping[str, object] = field(default_factory=dict)
    proposal_family_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("anchor_id", "view_id", "concept_id", "vocabulary"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, value)
        if self.region.view_id != self.view_id:
            raise ValueError("anchor and region view IDs must match")
        object.__setattr__(self, "score", _unit_interval(self.score, "score"))
        tracks = self.track_ids or self.region.track_ids
        object.__setattr__(
            self, "track_ids", _sorted_unique(tuple(tracks), "track_ids")
        )
        object.__setattr__(
            self,
            "source_hypothesis_ids",
            _sorted_unique(tuple(self.source_hypothesis_ids), "source_hypothesis_ids"),
        )
        families = self.proposal_family_ids or (self.anchor_id,)
        object.__setattr__(
            self,
            "proposal_family_ids",
            _sorted_unique(tuple(families), "proposal_family_ids"),
        )
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


@dataclass(frozen=True, slots=True)
class SphericalAnchorPairCues:
    """Optional, validity-explicit cues for one ordered anchor pair.

    ``None`` means that a cue was unavailable.  A numeric zero means that the
    cue was evaluated and found no agreement.  Those states are intentionally
    distinct.
    """

    anchor_id_a: str
    anchor_id_b: str
    geometric_compatibility: float | None = None
    transfer_iou: float | None = None
    track_consistency: float | None = None
    appearance_cosine_similarity: float | None = None
    geometric_contradiction: bool = False
    depth_separated: bool = False
    track_contradiction: bool = False
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not self.anchor_id_a.strip()
            or not self.anchor_id_b.strip()
            or self.anchor_id_a >= self.anchor_id_b
        ):
            raise ValueError("anchor IDs must be non-empty and strictly ordered")
        for name in (
            "geometric_compatibility",
            "transfer_iou",
            "track_consistency",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _unit_interval(value, name))
        if self.appearance_cosine_similarity is not None:
            similarity = float(self.appearance_cosine_similarity)
            if not math.isfinite(similarity) or not -1.0 <= similarity <= 1.0:
                raise ValueError("appearance_cosine_similarity must lie within [-1,1]")
            object.__setattr__(self, "appearance_cosine_similarity", similarity)
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


@dataclass(frozen=True, slots=True)
class SphericalAnchorAffinity:
    anchor_id_a: str
    anchor_id_b: str
    hard_compatible: bool
    hard_reasons: tuple[str, ...]
    factors: Mapping[str, float]
    available_association_cues: tuple[str, ...]
    affinity: float

    def __post_init__(self) -> None:
        if self.anchor_id_a >= self.anchor_id_b:
            raise ValueError("anchor IDs must be strictly ordered")
        factors = {
            str(name): _unit_interval(value, str(name))
            for name, value in self.factors.items()
        }
        cues = tuple(sorted(set(self.available_association_cues)))
        if not self.hard_compatible and self.affinity != 0.0:
            raise ValueError("hard-incompatible pairs must have zero affinity")
        object.__setattr__(self, "factors", MappingProxyType(factors))
        object.__setattr__(self, "available_association_cues", cues)
        object.__setattr__(self, "affinity", _unit_interval(self.affinity, "affinity"))


@dataclass(frozen=True, slots=True)
class SphericalAnchorFusionConfig:
    """Frozen policy for anchor suppression, association, and FOV refitting."""

    intra_view_nms_iou: float = 0.85
    maximum_anchors_per_view_concept: int = 6
    pair_affinity_threshold: float = 0.20
    complete_link_threshold: float = 0.20
    cluster_nms_iou: float = 0.55
    minimum_views_for_confirmation: int = 2
    minimum_depth_coverage: float = 0.25
    center_reprojection_sigma_deg: float = 1.0
    maximum_center_reprojection_error_deg: float = 2.0
    consensus_threshold: float = 0.50
    consensus_coverage: float = 0.90
    consensus_margin_deg: float = 3.0
    minimum_final_fov_deg: float = 8.0
    maximum_final_fov_deg: float = 120.0
    evaluation_shape_hw: tuple[int, int] = (128, 256)
    transfer_weight: float = 1.0
    track_weight: float = 1.0
    appearance_weight: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "intra_view_nms_iou",
            "pair_affinity_threshold",
            "complete_link_threshold",
            "cluster_nms_iou",
            "minimum_depth_coverage",
            "consensus_threshold",
            "consensus_coverage",
        ):
            _unit_interval(getattr(self, name), name)
        if self.maximum_anchors_per_view_concept < 1:
            raise ValueError("maximum_anchors_per_view_concept must be positive")
        if self.minimum_views_for_confirmation < 2:
            raise ValueError("minimum_views_for_confirmation must be at least two")
        for name in (
            "center_reprojection_sigma_deg",
            "maximum_center_reprojection_error_deg",
            "consensus_margin_deg",
            "minimum_final_fov_deg",
            "maximum_final_fov_deg",
            "transfer_weight",
            "track_weight",
            "appearance_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.minimum_final_fov_deg > self.maximum_final_fov_deg:
            raise ValueError("minimum final FOV cannot exceed maximum final FOV")
        if self.maximum_final_fov_deg >= 180.0:
            raise ValueError("maximum final FOV must be less than 180 degrees")
        shape = tuple(self.evaluation_shape_hw)
        if len(shape) != 2 or any(int(value) < 1 for value in shape):
            raise ValueError("evaluation_shape_hw must contain two positive values")
        object.__setattr__(self, "evaluation_shape_hw", (int(shape[0]), int(shape[1])))


@dataclass(frozen=True, slots=True)
class SphericalAnchorCluster:
    cluster_id: str
    concept_id: str
    vocabulary: str
    anchor_ids: tuple[str, ...]
    view_ids: tuple[str, ...]
    source_hypothesis_ids: tuple[str, ...]
    proposal_family_ids: tuple[str, ...]
    score: float
    state: str
    consensus_regions: tuple[HypothesisGnomonicRegion, ...]
    minimum_pair_affinity: float | None

    def __post_init__(self) -> None:
        if self.state not in {"proposed", "confirmed"}:
            raise ValueError("anchor cluster state must be proposed or confirmed")
        object.__setattr__(
            self, "anchor_ids", _sorted_unique(self.anchor_ids, "anchor_ids")
        )
        object.__setattr__(self, "view_ids", _sorted_unique(self.view_ids, "view_ids"))
        object.__setattr__(
            self,
            "source_hypothesis_ids",
            _sorted_unique(self.source_hypothesis_ids, "source_hypothesis_ids"),
        )
        object.__setattr__(
            self,
            "proposal_family_ids",
            _sorted_unique(self.proposal_family_ids, "proposal_family_ids"),
        )
        if len(self.anchor_ids) != len(self.view_ids):
            raise ValueError("a cluster must contain exactly one anchor per view")
        if tuple(region.view_id for region in self.consensus_regions) != self.view_ids:
            raise ValueError("consensus regions must align with sorted cluster views")
        object.__setattr__(self, "score", _unit_interval(self.score, "score"))
        if self.minimum_pair_affinity is not None:
            object.__setattr__(
                self,
                "minimum_pair_affinity",
                _unit_interval(self.minimum_pair_affinity, "minimum_pair_affinity"),
            )


AnchorDecisionAction = Literal["retained", "suppressed-overlap", "suppressed-capacity"]


@dataclass(frozen=True, slots=True)
class SphericalAnchorDecision:
    """One deterministic intra-view NMS decision without retaining a mask."""

    anchor: SphericalFOVAnchor
    action: AnchorDecisionAction
    representative_anchor_id: str | None
    solid_angle_iou: float | None

    def __post_init__(self) -> None:
        if self.action not in {
            "retained",
            "suppressed-overlap",
            "suppressed-capacity",
        }:
            raise ValueError("unsupported anchor decision action")
        if self.action == "retained":
            if self.representative_anchor_id != self.anchor.anchor_id:
                raise ValueError("retained anchors must represent themselves")
        elif self.action == "suppressed-overlap":
            if not self.representative_anchor_id:
                raise ValueError("overlap suppression requires a representative")
            if self.representative_anchor_id == self.anchor.anchor_id:
                raise ValueError("a suppressed anchor cannot represent itself")
        elif self.representative_anchor_id is not None:
            raise ValueError("capacity suppression has no representative")
        if self.solid_angle_iou is not None:
            object.__setattr__(
                self,
                "solid_angle_iou",
                _unit_interval(self.solid_angle_iou, "solid_angle_iou"),
            )
        if self.action == "suppressed-overlap" and self.solid_angle_iou is None:
            raise ValueError("overlap suppression requires solid-angle IoU")


ClusterDecisionAction = Literal["retained", "suppressed-family-overlap"]


@dataclass(frozen=True, slots=True)
class SphericalAnchorClusterDecision:
    """One deterministic cluster-level NMS decision without retaining masks."""

    cluster: SphericalAnchorCluster
    action: ClusterDecisionAction
    representative_cluster_id: str
    minimum_view_iou: float

    def __post_init__(self) -> None:
        if self.action not in {
            "retained",
            "suppressed-family-overlap",
        }:
            raise ValueError("unsupported cluster decision action")
        if not self.representative_cluster_id.strip():
            raise ValueError("representative_cluster_id must be non-empty")
        if self.action == "retained":
            if self.representative_cluster_id != self.cluster.cluster_id:
                raise ValueError("retained clusters must represent themselves")
        elif self.representative_cluster_id == self.cluster.cluster_id:
            raise ValueError("a suppressed cluster cannot represent itself")
        object.__setattr__(
            self,
            "minimum_view_iou",
            _unit_interval(self.minimum_view_iou, "minimum_view_iou"),
        )


@dataclass(frozen=True, slots=True)
class SphericalAnchorFusionResult:
    retained_anchors: tuple[SphericalFOVAnchor, ...]
    suppressed_anchor_ids: tuple[str, ...]
    anchor_decisions: tuple[SphericalAnchorDecision, ...]
    pair_affinities: tuple[SphericalAnchorAffinity, ...]
    clusters: tuple[SphericalAnchorCluster, ...]
    suppressed_cluster_ids: tuple[str, ...]
    cluster_decisions: tuple[SphericalAnchorClusterDecision, ...]
    diagnostics: Mapping[str, int | float | str]
    interface: str = SPHERICAL_ANCHOR_FUSION_INTERFACE
    stability: str = "experimental"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "suppressed_anchor_ids",
            _sorted_unique(self.suppressed_anchor_ids, "suppressed_anchor_ids"),
        )
        object.__setattr__(
            self,
            "suppressed_cluster_ids",
            _sorted_unique(self.suppressed_cluster_ids, "suppressed_cluster_ids"),
        )
        anchor_decisions = tuple(
            sorted(self.anchor_decisions, key=lambda item: item.anchor.anchor_id)
        )
        anchor_ids = tuple(item.anchor.anchor_id for item in anchor_decisions)
        if len(set(anchor_ids)) != len(anchor_ids):
            raise ValueError("anchor decisions must contain unique anchor IDs")
        retained_anchor_ids = tuple(
            sorted(item.anchor_id for item in self.retained_anchors)
        )
        decided_retained = tuple(
            item.anchor.anchor_id
            for item in anchor_decisions
            if item.action == "retained"
        )
        decided_suppressed = tuple(
            item.anchor.anchor_id
            for item in anchor_decisions
            if item.action != "retained"
        )
        if decided_retained != retained_anchor_ids:
            raise ValueError("retained anchors and anchor decisions disagree")
        if decided_suppressed != self.suppressed_anchor_ids:
            raise ValueError("suppressed anchors and anchor decisions disagree")
        object.__setattr__(self, "anchor_decisions", anchor_decisions)

        cluster_decisions = tuple(
            sorted(self.cluster_decisions, key=lambda item: item.cluster.cluster_id)
        )
        cluster_ids = tuple(item.cluster.cluster_id for item in cluster_decisions)
        if len(set(cluster_ids)) != len(cluster_ids):
            raise ValueError("cluster decisions must contain unique cluster IDs")
        retained_cluster_ids = tuple(sorted(item.cluster_id for item in self.clusters))
        decided_cluster_retained = tuple(
            item.cluster.cluster_id
            for item in cluster_decisions
            if item.action == "retained"
        )
        decided_cluster_suppressed = tuple(
            item.cluster.cluster_id
            for item in cluster_decisions
            if item.action != "retained"
        )
        if decided_cluster_retained != retained_cluster_ids:
            raise ValueError("retained clusters and cluster decisions disagree")
        if decided_cluster_suppressed != self.suppressed_cluster_ids:
            raise ValueError("suppressed clusters and cluster decisions disagree")
        object.__setattr__(self, "cluster_decisions", cluster_decisions)
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics))
        )


def generate_spherical_fov_anchors(
    base_region: HypothesisGnomonicRegion,
    *,
    concept_id: str,
    vocabulary: str,
    score: float,
    anchor_prefix: str,
    scale_factors: tuple[float, ...] = (0.75, 1.0, 1.33),
    aspect_ratios: tuple[float, ...] = (0.5, 1.0, 2.0),
    minimum_fov_deg: float = 8.0,
    maximum_fov_deg: float = 120.0,
    source_hypothesis_ids: tuple[str, ...] = (),
    proposal_family_id: str | None = None,
) -> tuple[SphericalFOVAnchor, ...]:
    """Generate deterministic scale/aspect alternatives around one direction."""

    if not anchor_prefix.strip():
        raise ValueError("anchor_prefix must be non-empty")
    for name, values in (
        ("scale_factors", scale_factors),
        ("aspect_ratios", aspect_ratios),
    ):
        if not values or any(
            not math.isfinite(value) or value <= 0.0 for value in values
        ):
            raise ValueError(f"{name} must contain finite positive values")
    if minimum_fov_deg <= 0.0 or maximum_fov_deg >= 180.0:
        raise ValueError("FOV limits must lie inside (0,180)")
    if minimum_fov_deg > maximum_fov_deg:
        raise ValueError("minimum_fov_deg cannot exceed maximum_fov_deg")
    if proposal_family_id is None:
        family_id = _stable_id(
            "proposal-family",
            anchor_prefix,
            base_region.view_id,
            concept_id,
            round(base_region.spec.center_lat_deg, 9),
            round(base_region.spec.center_lon_deg, 9),
        )
    else:
        family_id = str(proposal_family_id).strip()
        if not family_id:
            raise ValueError("proposal_family_id must be non-empty when provided")

    anchors: dict[str, SphericalFOVAnchor] = {}
    variants_by_anchor: dict[str, list[tuple[float, float]]] = {}
    for scale in sorted(set(float(value) for value in scale_factors)):
        for aspect in sorted(set(float(value) for value in aspect_ratios)):
            horizontal = float(base_region.spec.hfov_deg) * scale * math.sqrt(aspect)
            vertical = float(base_region.spec.vfov_deg) * scale / math.sqrt(aspect)
            horizontal = min(max(horizontal, minimum_fov_deg), maximum_fov_deg)
            vertical = min(max(vertical, minimum_fov_deg), maximum_fov_deg)
            spec = replace(
                base_region.spec,
                hfov_deg=horizontal,
                vfov_deg=vertical,
            )
            region = replace(
                base_region,
                spec=spec,
                extent_source=(
                    f"spherical-fov-anchor:scale={scale:.6g}:aspect={aspect:.6g}"
                ),
            )
            anchor_id = _stable_id(
                anchor_prefix,
                base_region.view_id,
                concept_id,
                round(spec.center_lat_deg, 9),
                round(spec.center_lon_deg, 9),
                round(horizontal, 9),
                round(vertical, 9),
            )
            candidate = SphericalFOVAnchor(
                anchor_id=anchor_id,
                view_id=base_region.view_id,
                concept_id=concept_id,
                vocabulary=vocabulary,
                region=region,
                score=float(
                    np.clip(
                        score
                        * math.exp(
                            -0.15 * (abs(math.log(scale)) + 0.5 * abs(math.log(aspect)))
                        ),
                        0.0,
                        1.0,
                    )
                ),
                source_hypothesis_ids=tuple(sorted(set(source_hypothesis_ids))),
                provenance={"scale_factor": scale, "aspect_ratio": aspect},
                proposal_family_ids=(family_id,),
            )
            variants_by_anchor.setdefault(anchor_id, []).append((scale, aspect))
            existing = anchors.get(anchor_id)
            if existing is None or candidate.score > existing.score:
                anchors[anchor_id] = candidate
    return tuple(
        replace(
            anchors[anchor_id],
            provenance={
                **dict(anchors[anchor_id].provenance),
                "collapsed_generator_variants": tuple(variants_by_anchor[anchor_id]),
            },
        )
        for anchor_id in sorted(anchors)
    )


def _mask_iou(
    first: HypothesisGnomonicRegion,
    second: HypothesisGnomonicRegion,
    shape_hw: tuple[int, int],
) -> float:
    one = spherical_region_mask(first.spec, shape_hw)
    two = spherical_region_mask(second.spec, shape_hw)
    weights = spherical_solid_angle_weights(shape_hw)
    union = float(weights[one | two].sum())
    return float(weights[one & two].sum()) / union if union > 0.0 else 1.0


def spherical_anchor_pair_affinity(
    first: SphericalFOVAnchor,
    second: SphericalFOVAnchor,
    *,
    cues: SphericalAnchorPairCues | None = None,
    config: SphericalAnchorFusionConfig | None = None,
) -> SphericalAnchorAffinity:
    """Combine alternative association cues without confusing missing and zero."""

    policy = config or SphericalAnchorFusionConfig()
    id_a, id_b = sorted((first.anchor_id, second.anchor_id))
    if cues is not None and (cues.anchor_id_a, cues.anchor_id_b) != (id_a, id_b):
        raise ValueError("pair cues do not identify the supplied anchors")
    by_id = {first.anchor_id: first, second.anchor_id: second}
    one, two = by_id[id_a], by_id[id_b]
    reasons = []
    if one.view_id == two.view_id:
        reasons.append("same-view-alternatives")
    if one.concept_id != two.concept_id:
        reasons.append("incompatible-concept")
    if one.vocabulary != two.vocabulary:
        reasons.append("incompatible-vocabulary")
    if cues is not None and cues.geometric_contradiction:
        reasons.append("geometric-contradiction")
    if cues is not None and cues.depth_separated:
        reasons.append("depth-separated-support")
    if cues is not None and cues.track_contradiction:
        reasons.append("track-contradiction")
    reasons = sorted(set(reasons))

    factors: dict[str, float] = {}
    if cues is not None and cues.geometric_compatibility is not None:
        factors["geometric"] = cues.geometric_compatibility
    association: list[tuple[str, float, float]] = []
    if cues is not None and cues.transfer_iou is not None:
        factors["transfer"] = cues.transfer_iou
        association.append(("transfer", cues.transfer_iou, policy.transfer_weight))
    if cues is not None and cues.track_consistency is not None:
        factors["track"] = cues.track_consistency
        association.append(("track", cues.track_consistency, policy.track_weight))
    if cues is not None and cues.appearance_cosine_similarity is not None:
        appearance = 0.5 * (1.0 + cues.appearance_cosine_similarity)
        factors["appearance"] = appearance
        association.append(("appearance", appearance, policy.appearance_weight))

    # Transfer, tracks, and appearance are alternative identity evidence.  A
    # reliable zero contributes no support, but does not erase a different,
    # independently strong cue.  Missing cues never enter the expression.
    remaining = 1.0
    for _, value, weight in association:
        remaining *= (1.0 - value) ** weight
    identity_support = 1.0 - remaining if association else 0.0
    geometric = factors.get("geometric", 1.0)
    affinity = geometric * identity_support
    if reasons:
        affinity = 0.0
    return SphericalAnchorAffinity(
        anchor_id_a=id_a,
        anchor_id_b=id_b,
        hard_compatible=not reasons,
        hard_reasons=tuple(reasons),
        factors=factors,
        available_association_cues=tuple(name for name, _, _ in association),
        affinity=float(np.clip(affinity, 0.0, 1.0)),
    )


def _merge_cues(
    automatic: SphericalAnchorPairCues,
    explicit: SphericalAnchorPairCues | None,
) -> SphericalAnchorPairCues:
    if explicit is None:
        return automatic
    return SphericalAnchorPairCues(
        automatic.anchor_id_a,
        automatic.anchor_id_b,
        geometric_compatibility=(
            explicit.geometric_compatibility
            if explicit.geometric_compatibility is not None
            else automatic.geometric_compatibility
        ),
        transfer_iou=(
            explicit.transfer_iou
            if explicit.transfer_iou is not None
            else automatic.transfer_iou
        ),
        track_consistency=(
            explicit.track_consistency
            if explicit.track_consistency is not None
            else automatic.track_consistency
        ),
        appearance_cosine_similarity=(
            explicit.appearance_cosine_similarity
            if explicit.appearance_cosine_similarity is not None
            else automatic.appearance_cosine_similarity
        ),
        geometric_contradiction=(
            automatic.geometric_contradiction or explicit.geometric_contradiction
        ),
        depth_separated=automatic.depth_separated or explicit.depth_separated,
        track_contradiction=(
            automatic.track_contradiction or explicit.track_contradiction
        ),
        provenance={
            "automatic": dict(automatic.provenance),
            "caller": dict(explicit.provenance),
        },
    )


def _automatic_pair_cues(
    first: SphericalFOVAnchor,
    second: SphericalFOVAnchor,
    *,
    pose_by_view: Mapping[str, SphericalCameraPose],
    depth_by_view: Mapping[str, RegisteredRadialDepth],
    config: SphericalAnchorFusionConfig,
) -> tuple[SphericalAnchorPairCues, SymmetricRegionTransfer | None]:
    id_a, id_b = sorted((first.anchor_id, second.anchor_id))
    by_id = {first.anchor_id: first, second.anchor_id: second}
    one, two = by_id[id_a], by_id[id_b]
    geometric: float | None = None
    geometric_contradiction = False
    geometry_provenance: dict[str, object] = {}
    if (
        one.view_id != two.view_id
        and one.view_id in pose_by_view
        and two.view_id in pose_by_view
    ):
        triangulation = triangulate_registered_bearing_track(
            RegisteredBearingTrack(
                f"anchor-centres:{id_a}:{id_b}",
                (
                    RegisteredBearingObservation(
                        one.view_id, 0, one.region.center_bearing_xyz
                    ),
                    RegisteredBearingObservation(
                        two.view_id, 1, two.region.center_bearing_xyz
                    ),
                ),
            ),
            (pose_by_view[one.view_id], pose_by_view[two.view_id]),
            SphericalObjectHypothesisConfig(
                maximum_triangulation_reprojection_error_deg=(
                    config.maximum_center_reprojection_error_deg
                )
            ),
        )
        geometry_provenance = {
            "state": triangulation.state,
            "reasons": triangulation.reasons,
            "maximum_pairwise_parallax_deg": (
                triangulation.maximum_pairwise_parallax_deg
            ),
            "maximum_angular_residual_deg": (
                triangulation.maximum_angular_residual_deg
            ),
        }
        if triangulation.accepted:
            residual = float(triangulation.maximum_angular_residual_deg or 0.0)
            geometric = math.exp(
                -0.5 * (residual / config.center_reprojection_sigma_deg) ** 2
            )
        elif triangulation.state == "rejected":
            geometric_contradiction = True

    track: float | None = None
    if one.track_ids and two.track_ids:
        union = set(one.track_ids) | set(two.track_ids)
        track = len(set(one.track_ids) & set(two.track_ids)) / len(union)

    transfer: SymmetricRegionTransfer | None = None
    transfer_iou: float | None = None
    transfer_provenance: dict[str, object] = {}
    if (
        one.view_id != two.view_id
        and one.view_id in depth_by_view
        and two.view_id in depth_by_view
        and one.view_id in pose_by_view
        and two.view_id in pose_by_view
    ):
        transfer = symmetric_spherical_region_transfer(
            one.region,
            two.region,
            depth_by_view[one.view_id],
            depth_by_view[two.view_id],
            pose_by_view[one.view_id],
            pose_by_view[two.view_id],
        )
        forward_coverage = (
            transfer.forward.source_supported_depth_count
            / transfer.forward.source_region_pixel_count
            if transfer.forward.source_region_pixel_count
            else 0.0
        )
        reverse_coverage = (
            transfer.reverse.source_supported_depth_count
            / transfer.reverse.source_region_pixel_count
            if transfer.reverse.source_region_pixel_count
            else 0.0
        )
        transfer_provenance = {
            "forward_depth_coverage": forward_coverage,
            "reverse_depth_coverage": reverse_coverage,
            "forward_iou": transfer.forward.solid_angle_iou,
            "reverse_iou": transfer.reverse.solid_angle_iou,
        }
        if min(forward_coverage, reverse_coverage) >= config.minimum_depth_coverage:
            transfer_iou = math.sqrt(
                transfer.forward.solid_angle_iou * transfer.reverse.solid_angle_iou
            )
        else:
            transfer = None

    cues = SphericalAnchorPairCues(
        id_a,
        id_b,
        geometric_compatibility=geometric,
        transfer_iou=transfer_iou,
        track_consistency=track,
        geometric_contradiction=geometric_contradiction,
        provenance={
            "center_geometry": geometry_provenance,
            "registered_transfer": transfer_provenance,
        },
    )
    return cues, transfer


def _intra_view_nms(
    anchors: tuple[SphericalFOVAnchor, ...],
    config: SphericalAnchorFusionConfig,
    shape_by_view: Mapping[str, tuple[int, int]],
) -> tuple[
    tuple[SphericalFOVAnchor, ...],
    tuple[str, ...],
    tuple[SphericalAnchorDecision, ...],
]:
    retained: list[SphericalFOVAnchor] = []
    suppressed: list[str] = []
    decisions: list[SphericalAnchorDecision] = []
    groups: dict[tuple[str, str, str], list[SphericalFOVAnchor]] = {}
    for anchor in anchors:
        groups.setdefault(
            (anchor.view_id, anchor.vocabulary, anchor.concept_id), []
        ).append(anchor)
    for key in sorted(groups):
        view_id = key[0]
        shape = shape_by_view.get(view_id, config.evaluation_shape_hw)
        ordered = sorted(
            groups[key],
            key=lambda item: (
                -item.score,
                item.region.spec.hfov_deg * item.region.spec.vfov_deg,
                item.anchor_id,
            ),
        )
        group_retained: list[SphericalFOVAnchor] = []
        for anchor in ordered:
            duplicate_index = None
            duplicate_iou = None
            for index, other in enumerate(group_retained):
                overlap = _mask_iou(anchor.region, other.region, shape)
                if overlap >= config.intra_view_nms_iou:
                    duplicate_index = index
                    duplicate_iou = overlap
                    break
            if duplicate_index is not None:
                representative = group_retained[duplicate_index]
                group_retained[duplicate_index] = replace(
                    representative,
                    track_ids=tuple(
                        sorted(set(representative.track_ids) | set(anchor.track_ids))
                    ),
                    source_hypothesis_ids=tuple(
                        sorted(
                            set(representative.source_hypothesis_ids)
                            | set(anchor.source_hypothesis_ids)
                        )
                    ),
                    proposal_family_ids=tuple(
                        sorted(
                            set(representative.proposal_family_ids)
                            | set(anchor.proposal_family_ids)
                        )
                    ),
                    provenance={
                        **dict(representative.provenance),
                        "intra_view_merged_anchor_ids": tuple(
                            sorted(
                                set(
                                    representative.provenance.get(
                                        "intra_view_merged_anchor_ids", ()
                                    )
                                )
                                | {anchor.anchor_id}
                            )
                        ),
                    },
                )
                suppressed.append(anchor.anchor_id)
                decisions.append(
                    SphericalAnchorDecision(
                        anchor,
                        "suppressed-overlap",
                        representative.anchor_id,
                        duplicate_iou,
                    )
                )
            elif len(group_retained) >= config.maximum_anchors_per_view_concept:
                suppressed.append(anchor.anchor_id)
                decisions.append(
                    SphericalAnchorDecision(
                        anchor,
                        "suppressed-capacity",
                        None,
                        None,
                    )
                )
            else:
                group_retained.append(anchor)
        decisions.extend(
            SphericalAnchorDecision(item, "retained", item.anchor_id, 1.0)
            for item in group_retained
        )
        retained.extend(group_retained)
    return (
        tuple(sorted(retained, key=lambda item: item.anchor_id)),
        tuple(sorted(suppressed)),
        tuple(sorted(decisions, key=lambda item: item.anchor.anchor_id)),
    )


def _complete_link_clusters(
    anchors: tuple[SphericalFOVAnchor, ...],
    affinity_by_pair: Mapping[tuple[str, str], SphericalAnchorAffinity],
    threshold: float,
) -> tuple[tuple[SphericalFOVAnchor, ...], ...]:
    ordered = sorted(anchors, key=lambda item: (-item.score, item.anchor_id))
    clusters: list[list[SphericalFOVAnchor]] = []
    for anchor in ordered:
        candidates = []
        for index, cluster in enumerate(clusters):
            if any(item.view_id == anchor.view_id for item in cluster):
                continue
            pair_values = []
            compatible = True
            for item in cluster:
                pair = tuple(sorted((anchor.anchor_id, item.anchor_id)))
                affinity = affinity_by_pair[pair]
                if not affinity.hard_compatible or affinity.affinity < threshold:
                    compatible = False
                    break
                pair_values.append(affinity.affinity)
            if compatible:
                candidates.append(
                    (
                        min(pair_values),
                        float(np.mean(pair_values)),
                        -len(cluster),
                        -index,
                        index,
                    )
                )
        if candidates:
            index = max(candidates)[-1]
            clusters[index].append(anchor)
        else:
            clusters.append([anchor])
    return tuple(
        tuple(sorted(cluster, key=lambda item: (item.view_id, item.anchor_id)))
        for cluster in clusters
    )


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    threshold = quantile * float(cumulative[-1])
    index = min(
        int(np.searchsorted(cumulative, threshold, side="left")), len(values) - 1
    )
    return float(ordered_values[index])


def _fit_consensus_region(
    local: SphericalFOVAnchor,
    consensus: np.ndarray,
    config: SphericalAnchorFusionConfig,
) -> HypothesisGnomonicRegion:
    if not np.any(consensus):
        return local.region
    rows, columns = np.nonzero(consensus)
    pixels = np.column_stack((columns.astype(np.float64), rows.astype(np.float64)))
    bearings = np.asarray(erp_pixels_to_rays(pixels, consensus.shape), dtype=np.float64)
    area = spherical_solid_angle_weights(consensus.shape)[rows, columns]
    vector = np.sum(bearings * area[:, None], axis=0)
    if float(np.linalg.norm(vector)) <= 1.0e-12:
        vector = np.asarray(local.region.center_bearing_xyz, dtype=np.float64)
    center = vector / np.linalg.norm(vector)
    latitude = math.degrees(math.asin(float(np.clip(center[1], -1.0, 1.0))))
    longitude = math.degrees(math.atan2(float(center[0]), float(center[2])))
    probe = GnomonicSpec(
        center_lat_deg=latitude,
        center_lon_deg=longitude,
        hfov_deg=config.maximum_final_fov_deg,
        vfov_deg=config.maximum_final_fov_deg,
        output_shape_hw=local.region.spec.output_shape_hw,
    )
    rotation = np.asarray(gnomonic_rotation(probe), dtype=np.float64)
    tangent = bearings @ rotation
    positive = tangent[:, 2] > 1.0e-12
    if not np.any(positive):
        return local.region
    bearings = bearings[positive]
    area = area[positive]
    tangent = tangent[positive]
    horizontal = np.degrees(np.abs(np.arctan2(tangent[:, 0], tangent[:, 2])))
    vertical = np.degrees(np.abs(np.arctan2(tangent[:, 1], tangent[:, 2])))
    # Using q=(1+coverage)/2 on each axis gives a union-bound guarantee that
    # at least ``coverage`` of the weighted samples survive both bounds.
    axis_quantile = 0.5 * (1.0 + config.consensus_coverage)
    half_horizontal = _weighted_quantile(horizontal, area, axis_quantile)
    half_vertical = _weighted_quantile(vertical, area, axis_quantile)
    horizontal_fov = min(
        max(
            2.0 * (half_horizontal + config.consensus_margin_deg),
            config.minimum_final_fov_deg,
        ),
        config.maximum_final_fov_deg,
    )
    vertical_fov = min(
        max(
            2.0 * (half_vertical + config.consensus_margin_deg),
            config.minimum_final_fov_deg,
        ),
        config.maximum_final_fov_deg,
    )
    spec = GnomonicSpec(
        center_lat_deg=latitude,
        center_lon_deg=longitude,
        hfov_deg=horizontal_fov,
        vfov_deg=vertical_fov,
        output_shape_hw=local.region.spec.output_shape_hw,
    )
    valid = np.asarray(rays_to_gnomonic_pixels(bearings, spec).valid, dtype=bool)
    coverage = float(np.average(valid, weights=area))
    angular_radius = float(
        np.degrees(np.max(np.arccos(np.clip(bearings @ center, -1.0, 1.0))))
    )
    if len(bearings) > 512:
        selected = np.linspace(0, len(bearings) - 1, 512).astype(np.int64)
        retained_bearings = bearings[selected]
    else:
        retained_bearings = bearings
    return HypothesisGnomonicRegion(
        view_id=local.view_id,
        spec=spec,
        center_bearing_xyz=center,
        angular_radius_deg=angular_radius,
        support_bearings_xyz=retained_bearings,
        track_feature_indices=local.region.track_feature_indices,
        sparse_support_coverage=coverage,
        semantic_evidence_available=local.region.semantic_evidence_available,
        semantic_supporting=local.region.semantic_supporting,
        extent_source="multiview-spherical-anchor-consensus",
    )


def _consensus_regions(
    cluster: tuple[SphericalFOVAnchor, ...],
    *,
    affinity_by_pair: Mapping[tuple[str, str], SphericalAnchorAffinity],
    transfer_by_pair: Mapping[tuple[str, str], SymmetricRegionTransfer],
    depth_by_view: Mapping[str, RegisteredRadialDepth],
    config: SphericalAnchorFusionConfig,
) -> tuple[HypothesisGnomonicRegion, ...]:
    result = []
    for target in cluster:
        shape = (
            depth_by_view[target.view_id].shape_hw
            if target.view_id in depth_by_view
            else config.evaluation_shape_hw
        )
        local_mask = spherical_region_mask(target.region.spec, shape)
        weighted = local_mask.astype(np.float64)
        total_weight = 1.0
        for source in cluster:
            if source.anchor_id == target.anchor_id:
                continue
            pair = tuple(sorted((source.anchor_id, target.anchor_id)))
            transfer = transfer_by_pair.get(pair)
            if transfer is None:
                continue
            if transfer.forward.target_view_id == target.view_id:
                transferred = transfer.forward.transferred_mask
            elif transfer.reverse.target_view_id == target.view_id:
                transferred = transfer.reverse.transferred_mask
            else:  # pragma: no cover - guarded by the transfer contract
                continue
            weight = affinity_by_pair[pair].affinity
            if transferred.shape != shape:
                raise ValueError("transferred consensus mask shape mismatch")
            weighted += weight * transferred.astype(np.float64)
            total_weight += weight
        consensus = weighted >= config.consensus_threshold * total_weight
        result.append(_fit_consensus_region(target, consensus, config))
    return tuple(sorted(result, key=lambda item: item.view_id))


def _cluster_nms(
    clusters: list[SphericalAnchorCluster],
    config: SphericalAnchorFusionConfig,
) -> tuple[
    tuple[SphericalAnchorCluster, ...],
    tuple[str, ...],
    tuple[SphericalAnchorClusterDecision, ...],
]:
    retained: list[SphericalAnchorCluster] = []
    suppressed: list[str] = []
    decisions: list[SphericalAnchorClusterDecision] = []
    for candidate in sorted(clusters, key=lambda item: (-item.score, item.cluster_id)):
        duplicate: (
            tuple[SphericalAnchorCluster, ClusterDecisionAction, float] | None
        ) = None
        for selected in retained:
            if (
                candidate.concept_id != selected.concept_id
                or candidate.vocabulary != selected.vocabulary
                or candidate.view_ids != selected.view_ids
            ):
                continue
            same_proposal_family = bool(candidate.proposal_family_ids) and (
                candidate.proposal_family_ids == selected.proposal_family_ids
            )
            overlap = min(
                _mask_iou(first, second, config.evaluation_shape_hw)
                for first, second in zip(
                    candidate.consensus_regions,
                    selected.consensus_regions,
                    strict=True,
                )
            )
            if same_proposal_family and overlap >= config.cluster_nms_iou:
                duplicate = (
                    selected,
                    "suppressed-family-overlap",
                    overlap,
                )
                break
        if duplicate is not None:
            representative, action, overlap = duplicate
            suppressed.append(candidate.cluster_id)
            decisions.append(
                SphericalAnchorClusterDecision(
                    candidate,
                    action,
                    representative.cluster_id,
                    overlap,
                )
            )
        else:
            retained.append(candidate)
            decisions.append(
                SphericalAnchorClusterDecision(
                    candidate,
                    "retained",
                    candidate.cluster_id,
                    1.0,
                )
            )
    return (
        tuple(retained),
        tuple(sorted(suppressed)),
        tuple(sorted(decisions, key=lambda item: item.cluster.cluster_id)),
    )


def fuse_registered_spherical_anchors(
    anchors: tuple[SphericalFOVAnchor, ...],
    *,
    poses: tuple[SphericalCameraPose, ...],
    radial_depth: tuple[RegisteredRadialDepth, ...] = (),
    pair_cues: tuple[SphericalAnchorPairCues, ...] = (),
    config: SphericalAnchorFusionConfig | None = None,
) -> SphericalAnchorFusionResult:
    """Associate per-view anchors and refit registered multiview consensus FOVs."""

    policy = config or SphericalAnchorFusionConfig()
    anchor_by_id = {item.anchor_id: item for item in anchors}
    if len(anchor_by_id) != len(anchors):
        raise ValueError("anchor IDs must be unique")
    pose_by_view = {item.panorama_id: item for item in poses}
    if len(pose_by_view) != len(poses):
        raise ValueError("registered pose IDs must be unique")
    depth_by_view = {item.view_id: item for item in radial_depth}
    if len(depth_by_view) != len(radial_depth):
        raise ValueError("radial depth view IDs must be unique")
    unknown_views = sorted({item.view_id for item in anchors} - set(pose_by_view))
    if unknown_views:
        raise ValueError(f"anchors reference unregistered views: {unknown_views}")
    unknown_depth = sorted(set(depth_by_view) - set(pose_by_view))
    if unknown_depth:
        raise ValueError(f"radial depth references unregistered views: {unknown_depth}")
    cue_by_pair = {(item.anchor_id_a, item.anchor_id_b): item for item in pair_cues}
    if len(cue_by_pair) != len(pair_cues):
        raise ValueError("pair cues must identify unique anchor pairs")
    unknown_cues = sorted(
        {
            anchor_id
            for pair in cue_by_pair
            for anchor_id in pair
            if anchor_id not in anchor_by_id
        }
    )
    if unknown_cues:
        raise ValueError(f"pair cues reference unknown anchors: {unknown_cues}")

    shape_by_view = {
        view_id: depth.shape_hw for view_id, depth in depth_by_view.items()
    }
    retained, suppressed, anchor_decisions = _intra_view_nms(
        anchors, policy, shape_by_view
    )
    affinity_by_pair: dict[tuple[str, str], SphericalAnchorAffinity] = {}
    transfer_by_pair: dict[tuple[str, str], SymmetricRegionTransfer] = {}
    resolved_cues = []
    for first, second in combinations(retained, 2):
        pair = tuple(sorted((first.anchor_id, second.anchor_id)))
        if (
            first.view_id == second.view_id
            or first.concept_id != second.concept_id
            or first.vocabulary != second.vocabulary
        ):
            automatic = SphericalAnchorPairCues(*pair)
            transfer = None
        else:
            automatic, transfer = _automatic_pair_cues(
                first,
                second,
                pose_by_view=pose_by_view,
                depth_by_view=depth_by_view,
                config=policy,
            )
        cues = _merge_cues(automatic, cue_by_pair.get(pair))
        resolved_cues.append(cues)
        affinity_by_pair[pair] = spherical_anchor_pair_affinity(
            first, second, cues=cues, config=policy
        )
        if transfer is not None:
            transfer_by_pair[pair] = transfer

    raw_clusters = _complete_link_clusters(
        retained, affinity_by_pair, policy.complete_link_threshold
    )
    clusters = []
    for members in raw_clusters:
        pair_values = [
            affinity_by_pair[
                tuple(sorted((first.anchor_id, second.anchor_id)))
            ].affinity
            for first, second in combinations(members, 2)
        ]
        minimum_pair = min(pair_values) if pair_values else None
        local_score = math.exp(
            sum(math.log(max(item.score, 1.0e-12)) for item in members) / len(members)
        )
        score = (
            math.sqrt(local_score * minimum_pair)
            if minimum_pair is not None
            else local_score
        )
        views = tuple(sorted(item.view_id for item in members))
        consensus = _consensus_regions(
            members,
            affinity_by_pair=affinity_by_pair,
            transfer_by_pair=transfer_by_pair,
            depth_by_view=depth_by_view,
            config=policy,
        )
        anchor_ids = tuple(sorted(item.anchor_id for item in members))
        source_ids = tuple(
            sorted(
                {
                    source_id
                    for item in members
                    for source_id in item.source_hypothesis_ids
                }
            )
        )
        proposal_family_ids = tuple(
            sorted(
                {
                    family_id
                    for item in members
                    for family_id in item.proposal_family_ids
                }
            )
        )
        clusters.append(
            SphericalAnchorCluster(
                cluster_id=_stable_id("anchor-cluster", anchor_ids),
                concept_id=members[0].concept_id,
                vocabulary=members[0].vocabulary,
                anchor_ids=anchor_ids,
                view_ids=views,
                source_hypothesis_ids=source_ids,
                proposal_family_ids=proposal_family_ids,
                score=float(np.clip(score, 0.0, 1.0)),
                state=(
                    "confirmed"
                    if len(views) >= policy.minimum_views_for_confirmation
                    else "proposed"
                ),
                consensus_regions=consensus,
                minimum_pair_affinity=minimum_pair,
            )
        )
    raw_cluster_count = len(clusters)
    selected_clusters, suppressed_clusters, cluster_decisions = _cluster_nms(
        clusters, policy
    )
    positive_pairs = sum(
        item.hard_compatible and item.affinity >= policy.pair_affinity_threshold
        for item in affinity_by_pair.values()
    )
    return SphericalAnchorFusionResult(
        retained_anchors=retained,
        suppressed_anchor_ids=suppressed,
        anchor_decisions=anchor_decisions,
        pair_affinities=tuple(
            affinity_by_pair[pair] for pair in sorted(affinity_by_pair)
        ),
        clusters=selected_clusters,
        suppressed_cluster_ids=suppressed_clusters,
        cluster_decisions=cluster_decisions,
        diagnostics={
            "input_anchor_count": len(anchors),
            "retained_anchor_count": len(retained),
            "intra_view_suppressed_count": len(suppressed),
            "pair_count": len(affinity_by_pair),
            "positive_pair_count": positive_pairs,
            "raw_cluster_count": raw_cluster_count,
            "cluster_count": len(selected_clusters),
            "cross_view_cluster_suppressed_count": len(suppressed_clusters),
            "confirmed_cluster_count": sum(
                item.state == "confirmed" for item in selected_clusters
            ),
            "transient_masks_retained": 0,
            "clustering": "deterministic-complete-link-one-anchor-per-view",
        },
    )


__all__ = [
    "SPHERICAL_ANCHOR_FUSION_INTERFACE",
    "SphericalAnchorAffinity",
    "SphericalAnchorCluster",
    "SphericalAnchorClusterDecision",
    "SphericalAnchorDecision",
    "SphericalAnchorFusionConfig",
    "SphericalAnchorFusionResult",
    "SphericalAnchorPairCues",
    "SphericalFOVAnchor",
    "fuse_registered_spherical_anchors",
    "generate_spherical_fov_anchors",
    "spherical_anchor_pair_affinity",
]
