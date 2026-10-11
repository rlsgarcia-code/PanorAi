"""Compatibility conversion from object-localization/v1 to graph/v1."""

from __future__ import annotations

import math

import numpy as np

from panorai.graph import (
    EvidenceMode,
    EvidenceSource,
    PairFeatureEvidence,
    PairwiseLocalizationConfig,
    PairwiseLocalizationInput,
    RegionAssociationConfig,
    RegionAssociationInput,
    RelativePoseEdge,
    SemanticRegionNode,
    associate_regions,
    localize_region_pair,
)
from panorai.graph.observations import deterministic_id

from ._association import _stable_id, object_hypothesis_id
from ._models import (
    ObjectHypothesis,
    ObjectLocalizationConfig,
    ObjectLocalizationResult,
    PairObjectLocalizationInput,
    RegionAssociation,
    SemanticRegionObservation,
    SpatialLocationHypothesis,
)


def _bearing(
    region: SemanticRegionObservation, feature_bearings: np.ndarray
) -> np.ndarray:
    if region.centroid_bearing is not None:
        return region.centroid_bearing
    if len(region.feature_indices):
        value = np.sum(feature_bearings[region.feature_indices], axis=0)
        norm = float(np.linalg.norm(value))
        if norm > 1e-12:
            return value / norm
    # Association can still be evaluated for a region without angular support.
    # This sentinel bearing is never reported as a localized result because the
    # compatibility conversion retains the legacy unobservable semantics.
    return np.asarray([1.0, 0.0, 0.0])


def _contracts(
    evidence: PairObjectLocalizationInput,
    *,
    require_accepted_pose: bool,
) -> tuple[
    PairFeatureEvidence,
    RelativePoseEdge,
    tuple[SemanticRegionNode, ...],
    tuple[SemanticRegionNode, ...],
]:
    feature_source = EvidenceSource(
        source_id=deterministic_id(
            "source", evidence.matches.interface, evidence.view_id_a, evidence.view_id_b
        ),
        modality="spherical-features",
        mode=EvidenceMode.ESTIMATED,
        method_id=evidence.matches.matcher_name,
        interface_version=evidence.matches.interface,
        calibration_id=evidence.matches.provenance.calibration_id,
        correlation_group=deterministic_id(
            "correlation", evidence.view_id_a, evidence.view_id_b, "matches"
        ),
    )
    feature_evidence = PairFeatureEvidence(
        evidence_id=deterministic_id(
            "feature-evidence", evidence.view_id_a, evidence.view_id_b
        ),
        view_id_a=evidence.view_id_a,
        view_id_b=evidence.view_id_b,
        feature_indices_a=evidence.matches.feature_indices_a,
        feature_indices_b=evidence.matches.feature_indices_b,
        bearings_a=evidence.matches.bearings_a,
        bearings_b=evidence.matches.bearings_b,
        valid=evidence.matches.valid,
        pose_inlier_mask=evidence.pose.inlier_mask,
        pose_residuals_rad=evidence.pose.residuals_rad,
        descriptor_type=evidence.features_a.descriptor_type,
        descriptor_metric=evidence.features_a.descriptor_metric,
        source=feature_source,
    )
    pose_source = EvidenceSource(
        source_id=deterministic_id(
            "source", evidence.pose.interface, evidence.view_id_a, evidence.view_id_b
        ),
        modality="relative-pose",
        mode=EvidenceMode.ESTIMATED,
        method_id=evidence.pose.robust_estimator,
        interface_version=evidence.pose.interface,
        correlation_group=feature_source.correlation_group,
    )
    pose = RelativePoseEdge(
        edge_id=deterministic_id("pose", evidence.view_id_a, evidence.view_id_b),
        view_id_a=evidence.view_id_a,
        view_id_b=evidence.view_id_b,
        rotation_b_from_a=evidence.pose.rotation,
        translation_direction_b_from_a=evidence.pose.translation_direction,
        accepted=(evidence.pose.quality_report.accepted or not require_accepted_pose),
        degenerate=evidence.pose.degenerate,
        source=pose_source,
        feature_evidence_id=feature_evidence.evidence_id,
        translation_scale=evidence.translation_scale,
        units=(
            evidence.metric_units if evidence.translation_scale is not None else None
        ),
        failure_reasons=evidence.pose.degeneracy_reasons,
    )
    semantic_source_a = EvidenceSource(
        source_id=deterministic_id("source", evidence.view_id_a, "semantic-regions"),
        modality="semantic-regions",
        mode=EvidenceMode.EXTERNAL,
        method_id="object-localization-v1-compat",
        interface_version="panorai-object-localization/v1",
        frame=evidence.view_id_a,
    )
    semantic_source_b = EvidenceSource(
        source_id=deterministic_id("source", evidence.view_id_b, "semantic-regions"),
        modality="semantic-regions",
        mode=EvidenceMode.EXTERNAL,
        method_id="object-localization-v1-compat",
        interface_version="panorai-object-localization/v1",
        frame=evidence.view_id_b,
    )

    def convert(
        item: SemanticRegionObservation,
        source: EvidenceSource,
        bearings: np.ndarray,
    ) -> SemanticRegionNode:
        return SemanticRegionNode(
            region_id=item.region_id,
            view_id=item.view_id,
            class_id=item.class_id,
            class_name=item.class_name,
            vocabulary=evidence.query.vocabulary,
            semantic_score=item.semantic_score,
            feature_indices=item.feature_indices,
            membership_weights=item.membership_weights,
            centroid_bearing=_bearing(item, bearings),
            source=source,
        )

    regions_a = tuple(
        convert(item, semantic_source_a, evidence.features_a.bearings)
        for item in evidence.regions_a
    )
    regions_b = tuple(
        convert(item, semantic_source_b, evidence.features_b.bearings)
        for item in evidence.regions_b
    )
    return feature_evidence, pose, regions_a, regions_b


def _association_config(config: ObjectLocalizationConfig) -> RegionAssociationConfig:
    return RegionAssociationConfig(
        min_region_matches=config.min_region_matches,
        min_region_pose_inliers=config.min_region_pose_inliers,
        max_median_pose_residual_deg=config.max_median_pose_residual_deg,
        ambiguity_margin=config.ambiguity_margin,
        require_accepted_pose=config.require_accepted_pose,
    )


def _localization_config(
    config: ObjectLocalizationConfig,
) -> PairwiseLocalizationConfig:
    return PairwiseLocalizationConfig(
        min_triangulated_points=config.min_triangulated_points,
        min_triangulation_angle_deg=config.min_triangulation_angle_deg,
        max_reprojection_error_deg=config.max_reprojection_error_deg,
    )


def estimate_via_graph(
    evidence: PairObjectLocalizationInput,
    config: ObjectLocalizationConfig,
) -> ObjectLocalizationResult:
    features, pose, regions_a, regions_b = _contracts(
        evidence, require_accepted_pose=config.require_accepted_pose
    )
    graph_result = associate_regions(
        RegionAssociationInput(
            regions_a=regions_a,
            regions_b=regions_b,
            feature_evidence=features,
            pose=pose,
            query_class_ids=evidence.query.class_ids,
        ),
        _association_config(config),
    )
    associations = tuple(
        RegionAssociation(
            association_id=_stable_id(
                "association",
                candidate.view_id_a,
                candidate.region_id_a,
                candidate.view_id_b,
                candidate.region_id_b,
            ),
            region_id_a=candidate.region_id_a,
            region_id_b=candidate.region_id_b,
            class_id=candidate.class_id,
            class_name=candidate.class_name,
            match_indices=candidate.match_indices,
            inlier_match_indices=candidate.inlier_match_indices,
            match_count=len(candidate.match_indices),
            inlier_count=len(candidate.inlier_match_indices),
            semantic_score=candidate.semantic_score,
            median_pose_residual_deg=(
                None
                if candidate.median_pose_residual_rad is None
                else math.degrees(candidate.median_pose_residual_rad)
            ),
            ranking_score=candidate.ranking_score,
            state=candidate.state,
            reasons=candidate.reasons,
        )
        for candidate in graph_result.candidates
    )
    candidate_by_pair = {
        (item.region_id_a, item.region_id_b): item for item in graph_result.candidates
    }
    region_a_by_id = {item.region_id: item for item in regions_a}
    region_b_by_id = {item.region_id: item for item in regions_b}
    old_region_a_by_id = {item.region_id: item for item in evidence.regions_a}
    hypotheses: list[ObjectHypothesis] = []
    for association in associations:
        if association.state != "accepted":
            continue
        candidate = candidate_by_pair[
            (association.region_id_a, association.region_id_b)
        ]
        localized = localize_region_pair(
            PairwiseLocalizationInput(
                candidate=candidate,
                region_a=region_a_by_id[association.region_id_a],
                region_b=region_b_by_id[association.region_id_b],
                feature_evidence=features,
                pose=pose,
            ),
            _localization_config(config),
        )
        details = localized
        original_region = old_region_a_by_id[association.region_id_a]
        fallback_bearing = (
            original_region.centroid_bearing
            if original_region.centroid_bearing is not None
            else (
                region_a_by_id[association.region_id_a].centroid_bearing
                if len(original_region.feature_indices)
                else None
            )
        )
        if len(details.triangulated_points) >= config.min_triangulated_points:
            cloud = details.triangulated_points
            position = np.median(cloud, axis=0)
            covariance = (
                np.cov(cloud, rowvar=False, ddof=1)
                if len(cloud) > 1
                else np.zeros((3, 3), dtype=np.float64)
            )
            uncertainty = math.sqrt(
                max(float(np.linalg.eigvalsh(covariance).max()), 0.0)
            )
            median_parallax = float(np.median(details.parallax_deg))
            median_error = float(np.median(details.reprojection_error_deg))
            support_factor = min(1.0, len(cloud) / config.min_triangulated_points)
            parallax_factor = min(
                1.0,
                median_parallax / max(config.min_triangulation_angle_deg, 1e-12),
            )
            error_factor = math.exp(
                -median_error / max(config.max_reprojection_error_deg, 1e-12)
            )
            location = SpatialLocationHypothesis(
                state="localized",
                mode=(
                    "scale_free_3d"
                    if evidence.translation_scale is None
                    else "metric_3d"
                ),
                frame=evidence.view_id_a,
                units=(
                    "normalized_baseline"
                    if evidence.translation_scale is None
                    else evidence.metric_units
                ),
                position_xyz=position,
                covariance_xyz=covariance,
                bearing_xyz=fallback_bearing,
                uncertainty_radius=uncertainty,
                supporting_match_indices=details.supporting_match_indices,
                triangulated_count=len(cloud),
                median_parallax_deg=median_parallax,
                median_reprojection_error_deg=median_error,
                location_score=float(support_factor * parallax_factor * error_factor),
            )
        else:
            location = SpatialLocationHypothesis(
                state="view_only" if fallback_bearing is not None else "unobservable",
                mode="bearing_only",
                frame=evidence.view_id_a,
                units="direction",
                position_xyz=None,
                covariance_xyz=None,
                bearing_xyz=fallback_bearing,
                uncertainty_radius=None,
                supporting_match_indices=details.supporting_match_indices,
                triangulated_count=len(details.triangulated_points),
                median_parallax_deg=(
                    float(np.median(details.parallax_deg))
                    if len(details.parallax_deg)
                    else None
                ),
                median_reprojection_error_deg=(
                    float(np.median(details.reprojection_error_deg))
                    if len(details.reprojection_error_deg)
                    else None
                ),
                location_score=0.0,
            )
        hypotheses.append(
            ObjectHypothesis(
                hypothesis_id=object_hypothesis_id(
                    evidence.view_id_a,
                    association.region_id_a,
                    evidence.view_id_b,
                    association.region_id_b,
                ),
                class_id=association.class_id,
                class_name=association.class_name,
                identity_state="confirmed",
                view_ids=(evidence.view_id_a, evidence.view_id_b),
                region_ids=(association.region_id_a, association.region_id_b),
                association_id=association.association_id,
                semantic_score=association.semantic_score,
                identity_score=association.ranking_score,
                location=location,
            )
        )
    failure_reasons = (
        ("relative-pose-not-accepted",)
        if config.require_accepted_pose and not evidence.pose.quality_report.accepted
        else ()
    )
    return ObjectLocalizationResult(
        query=evidence.query,
        hypotheses=tuple(hypotheses),
        associations=associations,
        failure_reasons=failure_reasons,
        diagnostics={
            "region_count_a": len(evidence.regions_a),
            "region_count_b": len(evidence.regions_b),
            "proposal_count": len(graph_result.candidates),
            "accepted_association_count": sum(
                item.state == "accepted" for item in associations
            ),
            "hypothesis_count": len(hypotheses),
            "query_vocabulary": evidence.query.vocabulary,
        },
    )


__all__ = ["estimate_via_graph"]
