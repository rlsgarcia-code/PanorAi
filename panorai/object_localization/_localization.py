"""Spatial hypotheses from an accepted region association."""

from __future__ import annotations

import math

import numpy as np

from ._models import (
    ObjectLocalizationConfig,
    PairObjectLocalizationInput,
    RegionAssociation,
    SemanticRegionObservation,
    SpatialLocationHypothesis,
)


def _angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _fallback_bearing(
    region: SemanticRegionObservation, evidence: PairObjectLocalizationInput
) -> np.ndarray | None:
    if region.centroid_bearing is not None:
        return region.centroid_bearing
    if len(region.feature_indices):
        value = np.sum(evidence.features_a.bearings[region.feature_indices], axis=0)
        norm = float(np.linalg.norm(value))
        if norm > 1e-12:
            return value / norm
    return None


def localize_region_association(
    association: RegionAssociation,
    evidence: PairObjectLocalizationInput,
    config: ObjectLocalizationConfig | None = None,
) -> SpatialLocationHypothesis:
    """Triangulate regional inliers or retain an honest angular hypothesis."""

    policy = config or ObjectLocalizationConfig()
    region_a = next(
        item for item in evidence.regions_a if item.region_id == association.region_id_a
    )
    bearing = _fallback_bearing(region_a, evidence)
    scale = 1.0 if evidence.translation_scale is None else evidence.translation_scale
    rotation = np.asarray(evidence.pose.rotation, dtype=np.float64)
    translation = (
        np.asarray(evidence.pose.translation_direction, dtype=np.float64) * scale
    )
    camera_b_in_a = -rotation.T @ translation

    points: list[np.ndarray] = []
    support: list[int] = []
    parallaxes: list[float] = []
    reprojection_errors: list[float] = []
    for match_index in association.inlier_match_indices:
        ray_a = np.asarray(evidence.matches.bearings_a[match_index], dtype=np.float64)
        ray_b = np.asarray(evidence.matches.bearings_b[match_index], dtype=np.float64)
        ray_b_in_a = rotation.T @ ray_b
        parallax = _angle_deg(ray_a, ray_b_in_a)
        if parallax < policy.min_triangulation_angle_deg:
            continue
        coefficients, *_ = np.linalg.lstsq(
            np.column_stack((ray_a, -ray_b_in_a)), camera_b_in_a, rcond=None
        )
        depth_a, depth_b = coefficients
        if depth_a <= 0 or depth_b <= 0:
            continue
        point_a = depth_a * ray_a
        point_from_b = camera_b_in_a + depth_b * ray_b_in_a
        point = 0.5 * (point_a + point_from_b)
        point_b = rotation @ point + translation
        if np.linalg.norm(point) <= 1e-12 or np.linalg.norm(point_b) <= 1e-12:
            continue
        error = max(
            _angle_deg(point / np.linalg.norm(point), ray_a),
            _angle_deg(point_b / np.linalg.norm(point_b), ray_b),
        )
        if error > policy.max_reprojection_error_deg:
            continue
        points.append(point)
        support.append(int(match_index))
        parallaxes.append(parallax)
        reprojection_errors.append(error)

    if len(points) >= policy.min_triangulated_points:
        cloud = np.stack(points)
        position = np.median(cloud, axis=0)
        covariance = (
            np.cov(cloud, rowvar=False, ddof=1)
            if len(cloud) > 1
            else np.zeros((3, 3), dtype=np.float64)
        )
        covariance = np.asarray(covariance, dtype=np.float64)
        uncertainty = math.sqrt(max(float(np.linalg.eigvalsh(covariance).max()), 0.0))
        median_parallax = float(np.median(parallaxes))
        median_error = float(np.median(reprojection_errors))
        support_factor = min(1.0, len(points) / policy.min_triangulated_points)
        parallax_factor = min(
            1.0, median_parallax / max(policy.min_triangulation_angle_deg, 1e-12)
        )
        error_factor = math.exp(
            -median_error / max(policy.max_reprojection_error_deg, 1e-12)
        )
        return SpatialLocationHypothesis(
            state="localized",
            mode="scale_free_3d" if evidence.translation_scale is None else "metric_3d",
            frame=evidence.view_id_a,
            units=(
                "normalized_baseline"
                if evidence.translation_scale is None
                else evidence.metric_units
            ),
            position_xyz=position,
            covariance_xyz=covariance,
            bearing_xyz=bearing,
            uncertainty_radius=uncertainty,
            supporting_match_indices=np.asarray(support),
            triangulated_count=len(points),
            median_parallax_deg=median_parallax,
            median_reprojection_error_deg=median_error,
            location_score=float(support_factor * parallax_factor * error_factor),
        )

    return SpatialLocationHypothesis(
        state="view_only" if bearing is not None else "unobservable",
        mode="bearing_only",
        frame=evidence.view_id_a,
        units="direction",
        position_xyz=None,
        covariance_xyz=None,
        bearing_xyz=bearing,
        uncertainty_radius=None,
        supporting_match_indices=np.asarray(support),
        triangulated_count=len(points),
        median_parallax_deg=(float(np.median(parallaxes)) if parallaxes else None),
        median_reprojection_error_deg=(
            float(np.median(reprojection_errors)) if reprojection_errors else None
        ),
        location_score=0.0,
    )
