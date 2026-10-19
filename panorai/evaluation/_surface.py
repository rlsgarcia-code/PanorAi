"""Deterministic geometric evaluation for reconstructed triangle surfaces."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from panorai.reconstruction import SurfaceMesh


@dataclass(frozen=True, slots=True)
class SurfaceReconstructionEvaluation:
    """Bidirectional point-to-triangle surface metrics for one scene."""

    accuracy_mean: float
    accuracy_median: float
    completeness_mean_distance: float
    completeness_median_distance: float
    precision: float
    recall: float
    fscore: float
    normal_consistency: float
    median_normal_error_deg: float
    observed_surface_coverage: float
    distance_threshold: float
    prediction_sample_count: int
    reference_sample_count: int
    units: str
    interface: str = "panorai-surface-evaluation/v1-experimental"

    @property
    def metrics(self) -> dict[str, float]:
        return {
            "accuracy_mean": self.accuracy_mean,
            "accuracy_median": self.accuracy_median,
            "completeness_mean_distance": self.completeness_mean_distance,
            "completeness_median_distance": self.completeness_median_distance,
            "precision": self.precision,
            "recall": self.recall,
            "fscore": self.fscore,
            "normal_consistency": self.normal_consistency,
            "median_normal_error_deg": self.median_normal_error_deg,
            "observed_surface_coverage": self.observed_surface_coverage,
        }


@dataclass(frozen=True, slots=True)
class SurfaceReconstructionEvaluationReport:
    """Per-scene results and an unweighted median across a frozen cohort."""

    per_scene: Mapping[str, SurfaceReconstructionEvaluation]
    aggregate: Mapping[str, float]
    scene_count: int
    distance_threshold: float
    units: str
    aggregation: str = "unweighted-scene-median"
    interface: str = "panorai-surface-evaluation-report/v1-experimental"

    def __post_init__(self) -> None:
        per_scene = dict(self.per_scene)
        if not per_scene or any(not key for key in per_scene):
            raise ValueError("per_scene must contain named scene evaluations")
        if len(per_scene) != self.scene_count:
            raise ValueError("scene_count must equal the number of scene evaluations")
        object.__setattr__(self, "per_scene", MappingProxyType(per_scene))
        object.__setattr__(self, "aggregate", MappingProxyType(dict(self.aggregate)))


def aggregate_surface_reconstruction(
    per_scene: Mapping[str, SurfaceReconstructionEvaluation],
) -> SurfaceReconstructionEvaluationReport:
    """Aggregate a preregistered cohort without weighting large meshes more.

    Every scene must use the same units and frozen distance threshold.  Each
    reported metric is the median of the scene-level metric, matching the
    promotion protocol's per-corpus median interpretation.
    """

    evaluations = dict(per_scene)
    if not evaluations:
        raise ValueError("per_scene must contain at least one evaluation")
    if any(not isinstance(key, str) or not key for key in evaluations):
        raise ValueError("scene IDs must be non-empty strings")
    if any(
        not isinstance(value, SurfaceReconstructionEvaluation)
        for value in evaluations.values()
    ):
        raise TypeError("per_scene values must be SurfaceReconstructionEvaluation")
    ordered = tuple(evaluations[key] for key in sorted(evaluations))
    threshold = ordered[0].distance_threshold
    units = ordered[0].units
    if any(item.distance_threshold != threshold for item in ordered[1:]):
        raise ValueError("all scenes must use the same distance_threshold")
    if any(item.units != units for item in ordered[1:]):
        raise ValueError("all scenes must use the same units")
    metric_names = tuple(ordered[0].metrics)
    aggregate = {
        name: float(np.median([item.metrics[name] for item in ordered]))
        for name in metric_names
    }
    return SurfaceReconstructionEvaluationReport(
        per_scene={key: evaluations[key] for key in sorted(evaluations)},
        aggregate=aggregate,
        scene_count=len(ordered),
        distance_threshold=threshold,
        units=units,
    )


def _closest_point_triangle(
    point: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return closest points and squared distances for one point, vectorized."""

    a = triangles[:, 0]
    b = triangles[:, 1]
    c = triangles[:, 2]
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    closest = np.empty_like(a)
    assigned = (d1 <= 0.0) & (d2 <= 0.0)
    closest[assigned] = a[assigned]

    bp = point - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    mask = (~assigned) & (d3 >= 0.0) & (d4 <= d3)
    closest[mask] = b[mask]
    assigned |= mask

    vc = d1 * d4 - d3 * d2
    mask = (~assigned) & (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    denominator = np.where(mask, d1 - d3, 1.0)
    v = d1 / denominator
    closest[mask] = a[mask] + v[mask, None] * ab[mask]
    assigned |= mask

    cp = point - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    mask = (~assigned) & (d6 >= 0.0) & (d5 <= d6)
    closest[mask] = c[mask]
    assigned |= mask

    vb = d5 * d2 - d1 * d6
    mask = (~assigned) & (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    denominator = np.where(mask, d2 - d6, 1.0)
    w = d2 / denominator
    closest[mask] = a[mask] + w[mask, None] * ac[mask]
    assigned |= mask

    va = d3 * d6 - d5 * d4
    mask = (~assigned) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0) & (va <= 0.0)
    denominator = np.where(mask, (d4 - d3) + (d5 - d6), 1.0)
    w = (d4 - d3) / denominator
    closest[mask] = b[mask] + w[mask, None] * (c[mask] - b[mask])
    assigned |= mask

    mask = ~assigned
    denominator = va + vb + vc
    denominator = np.where(np.abs(denominator) > 1e-20, denominator, 1.0)
    v = vb / denominator
    w = vc / denominator
    closest[mask] = a[mask] + ab[mask] * v[mask, None] + ac[mask] * w[mask, None]
    squared = np.einsum("ij,ij->i", closest - point, closest - point)
    return closest, squared


def _point_to_surface(
    points: np.ndarray, mesh: SurfaceMesh
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangles = mesh.vertices_xyz[mesh.faces]
    if not len(triangles):
        raise ValueError("surface evaluation requires meshes with faces")
    centroids = np.mean(triangles, axis=1)
    radii = np.max(np.linalg.norm(triangles - centroids[:, None, :], axis=2), axis=1)
    maximum_radius = float(np.max(radii))
    tree = cKDTree(centroids)
    distances = np.empty(len(points), dtype=np.float64)
    face_indices = np.empty(len(points), dtype=np.int64)
    closest_points = np.empty_like(points, dtype=np.float64)
    for index, point in enumerate(points):
        _, nearest = tree.query(point, k=1)
        _, initial_squared = _closest_point_triangle(
            point, triangles[np.asarray([nearest])]
        )
        bound = math.sqrt(float(initial_squared[0])) + maximum_radius
        candidates = np.asarray(tree.query_ball_point(point, bound), dtype=np.int64)
        candidate_points, squared = _closest_point_triangle(
            point, triangles[candidates]
        )
        selected = int(np.argmin(squared))
        distances[index] = math.sqrt(float(squared[selected]))
        face_indices[index] = candidates[selected]
        closest_points[index] = candidate_points[selected]
    return distances, face_indices, closest_points


def _surface_normals_at(
    mesh: SurfaceMesh, face_indices: np.ndarray, points: np.ndarray
) -> np.ndarray:
    triangles = mesh.vertices_xyz[mesh.faces[face_indices]]
    vertex_normals = mesh.normals_xyz[mesh.faces[face_indices]]
    v0 = triangles[:, 1] - triangles[:, 0]
    v1 = triangles[:, 2] - triangles[:, 0]
    v2 = points - triangles[:, 0]
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    denominator = d00 * d11 - d01 * d01
    if np.any(np.abs(denominator) <= 1e-20):
        raise ValueError("surface evaluation does not accept degenerate faces")
    weight_b = (d11 * d20 - d01 * d21) / denominator
    weight_c = (d00 * d21 - d01 * d20) / denominator
    weight_a = 1.0 - weight_b - weight_c
    normals = (
        vertex_normals[:, 0] * weight_a[:, None]
        + vertex_normals[:, 1] * weight_b[:, None]
        + vertex_normals[:, 2] * weight_c[:, None]
    )
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 1e-15):
        raise ValueError("surface interpolation produced a zero normal")
    return normals / lengths[:, None]


def evaluate_surface_reconstruction(
    prediction: SurfaceMesh,
    reference: SurfaceMesh,
    *,
    distance_threshold: float,
    reference_observed: Any | None = None,
) -> SurfaceReconstructionEvaluation:
    """Evaluate one reconstructed mesh against a world-frame reference mesh.

    Distances are exact point-to-triangle distances from each mesh's vertices
    to the other surface. ``reference_observed`` can restrict completeness to
    externally established observable reference vertices; it is never derived
    from color, coordinate values, or the prediction.
    """

    from panorai.reconstruction import SurfaceMesh

    if not isinstance(prediction, SurfaceMesh) or not isinstance(
        reference, SurfaceMesh
    ):
        raise TypeError("prediction and reference must be SurfaceMesh objects")
    if prediction.units != reference.units or prediction.frame != reference.frame:
        raise ValueError("prediction and reference must share units and frame")
    if not math.isfinite(distance_threshold) or distance_threshold <= 0.0:
        raise ValueError("distance_threshold must be positive and finite")
    if len(prediction.vertices_xyz) == 0 or len(reference.vertices_xyz) == 0:
        raise ValueError("surface evaluation requires non-empty meshes")
    if reference_observed is None:
        observed = np.ones(len(reference.vertices_xyz), dtype=bool)
    else:
        observed = np.asarray(reference_observed)
        if observed.dtype != np.bool_ or observed.shape != (
            len(reference.vertices_xyz),
        ):
            raise ValueError("reference_observed must be a boolean mask over vertices")
    if not np.any(observed):
        raise ValueError("reference_observed must contain at least one observed vertex")

    accuracy_distances, reference_faces, closest_reference_points = _point_to_surface(
        prediction.vertices_xyz, reference
    )
    completeness_distances, _, _ = _point_to_surface(
        reference.vertices_xyz[observed], prediction
    )
    precision = float(np.mean(accuracy_distances <= distance_threshold))
    recall = float(np.mean(completeness_distances <= distance_threshold))
    fscore = (
        0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )

    prediction_normals = np.asarray(prediction.normals_xyz, dtype=np.float64)
    reference_normals = _surface_normals_at(
        reference, reference_faces, closest_reference_points
    )
    cosine = np.clip(
        np.einsum("ij,ij->i", prediction_normals, reference_normals),
        -1.0,
        1.0,
    )
    normal_errors = np.degrees(np.arccos(cosine))
    return SurfaceReconstructionEvaluation(
        accuracy_mean=float(np.mean(accuracy_distances)),
        accuracy_median=float(np.median(accuracy_distances)),
        completeness_mean_distance=float(np.mean(completeness_distances)),
        completeness_median_distance=float(np.median(completeness_distances)),
        precision=precision,
        recall=recall,
        fscore=fscore,
        normal_consistency=float(np.mean(cosine)),
        median_normal_error_deg=float(np.median(normal_errors)),
        observed_surface_coverage=float(np.mean(observed)),
        distance_threshold=float(distance_threshold),
        prediction_sample_count=len(prediction.vertices_xyz),
        reference_sample_count=int(np.count_nonzero(observed)),
        units=prediction.units,
    )


__all__ = [
    "SurfaceReconstructionEvaluation",
    "SurfaceReconstructionEvaluationReport",
    "aggregate_surface_reconstruction",
    "evaluate_surface_reconstruction",
]
