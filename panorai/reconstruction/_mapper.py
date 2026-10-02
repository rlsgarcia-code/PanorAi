"""Global spherical reconstruction from PanorAi pairwise evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Iterable, Sequence

import numpy as np

from ._math import (
    angular_error,
    rotation_exp,
    rotation_log,
    spherical_log_residual,
)
from ._models import (
    SphericalCameraPose,
    SphericalGlobalMapperOptions,
    SphericalPairwisePoseEdge,
    SphericalReconstructionDiagnostics,
    SphericalReconstructionResult,
    SphericalTrack,
    SphericalTrackObservation,
)


@dataclass(slots=True)
class _Observation:
    panorama_id: str
    feature_index: int
    bearing: np.ndarray


@dataclass(slots=True)
class _Track:
    observations: list[_Observation]


class SphericalGlobalMapper:
    """PanorAi-owned Experimental global mapper for central panoramas."""

    def __init__(
        self,
        relative_pose_estimator: Any | None = None,
        options: SphericalGlobalMapperOptions | None = None,
    ) -> None:
        if relative_pose_estimator is None:
            from panorai.estimators import SphericalRelativePoseEstimator

            relative_pose_estimator = SphericalRelativePoseEstimator()
        if not hasattr(relative_pose_estimator, "estimate"):
            raise TypeError("relative_pose_estimator must provide estimate()")
        self.relative_pose_estimator = relative_pose_estimator
        self.options = options or SphericalGlobalMapperOptions()

    def estimate_pairwise(
        self, matches: Sequence[Any] | Iterable[Any]
    ) -> tuple[SphericalPairwisePoseEdge, ...]:
        """Estimate one relative pose for every supplied match set.

        Pairs for which the relative estimator returns ``None`` are omitted.
        Rejected but numerically successful poses remain available for audit;
        normal global reconstruction admission filters them later.
        """

        results = []
        for match_set in tuple(matches):
            if not hasattr(match_set, "to_bearing_correspondences"):
                raise TypeError(
                    "each match set must provide to_bearing_correspondences()"
                )
            pose = self.relative_pose_estimator.estimate(
                match_set.to_bearing_correspondences()
            )
            if pose is not None:
                results.append(SphericalPairwisePoseEdge(match_set, pose))
        return tuple(results)

    def reconstruct(
        self,
        *,
        matches: Sequence[Any] | Iterable[Any] | None = None,
        edges: Sequence[SphericalPairwisePoseEdge]
        | Iterable[SphericalPairwisePoseEdge]
        | None = None,
        panorama_ids: Sequence[str] | None = None,
        reference_id: str | None = None,
    ) -> SphericalReconstructionResult:
        """Reconstruct an arbitrary-scale panorama rig and sparse points.

        Exactly one of ``matches`` and ``edges`` is required. Invalid input is
        rejected. Geometric insufficiency returns an unsuccessful result with
        explicit reasons and no partial geometry.
        """

        if (matches is None) == (edges is None):
            raise ValueError("provide exactly one of matches or edges")
        input_matches = tuple(matches) if matches is not None else ()
        pairwise = (
            self.estimate_pairwise(input_matches)
            if matches is not None
            else tuple(edges or ())
        )
        if not all(isinstance(item, SphericalPairwisePoseEdge) for item in pairwise):
            raise TypeError("edges must contain SphericalPairwisePoseEdge objects")
        _validate_unique_edges(pairwise)

        all_ids = set(str(item) for item in (panorama_ids or ()))
        for edge in pairwise:
            all_ids.update(edge.pair)
        diagnostics = SphericalReconstructionDiagnostics(
            input_edge_count=len(input_matches)
            if matches is not None
            else len(pairwise),
            successful_edge_count=len(pairwise),
        )
        admitted_indices = tuple(
            index
            for index, edge in enumerate(pairwise)
            if self.options.edge_admission == "successful" or edge.accepted
        )
        rejected = tuple(
            pairwise[index].pair
            for index in range(len(pairwise))
            if index not in admitted_indices
        )
        diagnostics = replace(
            diagnostics,
            admitted_edge_count=len(admitted_indices),
            rejected_edge_pairs=rejected,
        )
        if not admitted_indices:
            return self._failure(
                pairwise,
                admitted_indices,
                diagnostics,
                "no-admitted-pairwise-edges",
            )

        admitted = [pairwise[index] for index in admitted_indices]
        component = _largest_component(admitted)
        if len(component) < self.options.min_panoramas:
            return self._failure(
                pairwise,
                admitted_indices,
                replace(
                    diagnostics,
                    excluded_panoramas=tuple(sorted(all_ids - component)),
                ),
                "insufficient-connected-panoramas",
            )
        if reference_id is not None and reference_id not in component:
            raise ValueError("reference_id must belong to the selected component")
        reference = reference_id or min(component)
        working = [edge for edge in admitted if set(edge.pair) <= component]

        rotations, rotation_costs, rotation_errors = _average_rotations(
            working, reference, self.options
        )
        kept = [
            edge
            for edge, error in zip(working, rotation_errors)
            if error <= math.radians(self.options.rotation_max_error_deg)
        ]
        filtered_pairs = tuple(
            edge.pair
            for edge, error in zip(working, rotation_errors)
            if error > math.radians(self.options.rotation_max_error_deg)
        )
        if filtered_pairs:
            component = _largest_component(kept)
            if reference not in component:
                if reference_id is not None:
                    return self._failure(
                        pairwise,
                        admitted_indices,
                        replace(diagnostics, rotation_filtered_pairs=filtered_pairs),
                        "reference-disconnected-after-rotation-filtering",
                    )
                reference = min(component) if component else reference
            if len(component) < self.options.min_panoramas:
                return self._failure(
                    pairwise,
                    admitted_indices,
                    replace(diagnostics, rotation_filtered_pairs=filtered_pairs),
                    "insufficient-panoramas-after-rotation-filtering",
                )
            working = [edge for edge in kept if set(edge.pair) <= component]
            rotations, rotation_costs, _ = _average_rotations(
                working, reference, self.options
            )

        diagnostics = replace(
            diagnostics,
            rotation_filtered_pairs=filtered_pairs,
            excluded_panoramas=tuple(sorted(all_ids - component)),
            rotation_initial_cost=rotation_costs[0],
            rotation_final_cost=rotation_costs[1],
        )
        tracks, candidate_count, conflict_count = _build_tracks(
            working, component, self.options.min_track_length
        )
        diagnostics = replace(
            diagnostics,
            candidate_match_count=candidate_count,
            track_count=len(tracks),
            track_conflict_count=conflict_count,
        )
        if not tracks:
            return self._failure(
                pairwise,
                admitted_indices,
                diagnostics,
                "no-consistent-multiview-tracks",
            )

        centers, position_costs = _initialize_centers(
            working, rotations, reference, self.options
        )
        centers, points, position_costs_bata, scale_anchor = _global_position(
            tracks, rotations, centers, reference, self.options
        )
        diagnostics = replace(
            diagnostics,
            position_initial_cost=position_costs[0] + position_costs_bata[0],
            position_final_cost=position_costs[1] + position_costs_bata[1],
            scale_anchor=scale_anchor,
        )

        active = [np.ones(len(track.observations), dtype=bool) for track in tracks]
        reasons: list[list[str | None]] = [
            [None] * len(track.observations) for track in tracks
        ]
        bundle_costs: list[tuple[str, float, float]] = []
        ba_anchor = _ba_scale_anchor(centers, reference, working)
        for iteration in range(self.options.max_refinement_rounds):
            rotations, centers, points, costs = _bundle_adjust(
                tracks,
                active,
                rotations,
                centers,
                points,
                reference,
                ba_anchor,
                self.options,
                joint=False,
            )
            bundle_costs.append((f"fixed-rotation-{iteration}", *costs))
            rotations, centers, points, costs = _bundle_adjust(
                tracks,
                active,
                rotations,
                centers,
                points,
                reference,
                ba_anchor,
                self.options,
                joint=True,
            )
            bundle_costs.append((f"joint-{iteration}", *costs))
            changed = _filter_observations(
                tracks,
                active,
                reasons,
                rotations,
                centers,
                points,
                self.options,
            )
            if not any(mask.sum() >= self.options.min_track_length for mask in active):
                return self._failure(
                    pairwise,
                    admitted_indices,
                    replace(diagnostics, bundle_costs=tuple(bundle_costs)),
                    "all-tracks-rejected-during-refinement",
                )
            points = _retriangulate(tracks, active, rotations, centers, points)
            if not changed:
                break

        rotations, centers, points, costs = _bundle_adjust(
            tracks,
            active,
            rotations,
            centers,
            points,
            reference,
            ba_anchor,
            self.options,
            joint=True,
        )
        bundle_costs.append(("joint-final", *costs))
        public_tracks, final_points, filtered_count = _public_tracks(
            tracks, active, reasons, rotations, centers, points, self.options
        )
        if not public_tracks:
            return self._failure(
                pairwise,
                admitted_indices,
                replace(diagnostics, bundle_costs=tuple(bundle_costs)),
                "no-tracks-after-final-bundle-adjustment",
            )
        camera_poses = tuple(
            SphericalCameraPose(
                panorama_id, rotations[panorama_id], centers[panorama_id]
            )
            for panorama_id in sorted(component)
        )
        final_admitted = tuple(
            index for index in admitted_indices if pairwise[index] in working
        )
        diagnostics = replace(
            diagnostics,
            admitted_edge_count=len(working),
            track_count=len(public_tracks),
            filtered_observation_count=filtered_count,
            bundle_costs=tuple(bundle_costs),
            ba_scale_anchor=ba_anchor,
            stage_messages=(
                "rotation-averaging",
                "conflict-free-track-union",
                "pairwise-translation-initialization",
                "bata-camera-point-positioning",
                "fixed-rotation-bundle-adjustment",
                "joint-spherical-bundle-adjustment",
                "filter-retriangulate-final-refinement",
            ),
        )
        return SphericalReconstructionResult(
            success=True,
            failure_reasons=(),
            poses=camera_poses,
            tracks=public_tracks,
            pairwise_edges=pairwise,
            admitted_edge_indices=final_admitted,
            reference_panorama_id=reference,
            diagnostics=diagnostics,
            options=self.options,
            _points_xyz=final_points,
        )

    def _failure(
        self,
        edges: tuple[SphericalPairwisePoseEdge, ...],
        admitted: tuple[int, ...],
        diagnostics: SphericalReconstructionDiagnostics,
        *reasons: str,
    ) -> SphericalReconstructionResult:
        return SphericalReconstructionResult(
            success=False,
            failure_reasons=tuple(reasons),
            poses=(),
            tracks=(),
            pairwise_edges=edges,
            admitted_edge_indices=admitted,
            reference_panorama_id=None,
            diagnostics=diagnostics,
            options=self.options,
        )

    def __repr__(self) -> str:
        return (
            "SphericalGlobalMapper("
            f"edge_admission={self.options.edge_admission!r}, "
            f"max_reprojection_error_deg="
            f"{self.options.max_reprojection_error_deg!r})"
        )


def _validate_unique_edges(edges: Sequence[SphericalPairwisePoseEdge]) -> None:
    seen = set()
    for edge in edges:
        key = tuple(sorted(edge.pair))
        if key in seen:
            raise ValueError(f"duplicate unordered panorama pair: {key}")
        seen.add(key)


def _largest_component(edges: Sequence[SphericalPairwisePoseEdge]) -> set[str]:
    adjacency: dict[str, set[str]] = {}
    for edge in edges:
        a, b = edge.pair
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    components = []
    unseen = set(adjacency)
    while unseen:
        start = min(unseen)
        stack = [start]
        component = set()
        while stack:
            item = stack.pop()
            if item in component:
                continue
            component.add(item)
            stack.extend(sorted(adjacency[item] - component, reverse=True))
        unseen -= component
        edge_weight = sum(
            int(edge.pose.num_inliers) for edge in edges if set(edge.pair) <= component
        )
        components.append(
            (len(component), edge_weight, tuple(sorted(component)), component)
        )
    if not components:
        return set()
    components.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return components[0][3]


def _edge_weight(edge: SphericalPairwisePoseEdge, median: float) -> float:
    return max(1e-6, float(edge.pose.num_inliers) / max(median, 1.0))


def _average_rotations(
    edges: Sequence[SphericalPairwisePoseEdge],
    reference: str,
    options: SphericalGlobalMapperOptions,
) -> tuple[dict[str, np.ndarray], tuple[float, float], list[float]]:
    from scipy.optimize import least_squares  # type: ignore[import-untyped]

    ids = sorted({item for edge in edges for item in edge.pair})
    rotations = _rotation_tree_initialization(edges, reference)
    variable_ids = [item for item in ids if item != reference]
    offsets = {item: 3 * index for index, item in enumerate(variable_ids)}
    median = float(np.median([edge.pose.num_inliers for edge in edges]))

    def unpack(parameters: np.ndarray) -> dict[str, np.ndarray]:
        result = {reference: np.eye(3)}
        for panorama_id in variable_ids:
            start = offsets[panorama_id]
            result[panorama_id] = (
                rotation_exp(parameters[start : start + 3]) @ rotations[panorama_id]
            )
        return result

    def residual(parameters: np.ndarray) -> np.ndarray:
        current = unpack(parameters)
        values: list[float] = []
        for edge in edges:
            a, b = edge.pair
            error = rotation_log(np.asarray(edge.pose.R).T @ current[b] @ current[a].T)
            values.extend(math.sqrt(_edge_weight(edge, median)) * error)
        return np.asarray(values)

    initial = np.zeros(3 * len(variable_ids), dtype=np.float64)
    initial_cost = float(np.mean(residual(initial) ** 2))
    solved = least_squares(
        residual,
        initial,
        loss="cauchy",
        f_scale=math.radians(options.rotation_robust_scale_deg),
        max_nfev=options.rotation_max_iterations,
    )
    result = unpack(solved.x)
    final_cost = float(np.mean(residual(solved.x) ** 2))
    errors = [
        float(
            np.linalg.norm(
                rotation_log(
                    np.asarray(edge.pose.R).T
                    @ result[edge.panorama_id_b]
                    @ result[edge.panorama_id_a].T
                )
            )
        )
        for edge in edges
    ]
    return result, (initial_cost, final_cost), errors


def _rotation_tree_initialization(
    edges: Sequence[SphericalPairwisePoseEdge], reference: str
) -> dict[str, np.ndarray]:
    rotations = {reference: np.eye(3)}
    ids = {item for edge in edges for item in edge.pair}
    while set(rotations) != ids:
        candidates = []
        for edge in edges:
            a, b = edge.pair
            crossing = (a in rotations) != (b in rotations)
            if crossing:
                candidates.append(
                    (-int(edge.pose.num_inliers), tuple(sorted(edge.pair)), edge)
                )
        if not candidates:
            raise ValueError("rotation graph is disconnected")
        _, _, edge = min(candidates, key=lambda item: (item[0], item[1]))
        a, b = edge.pair
        relative = np.asarray(edge.pose.R, dtype=np.float64)
        if a in rotations:
            rotations[b] = relative @ rotations[a]
        else:
            rotations[a] = relative.T @ rotations[b]
    return rotations


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[tuple[str, int], tuple[str, int]] = {}
        self.panoramas: dict[tuple[str, int], set[str]] = {}

    def add(self, item: tuple[str, int]) -> None:
        if item not in self.parent:
            self.parent[item] = item
            self.panoramas[item] = {item[0]}

    def find(self, item: tuple[str, int]) -> tuple[str, int]:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, first: tuple[str, int], second: tuple[str, int]) -> bool:
        left, right = self.find(first), self.find(second)
        if left == right:
            return True
        if self.panoramas[left] & self.panoramas[right]:
            return False
        root, child = sorted((left, right))
        self.parent[child] = root
        self.panoramas[root] |= self.panoramas.pop(child)
        return True


def _build_tracks(
    edges: Sequence[SphericalPairwisePoseEdge],
    component: set[str],
    min_length: int,
) -> tuple[list[_Track], int, int]:
    candidates = []
    bearings: dict[tuple[str, int], np.ndarray] = {}
    for edge in edges:
        if not set(edge.pair) <= component:
            continue
        matches = edge.matches
        active = np.asarray(matches.valid, dtype=bool) & np.asarray(
            edge.pose.inlier_mask, dtype=bool
        )
        arrays = (
            np.asarray(matches.feature_indices_a),
            np.asarray(matches.feature_indices_b),
            np.asarray(matches.bearings_a),
            np.asarray(matches.bearings_b),
            np.asarray(matches.descriptor_distances),
        )
        if any(array.shape[0] != len(active) for array in arrays):
            raise ValueError("match fields must align with the pairwise inlier mask")
        for row in np.flatnonzero(active):
            left = (edge.panorama_id_a, int(arrays[0][row]))
            right = (edge.panorama_id_b, int(arrays[1][row]))
            for key, value in ((left, arrays[2][row]), (right, arrays[3][row])):
                bearing = np.asarray(value, dtype=np.float64)
                if bearing.shape != (3,) or not np.all(np.isfinite(bearing)):
                    raise ValueError(
                        "valid match bearings must be finite with shape (3,)"
                    )
                norm = float(np.linalg.norm(bearing))
                if norm <= 0.0:
                    raise ValueError("valid match bearings must be non-zero")
                bearing = bearing / norm
                previous = bearings.get(key)
                if previous is not None and not np.allclose(
                    previous, bearing, atol=1e-8, rtol=1e-8
                ):
                    raise ValueError("one feature index carries inconsistent bearings")
                bearings[key] = bearing
            distance = float(arrays[4][row])
            if not math.isfinite(distance):
                raise ValueError("valid descriptor distances must be finite")
            candidates.append(
                (
                    -int(edge.pose.num_inliers),
                    distance,
                    tuple(sorted(edge.pair)),
                    left,
                    right,
                )
            )
    candidates.sort()
    union = _UnionFind()
    conflicts = 0
    for _, _, _, left, right in candidates:
        union.add(left)
        union.add(right)
        if not union.union(left, right):
            conflicts += 1
    groups: dict[tuple[str, int], list[tuple[str, int]]] = {}
    for item in sorted(union.parent):
        groups.setdefault(union.find(item), []).append(item)
    ordered = sorted(
        (tuple(sorted(items)) for items in groups.values() if len(items) >= min_length)
    )
    tracks = [
        _Track(
            [
                _Observation(panorama, index, bearings[(panorama, index)])
                for panorama, index in items
            ]
        )
        for items in ordered
    ]
    return tracks, len(candidates), conflicts


def _initialize_centers(
    edges: Sequence[SphericalPairwisePoseEdge],
    rotations: dict[str, np.ndarray],
    reference: str,
    options: SphericalGlobalMapperOptions,
) -> tuple[dict[str, np.ndarray], tuple[float, float]]:
    from scipy.optimize import least_squares  # type: ignore[import-untyped]

    ids = sorted(rotations)
    centers = {reference: np.zeros(3)}
    while len(centers) < len(ids):
        candidates = [
            edge
            for edge in edges
            if (edge.panorama_id_a in centers) != (edge.panorama_id_b in centers)
        ]
        if not candidates:
            raise ValueError("translation graph is disconnected")
        edge = min(candidates, key=lambda item: (-item.pose.num_inliers, item.pair))
        a, b = edge.pair
        direction: np.ndarray = np.asarray(
            -rotations[b].T @ np.asarray(edge.pose.t, dtype=np.float64),
            dtype=np.float64,
        )
        direction = direction / np.linalg.norm(direction)
        if a in centers:
            centers[b] = centers[a] + direction
        else:
            centers[a] = centers[b] - direction

    variable_ids = [item for item in ids if item != reference]
    center_offset = {item: 3 * index for index, item in enumerate(variable_ids)}
    sorted_edges = sorted(edges, key=lambda item: tuple(sorted(item.pair)))
    anchor = sorted_edges[0]
    scale_edges = [edge for edge in sorted_edges if edge is not anchor]
    scale_offset = {
        id(edge): 3 * len(variable_ids) + index
        for index, edge in enumerate(scale_edges)
    }
    initial = np.concatenate(
        [
            *(centers[item] for item in variable_ids),
            np.zeros(len(scale_edges), dtype=np.float64),
        ]
    )
    median = float(np.median([edge.pose.num_inliers for edge in edges]))

    def unpack(parameters: np.ndarray) -> dict[str, np.ndarray]:
        result = {reference: np.zeros(3)}
        for item in variable_ids:
            start = center_offset[item]
            result[item] = parameters[start : start + 3]
        return result

    def residual(parameters: np.ndarray) -> np.ndarray:
        current = unpack(parameters)
        values: list[float] = []
        for edge in sorted_edges:
            a, b = edge.pair
            direction = np.asarray(
                -rotations[b].T @ np.asarray(edge.pose.t, dtype=np.float64),
                dtype=np.float64,
            )
            direction = direction / np.linalg.norm(direction)
            scale = (
                1.0 if edge is anchor else math.exp(parameters[scale_offset[id(edge)]])
            )
            values.extend(
                math.sqrt(_edge_weight(edge, median))
                * (current[b] - current[a] - scale * direction)
            )
        return np.asarray(values)

    before = float(np.mean(residual(initial) ** 2))
    solved = least_squares(
        residual,
        initial,
        loss="cauchy",
        f_scale=0.1,
        max_nfev=options.bundle_max_nfev,
    )
    return unpack(solved.x), (before, float(np.mean(residual(solved.x) ** 2)))


def _triangulate(
    observations: Sequence[_Observation], rotations, centers
) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=np.float64)
    rhs = np.zeros(3, dtype=np.float64)
    for observation in observations:
        ray = rotations[observation.panorama_id].T @ observation.bearing
        projector = np.eye(3) - np.outer(ray, ray)
        matrix += projector
        rhs += projector @ centers[observation.panorama_id]
    return np.linalg.lstsq(matrix, rhs, rcond=None)[0]


def _global_position(
    tracks: Sequence[_Track],
    rotations: dict[str, np.ndarray],
    centers: dict[str, np.ndarray],
    reference: str,
    options: SphericalGlobalMapperOptions,
) -> tuple[dict[str, np.ndarray], np.ndarray, tuple[float, float], str]:
    from scipy.optimize import least_squares  # type: ignore[import-untyped]
    from scipy.sparse import lil_matrix  # type: ignore[import-untyped]

    ids = sorted(rotations)
    variable_ids = [item for item in ids if item != reference]
    points = np.stack(
        [_triangulate(track.observations, rotations, centers) for track in tracks]
    )
    observations = [
        (track_index, observation)
        for track_index, track in enumerate(tracks)
        for observation in track.observations
    ]
    initial_depths = []
    for track_index, observation in observations:
        ray = rotations[observation.panorama_id].T @ observation.bearing
        depth = float(
            np.dot(points[track_index] - centers[observation.panorama_id], ray)
        )
        initial_depths.append(max(depth, 1e-3))
    anchor_depth = initial_depths[0]
    factor = 1.0 / anchor_depth
    centers = {key: value * factor for key, value in centers.items()}
    points = points * factor
    initial_depths = [value * factor for value in initial_depths]

    center_base = 0
    point_base = 3 * len(variable_ids)
    depth_base = point_base + 3 * len(tracks)
    center_offset = {item: center_base + 3 * i for i, item in enumerate(variable_ids)}
    depth_offset = {
        obs_index: depth_base + obs_index - 1
        for obs_index in range(1, len(observations))
    }
    parameters = np.empty(depth_base + len(observations) - 1, dtype=np.float64)
    for item in variable_ids:
        start = center_offset[item]
        parameters[start : start + 3] = centers[item]
    parameters[point_base:depth_base] = points.reshape(-1)
    for obs_index in range(1, len(observations)):
        parameters[depth_offset[obs_index]] = math.log(
            max(initial_depths[obs_index], 1e-8)
        )

    def unpack(values: np.ndarray):
        current_centers = {reference: np.zeros(3)}
        for item in variable_ids:
            start = center_offset[item]
            current_centers[item] = values[start : start + 3]
        current_points = values[point_base:depth_base].reshape((-1, 3))
        depths = np.ones(len(observations), dtype=np.float64)
        for obs_index in range(1, len(observations)):
            depths[obs_index] = math.exp(values[depth_offset[obs_index]])
        return current_centers, current_points, depths

    def residual(values: np.ndarray) -> np.ndarray:
        current_centers, current_points, depths = unpack(values)
        result = np.empty((len(observations), 3), dtype=np.float64)
        for obs_index, (track_index, observation) in enumerate(observations):
            ray = rotations[observation.panorama_id].T @ observation.bearing
            result[obs_index] = (
                current_points[track_index]
                - current_centers[observation.panorama_id]
                - depths[obs_index] * ray
            )
        return result.ravel()

    sparsity = lil_matrix((3 * len(observations), len(parameters)), dtype=int)
    for obs_index, (track_index, observation) in enumerate(observations):
        rows = slice(3 * obs_index, 3 * obs_index + 3)
        if observation.panorama_id != reference:
            start = center_offset[observation.panorama_id]
            sparsity[rows, start : start + 3] = 1
        start = point_base + 3 * track_index
        sparsity[rows, start : start + 3] = 1
        if obs_index:
            sparsity[rows, depth_offset[obs_index]] = 1
    before = float(np.mean(residual(parameters) ** 2))
    lower = np.full(len(parameters), -np.inf)
    upper = np.full(len(parameters), np.inf)
    if len(observations) > 1:
        lower[depth_base:] = -20.0
        upper[depth_base:] = 20.0
    solved = least_squares(
        residual,
        parameters,
        jac_sparsity=sparsity.tocsr(),
        bounds=(lower, upper),
        loss="cauchy",
        f_scale=0.05,
        max_nfev=options.bundle_max_nfev,
    )
    final_centers, final_points, _ = unpack(solved.x)
    anchor_observation = observations[0][1]
    anchor = f"{anchor_observation.panorama_id}:{anchor_observation.feature_index}"
    return (
        final_centers,
        final_points.copy(),
        (before, float(np.mean(residual(solved.x) ** 2))),
        anchor,
    )


def _ba_scale_anchor(
    centers: dict[str, np.ndarray],
    reference: str,
    edges: Sequence[SphericalPairwisePoseEdge],
) -> str:
    neighbours = []
    for edge in edges:
        if reference in edge.pair:
            other = (
                edge.panorama_id_b
                if edge.panorama_id_a == reference
                else edge.panorama_id_a
            )
            neighbours.append((-int(edge.pose.num_inliers), other))
    panorama_id = (
        min(neighbours)[1] if neighbours else sorted(set(centers) - {reference})[0]
    )
    axis = int(np.argmax(np.abs(centers[panorama_id])))
    return f"{panorama_id}:{axis}"


def _bundle_adjust(
    tracks: Sequence[_Track],
    active: Sequence[np.ndarray],
    rotations: dict[str, np.ndarray],
    centers: dict[str, np.ndarray],
    points: np.ndarray,
    reference: str,
    scale_anchor: str,
    options: SphericalGlobalMapperOptions,
    *,
    joint: bool,
) -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, tuple[float, float]
]:
    from scipy.optimize import least_squares  # type: ignore[import-untyped]
    from scipy.sparse import lil_matrix  # type: ignore[import-untyped]

    ids = sorted(rotations)
    variable_ids = [item for item in ids if item != reference]
    anchor_id, anchor_axis_text = scale_anchor.rsplit(":", 1)
    anchor_axis = int(anchor_axis_text)
    rotation_offset = (
        {item: 3 * index for index, item in enumerate(variable_ids)} if joint else {}
    )
    center_base = 3 * len(variable_ids) if joint else 0
    center_components = []
    for item in variable_ids:
        for axis in range(3):
            if not (item == anchor_id and axis == anchor_axis):
                center_components.append((item, axis))
    center_offset = {
        item: center_base + index for index, item in enumerate(center_components)
    }
    point_base = center_base + len(center_components)
    parameters = np.empty(point_base + 3 * len(tracks), dtype=np.float64)
    if joint:
        parameters[:center_base] = 0.0
    for item, axis in center_components:
        parameters[center_offset[(item, axis)]] = centers[item][axis]
    parameters[point_base:] = points.reshape(-1)
    observation_rows = [
        (track_index, obs_index, observation)
        for track_index, track in enumerate(tracks)
        for obs_index, observation in enumerate(track.observations)
        if active[track_index][obs_index]
    ]

    def unpack(values: np.ndarray):
        current_rotations = {reference: rotations[reference]}
        for item in variable_ids:
            if joint:
                start = rotation_offset[item]
                current_rotations[item] = (
                    rotation_exp(values[start : start + 3]) @ rotations[item]
                )
            else:
                current_rotations[item] = rotations[item]
        current_centers = {key: value.copy() for key, value in centers.items()}
        current_centers[reference] = np.zeros(3)
        for item, axis in center_components:
            current_centers[item][axis] = values[center_offset[(item, axis)]]
        current_points = values[point_base:].reshape((-1, 3))
        return current_rotations, current_centers, current_points

    def residual(values: np.ndarray) -> np.ndarray:
        current_rotations, current_centers, current_points = unpack(values)
        result = np.empty((len(observation_rows), 2), dtype=np.float64)
        for row, (track_index, _, observation) in enumerate(observation_rows):
            vector = current_rotations[observation.panorama_id] @ (
                current_points[track_index] - current_centers[observation.panorama_id]
            )
            norm = float(np.linalg.norm(vector))
            if norm <= 1e-12:
                result[row] = math.pi
            else:
                result[row] = spherical_log_residual(observation.bearing, vector / norm)
        return result.ravel()

    if not observation_rows:
        return rotations, centers, points, (math.inf, math.inf)
    sparsity = lil_matrix((2 * len(observation_rows), len(parameters)), dtype=int)
    for row, (track_index, _, observation) in enumerate(observation_rows):
        rows = slice(2 * row, 2 * row + 2)
        if joint and observation.panorama_id != reference:
            start = rotation_offset[observation.panorama_id]
            sparsity[rows, start : start + 3] = 1
        if observation.panorama_id != reference:
            for axis in range(3):
                key = (observation.panorama_id, axis)
                if key in center_offset:
                    sparsity[rows, center_offset[key]] = 1
        start = point_base + 3 * track_index
        sparsity[rows, start : start + 3] = 1
    initial_residuals = residual(parameters)
    before = _robust_least_squares_cost(
        initial_residuals,
        options.bundle_loss,
        math.radians(options.bundle_loss_scale_deg),
    )
    solved = least_squares(
        residual,
        parameters,
        jac_sparsity=sparsity.tocsr(),
        loss=options.bundle_loss,
        f_scale=math.radians(options.bundle_loss_scale_deg),
        max_nfev=options.bundle_max_nfev,
    )
    final_rotations, final_centers, final_points = unpack(solved.x)
    final_cost = _robust_least_squares_cost(
        residual(solved.x),
        options.bundle_loss,
        math.radians(options.bundle_loss_scale_deg),
    )
    return (
        final_rotations,
        final_centers,
        final_points.copy(),
        (before, final_cost),
    )


def _robust_least_squares_cost(
    residuals: np.ndarray,
    loss: str,
    scale: float,
) -> float:
    """Return SciPy's normalized robust objective for diagnostic reporting."""

    squared = np.square(np.asarray(residuals, dtype=np.float64) / scale)
    if loss == "linear":
        rho = squared
    elif loss == "soft_l1":
        rho = 2.0 * (np.sqrt(1.0 + squared) - 1.0)
    elif loss == "huber":
        rho = np.where(squared <= 1.0, squared, 2.0 * np.sqrt(squared) - 1.0)
    elif loss == "cauchy":
        rho = np.log1p(squared)
    else:  # validated by SphericalGlobalMapperOptions
        raise ValueError(f"unsupported bundle loss: {loss}")
    return float(scale**2 * np.mean(rho))


def _observation_error(observation, rotation, center, point) -> float:
    vector = rotation @ (point - center)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return math.inf
    return angular_error(observation.bearing, vector / norm)


def _track_angle(track, mask, rotations) -> float:
    rays = [
        rotations[observation.panorama_id].T @ observation.bearing
        for observation, keep in zip(track.observations, mask)
        if keep
    ]
    if len(rays) < 2:
        return 0.0
    return max(
        math.acos(float(np.clip(np.dot(first, second), -1.0, 1.0)))
        for index, first in enumerate(rays)
        for second in rays[index + 1 :]
    )


def _filter_observations(
    tracks,
    active,
    reasons,
    rotations,
    centers,
    points,
    options,
) -> bool:
    changed = False
    threshold = math.radians(options.max_reprojection_error_deg)
    min_angle = math.radians(options.min_triangulation_angle_deg)
    for track_index, track in enumerate(tracks):
        for obs_index, observation in enumerate(track.observations):
            if not active[track_index][obs_index]:
                continue
            error = _observation_error(
                observation,
                rotations[observation.panorama_id],
                centers[observation.panorama_id],
                points[track_index],
            )
            if error > threshold:
                active[track_index][obs_index] = False
                reasons[track_index][obs_index] = "reprojection-error"
                changed = True
        if active[track_index].sum() < options.min_track_length:
            for obs_index in np.flatnonzero(active[track_index]):
                active[track_index][obs_index] = False
                reasons[track_index][obs_index] = "track-too-short"
                changed = True
        elif _track_angle(track, active[track_index], rotations) < min_angle:
            for obs_index in np.flatnonzero(active[track_index]):
                active[track_index][obs_index] = False
                reasons[track_index][obs_index] = "low-triangulation-angle"
                changed = True
    return changed


def _retriangulate(tracks, active, rotations, centers, points):
    result = points.copy()
    for index, (track, mask) in enumerate(zip(tracks, active)):
        observations = [
            observation for observation, keep in zip(track.observations, mask) if keep
        ]
        if len(observations) >= 2:
            result[index] = _triangulate(observations, rotations, centers)
    return result


def _public_tracks(
    tracks, active, reasons, rotations, centers, points, options
) -> tuple[tuple[SphericalTrack, ...], np.ndarray, int]:
    public: list[SphericalTrack] = []
    selected_points: list[np.ndarray] = []
    filtered = 0
    for track_index, track in enumerate(tracks):
        if active[track_index].sum() < options.min_track_length:
            filtered += len(track.observations)
            continue
        observations: list[SphericalTrackObservation] = []
        for obs_index, observation in enumerate(track.observations):
            keep = bool(active[track_index][obs_index])
            residual = _observation_error(
                observation,
                rotations[observation.panorama_id],
                centers[observation.panorama_id],
                points[track_index],
            )
            if not keep:
                filtered += 1
            observations.append(
                SphericalTrackObservation(
                    panorama_id=observation.panorama_id,
                    feature_index=observation.feature_index,
                    bearing_xyz=observation.bearing,
                    residual_rad=residual,
                    active=keep,
                    rejection_reason=reasons[track_index][obs_index],
                )
            )
        track_id = f"track-{len(public):06d}"
        public.append(
            SphericalTrack(
                track_id=track_id,
                point_xyz=points[track_index],
                observations=tuple(observations),
            )
        )
        selected_points.append(points[track_index])
    array = (
        np.stack(selected_points)
        if selected_points
        else np.empty((0, 3), dtype=np.float64)
    )
    return tuple(public), array, filtered
