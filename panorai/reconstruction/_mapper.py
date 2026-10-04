"""Global spherical reconstruction from PanorAi pairwise evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Iterable, Sequence

import numpy as np

from ._native import (
    resolve_bundle_backend,
    spherical_ba_residual_jacobians,
)
from ._math import (
    angular_error,
    rotation_exp,
    rotation_log,
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


@dataclass(slots=True)
class _BearingPosition:
    centers: dict[str, np.ndarray]
    points: np.ndarray
    depths: np.ndarray
    costs: tuple[float, float]
    positive_depth_ratio: float
    min_camera_positive_depth_ratio: float
    anchors_tested: int
    scale_anchor: str


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
        self.bundle_compute_backend = resolve_bundle_backend(
            self.options.bundle_compute_backend
        )

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
        edges: (
            Sequence[SphericalPairwisePoseEdge]
            | Iterable[SphericalPairwisePoseEdge]
            | None
        ) = None,
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
            input_edge_count=(
                len(input_matches) if matches is not None else len(pairwise)
            ),
            successful_edge_count=len(pairwise),
            bundle_compute_backend=self.bundle_compute_backend,
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
        if reference_id is not None and reference_id not in all_ids:
            raise ValueError("reference_id must identify an input panorama")
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
            return self._failure(
                pairwise,
                admitted_indices,
                replace(
                    diagnostics,
                    excluded_panoramas=tuple(sorted(all_ids - component)),
                ),
                "reference-not-in-selected-component",
            )
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

        translation_filtered: list[tuple[str, str]] = []
        translation_flipped: list[tuple[str, str]] = []
        translation_errors: dict[tuple[str, str], float] = {}
        bearing_position: _BearingPosition | None = None
        direction_signs: dict[int, float] = {}
        for round_index in range(self.options.translation_consistency_rounds):
            bearing_position = _bearing_position_initialization(
                tracks, rotations, reference, self.options
            )
            if (
                bearing_position.positive_depth_ratio
                < self.options.translation_min_positive_depth_ratio
            ):
                diagnostics = replace(
                    diagnostics,
                    translation_positive_depth_ratio=(
                        bearing_position.positive_depth_ratio
                    ),
                    bearing_position_anchors_tested=(bearing_position.anchors_tested),
                    bearing_position_min_camera_positive_depth_ratio=(
                        bearing_position.min_camera_positive_depth_ratio
                    ),
                    scale_anchor=bearing_position.scale_anchor,
                )
                return self._failure(
                    pairwise,
                    admitted_indices,
                    diagnostics,
                    "insufficient-positive-depth-support",
                )
            direction_signs, current_errors, current_flipped = (
                _resolve_translation_orientations(
                    working, rotations, bearing_position.centers
                )
            )
            translation_errors.update(current_errors)
            translation_flipped.extend(current_flipped)
            threshold = self.options.translation_max_error_deg
            bad_pairs = {
                pair for pair, error in current_errors.items() if error > threshold
            }
            if not bad_pairs:
                break
            if round_index + 1 >= self.options.translation_consistency_rounds:
                translation_filtered.extend(sorted(bad_pairs))
                return self._failure(
                    pairwise,
                    admitted_indices,
                    replace(
                        diagnostics,
                        translation_filtered_pairs=tuple(
                            sorted(set(translation_filtered))
                        ),
                        translation_axis_errors_deg=tuple(
                            sorted(translation_errors.items())
                        ),
                        translation_positive_depth_ratio=(
                            bearing_position.positive_depth_ratio
                        ),
                        bearing_position_anchors_tested=(
                            bearing_position.anchors_tested
                        ),
                        bearing_position_min_camera_positive_depth_ratio=(
                            bearing_position.min_camera_positive_depth_ratio
                        ),
                    ),
                    "translation-consistency-not-converged",
                )
            bad_edges = [edge for edge in working if edge.pair in bad_pairs]
            worst = min(
                bad_edges,
                key=lambda edge: (
                    -current_errors[edge.pair],
                    int(edge.pose.num_inliers),
                    tuple(sorted(edge.pair)),
                ),
            )
            translation_filtered.append(worst.pair)
            kept = [edge for edge in working if edge is not worst]
            component = _largest_component(kept)
            if reference not in component:
                if reference_id is not None:
                    return self._failure(
                        pairwise,
                        admitted_indices,
                        replace(
                            diagnostics,
                            translation_filtered_pairs=tuple(
                                sorted(set(translation_filtered))
                            ),
                        ),
                        "reference-disconnected-after-translation-filtering",
                    )
                reference = min(component) if component else reference
            if len(component) < self.options.min_panoramas:
                return self._failure(
                    pairwise,
                    admitted_indices,
                    replace(
                        diagnostics,
                        translation_filtered_pairs=tuple(
                            sorted(set(translation_filtered))
                        ),
                    ),
                    "insufficient-panoramas-after-translation-filtering",
                )
            working = [edge for edge in kept if set(edge.pair) <= component]
            rotations, rotation_costs, _ = _average_rotations(
                working, reference, self.options
            )
            tracks, candidate_count, conflict_count = _build_tracks(
                working, component, self.options.min_track_length
            )
            diagnostics = replace(
                diagnostics,
                excluded_panoramas=tuple(sorted(all_ids - component)),
                candidate_match_count=candidate_count,
                track_count=len(tracks),
                track_conflict_count=conflict_count,
            )
            if not tracks:
                return self._failure(
                    pairwise,
                    admitted_indices,
                    diagnostics,
                    "no-tracks-after-translation-filtering",
                )

        assert bearing_position is not None
        diagnostics = replace(
            diagnostics,
            translation_filtered_pairs=tuple(sorted(set(translation_filtered))),
            translation_flipped_pairs=tuple(sorted(set(translation_flipped))),
            translation_axis_errors_deg=tuple(sorted(translation_errors.items())),
            translation_positive_depth_ratio=bearing_position.positive_depth_ratio,
            bearing_position_anchors_tested=bearing_position.anchors_tested,
            bearing_position_min_camera_positive_depth_ratio=(
                bearing_position.min_camera_positive_depth_ratio
            ),
        )
        centers, position_costs = _initialize_centers(
            working,
            rotations,
            reference,
            self.options,
            initial_centers=bearing_position.centers,
            direction_signs=direction_signs,
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
        final_diagnostics = _final_reprojection_diagnostics(public_tracks)
        if not public_tracks:
            return self._failure(
                pairwise,
                admitted_indices,
                replace(diagnostics, bundle_costs=tuple(bundle_costs)),
                "no-tracks-after-final-bundle-adjustment",
            )
        unsupported = _unsupported_panoramas(
            public_tracks,
            component,
            self.options.min_active_tracks_per_panorama,
        )
        if unsupported:
            return self._failure(
                pairwise,
                admitted_indices,
                replace(
                    diagnostics,
                    filtered_observation_count=filtered_count,
                    bundle_costs=tuple(bundle_costs),
                    **final_diagnostics,
                ),
                "panorama-without-active-track-support:" + ",".join(unsupported),
            )
        camera_poses = tuple(
            SphericalCameraPose(
                panorama_id, rotations[panorama_id], centers[panorama_id]
            )
            for panorama_id in sorted(component)
        )
        stage_messages = (
            "rotation-averaging",
            "conflict-free-track-union",
            "track-bearing-linear-initialization",
            "multiview-translation-orientation",
            "pairwise-translation-refinement",
            "bata-camera-point-positioning",
            "fixed-rotation-bundle-adjustment",
            "joint-spherical-bundle-adjustment",
            "filter-retriangulate-final-refinement",
        )
        diagnostics = replace(
            diagnostics,
            admitted_edge_count=len(working),
            track_count=len(public_tracks),
            filtered_observation_count=filtered_count,
            bundle_costs=tuple(bundle_costs),
            ba_scale_anchor=ba_anchor,
            **final_diagnostics,
            stage_messages=stage_messages,
        )
        if (
            self.options.require_multiview_corroboration
            and self.options.min_track_length
            < self.options.multiview_corroboration_min_track_length
        ):
            corroboration_options = replace(
                self.options,
                min_track_length=(
                    self.options.multiview_corroboration_min_track_length
                ),
                require_multiview_corroboration=False,
            )
            corroboration = SphericalGlobalMapper(
                relative_pose_estimator=self.relative_pose_estimator,
                options=corroboration_options,
            ).reconstruct(
                edges=tuple(working),
                panorama_ids=tuple(sorted(component)),
                reference_id=reference,
            )
            if not corroboration.success:
                diagnostics = replace(
                    diagnostics,
                    multiview_corroboration_passed=False,
                    multiview_corroboration_failure_reasons=(
                        corroboration.failure_reasons
                    ),
                )
                return self._failure(
                    pairwise,
                    admitted_indices,
                    diagnostics,
                    "multiview-corroboration-failed",
                    *corroboration.failure_reasons,
                )
            primary_panorama_ids = {pose.panorama_id for pose in camera_poses}
            corroboration_panorama_ids = {
                pose.panorama_id for pose in corroboration.poses
            }
            if primary_panorama_ids != corroboration_panorama_ids:
                reason = "corroborating-reconstruction-panorama-set-mismatch"
                diagnostics = replace(
                    diagnostics,
                    multiview_corroboration_passed=False,
                    multiview_corroboration_track_count=len(corroboration.tracks),
                    multiview_corroboration_failure_reasons=(reason,),
                )
                return self._failure(
                    pairwise,
                    admitted_indices,
                    diagnostics,
                    "multiview-corroboration-failed",
                    reason,
                )
            position_errors = _position_direction_disagreements(
                camera_poses, corroboration.poses
            )
            position_p90 = float(np.quantile(position_errors, 0.9))
            if (
                position_p90
                > self.options.multiview_corroboration_max_position_error_deg
            ):
                diagnostics = replace(
                    diagnostics,
                    multiview_corroboration_passed=False,
                    multiview_corroboration_track_count=len(corroboration.tracks),
                    multiview_corroboration_position_p90_deg=position_p90,
                    multiview_corroboration_failure_reasons=(
                        "position-direction-disagreement",
                    ),
                )
                return self._failure(
                    pairwise,
                    admitted_indices,
                    diagnostics,
                    "multiview-corroboration-position-disagreement",
                )
            diagnostics = replace(
                diagnostics,
                multiview_corroboration_passed=True,
                multiview_corroboration_track_count=len(corroboration.tracks),
                multiview_corroboration_position_p90_deg=position_p90,
                stage_messages=(*stage_messages, "three-view-track-corroboration"),
            )
        final_admitted = tuple(
            index for index in admitted_indices if pairwise[index] in working
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


def _bearing_position_initialization(
    tracks: Sequence[_Track],
    rotations: dict[str, np.ndarray],
    reference: str,
    options: SphericalGlobalMapperOptions,
) -> _BearingPosition:
    """Solve the BATA incidence equations without pairwise translations.

    Several geometrically strong observations are tried as positive unit-depth
    scale anchors.  This removes the former dependency on whichever track was
    ordered first.  A short Cauchy IRLS pass limits inconsistent observations;
    the selected solution maximizes per-camera and global positive-depth
    support before considering its scale-normalized residual.
    """

    from scipy.sparse import coo_matrix  # type: ignore[import-untyped]
    from scipy.sparse.linalg import lsqr  # type: ignore[import-untyped]

    ids = sorted(rotations)
    variable_ids = [item for item in ids if item != reference]
    observations = [
        (track_index, observation)
        for track_index, track in enumerate(tracks)
        for observation in track.observations
    ]
    if not observations:
        raise ValueError("bearing position requires at least one observation")
    center_offset = {item: 3 * index for index, item in enumerate(variable_ids)}
    point_base = 3 * len(variable_ids)
    depth_base = point_base + 3 * len(tracks)
    camera_observations: dict[str, list[int]] = {item: [] for item in ids}
    track_spreads: list[float] = []
    observation_offset = 0
    for track in tracks:
        rays = [
            rotations[item.panorama_id].T @ item.bearing for item in track.observations
        ]
        spread = max(
            (
                math.acos(float(np.clip(np.dot(first, second), -1.0, 1.0)))
                for index, first in enumerate(rays)
                for second in rays[index + 1 :]
            ),
            default=0.0,
        )
        track_spreads.append(spread)
        for local_index, observation in enumerate(track.observations):
            camera_observations[observation.panorama_id].append(
                observation_offset + local_index
            )
        observation_offset += len(track.observations)

    track_offsets = np.cumsum(
        np.asarray((0, *(len(track.observations) for track in tracks)), dtype=np.int64)
    )
    ranked_tracks = sorted(
        range(len(tracks)),
        key=lambda index: (
            -len(tracks[index].observations),
            -track_spreads[index],
            tuple(
                (observation.panorama_id, observation.feature_index)
                for observation in tracks[index].observations
            ),
        ),
    )
    anchor_candidates: list[int] = []
    for track_index in ranked_tracks:
        track = tracks[track_index]
        local_index = next(
            (
                index
                for index, observation in enumerate(track.observations)
                if observation.panorama_id == reference
            ),
            0,
        )
        anchor_candidates.append(int(track_offsets[track_index] + local_index))
        if len(anchor_candidates) >= options.bearing_position_anchor_trials:
            break

    candidates: list[tuple[tuple[float, float, float, str], _BearingPosition]] = []
    for anchor_index in anchor_candidates:
        depth_offset: dict[int, int] = {}
        next_offset = depth_base
        for obs_index in range(len(observations)):
            if obs_index != anchor_index:
                depth_offset[obs_index] = next_offset
                next_offset += 1
        parameter_count = next_offset
        rows: list[int] = []
        columns: list[int] = []
        data: list[float] = []
        rhs = np.zeros(3 * len(observations), dtype=np.float64)
        for obs_index, (track_index, observation) in enumerate(observations):
            ray = rotations[observation.panorama_id].T @ observation.bearing
            ray = ray / np.linalg.norm(ray)
            for axis in range(3):
                row = 3 * obs_index + axis
                rows.append(row)
                columns.append(point_base + 3 * track_index + axis)
                data.append(1.0)
                if observation.panorama_id != reference:
                    rows.append(row)
                    columns.append(center_offset[observation.panorama_id] + axis)
                    data.append(-1.0)
                if obs_index == anchor_index:
                    rhs[row] = ray[axis]
                else:
                    rows.append(row)
                    columns.append(depth_offset[obs_index])
                    data.append(-float(ray[axis]))
        matrix = coo_matrix(
            (data, (rows, columns)),
            shape=(3 * len(observations), parameter_count),
            dtype=np.float64,
        ).tocsr()
        observation_weights = np.ones(len(observations), dtype=np.float64)
        solution = np.zeros(parameter_count, dtype=np.float64)
        initial_cost = math.inf
        final_cost = math.inf
        residual_norms = np.full(len(observations), math.inf)
        for iteration in range(options.bearing_position_irls_steps):
            row_weights = np.repeat(np.sqrt(observation_weights), 3)
            weighted_matrix = matrix.multiply(row_weights[:, None])
            weighted_rhs = rhs * row_weights
            solution = lsqr(
                weighted_matrix,
                weighted_rhs,
                atol=1e-11,
                btol=1e-11,
                iter_lim=max(100, 5 * parameter_count),
            )[0]
            residual_vectors = (matrix @ solution - rhs).reshape((-1, 3))
            residual_norms = np.linalg.norm(residual_vectors, axis=1)
            cost = float(np.mean(residual_norms**2))
            if iteration == 0:
                initial_cost = cost
            final_cost = cost
            scale = max(
                1e-8,
                1.4826
                * float(np.median(np.abs(residual_norms - np.median(residual_norms)))),
                float(np.median(residual_norms)),
            )
            observation_weights = 1.0 / (1.0 + (residual_norms / (2.5 * scale)) ** 2)
        if not np.all(np.isfinite(solution)):
            continue
        centers = {reference: np.zeros(3, dtype=np.float64)}
        for panorama_id in variable_ids:
            start = center_offset[panorama_id]
            centers[panorama_id] = solution[start : start + 3].copy()
        points = solution[point_base:depth_base].reshape((-1, 3)).copy()
        depths = np.ones(len(observations), dtype=np.float64)
        for obs_index, offset in depth_offset.items():
            depths[obs_index] = solution[offset]
        positive = depths > 1e-8
        positive_ratio = float(np.mean(positive))
        camera_ratios = [
            float(np.mean(positive[indices]))
            for indices in camera_observations.values()
            if indices
        ]
        minimum_camera_ratio = min(camera_ratios, default=0.0)
        scene_scale = max(float(np.median(np.abs(depths))), 1e-12)
        normalized_cost = float(np.mean((residual_norms / scene_scale) ** 2))
        anchor_observation = observations[anchor_index][1]
        anchor_name = (
            f"{anchor_observation.panorama_id}:{anchor_observation.feature_index}"
        )
        value = _BearingPosition(
            centers=centers,
            points=points,
            depths=depths,
            costs=(initial_cost, final_cost),
            positive_depth_ratio=positive_ratio,
            min_camera_positive_depth_ratio=minimum_camera_ratio,
            anchors_tested=len(anchor_candidates),
            scale_anchor=anchor_name,
        )
        score = (
            -minimum_camera_ratio,
            -positive_ratio,
            normalized_cost,
            anchor_name,
        )
        candidates.append((score, value))
    if not candidates:
        raise ValueError("bearing position produced no finite anchor solution")
    return min(candidates, key=lambda item: item[0])[1]


def _resolve_translation_orientations(
    edges: Sequence[SphericalPairwisePoseEdge],
    rotations: dict[str, np.ndarray],
    centers: dict[str, np.ndarray],
) -> tuple[
    dict[int, float],
    dict[tuple[str, str], float],
    list[tuple[str, str]],
]:
    signs: dict[int, float] = {}
    errors: dict[tuple[str, str], float] = {}
    flipped: list[tuple[str, str]] = []
    for edge in edges:
        a, b = edge.pair
        displacement = centers[b] - centers[a]
        displacement_norm = float(np.linalg.norm(displacement))
        if displacement_norm <= 1e-10:
            errors[edge.pair] = 90.0
            signs[id(edge)] = 1.0
            continue
        displacement = np.asarray(displacement / displacement_norm, dtype=np.float64)
        direction = np.asarray(
            -rotations[b].T @ np.asarray(edge.pose.t, dtype=np.float64),
            dtype=np.float64,
        )
        direction = direction / float(np.linalg.norm(direction))
        cosine = float(np.clip(np.dot(displacement, direction), -1.0, 1.0))
        sign = 1.0 if cosine >= 0.0 else -1.0
        signs[id(edge)] = sign
        if sign < 0.0:
            flipped.append(edge.pair)
        errors[edge.pair] = math.degrees(math.acos(abs(cosine)))
    return signs, errors, flipped


def _initialize_centers(
    edges: Sequence[SphericalPairwisePoseEdge],
    rotations: dict[str, np.ndarray],
    reference: str,
    options: SphericalGlobalMapperOptions,
    *,
    initial_centers: dict[str, np.ndarray] | None = None,
    direction_signs: dict[int, float] | None = None,
) -> tuple[dict[str, np.ndarray], tuple[float, float]]:
    from scipy.optimize import least_squares  # type: ignore[import-untyped]

    ids = sorted(rotations)
    centers = (
        {
            key: np.array(value, dtype=np.float64, copy=True)
            for key, value in initial_centers.items()
        }
        if initial_centers is not None
        else {reference: np.zeros(3)}
    )
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
        direction = (
            (direction_signs or {}).get(id(edge), 1.0)
            * direction
            / np.linalg.norm(direction)
        )
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
            direction = (
                (direction_signs or {}).get(id(edge), 1.0)
                * direction
                / np.linalg.norm(direction)
            )
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
    camera_index = {item: index for index, item in enumerate(ids)}
    observation_track_indices = np.fromiter(
        (track_index for track_index, _ in observations), dtype=np.int64
    )
    observation_camera_indices = np.fromiter(
        (camera_index[observation.panorama_id] for _, observation in observations),
        dtype=np.int64,
    )
    world_rays = np.stack(
        [
            rotations[observation.panorama_id].T @ observation.bearing
            for _, observation in observations
        ]
    )
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
        depths[1:] = np.exp(values[depth_base:])
        return current_centers, current_points, depths

    def residual(values: np.ndarray) -> np.ndarray:
        current_centers, current_points, depths = unpack(values)
        center_values = np.stack([current_centers[item] for item in ids])
        result = (
            current_points[observation_track_indices]
            - center_values[observation_camera_indices]
            - depths[:, None] * world_rays
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


def _spherical_log_residual_batch(
    measured: np.ndarray, predicted: np.ndarray
) -> np.ndarray:
    """Vectorized equivalent of :func:`spherical_log_residual` for unit rows."""

    measured = np.asarray(measured, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    cosine = np.clip(np.sum(measured * predicted, axis=1), -1.0, 1.0)
    angle = np.arccos(cosine)
    tangent = predicted - cosine[:, None] * measured
    tangent_norm = np.linalg.norm(tangent, axis=1)

    axes = np.zeros_like(measured)
    axes[np.arange(len(measured)), np.argmin(np.abs(measured), axis=1)] = 1.0
    first = np.cross(measured, axes)
    first /= np.linalg.norm(first, axis=1, keepdims=True)
    second = np.cross(measured, first)

    vectors = np.empty_like(tangent)
    regular = tangent_norm >= 1e-12
    vectors[regular] = (
        angle[regular, None] / tangent_norm[regular, None] * tangent[regular]
    )
    coincident = ~regular & (angle < 1e-8)
    vectors[coincident] = tangent[coincident]
    antipodal = ~regular & ~coincident
    vectors[antipodal] = angle[antipodal, None] * first[antipodal]
    return np.stack(
        (np.sum(first * vectors, axis=1), np.sum(second * vectors, axis=1)),
        axis=1,
    )


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
    from scipy.sparse import csr_matrix, lil_matrix  # type: ignore[import-untyped]

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
    camera_index = {item: index for index, item in enumerate(ids)}
    observation_track_indices = np.fromiter(
        (track_index for track_index, _, _ in observation_rows), dtype=np.int64
    )
    observation_camera_indices = np.fromiter(
        (
            camera_index[observation.panorama_id]
            for _, _, observation in observation_rows
        ),
        dtype=np.int64,
    )
    measured_bearings = np.stack(
        [observation.bearing for _, _, observation in observation_rows]
    )
    resolved_backend = resolve_bundle_backend(options.bundle_compute_backend)

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
        rotation_values = np.stack([current_rotations[item] for item in ids])
        center_values = np.stack([current_centers[item] for item in ids])
        deltas = (
            current_points[observation_track_indices]
            - center_values[observation_camera_indices]
        )
        vectors = np.einsum(
            "nij,nj->ni", rotation_values[observation_camera_indices], deltas
        )
        norms = np.linalg.norm(vectors, axis=1)
        result = np.full((len(observation_rows), 2), math.pi, dtype=np.float64)
        valid = norms > 1e-12
        if np.any(valid):
            result[valid] = _spherical_log_residual_batch(
                measured_bearings[valid], vectors[valid] / norms[valid, None]
            )
        return result.ravel()

    native_cache: dict[str, Any] = {}

    def native_evaluation(values: np.ndarray):
        cached_values = native_cache.get("values")
        if cached_values is not None and np.array_equal(values, cached_values):
            return native_cache["result"]
        current_rotations, current_centers, current_points = unpack(values)
        rotation_values = np.stack([current_rotations[item] for item in ids])
        center_values = np.stack([current_centers[item] for item in ids])
        rotation_deltas = np.zeros((len(ids), 3), dtype=np.float64)
        if joint:
            for item in variable_ids:
                start = rotation_offset[item]
                rotation_deltas[camera_index[item]] = values[start : start + 3]
        result = spherical_ba_residual_jacobians(
            measured_bearings,
            rotation_values,
            center_values,
            current_points,
            observation_camera_indices,
            observation_track_indices,
            rotation_deltas,
        )
        native_cache["values"] = np.array(values, copy=True)
        native_cache["result"] = result
        return result

    def native_residual(values: np.ndarray) -> np.ndarray:
        # SciPy scales the returned residual array in place for robust losses.
        # Keep the cached native result immutable across the paired fun/jac calls.
        return native_evaluation(values)[0].ravel().copy()

    def native_jacobian(values: np.ndarray):
        _, rotation_blocks, center_blocks, point_blocks = native_evaluation(values)
        data_parts: list[np.ndarray] = []
        row_parts: list[np.ndarray] = []
        column_parts: list[np.ndarray] = []
        observation_indices = np.arange(len(observation_rows), dtype=np.int64)
        block_rows = (
            2 * observation_indices[:, None, None]
            + np.arange(2, dtype=np.int64)[None, :, None]
        )
        if joint:
            variable_mask = observation_camera_indices != camera_index[reference]
            selected = observation_indices[variable_mask]
            if len(selected):
                row_parts.append(
                    np.broadcast_to(
                        block_rows[variable_mask], (len(selected), 2, 3)
                    ).ravel()
                )
                starts = np.asarray(
                    [
                        rotation_offset[ids[index]]
                        for index in observation_camera_indices[variable_mask]
                    ],
                    dtype=np.int64,
                )
                columns = (
                    starts[:, None, None] + np.arange(3, dtype=np.int64)[None, None, :]
                )
                column_parts.append(
                    np.broadcast_to(columns, (len(selected), 2, 3)).ravel()
                )
                data_parts.append(rotation_blocks[variable_mask].ravel())
        for axis in range(3):
            selected_mask = np.asarray(
                [
                    (ids[index], axis) in center_offset
                    for index in observation_camera_indices
                ],
                dtype=bool,
            )
            selected = observation_indices[selected_mask]
            if not len(selected):
                continue
            row_parts.append(
                (2 * selected[:, None] + np.arange(2, dtype=np.int64)[None, :]).ravel()
            )
            column_parts.append(
                np.repeat(
                    np.asarray(
                        [
                            center_offset[(ids[index], axis)]
                            for index in observation_camera_indices[selected_mask]
                        ],
                        dtype=np.int64,
                    ),
                    2,
                )
            )
            data_parts.append(center_blocks[selected_mask, :, axis].ravel())
        row_parts.append(
            np.broadcast_to(block_rows, (len(observation_rows), 2, 3)).ravel()
        )
        point_columns = (
            point_base
            + 3 * observation_track_indices[:, None, None]
            + np.arange(3, dtype=np.int64)[None, None, :]
        )
        column_parts.append(
            np.broadcast_to(point_columns, (len(observation_rows), 2, 3)).ravel()
        )
        data_parts.append(point_blocks.ravel())
        return csr_matrix(
            (
                np.concatenate(data_parts),
                (np.concatenate(row_parts), np.concatenate(column_parts)),
            ),
            shape=(2 * len(observation_rows), len(parameters)),
        )

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
    objective = native_residual if resolved_backend == "native" else residual
    initial_residuals = objective(parameters)
    before = _robust_least_squares_cost(
        initial_residuals,
        options.bundle_loss,
        math.radians(options.bundle_loss_scale_deg),
    )
    solver_arguments: dict[str, Any] = {
        "loss": options.bundle_loss,
        "f_scale": math.radians(options.bundle_loss_scale_deg),
        "max_nfev": options.bundle_max_nfev,
    }
    if resolved_backend == "native":
        solver_arguments["jac"] = native_jacobian
    else:
        solver_arguments["jac_sparsity"] = sparsity.tocsr()
    solved = least_squares(objective, parameters, **solver_arguments)
    final_rotations, final_centers, final_points = unpack(solved.x)
    final_cost = _robust_least_squares_cost(
        objective(solved.x),
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


def _final_reprojection_diagnostics(
    tracks: Sequence[SphericalTrack],
) -> dict[str, Any]:
    residuals: list[float] = []
    by_camera: dict[str, list[float]] = {}
    active_tracks: dict[str, set[str]] = {}
    for track in tracks:
        for observation in track.observations:
            if not observation.active:
                continue
            value = math.degrees(float(observation.residual_rad))
            residuals.append(value)
            by_camera.setdefault(observation.panorama_id, []).append(value)
            active_tracks.setdefault(observation.panorama_id, set()).add(track.track_id)
    if not residuals:
        return {
            "reprojection_median_deg": None,
            "reprojection_p90_deg": None,
            "reprojection_max_deg": None,
            "camera_max_reprojection_deg": (),
            "camera_active_track_counts": (),
        }
    values = np.asarray(residuals, dtype=np.float64)
    return {
        "reprojection_median_deg": float(np.median(values)),
        "reprojection_p90_deg": float(np.quantile(values, 0.9)),
        "reprojection_max_deg": float(np.max(values)),
        "camera_max_reprojection_deg": tuple(
            (panorama_id, max(camera_values))
            for panorama_id, camera_values in sorted(by_camera.items())
        ),
        "camera_active_track_counts": tuple(
            (panorama_id, len(track_ids))
            for panorama_id, track_ids in sorted(active_tracks.items())
        ),
    }


def _unsupported_panoramas(
    tracks: Sequence[SphericalTrack],
    panorama_ids: set[str],
    minimum: int,
) -> tuple[str, ...]:
    counts = {panorama_id: 0 for panorama_id in panorama_ids}
    for track in tracks:
        seen = {
            observation.panorama_id
            for observation in track.observations
            if observation.active
        }
        for panorama_id in seen:
            if panorama_id in counts:
                counts[panorama_id] += 1
    return tuple(
        panorama_id for panorama_id, count in sorted(counts.items()) if count < minimum
    )


def _position_direction_disagreements(
    primary: Sequence[SphericalCameraPose],
    corroboration: Sequence[SphericalCameraPose],
) -> np.ndarray:
    """Return gauge-invariant pairwise center-direction disagreement in degrees."""

    primary_centers = {pose.panorama_id: pose.center for pose in primary}
    corroboration_centers = {pose.panorama_id: pose.center for pose in corroboration}
    if set(primary_centers) != set(corroboration_centers):
        raise ValueError("corroborating reconstruction must contain the same panoramas")
    panorama_ids = sorted(primary_centers)
    errors: list[float] = []
    for index, panorama_a in enumerate(panorama_ids):
        for panorama_b in panorama_ids[index + 1 :]:
            primary_direction = (
                primary_centers[panorama_b] - primary_centers[panorama_a]
            )
            corroboration_direction = (
                corroboration_centers[panorama_b] - corroboration_centers[panorama_a]
            )
            primary_norm = float(np.linalg.norm(primary_direction))
            corroboration_norm = float(np.linalg.norm(corroboration_direction))
            if primary_norm <= 1e-12 or corroboration_norm <= 1e-12:
                errors.append(180.0)
                continue
            cosine = float(
                np.clip(
                    np.dot(primary_direction, corroboration_direction)
                    / (primary_norm * corroboration_norm),
                    -1.0,
                    1.0,
                )
            )
            errors.append(math.degrees(math.acos(cosine)))
    if not errors:
        raise ValueError("corroboration requires at least two panorama centers")
    return np.asarray(errors, dtype=np.float64)
