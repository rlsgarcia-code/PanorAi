"""Incremental central-ERP tracking, local mapping and spherical BA."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import math
from typing import Any, Iterable

import numpy as np

from panorai.features import SphericalFeatureMatches, SphericalFeaturePipeline
from panorai.features._models import SphericalFeatureSet
from panorai.reconstruction import (
    SphericalGlobalMapper,
    SphericalPairwisePoseEdge,
)

from ._local_map import (
    LocalBundleAdjuster,
    LocalSphericalMap,
    ObservationState,
    PoseState,
    observation_error_deg,
    refine_absolute_pose,
    triangulate_bearings,
)
from ._models import (
    SphericalIncrementalSLAMDiagnostics,
    SphericalIncrementalSLAMOptions,
    SphericalIncrementalSLAMResult,
    SphericalKeyframeSummary,
    SphericalSLAMPose,
    SphericalTrackingResult,
)


@dataclass(slots=True)
class _KeyframeRecord:
    features: SphericalFeatureSet
    pose: PoseState


@dataclass(slots=True)
class _FrameRecord:
    frame_id: str
    timestamp_s: float
    state: str
    pose: PoseState | None
    reference_frame_id: str | None
    is_keyframe: bool
    feature_count: int
    match_count: int
    inlier_count: int
    map_correspondence_count: int
    median_parallax_deg: float | None
    local_map_point_count: int
    reasons: tuple[str, ...]
    anchor_frame_id: str | None = None
    rotation_frame_from_anchor: np.ndarray | None = None
    center_delta_in_anchor: np.ndarray | None = None


@dataclass(slots=True)
class _PairEvaluation:
    reference: _KeyframeRecord
    matches: SphericalFeatureMatches
    relative_pose: Any | None

    @property
    def admitted(self) -> bool:
        return self.relative_pose is not None


class SphericalIncrementalSLAM:
    """Experimental online visual SLAM for central ERP images.

    ``add_frame`` returns a pose/status immediately. Descriptor tracking,
    landmark association, keyframe selection and local BA are incremental.
    Global mapping is used only for accepted loop correction and final graph
    refinement; it is not the mechanism that creates online poses.
    """

    def __init__(
        self,
        *,
        feature_pipeline: SphericalFeaturePipeline | None = None,
        relative_pose_estimator: Any | None = None,
        options: SphericalIncrementalSLAMOptions | None = None,
    ) -> None:
        if feature_pipeline is None:
            feature_pipeline = SphericalFeaturePipeline.from_preset(
                "sift-flann",
                face_sampler="cube",
                face_fov_deg=100.0,
                face_shape_hw=(320, 320),
                edge_margin_px=8,
                max_features=2048,
            )
        if not isinstance(feature_pipeline, SphericalFeaturePipeline):
            raise TypeError("feature_pipeline must be SphericalFeaturePipeline")
        if relative_pose_estimator is None:
            from panorai.estimators import SphericalRelativePoseEstimator

            relative_pose_estimator = SphericalRelativePoseEstimator()
        if not hasattr(relative_pose_estimator, "estimate"):
            raise TypeError("relative_pose_estimator must provide estimate()")
        self.feature_pipeline = feature_pipeline
        self.relative_pose_estimator = relative_pose_estimator
        self.options = options or SphericalIncrementalSLAMOptions()
        mapper_options = replace(
            self.options.mapper,
            edge_admission=self.options.edge_admission,
        )
        self.global_mapper = SphericalGlobalMapper(
            relative_pose_estimator=relative_pose_estimator,
            options=mapper_options,
        )
        self.local_ba = LocalBundleAdjuster(self.options)
        self.reset()

    def reset(self) -> None:
        """Clear all trajectory and map state while retaining configuration."""

        self._input_count = 0
        self._seen_ids: set[str] = set()
        self._last_timestamp: float | None = None
        self._keyframes: list[_KeyframeRecord] = []
        self._poses: dict[str, PoseState] = {}
        self._frames: list[_FrameRecord] = []
        self._map = LocalSphericalMap()
        self._edges: dict[tuple[str, str], SphericalPairwisePoseEdge] = {}
        self._ba_reports = []
        self._relocalizations = 0
        self._loop_closures = 0
        self._messages: list[str] = []
        self._last_global = None
        self._finished = False

    @classmethod
    def from_preset(
        cls,
        preset: str = "sift-flann",
        *,
        face_sampler: str = "cube",
        face_fov_deg: float = 100.0,
        face_shape_hw: int | tuple[int, int] = (320, 320),
        max_features: int = 2048,
        options: SphericalIncrementalSLAMOptions | None = None,
        relative_pose_estimator: Any | None = None,
    ) -> "SphericalIncrementalSLAM":
        pipeline = SphericalFeaturePipeline.from_preset(
            preset,
            face_sampler=face_sampler,
            face_fov_deg=face_fov_deg,
            face_shape_hw=face_shape_hw,
            edge_margin_px=8,
            max_features=max_features,
        )
        return cls(
            feature_pipeline=pipeline,
            relative_pose_estimator=relative_pose_estimator,
            options=options,
        )

    def add_frame(
        self,
        panorama: Any,
        *,
        timestamp_s: float,
        frame_id: str | None = None,
        validity_mask: Any | None = None,
    ) -> SphericalTrackingResult:
        """Extract spherical features and increment the map by one ERP frame."""

        identity = frame_id or f"frame-{self._input_count:06d}"
        self._validate_input_identity(identity, timestamp_s)
        features = self.feature_pipeline.extract(
            panorama,
            panorama_id=identity,
            validity_mask=validity_mask,
        )
        return self.add_features(features, timestamp_s=timestamp_s)

    def add_features(
        self,
        features: SphericalFeatureSet,
        *,
        timestamp_s: float,
    ) -> SphericalTrackingResult:
        """Advanced incremental route for already extracted spherical features."""

        if self._finished:
            raise RuntimeError("the session is finished; call reset() before reuse")
        if not isinstance(features, SphericalFeatureSet):
            raise TypeError("features must be SphericalFeatureSet")
        identity = features.panorama_id
        timestamp = self._validate_input_identity(identity, timestamp_s)
        owned_features = copy.deepcopy(features)
        self._input_count += 1
        self._seen_ids.add(identity)
        self._last_timestamp = timestamp
        if len(owned_features) < self.options.min_frame_features:
            return self._record_lost(
                identity,
                timestamp,
                len(owned_features),
                reasons=("insufficient-features",),
            )

        if not self._keyframes:
            pose = PoseState(identity, timestamp, np.eye(3), np.zeros(3))
            self._poses[identity] = pose
            self._keyframes.append(_KeyframeRecord(owned_features, pose))
            record = _FrameRecord(
                frame_id=identity,
                timestamp_s=timestamp,
                state="initializing",
                pose=pose,
                reference_frame_id=None,
                is_keyframe=True,
                feature_count=len(owned_features),
                match_count=0,
                inlier_count=0,
                map_correspondence_count=0,
                median_parallax_deg=None,
                local_map_point_count=self._map.active_point_count,
                reasons=("world-anchor-created",),
            )
            self._frames.append(record)
            return self._snapshot_frame(record)

        local_references = self._keyframes[-self.options.local_window_size :]
        evaluations = self._evaluate_references(local_references, owned_features)
        admitted = [item for item in evaluations if self._admitted(item.relative_pose)]
        relocalized = False
        if not admitted:
            excluded = {item.pose.frame_id for item in local_references}
            relocalization = self._find_relocalization(owned_features, excluded)
            if relocalization is not None:
                evaluations.append(relocalization)
                admitted = [relocalization]
                relocalized = True
        if not admitted:
            match_count = max((len(item.matches) for item in evaluations), default=0)
            return self._record_lost(
                identity,
                timestamp,
                len(owned_features),
                match_count=match_count,
                reasons=("no-admitted-relative-pose",),
            )

        best = max(
            admitted,
            key=lambda item: (
                int(item.relative_pose.num_inliers),
                len(item.matches),
                item.reference.pose.frame_id,
            ),
        )
        initial_pose = self._compose_relative_pose(best, identity, timestamp)
        correspondences = self._map_correspondences(evaluations)
        tracked_pose = initial_pose
        pose_reasons: list[str] = []
        if len(correspondences) >= self.options.min_map_correspondences:
            refined = self._refine_from_local_map(
                initial_pose,
                best,
                correspondences,
            )
            if (
                refined.success
                and refined.p90_error_deg <= self.options.max_reprojection_error_deg
            ):
                tracked_pose = PoseState(
                    identity,
                    timestamp,
                    refined.rotation,
                    refined.center,
                )
                pose_reasons.append("spherical-2d3d-refined")
            elif len(self._keyframes) > 1:
                return self._record_lost(
                    identity,
                    timestamp,
                    len(owned_features),
                    match_count=len(best.matches),
                    inlier_count=int(best.relative_pose.num_inliers),
                    map_correspondence_count=len(correspondences),
                    reasons=("map-pose-refinement-failed",),
                )
        elif len(self._keyframes) > 1:
            pose_reasons.append("relative-pose-fallback")

        median_parallax = self._median_parallax(best, tracked_pose)
        tracked_ratio = len(correspondences) / max(
            1, int(best.relative_pose.num_inliers)
        )
        make_keyframe = self._should_create_keyframe(
            timestamp,
            median_parallax,
            tracked_ratio,
        )
        self._poses[identity] = tracked_pose
        if relocalized:
            self._relocalizations += 1
            make_keyframe = True
            state = "relocalized"
            pose_reasons.append("global-keyframe-relocalization")
        else:
            state = "keyframe" if make_keyframe else "tracking"

        if make_keyframe:
            current = _KeyframeRecord(owned_features, tracked_pose)
            self._keyframes.append(current)
            for evaluation in evaluations:
                if self._admitted(evaluation.relative_pose):
                    self._store_edge(evaluation)
                    self._integrate_pair(evaluation, current)
            report = self.local_ba.optimize(
                self._poses,
                self._map,
                [
                    item.pose.frame_id
                    for item in self._keyframes[-self.options.local_window_size :]
                ],
            )
            if report is not None:
                self._ba_reports.append(report)
                pose_reasons.append("local-ba")
            loop = self._detect_loop(current)
            if loop is not None:
                self._store_edge(loop)
                self._integrate_pair(loop, current)
                self._loop_closures += 1
                pose_reasons.append("loop-closure")
                if self.options.optimize_global_on_loop:
                    self._apply_global_correction(reason="accepted-loop")

        reference_pose = best.reference.pose
        rotation_frame_from_anchor = tracked_pose.rotation @ reference_pose.rotation.T
        center_delta_in_anchor = reference_pose.rotation @ (
            tracked_pose.center - reference_pose.center
        )
        record = _FrameRecord(
            frame_id=identity,
            timestamp_s=timestamp,
            state=state,
            pose=tracked_pose,
            reference_frame_id=reference_pose.frame_id,
            is_keyframe=make_keyframe,
            feature_count=len(owned_features),
            match_count=len(best.matches),
            inlier_count=int(best.relative_pose.num_inliers),
            map_correspondence_count=len(correspondences),
            median_parallax_deg=median_parallax,
            local_map_point_count=self._map.active_point_count,
            reasons=tuple(pose_reasons),
            anchor_frame_id=None if make_keyframe else reference_pose.frame_id,
            rotation_frame_from_anchor=(
                None if make_keyframe else rotation_frame_from_anchor
            ),
            center_delta_in_anchor=(None if make_keyframe else center_delta_in_anchor),
        )
        self._frames.append(record)
        return self._snapshot_frame(record)

    def run(
        self,
        frames: Iterable[tuple[str, float, Any]],
    ) -> SphericalIncrementalSLAMResult:
        for frame_id, timestamp_s, panorama in frames:
            self.add_frame(
                panorama,
                timestamp_s=timestamp_s,
                frame_id=frame_id,
            )
        return self.finish()

    def finish(self) -> SphericalIncrementalSLAMResult:
        """Freeze the session and optionally refine the complete keyframe graph."""

        if self._finished:
            raise RuntimeError("finish() may be called only once per session")
        self._finished = True
        if (
            self.options.global_refine_on_finish
            and len(self._keyframes) >= self.options.mapper.min_panoramas
            and self._edges
        ):
            self._apply_global_correction(reason="final-keyframe-graph")
        active_points = self._map.active_point_count
        success = len(self._keyframes) >= 2 and active_points > 0
        reasons = () if success else self._failure_reasons(active_points)
        return SphericalIncrementalSLAMResult(
            success=success,
            failure_reasons=reasons,
            frames=tuple(self._snapshot_frame(item) for item in self._frames),
            keyframes=tuple(self._keyframe_summaries()),
            map_points=self._map.snapshot(),
            diagnostics=self._diagnostics(),
            options=self.options,
            global_reconstruction=self._last_global,
        )

    @property
    def current_pose(self) -> SphericalSLAMPose | None:
        for frame in reversed(self._frames):
            if frame.pose is not None:
                return self._public_pose(frame.pose)
        return None

    @property
    def local_map(self):
        return self._map.snapshot()

    def describe(self) -> dict[str, Any]:
        return {
            "interface": "panorai-spherical-incremental-slam/v1",
            "input": "central-equirectangular-image",
            "pose_convention": "x_frame=R_world_to_frame@(X-C_world)",
            "scale": "arbitrary",
            "feature_pipeline": self.feature_pipeline.describe(),
            "options": self.options.to_dict(),
            "state": self._diagnostics().to_dict(),
        }

    def _validate_input_identity(self, frame_id: str, timestamp_s: float) -> float:
        if self._finished:
            raise RuntimeError("the session is finished; call reset() before reuse")
        if not isinstance(frame_id, str) or not frame_id.strip():
            raise TypeError("frame_id must be a non-empty string")
        if frame_id != frame_id.strip():
            raise ValueError("frame_id cannot have surrounding whitespace")
        timestamp = float(timestamp_s)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp_s must be finite")
        if frame_id in self._seen_ids:
            raise ValueError(f"duplicate frame_id: {frame_id!r}")
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("frame timestamps must increase strictly")
        return timestamp

    def _evaluate_references(
        self,
        references: Iterable[_KeyframeRecord],
        current: SphericalFeatureSet,
    ) -> list[_PairEvaluation]:
        results = []
        for reference in reversed(tuple(references)):
            matches = self.feature_pipeline.match(reference.features, current)
            results.append(_PairEvaluation(reference, matches, None))
        ranked = sorted(
            range(len(results)),
            key=lambda index: (
                -len(results[index].matches),
                results[index].reference.pose.frame_id,
            ),
        )
        for index in ranked[: self.options.max_tracking_pose_candidates]:
            evaluation = results[index]
            if len(evaluation.matches) >= self.options.min_pair_matches:
                evaluation.relative_pose = self.relative_pose_estimator.estimate(
                    evaluation.matches.to_bearing_correspondences()
                )
        return results

    def _find_relocalization(
        self,
        current: SphericalFeatureSet,
        excluded_ids: set[str],
    ) -> _PairEvaluation | None:
        candidates = []
        for reference in self._keyframes:
            if reference.pose.frame_id in excluded_ids:
                continue
            matches = self.feature_pipeline.match(reference.features, current)
            if len(matches) >= self.options.min_pair_matches:
                candidates.append(
                    (len(matches), reference.pose.frame_id, reference, matches)
                )
        candidates.sort(key=lambda item: (-item[0], item[1]))
        for _, _, reference, matches in candidates[: self.options.max_loop_candidates]:
            pose = self.relative_pose_estimator.estimate(
                matches.to_bearing_correspondences()
            )
            if self._admitted(pose):
                return _PairEvaluation(reference, matches, pose)
        return None

    def _detect_loop(self, current: _KeyframeRecord) -> _PairEvaluation | None:
        current_index = len(self._keyframes) - 1
        cutoff = current_index - self.options.loop_min_keyframe_separation
        if cutoff < 0:
            return None
        candidates = []
        for reference in self._keyframes[: cutoff + 1]:
            pair = (reference.pose.frame_id, current.pose.frame_id)
            if pair in self._edges:
                continue
            matches = self.feature_pipeline.match(reference.features, current.features)
            if len(matches) >= self.options.loop_min_matches:
                candidates.append(
                    (len(matches), reference.pose.frame_id, reference, matches)
                )
        candidates.sort(key=lambda item: (-item[0], item[1]))
        for _, _, reference, matches in candidates[: self.options.max_loop_candidates]:
            pose = self.relative_pose_estimator.estimate(
                matches.to_bearing_correspondences()
            )
            if self._admitted(pose):
                return _PairEvaluation(reference, matches, pose)
        return None

    def _admitted(self, pose: Any | None) -> bool:
        if pose is None:
            return False
        return self.options.edge_admission == "successful" or bool(
            pose.quality_report.accepted
        )

    def _compose_relative_pose(
        self,
        evaluation: _PairEvaluation,
        frame_id: str,
        timestamp_s: float,
    ) -> PoseState:
        reference = evaluation.reference.pose
        relative = evaluation.relative_pose
        rotation = np.asarray(relative.R) @ reference.rotation
        baseline = self._baseline_guess()
        center = reference.center - rotation.T @ (baseline * np.asarray(relative.t))
        return PoseState(frame_id, timestamp_s, rotation, center)

    def _baseline_guess(self) -> float:
        distances = []
        for left, right in zip(self._keyframes[:-1], self._keyframes[1:]):
            distance = float(np.linalg.norm(right.pose.center - left.pose.center))
            if math.isfinite(distance) and distance > 1e-9:
                distances.append(distance)
        return (
            float(np.median(distances[-self.options.local_window_size :]))
            if distances
            else self.options.initial_baseline
        )

    def _refine_from_local_map(
        self,
        initial_pose: PoseState,
        evaluation: _PairEvaluation,
        correspondences: list[tuple[np.ndarray, np.ndarray, int, str]],
    ):
        items = tuple((point, bearing) for point, bearing, _, _ in correspondences)
        primary = refine_absolute_pose(
            initial_pose,
            items,
            loss=self.options.local_ba_loss,
            loss_scale_deg=self.options.local_ba_loss_scale_deg,
            max_nfev=self.options.local_ba_max_nfev,
        )
        if (
            primary.success
            and primary.p90_error_deg <= self.options.max_reprojection_error_deg
        ):
            return primary
        estimates = [primary]
        reference = evaluation.reference.pose
        direction = initial_pose.center - reference.center
        norm = float(np.linalg.norm(direction))
        if norm > 1e-12:
            direction /= norm
            baseline = self._baseline_guess()
            estimates.extend(
                refine_absolute_pose(
                    PoseState(
                        initial_pose.frame_id,
                        initial_pose.timestamp_s,
                        initial_pose.rotation.copy(),
                        reference.center + multiplier * baseline * direction,
                    ),
                    items,
                    loss=self.options.local_ba_loss,
                    loss_scale_deg=self.options.local_ba_loss_scale_deg,
                    max_nfev=self.options.local_ba_max_nfev,
                )
                for multiplier in (0.25, 0.5, 2.0, 4.0, 8.0)
            )
        return min(
            estimates,
            key=lambda item: (
                not item.success,
                item.p90_error_deg,
                item.final_cost,
            ),
        )

    def _map_correspondences(
        self,
        evaluations: Iterable[_PairEvaluation],
    ) -> list[tuple[np.ndarray, np.ndarray, int, str]]:
        candidates = []
        for evaluation in evaluations:
            if not self._admitted(evaluation.relative_pose):
                continue
            matches = evaluation.matches
            pose_inliers = np.asarray(
                evaluation.relative_pose.inlier_mask,
                dtype=bool,
            )
            if pose_inliers.shape != matches.valid.shape:
                raise ValueError(
                    "relative-pose inlier mask must align with feature matches"
                )
            for row in np.flatnonzero(matches.valid & pose_inliers):
                point_id = self._map.point_for_feature(
                    evaluation.reference.pose.frame_id,
                    int(matches.feature_indices_a[row]),
                )
                if point_id is None:
                    continue
                point = self._map.points[point_id]
                candidates.append(
                    (
                        float(matches.descriptor_distances[row]),
                        int(matches.feature_indices_b[row]),
                        point_id,
                        point.position,
                        np.asarray(matches.bearings_b[row]),
                    )
                )
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        used_features: set[int] = set()
        used_points: set[str] = set()
        result = []
        for _, feature_index, point_id, point, bearing in candidates:
            if feature_index in used_features or point_id in used_points:
                continue
            used_features.add(feature_index)
            used_points.add(point_id)
            result.append((point.copy(), bearing.copy(), feature_index, point_id))
        return result

    def _median_parallax(
        self,
        evaluation: _PairEvaluation,
        current_pose: PoseState,
    ) -> float:
        pose = evaluation.relative_pose
        active = np.flatnonzero(pose.inlier_mask & evaluation.matches.valid)
        if not len(active):
            return 0.0
        reference = evaluation.reference.pose
        left = (reference.rotation.T @ evaluation.matches.bearings_a[active].T).T
        right = (current_pose.rotation.T @ evaluation.matches.bearings_b[active].T).T
        dots = np.sum(left * right, axis=1)
        return float(np.degrees(np.median(np.arccos(np.clip(dots, -1.0, 1.0)))))

    def _should_create_keyframe(
        self,
        timestamp_s: float,
        median_parallax_deg: float,
        tracked_ratio: float,
    ) -> bool:
        if len(self._keyframes) == 1:
            return True
        elapsed = timestamp_s - self._keyframes[-1].pose.timestamp_s
        if elapsed < self.options.keyframe_min_interval_s:
            return False
        return bool(
            elapsed >= self.options.keyframe_max_interval_s
            or median_parallax_deg >= self.options.keyframe_min_parallax_deg
            or tracked_ratio <= self.options.keyframe_min_tracked_ratio
        )

    def _store_edge(self, evaluation: _PairEvaluation) -> None:
        edge = SphericalPairwisePoseEdge(
            evaluation.matches,
            evaluation.relative_pose,
        )
        pair = edge.pair
        previous = self._edges.get(pair)
        if previous is None or edge.pose.num_inliers > previous.pose.num_inliers:
            self._edges[pair] = edge

    def _integrate_pair(
        self,
        evaluation: _PairEvaluation,
        current: _KeyframeRecord,
    ) -> None:
        matches = evaluation.matches
        pose = evaluation.relative_pose
        active = np.flatnonzero(pose.inlier_mask & matches.valid)
        active = sorted(
            (int(row) for row in active),
            key=lambda row: (
                float(matches.descriptor_distances[row]),
                int(matches.feature_indices_a[row]),
                int(matches.feature_indices_b[row]),
            ),
        )
        reference = evaluation.reference
        for row in active:
            feature_a = int(matches.feature_indices_a[row])
            feature_b = int(matches.feature_indices_b[row])
            bearing_a = np.asarray(matches.bearings_a[row], dtype=np.float64)
            bearing_b = np.asarray(matches.bearings_b[row], dtype=np.float64)
            point_a = self._map.point_for_feature(reference.pose.frame_id, feature_a)
            point_b = self._map.point_for_feature(current.pose.frame_id, feature_b)
            if point_a is not None and point_b is None:
                point = self._map.points[point_a]
                if (
                    observation_error_deg(current.pose, point.position, bearing_b)
                    <= self.options.max_reprojection_error_deg
                ):
                    self._map.add_observation(
                        point_a,
                        ObservationState(current.pose.frame_id, feature_b, bearing_b),
                    )
                continue
            if point_b is not None and point_a is None:
                point = self._map.points[point_b]
                if (
                    observation_error_deg(reference.pose, point.position, bearing_a)
                    <= self.options.max_reprojection_error_deg
                ):
                    self._map.add_observation(
                        point_b,
                        ObservationState(reference.pose.frame_id, feature_a, bearing_a),
                    )
                continue
            if point_a is not None or point_b is not None:
                continue
            point = triangulate_bearings(
                ((reference.pose, bearing_a), (current.pose, bearing_b)),
                min_angle_deg=self.options.min_triangulation_angle_deg,
                max_error_deg=self.options.max_reprojection_error_deg,
            )
            if point is not None:
                self._map.add_point(
                    point,
                    (
                        ObservationState(reference.pose.frame_id, feature_a, bearing_a),
                        ObservationState(current.pose.frame_id, feature_b, bearing_b),
                    ),
                )

    def _apply_global_correction(self, *, reason: str) -> bool:
        if len(self._edges) < 2:
            return False
        result = self.global_mapper.reconstruct(
            edges=tuple(self._edges[pair] for pair in sorted(self._edges)),
            panorama_ids=tuple(item.pose.frame_id for item in self._keyframes),
            reference_id=self._keyframes[0].pose.frame_id,
        )
        self._last_global = result
        if not result.success:
            self._messages.append(
                f"global-correction-failed:{reason}:{','.join(result.failure_reasons)}"
            )
            return False
        for public_pose in result.poses:
            pose = self._poses.get(public_pose.panorama_id)
            if pose is None:
                continue
            pose.rotation = np.asarray(public_pose.R).copy()
            pose.center = np.asarray(public_pose.center).copy()
        self._refresh_non_keyframes()
        for point in self._map.points.values():
            active = [
                (self._poses[item.frame_id], item.bearing)
                for item in point.observations.values()
                if item.active and item.frame_id in self._poses
            ]
            retriangulated = triangulate_bearings(
                active,
                min_angle_deg=self.options.min_triangulation_angle_deg,
                max_error_deg=self.options.max_reprojection_error_deg,
            )
            if retriangulated is None:
                point.active = False
            else:
                point.position = retriangulated
        report = self.local_ba.optimize(
            self._poses,
            self._map,
            [
                item.pose.frame_id
                for item in self._keyframes[-self.options.local_window_size :]
            ],
        )
        if report is not None:
            self._ba_reports.append(report)
        self._messages.append(f"global-correction-applied:{reason}")
        return True

    def _refresh_non_keyframes(self) -> None:
        for frame in self._frames:
            if (
                frame.is_keyframe
                or frame.pose is None
                or frame.anchor_frame_id is None
                or frame.rotation_frame_from_anchor is None
                or frame.center_delta_in_anchor is None
            ):
                continue
            anchor = self._poses.get(frame.anchor_frame_id)
            if anchor is None:
                continue
            frame.pose.rotation = frame.rotation_frame_from_anchor @ anchor.rotation
            frame.pose.center = (
                anchor.center + anchor.rotation.T @ frame.center_delta_in_anchor
            )

    def _record_lost(
        self,
        frame_id: str,
        timestamp_s: float,
        feature_count: int,
        *,
        match_count: int = 0,
        inlier_count: int = 0,
        map_correspondence_count: int = 0,
        reasons: tuple[str, ...],
    ) -> SphericalTrackingResult:
        record = _FrameRecord(
            frame_id=frame_id,
            timestamp_s=timestamp_s,
            state="lost",
            pose=None,
            reference_frame_id=None,
            is_keyframe=False,
            feature_count=feature_count,
            match_count=match_count,
            inlier_count=inlier_count,
            map_correspondence_count=map_correspondence_count,
            median_parallax_deg=None,
            local_map_point_count=self._map.active_point_count,
            reasons=reasons,
        )
        self._frames.append(record)
        return self._snapshot_frame(record)

    def _snapshot_frame(self, record: _FrameRecord) -> SphericalTrackingResult:
        return SphericalTrackingResult(
            frame_id=record.frame_id,
            timestamp_s=record.timestamp_s,
            state=record.state,
            pose=None if record.pose is None else self._public_pose(record.pose),
            reference_frame_id=record.reference_frame_id,
            is_keyframe=record.is_keyframe,
            feature_count=record.feature_count,
            match_count=record.match_count,
            inlier_count=record.inlier_count,
            map_correspondence_count=record.map_correspondence_count,
            median_parallax_deg=record.median_parallax_deg,
            local_map_point_count=record.local_map_point_count,
            reasons=record.reasons,
        )

    @staticmethod
    def _public_pose(pose: PoseState) -> SphericalSLAMPose:
        return SphericalSLAMPose(
            frame_id=pose.frame_id,
            timestamp_s=pose.timestamp_s,
            rotation_world_to_camera=pose.rotation,
            center_world=pose.center,
        )

    def _keyframe_summaries(self) -> list[SphericalKeyframeSummary]:
        result = []
        for item in self._keyframes:
            count = sum(
                point.active and item.pose.frame_id in point.observations
                for point in self._map.points.values()
            )
            result.append(
                SphericalKeyframeSummary(
                    frame_id=item.pose.frame_id,
                    timestamp_s=item.pose.timestamp_s,
                    pose=self._public_pose(item.pose),
                    feature_count=len(item.features),
                    active_landmark_count=int(count),
                )
            )
        return result

    def _diagnostics(self) -> SphericalIncrementalSLAMDiagnostics:
        tracked = sum(item.pose is not None for item in self._frames)
        lost = sum(item.state == "lost" for item in self._frames)
        return SphericalIncrementalSLAMDiagnostics(
            input_frame_count=self._input_count,
            tracked_frame_count=tracked,
            lost_frame_count=lost,
            keyframe_count=len(self._keyframes),
            map_point_count=len(self._map.points),
            active_map_point_count=self._map.active_point_count,
            relocalization_count=self._relocalizations,
            loop_closure_count=self._loop_closures,
            pairwise_edge_count=len(self._edges),
            local_ba_reports=tuple(self._ba_reports),
            stage_messages=tuple(self._messages),
        )

    def _failure_reasons(self, active_points: int) -> tuple[str, ...]:
        reasons = []
        if len(self._keyframes) < 2:
            reasons.append("insufficient-keyframes")
        if active_points == 0:
            reasons.append("no-active-map-points")
        return tuple(reasons or ("incremental-map-incomplete",))


__all__ = ["SphericalIncrementalSLAM"]
