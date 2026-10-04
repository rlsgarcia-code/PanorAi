"""Incremental visual front end with a PanorAi global spherical back end."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable

import numpy as np

from panorai.features import (
    MatchProvenance,
    OpenCVFeatureBackend,
    SphericalBearingCorrespondences,
    SphericalFeatureMatches,
)
from panorai.reconstruction import SphericalGlobalMapper

from ._fisheye import EquidistantFisheyeCamera
from ._models import (
    SphericalImageFrame,
    SphericalSLAMDiagnostics,
    SphericalSLAMFrameSummary,
    SphericalSLAMOptions,
    SphericalSLAMPose,
    SphericalSLAMResult,
)


@dataclass(slots=True)
class _FrameFeatures:
    frame_id: str
    timestamp_s: float
    pixels_xy: np.ndarray
    bearings_xyz: np.ndarray
    responses: np.ndarray
    descriptors: np.ndarray
    descriptor_type: str
    descriptor_metric: str
    checksum: str


class SphericalVisualSLAM:
    """Experimental calibrated-fisheye visual SLAM session.

    ``add_frame`` incrementally extracts and matches features. ``finish`` runs
    PanorAi's global spherical mapper over the retained temporal graph. Every
    input frame is a keyframe; callers control temporal sampling explicitly.
    """

    def __init__(
        self,
        camera: EquidistantFisheyeCamera,
        *,
        options: SphericalSLAMOptions | None = None,
        backend: OpenCVFeatureBackend | None = None,
        relative_pose_estimator: Any | None = None,
    ) -> None:
        if not isinstance(camera, EquidistantFisheyeCamera):
            raise TypeError("camera must be an EquidistantFisheyeCamera")
        self.camera = camera
        self.options = options or SphericalSLAMOptions()
        self.backend = backend or OpenCVFeatureBackend()
        self.backend.require_version("4.9.0")
        self.mapper = SphericalGlobalMapper(
            relative_pose_estimator=relative_pose_estimator,
            options=self.options.mapper,
        )
        self._support_mask = self.camera.valid_pixel_mask(
            edge_margin_px=self.options.edge_margin_px
        )
        self.reset()

    def reset(self) -> None:
        """Clear session state while retaining immutable configuration."""

        self._input_count = 0
        self._features: list[_FrameFeatures] = []
        self._matches: list[SphericalFeatureMatches] = []
        self._dropped: list[tuple[str, str]] = []
        self._candidate_pairs = 0
        self._pair_counts: list[tuple[tuple[str, str], int]] = []
        self._seen_frame_ids: set[str] = set()
        self._last_timestamp: float | None = None
        self._finished = False

    def add_frame(
        self,
        image: Any,
        *,
        timestamp_s: float,
        frame_id: str | None = None,
    ) -> SphericalSLAMFrameSummary | None:
        """Extract one frame and connect it to the recent keyframe window."""

        if self._finished:
            raise RuntimeError("the session is finished; call reset() before reuse")
        identity = frame_id or f"frame-{self._input_count:06d}"
        frame = SphericalImageFrame(identity, timestamp_s, np.asarray(image))
        if tuple(frame.image.shape[:2]) != self.camera.image_shape_hw:
            raise ValueError(
                "frame spatial shape must match camera.image_shape_hw; "
                f"got {frame.image.shape[:2]}"
            )
        if frame.frame_id in self._seen_frame_ids:
            raise ValueError(f"duplicate frame_id: {frame.frame_id!r}")
        if (
            self._last_timestamp is not None
            and frame.timestamp_s <= self._last_timestamp
        ):
            raise ValueError("frame timestamps must increase strictly")
        self._input_count += 1
        self._seen_frame_ids.add(frame.frame_id)
        self._last_timestamp = frame.timestamp_s
        extracted = self._extract(frame)
        if len(extracted.bearings_xyz) < self.options.min_features_per_frame:
            self._dropped.append((frame.frame_id, "insufficient-features"))
            return None

        for previous in self._features[-self.options.temporal_window :]:
            self._candidate_pairs += 1
            matches = self._match(previous, extracted)
            self._pair_counts.append(
                ((previous.frame_id, extracted.frame_id), len(matches))
            )
            if len(matches) >= self.options.min_matches_per_pair:
                self._matches.append(matches)
        self._features.append(extracted)
        return SphericalSLAMFrameSummary(
            frame_id=extracted.frame_id,
            timestamp_s=extracted.timestamp_s,
            feature_count=len(extracted.bearings_xyz),
            checksum_sha256=extracted.checksum,
        )

    def run(self, frames: Iterable[SphericalImageFrame]) -> SphericalSLAMResult:
        """Process a finite frame stream and finalize its global map."""

        for frame in frames:
            if not isinstance(frame, SphericalImageFrame):
                raise TypeError("frames must contain SphericalImageFrame objects")
            self.add_frame(
                frame.image,
                timestamp_s=frame.timestamp_s,
                frame_id=frame.frame_id,
            )
        return self.finish()

    def finish(self) -> SphericalSLAMResult:
        """Freeze the temporal graph and run global spherical reconstruction."""

        if self._finished:
            raise RuntimeError("finish() may be called only once per session")
        self._finished = True
        summaries = tuple(
            SphericalSLAMFrameSummary(
                frame_id=item.frame_id,
                timestamp_s=item.timestamp_s,
                feature_count=len(item.bearings_xyz),
                checksum_sha256=item.checksum,
            )
            for item in self._features
        )
        diagnostics = SphericalSLAMDiagnostics(
            input_frame_count=self._input_count,
            keyframe_count=len(self._features),
            dropped_frames=tuple(self._dropped),
            candidate_pair_count=self._candidate_pairs,
            retained_pair_count=len(self._matches),
            pair_match_counts=tuple(self._pair_counts),
            backend_name=self.backend.name,
            backend_version=self.backend.version,
        )
        if len(self._features) < self.options.mapper.min_panoramas:
            return SphericalSLAMResult(
                success=False,
                failure_reasons=("insufficient-keyframes",),
                poses=(),
                keyframes=summaries,
                reconstruction=None,
                diagnostics=diagnostics,
                options=self.options,
            )
        reconstruction = self.mapper.reconstruct(
            matches=tuple(self._matches),
            panorama_ids=tuple(item.frame_id for item in self._features),
        )
        if not reconstruction.success:
            return SphericalSLAMResult(
                success=False,
                failure_reasons=tuple(reconstruction.failure_reasons),
                poses=(),
                keyframes=summaries,
                reconstruction=reconstruction,
                diagnostics=diagnostics,
                options=self.options,
            )
        timestamps = {item.frame_id: item.timestamp_s for item in self._features}
        poses = tuple(
            SphericalSLAMPose(
                frame_id=pose.panorama_id,
                timestamp_s=timestamps[pose.panorama_id],
                rotation_world_to_camera=pose.R,
                center_world=pose.center,
            )
            for pose in reconstruction.poses
        )
        poses = tuple(sorted(poses, key=lambda item: item.timestamp_s))
        return SphericalSLAMResult(
            success=True,
            failure_reasons=(),
            poses=poses,
            keyframes=summaries,
            reconstruction=reconstruction,
            diagnostics=diagnostics,
            options=self.options,
        )

    def _extract(self, frame: SphericalImageFrame) -> _FrameFeatures:
        image = _opencv_image(frame.image)
        detected = self.backend.detect_and_describe(
            image,
            self._support_mask.astype(np.uint8) * 255,
            self.options.extractor,
        )
        projection = self.camera.pixels_to_rays(detected["pixels_xy"])
        pixels = detected["pixels_xy"]
        ix = np.floor(pixels[:, 0] + 0.5).astype(np.int64)
        iy = np.floor(pixels[:, 1] + 0.5).astype(np.int64)
        inside = (
            (ix >= 0)
            & (ix < self._support_mask.shape[1])
            & (iy >= 0)
            & (iy < self._support_mask.shape[0])
        )
        supported = np.zeros(len(pixels), dtype=bool)
        supported[inside] = self._support_mask[iy[inside], ix[inside]]
        keep = np.flatnonzero(projection.valid & supported)
        checksum = hashlib.sha256(
            np.ascontiguousarray(frame.image).view(np.uint8)
        ).hexdigest()
        metadata = detected["metadata"]
        return _FrameFeatures(
            frame_id=frame.frame_id,
            timestamp_s=frame.timestamp_s,
            pixels_xy=np.asarray(pixels[keep], dtype=np.float64),
            bearings_xyz=np.asarray(projection.rays_xyz[keep], dtype=np.float64),
            responses=np.asarray(detected["responses"][keep], dtype=np.float32),
            descriptors=np.asarray(detected["descriptors"][keep]),
            descriptor_type=str(metadata["type"]),
            descriptor_metric=str(metadata["metric"]),
            checksum=checksum,
        )

    def _match(
        self, left: _FrameFeatures, right: _FrameFeatures
    ) -> SphericalFeatureMatches:
        if left.descriptor_type != right.descriptor_type:
            raise RuntimeError("descriptor type changed within one SLAM session")
        raw, method = self.backend.match(
            left.descriptors,
            right.descriptors,
            left.descriptor_metric,
            self.options.matcher,
        )
        # Descriptor matchers may return multiple queries for one train row.
        # A track can contain at most one feature per frame, so retain the
        # deterministic minimum-distance one-to-one assignment.
        chosen = []
        used_query: set[int] = set()
        used_train: set[int] = set()
        for item in sorted(
            raw,
            key=lambda value: (
                value["distance"],
                value["query_idx"],
                value["train_idx"],
            ),
        ):
            query = int(item["query_idx"])
            train = int(item["train_idx"])
            if query in used_query or train in used_train:
                continue
            used_query.add(query)
            used_train.add(train)
            chosen.append(item)
        chosen.sort(key=lambda value: (value["query_idx"], value["train_idx"]))
        indices_a = np.asarray([item["query_idx"] for item in chosen], dtype=np.int64)
        indices_b = np.asarray([item["train_idx"] for item in chosen], dtype=np.int64)
        count = len(chosen)
        distances = np.asarray([item["distance"] for item in chosen], dtype=np.float32)
        ratios = (
            np.asarray([item["ratio_score"] for item in chosen], dtype=np.float32)
            if chosen and any(item["ratio_score"] is not None for item in chosen)
            else None
        )
        mutual = (
            np.asarray([bool(item["mutual"]) for item in chosen], dtype=bool)
            if chosen and any(item["mutual"] is not None for item in chosen)
            else None
        )
        face_pair = (self.camera.camera_id, self.camera.camera_id)
        provenance = MatchProvenance(
            interface="panorai-spherical-slam/v1",
            source_checksums=(left.checksum, right.checksum),
            face_pairs=tuple(face_pair for _ in chosen),
            face_pair_groups=tuple((face_pair,) for _ in chosen),
            deduplicated=True,
            selection_reason="ratio-test-then-one-to-one-minimum-distance",
        )
        return SphericalFeatureMatches(
            panorama_id_a=left.frame_id,
            panorama_id_b=right.frame_id,
            feature_indices_a=indices_a,
            feature_indices_b=indices_b,
            bearings_a=(
                left.bearings_xyz[indices_a]
                if count
                else np.empty((0, 3), dtype=np.float64)
            ),
            bearings_b=(
                right.bearings_xyz[indices_b]
                if count
                else np.empty((0, 3), dtype=np.float64)
            ),
            descriptor_distances=distances,
            ratio_scores=ratios,
            mutual=mutual,
            valid=np.ones(count, dtype=bool),
            matcher_name=method,
            matcher_config=self.options.matcher.to_dict(),
            backend_name=self.backend.name,
            backend_version=self.backend.version,
            provenance=provenance,
            keypoint_responses=(
                np.stack(
                    (left.responses[indices_a], right.responses[indices_b]), axis=1
                )
                if count
                else np.empty((0, 2), dtype=np.float32)
            ),
            face_ids_a=np.full(count, self.camera.camera_id, dtype=object),
            face_ids_b=np.full(count, self.camera.camera_id, dtype=object),
        )


def _opencv_image(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image)
    if values.dtype != np.uint8:
        if np.issubdtype(values.dtype, np.floating):
            finite = np.isfinite(values)
            if not finite.all():
                raise ValueError("floating input image must be finite")
            scale = 255.0 if values.max(initial=0.0) <= 1.0 else 1.0
            values = np.clip(values * scale, 0.0, 255.0).astype(np.uint8)
        else:
            values = np.clip(values, 0, 255).astype(np.uint8)
    if values.ndim == 3 and values.shape[2] == 1:
        values = values[..., 0]
    elif values.ndim == 3:
        import cv2

        code = cv2.COLOR_BGRA2GRAY if values.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        values = cv2.cvtColor(values, code)
    return np.ascontiguousarray(values)


__all__ = ["SphericalVisualSLAM", "SphericalBearingCorrespondences"]
