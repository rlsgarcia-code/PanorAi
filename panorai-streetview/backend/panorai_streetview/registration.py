"""Auditable pair registration through documented PanorAi public APIs."""

from __future__ import annotations

from hashlib import sha256
import json
from math import degrees
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock, RLock
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image

from panorai.applications import SphericalRelativePose
from panorai.features import (
    MatchProvenance,
    SphericalBearingCorrespondences,
    SphericalFeatureMatches,
    SphericalFeaturePipeline,
)
from panorai.geometry import erp_pixels_to_rays
from panorai.reconstruction import SphericalPairwisePoseEdge

from .models import DenseAnnotation, MatchReview, RegistrationRequest, utc_now
from .store import ProjectStore, SAFE_ID


_DETECTION_LIMIT_PER_IMAGE = 900


class PairRegistrationService:
    """Run, persist, review, and re-estimate one panorama pair."""

    def __init__(self, store: ProjectStore) -> None:
        self.store = store
        self.directory = store.root / "registrations"
        self.pose_application = SphericalRelativePose.reference()
        self._pair_locks: dict[tuple[str, str], Any] = {}
        self._pair_locks_guard = Lock()

    def pair_lock(self, project_id: str, pair_id: str):
        """Return the re-entrant lock for one persisted pair registration."""

        if not SAFE_ID.fullmatch(project_id) or not SAFE_ID.fullmatch(pair_id):
            raise KeyError(pair_id)
        key = (project_id, pair_id)
        with self._pair_locks_guard:
            return self._pair_locks.setdefault(key, RLock())

    def evidence_signature(self, project, pair) -> dict[str, Any]:
        return _effective_pair_evidence_signature(project, pair)

    def evidence_is_stale(self, project, pair, record: dict[str, Any]) -> bool:
        return _has_newer_pair_evidence(project, pair, record)

    def _path(self, project_id: str, pair_id: str, *, create: bool = False) -> Path:
        if not SAFE_ID.fullmatch(project_id) or not SAFE_ID.fullmatch(pair_id):
            raise KeyError(pair_id)
        directory = self.directory / project_id
        if create:
            directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{pair_id}.json"

    def load(self, project_id: str, pair_id: str) -> dict[str, Any]:
        path = self._path(project_id, pair_id)
        if not path.is_file():
            raise KeyError(pair_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def _history_directory(
        self, project_id: str, pair_id: str, *, create: bool = False
    ) -> Path:
        current = self._path(project_id, pair_id, create=create)
        directory = current.parent / f"{pair_id}.history"
        if create:
            directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _write(self, record: dict[str, Any], *, snapshot: bool = True) -> None:
        target = self._path(record["project_id"], record["pair_id"], create=True)
        payload = json.dumps(record, indent=2, allow_nan=False) + "\n"
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=target.parent, delete=False
        ) as handle:
            handle.write(payload)
            temporary = Path(handle.name)
        temporary.replace(target)
        if snapshot:
            history = self._history_directory(
                record["project_id"], record["pair_id"], create=True
            )
            revision = int(record["revision"])
            revision_path = history / f"revision-{revision:04d}.json"
            with NamedTemporaryFile(
                "w", encoding="utf-8", dir=history, delete=False
            ) as handle:
                handle.write(payload)
                temporary = Path(handle.name)
            temporary.replace(revision_path)

    def history(self, project_id: str, pair_id: str) -> dict[str, Any]:
        self._resolve(project_id, pair_id)
        directory = self._history_directory(project_id, pair_id)
        revisions = []
        if directory.is_dir():
            for path in sorted(directory.glob("revision-*.json"), reverse=True):
                record = json.loads(path.read_text(encoding="utf-8"))
                pose = record.get("pose")
                revisions.append(
                    {
                        "revision": record["revision"],
                        "updated_at": record["updated_at"],
                        "region_mode": record["region_mode"],
                        "accepted": bool(pose and pose.get("accepted")),
                        "num_inliers": 0 if pose is None else pose["num_inliers"],
                        "quality_score": (
                            None
                            if pose is None
                            else pose["quality"]["raw_quality_score"]
                        ),
                        "median_parallax_deg": (
                            None if pose is None else pose["median_parallax_deg"]
                        ),
                        "accepted_reviews": sum(
                            item["review_state"] == "accepted"
                            for item in record["matches"]
                        ),
                        "rejected_reviews": sum(
                            item["review_state"] == "rejected"
                            for item in record["matches"]
                        ),
                        "manual_correspondence_count": record.get(
                            "manual_correspondence_count", 0
                        ),
                    }
                )
        return {
            "schema": "panorai-streetview-registration-history/v1",
            "project_id": project_id,
            "pair_id": pair_id,
            "revisions": revisions,
        }

    def run(
        self,
        project_id: str,
        pair_id: str,
        request: RegistrationRequest,
    ) -> dict[str, Any]:
        with self.store.project_lock(project_id):
            with self.pair_lock(project_id, pair_id):
                return self._run(project_id, pair_id, request)

    def _run(
        self,
        project_id: str,
        pair_id: str,
        request: RegistrationRequest,
    ) -> dict[str, Any]:
        project, pair, reference, source = self._resolve(project_id, pair_id)
        previous = self._load_optional(project_id, pair_id)
        previous_reviews = {
            (item["feature_index_reference"], item["feature_index_source"]): item[
                "review_state"
            ]
            for item in (previous or {}).get("matches", [])
        }
        with Image.open(self.store.image_path(project_id, reference.id)) as image:
            reference_pixels = np.asarray(image.convert("RGB"))
        with Image.open(self.store.image_path(project_id, source.id)) as image:
            source_pixels = np.asarray(image.convert("RGB"))

        pipeline = SphericalFeaturePipeline.for_relative_pose()
        reference_features = pipeline.extract(
            reference_pixels, panorama_id=reference.id
        )
        source_features = pipeline.extract(source_pixels, panorama_id=source.id)
        matches = pipeline.match(reference_features, source_features)
        reference_xy = reference_features.source_erp_xy[matches.feature_indices_a]
        source_xy = source_features.source_erp_xy[matches.feature_indices_b]

        match_records = []
        for index in range(len(matches)):
            key = (
                int(matches.feature_indices_a[index]),
                int(matches.feature_indices_b[index]),
            )
            match_records.append(
                {
                    "index": index,
                    "feature_index_reference": key[0],
                    "feature_index_source": key[1],
                    "reference_xy": reference_xy[index].astype(float).tolist(),
                    "source_xy": source_xy[index].astype(float).tolist(),
                    "bearing_reference": matches.bearings_a[index]
                    .astype(float)
                    .tolist(),
                    "bearing_source": matches.bearings_b[index].astype(float).tolist(),
                    "descriptor_distance": float(matches.descriptor_distances[index]),
                    "keypoint_response_reference": float(
                        reference_features.features[key[0]].response
                    ),
                    "keypoint_response_source": float(
                        source_features.features[key[1]].response
                    ),
                    "face_id_reference": reference_features.features[key[0]].face_id,
                    "face_id_source": source_features.features[key[1]].face_id,
                    "ratio_score": (
                        None
                        if matches.ratio_scores is None
                        else float(matches.ratio_scores[index])
                    ),
                    "auto_valid": bool(matches.valid[index]),
                    "region_match": False,
                    "review_state": previous_reviews.get(key, "unreviewed"),
                    "inlier": False,
                    "residual_deg": None,
                }
            )

        record = {
            "schema": "panorai-streetview-pair-registration/v1",
            "id": f"registration_{uuid4().hex[:12]}",
            "project_id": project.id,
            "pair_id": pair.id,
            "revision": int((previous or {}).get("revision", 0)) + 1,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "region_mode": request.region_mode,
            "engine": {
                "feature_pipeline": pipeline.describe(),
                "relative_pose": self.pose_application.describe(),
            },
            "feature_counts": {
                "reference": len(reference_features),
                "source": len(source_features),
            },
            "detection_payload": {
                "bounded": True,
                "limit_per_image": _DETECTION_LIMIT_PER_IMAGE,
            },
            "detections": {
                "reference": _serialize_detections(
                    reference_features, matches.feature_indices_a
                ),
                "source": _serialize_detections(
                    source_features, matches.feature_indices_b
                ),
            },
            "matches": match_records,
            "object_summaries": [],
            "manual_correspondence_count": 0,
            "active_correspondence_count": 0,
            "pending_match_reviews": [],
            "revision_comparison": None,
            "pose": None,
        }
        self._estimate(record, project, pair, request.region_mode)
        record["revision_comparison"] = _revision_comparison(previous, record)
        self._write(record)
        return record

    def reestimate(
        self,
        project_id: str,
        pair_id: str,
        request: RegistrationRequest,
    ) -> dict[str, Any]:
        with self.store.project_lock(project_id):
            with self.pair_lock(project_id, pair_id):
                return self._reestimate(project_id, pair_id, request)

    def _reestimate(
        self,
        project_id: str,
        pair_id: str,
        request: RegistrationRequest,
    ) -> dict[str, Any]:
        project, pair, _, _ = self._resolve(project_id, pair_id)
        record = self.load(project_id, pair_id)
        previous = json.loads(json.dumps(record))
        record["id"] = f"registration_{uuid4().hex[:12]}"
        record["revision"] = int(record["revision"]) + 1
        record["updated_at"] = utc_now()
        record["region_mode"] = request.region_mode
        record["pending_match_reviews"] = []
        self._estimate(record, project, pair, request.region_mode)
        record["revision_comparison"] = _revision_comparison(previous, record)
        self._write(record)
        return record

    def review_match(
        self,
        project_id: str,
        pair_id: str,
        match_index: int,
        review: MatchReview,
    ) -> dict[str, Any]:
        with self.store.project_lock(project_id):
            with self.pair_lock(project_id, pair_id):
                return self._review_match(project_id, pair_id, match_index, review)

    def _review_match(
        self,
        project_id: str,
        pair_id: str,
        match_index: int,
        review: MatchReview,
    ) -> dict[str, Any]:
        record = self.load(project_id, pair_id)
        if int(record["revision"]) != review.revision:
            raise ValueError(
                "registration revision changed; reload matches before reviewing"
            )
        matches = record["matches"]
        if match_index < 0 or match_index >= len(matches):
            raise KeyError(str(match_index))
        matches[match_index]["review_state"] = review.state
        pending = set(record.get("pending_match_reviews", []))
        pending.add(match_index)
        record["pending_match_reviews"] = sorted(pending)
        record["updated_at"] = utc_now()
        self._write(record, snapshot=False)
        return record

    def mapper_edge(self, project_id: str, pair_id: str) -> SphericalPairwisePoseEdge:
        edge, _ = self.mapper_edge_snapshot(project_id, pair_id)
        return edge

    def mapper_edge_snapshot(
        self, project_id: str, pair_id: str
    ) -> tuple[SphericalPairwisePoseEdge, dict[str, Any]]:
        """Return an edge and the exact persisted revision used to rebuild it."""

        with self.store.project_lock(project_id):
            with self.pair_lock(project_id, pair_id):
                return self._mapper_edge_snapshot(project_id, pair_id)

    def _mapper_edge_snapshot(
        self, project_id: str, pair_id: str
    ) -> tuple[SphericalPairwisePoseEdge, dict[str, Any]]:
        """Rebuild one typed public mapper edge from an accepted revision.

        The pose is deterministically re-estimated from persisted bearings and
        the current human evidence. This avoids deserializing private estimator
        state while preserving PanorAi feature indices as multiview identity.
        """

        project, pair, reference, source = self._resolve(project_id, pair_id)
        record = self.load(project_id, pair_id)
        if record.get("pending_match_reviews"):
            raise ValueError("pair has match reviews pending re-estimation")
        if _has_newer_pair_evidence(project, pair, record):
            raise ValueError("pair has human evidence pending re-estimation")
        if not record.get("pose") or not record["pose"].get("accepted"):
            raise ValueError("pair does not have an accepted pose revision")
        paired_regions = _paired_regions(project, pair)
        region_mode = record.get("region_mode", "all")
        if region_mode == "restrict" and not paired_regions:
            raise ValueError("restricted pair no longer has paired object regions")

        rows = list(record["matches"])
        bearings_a = [item["bearing_reference"] for item in rows]
        bearings_b = [item["bearing_source"] for item in rows]
        indices_a = [item["feature_index_reference"] for item in rows]
        indices_b = [item["feature_index_source"] for item in rows]
        distances = [item["descriptor_distance"] for item in rows]
        responses = [
            [
                item.get("keypoint_response_reference", 1.0),
                item.get("keypoint_response_source", 1.0),
            ]
            for item in rows
        ]
        faces_a = [item.get("face_id_reference", "persisted") for item in rows]
        faces_b = [item.get("face_id_source", "persisted") for item in rows]
        valid = np.asarray(
            [_match_is_active(item) for item in rows],
            dtype=bool,
        )
        weights = np.ones(len(rows), dtype=np.float64)
        for index, item in enumerate(rows):
            in_region = _match_in_paired_region(item, paired_regions)
            if item["review_state"] == "accepted":
                weights[index] = 3.0
            if in_region and region_mode == "prioritize":
                weights[index] *= 4.0
            if region_mode == "restrict" and not in_region:
                valid[index] = False

        manual = [
            item
            for item in project.manual_correspondences
            if item.pair_id == pair.id and item.state == "accepted"
        ]
        if manual:
            manual_a = np.asarray(
                erp_pixels_to_rays(
                    np.asarray([item.reference_xy for item in manual]),
                    (reference.height, reference.width),
                )
            )
            manual_b = np.asarray(
                erp_pixels_to_rays(
                    np.asarray([item.source_xy for item in manual]),
                    (source.height, source.width),
                )
            )
            for index, item in enumerate(manual):
                identifier = item.id or f"manual-{index}"
                indices_a.append(_manual_feature_index(identifier, reference.id))
                indices_b.append(_manual_feature_index(identifier, source.id))
                bearings_a.append(manual_a[index].tolist())
                bearings_b.append(manual_b[index].tolist())
                distances.append(0.0)
                responses.append([1.0, 1.0])
                faces_a.append("manual")
                faces_b.append("manual")
            valid = np.concatenate((valid, np.ones(len(manual), dtype=bool)))
            weights = np.concatenate((weights, np.full(len(manual), 8.0)))

        bearing_array_a = np.asarray(bearings_a, dtype=np.float64).reshape((-1, 3))
        bearing_array_b = np.asarray(bearings_b, dtype=np.float64).reshape((-1, 3))
        pose = self.pose_application.estimate_correspondences(
            SphericalBearingCorrespondences(
                bearing_array_a,
                bearing_array_b,
                weights,
                valid,
            )
        )
        if pose is None or not pose.quality_report.accepted:
            raise ValueError("accepted pair no longer reproduces an accepted pose")

        face_pairs = tuple(zip(faces_a, faces_b, strict=True))
        matcher = (
            record.get("engine", {}).get("feature_pipeline", {}).get("matcher", {})
        )
        match_set = SphericalFeatureMatches(
            panorama_id_a=reference.id,
            panorama_id_b=source.id,
            feature_indices_a=np.asarray(indices_a, dtype=np.int64),
            feature_indices_b=np.asarray(indices_b, dtype=np.int64),
            bearings_a=bearing_array_a,
            bearings_b=bearing_array_b,
            descriptor_distances=np.asarray(distances, dtype=np.float32),
            ratio_scores=None,
            mutual=None,
            valid=valid,
            matcher_name=str(matcher.get("method", "persisted-panorai-matches")),
            matcher_config={
                "registration_revision": record["revision"],
                "region_mode": region_mode,
            },
            backend_name="panorai-streetview-public-adapter",
            backend_version="1",
            provenance=MatchProvenance(
                interface="panorai-streetview-persisted-matches/v1",
                source_checksums=(reference.sha256, source.sha256),
                face_pairs=face_pairs,
                face_pair_groups=tuple((item,) for item in face_pairs),
                deduplicated=True,
                selection_reason="persisted-panorai-feature-index",
            ),
            keypoint_responses=np.asarray(responses, dtype=np.float32),
            face_ids_a=np.asarray(faces_a, dtype=object),
            face_ids_b=np.asarray(faces_b, dtype=object),
        )
        return SphericalPairwisePoseEdge(match_set, pose), record

    def bind_range_anchor(
        self,
        project_id: str,
        annotation: DenseAnnotation,
        *,
        max_distance_px: float = 48.0,
    ) -> DenseAnnotation:
        """Bind a human click to the nearest accepted inlier feature observation."""

        if annotation.kind != "range_anchor":
            return annotation
        annotation = annotation.model_copy(
            update={
                "feature_index": None,
                "binding_pair_id": None,
                "binding_distance_px": None,
                "engine_effect": "staged",
            }
        )
        project = self.store.load(project_id)
        image = next(
            (item for item in project.images if item.id == annotation.image_id), None
        )
        if image is None:
            raise ValueError("dense annotation image_id does not belong to project")
        anchor = annotation.points_xy[0]
        candidates: list[tuple[float, str, int]] = []
        for pair in project.pairs:
            if image.id not in {pair.reference_image_id, pair.source_image_id}:
                continue
            try:
                record = self.load(project_id, pair.id)
            except KeyError:
                continue
            if (
                not record.get("pose")
                or not record["pose"].get("accepted")
                or record.get("pending_match_reviews")
                or _has_newer_pair_evidence(project, pair, record)
            ):
                continue
            reference_side = image.id == pair.reference_image_id
            for match in record["matches"]:
                if not match.get("inlier") or match.get("review_state") == "rejected":
                    continue
                xy = match["reference_xy" if reference_side else "source_xy"]
                dx = abs(float(xy[0]) - anchor[0])
                dx = min(dx, image.width - dx)
                dy = float(xy[1]) - anchor[1]
                distance = float(np.hypot(dx, dy))
                feature_index = int(
                    match[
                        "feature_index_reference"
                        if reference_side
                        else "feature_index_source"
                    ]
                )
                candidates.append((distance, pair.id, feature_index))
        if not candidates:
            return annotation
        distance, pair_id, feature_index = min(candidates)
        if distance > max_distance_px:
            return annotation
        return annotation.model_copy(
            update={
                "feature_index": feature_index,
                "binding_pair_id": pair_id,
                "binding_distance_px": distance,
                "engine_effect": "range_prior",
            }
        )

    def validated_range_prior_records(self, project) -> list[dict[str, Any]]:
        """Return range anchors whose persisted feature binding is still current."""

        pairs = {item.id: item for item in project.pairs}
        records = []
        for annotation in project.dense_annotations:
            if (
                annotation.state != "accepted"
                or annotation.engine_effect != "range_prior"
                or annotation.feature_index is None
                or annotation.radial_range is None
                or annotation.binding_pair_id is None
            ):
                continue
            pair = pairs.get(annotation.binding_pair_id)
            if pair is None or annotation.image_id not in {
                pair.reference_image_id,
                pair.source_image_id,
            }:
                continue
            try:
                registration = self.load(project.id, pair.id)
            except KeyError:
                continue
            if (
                not registration.get("pose")
                or not registration["pose"].get("accepted")
                or registration.get("pending_match_reviews")
                or _has_newer_pair_evidence(project, pair, registration)
            ):
                continue
            reference_side = annotation.image_id == pair.reference_image_id
            feature_key = (
                "feature_index_reference" if reference_side else "feature_index_source"
            )
            if not any(
                match.get("inlier")
                and match.get("review_state") != "rejected"
                and int(match[feature_key]) == annotation.feature_index
                for match in registration.get("matches", ())
            ):
                continue
            records.append(
                {
                    "annotation_id": annotation.id,
                    "panorama_id": annotation.image_id,
                    "feature_index": annotation.feature_index,
                    "range_m": annotation.radial_range,
                    "binding_pair_id": annotation.binding_pair_id,
                }
            )
        return records

    def _load_optional(self, project_id: str, pair_id: str) -> dict[str, Any] | None:
        try:
            return self.load(project_id, pair_id)
        except KeyError:
            return None

    def _resolve(self, project_id: str, pair_id: str):
        project = self.store.load(project_id)
        if project.workflow not in {
            "spherical_registered",
            "spherical_unregistered",
        }:
            raise ValueError("pair registration requires a spherical ERP workflow")
        pair = next((item for item in project.pairs if item.id == pair_id), None)
        if pair is None:
            raise KeyError(pair_id)
        reference = next(
            item for item in project.images if item.id == pair.reference_image_id
        )
        source = next(
            item for item in project.images if item.id == pair.source_image_id
        )
        return project, pair, reference, source

    def _estimate(self, record, project, pair, region_mode: str) -> None:
        matches = record["matches"]
        paired_regions = _paired_regions(project, pair)
        if region_mode == "restrict" and not paired_regions:
            raise ValueError(
                "region restriction requires the same accepted object in both images"
            )

        reference_bearings = np.asarray(
            [item["bearing_reference"] for item in matches], dtype=np.float64
        ).reshape((-1, 3))
        source_bearings = np.asarray(
            [item["bearing_source"] for item in matches], dtype=np.float64
        ).reshape((-1, 3))
        valid = np.asarray(
            [_match_is_active(item) for item in matches],
            dtype=bool,
        )
        weights = np.ones(len(matches), dtype=np.float64)
        for item in matches:
            item["region_match"] = _match_in_paired_region(item, paired_regions)
            item["inlier"] = False
            item["residual_deg"] = None
            if item["review_state"] == "accepted":
                weights[item["index"]] = 3.0
            if item["region_match"] and region_mode == "prioritize":
                weights[item["index"]] *= 4.0
        if region_mode == "restrict":
            valid &= np.asarray([item["region_match"] for item in matches], dtype=bool)

        manual = [
            item
            for item in project.manual_correspondences
            if item.pair_id == pair.id and item.state == "accepted"
        ]
        if manual:
            reference_image = next(
                item for item in project.images if item.id == pair.reference_image_id
            )
            source_image = next(
                item for item in project.images if item.id == pair.source_image_id
            )
            manual_reference = np.asarray(
                erp_pixels_to_rays(
                    np.asarray([item.reference_xy for item in manual]),
                    (reference_image.height, reference_image.width),
                )
            )
            manual_source = np.asarray(
                erp_pixels_to_rays(
                    np.asarray([item.source_xy for item in manual]),
                    (source_image.height, source_image.width),
                )
            )
            reference_bearings = np.concatenate(
                (reference_bearings, manual_reference), axis=0
            )
            source_bearings = np.concatenate((source_bearings, manual_source), axis=0)
            valid = np.concatenate((valid, np.ones(len(manual), dtype=bool)))
            weights = np.concatenate((weights, np.full(len(manual), 8.0)))

        record["object_summaries"] = _object_summaries(
            project, pair, record, paired_regions
        )
        record["manual_correspondence_count"] = len(manual)
        record["evidence_signature"] = _effective_pair_evidence_signature(project, pair)
        record["active_correspondence_count"] = int(valid.sum())
        record.setdefault("engine", {})["relative_pose"] = (
            self.pose_application.describe()
        )
        pose = self.pose_application.estimate_correspondences(
            SphericalBearingCorrespondences(
                reference_bearings,
                source_bearings,
                weights,
                valid,
            )
        )
        if pose is None:
            record["pose"] = None
            record["quality_gates"] = []
            record["recommended_edit"] = {
                "tool": "manual_match",
                "title": "Acrescente correspondências distribuídas",
                "detail": (
                    "O perfil PanorAi não encontrou uma hipótese. Marque pontos "
                    "em regiões distintas ou escolha outro par."
                ),
            }
            return

        for index, item in enumerate(matches):
            item["inlier"] = bool(pose.inlier_mask[index])
            item["residual_deg"] = (
                degrees(float(pose.residuals_rad[index]))
                if np.isfinite(pose.residuals_rad[index])
                else None
            )
        auto_count = len(matches)
        record["pose"] = {
            "rotation": pose.rotation.astype(float).tolist(),
            "translation_direction": pose.translation_direction.astype(float).tolist(),
            "essential_matrix": pose.essential_matrix.astype(float).tolist(),
            "num_inliers": pose.num_inliers,
            "auto_inlier_count": int(pose.inlier_mask[:auto_count].sum()),
            "manual_inlier_count": int(pose.inlier_mask[auto_count:].sum()),
            "median_parallax_deg": pose.median_parallax_deg,
            "cheirality_ratio": pose.cheirality_ratio,
            "degenerate": pose.degenerate,
            "degeneracy_reasons": list(pose.degeneracy_reasons),
            "accepted": pose.quality_report.accepted,
            "quality": pose.quality_report.to_dict(),
            "diagnostics": pose.describe(),
            "acceptance_policy": pose.acceptance_policy.to_dict(),
            "compute_backend": pose.compute_backend,
            "translation_scale": "unobservable",
            "pose_convention": "source-from-reference",
        }
        record["quality_gates"] = _quality_gates(pose)
        record["recommended_edit"] = _recommended_edit(record["quality_gates"])


def _serialize_detections(
    features, matched_indices: np.ndarray
) -> list[dict[str, Any]]:
    required = set(int(value) for value in np.asarray(matched_indices).tolist())
    ranked = sorted(
        range(len(features.features)),
        key=lambda index: (-features.features[index].response, index),
    )
    selected = list(required)
    selected.extend(index for index in ranked if index not in required)
    selected = selected[:_DETECTION_LIMIT_PER_IMAGE]
    selected.sort()
    return [
        {
            "index": index,
            "xy": features.features[index].source_erp_xy.astype(float).tolist(),
            "response": float(features.features[index].response),
            "scale": float(features.features[index].scale),
            "face_id": features.features[index].face_id,
        }
        for index in selected
        if features.features[index].source_erp_xy is not None
    ]


def _manual_feature_index(correspondence_id: str, panorama_id: str) -> int:
    digest = sha256(f"{correspondence_id}:{panorama_id}".encode()).hexdigest()
    return 10_000_000 + int(digest[:14], 16)


def _has_newer_pair_evidence(project, pair, record: dict[str, Any]) -> bool:
    current_signature = _effective_pair_evidence_signature(project, pair)
    if isinstance(record.get("evidence_signature"), dict):
        return record["evidence_signature"] != current_signature
    timestamps = [
        item.created_at
        for item in project.manual_correspondences
        if item.pair_id == pair.id
        and item.state == "accepted"
        and item.created_at is not None
    ]
    for reference, source in _paired_regions(project, pair).values():
        timestamps.extend(
            item.created_at
            for item in (reference, source)
            if item.created_at is not None
        )
    return bool(timestamps and max(timestamps) > str(record.get("updated_at", "")))


def _effective_pair_evidence_signature(project, pair) -> dict[str, Any]:
    regions = []
    for object_id, (reference, source) in sorted(
        _paired_regions(project, pair).items()
    ):
        regions.append(
            {
                "object_id": object_id,
                "reference_region_id": reference.id,
                "source_region_id": source.id,
                "reference_bbox_xyxy": list(reference.bbox_xyxy),
                "source_bbox_xyxy": list(source.bbox_xyxy),
            }
        )
    manual = sorted(
        (
            item
            for item in project.manual_correspondences
            if item.pair_id == pair.id and item.state == "accepted"
        ),
        key=lambda item: item.id or "",
    )
    return {
        "object_regions": regions,
        "manual_correspondences": [
            {
                "id": item.id,
                "reference_xy": list(item.reference_xy),
                "source_xy": list(item.source_xy),
            }
            for item in manual
        ],
    }


def _paired_regions(project, pair) -> dict[str, tuple[Any, Any]]:
    accepted = [item for item in project.object_regions if item.state == "accepted"]
    result = {}
    for object_id in {item.object_id for item in accepted}:
        reference = next(
            (
                item
                for item in accepted
                if item.object_id == object_id
                and item.image_id == pair.reference_image_id
            ),
            None,
        )
        source = next(
            (
                item
                for item in accepted
                if item.object_id == object_id and item.image_id == pair.source_image_id
            ),
            None,
        )
        if reference is not None and source is not None:
            result[object_id] = (reference, source)
    return result


def _inside(point: list[float], bbox: tuple[float, float, float, float]) -> bool:
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def _match_in_paired_region(item, paired_regions) -> bool:
    return any(
        _inside(item["reference_xy"], reference.bbox_xyxy)
        and _inside(item["source_xy"], source.bbox_xyxy)
        for reference, source in paired_regions.values()
    )


def _match_is_active(item: dict[str, Any]) -> bool:
    """Apply an explicit human review after, and above, the matcher default."""

    review_state = item["review_state"]
    return review_state != "rejected" and (
        bool(item["auto_valid"]) or review_state == "accepted"
    )


def _object_summaries(project, pair, record, paired_regions) -> list[dict[str, Any]]:
    detections_reference = record["detections"]["reference"]
    detections_source = record["detections"]["source"]
    summaries = []
    for object_id, (reference, source) in sorted(paired_regions.items()):
        summaries.append(
            {
                "object_id": object_id,
                "label": reference.label,
                "reference_region_id": reference.id,
                "source_region_id": source.id,
                "reference_bbox_xyxy": list(reference.bbox_xyxy),
                "source_bbox_xyxy": list(source.bbox_xyxy),
                "reference_detection_count": sum(
                    _inside(item["xy"], reference.bbox_xyxy)
                    for item in detections_reference
                ),
                "source_detection_count": sum(
                    _inside(item["xy"], source.bbox_xyxy) for item in detections_source
                ),
                "match_count": sum(
                    _inside(item["reference_xy"], reference.bbox_xyxy)
                    and _inside(item["source_xy"], source.bbox_xyxy)
                    for item in record["matches"]
                ),
            }
        )
    return summaries


def _revision_comparison(
    previous: dict[str, Any] | None, current: dict[str, Any]
) -> dict[str, Any] | None:
    if previous is None:
        return None
    before = previous.get("pose")
    after = current.get("pose")
    if before is None and after is None:
        return {
            "from_revision": previous["revision"],
            "to_revision": current["revision"],
            "solution_changed": False,
            "inlier_delta": 0,
            "quality_delta": None,
            "parallax_delta_deg": None,
            "accepted_before": False,
            "accepted_after": False,
        }
    return {
        "from_revision": previous["revision"],
        "to_revision": current["revision"],
        "solution_changed": (before is None) != (after is None),
        "inlier_delta": (0 if after is None else after["num_inliers"])
        - (0 if before is None else before["num_inliers"]),
        "quality_delta": (
            None
            if before is None or after is None
            else after["quality"]["raw_quality_score"]
            - before["quality"]["raw_quality_score"]
        ),
        "parallax_delta_deg": (
            None
            if before is None or after is None
            else after["median_parallax_deg"] - before["median_parallax_deg"]
        ),
        "accepted_before": bool(before and before["accepted"]),
        "accepted_after": bool(after and after["accepted"]),
    }


def _quality_gates(pose) -> list[dict[str, Any]]:
    quality = pose.quality_report
    policy = pose.acceptance_policy
    stability = quality.stability
    model = quality.model_competition
    values = (
        (
            "inliers",
            "Inliers",
            quality.num_inliers,
            policy.min_inliers,
            "minimum",
            quality.num_inliers >= policy.min_inliers,
            "Revise falsos matches e acrescente correspondências manuais.",
        ),
        (
            "inlier-ratio",
            "Razão de inliers",
            quality.inlier_ratio,
            policy.min_inlier_ratio,
            "minimum",
            quality.inlier_ratio >= policy.min_inlier_ratio,
            "Rejeite outliers ou restrinja o matching ao objeto compartilhado.",
        ),
        (
            "coverage",
            "Cobertura angular",
            min(quality.coverage_entropy_a, quality.coverage_entropy_b),
            policy.min_coverage_entropy,
            "minimum",
            min(quality.coverage_entropy_a, quality.coverage_entropy_b)
            >= policy.min_coverage_entropy,
            "Distribua novos matches por regiões afastadas do panorama.",
        ),
        (
            "residual",
            "Resíduo mediano",
            quality.median_residual_deg,
            policy.max_median_residual_deg,
            "maximum",
            quality.median_residual_deg <= policy.max_median_residual_deg,
            "Inspecione as linhas de maior resíduo e rejeite inconsistências.",
        ),
        (
            "parallax",
            "Paralaxe",
            quality.median_parallax_deg,
            policy.min_median_parallax_deg,
            "minimum",
            quality.median_parallax_deg >= policy.min_median_parallax_deg,
            "Escolha uma vista com maior deslocamento ou maior variação de profundidade.",
        ),
        (
            "cheirality",
            "Cheiralidade",
            quality.cheirality_ratio,
            policy.min_cheirality_ratio,
            "minimum",
            quality.cheirality_ratio >= policy.min_cheirality_ratio,
            "Revise a orientação de t e correspondências em profundidades distintas.",
        ),
        (
            "rotation-stability",
            "Estabilidade de R (P90)",
            stability.rotation_p90_deg,
            policy.max_stability_rotation_p90_deg,
            "maximum",
            stability.successful_trials == stability.requested_trials
            and stability.rotation_p90_deg <= policy.max_stability_rotation_p90_deg,
            "Adicione estrutura distribuída e remova matches ambíguos.",
        ),
        (
            "translation-stability",
            "Estabilidade de t (P90)",
            stability.translation_p90_deg,
            policy.max_stability_translation_p90_deg,
            "maximum",
            stability.successful_trials == stability.requested_trials
            and stability.translation_p90_deg
            <= policy.max_stability_translation_p90_deg,
            "Inclua matches em planos e profundidades diferentes.",
        ),
        (
            "model",
            "Modelo preferido",
            model.preferred_model,
            "essential" if policy.require_essential_preferred else "any",
            "equals",
            not policy.require_essential_preferred
            or model.preferred_model == "essential",
            "Escolha outro par; rotação pura ou cena planar domina este par.",
        ),
    )
    return [
        {
            "id": gate_id,
            "label": label,
            "value": value,
            "threshold": threshold,
            "threshold_kind": threshold_kind,
            "passed": passed,
            "action": action,
        }
        for gate_id, label, value, threshold, threshold_kind, passed, action in values
    ]


def _recommended_edit(gates: list[dict[str, Any]]) -> dict[str, str]:
    failed = next((gate for gate in gates if not gate["passed"]), None)
    if failed is None:
        return {
            "tool": "inspect",
            "title": "Pose aceita; confirme a distribuição visual",
            "detail": (
                "Todos os gates do perfil de referência passaram. Inspecione os "
                "inliers antes de promover este par ao grafo."
            ),
        }
    tool_by_gate = {
        "inliers": "manual_match",
        "inlier-ratio": "object_box",
        "coverage": "manual_match",
        "residual": "matches",
        "parallax": "next_pair",
        "cheirality": "manual_match",
        "rotation-stability": "object_box",
        "translation-stability": "manual_match",
        "model": "next_pair",
    }
    return {
        "tool": tool_by_gate.get(failed["id"], "inspect"),
        "title": f"Ação sugerida · {failed['label']}",
        "detail": failed["action"],
    }
