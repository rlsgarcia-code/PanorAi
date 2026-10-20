"""Persisted public PanorAi global-map execution for the isolated application."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np

from panorai.reconstruction import (
    SphericalBaselinePrior,
    SphericalGlobalMapper,
    SphericalRangePrior,
)

from .models import utc_now
from .registration import PairRegistrationService, _has_newer_pair_evidence
from .store import ProjectStore, SAFE_ID


_DISPLAY_POINT_LIMIT = 500


class GlobalMapService:
    """Run and preserve one auditable map revision per project."""

    def __init__(
        self, store: ProjectStore, registration: PairRegistrationService
    ) -> None:
        self.store = store
        self.registration = registration
        self.directory = store.root / "global-maps"

    def _path(self, project_id: str, *, create: bool = False) -> Path:
        if not SAFE_ID.fullmatch(project_id):
            raise KeyError(project_id)
        if create:
            self.directory.mkdir(parents=True, exist_ok=True)
        return self.directory / f"{project_id}.json"

    def load(self, project_id: str) -> dict[str, Any]:
        project = self.store.load(project_id)
        path = self._path(project_id)
        if not path.is_file():
            raise KeyError(project_id)
        record = json.loads(path.read_text(encoding="utf-8"))
        input_pair_ids = set(record.get("input_edge_revisions", {}))
        component_image_ids = set(record.get("component_image_ids", ()))
        accepted_revisions = self._accepted_revision_map(project)
        current_component_revisions = {
            pair.id: accepted_revisions[pair.id]
            for pair in project.pairs
            if pair.id in accepted_revisions
            and bool(
                {pair.reference_image_id, pair.source_image_id} & component_image_ids
            )
        }
        stored_component_revisions = record.get(
            "component_edge_revisions", record.get("input_edge_revisions", {})
        )
        current_ranges = [
            item
            for item in self.registration.validated_range_prior_records(project)
            if item["panorama_id"] in component_image_ids
            and item["binding_pair_id"] in input_pair_ids
        ]
        range_candidates_changed = (
            record.get("range_prior_candidates", []) != current_ranges
        )
        range_change_affects_map = range_candidates_changed and (
            len(current_ranges) >= int(record.get("range_prior_min_count", 3))
            or int(record.get("range_priors_applied", 0)) > 0
        )
        record["stale"] = (
            stored_component_revisions != current_component_revisions
            or range_change_affects_map
        )
        return record

    def load_optional(self, project_id: str) -> dict[str, Any] | None:
        try:
            return self.load(project_id)
        except KeyError:
            return None

    def run(self, project_id: str) -> dict[str, Any]:
        with self.store.project_lock(project_id):
            return self._run(project_id)

    def _run(self, project_id: str) -> dict[str, Any]:
        project = self.store.load(project_id)
        previous = self.load_optional(project_id)
        edges = []
        inputs = []
        skipped = []
        pair_id_by_images = {}
        for pair in project.pairs:
            pair_id_by_images[
                frozenset((pair.reference_image_id, pair.source_image_id))
            ] = pair.id
            try:
                edge, registration = self.registration.mapper_edge_snapshot(
                    project.id, pair.id
                )
            except (KeyError, ValueError) as exc:
                skipped.append({"pair_id": pair.id, "reason": str(exc)})
                continue
            edges.append(edge)
            inputs.append(
                {
                    "pair_id": pair.id,
                    "revision": registration["revision"],
                    "reference_image_id": pair.reference_image_id,
                    "source_image_id": pair.source_image_id,
                    "num_inliers": registration["pose"]["num_inliers"],
                    "quality_score": registration["pose"]["quality"][
                        "raw_quality_score"
                    ],
                }
            )

        mapper = SphericalGlobalMapper()
        panorama_ids = tuple(image.id for image in project.images)
        discovery = mapper.reconstruct(
            edges=tuple(edges),
            panorama_ids=panorama_ids,
            range_priors=(),
            baseline_priors=(),
        )
        discovery_diagnostics = discovery.diagnostics.to_dict()
        edge_image_ids = {image_id for edge in edges for image_id in edge.pair}
        component = (
            {pose.panorama_id for pose in discovery.poses}
            if discovery.success
            else edge_image_ids
            - set(discovery_diagnostics.get("excluded_panoramas", ()))
        )
        selected = [
            (edge, item)
            for edge, item in zip(edges, inputs, strict=True)
            if set(edge.pair) <= component
        ]
        for _, item in zip(edges, inputs, strict=True):
            if {
                item["reference_image_id"],
                item["source_image_id"],
            } <= component:
                continue
            skipped.append(
                {
                    "pair_id": item["pair_id"],
                    "reason": "outside-selected-map-component",
                }
            )
        selected_edges = [edge for edge, _ in selected]
        selected_inputs = [item for _, item in selected]
        input_pair_ids = {
            inputs[int(index)]["pair_id"] for index in discovery.admitted_edge_indices
        }

        baselines = [
            SphericalBaselinePrior(
                pair.reference_image_id,
                pair.source_image_id,
                pair.baseline_m,
            )
            for pair in project.pairs
            if pair.baseline_m is not None
            and {pair.reference_image_id, pair.source_image_id} <= component
        ]
        range_records = [
            item
            for item in self.registration.validated_range_prior_records(project)
            if item["panorama_id"] in component
            and item["binding_pair_id"] in input_pair_ids
        ]
        range_priors = (
            tuple(
                SphericalRangePrior(
                    item["panorama_id"],
                    item["feature_index"],
                    item["range_m"],
                )
                for item in range_records
            )
            if len(range_records) >= mapper.options.range_prior_min_count
            else ()
        )
        rerun_with_scale = discovery.success and bool(range_priors or baselines)
        if rerun_with_scale:
            result = mapper.reconstruct(
                edges=tuple(selected_edges),
                panorama_ids=tuple(sorted(component)),
                range_priors=range_priors,
                baseline_priors=tuple(baselines),
            )
            result_inputs = selected_inputs
            applied_range_prior_count = len(range_priors)
        else:
            result = discovery
            result_inputs = inputs
            applied_range_prior_count = 0
        record = _serialize_result(
            project,
            result,
            result_inputs,
            skipped,
            pair_id_by_images,
            range_records,
            applied_range_prior_count,
            component,
            revision=1 if previous is None else int(previous["revision"]) + 1,
        )
        self._write(record)
        return record

    def _accepted_revision_map(
        self, project, *, pair_ids: set[str] | None = None
    ) -> dict[str, int]:
        result = {}
        for pair in project.pairs:
            if pair_ids is not None and pair.id not in pair_ids:
                continue
            try:
                record = self.registration.load(project.id, pair.id)
            except KeyError:
                continue
            pose = record.get("pose")
            if (
                pose
                and pose.get("accepted")
                and not record.get("pending_match_reviews")
                and not _has_newer_pair_evidence(project, pair, record)
            ):
                result[pair.id] = int(record["revision"])
        return result

    def _write(self, record: dict[str, Any]) -> None:
        target = self._path(record["project_id"], create=True)
        payload = json.dumps(record, indent=2, allow_nan=False) + "\n"
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=target.parent, delete=False
        ) as handle:
            handle.write(payload)
            temporary = Path(handle.name)
        temporary.replace(target)


def _serialize_result(
    project,
    result,
    inputs: list[dict[str, Any]],
    skipped: list[dict[str, str]],
    pair_id_by_images: dict[frozenset[str], str],
    range_records: list[dict[str, Any]],
    applied_range_prior_count: int,
    selected_component: set[str],
    *,
    revision: int,
) -> dict[str, Any]:
    diagnostics = result.diagnostics.to_dict()
    admitted_indices = tuple(
        int(index)
        for index in getattr(result, "admitted_edge_indices", range(len(inputs)))
    )
    consumed_inputs = [inputs[index] for index in admitted_indices]
    component_image_ids = (
        {pose.panorama_id for pose in result.poses}
        if result.success
        else set(selected_component)
    )
    consumed_pair_ids = {item["pair_id"] for item in consumed_inputs}
    component_inputs = [
        item
        for item in inputs
        if bool(
            {item["reference_image_id"], item["source_image_id"]} & component_image_ids
        )
    ]
    consumed_ranges = [
        item
        for item in range_records
        if item["panorama_id"] in component_image_ids
        and item["binding_pair_id"] in consumed_pair_ids
    ]
    poses = []
    display_points = []
    if result.success:
        centers = np.stack([pose.center for pose in result.poses])
        point_xyz = np.asarray(result.points_xyz)
        camera_xy, point_xy = _project_for_display(centers, point_xyz)
        active_tracks = dict(diagnostics.get("camera_active_track_counts", ()))
        max_errors = dict(diagnostics.get("camera_max_reprojection_deg", ()))
        for index, pose in enumerate(result.poses):
            poses.append(
                {
                    "image_id": pose.panorama_id,
                    "rotation_world_to_panorama": pose.R.astype(float).tolist(),
                    "center_world": pose.center.astype(float).tolist(),
                    "display_xy": camera_xy[index].astype(float).tolist(),
                    "active_track_count": int(active_tracks.get(pose.panorama_id, 0)),
                    "max_reprojection_deg": (
                        None
                        if pose.panorama_id not in max_errors
                        else float(max_errors[pose.panorama_id])
                    ),
                }
            )
        if len(point_xy):
            indices = np.linspace(
                0,
                len(point_xy) - 1,
                min(_DISPLAY_POINT_LIMIT, len(point_xy)),
                dtype=int,
            )
            display_points = point_xy[indices].astype(float).tolist()

    return {
        "schema": "panorai-streetview-global-map/v1",
        "project_id": project.id,
        "revision": revision,
        "created_at": utc_now(),
        "stale": False,
        "engine": {
            "application": "panorai.reconstruction.SphericalGlobalMapper",
            "interface": result.interface,
            "options": result.options.to_dict(),
        },
        "success": result.success,
        "failure_reasons": list(result.failure_reasons),
        "scale": result.scale,
        "reference_image_id": result.reference_panorama_id,
        "component_image_ids": sorted(component_image_ids),
        "component_edge_revisions": {
            item["pair_id"]: item["revision"] for item in component_inputs
        },
        "input_edge_revisions": {
            item["pair_id"]: item["revision"] for item in consumed_inputs
        },
        "range_prior_candidates": consumed_ranges,
        "range_priors_applied": applied_range_prior_count,
        "range_prior_min_count": result.options.range_prior_min_count,
        "edges": consumed_inputs,
        "skipped_pairs": skipped,
        "poses": poses,
        "points": {
            "total_count": int(len(result.points_xyz)),
            "displayed_count": len(display_points),
            "display_xy": display_points,
        },
        "track_count": len(result.tracks),
        "diagnostics": diagnostics,
        "recommended_edit": _recommended_edit(
            result,
            inputs,
            diagnostics,
            pair_id_by_images,
        ),
    }


def _project_for_display(
    centers: np.ndarray, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    origin = centers.mean(axis=0)
    combined = centers if not len(points) else np.concatenate((centers, points), axis=0)
    centered = combined - origin
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    basis = axes[:2].T
    projected = centered @ basis
    scale = float(np.max(np.linalg.norm(projected, axis=1)))
    if scale > 0.0:
        projected /= scale
    return projected[: len(centers)], projected[len(centers) :]


def _recommended_edit(
    result,
    inputs: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    pair_id_by_images: dict[frozenset[str], str],
) -> dict[str, Any]:
    def pair_for(value) -> str | None:
        return pair_id_by_images.get(frozenset(value)) if value else None

    if not result.success:
        reason = result.failure_reasons[0] if result.failure_reasons else "unknown"
        if reason in {"no-admitted-pairwise-edges", "insufficient-connected-panoramas"}:
            return {
                "kind": "add-edge",
                "pair_id": None,
                "title": "Conecte ao menos três panoramas",
                "detail": "Estime e aceite pares que ampliem o mesmo componente.",
            }
        filtered = diagnostics.get("rotation_filtered_pairs") or diagnostics.get(
            "translation_filtered_pairs"
        )
        if filtered:
            return {
                "kind": "inspect-pair",
                "pair_id": pair_for(filtered[0]),
                "title": "Revise a aresta globalmente inconsistente",
                "detail": "Inspecione inliers, objetos compartilhados e a direção de t deste par.",
            }
        if reason == "no-consistent-multiview-tracks":
            return {
                "kind": "close-loop",
                "pair_id": None,
                "title": "Crie tracks observados em três vistas",
                "detail": "Execute um terceiro par do mesmo triângulo e revise correspondências compartilhadas.",
            }
        return {
            "kind": "inspect-graph",
            "pair_id": None,
            "title": f"Mapa recusado · {reason}",
            "detail": "Use os diagnósticos do mapper para corrigir arestas antes de tentar novamente.",
        }

    camera_errors = diagnostics.get("camera_max_reprojection_deg", ())
    if camera_errors:
        worst_camera = max(camera_errors, key=lambda item: item[1])[0]
        incident = next(
            (
                item
                for item in sorted(inputs, key=lambda edge: edge["quality_score"])
                if worst_camera in {item["reference_image_id"], item["source_image_id"]}
            ),
            None,
        )
        if incident is not None:
            return {
                "kind": "inspect-pair",
                "pair_id": incident["pair_id"],
                "title": "Inspecione a câmera de maior resíduo",
                "detail": (
                    "O mapa passou; revise a aresta de menor qualidade incidente "
                    "antes de adicionar mais vistas."
                ),
            }
    return {
        "kind": "accepted",
        "pair_id": None,
        "title": "Mapa global aceito",
        "detail": "A geometria está pronta para inspeção e para receber novas arestas.",
    }
