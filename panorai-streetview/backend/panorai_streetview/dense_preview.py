"""Human-masked pair preview through the public PanorAi dense application."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
from PIL import Image

from panorai.applications import PanoramicDenseStereo
from panorai.stereo import colorize_spherical_range
from panorai.stereo.panoramic import PanoramicStereoPose

from .models import utc_now
from .registration import PairRegistrationService
from .store import ProjectStore, SAFE_ID


_PREVIEW_SHAPE = (128, 256)


class PairDensePreviewService:
    """Persist two auditable, independently masked dense directions per pair."""

    def __init__(
        self, store: ProjectStore, registration: PairRegistrationService
    ) -> None:
        self.store = store
        self.registration = registration
        self.directory = store.root / "dense-previews"
        self.application = PanoramicDenseStereo(
            route="sphere-native", pose_mode="provided"
        )

    def _path(self, project_id: str, pair_id: str, suffix: str) -> Path:
        if not SAFE_ID.fullmatch(project_id) or not SAFE_ID.fullmatch(pair_id):
            raise KeyError(pair_id)
        return self.directory / project_id / f"{pair_id}.{suffix}"

    def load(self, project_id: str, pair_id: str) -> dict[str, Any]:
        project, pair, reference, source = self._resolve(project_id, pair_id)
        path = self._path(project_id, pair_id, "json")
        if not path.is_file():
            raise KeyError(pair_id)
        record = json.loads(path.read_text(encoding="utf-8"))
        if "reverse" not in record or "source_input_mask" not in record:
            raise KeyError(pair_id)
        registration = self.registration.load(project_id, pair_id)
        record["stale"] = (
            record.get("registration_revision") != registration.get("revision")
            or record.get("registration_updated_at") != registration.get("updated_at")
            or self.registration.evidence_is_stale(project, pair, registration)
            or record.get("annotation_signature")
            != _annotation_signature(project, (reference.id, source.id))
        )
        return record

    def load_optional(self, project_id: str, pair_id: str) -> dict[str, Any] | None:
        try:
            return self.load(project_id, pair_id)
        except KeyError:
            return None

    def image_path(
        self, project_id: str, pair_id: str, *, revision: int | None = None
    ) -> Path:
        record = self.load(project_id, pair_id)
        if revision is not None and int(record["revision"]) != revision:
            raise ValueError("dense preview revision changed; reload before download")
        expected_name = f"{pair_id}.revision-{int(record['revision']):04d}.png"
        filename = record.get("visualization_file", f"{pair_id}.png")
        if "visualization_file" in record and filename != expected_name:
            raise KeyError(pair_id)
        path = self._path(project_id, pair_id, "png").with_name(filename)
        if not path.is_file():
            raise KeyError(pair_id)
        return path

    def run(self, project_id: str, pair_id: str) -> dict[str, Any]:
        with self.store.project_lock(project_id):
            with self.registration.pair_lock(project_id, pair_id):
                return self._run(project_id, pair_id)

    def _run(self, project_id: str, pair_id: str) -> dict[str, Any]:
        project, pair, reference, source = self._resolve(project_id, pair_id)
        edge, registration = self.registration.mapper_edge_snapshot(project_id, pair_id)
        reference_pixels = _preview_image(
            self.store.image_path(project_id, reference.id)
        )
        source_pixels = _preview_image(self.store.image_path(project_id, source.id))
        validity, mask_details = _reference_validity(
            project, reference.id, (reference.height, reference.width)
        )
        source_validity, source_mask_details = _reference_validity(
            project, source.id, (source.height, source.width)
        )
        baseline = pair.baseline_m
        scale_status = "metric" if baseline is not None else "scale-free"
        baseline_value = float(baseline) if baseline is not None else 1.0
        translation = np.asarray(edge.pose.t, dtype=np.float64)
        translation = translation / np.linalg.norm(translation) * baseline_value
        pose = PanoramicStereoPose(
            np.asarray(edge.pose.R, dtype=np.float64),
            translation,
            baseline_value,
            "m" if baseline is not None else "baseline-units",
            scale_status,
            "accepted-panorai-pair-revision",
        )
        result = self.application.run(
            reference_pixels,
            source_pixels,
            pose=pose,
            reference_validity=validity,
        )
        inverse_rotation = np.asarray(edge.pose.R, dtype=np.float64).T
        inverse_pose = PanoramicStereoPose(
            inverse_rotation,
            -inverse_rotation @ translation,
            baseline_value,
            "m" if baseline is not None else "baseline-units",
            scale_status,
            "accepted-panorai-pair-revision-inverse",
        )
        reverse = self.application.run(
            source_pixels,
            reference_pixels,
            pose=inverse_pose,
            reference_validity=source_validity,
        )
        valid = np.asarray(result.validity, dtype=bool)
        confidence = np.asarray(result.confidence)
        ranges = np.asarray(result.radial_range)
        reverse_valid = np.asarray(reverse.validity, dtype=bool)
        reverse_confidence = np.asarray(reverse.confidence)
        reverse_ranges = np.asarray(reverse.radial_range)
        previous = self.load_optional(project_id, pair_id)
        record = {
            "schema": "panorai-streetview-dense-preview/v1",
            "project_id": project_id,
            "pair_id": pair_id,
            "revision": 1 if previous is None else int(previous["revision"]) + 1,
            "created_at": utc_now(),
            "stale": False,
            "engine": self.application.describe(),
            "registration_revision": registration["revision"],
            "registration_updated_at": registration["updated_at"],
            "annotation_signature": _annotation_signature(
                project, (reference.id, source.id)
            ),
            "reference_image_id": reference.id,
            "source_image_id": source.id,
            "preview_shape_hw": list(_PREVIEW_SHAPE),
            "scale_status": result.pose.scale_status,
            "units": result.pose.units,
            "input_mask": mask_details,
            "source_input_mask": source_mask_details,
            "valid_pixel_count": int(valid.sum()),
            "valid_fraction": float(valid.mean()),
            "valid_outside_input_count": int(np.count_nonzero(valid & ~validity)),
            "median_range": float(np.median(ranges[valid])) if np.any(valid) else None,
            "median_confidence": (
                float(np.median(confidence[valid])) if np.any(valid) else None
            ),
            "range_min": float(np.min(ranges[valid])) if np.any(valid) else None,
            "range_max": float(np.max(ranges[valid])) if np.any(valid) else None,
            "reverse": _direction_summary(
                reverse_valid,
                reverse_ranges,
                reverse_confidence,
                source_validity,
                reverse.pose,
            ),
            "diagnostics": {
                "route": self.application.route,
                "pipeline_stages": list(result.diagnostics.get("pipeline_stages", ())),
                "pose_provenance": result.pose.provenance,
            },
        }
        visualization = np.concatenate(
            (
                colorize_spherical_range(ranges, valid),
                colorize_spherical_range(reverse_ranges, reverse_valid),
            ),
            axis=1,
        )
        self._write(record, visualization)
        return record

    def _resolve(self, project_id: str, pair_id: str):
        project = self.store.load(project_id)
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

    def _write(self, record: dict[str, Any], visualization: np.ndarray) -> None:
        json_path = self._path(record["project_id"], record["pair_id"], "json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        image_path = self._path(
            record["project_id"],
            record["pair_id"],
            f"revision-{int(record['revision']):04d}.png",
        )
        record["visualization_file"] = image_path.name
        with NamedTemporaryFile("wb", dir=image_path.parent, delete=False) as handle:
            Image.fromarray(visualization).save(handle, format="PNG")
            temporary_image = Path(handle.name)
        temporary_image.replace(image_path)
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=json_path.parent, delete=False
        ) as handle:
            json.dump(record, handle, indent=2, allow_nan=False)
            handle.write("\n")
            temporary_json = Path(handle.name)
        temporary_json.replace(json_path)


def _preview_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(
            image.convert("RGB").resize(
                (_PREVIEW_SHAPE[1], _PREVIEW_SHAPE[0]), Image.Resampling.LANCZOS
            )
        )


def _annotation_signature(project, image_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    selected = set(image_ids)
    return [
        {
            "id": item.id,
            "image_id": item.image_id,
            "kind": item.kind,
            "points_xy": [list(point) for point in item.points_xy],
            "state": item.state,
        }
        for item in project.dense_annotations
        if item.image_id in selected
        and item.state == "accepted"
        and item.kind in {"include_region", "exclude_region"}
    ]


def _direction_summary(
    valid: np.ndarray,
    ranges: np.ndarray,
    confidence: np.ndarray,
    input_validity: np.ndarray,
    pose: PanoramicStereoPose,
) -> dict[str, Any]:
    return {
        "scale_status": pose.scale_status,
        "units": pose.units,
        "valid_pixel_count": int(valid.sum()),
        "valid_fraction": float(valid.mean()),
        "valid_outside_input_count": int(np.count_nonzero(valid & ~input_validity)),
        "median_range": float(np.median(ranges[valid])) if np.any(valid) else None,
        "median_confidence": (
            float(np.median(confidence[valid])) if np.any(valid) else None
        ),
        "range_min": float(np.min(ranges[valid])) if np.any(valid) else None,
        "range_max": float(np.max(ranges[valid])) if np.any(valid) else None,
        "pose_provenance": pose.provenance,
    }


def _reference_validity(
    project, reference_id: str, source_shape: tuple[int, int]
) -> tuple[np.ndarray, dict[str, Any]]:
    annotations = [
        item
        for item in project.dense_annotations
        if item.image_id == reference_id
        and item.state == "accepted"
        and item.kind in {"include_region", "exclude_region"}
    ]
    includes = [item for item in annotations if item.kind == "include_region"]
    excludes = [item for item in annotations if item.kind == "exclude_region"]
    validity = (
        np.zeros(_PREVIEW_SHAPE, dtype=bool)
        if includes
        else np.ones(_PREVIEW_SHAPE, dtype=bool)
    )
    for item in includes:
        _paint_rectangle(validity, item.points_xy, source_shape, True)
    for item in excludes:
        _paint_rectangle(validity, item.points_xy, source_shape, False)
    return validity, {
        "eligible_pixel_count": int(validity.sum()),
        "eligible_fraction": float(validity.mean()),
        "include_annotation_ids": [item.id for item in includes],
        "exclude_annotation_ids": [item.id for item in excludes],
        "semantics": "explicit-reference-validity",
    }


def _paint_rectangle(
    mask: np.ndarray,
    points: list[tuple[float, float]],
    source_shape: tuple[int, int],
    value: bool,
) -> None:
    height, width = source_shape
    xs = sorted(point[0] / width * mask.shape[1] for point in points)
    ys = sorted(point[1] / height * mask.shape[0] for point in points)
    x0 = max(0, min(mask.shape[1] - 1, int(np.floor(xs[0]))))
    x1 = max(x0 + 1, min(mask.shape[1], int(np.ceil(xs[-1]))))
    y0 = max(0, min(mask.shape[0] - 1, int(np.floor(ys[0]))))
    y1 = max(y0 + 1, min(mask.shape[0], int(np.ceil(ys[-1]))))
    mask[y0:y1, x0:x1] = value
