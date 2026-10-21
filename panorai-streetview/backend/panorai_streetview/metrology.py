"""Adapter from Street View records to PanorAi single-view metrology."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock, RLock
from typing import Any

import numpy as np

from panorai.experimental.single_view_metrology import (
    FloorCalibration,
    DirectionFamilyEstimate,
    SphericalLineObservation,
    calibrate_floor,
    measure_floor_point,
    measure_vertical,
)

from .models import (
    FloorCalibrationRecord,
    FloorCalibrationRequest,
    MetrologyFamilyRecord,
    utc_now,
)
from .store import ProjectStore, SAFE_ID


class CalibrationRevisionConflict(ValueError):
    """The client measured against a calibration that is no longer current."""


class MetrologyService:
    def __init__(self, store: ProjectStore) -> None:
        self.store = store
        self.directory = store.root / "metrology"
        self.directory.mkdir(parents=True, exist_ok=True)
        self._image_locks: dict[tuple[str, str], Any] = {}
        self._image_locks_guard = Lock()

    def image_lock(self, project_id: str, image_id: str):
        """Return the re-entrant lock for one image's immutable revisions."""

        if not SAFE_ID.fullmatch(project_id) or not SAFE_ID.fullmatch(image_id):
            raise KeyError("invalid project or image ID")
        key = (project_id, image_id)
        with self._image_locks_guard:
            return self._image_locks.setdefault(key, RLock())

    def _paths(self, project_id: str, image_id: str) -> list[Path]:
        if not SAFE_ID.fullmatch(project_id) or not SAFE_ID.fullmatch(image_id):
            raise KeyError("invalid project or image ID")
        return sorted(self.directory.glob(f"{project_id}.{image_id}.r*.json"))

    def _write(self, record: FloorCalibrationRecord) -> None:
        target = self.directory / (
            f"{record.project_id}.{record.image_id}.r{record.revision:04d}.json"
        )
        if target.exists():
            raise RuntimeError("floor calibration revisions are immutable")
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.directory, delete=False
        ) as handle:
            handle.write(record.model_dump_json(indent=2, by_alias=True))
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(target)

    def load_record(self, project_id: str, image_id: str) -> FloorCalibrationRecord:
        self._validate_image(project_id, image_id)
        paths = self._paths(project_id, image_id)
        if not paths:
            raise KeyError("floor calibration has not been run")
        return max(
            (
                FloorCalibrationRecord.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
                for path in paths
            ),
            key=lambda item: item.revision,
        )

    def _validate_image(self, project_id: str, image_id: str):
        project = self.store.load(project_id)
        if project.workflow not in {
            "spherical_registered",
            "spherical_unregistered",
        }:
            raise ValueError("single-view metrology requires a spherical ERP workflow")
        image = next((item for item in project.images if item.id == image_id), None)
        if image is None:
            raise KeyError(image_id)
        return project, image

    @staticmethod
    def _validate_point(image, point: tuple[float, float], label: str) -> None:
        if not (0 <= point[0] < image.width and 0 <= point[1] < image.height):
            raise ValueError(f"{label} must lie inside the source image")

    def calibrate(
        self, project_id: str, request: FloorCalibrationRequest
    ) -> dict[str, Any]:
        with self.image_lock(project_id, request.image_id):
            project, image = self._validate_image(project_id, request.image_id)
            self._validate_point(image, request.floor_hint_xy, "floor hint")
            for index, line in enumerate(request.lines, start=1):
                self._validate_point(
                    image, line.start_xy, f"metrology line {index} start"
                )
                self._validate_point(image, line.end_xy, f"metrology line {index} end")
            calibration = calibrate_floor(
                [
                    SphericalLineObservation(item.family, item.start_xy, item.end_xy)
                    for item in request.lines
                ],
                (image.height, image.width),
                family_a="floor_a",
                family_b="floor_b",
                camera_height_m=request.camera_height_m,
                floor_hint_xy=request.floor_hint_xy,
                max_residual_deg=request.max_residual_deg,
            )
            paths = self._paths(project_id, image.id)
            previous_revisions = [
                FloorCalibrationRecord.model_validate_json(
                    path.read_text(encoding="utf-8")
                ).revision
                for path in paths
            ]
            revision = max(previous_revisions, default=0) + 1
            record = self._record(
                calibration,
                project_id=project_id,
                image_id=image.id,
                revision=revision,
                request=request,
                engine=project.panorai_runtime,
            )
            self._write(record)
            return record.model_dump(mode="json", by_alias=True)

    def load(
        self, project_id: str, image_id: str, *, revision: int | None = None
    ) -> FloorCalibration:
        record = self.load_record(project_id, image_id)
        if revision is not None and record.revision != revision:
            raise CalibrationRevisionConflict(
                "floor calibration revision changed; reload measurements"
            )
        families = tuple(
            DirectionFamilyEstimate(
                family=item.family,
                direction_panorama=np.asarray(
                    item.direction_panorama, dtype=np.float64
                ),
                inlier_indices=tuple(item.inlier_indices),
                outlier_indices=tuple(item.outlier_indices),
                residuals_deg=tuple(item.residuals_deg),
                median_residual_deg=item.median_residual_deg,
                p90_residual_deg=item.p90_residual_deg,
            )
            for item in record.families
        )
        if len(families) != 2:
            raise ValueError("stored floor calibration requires two direction families")
        return FloorCalibration(
            shape_hw=record.shape_hw,
            rotation_panorama_to_world=np.asarray(
                record.rotation_panorama_to_world, dtype=np.float64
            ),
            camera_height_m=record.camera_height_m,
            floor_normal_panorama=np.asarray(
                record.floor_normal_panorama, dtype=np.float64
            ),
            direction_families=(families[0], families[1]),
            floor_hint_xy=record.floor_hint_xy,
            family_separation_deg=record.family_separation_deg,
            floor_normal_uncertainty_deg=record.floor_normal_uncertainty_deg,
        )

    @staticmethod
    def _record(
        calibration: FloorCalibration,
        *,
        project_id: str,
        image_id: str,
        revision: int,
        request: FloorCalibrationRequest,
        engine: dict[str, Any],
    ) -> FloorCalibrationRecord:
        return FloorCalibrationRecord(
            project_id=project_id,
            image_id=image_id,
            revision=revision,
            created_at=utc_now(),
            engine={
                **engine,
                "public_api": "panorai.experimental.single_view_metrology",
            },
            shape_hw=calibration.shape_hw,
            camera_height_m=calibration.camera_height_m,
            floor_hint_xy=calibration.floor_hint_xy,
            lines=request.lines,
            rotation_panorama_to_world=calibration.rotation_panorama_to_world.tolist(),
            floor_normal_panorama=tuple(
                float(value) for value in calibration.floor_normal_panorama
            ),
            family_separation_deg=calibration.family_separation_deg,
            floor_normal_uncertainty_deg=calibration.floor_normal_uncertainty_deg,
            families=[
                MetrologyFamilyRecord(
                    family=item.family,
                    direction_panorama=tuple(
                        float(value) for value in item.direction_panorama
                    ),
                    inlier_indices=list(item.inlier_indices),
                    outlier_indices=list(item.outlier_indices),
                    residuals_deg=list(item.residuals_deg),
                    median_residual_deg=item.median_residual_deg,
                    p90_residual_deg=item.p90_residual_deg,
                )
                for item in calibration.direction_families
            ],
        )

    def history(self, project_id: str, image_id: str) -> dict[str, Any]:
        self._validate_image(project_id, image_id)
        revisions = [
            FloorCalibrationRecord.model_validate_json(path.read_text(encoding="utf-8"))
            for path in self._paths(project_id, image_id)
        ]
        revisions.sort(key=lambda item: item.revision)
        return {
            "schema": "panorai-streetview-floor-calibration-history/v1",
            "project_id": project_id,
            "image_id": image_id,
            "revisions": [
                {
                    "revision": item.revision,
                    "created_at": item.created_at,
                    "camera_height_m": item.camera_height_m,
                    "floor_normal_uncertainty_deg": item.floor_normal_uncertainty_deg,
                    "family_separation_deg": item.family_separation_deg,
                }
                for item in revisions
            ],
        }

    def floor_point(
        self,
        project_id: str,
        image_id: str,
        xy: tuple[float, float],
        *,
        calibration_revision: int,
    ) -> dict[str, Any]:
        _, image = self._validate_image(project_id, image_id)
        self._validate_point(image, xy, "floor point")
        result = measure_floor_point(
            self.load(project_id, image_id, revision=calibration_revision), xy
        )
        return {
            "calibration_revision": calibration_revision,
            "point_world_m": result.point_world_m.tolist(),
            "horizontal_distance_m": result.horizontal_distance_m,
            "radial_distance_m": result.radial_distance_m,
        }

    def vertical(
        self,
        project_id: str,
        image_id: str,
        base: tuple[float, float],
        top: tuple[float, float],
        *,
        calibration_revision: int,
        max_horizontal_residual_m: float | None = 0.15,
    ) -> dict[str, Any]:
        _, image = self._validate_image(project_id, image_id)
        self._validate_point(image, base, "vertical base")
        self._validate_point(image, top, "vertical top")
        result = measure_vertical(
            self.load(project_id, image_id, revision=calibration_revision),
            base,
            top,
            max_horizontal_residual_m=max_horizontal_residual_m,
        )
        return {
            "calibration_revision": calibration_revision,
            "base_world_m": result.base_world_m.tolist(),
            "top_world_m": result.top_world_m.tolist(),
            "height_m": result.height_m,
            "horizontal_residual_m": result.horizontal_residual_m,
        }
