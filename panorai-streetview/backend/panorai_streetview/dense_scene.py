"""Metric multiview reconstruction through public PanorAi scene contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
from PIL import Image

from panorai.applications import SphericalDenseReconstruction
from panorai.data import EquirectangularImage
from panorai.reconstruction import (
    RegisteredSphericalScene,
    RegisteredSphericalView,
    SphericalCameraPose,
    SphericalDenseReconstructionResult,
)
from panorai.stereo import SphericalStereoOptions
from panorai.stereo.panoramic import (
    FullSphereStereoOptions,
    OverlappingCubemapPlanner,
)

from .dense_preview import _annotation_signature, _reference_validity
from .global_map import GlobalMapService
from .models import utc_now
from .store import ProjectStore, SAFE_ID


_SCENE_SHAPE = (128, 256)
_DISPLAY_VERTEX_LIMIT = 1_200


class DenseSceneService:
    """Persist one auditable public dense-scene result per project."""

    def __init__(
        self,
        store: ProjectStore,
        global_map: GlobalMapService,
        application_factory: (
            Callable[[RegisteredSphericalScene, str], Any] | None
        ) = None,
    ) -> None:
        self.store = store
        self.global_map = global_map
        self.directory = store.root / "dense-scenes"
        self.application_factory = application_factory or _dense_application

    def _path(self, project_id: str, suffix: str) -> Path:
        if not SAFE_ID.fullmatch(project_id):
            raise KeyError(project_id)
        return self.directory / f"{project_id}.{suffix}"

    def load(self, project_id: str) -> dict[str, Any]:
        project = self.store.load(project_id)
        path = self._path(project_id, "json")
        if not path.is_file():
            raise KeyError(project_id)
        record = json.loads(path.read_text(encoding="utf-8"))
        map_record = self.global_map.load_optional(project_id)
        mapped_image_ids = (
            tuple(item["image_id"] for item in map_record.get("poses", ()))
            if map_record is not None
            else ()
        )
        record["stale"] = (
            map_record is None
            or map_record.get("stale", True)
            or record.get("global_map_revision") != map_record.get("revision")
            or record.get("global_map_created_at") != map_record.get("created_at")
            or record.get("annotation_signature")
            != _annotation_signature(project, mapped_image_ids)
        )
        return record

    def load_optional(self, project_id: str) -> dict[str, Any] | None:
        try:
            return self.load(project_id)
        except KeyError:
            return None

    def mesh_path(self, project_id: str, *, revision: int | None = None) -> Path:
        record = self.load(project_id)
        if revision is not None and int(record["revision"]) != revision:
            raise ValueError("dense scene revision changed; reload before download")
        if not record.get("surface", {}).get("file_available"):
            raise KeyError(project_id)
        expected_name = f"{project_id}.revision-{int(record['revision']):04d}.ply"
        filename = record["surface"].get("file_name", f"{project_id}.ply")
        if "file_name" in record["surface"] and filename != expected_name:
            raise KeyError(project_id)
        path = self._path(project_id, "ply").with_name(filename)
        if not path.is_file():
            raise KeyError(project_id)
        return path

    def run(self, project_id: str, *, profile: str = "interactive") -> dict[str, Any]:
        with self.store.project_lock(project_id):
            return self._run(project_id, profile=profile)

    def _run(self, project_id: str, *, profile: str) -> dict[str, Any]:
        project = self.store.load(project_id)
        map_record = self.global_map.load(project_id)
        _validate_metric_map(map_record)
        scene, mask_records = self._scene(project, map_record)
        application = self.application_factory(scene, profile)
        result = application.run(scene)
        if not isinstance(result, SphericalDenseReconstructionResult):
            raise RuntimeError(
                "SphericalDenseReconstruction returned an unexpected result"
            )
        previous = self.load_optional(project_id)
        surface_record = _surface_summary(result)
        record = {
            "schema": "panorai-streetview-dense-scene/v1",
            "project_id": project_id,
            "revision": 1 if previous is None else int(previous["revision"]) + 1,
            "created_at": utc_now(),
            "stale": False,
            "profile": profile,
            "engine": _jsonable(application.describe()),
            "global_map_revision": map_record["revision"],
            "global_map_created_at": map_record.get("created_at"),
            "global_map_scale": map_record["scale"],
            "scene": {
                "units": scene.units,
                "frame": scene.frame,
                "view_count": len(scene.views),
                "view_ids": list(scene.panorama_ids),
                "shape_hw": list(_SCENE_SHAPE),
            },
            "annotation_signature": _annotation_signature(project, scene.panorama_ids),
            "view_masks": mask_records,
            "status": result.status,
            "accepted": result.accepted,
            "reasons": list(result.reasons),
            "range_maps": _range_summaries(result),
            "surface": surface_record,
            "diagnostics": _jsonable(dict(result.diagnostics)),
        }
        self._write(record, result)
        return record

    def _scene(self, project, map_record):
        image_by_id = {item.id: item for item in project.images}
        views = []
        masks = []
        for pose_record in map_record["poses"]:
            image_id = pose_record["image_id"]
            image_record = image_by_id.get(image_id)
            if image_record is None:
                raise ValueError(f"global map references unknown image {image_id!r}")
            pixels = _scene_image(self.store.image_path(project.id, image_id))
            validity, details = _reference_validity(
                project,
                image_id,
                (image_record.height, image_record.width),
            )
            image = EquirectangularImage(pixels, valid=validity)
            pose = SphericalCameraPose(
                panorama_id=image_id,
                rotation_world_to_panorama=np.asarray(
                    pose_record["rotation_world_to_panorama"], dtype=np.float64
                ),
                center_world=np.asarray(pose_record["center_world"], dtype=np.float64),
            )
            views.append(
                RegisteredSphericalView(
                    image=image,
                    pose=pose,
                    validity=validity,
                )
            )
            masks.append({"image_id": image_id, **details})
        scene = RegisteredSphericalScene(
            views=tuple(views),
            units="m",
            frame=f"panorai-global-map-revision-{map_record['revision']}",
        )
        return scene, masks

    def _write(
        self,
        record: dict[str, Any],
        result: SphericalDenseReconstructionResult,
    ) -> None:
        json_path = self._path(record["project_id"], "json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        if result.accepted and result.surface is not None:
            mesh_path = self._path(
                record["project_id"],
                f"revision-{int(record['revision']):04d}.ply",
            )
            record["surface"]["file_name"] = mesh_path.name
            with NamedTemporaryFile(
                "w", encoding="ascii", dir=json_path.parent, delete=False
            ) as handle:
                _write_ply(handle, result.surface)
                temporary_mesh = Path(handle.name)
            temporary_mesh.replace(mesh_path)
        payload = json.dumps(record, indent=2, allow_nan=False) + "\n"
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=json_path.parent, delete=False
        ) as handle:
            handle.write(payload)
            temporary = Path(handle.name)
        temporary.replace(json_path)


def _validate_metric_map(record: dict[str, Any]) -> None:
    if not record.get("success") or record.get("stale"):
        raise ValueError("dense scene requires a current successful global map")
    if not str(record.get("scale", "")).startswith("metric"):
        raise ValueError("dense scene requires metric global-map scale")
    if len(record.get("poses", ())) < 3:
        raise ValueError("dense scene requires at least three global-map poses")


def _scene_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        resized = image.convert("RGB").resize(
            (_SCENE_SHAPE[1], _SCENE_SHAPE[0]), Image.Resampling.LANCZOS
        )
        return np.asarray(resized, dtype=np.float32) / 255.0


def _scene_scale(scene: RegisteredSphericalScene) -> float:
    centers = np.stack([item.pose.center for item in scene.views])
    distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)
    positive = distances[distances > 1e-9]
    if not len(positive):
        raise ValueError("dense scene camera centres must be distinct")
    return float(np.median(positive))


def _dense_application(
    scene: RegisteredSphericalScene, profile: str
) -> SphericalDenseReconstruction:
    baseline = _scene_scale(scene)
    voxel_size = float(np.clip(baseline / 20.0, 0.04, 0.25))
    if profile == "reviewed":
        return SphericalDenseReconstruction.reviewed_profile(
            voxel_size=voxel_size,
            filter_backend="auto",
        )
    if profile != "interactive":
        raise ValueError("dense profile must be 'interactive' or 'reviewed'")
    source_requirement = min(2, len(scene.views) - 1)
    options = FullSphereStereoOptions(
        planner=OverlappingCubemapPlanner(face_size=48, face_fov_deg=100.0),
        stereo=SphericalStereoOptions(
            min_range=max(0.20, baseline * 0.10),
            max_range=max(4.0, baseline * 5.0),
            num_hypotheses=24,
            window_size=5,
            pole_margin_fraction=0.0,
            min_texture_std=0.005,
            min_confidence=0.01,
            max_matching_cost=0.75,
            bidirectional_consistency=False,
            filter_backend="numpy",
            pyramid_levels=1,
        ),
        min_parallax_deg=0.10,
        min_source_evidence=source_requirement,
        source_aggregation="median",
        matching_cost="intensity-gradient",
        local_aggregation="none",
        regularizer="none",
        erp_feature_tap_deg=None,
        face_cost_tap_deg=None,
        min_range_consensus_sources=source_requirement,
    )
    return SphericalDenseReconstruction(
        voxel_size=voxel_size,
        mvs_options=options,
        max_voxels=4_000_000,
        min_surface_views=2,
        min_valid_weight=1.0,
        weighting="uniform",
        colorize=True,
        retain_intermediates="ranges",
    )


def _range_summaries(
    result: SphericalDenseReconstructionResult,
) -> list[dict[str, Any]]:
    if result.ranges is None:
        return []
    records = []
    for item in result.ranges.maps:
        valid = np.asarray(item.validity, dtype=bool)
        radial = np.asarray(item.radial_range)
        confidence = np.asarray(item.confidence)
        records.append(
            {
                "image_id": item.reference_id,
                "valid_pixel_count": int(valid.sum()),
                "valid_fraction": float(valid.mean()),
                "median_range_m": (
                    float(np.median(radial[valid])) if np.any(valid) else None
                ),
                "median_confidence": (
                    float(np.median(confidence[valid])) if np.any(valid) else None
                ),
                "source_ids": list(item.source_ids),
            }
        )
    return records


def _surface_summary(result: SphericalDenseReconstructionResult) -> dict[str, Any]:
    surface = result.surface
    if not result.accepted or surface is None:
        return {
            "file_available": False,
            "vertex_count": 0,
            "face_count": 0,
            "bounds_xyz": None,
            "colored_vertex_fraction": 0.0,
            "display_xy": [],
            "display_rgb": [],
        }
    vertices = np.asarray(surface.vertices_xyz)
    indices = np.linspace(
        0,
        max(0, len(vertices) - 1),
        min(_DISPLAY_VERTEX_LIMIT, len(vertices)),
        dtype=int,
    )
    sampled = vertices[indices] if len(indices) else np.empty((0, 3))
    display = _project_vertices(sampled)
    colors = (
        np.asarray(surface.colors_rgb)[indices]
        if surface.colors_rgb is not None and len(indices)
        else np.full((len(indices), 3), 0.7, dtype=np.float32)
    )
    colored_fraction = (
        float(np.asarray(surface.color_validity, dtype=bool).mean())
        if surface.color_validity is not None and len(surface.color_validity)
        else 0.0
    )
    return {
        "file_available": True,
        "vertex_count": int(len(vertices)),
        "face_count": int(len(surface.faces)),
        "bounds_xyz": [
            vertices.min(axis=0).astype(float).tolist(),
            vertices.max(axis=0).astype(float).tolist(),
        ],
        "colored_vertex_fraction": colored_fraction,
        "display_xy": display.astype(float).tolist(),
        "display_rgb": np.clip(colors * 255.0, 0, 255).astype(int).tolist(),
    }


def _project_vertices(vertices: np.ndarray) -> np.ndarray:
    if not len(vertices):
        return np.empty((0, 2), dtype=np.float64)
    centered = vertices - vertices.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ axes[:2].T
    scale = float(np.max(np.linalg.norm(projected, axis=1)))
    return projected / scale if scale > 0.0 else projected


def _write_ply(handle, surface) -> None:
    vertices = np.asarray(surface.vertices_xyz)
    faces = np.asarray(surface.faces)
    colors = (
        np.clip(np.asarray(surface.colors_rgb) * 255.0, 0, 255).astype(np.uint8)
        if surface.colors_rgb is not None
        else np.full((len(vertices), 3), 180, dtype=np.uint8)
    )
    handle.write("ply\nformat ascii 1.0\n")
    handle.write(f"element vertex {len(vertices)}\n")
    handle.write("property float x\nproperty float y\nproperty float z\n")
    handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
    handle.write(f"element face {len(faces)}\n")
    handle.write("property list uchar int vertex_indices\nend_header\n")
    for point, color in zip(vertices, colors, strict=True):
        handle.write(
            f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
            f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
        )
    for face in faces:
        handle.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
