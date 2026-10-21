"""Auditable conventional-image to ERP conversion through public PanorAi APIs."""

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

from panorai.applications import PanoramaStitcher
from panorai.stitching import (
    PanoramaStitchingResult,
    StitchingOptions,
    StitchingRegistrationError,
)

from .models import ImagePairInput, ProjectRecord, utc_now
from .store import ProjectStore, SAFE_ID


_INPUT_MAX_EDGE = 960
_OUTPUT_SHAPE = (256, 512)


class StitchingService:
    """Persist public rotation-only stitching results for capture groups."""

    def __init__(
        self,
        store: ProjectStore,
        application_factory: Callable[[StitchingOptions], Any] | None = None,
    ) -> None:
        self.store = store
        self.directory = store.root / "stitching"
        self.application_factory = application_factory or PanoramaStitcher

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
        if record.get("schema") == "panorai-streetview-stitching/v1":
            record = _upgrade_v1(record)
        record["stale"] = record.get("input_signature") != _input_signature(project)
        return record

    def load_optional(self, project_id: str) -> dict[str, Any] | None:
        try:
            return self.load(project_id)
        except KeyError:
            return None

    def image_path(
        self,
        project_id: str,
        group_id: str = "capture-1",
        *,
        revision: int | None = None,
    ) -> Path:
        record = self.load(project_id)
        if revision is not None and int(record["revision"]) != revision:
            raise ValueError("stitching revision changed; reload before download")
        group = next(
            (item for item in record["groups"] if item["group_id"] == group_id),
            None,
        )
        if group is None or not group.get("accepted"):
            raise KeyError(project_id)
        legacy_path = self._path(project_id, f"{group_id}.png")
        filename = (group.get("output") or {}).get("file_name")
        if filename is not None:
            expected = self._path(
                project_id,
                f"{group_id}.revision-{int(record['revision']):04d}.png",
            )
            if filename != expected.name:
                raise KeyError(project_id)
            path = expected
        else:
            path = legacy_path
        if not path.is_file() and filename is None and group_id == "capture-1":
            path = self._path(project_id, "png")
        if not path.is_file():
            raise KeyError(project_id)
        return path

    def run(self, project_id: str) -> dict[str, Any]:
        with self.store.project_lock(project_id):
            return self._run(project_id)

    def _run(self, project_id: str) -> dict[str, Any]:
        project = self.store.load(project_id)
        if project.workflow not in {
            "perspective_calibrated",
            "perspective_uncalibrated",
        }:
            raise ValueError("stitching requires a perspective-image project")
        options = StitchingOptions(
            feature_method="sift",
            max_features=2_500,
            ratio_test=0.8,
            min_pair_matches=12,
            min_pair_inliers=8,
            max_transfer_error_px=4.0,
            max_angular_error_deg=1.5,
            output_shape_hw=_OUTPUT_SHAPE,
            random_seed=17,
        )
        grouped = _capture_groups(project)
        previous = self.load_optional(project_id)
        group_records = []
        results: dict[str, PanoramaStitchingResult | None] = {}
        for group_id, images in grouped:
            group_record, result = self._run_group(project, group_id, images, options)
            group_records.append(group_record)
            results[group_id] = result
        accepted_count = sum(bool(item["accepted"]) for item in group_records)
        rejected_count = len(group_records) - accepted_count
        record = {
            "schema": "panorai-streetview-stitching/v2",
            "project_id": project_id,
            "revision": 1 if previous is None else int(previous["revision"]) + 1,
            "created_at": utc_now(),
            "stale": False,
            "workflow": project.workflow,
            "model": "central-rotation-only",
            "engine": group_records[0]["engine"],
            "input_signature": _input_signature(project),
            "status": (
                "accepted"
                if rejected_count == 0
                else "partial"
                if accepted_count
                else "rejected"
            ),
            "accepted_group_count": accepted_count,
            "rejected_group_count": rejected_count,
            "ready_for_promotion": accepted_count >= 2,
            "promoted_project_id": None,
            "groups": group_records,
        }
        self._write(record, results)
        return record

    def _run_group(self, project, group_id, images, options):
        arrays = tuple(
            _load_source(self.store.image_path(project.id, item.id)) for item in images
        )
        camera_inputs = tuple(_camera_input(item) for item in images)
        fovs = tuple(item[0] for item in camera_inputs)
        camera_errors = tuple(item[1] for item in camera_inputs if item[1] is not None)
        application = self.application_factory(options)
        base = {
            "group_id": group_id,
            "engine": _jsonable(application.describe()),
            "sources": [
                {
                    "image_id": item.id,
                    "filename": item.filename,
                    "sha256": item.sha256,
                    "shape_hw": [item.height, item.width],
                    "working_shape_hw": list(array.shape[:2]),
                    "horizontal_fov_deg": fov,
                    "fov_source": "supplied" if fov is not None else "estimated",
                }
                for item, array, fov in zip(images, arrays, fovs, strict=True)
            ],
        }
        if len(images) < 2:
            return _rejected_group(
                base, "capture group requires at least two source images"
            ), None
        if camera_errors:
            return _rejected_group(base, "; ".join(camera_errors)), None
        if project.workflow == "perspective_calibrated" and any(
            value is None for value in fovs
        ):
            missing = [
                item.filename
                for item, value in zip(images, fovs, strict=True)
                if value is None
            ]
            return _rejected_group(
                base,
                "calibrated perspective images require horizontal FOV or fx: "
                + ", ".join(missing),
            ), None
        try:
            result = application.run(
                arrays,
                image_ids=tuple(item.id for item in images),
                horizontal_fov_deg=fovs,
            )
            if not isinstance(result, PanoramaStitchingResult):
                raise RuntimeError("PanoramaStitcher returned an unexpected result")
            return {**base, **_accepted_result(result)}, result
        except (StitchingRegistrationError, ValueError) as exc:
            return _rejected_group(base, str(exc)), None

    def promote(self, project_id: str) -> ProjectRecord:
        with self.store.project_lock(project_id):
            return self._promote(project_id)

    def _promote(self, project_id: str) -> ProjectRecord:
        source = self.store.load(project_id)
        record = self.load(project_id)
        if record["stale"]:
            raise ValueError("stitching result is stale; rerun before promotion")
        if record.get("promoted_project_id"):
            try:
                return self.store.load(record["promoted_project_id"])
            except KeyError:
                pass
        accepted = [item for item in record["groups"] if item["accepted"]]
        if len(accepted) < 2:
            raise ValueError("promotion requires at least two accepted capture groups")
        filenames = [f"{item['group_id']}.png" for item in accepted]
        uploads = [
            (
                filename,
                "image/png",
                self.image_path(project_id, item["group_id"]).read_bytes(),
            )
            for filename, item in zip(filenames, accepted, strict=True)
        ]
        metadata = {
            filename: {
                "projection": "equirectangular",
                "source_perspective_project_id": project_id,
                "source_stitching_revision": record["revision"],
                "source_capture_group": item["group_id"],
                "pose_status": "unknown",
            }
            for filename, item in zip(filenames, accepted, strict=True)
        }
        pair_inputs = []
        pair_index = 1
        for left in range(len(filenames)):
            for right in range(left + 1, len(filenames)):
                pair_inputs.append(
                    ImagePairInput(
                        id=f"candidate-{pair_index:02d}",
                        reference_filename=filenames[left],
                        source_filename=filenames[right],
                        provenance={
                            "source": "panorai-streetview-stitching-promotion",
                            "pose_status": "unknown",
                            "stitching_revision": record["revision"],
                        },
                    )
                )
                pair_index += 1
        name_suffix = " · ERPs"
        promoted_name = f"{source.name[: 120 - len(name_suffix)].rstrip()}{name_suffix}"
        promoted = self.store.create(
            name=promoted_name,
            workflow="spherical_unregistered",
            uploads=uploads,
            metadata=metadata,
            pair_inputs=pair_inputs,
        )
        record["promoted_project_id"] = promoted.id
        record["promoted_at"] = utc_now()
        self._write_record(record)
        return promoted

    def _write(
        self,
        record: dict[str, Any],
        results: dict[str, PanoramaStitchingResult | None],
    ) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        for group_id, result in results.items():
            if result is None:
                continue
            image_path = self._path(
                record["project_id"],
                f"{group_id}.revision-{int(record['revision']):04d}.png",
            )
            _save_panorama(result, image_path)
            group = next(
                item for item in record["groups"] if item["group_id"] == group_id
            )
            group["output"]["file_name"] = image_path.name
        self._write_record(record)

    def _write_record(self, record: dict[str, Any]) -> None:
        path = self._path(record["project_id"], "json")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(record, indent=2, allow_nan=False) + "\n"
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            handle.write(payload)
            temporary = Path(handle.name)
        temporary.replace(path)


def _capture_groups(project: ProjectRecord) -> list[tuple[str, list[Any]]]:
    explicit = any(
        "capture_group" in item.metadata or "group_id" in item.metadata
        for item in project.images
    )
    groups: dict[str, list[Any]] = {}
    for item in project.images:
        raw = item.metadata.get("capture_group", item.metadata.get("group_id"))
        if explicit and raw is None:
            raise ValueError(
                f"{item.filename} is missing capture_group while other images define it"
            )
        group_id = "capture-1" if raw is None else str(raw).strip()
        if not SAFE_ID.fullmatch(group_id):
            raise ValueError(
                "capture_group must use only letters, digits, hyphen or underscore"
            )
        groups.setdefault(group_id, []).append(item)
    if not groups:
        raise ValueError("stitching requires at least one capture group")
    return list(groups.items())


def _rejected_group(base: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        **base,
        "status": "rejected",
        "accepted": False,
        "reasons": [reason],
        "output": None,
        "diagnostics": None,
        "intrinsics": [],
        "orientations": [],
        "pairwise": [],
    }


def _upgrade_v1(record: dict[str, Any]) -> dict[str, Any]:
    group = {
        "group_id": "capture-1",
        "engine": record.get("engine", {}),
        "sources": record.get("sources", []),
        "status": record.get("status", "rejected"),
        "accepted": bool(record.get("accepted")),
        "reasons": record.get("reasons", []),
        "output": record.get("output"),
        "diagnostics": record.get("diagnostics"),
        "intrinsics": record.get("intrinsics", []),
        "orientations": record.get("orientations", []),
        "pairwise": record.get("pairwise", []),
    }
    signature = [
        {**item, "capture_group": "capture-1"}
        for item in record.get("input_signature", [])
    ]
    accepted_count = int(group["accepted"])
    return {
        "schema": "panorai-streetview-stitching/v2",
        "project_id": record["project_id"],
        "revision": record["revision"],
        "created_at": record["created_at"],
        "stale": record.get("stale", False),
        "workflow": record["workflow"],
        "model": record.get("model", "central-rotation-only"),
        "engine": record.get("engine", {}),
        "input_signature": signature,
        "status": "accepted" if accepted_count else "rejected",
        "accepted_group_count": accepted_count,
        "rejected_group_count": 1 - accepted_count,
        "ready_for_promotion": False,
        "promoted_project_id": None,
        "groups": [group],
    }


def _load_source(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        converted = image.convert("RGB")
        scale = min(1.0, _INPUT_MAX_EDGE / max(converted.size))
        if scale < 1.0:
            converted = converted.resize(
                (
                    max(2, round(converted.width * scale)),
                    max(2, round(converted.height * scale)),
                ),
                Image.Resampling.LANCZOS,
            )
        return np.asarray(converted, dtype=np.uint8)


def _horizontal_fov(image) -> float | None:
    metadata = image.metadata
    distortion = metadata.get("distortion")
    if distortion not in {None, "none"}:
        raise ValueError(
            f"{image.filename} uses unsupported distortion {distortion!r}; "
            "PanoramaStitcher requires centred pinhole inputs"
        )
    intrinsics = metadata.get("intrinsics", {})
    if intrinsics is None:
        intrinsics = {}
    if not isinstance(intrinsics, dict):
        raise ValueError(f"intrinsics for {image.filename} must be an object")
    value = metadata.get("horizontal_fov_deg", intrinsics.get("horizontal_fov_deg"))
    if value is None and intrinsics.get("fx") is not None:
        fx = float(intrinsics["fx"])
        if not math.isfinite(fx) or fx <= 0.0:
            raise ValueError(f"fx for {image.filename} must be positive and finite")
        value = math.degrees(2.0 * math.atan(image.width / (2.0 * fx)))
    if value is None:
        return None
    fov = float(value)
    if not math.isfinite(fov) or not 1.0 < fov < 179.0:
        raise ValueError(
            f"horizontal FOV for {image.filename} must be in (1, 179) degrees"
        )
    return fov


def _input_signature(project: ProjectRecord) -> list[dict[str, Any]]:
    signature = []
    for item in project.images:
        horizontal_fov, camera_error = _camera_input(item)
        signature.append(
            {
                "image_id": item.id,
                "sha256": item.sha256,
                "horizontal_fov_deg": horizontal_fov,
                "camera_error": camera_error,
                "capture_group": str(
                    item.metadata.get(
                        "capture_group", item.metadata.get("group_id", "capture-1")
                    )
                ),
            }
        )
    return signature


def _camera_input(image) -> tuple[float | None, str | None]:
    try:
        return _horizontal_fov(image), None
    except (OverflowError, TypeError, ValueError) as exc:
        return None, str(exc)


def _accepted_result(result: PanoramaStitchingResult) -> dict[str, Any]:
    panorama = result.panorama
    support = np.asarray(panorama.support_mask, dtype=bool)
    validity = np.asarray(panorama.validity("image"), dtype=bool)
    registered = result.registered
    return {
        "status": "accepted",
        "accepted": True,
        "reasons": [],
        "output": {
            "shape_hw": list(panorama.shape[:2]),
            "support_fraction": float(support.mean()),
            "valid_fraction": float(validity.mean()),
            "overlap_pixel_fraction": float(
                np.asarray(result.support_count >= 2, dtype=bool).mean()
            ),
            "file_available": True,
        },
        "diagnostics": _jsonable(asdict(result.diagnostics)),
        "intrinsics": [item.to_dict() for item in registered.intrinsics],
        "orientations": [
            {
                "image_id": source.image_id,
                "camera_to_world": np.asarray(rotation).tolist(),
            }
            for source, rotation in zip(
                registered.source.images,
                registered.camera_to_world,
                strict=True,
            )
        ],
        "pairwise": [
            {
                "image_a_id": registered.source.images[item.index_a].image_id,
                "image_b_id": registered.source.images[item.index_b].image_id,
                "num_inliers": item.num_inliers,
                "median_angular_error_deg": (item.calibration.median_angular_error_deg),
                "estimated_fov": item.calibration.estimated_fov,
                "observable": item.calibration.observable,
                "spatial_coverage": item.calibration.spatial_coverage,
            }
            for item in registered.pairwise
        ],
    }


def _save_panorama(result: PanoramaStitchingResult, path: Path) -> None:
    pixels = np.asarray(result.panorama.data)
    if np.issubdtype(pixels.dtype, np.floating):
        maximum = float(np.nanmax(pixels)) if pixels.size else 0.0
        scale = 255.0 if maximum <= 1.0 + 1e-6 else 1.0
        pixels = np.clip(pixels * scale, 0.0, 255.0).astype(np.uint8)
    else:
        pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    if pixels.ndim == 2:
        pixels = np.repeat(pixels[..., None], 3, axis=-1)
    if pixels.shape[-1] == 4:
        pixels = pixels[..., :3]
    valid = np.asarray(result.panorama.validity("image"), dtype=bool)
    output = pixels.copy()
    output[~valid] = 0
    with NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        Image.fromarray(output, mode="RGB").save(handle, format="PNG")
        temporary = Path(handle.name)
    temporary.replace(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
