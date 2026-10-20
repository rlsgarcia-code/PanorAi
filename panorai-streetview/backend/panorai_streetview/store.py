"""Checksum-bound local project storage with atomic JSON manifests."""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from threading import Lock, RLock
from typing import Any
from uuid import uuid4

from PIL import Image, UnidentifiedImageError

from .engines import runtime_fingerprint
from .models import (
    DenseAnnotation,
    ImageRecord,
    ImagePairInput,
    ImagePairRecord,
    ManualCorrespondence,
    ObjectRegion,
    ProjectRecord,
    Workflow,
    is_valid_registered_pose,
    utc_now,
)


SAFE_ID = re.compile(r"^[a-zA-Z0-9_-]+$")
IMAGE_FORMATS = {
    "bmp": (".bmp", "image/bmp"),
    "gif": (".gif", "image/gif"),
    "jpeg": (".jpg", "image/jpeg"),
    "png": (".png", "image/png"),
    "tiff": (".tiff", "image/tiff"),
    "webp": (".webp", "image/webp"),
}
MAX_DECODED_IMAGE_WIDTH = 32_768
MAX_DECODED_IMAGE_HEIGHT = 16_384
MAX_DECODED_IMAGE_PIXELS = 64 * 1024 * 1024


class ProjectStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.projects_dir = self.root / "projects"
        self.objects_dir = self.root / "objects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        self._project_locks: dict[str, Any] = {}
        self._project_locks_guard = Lock()

    def project_lock(self, project_id: str):
        """Return the re-entrant lock protecting one project manifest transaction."""

        if not SAFE_ID.fullmatch(project_id):
            raise KeyError(project_id)
        with self._project_locks_guard:
            return self._project_locks.setdefault(project_id, RLock())

    def _project_path(self, project_id: str) -> Path:
        if not SAFE_ID.fullmatch(project_id):
            raise KeyError(project_id)
        return self.projects_dir / f"{project_id}.json"

    def _write(self, project: ProjectRecord) -> None:
        target = self._project_path(project.id)
        payload = project.model_dump_json(indent=2, by_alias=True)
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.projects_dir, delete=False
        ) as handle:
            handle.write(payload)
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(target)

    def list(self) -> list[dict[str, Any]]:
        projects = [self.load(path.stem) for path in self.projects_dir.glob("*.json")]
        projects.sort(key=lambda item: item.updated_at, reverse=True)
        return [project.model_dump(mode="json", by_alias=True) for project in projects]

    def load(self, project_id: str) -> ProjectRecord:
        path = self._project_path(project_id)
        if not path.is_file():
            raise KeyError(project_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        # APP-239: old local drafts may contain the retired line/VP evidence.
        # It is ignored on read and disappears on the next normal project write.
        payload.pop("constraints", None)
        return ProjectRecord.model_validate(payload)

    def create(
        self,
        *,
        name: str,
        workflow: Workflow,
        uploads: list[tuple[str, str, bytes]],
        metadata: dict[str, Any],
        pair_inputs: list[ImagePairInput] | None = None,
    ) -> ProjectRecord:
        if not name.strip():
            raise ValueError("project name is required")
        if not uploads:
            raise ValueError("at least one image is required")
        if workflow == "spherical_unregistered" and any(
            isinstance(value, dict) and "pose" in value for value in metadata.values()
        ):
            raise ValueError(
                "unregistered ERP projects must not contain supplied poses; "
                "use the registered workflow for pose-bearing inputs"
            )
        prepared = [
            self._ingest(index, item, metadata) for index, item in enumerate(uploads)
        ]
        records = [record for record, _payload in prepared]
        pairs = self._resolve_pairs(records, pair_inputs or [], workflow)
        timestamp = utc_now()
        project = ProjectRecord(
            id=f"sv_{uuid4().hex[:12]}",
            name=name.strip(),
            workflow=workflow,
            created_at=timestamp,
            updated_at=timestamp,
            panorai_runtime=runtime_fingerprint(),
            images=records,
            pairs=pairs,
            object_regions=[],
            manual_correspondences=[],
            dense_annotations=[],
            status="draft",
        )
        for record, payload in prepared:
            target = self.objects_dir / record.object_name
            if not target.exists():
                target.write_bytes(payload)
        self._write(project)
        return project

    @staticmethod
    def _resolve_pairs(
        records: list[ImageRecord],
        pair_inputs: list[ImagePairInput],
        workflow: Workflow,
    ) -> list[ImagePairRecord]:
        if not pair_inputs:
            return []
        by_filename: dict[str, ImageRecord] = {}
        for record in records:
            if record.filename in by_filename:
                raise ValueError(
                    "pair-addressed projects require unique upload filenames"
                )
            by_filename[record.filename] = record
        resolved = []
        seen_endpoints: set[frozenset[str]] = set()
        for pair in pair_inputs:
            try:
                reference = by_filename[pair.reference_filename]
                source = by_filename[pair.source_filename]
            except KeyError as exc:
                raise ValueError(
                    f"pair {pair.id} references an unknown upload filename"
                ) from exc
            endpoint_key = frozenset((reference.id, source.id))
            if endpoint_key in seen_endpoints:
                raise ValueError(
                    "duplicate unordered image pair: "
                    f"{pair.reference_filename!r} and {pair.source_filename!r}"
                )
            seen_endpoints.add(endpoint_key)
            resolved.append(
                ImagePairRecord(
                    id=pair.id,
                    reference_image_id=reference.id,
                    source_image_id=source.id,
                    relation=(
                        "registered_overlap"
                        if workflow == "spherical_registered"
                        else "overlap_candidate"
                    ),
                    baseline_m=pair.baseline_m,
                    minimum_directional_overlap=pair.minimum_directional_overlap,
                    provenance=pair.provenance,
                )
            )
        return resolved

    def _ingest(
        self,
        index: int,
        upload: tuple[str, str, bytes],
        metadata: dict[str, Any],
    ) -> tuple[ImageRecord, bytes]:
        filename, _claimed_media_type, payload = upload
        if not payload:
            raise ValueError(f"{filename} is empty")
        if len(payload) > 160 * 1024 * 1024:
            raise ValueError(f"{filename} exceeds the 160 MiB limit")
        try:
            with Image.open(BytesIO(payload)) as image:
                width, height = image.size
                if (
                    width > MAX_DECODED_IMAGE_WIDTH
                    or height > MAX_DECODED_IMAGE_HEIGHT
                    or width * height > MAX_DECODED_IMAGE_PIXELS
                ):
                    raise ValueError(
                        f"{filename} exceeds decoded image limits "
                        f"({MAX_DECODED_IMAGE_WIDTH}x{MAX_DECODED_IMAGE_HEIGHT}, "
                        f"{MAX_DECODED_IMAGE_PIXELS} pixels)"
                    )
                image.load()
                detected_format = (image.format or "unknown").lower()
        except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"{filename} is not a readable image") from exc
        try:
            suffix, media_type = IMAGE_FORMATS[detected_format]
        except KeyError as exc:
            raise ValueError(
                f"{filename} uses unsupported decoded image format {detected_format!r}"
            ) from exc
        digest = sha256(payload).hexdigest()
        object_name = f"{digest}{suffix}"
        image_metadata = metadata.get(filename, {})
        if not isinstance(image_metadata, dict):
            raise ValueError(f"metadata for {filename} must be an object")
        return (
            ImageRecord(
                id=f"img_{index + 1:04d}_{digest[:8]}",
                filename=Path(filename).name,
                object_name=object_name,
                sha256=digest,
                media_type=media_type,
                width=width,
                height=height,
                size_bytes=len(payload),
                metadata=image_metadata,
            ),
            payload,
        )

    def add_object_region(self, project_id: str, region: ObjectRegion) -> ProjectRecord:
        with self.project_lock(project_id):
            return self._add_object_region(project_id, region)

    def _add_object_region(
        self, project_id: str, region: ObjectRegion
    ) -> ProjectRecord:
        project = self.load(project_id)
        image = next(
            (item for item in project.images if item.id == region.image_id), None
        )
        if image is None:
            raise ValueError("object region references an unknown image")
        x0, y0, x1, y1 = region.bbox_xyxy
        if x0 < 0 or y0 < 0 or x1 > image.width or y1 > image.height:
            raise ValueError("object region must lie inside the source image")
        stored = region.model_copy(
            update={
                "id": region.id or f"object_region_{uuid4().hex[:12]}",
                "created_at": utc_now(),
            }
        )
        retained = [
            item
            for item in project.object_regions
            if not (
                item.object_id == stored.object_id and item.image_id == stored.image_id
            )
        ]
        updated = project.model_copy(
            update={
                "object_regions": [*retained, stored],
                "updated_at": utc_now(),
            }
        )
        self._write(updated)
        return updated

    def add_manual_correspondence(
        self, project_id: str, correspondence: ManualCorrespondence
    ) -> ProjectRecord:
        with self.project_lock(project_id):
            return self._add_manual_correspondence(project_id, correspondence)

    def _add_manual_correspondence(
        self, project_id: str, correspondence: ManualCorrespondence
    ) -> ProjectRecord:
        project = self.load(project_id)
        pair = next(
            (item for item in project.pairs if item.id == correspondence.pair_id),
            None,
        )
        if pair is None:
            raise ValueError("manual correspondence references an unknown pair")
        reference = next(
            item for item in project.images if item.id == pair.reference_image_id
        )
        source = next(
            item for item in project.images if item.id == pair.source_image_id
        )
        for name, point, image in (
            ("reference", correspondence.reference_xy, reference),
            ("source", correspondence.source_xy, source),
        ):
            if not (0 <= point[0] < image.width and 0 <= point[1] < image.height):
                raise ValueError(f"manual {name} point must lie inside its image")
        stored = correspondence.model_copy(
            update={
                "id": f"manual_match_{uuid4().hex[:12]}",
                "created_at": utc_now(),
            }
        )
        updated = project.model_copy(
            update={
                "manual_correspondences": [
                    *project.manual_correspondences,
                    stored,
                ],
                "updated_at": utc_now(),
            }
        )
        self._write(updated)
        return updated

    def add_dense_annotation(
        self, project_id: str, annotation: DenseAnnotation
    ) -> ProjectRecord:
        with self.project_lock(project_id):
            return self._add_dense_annotation(project_id, annotation)

    def _add_dense_annotation(
        self, project_id: str, annotation: DenseAnnotation
    ) -> ProjectRecord:
        project = self.load(project_id)
        image = next(
            (item for item in project.images if item.id == annotation.image_id), None
        )
        if image is None:
            raise ValueError("dense annotation references an unknown image")
        if any(
            not (0 <= point[0] < image.width and 0 <= point[1] < image.height)
            for point in annotation.points_xy
        ):
            raise ValueError("dense annotation must lie inside the source image")
        stored = annotation.model_copy(
            update={
                "id": annotation.id or f"dense_{uuid4().hex[:12]}",
                "created_at": utc_now(),
            }
        )
        updated = project.model_copy(
            update={
                "dense_annotations": [*project.dense_annotations, stored],
                "updated_at": utc_now(),
            }
        )
        self._write(updated)
        return updated

    def image_path(self, project_id: str, image_id: str) -> Path:
        return self.image_asset(project_id, image_id)[0]

    def image_asset(self, project_id: str, image_id: str) -> tuple[Path, str]:
        project = self.load(project_id)
        image = next((item for item in project.images if item.id == image_id), None)
        if image is None:
            raise KeyError(image_id)
        path = self.objects_dir / image.object_name
        if not path.is_file():
            raise KeyError(image_id)
        expected = next(
            (
                media_type
                for suffix, media_type in IMAGE_FORMATS.values()
                if path.suffix.lower() == suffix
            ),
            None,
        )
        if expected is None or image.media_type != expected:
            raise KeyError(image_id)
        return path, expected

    def scene_manifest(self, project_id: str) -> dict[str, Any]:
        project = self.load(project_id)
        if project.workflow != "spherical_registered":
            raise ValueError(
                "scene manifests are currently generated only for registered ERPs"
            )
        nodes = []
        for image in project.images:
            pose = image.metadata.get("pose")
            if not is_valid_registered_pose(pose):
                raise ValueError(f"missing or invalid rigid pose for {image.filename}")
            assert isinstance(pose, dict)
            nodes.append(
                {
                    "id": image.id,
                    "image_sha256": image.sha256,
                    "image_url": f"/api/v1/projects/{project.id}/images/{image.id}",
                    "rotation_world_to_panorama": pose.get(
                        "rotation_world_to_panorama"
                    ),
                    "center_world": pose.get("center_world"),
                }
            )
        edges = [
            {
                "id": pair.id,
                "reference_image_id": pair.reference_image_id,
                "source_image_id": pair.source_image_id,
                "relation": pair.relation,
                "baseline_m": pair.baseline_m,
                "minimum_directional_overlap": pair.minimum_directional_overlap,
                "provenance": pair.provenance,
            }
            for pair in project.pairs
        ]
        if not edges:
            raise ValueError(
                "registered Street View scenes require an explicit pair graph"
            )
        return {
            "schema": "panorai-streetview-scene/v1",
            "project_id": project.id,
            "coordinate_contract": "x_panorama = R_world_to_panorama @ (X_world - center_world)",
            "scale": "as-supplied",
            "nodes": nodes,
            "edges": edges,
            "navigation": "explicit registered overlap pairs",
        }
