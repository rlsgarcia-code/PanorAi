"""FastAPI entry point for the isolated PanorAi Street View workspace."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from .dense_preview import PairDensePreviewService
from .dense_scene import DenseSceneService
from .engines import engine_capabilities
from .global_map import GlobalMapService
from .models import (
    DenseAnnotation,
    DenseSceneRequest,
    FloorCalibrationRequest,
    FloorMeasurementRequest,
    ImagePairInput,
    ManualCorrespondence,
    MatchReview,
    ObjectRegion,
    RegistrationRequest,
    Workflow,
    VerticalMeasurementRequest,
)
from .metrology import CalibrationRevisionConflict, MetrologyService
from .planner import build_plan
from .registration import PairRegistrationService
from .setup import scene_setup
from .stitching import StitchingService
from .store import ProjectStore


MAX_PROJECT_IMAGES = 64
MAX_IMAGE_BYTES = 160 * 1024 * 1024
MAX_PROJECT_UPLOAD_BYTES = 512 * 1024 * 1024


def _validation_detail(value):
    if isinstance(value, dict):
        return {str(key): _validation_detail(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_validation_detail(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def create_app(data_root: Path | None = None) -> FastAPI:
    root = data_root or Path(
        os.environ.get(
            "PANORAI_STREETVIEW_DATA",
            Path(__file__).resolve().parents[2] / ".local",
        )
    )
    store = ProjectStore(root)
    registration = PairRegistrationService(store)
    global_map = GlobalMapService(store, registration)
    dense_preview = PairDensePreviewService(store, registration)
    dense_scene = DenseSceneService(store, global_map)
    stitching = StitchingService(store)
    metrology = MetrologyService(store)
    application = FastAPI(
        title="PanorAi Street View Backend",
        version="0.1.0",
        description="Project intake and build planning over public PanorAi APIs.",
    )
    application.state.store = store
    application.state.registration = registration
    application.state.global_map = global_map
    application.state.dense_preview = dense_preview
    application.state.dense_scene = dense_scene
    application.state.stitching = stitching
    application.state.metrology = metrology

    @application.exception_handler(RequestValidationError)
    async def request_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"detail": _validation_detail(exc.errors())},
        )

    @application.get("/api/v1/projects/{project_id}/metrology/{image_id}")
    def get_metrology(project_id: str, image_id: str) -> dict:
        try:
            return metrology.load_record(project_id, image_id).model_dump(
                mode="json", by_alias=True
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="floor calibration has not been run"
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.get("/api/v1/projects/{project_id}/metrology/{image_id}/history")
    def get_metrology_history(project_id: str, image_id: str) -> dict:
        try:
            return metrology.history(project_id, image_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="project or image not found"
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.post("/api/v1/projects/{project_id}/metrology/calibrate")
    def calibrate_metrology(project_id: str, payload: FloorCalibrationRequest) -> dict:
        try:
            return metrology.calibrate(project_id, payload)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="project or image not found"
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.post("/api/v1/projects/{project_id}/metrology/floor-point")
    def measure_metrology_floor(
        project_id: str, payload: FloorMeasurementRequest
    ) -> dict:
        try:
            return metrology.floor_point(
                project_id,
                payload.image_id,
                payload.point_xy,
                calibration_revision=payload.calibration_revision,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except CalibrationRevisionConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.post("/api/v1/projects/{project_id}/metrology/vertical")
    def measure_metrology_vertical(
        project_id: str, payload: VerticalMeasurementRequest
    ) -> dict:
        try:
            return metrology.vertical(
                project_id,
                payload.image_id,
                payload.base_xy,
                payload.top_xy,
                calibration_revision=payload.calibration_revision,
                max_horizontal_residual_m=payload.max_horizontal_residual_m,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except CalibrationRevisionConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.get("/api/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "panorai-streetview/v1"}

    @application.get("/api/v1/capabilities")
    def capabilities() -> dict:
        return engine_capabilities()

    @application.get("/api/v1/projects")
    def list_projects() -> dict:
        return {
            "schema": "panorai-streetview-project-list/v1",
            "projects": store.list(),
        }

    @application.post("/api/v1/projects")
    async def create_project(
        name: Annotated[str, Form()],
        workflow: Annotated[Workflow, Form()],
        images: Annotated[list[UploadFile], File()],
        metadata: Annotated[str, Form()] = "{}",
        pairs: Annotated[str, Form()] = "[]",
    ) -> dict:
        try:
            decoded_metadata = json.loads(metadata)
            if not isinstance(decoded_metadata, dict):
                raise ValueError("metadata must encode a JSON object keyed by filename")
            decoded_pairs = json.loads(pairs)
            if not isinstance(decoded_pairs, list):
                raise ValueError("pairs must encode a JSON array")
            pair_inputs = [
                ImagePairInput.model_validate(item) for item in decoded_pairs
            ]
            if len(images) > MAX_PROJECT_IMAGES:
                raise ValueError(
                    f"a project accepts at most {MAX_PROJECT_IMAGES} images"
                )
            uploads = []
            total_bytes = 0
            try:
                for index, image in enumerate(images):
                    filename = image.filename or f"image-{index + 1}.bin"
                    payload = await image.read(MAX_IMAGE_BYTES + 1)
                    if len(payload) > MAX_IMAGE_BYTES:
                        raise ValueError(f"{filename} exceeds the 160 MiB limit")
                    total_bytes += len(payload)
                    if total_bytes > MAX_PROJECT_UPLOAD_BYTES:
                        raise ValueError(
                            "project uploads exceed the 512 MiB aggregate limit"
                        )
                    uploads.append(
                        (
                            filename,
                            image.content_type or "application/octet-stream",
                            payload,
                        )
                    )
            finally:
                for image in images:
                    await image.close()
            project = store.create(
                name=name,
                workflow=workflow,
                uploads=uploads,
                metadata=decoded_metadata,
                pair_inputs=pair_inputs,
            )
            return project.model_dump(mode="json", by_alias=True)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.get("/api/v1/projects/{project_id}")
    def get_project(project_id: str) -> dict:
        try:
            return store.load(project_id).model_dump(mode="json", by_alias=True)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc

    @application.get("/api/v1/projects/{project_id}/build-plan")
    def get_build_plan(project_id: str) -> dict:
        try:
            return build_plan(store.load(project_id)).model_dump(
                mode="json", by_alias=True
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc

    @application.get("/api/v1/projects/{project_id}/setup")
    def get_scene_setup(project_id: str) -> dict:
        try:
            return scene_setup(
                store.load(project_id),
                registration,
                global_map.load_optional(project_id),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc

    @application.get("/api/v1/projects/{project_id}/global-map")
    def get_global_map(project_id: str) -> dict:
        try:
            return global_map.load(project_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="global map has not been run"
            ) from exc

    @application.post("/api/v1/projects/{project_id}/global-map")
    def run_global_map(project_id: str) -> dict:
        try:
            return global_map.run(project_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.get("/api/v1/projects/{project_id}/pairs/{pair_id}/dense-preview")
    def get_dense_preview(project_id: str, pair_id: str) -> dict:
        try:
            return dense_preview.load(project_id, pair_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="dense preview has not been run"
            ) from exc

    @application.post("/api/v1/projects/{project_id}/pairs/{pair_id}/dense-preview")
    def run_dense_preview(project_id: str, pair_id: str) -> dict:
        try:
            return dense_preview.run(project_id, pair_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="project or pair not found"
            ) from exc
        except (RuntimeError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.get(
        "/api/v1/projects/{project_id}/pairs/{pair_id}/dense-preview/image"
    )
    def get_dense_preview_image(
        project_id: str, pair_id: str, revision: int | None = None
    ) -> FileResponse:
        try:
            return FileResponse(
                dense_preview.image_path(project_id, pair_id, revision=revision),
                media_type="image/png",
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="dense preview image not found"
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.get("/api/v1/projects/{project_id}/dense-scene")
    def get_dense_scene(project_id: str) -> dict:
        try:
            return dense_scene.load(project_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="dense scene has not been run"
            ) from exc

    @application.post("/api/v1/projects/{project_id}/dense-scene")
    def run_dense_scene(project_id: str, payload: DenseSceneRequest) -> dict:
        try:
            return dense_scene.run(project_id, profile=payload.profile)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="project or global map not found"
            ) from exc
        except (RuntimeError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.get("/api/v1/projects/{project_id}/dense-scene/mesh")
    def get_dense_scene_mesh(
        project_id: str, revision: int | None = None
    ) -> FileResponse:
        try:
            return FileResponse(
                dense_scene.mesh_path(project_id, revision=revision),
                media_type="application/octet-stream",
                filename=f"{project_id}-dense-scene.ply",
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="accepted dense-scene mesh not found"
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.get("/api/v1/projects/{project_id}/stitching")
    def get_stitching(project_id: str) -> dict:
        try:
            return stitching.load(project_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="stitching has not been run"
            ) from exc

    @application.post("/api/v1/projects/{project_id}/stitching")
    def run_stitching(project_id: str) -> dict:
        try:
            return stitching.run(project_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        except (RuntimeError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.get("/api/v1/projects/{project_id}/stitching/image")
    def get_stitching_image(
        project_id: str,
        group_id: str = "capture-1",
        revision: int | None = None,
    ) -> FileResponse:
        try:
            return FileResponse(
                stitching.image_path(project_id, group_id, revision=revision),
                media_type="image/png",
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="accepted stitched panorama not found"
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.get("/api/v1/projects/{project_id}/stitching/groups/{group_id}/image")
    def get_stitching_group_image(
        project_id: str, group_id: str, revision: int | None = None
    ) -> FileResponse:
        try:
            return FileResponse(
                stitching.image_path(project_id, group_id, revision=revision),
                media_type="image/png",
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="accepted stitched panorama not found"
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.post("/api/v1/projects/{project_id}/stitching/promote")
    def promote_stitching(project_id: str) -> dict:
        try:
            return stitching.promote(project_id).model_dump(mode="json", by_alias=True)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="project or stitching result not found"
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.post("/api/v1/projects/{project_id}/object-regions")
    def add_object_region(project_id: str, payload: ObjectRegion) -> dict:
        try:
            return store.add_object_region(project_id, payload).model_dump(
                mode="json", by_alias=True
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.post("/api/v1/projects/{project_id}/manual-correspondences")
    def add_manual_correspondence(
        project_id: str, payload: ManualCorrespondence
    ) -> dict:
        try:
            return store.add_manual_correspondence(project_id, payload).model_dump(
                mode="json", by_alias=True
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.post("/api/v1/projects/{project_id}/dense-annotations")
    def add_dense_annotation(project_id: str, payload: DenseAnnotation) -> dict:
        try:
            with store.project_lock(project_id):
                if payload.kind in {"include_region", "exclude_region"}:
                    payload = payload.model_copy(
                        update={"engine_effect": "pair_validity"}
                    )
                bound = registration.bind_range_anchor(project_id, payload)
                project = store.load(project_id)
                if bound.engine_effect == "range_prior" and any(
                    item.state == "accepted"
                    and item.engine_effect == "range_prior"
                    and item.image_id == bound.image_id
                    and item.feature_index == bound.feature_index
                    for item in project.dense_annotations
                ):
                    raise ValueError(
                        "a range prior already exists for this panorama feature"
                    )
                return store.add_dense_annotation(project_id, bound).model_dump(
                    mode="json", by_alias=True
                )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.get("/api/v1/projects/{project_id}/pairs/{pair_id}/registration")
    def get_pair_registration(project_id: str, pair_id: str) -> dict:
        try:
            return registration.load(project_id, pair_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="pair registration has not been run"
            ) from exc

    @application.get(
        "/api/v1/projects/{project_id}/pairs/{pair_id}/registration/history"
    )
    def get_pair_registration_history(project_id: str, pair_id: str) -> dict:
        try:
            return registration.history(project_id, pair_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="project or pair not found"
            ) from exc

    @application.post("/api/v1/projects/{project_id}/pairs/{pair_id}/registration")
    def run_pair_registration(
        project_id: str, pair_id: str, payload: RegistrationRequest
    ) -> dict:
        try:
            return registration.run(project_id, pair_id, payload)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="project or pair not found"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.post(
        "/api/v1/projects/{project_id}/pairs/{pair_id}/registration/reestimate"
    )
    def reestimate_pair_registration(
        project_id: str, pair_id: str, payload: RegistrationRequest
    ) -> dict:
        try:
            return registration.reestimate(project_id, pair_id, payload)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="run the pair registration first"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.patch(
        "/api/v1/projects/{project_id}/pairs/{pair_id}/registration/matches/{match_index}"
    )
    def review_pair_match(
        project_id: str,
        pair_id: str,
        match_index: int,
        payload: MatchReview,
    ) -> dict:
        try:
            return registration.review_match(project_id, pair_id, match_index, payload)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="registration or match not found"
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.get("/api/v1/projects/{project_id}/images/{image_id}")
    def get_image(project_id: str, image_id: str) -> FileResponse:
        try:
            path, media_type = store.image_asset(project_id, image_id)
            return FileResponse(
                path,
                media_type=media_type,
                headers={"X-Content-Type-Options": "nosniff"},
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="image not found") from exc

    @application.post("/api/v1/projects/{project_id}/scene-manifest")
    def create_scene_manifest(project_id: str) -> dict:
        try:
            return store.scene_manifest(project_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return application


app = create_app()
