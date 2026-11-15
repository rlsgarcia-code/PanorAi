"""FastAPI entry point for the local PanorAi workbench."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import subprocess
import time
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response, StreamingResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
import numpy as np
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict

from .catalog import experiment_catalog, runtime_fingerprint, workspace_catalog
from .discovery_surface import (
    discovery_catalog as goal_discovery_catalog,
    recommendations as goal_recommendations,
)
from .runs import STORE, cancel_run as cancel_persisted_run, schedule_run
from .service import (
    analyze_pair,
    capabilities,
    lift_erp_points,
    single_view_academy_status,
)
from .known_height import preview_known_height_relation
from .line_refinement import refine_erp_line
from .planar import PlanarSceneStore
from .planar_discovery import discover_planar_orientations
from .resources import install_sam21_dependencies, sam21_runtime_status
from .operational import MIRROR
from .samples import (
    DEFAULT_ERP,
    KNOWN_HEIGHT_ROOM_ID,
    PAIR_ID,
    STEREO_TEXTURE,
    known_height_room_sample,
    known_pose_planar_scene,
    known_pose_pair_pngs,
    sample_catalog,
)


app = FastAPI(
    title="PanorAi Workbench Backend",
    version="0.1.0",
    description="Local inspection endpoints over explicit PanorAi public APIs.",
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver", "backend"],
)


class Sam21InstallRequest(BaseModel):
    """A closed request: browsers cannot choose packages or commands."""

    model_config = ConfigDict(extra="forbid")
    accept_environment_change: bool


def _planar_store() -> PlanarSceneStore:
    return PlanarSceneStore(STORE.project)


@app.post("/api/v1/planar/scenes")
async def create_planar_scene(
    document: Annotated[str, Form()],
    images: Annotated[list[UploadFile], File()],
) -> dict:
    try:
        manifest = json.loads(document)
        inputs = []
        for upload in images:
            payload = await upload.read()
            decoded = _decode_image(payload, upload.filename or "image.png")
            record = STORE.ingest(payload, upload.filename or "image.png")
            record["image_shape_hw"] = list(decoded.shape[:2])
            inputs.append(record)
        return await run_in_threadpool(_planar_store().save, manifest, inputs)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/v1/planar/scenes/{scene_id}/{revision}")
def get_planar_scene(scene_id: str, revision: str) -> dict:
    try:
        payload = _planar_store().load(scene_id, revision)
        return {
            "scene_id": scene_id,
            "revision": revision,
            "document": payload["document"],
        }
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/planar/scenes/{scene_id}/{revision}/images/{index}")
def get_planar_image(scene_id: str, revision: str, index: int) -> FileResponse:
    try:
        payload = _planar_store().load(scene_id, revision)
        if index < 0 or index >= len(payload["inputs"]):
            raise KeyError("image index")
        path = Path(payload["inputs"][index]["path"]).resolve()
        if STORE.inputs.resolve() not in path.parents:
            raise ValueError("invalid stored image")
        return FileResponse(path)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/v1/planar/scenes/{scene_id}/{revision}/revisions")
async def revise_planar_scene(
    scene_id: str, revision: str, document: dict
) -> dict:
    try:
        previous = _planar_store().load(scene_id, revision)
        return await run_in_threadpool(
            _planar_store().save,
            document,
            previous["inputs"],
            scene_id=scene_id,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/v1/planar/scenes/{scene_id}/{revision}/runs")
def create_planar_run(scene_id: str, revision: str, parameters: dict) -> dict:
    try:
        snapshot = _planar_store().load(scene_id, revision)
        run = STORE.create(
            experiment_id="human-guided-planar",
            workspace="multi",
            parameters=parameters,
            inputs=snapshot["inputs"],
        )
        run["planar_scene"] = snapshot["document"]
        STORE.save(run)
        return schedule_run(run["id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _decode_image(payload: bytes, name: str) -> np.ndarray:
    if len(payload) > 160 * 1024 * 1024:
        raise ValueError(f"{name} exceeds the 160 MiB upload limit")
    try:
        with Image.open(BytesIO(payload)) as image:
            image.load()
            return np.asarray(image.convert("RGB"))
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"{name} is not a readable image") from exc


@app.get("/api/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "panorai-workbench/v0"}


@app.get("/api/v1/health/live")
def health_live() -> dict[str, str]:
    return {"status": "ok", "service": "panorai-workbench/v0"}


@app.get("/api/v1/health/ready")
def health_ready() -> Response:
    persistence = MIRROR.readiness()
    ready = bool(persistence["ready"])
    return Response(
        content=json.dumps(
            {
                "status": "ready" if ready else "not-ready",
                "service": "panorai-workbench/v0",
                "persistence": persistence,
            },
            ensure_ascii=False,
        ),
        status_code=200 if ready else 503,
        media_type="application/json",
    )


@app.get("/api/v1/samples")
def get_samples() -> dict:
    """List redistributable examples available without remote I/O."""

    return sample_catalog()


@app.get("/api/v1/samples/nature-reserve-forest-cc0/image")
def get_default_erp_sample() -> FileResponse:
    if not DEFAULT_ERP.is_file():
        raise HTTPException(status_code=404, detail="default ERP sample is unavailable")
    return FileResponse(
        DEFAULT_ERP,
        media_type="image/jpeg",
        filename="nature-reserve-forest-erp.jpg",
        headers={
            "X-PanorAi-Sample-License": "CC0-1.0",
            "X-PanorAi-Sample-Source": "Poly-Haven",
        },
    )


@app.get("/api/v1/samples/poly-haven-studio-cc0/image")
def get_indoor_erp_sample() -> FileResponse:
    if not STEREO_TEXTURE.is_file():
        raise HTTPException(status_code=404, detail="indoor ERP sample is unavailable")
    return FileResponse(
        STEREO_TEXTURE,
        media_type="image/jpeg",
        filename="poly-haven-studio-erp.jpg",
        headers={
            "X-PanorAi-Sample-License": "CC0-1.0",
            "X-PanorAi-Sample-Source": "Poly-Haven",
        },
    )


@app.get(f"/api/v1/samples/{PAIR_ID}/{{side}}.png")
def get_known_pose_pair_sample(side: str) -> Response:
    if side not in {"a", "b"}:
        raise HTTPException(status_code=404, detail="sample side must be a or b")
    try:
        image_a, image_b = known_pose_pair_pngs()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="stereo sample is unavailable") from exc
    return Response(
        image_a if side == "a" else image_b,
        media_type="image/png",
        headers={
            "Content-Disposition": f'inline; filename="studio-known-pose-{side}.png"',
            "X-PanorAi-Sample-License": "CC0-1.0",
            "X-PanorAi-Sample-Pose": "known",
        },
    )


@app.get(f"/api/v1/samples/{PAIR_ID}/planar-scene")
def get_known_pose_planar_scene() -> dict:
    return known_pose_planar_scene()


@app.get(f"/api/v1/samples/{KNOWN_HEIGHT_ROOM_ID}/image.png")
def get_known_height_room_image() -> Response:
    image, _ = known_height_room_sample()
    return Response(
        image,
        media_type="image/png",
        headers={
            "Content-Disposition": 'inline; filename="known-height-room.png"',
            "X-PanorAi-Sample-License": "MIT",
            "X-PanorAi-Sample-Geometry": "known-height-analytic-room",
        },
    )


@app.get(f"/api/v1/samples/{KNOWN_HEIGHT_ROOM_ID}/scene")
def get_known_height_room_scene() -> dict:
    _, document = known_height_room_sample()
    return document


@app.get("/api/v1/resources/sam21")
def get_sam21_resource() -> dict:
    return sam21_runtime_status()


@app.post("/api/v1/resources/sam21/install")
async def install_sam21_resource(request: Sam21InstallRequest) -> dict:
    """Install only the audited SAM extra after explicit confirmation."""

    try:
        return await run_in_threadpool(
            install_sam21_dependencies,
            accept_environment_change=request.accept_environment_change,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (RuntimeError, subprocess.SubprocessError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/v1/interactions/refine-line")
async def refine_interaction_line(
    request: Annotated[str, Form()],
    image: Annotated[UploadFile, File()],
) -> dict:
    """Suggest, but never apply, an edge-supported line refinement."""

    try:
        document = json.loads(request)
        payload = await image.read()
        decoded = _decode_image(payload, image.filename or "image.png")
        return await run_in_threadpool(refine_erp_line, decoded, document)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/v1/interactions/discover-planar-orientations")
async def discover_interaction_planar_orientations(
    request: Annotated[str, Form()],
    image: Annotated[UploadFile, File()],
) -> dict:
    """Discover reviewable lines, directions and plane orientations."""

    try:
        document = json.loads(request)
        payload = await image.read()
        decoded = _decode_image(payload, image.filename or "image.png")
        return await run_in_threadpool(
            discover_planar_orientations,
            decoded,
            document,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/v1/interactions/preview-planar-relation")
async def preview_interaction_planar_relation(document: dict) -> dict:
    """Project a cursor contact through the public preview operation."""

    try:
        return await run_in_threadpool(preview_known_height_relation, document)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/v1/capabilities")
def get_capabilities() -> dict:
    try:
        return capabilities()
    except ImportError as exc:
        raise HTTPException(
            status_code=503, detail=f"PanorAi is unavailable: {exc}"
        ) from exc


@app.get("/api/v1/runtime")
def get_runtime() -> dict:
    """Identify the exact PanorAi installation exercised by this backend."""

    try:
        return runtime_fingerprint()
    except ImportError as exc:
        raise HTTPException(
            status_code=503, detail=f"PanorAi is unavailable: {exc}"
        ) from exc


@app.get("/api/v1/experiments")
def get_experiments() -> dict:
    """Return public surfaces plus the separate local adapter overlay."""

    try:
        return experiment_catalog()
    except ImportError as exc:
        raise HTTPException(
            status_code=503, detail=f"PanorAi is unavailable: {exc}"
        ) from exc


@app.get("/api/v1/catalog")
def get_catalog() -> dict:
    """Canonical Workbench projection of the installed PanorAi catalog."""

    return get_experiments()


@app.get("/api/v1/discovery")
def get_discovery() -> dict:
    """Return the goal-oriented projection of the installed catalog."""

    try:
        return goal_discovery_catalog()
    except ImportError as exc:
        raise HTTPException(
            status_code=503, detail=f"PanorAi is unavailable: {exc}"
        ) from exc


@app.post("/api/v1/discovery/recommendations")
def post_discovery_recommendations(context: dict) -> dict:
    """Rank recipes from typed metadata without inspecting image content."""

    try:
        return goal_recommendations(context)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/v1/workspaces/{workspace}")
def get_workspace(workspace: str) -> dict:
    try:
        return workspace_catalog(workspace)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/runs")
def list_runs(limit: int = 50) -> dict:
    return {
        "schema": "panorai-workbench-run-list/v1",
        "project": "scratch",
        "runs": STORE.list(limit=max(1, min(limit, 200))),
    }


@app.post("/api/v1/runs")
async def create_run(
    experiment_id: Annotated[str, Form()],
    workspace: Annotated[str, Form()],
    parameters: Annotated[str, Form()] = "{}",
    image_a: Annotated[UploadFile | None, File()] = None,
    image_b: Annotated[UploadFile | None, File()] = None,
) -> dict:
    try:
        decoded_parameters = json.loads(parameters)
        if not isinstance(decoded_parameters, dict):
            raise ValueError("parameters must encode a JSON object")
        inputs = []
        for image, fallback in ((image_a, "view-a.png"), (image_b, "view-b.png")):
            if image is None:
                continue
            payload = await image.read()
            decoded = _decode_image(payload, image.filename or fallback)
            stored = STORE.ingest(payload, image.filename or fallback)
            stored["image_shape_hw"] = [int(decoded.shape[0]), int(decoded.shape[1])]
            inputs.append(stored)
        run = STORE.create(
            experiment_id=experiment_id,
            workspace=workspace,
            parameters=decoded_parameters,
            inputs=inputs,
        )
        return schedule_run(run["id"])
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/v2/runs")
async def create_component_run(request: Request) -> dict:
    """Create a generic single-component run or typed ordered plan."""

    form = None
    try:
        form = await request.form()
        raw_plan = form.get("plan")
        if not isinstance(raw_plan, str):
            raise ValueError("plan must be a JSON form field")
        plan = json.loads(raw_plan)
        if not isinstance(plan, dict):
            raise ValueError("plan must encode a JSON object")
        file_fields = plan.get("file_fields", [])
        if not isinstance(file_fields, list) or not all(
            isinstance(item, str) for item in file_fields
        ):
            raise ValueError("file_fields must be a string array")
        uploads = [item for item in form.getlist("files") if hasattr(item, "read")]
        if len(uploads) != len(file_fields):
            raise ValueError("file_fields must identify every uploaded file")
        inputs: list[dict] = []
        for field, upload in zip(file_fields, uploads, strict=True):
            payload = await upload.read()
            filename = upload.filename or f"{field}.bin"
            stored = STORE.ingest(payload, filename)
            stored["field"] = field
            if (upload.content_type or "").startswith("image/"):
                decoded = _decode_image(payload, filename)
                stored["image_shape_hw"] = [
                    int(decoded.shape[0]),
                    int(decoded.shape[1]),
                ]
            inputs.append(stored)
        if "steps" in plan:
            steps = plan["steps"]
            if not isinstance(steps, list):
                raise ValueError("steps must be an array")
            run = STORE.create_plan(steps=steps, inputs=inputs)
        else:
            component_id = plan.get("component_id")
            configuration = plan.get("configuration", {})
            input_values = plan.get("input_values", {})
            if not isinstance(component_id, str):
                raise ValueError("component_id must be a string")
            if not isinstance(configuration, dict) or not isinstance(
                input_values, dict
            ):
                raise ValueError("configuration and input_values must be objects")
            run = STORE.create_component(
                component_id=component_id,
                configuration=configuration,
                input_values=input_values,
                inputs=inputs,
            )
        return schedule_run(run["id"])
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        if form is not None:
            await form.close()


@app.get("/api/v1/runs/{run_id}")
def get_run(run_id: str) -> dict:
    try:
        return STORE.load(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc


@app.get("/api/v1/runs/{run_id}/events")
def get_run_events(run_id: str) -> StreamingResponse:
    try:
        run = STORE.load(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc

    def stream():
        delivered = 0
        current = run
        while True:
            for event in current["events"][delivered:]:
                yield (
                    "event: progress\ndata: "
                    + json.dumps(event, ensure_ascii=False)
                    + "\n\n"
                )
            delivered = len(current["events"])
            if current["status"] in {"completed", "failed", "cancelled"}:
                yield (
                    "event: end\ndata: "
                    + json.dumps({"status": current["status"]})
                    + "\n\n"
                )
                return
            time.sleep(0.2)
            try:
                current = STORE.load(run_id)
            except KeyError:
                yield "event: end\ndata: " + json.dumps({"status": "missing"}) + "\n\n"
                return

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/v1/runs/{run_id}/cancel")
def cancel_run(run_id: str) -> dict:
    try:
        return cancel_persisted_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc


@app.get("/api/v1/runs/{run_id}/artifacts/{artifact_id}")
def get_run_artifact(run_id: str, artifact_id: str) -> FileResponse:
    try:
        path = STORE.artifact_path(run_id, artifact_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="artifact not found") from exc
    return FileResponse(path)


@app.get("/api/v1/runs/{run_id}/inputs/{index}")
def get_run_input(run_id: str, index: int) -> FileResponse:
    """Serve the deduplicated source used by a run for visual inspection."""

    try:
        path = STORE.input_path(run_id, index)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run input not found") from exc
    return FileResponse(path)


@app.get("/api/v1/academy/single-view/status")
def get_single_view_academy_status() -> dict:
    """Expose supported concepts and fail closed for missing PanorAi APIs."""

    try:
        return single_view_academy_status()
    except ImportError as exc:
        raise HTTPException(
            status_code=503, detail=f"PanorAi is unavailable: {exc}"
        ) from exc


@app.post("/api/v1/academy/single-view/rays")
def lift_single_view_points(payload: dict) -> dict:
    """Convert manually selected ERP points using PanorAi Main geometry only."""

    try:
        shape = payload.get("image_shape_hw")
        if not isinstance(shape, list) or len(shape) != 2:
            raise ValueError("image_shape_hw must be [height, width]")
        return lift_erp_points(payload.get("points_xy"), (int(shape[0]), int(shape[1])))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/v1/pairs/analyze")
async def analyze_pair_endpoint(
    image_a: Annotated[UploadFile, File()],
    image_b: Annotated[UploadFile, File()],
    profile: Annotated[str, Form()] = "relative-pose-reference",
    max_angular_error_deg: Annotated[float, Form()] = 0.5,
    random_seed: Annotated[int, Form()] = 7,
) -> dict:
    try:
        array_a = _decode_image(await image_a.read(), "image_a")
        array_b = _decode_image(await image_b.read(), "image_b")
        return await run_in_threadpool(
            analyze_pair,
            array_a,
            array_b,
            panorama_id_a=image_a.filename or "view-a",
            panorama_id_b=image_b.filename or "view-b",
            profile=profile,
            max_angular_error_deg=max_angular_error_deg,
            random_seed=random_seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/v1/semantics/status")
def semantic_status() -> dict:
    """Declare the planned boundary without pretending the runner exists."""

    return {
        "available": False,
        "status": "not-implemented",
        "planned_stages": [
            "dense semantic evidence",
            "spherical FOV anchors",
            "A-to-B and B-to-A region transfer",
            "cross-view anchor fusion",
            "optional segmentation review",
        ],
    }
