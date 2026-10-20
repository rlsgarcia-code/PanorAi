"""FastAPI entry point for the local PanorAi workbench."""

from __future__ import annotations

from io import BytesIO
import json
import time
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
import numpy as np
from PIL import Image, UnidentifiedImageError

from .catalog import experiment_catalog, runtime_fingerprint, workspace_catalog
from .runs import STORE, cancel_run as cancel_persisted_run, schedule_run
from .service import (
    analyze_pair,
    capabilities,
    lift_erp_points,
    single_view_academy_status,
)


app = FastAPI(
    title="PanorAi Workbench Backend",
    version="0.1.0",
    description="Local inspection endpoints over explicit PanorAi public APIs.",
)


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
