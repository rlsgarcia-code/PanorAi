"""Persistent, allow-listed experiment execution for the local Workbench."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import threading
from time import perf_counter
import uuid
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .catalog import adapter_for, experiment_catalog, runtime_fingerprint
from .discovery import component_index
from .planning import validate_plan
from .service import _jsonable, analyze_pair, lift_erp_points


TERMINAL_STATES = {"completed", "failed", "cancelled"}


def _application_root() -> Path:
    """Locate the checkout app root or a writable per-user installed root."""

    module_root = Path(__file__).resolve().parents[2]
    if (module_root / "package.json").is_file():
        return module_root
    launch_root = Path.cwd().resolve()
    for candidate in (launch_root, launch_root / "workbench"):
        if (candidate / "package.json").is_file():
            return candidate
    return (Path.home() / ".local" / "share" / "panorai-workbench").resolve()


_APPLICATION_ROOT = _application_root()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _store_root() -> Path:
    configured = os.environ.get("PANORAI_WORKBENCH_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    return (_APPLICATION_ROOT / "outputs" / "workbench").resolve()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _normalized_u8(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype == np.bool_:
        return array.astype(np.uint8) * 255
    if (
        np.issubdtype(array.dtype, np.integer)
        and array.min(initial=0) >= 0
        and array.max(initial=0) <= 255
    ):
        return array.astype(np.uint8)
    floating = np.asarray(array, dtype=np.float64)
    finite = np.isfinite(floating)
    if not finite.any():
        return np.zeros(floating.shape, dtype=np.uint8)
    low, high = np.percentile(floating[finite], [1.0, 99.0])
    if high <= low:
        high = low + 1.0
    scaled = np.clip((floating - low) / (high - low), 0.0, 1.0)
    scaled[~finite] = 0.0
    return np.round(scaled * 255.0).astype(np.uint8)


def _labels_rgb(labels: Any) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int64)
    red = (values * 37 + 23) % 251
    green = (values * 67 + 71) % 253
    blue = (values * 97 + 131) % 255
    rgb = np.stack((red, green, blue), axis=-1).astype(np.uint8)
    rgb[values <= 0] = 0
    return rgb


def _png_bytes(values: Any, *, labels: bool = False) -> bytes:
    array = _labels_rgb(values) if labels else _normalized_u8(values)
    if array.ndim == 3 and array.shape[-1] not in {3, 4}:
        array = array[..., 0]
    image = Image.fromarray(array)
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def _gray(image: np.ndarray) -> np.ndarray:
    rgb = image[..., :3].astype(np.float32) / 255.0
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def _input_support(image: np.ndarray, policy: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Resolve support without confusing valid black pixels with missing data.

    Automatic detection is deliberately narrow: it removes only runs of
    exactly-black pixels connected to both polar raster borders. Interior
    black objects remain valid, and an ordinary full ERP remains fully
    supported. The resolved mask is persisted with the run for auditability.
    """

    height, width = image.shape[:2]
    if policy == "full-sphere":
        support = np.ones((height, width), dtype=bool)
        return support, {
            "policy": policy,
            "detected_polar_voids": False,
            "valid_fraction": 1.0,
        }
    if policy != "auto-polar-voids":
        raise ValueError(f"unsupported input support policy {policy!r}")

    black = np.all(np.asarray(image)[..., :3] == 0, axis=-1)
    polar_border = (
        float(np.mean(black[0])) >= 0.98 and float(np.mean(black[-1])) >= 0.98
    )
    support = np.ones((height, width), dtype=bool)
    if polar_border:
        for x in range(width):
            observed = np.flatnonzero(~black[:, x])
            if observed.size == 0:
                support[:, x] = False
                continue
            support[: int(observed[0]), x] = False
            support[int(observed[-1]) + 1 :, x] = False
    invalid_fraction = 1.0 - float(np.mean(support))
    detected = bool(polar_border and invalid_fraction >= 0.005)
    if not detected:
        support.fill(True)
        invalid_fraction = 0.0
    return support, {
        "policy": policy,
        "detected_polar_voids": detected,
        "valid_fraction": 1.0 - invalid_fraction,
        "criterion": "exact-black runs connected to both polar borders",
    }


def _working_raster(
    image: np.ndarray,
    support: np.ndarray,
    requested_width: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Resize continuous RGB and discrete support with the correct kernels."""

    from panorai.ops import spherical as ops

    source_shape = tuple(int(value) for value in image.shape[:2])
    resized = requested_width != "original" and int(requested_width) < source_shape[1]
    if not resized:
        return (
            image,
            support,
            {
                "source_shape_hw": list(source_shape),
                "working_shape_hw": list(source_shape),
            },
        )
    target_width = int(requested_width)
    target_height = max(1, round(source_shape[0] * target_width / source_shape[1]))
    target_shape = (target_height, target_width)
    working_image = ops.Resize(target_shape)(image.astype(np.float32) / 255.0)
    working_support = ops.Resize(target_shape, interpolation="nearest")(support).astype(
        bool
    )
    return (
        working_image,
        working_support,
        {
            "source_shape_hw": list(source_shape),
            "working_shape_hw": list(target_shape),
        },
    )


class RunStore:
    """Small immutable run store with content-addressed image inputs."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or _store_root()
        self.project = self.root / "projects" / "scratch"
        self.inputs = self.project / "inputs"
        self.runs = self.project / "runs"
        self._lock = threading.RLock()
        self.inputs.mkdir(parents=True, exist_ok=True)
        self.runs.mkdir(parents=True, exist_ok=True)
        self._recover_interrupted_runs()

    def _recover_interrupted_runs(self) -> None:
        """Fail runs that cannot still have a worker after a backend restart."""

        for path in self.runs.glob("run_*/run.json"):
            run = json.loads(path.read_text(encoding="utf-8"))
            if run.get("status") not in {"created", "running"}:
                continue
            worker_pid = run.get("worker_pid")
            if not isinstance(worker_pid, int) or _process_alive(worker_pid):
                continue
            message = (
                "Run interrompido porque o backend foi encerrado durante a execução"
            )
            run["status"] = "failed"
            run["error"] = f"RunInterrupted: {message}"
            run["events"].append({"at": _now(), "state": "failed", "message": message})
            run["updated_at"] = _now()
            _atomic_json(path, run)

    def ingest(self, payload: bytes, filename: str) -> dict[str, Any]:
        digest = _sha256(payload)
        suffix = Path(filename).suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".npy"}:
            suffix = ".bin"
        path = self.inputs / f"{digest}{suffix}"
        if not path.exists():
            path.write_bytes(payload)
        return {
            "name": filename,
            "sha256": digest,
            "size_bytes": len(payload),
            "path": str(path),
        }

    def create(
        self,
        *,
        experiment_id: str,
        workspace: str,
        parameters: dict[str, Any],
        inputs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        catalog_payload = experiment_catalog()
        catalog = {item["id"]: item for item in catalog_payload["experiments"]}
        descriptor = catalog.get(experiment_id)
        if descriptor is None:
            raise ValueError(f"unknown experiment {experiment_id!r}")
        if workspace not in descriptor["workspace_targets"]:
            raise ValueError(f"{experiment_id!r} is not compatible with {workspace!r}")
        if descriptor["availability"] != "available":
            raise ValueError(descriptor["diagnostic"] or "experiment is unavailable")
        resolved = dict(descriptor["default_preset"])
        valid_parameters = {
            item["name"]: item for item in descriptor["parameter_schema"]
        }
        for name, value in parameters.items():
            if name.startswith("_"):
                resolved[name] = value
            elif name not in valid_parameters:
                raise ValueError(f"unknown parameter {name!r}")
            else:
                resolved[name] = value
        for name, schema in valid_parameters.items():
            resolved.setdefault(name, schema.get("default"))
            value = resolved[name]
            if schema["kind"] == "select" and value not in schema.get("choices", ()):
                raise ValueError(f"invalid value for parameter {name!r}")
            if schema["kind"] in {"integer", "number"}:
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ValueError(f"parameter {name!r} must be numeric")
                if "minimum" in schema and value < schema["minimum"]:
                    raise ValueError(f"parameter {name!r} is below its minimum")
                if "maximum" in schema and value > schema["maximum"]:
                    raise ValueError(f"parameter {name!r} is above its maximum")
        run_id = (
            f"run_{datetime.now().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )
        run = {
            "schema": "panorai-workbench-run/v2",
            "id": run_id,
            "project": "scratch",
            "experiment_id": experiment_id,
            "adapter_id": experiment_id,
            "surface_id": descriptor["surface_id"],
            "catalog": catalog_payload["catalog"],
            "workspace": workspace,
            "status": "created",
            "created_at": _now(),
            "updated_at": _now(),
            "parameters": resolved,
            "inputs": inputs,
            "runtime": runtime_fingerprint(),
            "public_api": descriptor["public_api"],
            "artifacts": [],
            "metrics": {},
            "warnings": [],
            "events": [{"at": _now(), "state": "created", "message": "Run criado"}],
        }
        _atomic_json(self.runs / run_id / "run.json", run)
        return run

    def create_component(
        self,
        *,
        component_id: str,
        configuration: dict[str, Any],
        input_values: dict[str, Any],
        inputs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        descriptor = component_index().get(component_id)
        if descriptor is None:
            raise ValueError(f"unknown component {component_id!r}")
        if descriptor["status"] != "ready":
            raise ValueError(descriptor["diagnostic"] or "component is not executable")
        configuration_names = {
            item["name"] for item in descriptor["configuration_schema"]
        }
        input_names = {item["name"] for item in descriptor["input_schema"]}
        unknown_configuration = set(configuration) - configuration_names
        unknown_values = set(input_values) - input_names
        upload_fields = {str(item.get("field")) for item in inputs}
        upload_roots = {item.split(".", 1)[0] for item in upload_fields}
        unknown_uploads = upload_roots - input_names
        if unknown_configuration:
            raise ValueError(
                "unknown configuration fields: "
                + ", ".join(sorted(unknown_configuration))
            )
        if unknown_values or unknown_uploads:
            raise ValueError(
                "unknown input fields: "
                + ", ".join(sorted(unknown_values | unknown_uploads))
            )
        for item in descriptor["input_schema"]:
            if (
                item["required"]
                and item["name"] not in input_values
                and item["name"] not in upload_roots
            ):
                raise ValueError(f"missing required input {item['name']!r}")

        catalog_payload = experiment_catalog()
        run_id = (
            f"run_{datetime.now().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )
        run = {
            "schema": "panorai-workbench-run/v3",
            "id": run_id,
            "project": "scratch",
            "experiment_id": component_id,
            "component_id": component_id,
            "adapter_id": None,
            "surface_id": descriptor["surface_id"],
            "catalog": catalog_payload["catalog"],
            "workspace": "runner",
            "status": "created",
            "created_at": _now(),
            "updated_at": _now(),
            "parameters": configuration,
            "configuration": configuration,
            "input_values": input_values,
            "inputs": inputs,
            "runtime": runtime_fingerprint(),
            "public_api": descriptor["public_api"],
            "artifacts": [],
            "metrics": {},
            "warnings": [],
            "events": [{"at": _now(), "state": "created", "message": "Run criado"}],
        }
        _atomic_json(self.runs / run_id / "run.json", run)
        return run

    def create_plan(
        self,
        *,
        steps: list[dict[str, Any]],
        inputs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        normalized = validate_plan(steps, inputs)
        final_descriptor = component_index()[normalized[-1]["component_id"]]
        catalog_payload = experiment_catalog()
        run_id = (
            f"run_{datetime.now().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )
        public_apis = [
            component_index()[step["component_id"]]["public_api"] for step in normalized
        ]
        run = {
            "schema": "panorai-workbench-run/v4",
            "id": run_id,
            "project": "scratch",
            "experiment_id": final_descriptor["id"],
            "component_id": final_descriptor["id"],
            "adapter_id": None,
            "surface_id": final_descriptor["surface_id"],
            "catalog": catalog_payload["catalog"],
            "workspace": "runner",
            "status": "created",
            "created_at": _now(),
            "updated_at": _now(),
            "parameters": {},
            "configuration": normalized[-1]["configuration"],
            "input_values": normalized[-1]["input_values"],
            "inputs": inputs,
            "plan": {"steps": normalized},
            "runtime": runtime_fingerprint(),
            "public_api": " → ".join(public_apis),
            "artifacts": [],
            "metrics": {},
            "warnings": [],
            "events": [{"at": _now(), "state": "created", "message": "Plano criado"}],
        }
        _atomic_json(self.runs / run_id / "run.json", run)
        return run

    def load(self, run_id: str) -> dict[str, Any]:
        if not run_id.startswith("run_") or "/" in run_id or ".." in run_id:
            raise KeyError(run_id)
        path = self.runs / run_id / "run.json"
        if not path.exists():
            raise KeyError(run_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, run: dict[str, Any]) -> None:
        run["updated_at"] = _now()
        _atomic_json(self.runs / run["id"] / "run.json", run)

    def list(self, *, limit: int = 50) -> list[dict[str, Any]]:
        paths = sorted(self.runs.glob("run_*/run.json"), reverse=True)[:limit]
        return [json.loads(path.read_text(encoding="utf-8")) for path in paths]

    def artifact_path(self, run_id: str, artifact_id: str) -> Path:
        run = self.load(run_id)
        item = next(
            (value for value in run["artifacts"] if value["id"] == artifact_id), None
        )
        if item is None:
            raise KeyError(artifact_id)
        path = (self.runs / run_id / item["file"]).resolve()
        if self.runs.resolve() not in path.parents:
            raise KeyError(artifact_id)
        return path

    def input_path(self, run_id: str, index: int) -> Path:
        run = self.load(run_id)
        if index < 0 or index >= len(run["inputs"]):
            raise KeyError(index)
        path = Path(run["inputs"][index]["path"]).resolve()
        if self.inputs.resolve() not in path.parents or not path.exists():
            raise KeyError(index)
        return path

    def cancel(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            run = self.load(run_id)
            if run["status"] not in TERMINAL_STATES:
                run["status"] = "cancelled"
                run["events"].append(
                    {
                        "at": _now(),
                        "state": "cancelled",
                        "message": "Cancelado pelo usuário",
                    }
                )
                self.save(run)
            return run

    def execute(self, run_id: str) -> dict[str, Any]:
        started = perf_counter()
        with self._lock:
            run = self.load(run_id)
            if run["status"] in TERMINAL_STATES:
                return run
            run["status"] = "running"
            run["worker_pid"] = os.getpid()
            run["started_at"] = _now()
            shape = (
                run["inputs"][0].get("image_shape_hw") if run.get("inputs") else None
            )
            dimensions = (
                f" · {shape[1]}×{shape[0]}"
                if isinstance(shape, list) and len(shape) == 2
                else ""
            )
            detail = (
                " · SLIC esférico integral pode levar vários minutos"
                if run["experiment_id"] in {"spherical-slic", "slic-op"}
                and run["parameters"].get("working_width") == "original"
                else ""
            )
            run["events"].append(
                {
                    "at": _now(),
                    "state": "running",
                    "message": f"Executando API pública{dimensions}{detail}",
                }
            )
            self.save(run)
        try:
            artifacts, metrics, warnings = _execute(run, self.runs / run_id)
            run = self.load(run_id)
            if run["status"] in TERMINAL_STATES:
                return run
            run["status"] = "completed"
            run["artifacts"] = artifacts
            run["metrics"] = metrics
            run["metrics"]["wall_time_s"] = round(perf_counter() - started, 3)
            run["warnings"] = warnings
            run["events"].append(
                {"at": _now(), "state": "completed", "message": "Artefatos persistidos"}
            )
        except Exception as exc:
            run = self.load(run_id)
            if run["status"] in TERMINAL_STATES:
                return run
            run["status"] = "failed"
            run["error"] = f"{type(exc).__name__}: {exc}"
            run["metrics"]["wall_time_s"] = round(perf_counter() - started, 3)
            run["events"].append(
                {"at": _now(), "state": "failed", "message": run["error"]}
            )
        run["finished_at"] = _now()
        self.save(run)
        return run


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _artifact(
    directory: Path,
    artifacts: list[dict[str, Any]],
    artifact_id: str,
    kind: str,
    label: str,
    payload: bytes | dict[str, Any],
    *,
    extension: str = "png",
) -> None:
    artifact_directory = directory / "artifacts"
    artifact_directory.mkdir(parents=True, exist_ok=True)
    filename = f"{artifact_id}.{extension}"
    path = artifact_directory / filename
    if isinstance(payload, bytes):
        path.write_bytes(payload)
        mime = "image/png" if extension == "png" else "application/octet-stream"
    else:
        path.write_text(
            json.dumps(
                _jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False
            ),
            encoding="utf-8",
        )
        mime = "application/json"
    artifacts.append(
        {
            "id": artifact_id,
            "type": kind,
            "label": label,
            "mime_type": mime,
            "file": f"artifacts/{filename}",
            "size_bytes": path.stat().st_size,
        }
    )


def _read_image(item: dict[str, Any]) -> np.ndarray:
    with Image.open(item["path"]) as image:
        image.load()
        return np.asarray(image.convert("RGB"))


def _operation_result(
    experiment_id: str, image: np.ndarray, params: dict[str, Any]
) -> tuple[Any, dict[str, Any]]:
    from panorai.ops import spherical as ops

    if experiment_id in {"filters-edges", "on-sphere"}:
        name = str(params.get("operation", "gaussian"))
        ksize = int(params.get("ksize", 5))
        sigma = float(params.get("sigma", 1.0))
        source = image.astype(np.float32) / 255.0
        choices = {
            "gaussian": lambda: ops.GaussianBlur(ksize=ksize, sigma=sigma),
            "box": lambda: ops.BoxBlur(ksize=ksize),
            "median": lambda: ops.MedianBlur(ksize=ksize),
            "bilateral": lambda: ops.BilateralFilter(
                diameter=ksize,
                sigma_color=max(sigma, 0.01),
                sigma_space=max(sigma, 0.01),
            ),
            "sobel": lambda: ops.Sobel(),
            "gradient": lambda: ops.Gradient(),
            "laplacian": lambda: ops.Laplacian(),
            "canny": lambda: ops.Canny(
                float(params.get("threshold_low", 0.1)),
                float(params.get("threshold_high", 0.3)),
                gaussian_ksize=ksize,
                gaussian_sigma=sigma,
            ),
        }
        if name not in choices:
            raise ValueError(f"unsupported filter operation {name!r}")
        operation = choices[name]()
        if experiment_id == "on-sphere":
            from panorai.data import EquirectangularImage
            from panorai.pipeline import OnSphere, Output, Pipeline

            pipeline = Pipeline(
                OnSphere(Output(operation, kind="raster"), output="raster")
            )
            run = pipeline.run(EquirectangularImage(source))
            values = run.output.values
            return values, {
                "pipeline": run.describe(),
                "operation": operation.describe(),
            }
        return operation(source), {"operation": operation.describe()}
    if experiment_id in {"spherical-slic", "slic-op"}:
        arguments = {
            "n_segments": int(params.get("n_segments", 100)),
            "compactness": float(params.get("compactness", 10.0)),
        }
        source_shape = tuple(int(value) for value in image.shape[:2])
        requested_width = str(params.get("working_width", "1024"))
        support, support_info = _input_support(
            image, str(params.get("input_support", "auto-polar-voids"))
        )
        working_image, working_support, resolution = _working_raster(
            image, support, requested_width
        )
        resized = tuple(working_image.shape[:2]) != source_shape
        if experiment_id == "spherical-slic":
            from panorai.applications import SphericalSLIC

            app = SphericalSLIC(
                **arguments, max_num_iter=int(params.get("max_num_iter", 10))
            )
            result = app.run(working_image, validity_mask=working_support)
            labels = result.values
            provenance = {"application": app.describe(), "config": app.to_config()}
        else:
            from panorai.image_processing import spherical_slic

            labels = spherical_slic(
                working_image,
                **arguments,
                mask=working_support,
            )
            provenance = {
                "public_api": "panorai.image_processing.spherical_slic",
                "parameters": arguments,
            }
        working_shape = tuple(int(value) for value in np.asarray(labels).shape[:2])
        if resized:
            labels = ops.Resize(source_shape, interpolation="nearest")(labels)
        provenance["workbench_resolution"] = {
            **resolution,
            "working_shape_hw": list(working_shape),
            "output_shape_hw": [int(value) for value in np.asarray(labels).shape[:2]],
            "restored_with": "panorai.ops.spherical.Resize(interpolation='nearest')"
            if resized
            else None,
        }
        provenance["input_support"] = support_info
        return labels, provenance
    if experiment_id in {"morphology", "region-ops"}:
        mask = _gray(image) >= float(params.get("threshold", 0.5))
        name = str(params.get("operation", "connected-components"))
        if experiment_id == "morphology":
            radius = float(params.get("radius_deg", 1.0))
            choices = {
                "dilate": ops.Dilate,
                "erode": ops.Erode,
                "opening": ops.Opening,
                "closing": ops.Closing,
            }
            if name == "fill-holes":
                operation = ops.FillHoles(max_hole_solid_angle_sr=0.01)
            else:
                operation = choices[name](ops.GeodesicDisk(radius))
            return operation(mask), {"operation": operation.describe()}
        if name == "boundaries":
            operation = ops.RegionBoundaries()
        elif name == "distance":
            operation = ops.GeodesicDistanceTransform()
        else:
            operation = ops.ConnectedComponents()
        return operation(mask), {"operation": operation.describe()}
    raise ValueError(f"no raster executor registered for {experiment_id!r}")


def _execute(
    run: dict[str, Any], directory: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    if run.get("plan"):
        from .executor import execute_plan

        return execute_plan(run, directory)
    if run.get("component_id"):
        from .executor import execute_component

        return execute_component(run, directory)

    experiment_id = run["experiment_id"]
    params = run["parameters"]
    inputs = run["inputs"]
    artifacts: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    warnings: list[str] = []
    images = [_read_image(item) for item in inputs]

    if experiment_id == "geometry-contract":
        if not images:
            raise ValueError("Geometry v1 requires one ERP")
        height, width = images[0].shape[:2]
        points = params.get("points_xy") or [[width / 2 - 0.5, height / 2 - 0.5]]
        payload = lift_erp_points(points, (height, width))
        _artifact(
            directory,
            artifacts,
            "bearings",
            "bearings",
            "Pixels e bearings",
            payload,
            extension="json",
        )
        metrics["points"] = len(payload["points_xy"])
    elif experiment_id == "gnomonic-projection":
        from panorai.geometry import GnomonicProjector, GnomonicSpec

        if not images:
            raise ValueError("Gnomonic projection requires one ERP")
        size = int(params.get("output_size", params.get("face_size", 768)))
        spec = GnomonicSpec(
            center_lon_deg=float(params.get("center_lon_deg", 0.0)),
            center_lat_deg=float(params.get("center_lat_deg", 0.0)),
            hfov_deg=float(params.get("hfov_deg", 90.0)),
            vfov_deg=float(params.get("vfov_deg", params.get("hfov_deg", 90.0))),
            output_shape_hw=(size, size),
        )
        result = GnomonicProjector(spec).project(images[0].astype(np.float32) / 255.0)
        _artifact(
            directory,
            artifacts,
            "gnomonic",
            "raster",
            "Vista gnomônica",
            _png_bytes(result.data),
        )
        _artifact(
            directory,
            artifacts,
            "support",
            "mask",
            "Suporte geométrico",
            _png_bytes(result.support_mask),
        )
        metrics["support_fraction"] = float(np.mean(result.support_mask))
    elif experiment_id == "projected-view-route":
        from panorai.data import EquirectangularImage
        from panorai.ops import spherical as ops
        from panorai.pipeline import PerView, Pipeline, ReconstructERP, SampleViews

        if not images:
            raise ValueError("Projected route requires one ERP")
        size = int(params.get("face_size", 512))
        pipeline = Pipeline(
            SampleViews(
                "fibonacci", count=6, size=size, fov=float(params.get("hfov_deg", 90.0))
            ),
            PerView(ops.Resize((size, size))),
            ReconstructERP(image="average"),
        )
        retain_views = params.get("retain", "views") == "views"
        result = pipeline.run(
            EquirectangularImage(images[0].astype(np.float32) / 255.0),
            retain=["sample_views"] if retain_views else None,
        )
        _artifact(
            directory,
            artifacts,
            "reconstruction",
            "raster",
            "ERP reconstruída",
            _png_bytes(result.output.image),
        )
        views = result["sample_views"] if retain_views else ()
        for index, face in enumerate(views):
            _artifact(
                directory,
                artifacts,
                f"view-{index}",
                "view",
                f"Vista {index + 1}",
                _png_bytes(face.image),
            )
        _artifact(
            directory,
            artifacts,
            "pipeline-trace",
            "pipeline-trace",
            "StageRecord",
            result.describe(),
            extension="json",
        )
        metrics["views"] = len(views)
    elif experiment_id == "cubemap-projection":
        from panorai.geometry import CubemapProjector, CubemapSpec

        if not images:
            raise ValueError("Cubemap projection requires one ERP")
        faces = CubemapProjector(
            CubemapSpec((int(params.get("face_size", 512)),) * 2)
        ).project(images[0].astype(np.float32) / 255.0)
        for name, result in faces.items():
            _artifact(
                directory,
                artifacts,
                f"face-{name}",
                "view",
                f"Face {name}",
                _png_bytes(result.data),
            )
            _artifact(
                directory,
                artifacts,
                f"support-{name}",
                "mask",
                f"Suporte {name}",
                _png_bytes(result.support_mask),
            )
        metrics["faces"] = len(faces)
    elif experiment_id in {"spherical-dog-features", "dog-op"}:
        if not images:
            raise ValueError("DoG requires one ERP")
        source_image = images[0]
        support, support_info = _input_support(
            source_image, str(params.get("input_support", "auto-polar-voids"))
        )
        working_image, working_support, resolution = _working_raster(
            source_image, support, str(params.get("working_width", "1024"))
        )
        source_height, source_width = source_image.shape[:2]
        working_height, working_width = working_image.shape[:2]
        scale_x = source_width / working_width
        scale_y = source_height / working_height
        if experiment_id == "spherical-dog-features":
            from panorai.applications import SphericalDoGFeatures

            app = SphericalDoGFeatures()
            config = replace(
                app.config,
                max_features=int(params.get("max_features", 1000)),
                contrast_threshold=float(
                    params.get("contrast_threshold", app.config.contrast_threshold)
                ),
            )
            configured = SphericalDoGFeatures(config)
            result = configured.run(
                working_image,
                panorama_id=inputs[0]["name"],
                validity_mask=working_support,
            )
            provenance = {
                "application": configured.describe(),
                "config": configured.to_config(),
                "workbench_resolution": resolution,
                "input_support": support_info,
            }
            observations = result.features
            rows = [
                {
                    "id": feature.feature_id,
                    "erp_xy": None
                    if feature.source_erp_xy is None
                    else (
                        np.asarray(feature.source_erp_xy) * [scale_x, scale_y]
                    ).tolist(),
                    "bearing_xyz": np.asarray(feature.bearing_xyz).tolist(),
                    "response": float(feature.response),
                    "scale": float(feature.scale),
                }
                for feature in observations
            ]
        else:
            from panorai.features import (
                SphericalDoGDetector,
                SphericalDoGDetectorConfig,
            )

            detector_config = SphericalDoGDetectorConfig(
                max_keypoints=int(params.get("max_features", 1000)),
                contrast_threshold=float(params.get("contrast_threshold", 0.012)),
            )
            detector = SphericalDoGDetector(detector_config)
            result = detector.detect(
                working_image,
                panorama_id=inputs[0]["name"],
                validity_mask=working_support,
            )
            provenance = {
                "public_api": "panorai.features.SphericalDoGDetector",
                "result": result.describe(),
                "workbench_resolution": resolution,
                "input_support": support_info,
            }
            rows = [
                {
                    "id": f"dog-{index:06d}",
                    "erp_xy": (
                        np.asarray(feature.source_erp_xy) * [scale_x, scale_y]
                    ).tolist(),
                    "bearing_xyz": np.asarray(feature.bearing_xyz).tolist(),
                    "response": float(feature.response),
                    "scale": float(feature.scale_deg),
                }
                for index, feature in enumerate(result.keypoints)
            ]
        overlay = Image.fromarray(source_image.copy())
        drawing = ImageDraw.Draw(overlay)
        for row in rows[:2000]:
            if row["erp_xy"] is None:
                continue
            x, y = row["erp_xy"]
            drawing.ellipse(
                (x - 3, y - 3, x + 3, y + 3), outline=(255, 200, 87), width=2
            )
        stream = BytesIO()
        overlay.save(stream, format="PNG")
        _artifact(
            directory,
            artifacts,
            "keypoints-overlay",
            "preview",
            "Keypoints",
            stream.getvalue(),
        )
        _artifact(
            directory,
            artifacts,
            "input-support",
            "mask",
            "Suporte da entrada",
            _png_bytes(support),
        )
        _artifact(
            directory,
            artifacts,
            "keypoints",
            "table",
            "Tabela de keypoints",
            {"features": rows},
            extension="json",
        )
        _artifact(
            directory,
            artifacts,
            "provenance",
            "log",
            "Proveniência",
            provenance,
            extension="json",
        )
        metrics["keypoints"] = len(rows)
        metrics["support_fraction"] = support_info["valid_fraction"]
        metrics["working_shape_hw"] = resolution["working_shape_hw"]
        if support_info["detected_polar_voids"]:
            warnings.append(
                "Vazios polares da captura foram excluídos por suporte explícito"
            )
    elif experiment_id == "relative-pose":
        if len(images) != 2:
            raise ValueError("Relative pose requires exactly two ERPs")
        result = analyze_pair(
            images[0],
            images[1],
            panorama_id_a=inputs[0]["name"],
            panorama_id_b=inputs[1]["name"],
            profile=str(params.get("profile", "relative-pose-reference")),
            max_angular_error_deg=float(params.get("max_angular_error_deg", 0.5)),
            random_seed=int(params.get("random_seed", 7)),
        )
        _artifact(
            directory,
            artifacts,
            "pair-analysis",
            "pose",
            "Pose e correspondências",
            result,
            extension="json",
        )
        metrics["matches"] = len(result["matches"])
        metrics["inliers"] = (
            0 if result["pose"] is None else result["pose"]["num_inliers"]
        )
    elif experiment_id == "classical-stereo":
        if len(images) != 2:
            raise ValueError("Classical stereo requires exactly two images")
        from panorai.applications import (
            ADCensusStereoDepth,
            CensusMGMStereoDepth,
            ELASStereoDepth,
            PatchMatchStereoDepth,
            StereoBMDepth,
            StereoSGBMDepth,
        )

        choices = {
            "bm": StereoBMDepth,
            "sgbm": StereoSGBMDepth,
            "ad-census": ADCensusStereoDepth,
            "census-mgm": CensusMGMStereoDepth,
            "elas": ELASStereoDepth,
            "patch-match": PatchMatchStereoDepth,
        }
        application_type = choices[str(params.get("algorithm", "sgbm"))]
        application = application_type()
        if hasattr(application.config, "max_disparity"):
            application = application_type(
                replace(
                    application.config,
                    max_disparity=int(params.get("max_disparity", 64)),
                )
            )
        result = application.run(images[0], images[1])
        _artifact(
            directory,
            artifacts,
            "disparity",
            "disparity",
            "Disparidade",
            _png_bytes(result.disparity),
        )
        _artifact(
            directory,
            artifacts,
            "validity",
            "mask",
            "Validade",
            _png_bytes(result.validity),
        )
        metrics["valid_fraction"] = float(np.mean(result.validity))
    elif experiment_id == "spherical-stereo":
        if len(images) != 2:
            raise ValueError("Spherical stereo requires exactly two ERPs")
        from panorai.applications import SphericalStereoDepth
        from panorai.stereo import SphericalStereoOptions

        options = SphericalStereoOptions(
            min_range=float(params.get("min_range", 0.3)),
            max_range=float(params.get("max_range", 12.0)),
            num_hypotheses=int(params.get("num_hypotheses", 96)),
        )
        result = SphericalStereoDepth(options).run(
            images[0],
            images[1],
            np.eye(3),
            np.array([float(params.get("baseline", 1.0)), 0.0, 0.0]),
        )
        _artifact(
            directory,
            artifacts,
            "range",
            "range",
            "Range radial",
            _png_bytes(result.range),
        )
        _artifact(
            directory,
            artifacts,
            "confidence",
            "confidence",
            "Confiança",
            _png_bytes(result.confidence),
        )
        _artifact(
            directory,
            artifacts,
            "validity",
            "mask",
            "Validade",
            _png_bytes(result.validity_mask),
        )
        metrics["valid_fraction"] = float(np.mean(result.validity_mask))
    elif experiment_id == "region-analysis-pipeline":
        from panorai.ops import spherical as ops
        from panorai.pipeline import AnalyzeRegions, Pipeline, RasterField, RasterGrid

        if not images:
            raise ValueError("Region analysis requires one ERP")
        labels = ops.ConnectedComponents()(
            _gray(images[0]) >= float(params.get("threshold", 0.5))
        )
        height, width = labels.shape
        support = np.ones((height, width), dtype=bool)
        field = RasterField(
            "labels",
            labels,
            "labels",
            support,
            support,
            RasterGrid("erp", (height, width), (height, width), "pixel-centres"),
        )
        result = Pipeline(AnalyzeRegions()).run(field)
        output = result.output
        _artifact(
            directory,
            artifacts,
            "labels",
            "labels",
            "Regiões",
            _png_bytes(labels, labels=True),
        )
        boundary_mask = ops.RegionBoundaries()(labels)
        _artifact(
            directory,
            artifacts,
            "boundaries",
            "mask",
            "Fronteiras",
            _png_bytes(boundary_mask),
        )
        _artifact(
            directory,
            artifacts,
            "regions",
            "table",
            "Medidas e adjacência",
            {"measurements": output.measurements, "adjacency": output.adjacency},
            extension="json",
        )
        _artifact(
            directory,
            artifacts,
            "pipeline-trace",
            "pipeline-trace",
            "StageRecord",
            result.describe(),
            extension="json",
        )
        metrics["regions"] = len(output.measurements.regions)
    elif experiment_id in {
        "filters-edges",
        "on-sphere",
        "spherical-slic",
        "slic-op",
        "morphology",
        "region-ops",
    }:
        if not images:
            raise ValueError("This experiment requires at least one ERP")
        target = str(params.get("_target", "a"))
        selected = (
            list(enumerate(images))
            if target == "both"
            else [
                (
                    1 if target == "b" and len(images) > 1 else 0,
                    images[1] if target == "b" and len(images) > 1 else images[0],
                )
            ]
        )
        for index, image in selected:
            values, provenance = _operation_result(experiment_id, image, params)
            labels = (
                experiment_id
                in {
                    "spherical-slic",
                    "slic-op",
                    "region-ops",
                    "region-analysis-pipeline",
                }
                and np.asarray(values).dtype != np.bool_
            )
            name = "a" if index == 0 else "b"
            _artifact(
                directory,
                artifacts,
                f"result-{name}",
                "labels" if labels else "raster",
                f"Resultado {name.upper()}",
                _png_bytes(values, labels=labels),
            )
            if labels:
                label_values = np.asarray(values)
                metrics[f"regions_{name}"] = int(
                    np.unique(label_values[label_values >= 0]).size
                )
            if experiment_id in {"spherical-slic", "slic-op"}:
                support, support_info = _input_support(
                    image, str(params.get("input_support", "auto-polar-voids"))
                )
                _artifact(
                    directory,
                    artifacts,
                    f"input-support-{name}",
                    "mask",
                    f"Suporte {name.upper()}",
                    _png_bytes(support),
                )
                metrics[f"support_fraction_{name}"] = support_info["valid_fraction"]
                if support_info["detected_polar_voids"]:
                    warnings.append(
                        f"Vazios polares da entrada {name.upper()} foram excluídos por suporte explícito"
                    )
            _artifact(
                directory,
                artifacts,
                f"provenance-{name}",
                "log",
                f"Proveniência {name.upper()}",
                provenance,
                extension="json",
            )
            resolution = provenance.get("workbench_resolution")
            if resolution:
                metrics[f"working_shape_{name}"] = resolution["working_shape_hw"]
                if resolution["source_shape_hw"] != resolution["working_shape_hw"]:
                    warnings.append(
                        "SLIC executado em resolução de trabalho e restaurado à ERP com nearest-neighbour"
                    )
        metrics["outputs"] = len(selected)
    else:
        descriptor = adapter_for(experiment_id)
        raise ValueError(
            f"adapter {descriptor['adapter_status']!r} requires inputs or dependencies not exposed by this generic runner"
        )
    return artifacts, metrics, warnings


STORE = RunStore()

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="panorai-workbench")
_FUTURES: dict[str, Future[dict[str, Any]]] = {}


def schedule_run(run_id: str) -> dict[str, Any]:
    """Queue a run on the single local device worker and return immediately."""

    future = _EXECUTOR.submit(STORE.execute, run_id)
    _FUTURES[run_id] = future
    future.add_done_callback(lambda _future: _FUTURES.pop(run_id, None))
    return STORE.load(run_id)


def cancel_run(run_id: str) -> dict[str, Any]:
    future = _FUTURES.get(run_id)
    if future is not None:
        future.cancel()
    return STORE.cancel(run_id)


__all__ = ["RunStore", "STORE", "TERMINAL_STATES", "schedule_run", "cancel_run"]
