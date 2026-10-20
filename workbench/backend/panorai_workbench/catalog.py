"""Workbench projection of the catalog shipped by the active PanorAi install.

PanorAi owns public-surface discovery and documentation.  The Workbench owns
only an allow-listed adapter overlay keyed by stable ``surface_id`` values.
Nothing received from the browser is ever interpreted as an import path or
callable name.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from importlib import import_module
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any

from panorai.catalog import PublicCatalog, public_catalog


def _field(
    name: str,
    label: str,
    kind: str,
    default: Any,
    **options: Any,
) -> dict[str, Any]:
    return {"name": name, "label": label, "kind": kind, "default": default, **options}


def _adapter(
    *,
    input_modes: tuple[str, ...],
    status: str,
    artifacts: tuple[str, ...],
    targets: tuple[str, ...],
    parameters: tuple[dict[str, Any], ...] = (),
    preset: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "input_modes": list(input_modes),
        "adapter_status": status,
        "parameter_schema": list(parameters),
        "artifact_types": list(artifacts),
        "workspace_targets": list(targets),
        "default_preset": preset or {},
    }


def _legacy_experiment(
    *,
    surface_id: str,
    public_symbol: str,
    title: str,
    runner: bool = False,
    automatic_surface: str | None = None,
    automatic_symbol: str | None = None,
    **adapter: Any,
) -> dict[str, Any]:
    """Describe one preserved URL/UI contract without duplicating API inventory."""

    return {
        **_adapter(**adapter),
        "surface_id": surface_id,
        "public_symbol": public_symbol,
        "title": title,
        "legacy_runner": runner,
        "automatic_target": (
            None
            if automatic_symbol is None
            else {
                "surface_id": automatic_surface or surface_id,
                "symbol": automatic_symbol,
            }
        ),
    }


_SINGLE_TARGETS = ("single", "two")
_IMAGE_OUTPUTS = ("raster", "preview", "metrics", "log")

# This is the versioned Workbench execution contract.  It deliberately names
# only known public PanorAi symbols; the browser never supplies an import path.
LEGACY_EXPERIMENTS: dict[str, dict[str, Any]] = {
    "spherical-slic": _legacy_experiment(
        surface_id="spherical-applications-slic",
        public_symbol="SphericalSLIC",
        title="Spherical SLIC",
        runner=True,
        automatic_symbol="SphericalSLIC",
        input_modes=("single",),
        status="ready",
        artifacts=("labels", "preview", "log"),
        targets=_SINGLE_TARGETS,
        parameters=(
            _field("n_segments", "Segmentos", "integer", 100, minimum=2, maximum=5000),
            _field(
                "compactness",
                "Compacidade",
                "number",
                10.0,
                minimum=0.01,
                maximum=100.0,
            ),
            _field("max_num_iter", "Iterações", "integer", 10, minimum=1, maximum=100),
            _field(
                "working_width",
                "Largura de trabalho",
                "select",
                "1024",
                choices=["512", "1024", "2048", "original"],
            ),
            _field(
                "input_support",
                "Suporte da entrada",
                "select",
                "auto-polar-voids",
                choices=["auto-polar-voids", "full-sphere"],
            ),
        ),
    ),
    "spherical-dog-features": _legacy_experiment(
        surface_id="spherical-applications",
        public_symbol="SphericalDoGFeatures",
        title="Spherical DoG Features",
        runner=True,
        automatic_symbol="SphericalDoGFeatures",
        input_modes=("single",),
        status="ready",
        artifacts=("keypoints", "preview", "table", "log"),
        targets=_SINGLE_TARGETS,
        parameters=(
            _field(
                "max_features",
                "Máximo de features",
                "integer",
                1000,
                minimum=10,
                maximum=10000,
            ),
            _field(
                "contrast_threshold",
                "Contraste mínimo",
                "number",
                0.012,
                minimum=0.0001,
                maximum=1.0,
            ),
            _field(
                "working_width",
                "Largura de trabalho",
                "select",
                "1024",
                choices=["512", "1024", "2048", "original"],
            ),
            _field(
                "input_support",
                "Suporte da entrada",
                "select",
                "auto-polar-voids",
                choices=["auto-polar-voids", "full-sphere"],
            ),
        ),
    ),
    "relative-pose": _legacy_experiment(
        surface_id="spherical-relative-pose",
        public_symbol="estimate_relative_pose",
        title="Spherical Relative Pose",
        runner=True,
        automatic_surface="spherical-applications",
        automatic_symbol="SphericalRelativePose",
        input_modes=("pair",),
        status="ready",
        artifacts=("keypoints", "matches", "pose", "table", "log"),
        targets=("two",),
        parameters=(
            _field(
                "profile",
                "Perfil",
                "select",
                "relative-pose-reference",
                choices=[
                    "relative-pose-reference",
                    "sift-flann",
                    "sift-bf",
                    "orb-hamming",
                    "akaze-hamming",
                ],
            ),
            _field(
                "max_angular_error_deg",
                "Erro angular máximo",
                "number",
                0.5,
                minimum=0.01,
                maximum=10.0,
            ),
            _field("random_seed", "Seed", "integer", 7, minimum=0, maximum=2147483647),
        ),
    ),
    "spherical-stereo": _legacy_experiment(
        surface_id="spherical-applications",
        public_symbol="SphericalStereoDepth",
        title="Spherical Stereo Depth",
        runner=True,
        automatic_symbol="SphericalStereoDepth",
        input_modes=("pair",),
        status="ready",
        artifacts=("range", "confidence", "validity", "preview", "log"),
        targets=("two",),
        parameters=(
            _field("min_range", "Range mínimo", "number", 0.3, minimum=0.01),
            _field("max_range", "Range máximo", "number", 12.0, minimum=0.02),
            _field(
                "num_hypotheses", "Hipóteses", "integer", 96, minimum=8, maximum=512
            ),
            _field("baseline", "Baseline", "number", 1.0, minimum=0.0001),
        ),
    ),
    "full-sphere-stereo": _legacy_experiment(
        surface_id="registered-full-sphere-stereo-application",
        public_symbol="FullSphereDenseStereo",
        title="Full Sphere Dense Stereo",
        input_modes=("pair", "multi"),
        status="requires-input",
        artifacts=("range", "support", "confidence", "preview", "log"),
        targets=("two", "multi"),
        parameters=(_field("baseline", "Baseline", "number", 1.0, minimum=0.0001),),
    ),
    "panoramic-stereo": _legacy_experiment(
        surface_id="spherical-applications",
        public_symbol="PanoramicDenseStereo",
        title="Panoramic Dense Stereo",
        input_modes=("pair",),
        status="requires-input",
        artifacts=("range", "confidence", "preview", "pipeline-trace"),
        targets=("two",),
        parameters=(
            _field(
                "route",
                "Rota",
                "select",
                "rectified-views",
                choices=["rectified-views", "direct-spherical"],
            ),
            _field("baseline", "Baseline", "number", 1.0, minimum=0.0001),
        ),
    ),
    "classical-stereo": _legacy_experiment(
        surface_id="spherical-applications",
        public_symbol="StereoSGBMDepth",
        title="Classical Stereo Family",
        runner=True,
        input_modes=("pair",),
        status="ready",
        artifacts=("disparity", "validity", "preview", "log"),
        targets=("two",),
        parameters=(
            _field(
                "algorithm",
                "Algoritmo",
                "select",
                "sgbm",
                choices=[
                    "bm",
                    "sgbm",
                    "ad-census",
                    "census-mgm",
                    "elas",
                    "patch-match",
                ],
            ),
            _field(
                "max_disparity",
                "Disparidade máxima",
                "integer",
                64,
                minimum=16,
                maximum=512,
                step=16,
            ),
        ),
    ),
    "mask-rcnn-workflow": _legacy_experiment(
        surface_id="spherical-mask-rcnn-taxonomy",
        public_symbol="SphericalMaskRCNN",
        title="Spherical Mask R-CNN",
        input_modes=("single",),
        status="requires-input",
        artifacts=("instances", "masks", "preview", "table"),
        targets=_SINGLE_TARGETS,
        parameters=(
            _field(
                "score_threshold",
                "Score mínimo",
                "number",
                0.5,
                minimum=0.0,
                maximum=1.0,
            ),
        ),
    ),
    "pipeline-core": _legacy_experiment(
        surface_id="spherical-pipeline-core",
        public_symbol="Pipeline",
        title="Typed Pipeline Core",
        input_modes=("single", "pair"),
        status="inspect",
        artifacts=("pipeline-trace", "config", "log"),
        targets=("single", "two"),
        parameters=(_field("retain", "Intermediários", "text", ""),),
    ),
    "on-sphere": _legacy_experiment(
        surface_id="spherical-pipeline-core",
        public_symbol="OnSphere",
        title="OnSphere",
        runner=True,
        input_modes=("single",),
        status="ready",
        artifacts=_IMAGE_OUTPUTS + ("pipeline-trace",),
        targets=_SINGLE_TARGETS,
        parameters=(
            _field(
                "operation",
                "Operação",
                "select",
                "gaussian",
                choices=["gaussian", "sobel", "laplacian", "canny"],
            ),
            _field("sigma", "Sigma", "number", 1.0, minimum=0.01, maximum=20.0),
        ),
    ),
    "projected-view-route": _legacy_experiment(
        surface_id="spherical-pipeline-core",
        public_symbol="SampleViews",
        title="Sample → PerView → Reconstruct",
        runner=True,
        input_modes=("single",),
        status="ready",
        artifacts=("views", "raster", "support", "pipeline-trace"),
        targets=_SINGLE_TARGETS,
        parameters=(
            _field(
                "hfov_deg", "FOV horizontal", "number", 90.0, minimum=5.0, maximum=150.0
            ),
            _field(
                "face_size",
                "Resolução da face",
                "integer",
                512,
                minimum=64,
                maximum=2048,
            ),
            _field(
                "retain", "Intermediários", "select", "views", choices=["none", "views"]
            ),
        ),
    ),
    "region-analysis-pipeline": _legacy_experiment(
        surface_id="spherical-pipeline-core",
        public_symbol="registered_pipeline_components",
        title="Spherical Region Analysis",
        runner=True,
        input_modes=("single",),
        status="ready",
        artifacts=("labels", "boundaries", "table", "pipeline-trace"),
        targets=_SINGLE_TARGETS,
        parameters=(
            _field("threshold", "Limiar", "number", 0.5, minimum=0.0, maximum=1.0),
        ),
    ),
    "cam-sam-pipeline": _legacy_experiment(
        surface_id="spherical-pipeline-semantic-instances",
        public_symbol="CAMGuidedSphericalInstances",
        title="CAM → Proposals → SAM → Fusion",
        input_modes=("single",),
        status="requires-input",
        artifacts=("cam", "proposals", "masks", "instances", "pipeline-trace"),
        targets=_SINGLE_TARGETS,
        parameters=(
            _field(
                "max_proposals",
                "Máximo de propostas",
                "integer",
                24,
                minimum=1,
                maximum=256,
            ),
        ),
    ),
    "filters-edges": _legacy_experiment(
        surface_id="spherical-ops-core",
        public_symbol="GaussianBlur",
        title="Filters and Edges",
        runner=True,
        input_modes=("single",),
        status="ready",
        artifacts=_IMAGE_OUTPUTS,
        targets=_SINGLE_TARGETS,
        parameters=(
            _field(
                "operation",
                "Operação",
                "select",
                "gaussian",
                choices=[
                    "gaussian",
                    "box",
                    "median",
                    "bilateral",
                    "sobel",
                    "gradient",
                    "laplacian",
                    "canny",
                ],
            ),
            _field("ksize", "Kernel", "integer", 5, minimum=3, maximum=31, step=2),
            _field("sigma", "Sigma", "number", 1.0, minimum=0.01, maximum=20.0),
            _field(
                "threshold_low", "Canny baixo", "number", 0.1, minimum=0.0, maximum=1.0
            ),
            _field(
                "threshold_high", "Canny alto", "number", 0.3, minimum=0.0, maximum=1.0
            ),
        ),
    ),
    "morphology": _legacy_experiment(
        surface_id="spherical-ops-core",
        public_symbol="functional",
        title="Spherical Morphology",
        runner=True,
        input_modes=("single",),
        status="ready",
        artifacts=("mask", "preview", "metrics"),
        targets=_SINGLE_TARGETS,
        parameters=(
            _field(
                "operation",
                "Operação",
                "select",
                "dilate",
                choices=["dilate", "erode", "opening", "closing", "fill-holes"],
            ),
            _field(
                "radius_deg",
                "Raio geodésico",
                "number",
                1.0,
                minimum=0.01,
                maximum=30.0,
            ),
            _field("threshold", "Limiar", "number", 0.5, minimum=0.0, maximum=1.0),
        ),
    ),
    "region-ops": _legacy_experiment(
        surface_id="spherical-ops-core",
        public_symbol="functional",
        title="Region Analysis Ops",
        runner=True,
        input_modes=("single",),
        status="ready",
        artifacts=("labels", "boundaries", "table", "preview"),
        targets=_SINGLE_TARGETS,
        parameters=(
            _field(
                "operation",
                "Operação",
                "select",
                "connected-components",
                choices=["connected-components", "boundaries", "distance", "measure"],
            ),
            _field("threshold", "Limiar", "number", 0.5, minimum=0.0, maximum=1.0),
        ),
    ),
    "slic-op": _legacy_experiment(
        surface_id="spherical-image-processing-slic",
        public_symbol="spherical_slic",
        title="SLIC",
        runner=True,
        automatic_symbol="spherical_slic",
        input_modes=("single",),
        status="ready",
        artifacts=("labels", "preview", "metrics"),
        targets=_SINGLE_TARGETS,
        parameters=(
            _field("n_segments", "Segmentos", "integer", 100, minimum=2, maximum=5000),
            _field(
                "compactness",
                "Compacidade",
                "number",
                10.0,
                minimum=0.01,
                maximum=100.0,
            ),
            _field(
                "working_width",
                "Largura de trabalho",
                "select",
                "1024",
                choices=["512", "1024", "2048", "original"],
            ),
            _field(
                "input_support",
                "Suporte da entrada",
                "select",
                "auto-polar-voids",
                choices=["auto-polar-voids", "full-sphere"],
            ),
        ),
    ),
    "dog-op": _legacy_experiment(
        surface_id="spherical-features-dog-detector",
        public_symbol="SphericalDoGDetector",
        title="DoG Detector",
        runner=True,
        input_modes=("single",),
        status="ready",
        artifacts=("keypoints", "preview", "table"),
        targets=_SINGLE_TARGETS,
        parameters=(
            _field(
                "max_features",
                "Máximo de features",
                "integer",
                1000,
                minimum=10,
                maximum=10000,
            ),
            _field(
                "contrast_threshold",
                "Contraste mínimo",
                "number",
                0.012,
                minimum=0.0001,
                maximum=1.0,
            ),
            _field(
                "working_width",
                "Largura de trabalho",
                "select",
                "1024",
                choices=["512", "1024", "2048", "original"],
            ),
            _field(
                "input_support",
                "Suporte da entrada",
                "select",
                "auto-polar-voids",
                choices=["auto-polar-voids", "full-sphere"],
            ),
        ),
    ),
    "torch-layers": _legacy_experiment(
        surface_id="spherical-ops-torch",
        public_symbol="Conv2d",
        title="Spherical Torch Layers",
        input_modes=("single",),
        status="requires-input",
        artifacts=("tensor", "raster", "metrics"),
        targets=_SINGLE_TARGETS,
        parameters=(
            _field(
                "layer",
                "Layer",
                "select",
                "conv2d",
                choices=["conv2d", "pool", "resize", "rotate"],
            ),
        ),
    ),
    "geometry-contract": _legacy_experiment(
        surface_id="canonical-geometry",
        public_symbol="erp_pixels_to_rays",
        title="Geometry v1",
        runner=True,
        input_modes=("single",),
        status="ready",
        artifacts=("bearings", "points", "table", "log"),
        targets=_SINGLE_TARGETS,
        parameters=(),
    ),
    "gnomonic-projection": _legacy_experiment(
        surface_id="canonical-geometry",
        public_symbol="GnomonicProjector",
        title="Gnomonic Projection",
        runner=True,
        automatic_symbol="equirectangular_to_gnomonic",
        input_modes=("single",),
        status="ready",
        artifacts=("raster", "mask"),
        targets=_SINGLE_TARGETS,
        parameters=(
            _field(
                "center_lon_deg",
                "Longitude",
                "number",
                0.0,
                minimum=-180.0,
                maximum=180.0,
            ),
            _field(
                "center_lat_deg", "Latitude", "number", 0.0, minimum=-90.0, maximum=90.0
            ),
            _field(
                "hfov_deg", "FOV horizontal", "number", 90.0, minimum=5.0, maximum=150.0
            ),
            _field(
                "vfov_deg", "FOV vertical", "number", 90.0, minimum=5.0, maximum=150.0
            ),
            _field(
                "output_size", "Resolução", "integer", 768, minimum=64, maximum=2048
            ),
        ),
    ),
    "cubemap-projection": _legacy_experiment(
        surface_id="canonical-geometry",
        public_symbol="CubemapProjector",
        title="Cubemap Projection",
        runner=True,
        automatic_symbol="equirectangular_to_cubemap",
        input_modes=("single",),
        status="ready",
        artifacts=("view", "mask"),
        targets=_SINGLE_TARGETS,
        parameters=(
            _field(
                "face_size",
                "Resolução da face",
                "integer",
                512,
                minimum=64,
                maximum=2048,
            ),
        ),
    ),
}

_adapter_index: dict[str, list[str]] = defaultdict(list)
for _adapter_id, _adapter in LEGACY_EXPERIMENTS.items():
    _adapter_index[_adapter["surface_id"]].append(_adapter_id)
_LEGACY_BY_SURFACE: dict[str, tuple[str, ...]] = {
    surface_id: tuple(adapter_ids) for surface_id, adapter_ids in _adapter_index.items()
}


class AdapterContractError(ValueError):
    """Raised when a local adapter points outside an installed catalog."""


def adapter_for(experiment_id: str) -> dict[str, Any]:
    """Return defensive adapter metadata for one allow-listed legacy id."""

    return {**LEGACY_EXPERIMENTS[experiment_id], "id": experiment_id}


def validate_adapter_contract(
    catalog: PublicCatalog | None = None,
    *,
    strict: bool = True,
) -> tuple[str, ...]:
    """Validate local surface references without hiding stale adapters."""

    installed = catalog or public_catalog()
    known = {surface.id for surface in installed.iter_surfaces(include_internal=True)}
    issues = tuple(
        f"adapter {adapter_id!r} references unknown surface_id {surface_id!r}"
        for adapter_id, adapter in LEGACY_EXPERIMENTS.items()
        for surface_id in (adapter["surface_id"],)
        if surface_id not in known
    )
    if strict and issues:
        raise AdapterContractError("; ".join(issues))
    return issues


def _resolve_public_symbol(
    module_name: str | None, symbol: str
) -> tuple[bool, str | None]:
    if module_name is None:
        return False, "surface has no public import"
    try:
        value: Any = import_module(module_name)
        for name in symbol.split("."):
            value = getattr(value, name)
        return True, None
    except (AttributeError, ImportError, ModuleNotFoundError) as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _dependency_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for distribution_name in ("numpy", "opencv-python-headless", "torch"):
        try:
            result[distribution_name] = metadata.version(distribution_name)
        except metadata.PackageNotFoundError:
            result[distribution_name] = None
    return result


def runtime_fingerprint() -> dict[str, Any]:
    import panorai

    module_path = Path(panorai.__file__).resolve()
    install_type = "source-checkout"
    direct_url: dict[str, Any] | None = None
    try:
        distribution = metadata.distribution("panorai")
        raw_direct_url = distribution.read_text("direct_url.json")
        if raw_direct_url:
            direct_url = json.loads(raw_direct_url)
        if direct_url and direct_url.get("dir_info", {}).get("editable"):
            install_type = "editable"
        elif direct_url and direct_url.get("vcs_info"):
            install_type = "vcs-install"
        elif "site-packages" in str(module_path) or "dist-packages" in str(module_path):
            install_type = "installed-distribution"
    except (metadata.PackageNotFoundError, json.JSONDecodeError):
        pass

    # Catalog discovery must stay lightweight and must not initialize optional
    # compute runtimes merely to draw a status pill. Individual runners own
    # backend/device initialization when they are explicitly executed.
    devices: dict[str, bool] = {"cpu": True, "mps": False, "cuda": False}

    version = panorai.__version__
    source_commit = None
    if "+g" in version:
        source_commit = version.split("+g", 1)[1].split(".", 1)[0]

    return {
        "schema": "panorai-workbench-runtime/v1",
        "panorai": {
            "version": version,
            "source_commit": source_commit,
            "module_path": str(module_path),
            "installation_type": install_type,
            "direct_url": direct_url,
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": platform.platform(),
        "dependencies": _dependency_versions(),
        "devices": devices,
    }


def _docs_payload(
    catalog: PublicCatalog,
    surface_id: str,
    component: str | None = None,
) -> dict[str, Any]:
    try:
        context = catalog.documentation_for(surface_id, component=component)
    except (KeyError, ValueError):
        context = catalog.documentation_for(surface_id)
    return {
        "primary": None if context.primary is None else asdict(context.primary),
        "learn": [asdict(link) for link in context.learn],
        "help": [asdict(link) for link in context.help],
        "context": [asdict(link) for link in context.context],
    }


def _surface_availability(
    surface: Any,
    dependency_versions: dict[str, str | None],
) -> tuple[str, str | None]:
    missing: list[str] = []
    for name in surface.requires:
        version = dependency_versions.get(name)
        if name not in dependency_versions:
            try:
                version = metadata.version(name)
            except metadata.PackageNotFoundError:
                version = None
            dependency_versions[name] = version
        if version is None:
            missing.append(name)
    if missing:
        return "dependency-missing", f"dependência ausente: {', '.join(missing)}"
    if surface.canonical_import is None:
        return "not-in-runtime", "surface has no public import"
    # The catalog is embedded in the exact installed distribution and is the
    # authority for declared imports. Importing every surface here would load
    # optional Torch/Open3D runtimes just to render a list.
    return "available", None


def _surface_title(surface: Any) -> str:
    return surface.id.replace("-", " ").title().replace("Api", "API")


def experiment_catalog(catalog: PublicCatalog | None = None) -> dict[str, Any]:
    """Return installed surfaces and the separate Workbench adapter overlay."""

    installed = catalog or public_catalog()
    from .discovery import discover_components

    runtime = runtime_fingerprint()
    dependencies = runtime["dependencies"]
    families = [dict(item) for item in installed.families]
    family_by_id = {item["id"]: item for item in families}
    surfaces: list[dict[str, Any]] = []
    surface_by_id: dict[str, dict[str, Any]] = {}
    catalog_surfaces = list(installed.iter_surfaces())
    for surface in catalog_surfaces:
        availability, diagnostic = _surface_availability(surface, dependencies)
        family = family_by_id.get(surface.family, {})
        row = {
            **surface.as_dict(),
            "title": _surface_title(surface),
            "summary": family.get("summary", "Public PanorAi surface."),
            "roles": list(surface.roles),
            "public_symbols": list(surface.public_symbols),
            "availability": availability,
            "diagnostic": diagnostic,
            "adapter_ids": list(_LEGACY_BY_SURFACE.get(surface.id, ())),
            "documentation": _docs_payload(installed, surface.id),
        }
        surfaces.append(row)
        surface_by_id[surface.id] = row

    components = discover_components(installed)
    component_by_target = {
        (item["surface_id"], item["symbol"]): item for item in components
    }
    component_ids_by_surface: dict[str, list[str]] = {}
    for component in components:
        component_ids_by_surface.setdefault(component["surface_id"], []).append(
            component["id"]
        )
    for surface in surfaces:
        surface["component_ids"] = component_ids_by_surface.get(surface["id"], [])

    contract_issues = validate_adapter_contract(installed, strict=False)
    issue_by_adapter = {issue.split("'", 2)[1]: issue for issue in contract_issues}
    experiments: list[dict[str, Any]] = []
    for experiment_id in LEGACY_EXPERIMENTS:
        adapter = adapter_for(experiment_id)
        surface = surface_by_id.get(adapter["surface_id"])
        raw_status = adapter["adapter_status"]
        diagnostic = issue_by_adapter.get(experiment_id)
        symbol_available = False
        if surface is not None:
            public_symbols = set(surface["public_symbols"])
            if adapter["public_symbol"] not in public_symbols:
                diagnostic = (
                    f"símbolo {adapter['public_symbol']!r} não pertence à superfície "
                    f"{adapter['surface_id']!r} deste catálogo"
                )
            elif raw_status == "ready":
                symbol_available, symbol_diagnostic = _resolve_public_symbol(
                    surface.get("canonical_import"), adapter["public_symbol"]
                )
                if not symbol_available:
                    diagnostic = symbol_diagnostic
            else:
                symbol_available = True
        status = raw_status
        if (
            surface is None
            or surface.get("availability") != "available"
            or not symbol_available
        ):
            status = "blocked"
        elif raw_status == "ready" and not adapter["legacy_runner"]:
            status = "blocked"
            diagnostic = "nenhum executor de compatibilidade está registrado"
        automatic_target = adapter.get("automatic_target")
        automatic_component = (
            None
            if automatic_target is None
            else component_by_target.get(
                (automatic_target["surface_id"], automatic_target["symbol"])
            )
        )
        automatic_ready = bool(
            automatic_component and automatic_component["status"] == "ready"
        )
        public_api = (
            f"{surface['canonical_import']}.{adapter['public_symbol']}"
            if surface and surface.get("canonical_import")
            else adapter["public_symbol"]
        )
        support = {
            "ready": "execute",
            "requires-input": "planned",
            "inspect": "inspect",
            "blocked": "inspect",
        }[status]
        experiments.append(
            {
                **adapter,
                "family": None if surface is None else surface["family"],
                "domain": None if surface is None else surface["domain"],
                "stability": None if surface is None else surface["tier"],
                "summary": (
                    "Adapter local para executar uma superfície pública do PanorAi."
                    if surface is not None
                    else "Adapter local com referência de catálogo inválida."
                ),
                "public_api": public_api,
                "availability": (
                    "not-in-runtime" if surface is None else surface["availability"]
                ),
                "adapter_status": status,
                "workbench_support": support,
                "execution_path": (
                    "automatic" if automatic_ready else "legacy-compatibility"
                ),
                "component_id": (
                    automatic_component["id"] if automatic_ready else None
                ),
                "automatic_route": (
                    automatic_component["route"] if automatic_ready else None
                ),
                "diagnostic": diagnostic
                or (None if surface is None else surface["diagnostic"]),
                "documentation": (
                    None
                    if surface is None
                    else _docs_payload(
                        installed, adapter["surface_id"], adapter["public_symbol"]
                    )
                ),
                "route": (
                    f"/single-view.html?experiment={experiment_id}"
                    if "single" in adapter["workspace_targets"]
                    else f"/two-view.html?experiment={experiment_id}"
                ),
            }
        )

    return {
        "schema": "panorai-workbench-catalog/v2",
        "catalog": {
            "interface": installed.interface,
            "contract_id": installed.contract_id,
            "release": installed.release,
            "series": installed.series,
        },
        "runtime": runtime,
        "documentation_site": installed.documentation_site.as_dict(),
        "families": families,
        "surfaces": surfaces,
        "components": components,
        "experiments": experiments,
        "contract_issues": list(contract_issues),
        "counts": {
            "surfaces": len(surfaces),
            "available_surfaces": sum(
                item["availability"] == "available" for item in surfaces
            ),
            "experiments": len(experiments),
            "components": len(components),
            "automatic_ready": sum(item["status"] == "ready" for item in components),
            "executable": sum(
                item["availability"] == "available"
                and item["adapter_status"] == "ready"
                for item in experiments
            ),
        },
    }


def workspace_catalog(workspace: str) -> dict[str, Any]:
    if workspace not in {"single", "two"}:
        raise ValueError("workspace must be 'single' or 'two'")
    payload = experiment_catalog()
    experiments = [
        item
        for item in payload["experiments"]
        if workspace in item["workspace_targets"]
    ]
    return {
        "schema": "panorai-workbench-workspace-catalog/v2",
        "workspace": workspace,
        "catalog": payload["catalog"],
        "runtime": payload["runtime"],
        "experiments": experiments,
        "counts": {
            "experiments": len(experiments),
            "ready": sum(
                item["availability"] == "available"
                and item["adapter_status"] == "ready"
                for item in experiments
            ),
        },
    }


__all__ = [
    "AdapterContractError",
    "LEGACY_EXPERIMENTS",
    "adapter_for",
    "experiment_catalog",
    "runtime_fingerprint",
    "validate_adapter_contract",
    "workspace_catalog",
]
