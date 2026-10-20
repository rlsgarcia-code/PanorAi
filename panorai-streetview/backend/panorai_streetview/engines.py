"""Allow-listed inspection of documented PanorAi engine surfaces."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


ENGINE_SPECS = (
    ("geometry", "panorai.geometry", "erp_pixels_to_rays", "stable"),
    ("features", "panorai.features", "SphericalFeaturePipeline", "stable"),
    (
        "relative_pose",
        "panorai.estimators",
        "SphericalRelativePoseEstimator",
        "experimental",
    ),
    (
        "global_mapper",
        "panorai.reconstruction",
        "SphericalGlobalMapper",
        "experimental",
    ),
    ("dense_stereo", "panorai.stereo", "SphericalDenseStereo", "experimental"),
    (
        "dense_reconstruction",
        "panorai.applications",
        "SphericalDenseReconstruction",
        "experimental",
    ),
    (
        "camera_calibration",
        "panorai.calibration",
        "calibrate_rotation_pair",
        "experimental",
    ),
    (
        "perspective_stitching",
        "panorai.applications",
        "PanoramaStitcher",
        "experimental",
    ),
)


def _distribution_version() -> str:
    try:
        return version("panorai")
    except PackageNotFoundError:
        import panorai

        return str(panorai.__version__)


def runtime_fingerprint() -> dict[str, Any]:
    import panorai

    return {
        "distribution": "panorai",
        "version": _distribution_version(),
        "module_file": str(Path(panorai.__file__).resolve()),
    }


def engine_capabilities() -> dict[str, Any]:
    engines: dict[str, Any] = {}
    for engine_id, module_name, symbol, stability in ENGINE_SPECS:
        available = False
        diagnostic = None
        try:
            module = import_module(module_name)
            available = hasattr(module, symbol)
            if not available:
                diagnostic = f"public symbol {module_name}.{symbol} is absent"
        except (ImportError, ModuleNotFoundError) as exc:
            diagnostic = f"public module {module_name} is unavailable: {exc}"
        engines[engine_id] = {
            "available": available,
            "public_api": f"{module_name}.{symbol}",
            "stability": stability,
            "diagnostic": diagnostic,
        }
    return {
        "schema": "panorai-streetview-engine-capabilities/v1",
        "runtime": runtime_fingerprint(),
        "policy": (
            "Computer-vision engines are invoked only through documented PanorAi public APIs. "
            "Missing calibration or stitching APIs block their build stages."
        ),
        "engines": engines,
    }
