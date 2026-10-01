import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


def test_geometry_import_does_not_load_legacy_or_heavy_trees() -> None:
    code = r"""
import json
import sys
import panorai.geometry

forbidden_roots = {"PIL", "cv2", "pydantic", "scipy", "skimage", "torch", "yaml"}
forbidden_panorai = (
    "panorai.blenders",
    "panorai.config",
    "panorai.data",
    "panorai.factory",
    "panorai.preprocessing",
    "panorai.projections",
    "panorai.samplers",
)
loaded_roots = {name.split(".")[0] for name in sys.modules}
assert forbidden_roots.isdisjoint(loaded_roots), forbidden_roots & loaded_roots
assert not any(
    name == prefix or name.startswith(prefix + ".")
    for name in sys.modules
    for prefix in forbidden_panorai
)
assert "EquirectangularImage" not in vars(sys.modules["panorai"])
print(json.dumps(sorted(name for name in sys.modules if name.startswith("panorai"))))
"""
    completed = _run(code)
    loaded = json.loads(completed.stdout)
    assert loaded == [
        "panorai",
        "panorai._version",
        "panorai.geometry",
        "panorai.geometry._contracts",
        "panorai.geometry._engine",
        "panorai.geometry._projectors",
        "panorai.geometry._typing",
    ]


def test_root_exports_are_lazy_discoverable_and_identical_to_direct_objects() -> None:
    code = r"""
import panorai
import sys

expected = [
    "EquirectangularImage",
    "GnomonicFace",
    "GnomonicFaceSet",
    "ConfigManager",
    "PanoraiFactory",
    "SphericalFeaturePipeline",
    "__version__",
]
assert panorai.__all__ == expected
assert set(expected).issubset(dir(panorai))
assert not any(name in vars(panorai) for name in expected[:-1])

namespace = {}
exec("from panorai import *", namespace)
from panorai.config.config_manager import ConfigManager
from panorai.data import EquirectangularImage, GnomonicFace, GnomonicFaceSet
from panorai.factory.panorai_factory import PanoraiFactory
from panorai.features import SphericalFeaturePipeline

assert namespace["EquirectangularImage"] is EquirectangularImage
assert namespace["GnomonicFace"] is GnomonicFace
assert namespace["GnomonicFaceSet"] is GnomonicFaceSet
assert namespace["ConfigManager"] is ConfigManager
assert namespace["PanoraiFactory"] is PanoraiFactory
assert namespace["SphericalFeaturePipeline"] is SphericalFeaturePipeline
assert namespace["__version__"] == panorai.__version__
assert panorai.EquirectangularImage is EquirectangularImage
assert panorai.GnomonicFace is GnomonicFace
assert panorai.GnomonicFaceSet is GnomonicFaceSet
assert panorai.ConfigManager is ConfigManager
assert panorai.PanoraiFactory is PanoraiFactory
assert panorai.SphericalFeaturePipeline is SphericalFeaturePipeline
"""
    _run(code)


def test_root_version_matches_distribution_metadata() -> None:
    _run(
        "import panorai; from importlib.metadata import version; "
        "assert panorai.__version__ == version('panorai')"
    )


def test_depth_adapter_import_does_not_load_optional_backends_or_vendor_trees() -> None:
    code = r"""
import json
import sys
import panorai.depth

forbidden_roots = {"torch", "open3d", "transformers", "mmcv", "mmengine"}
loaded_roots = {name.split(".")[0] for name in sys.modules}
assert forbidden_roots.isdisjoint(loaded_roots), forbidden_roots & loaded_roots
forbidden_panorai = (
    "panorai.depth.DepthAnythingV2",
    "panorai.depth.Dust3r",
    "panorai.depth.Metric3D",
    "panorai.depth.ZoeDepth_not_used",
    "panorai.depth.custom_data",
    "panorai.depth.trainers",
    "panorai.depth.training",
)
assert not any(
    name == prefix or name.startswith(prefix + ".")
    for name in sys.modules
    for prefix in forbidden_panorai
)
print(json.dumps(sorted(name for name in sys.modules if name.startswith("panorai.depth"))))
"""
    completed = _run(code)
    assert json.loads(completed.stdout) == [
        "panorai.depth",
        "panorai.depth._adapters",
        "panorai.depth.registry",
    ]


def test_import_profiler_emits_reproducible_source_schema() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/profile_imports.py",
            "--repetitions",
            "1",
            "--target",
            "panorai.geometry",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    report = json.loads(completed.stdout)
    result = report["results"][0]

    assert report["schema"] == "panorai-import-profile-v1"
    assert not report["require_installed"]
    assert result["target"] == "panorai.geometry"
    assert result["repetitions"] == 1
    assert result["cold_subprocess"]
    assert result["elapsed_seconds"]["median"] > 0
    assert result["rss_bytes"]["maximum"] > 0
    assert result["module_count"] >= len(result["top_level_modules"])
    assert result["panorai_modules"] == [
        "panorai",
        "panorai._version",
        "panorai.geometry",
        "panorai.geometry._contracts",
        "panorai.geometry._engine",
        "panorai.geometry._projectors",
        "panorai.geometry._typing",
    ]
    assert "cv2" not in result["top_level_modules"]
    assert "scipy" not in result["top_level_modules"]
    assert Path(result["origin"]).is_relative_to(ROOT)
