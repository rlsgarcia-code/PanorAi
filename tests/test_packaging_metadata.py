from pathlib import Path
from importlib.resources import files
import subprocess
import sys

try:
    import tomllib
except ImportError:  # pragma: no cover - exercised by the Python 3.10 CI job
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_core_metadata_keeps_heavy_backends_optional() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = metadata["project"]["dependencies"]
    lowered = "\n".join(dependencies).lower()
    assert "torch" not in lowered
    assert "open3d" not in lowered
    assert "joblib" not in lowered
    assert metadata["project"]["requires-python"] == ">=3.10"
    assert set(metadata["project"]["optional-dependencies"]) == {
        "torch", "pcd", "depth", "depth-demo", "dev", "docs"
    }

    depth = "\n".join(metadata["project"]["optional-dependencies"]["depth"]).lower()
    assert "torch" in depth
    assert "transformers" in depth
    for research_dependency in (
        "mmcv",
        "mmengine",
        "tensorboard",
        "wandb",
        "xformers",
    ):
        assert research_dependency not in depth


def test_adapter_only_package_discovery_excludes_research_trees() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    excluded = set(metadata["tool"]["setuptools"]["packages"]["find"]["exclude"])
    for package in (
        "panorai.depth.DepthAnythingV2",
        "panorai.depth.Dust3r",
        "panorai.depth.Metric3D",
        "panorai.depth.ZoeDepth_not_used",
        "panorai.depth.custom_data",
        "panorai.depth.trainers",
        "panorai.depth.training",
    ):
        assert package in excluded
        assert f"{package}.*" in excluded


def test_importing_core_does_not_load_optional_backends() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, panorai, panorai.geometry; "
            "assert 'torch' not in sys.modules; assert 'open3d' not in sys.modules",
        ],
        cwd=ROOT,
        check=True,
    )


def test_geometry_contract_resource_is_packaged() -> None:
    resource = files("panorai.geometry").joinpath("geometry-v1.yaml")
    assert resource.is_file()
    assert "contract: geometry-v1" in resource.read_text(encoding="utf-8")
