from pathlib import Path
from importlib.resources import files
import subprocess
import sys

import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_core_metadata_keeps_heavy_backends_optional() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = metadata["project"]["dependencies"]
    lowered = "\n".join(dependencies).lower()
    assert "torch" not in lowered
    assert "open3d" not in lowered
    assert "joblib" not in lowered
    assert metadata["project"]["requires-python"] == ">=3.11"
    assert "opencv-python-headless>=4.9,<5" in dependencies
    assert set(metadata["project"]["optional-dependencies"]) == {
        "torch",
        "deep-learning",
        "deep-learning-depth",
        "features",
        "pycolmap",
        "slam",
        "pcd",
        "depth",
        "depth-demo",
        "dev",
        "docs",
    }

    slam = "\n".join(metadata["project"]["optional-dependencies"]["slam"]).lower()
    assert "rosbags" in slam

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

    deep_learning = "\n".join(
        metadata["project"]["optional-dependencies"]["deep-learning"]
    ).lower()
    assert "torch>=2.2,<3" in deep_learning
    assert "torchvision>=0.17,<1" in deep_learning

    depth_learning = "\n".join(
        metadata["project"]["optional-dependencies"]["deep-learning-depth"]
    ).lower()
    for dependency in ("torch>=2.2,<3", "timm", "mmengine", "mmcv-lite", "iopath"):
        assert dependency in depth_learning
    assert "torchvision" not in depth_learning

    assert metadata["project"]["optional-dependencies"]["features"] == [
        "opencv-python-headless>=4.9,<5"
    ]


def test_supported_python_versions_match_native_wheel_selector() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    classifiers = set(metadata["project"]["classifiers"])
    expected_versions = {"3.11", "3.12", "3.13", "3.14"}

    declared_versions = {
        classifier.rsplit(" :: ", 1)[-1]
        for classifier in classifiers
        if classifier.startswith("Programming Language :: Python :: 3.")
    }
    assert declared_versions == expected_versions
    assert metadata["tool"]["cibuildwheel"]["build"] == "cp3{11,12,13,14}-*"
    assert metadata["tool"]["setuptools_scm"]["fallback_version"] == "3.4.0.dev0"


def test_macos_native_link_omits_local_build_identity() -> None:
    setup_source = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert 'native_link_args = (\n    ["-Wl,-S", "-Wl,-x"]' in setup_source
    assert "extra_link_args=native_link_args" in setup_source
    assert "*native_link_args" in setup_source


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
