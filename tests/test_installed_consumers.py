from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_installed_consumer.py"
CONSUMER = ROOT / "consumers" / "multimodal_arbitrary_n" / "consumer.py"
FEATURE_CONSUMER = ROOT / "consumers" / "features_pose_pycolmap" / "consumer.py"
COMPOSED_CONSUMERS = tuple(
    ROOT / "consumers" / name / "consumer.py"
    for name in (
        "spherical_slic",
        "spherical_dog",
        "spherical_evaluation",
        "spherical_region_analysis",
        "panorama_stitching",
        "spherical_stereo",
        "spherical_mask_rcnn_workflow",
        "composed_spherical_api",
    )
)


def test_installed_consumer_runner_rejects_missing_wheel(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--wheel",
            str(tmp_path / "missing.whl"),
            "--consumer",
            str(CONSUMER),
            "--source-root",
            str(ROOT),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "existing .whl file" in completed.stderr


def test_consumer_requires_installed_origin() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    completed = subprocess.run(
        [
            sys.executable,
            str(CONSUMER),
            "--backend",
            "numpy",
            "--source-root",
            str(ROOT),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "resolved inside the source checkout" in completed.stderr


def test_consumer_constants_are_arbitrary_n_and_rectangular() -> None:
    namespace: dict[str, object] = {"__name__": "consumer_contract_test"}
    exec(
        compile(CONSUMER.read_text(encoding="utf-8"), str(CONSUMER), "exec"), namespace
    )
    assert namespace["COUNT"] > 6
    assert namespace["VIEW_SHAPE"][0] != namespace["VIEW_SHAPE"][1]
    assert namespace["BLENDS"] == {
        "image": "gaussian",
        "depth": "average",
        "labels": "closest",
    }


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_consumer_output_contract_parser(backend: str) -> None:
    payload = json.loads(
        json.dumps(
            {
                "status": "ok",
                "consumer": "multimodal-arbitrary-n",
                "backend": backend,
                "view_count": 14,
            }
        )
    )
    assert payload["status"] == "ok"
    assert payload["backend"] == backend
    assert payload["view_count"] > 6


def test_features_consumer_uses_only_public_panorai_packages() -> None:
    source = FEATURE_CONSUMER.read_text(encoding="utf-8")
    assert "panorai.features import SphericalFeaturePipeline" in source
    assert "panorai.estimators import" in source
    assert "from panorai._" not in source
    assert "import panorai._" not in source
    assert "from panorai.features._" not in source
    assert "from panorai.estimators._" not in source
    assert '"competing-model:rotation-only"' in source


@pytest.mark.parametrize(
    "consumer", COMPOSED_CONSUMERS, ids=lambda path: path.parent.name
)
def test_composed_consumers_reject_source_checkout_imports(consumer: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    completed = subprocess.run(
        [
            sys.executable,
            str(consumer),
            "--source-root",
            str(ROOT),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "resolved inside the source checkout" in completed.stderr


@pytest.mark.parametrize(
    "consumer", COMPOSED_CONSUMERS, ids=lambda path: path.parent.name
)
def test_composed_consumers_use_only_public_panorai_imports(consumer: Path) -> None:
    source = consumer.read_text(encoding="utf-8")
    compile(source, str(consumer), "exec")
    assert "from panorai._" not in source
    assert "import panorai._" not in source
    assert '"status": "ok"' in source
    if "mask_rcnn" in consumer.parent.name:
        assert "evidence_scope" in source


def test_stable_slic_and_experimental_dog_profile_are_permanent_wheel_gates() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "consumers/spherical_slic/consumer.py" in workflow
    assert "consumers/spherical_dog/consumer.py" in workflow
    dog_consumer = (ROOT / "consumers" / "spherical_dog" / "consumer.py").read_text(
        encoding="utf-8"
    )
    assert "SphericalDoGFeatures.reviewed_profile" in dog_consumer
    assert 'method.stability == "experimental"' in dog_consumer
    assert "panorai-spherical-dog-features/reviewed-v1" in dog_consumer


def test_public_catalog_is_a_permanent_base_wheel_gate() -> None:
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    release = (ROOT / ".github" / "workflows" / "python-publish.yml").read_text(
        encoding="utf-8"
    )
    consumer = (ROOT / "consumers" / "public_catalog" / "consumer.py").read_text(
        encoding="utf-8"
    )

    assert "consumers/public_catalog/consumer.py" in ci
    assert "consumers/public_catalog/consumer.py" in release
    assert "public_catalog" in consumer
    assert "import torch" not in consumer
    assert "from torch" not in consumer
    assert '"consumer": "public-catalog"' in consumer


def test_stable_evaluation_and_region_analysis_are_permanent_wheel_gates() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "consumers/spherical_evaluation/consumer.py" in workflow
    assert "consumers/spherical_region_analysis/consumer.py" in workflow


def test_experimental_panorama_stitching_is_a_permanent_wheel_gate() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    consumer = (ROOT / "consumers" / "panorama_stitching" / "consumer.py").read_text(
        encoding="utf-8"
    )

    assert "consumers/panorama_stitching/consumer.py" in workflow
    assert "PanoramaStitcher" in consumer
    assert '"consumer": "panorama-stitching"' in consumer
