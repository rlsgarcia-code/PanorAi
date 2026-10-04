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
    exec(compile(CONSUMER.read_text(encoding="utf-8"), str(CONSUMER), "exec"), namespace)
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
