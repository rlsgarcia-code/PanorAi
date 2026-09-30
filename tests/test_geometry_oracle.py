from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


def test_geometry_oracle_runner_uses_source_explicitly() -> None:
    pytest.importorskip(
        "torch",
        reason="the mandatory gradient oracle is exercised in the Torch gate",
    )
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_geometry_oracle.py",
            "--source-checkout",
            "--require-torch-gradcheck",
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["contract"] == "geometry-v1"
    assert report["random_samples"] == 4096
    assert report["torch_gradcheck"] is True
    assert Path(report["origin"]).resolve().is_relative_to(root)
