import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
FIXTURES_ROOT = ROOT / "tests/fixtures/geometry/v1"


def _load_integrity_verifier():
    path = ROOT / "scripts/verify_geometry_fixture_integrity.py"
    spec = importlib.util.spec_from_file_location("geometry_fixture_integrity", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_geometry_fixture_manifest_and_lf_checkout_contract() -> None:
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "tests/fixtures/geometry/v1/*.json text eol=lf" in attributes.splitlines()

    verifier = _load_integrity_verifier()
    assert verifier.verify_fixture_integrity(FIXTURES_ROOT) > 0


def test_geometry_fixture_integrity_rejects_crlf_mutation(tmp_path: Path) -> None:
    copied_root = tmp_path / "geometry-v1"
    shutil.copytree(FIXTURES_ROOT, copied_root)
    analytic = copied_root / "analytic.json"
    analytic.write_bytes(analytic.read_bytes().replace(b"\n", b"\r\n"))

    verifier = _load_integrity_verifier()
    with pytest.raises(AssertionError, match="checksum mismatch: analytic.json"):
        verifier.verify_fixture_integrity(copied_root)


def test_geometry_v1_independent_runner() -> None:
    subprocess.run(
        [sys.executable, "scripts/run_geometry_conformance.py", "--source-checkout"],
        cwd=ROOT,
        check=True,
    )
