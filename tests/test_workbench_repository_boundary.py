from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_workbench_is_not_a_panorai_package_or_sdist_input() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert 'include = ["panorai", "panorai.*"]' in pyproject
    assert "prune workbench" in manifest
    assert (ROOT / "workbench" / "pyproject.toml").is_file()
    assert (ROOT / "workbench" / "package.json").is_file()
    assert '"panorai>=3.8.1.dev0,<4"' in (
        ROOT / "workbench" / "pyproject.toml"
    ).read_text(encoding="utf-8")


def test_root_package_has_no_workbench_runtime_dependency() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    root_project = pyproject.split("[project.optional-dependencies]", 1)[0]

    assert "fastapi" not in root_project.lower()
    assert "uvicorn" not in root_project.lower()
    assert "vite" not in root_project.lower()
