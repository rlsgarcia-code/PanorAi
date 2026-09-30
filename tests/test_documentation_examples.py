from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_documentation_examples.py"


def test_canonical_documentation_examples_execute_from_source() -> None:
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--source-checkout"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "documentation examples: OK" in completed.stdout


def test_torch_documentation_example_has_input_gradients() -> None:
    pytest.importorskip("torch")
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--source-checkout", "--torch"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "documentation examples: OK" in completed.stdout


def test_every_executable_section_is_included_in_public_docs() -> None:
    docs = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            ROOT / "docs/tutorials/00_quick_start.md",
            ROOT / "docs/tutorials/01_custom_pipeline.md",
            ROOT / "docs/how_to/data_modalities.rst",
        )
    )
    sections = (
        "FUNCTIONAL",
        "PROJECTOR",
        "CONTAINER",
        "WORKFLOW",
        "MODALITIES",
        "BLENDER",
        "CUBEMAP",
        "TORCH",
    )
    runner = RUNNER.read_text(encoding="utf-8")
    for section in sections:
        assert runner.count(f"# DOCS_{section}_START = None") == 1
        assert runner.count(f"# DOCS_{section}_END = None") == 1
        assert f":start-after: DOCS_{section}_START = None" in docs
        assert f":end-before: DOCS_{section}_END = None" in docs


def test_documentation_is_curated_without_warning_suppression() -> None:
    conf = (ROOT / "docs/conf.py").read_text(encoding="utf-8")
    stability = (ROOT / "docs/reference/stability.rst").read_text(encoding="utf-8")
    related = (ROOT / "docs/explanation/related_libraries.rst").read_text(
        encoding="utf-8"
    )

    assert "nitpicky = True" in conf
    assert "suppress_warnings = []" in conf
    for tier in ("Stable", "Compatibility", "Experimental", "Frozen/internal"):
        assert tier in stability
    for project in ("py360convert", "pyequilib", "OpenCV"):
        assert project in related
    assert "No performance superiority is claimed" in related
    assert "my_pano.jpg" not in "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "docs/tutorials").glob("*")
        if path.suffix in {".md", ".rst"}
    )


def test_readme_is_a_curated_entry_point_with_valid_local_links() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    proposal = (ROOT / "docs/explanation/workflow-evolution.md").read_text(
        encoding="utf-8"
    )
    normalized_proposal = " ".join(proposal.split())

    assert len(readme.splitlines()) <= 220
    for heading in (
        "## Quick start: canonical geometry",
        "## Workflow API: panorama to faces and back",
        "## Choose the right surface",
        "## Data modalities",
    ):
        assert readme.count(heading) == 1
    assert "MultiChannelHandler" not in readme
    assert "Do not stack RGB, labels, masks, or depth" in readme
    assert "proposal, not a current API commitment" in readme
    assert "not part of the PanorAi 3.x compatibility contract" in normalized_proposal

    relative_links = re.findall(r"\[[^]]+\]\(([^)]+)\)", readme)
    for target in relative_links:
        if "://" in target or target.startswith("#"):
            continue
        local_path = target.split("#", 1)[0]
        assert (ROOT / local_path).exists(), f"README link target is missing: {target}"


def test_readme_python_examples_execute_from_source() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    python_blocks = re.findall(r"```python\n(.*?)```", readme, flags=re.DOTALL)
    assert len(python_blocks) == 2

    for block in python_blocks:
        subprocess.run(
            [sys.executable, "-c", block],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
