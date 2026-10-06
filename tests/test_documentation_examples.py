from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_documentation_examples.py"
TUTORIAL_MEDIA = ROOT / "docs/_static/tutorials"


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
            ROOT / "docs/tutorials/spherical_capability_map.md",
            ROOT / "docs/tutorials/01_custom_pipeline.md",
            ROOT / "docs/tutorials/02_projection_foundations.md",
            ROOT / "docs/tutorials/spherical_image_processing.md",
            ROOT / "docs/tutorials/03_features_and_matching.md",
            ROOT / "docs/tutorials/04_two_view_geometry.md",
            ROOT / "docs/tutorials/05_multiview_reconstruction.md",
            ROOT / "docs/tutorials/06_spherical_dense_stereo.md",
            ROOT / "docs/tutorials/06_spherical_slam.md",
            ROOT / "docs/how_to/data_modalities.rst",
            ROOT / "docs/how_to/spherical_features.rst",
            ROOT / "docs/how_to/spherical_reconstruction.rst",
            ROOT / "docs/how_to/spherical_slam.rst",
        )
    )
    sections = (
        "FUNCTIONAL",
        "PROJECTOR",
        "CONTAINER",
        "WORKFLOW",
        "MODALITIES",
        "SPHERICAL_PROCESSING",
        "SPHERICAL_PREPROCESSING",
        "FEATURES",
        "SPHERICAL_DOG_SIFT",
        "RELATIVE_POSE",
        "STEREO",
        "TRIANGULATION",
        "METRIC_FLOOR",
        "RECONSTRUCTION",
        "SLAM",
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


def test_every_tutorial_links_to_its_user_first_spherical_cv_section() -> None:
    capability_map = (ROOT / "docs/tutorials/spherical_capability_map.md").read_text(
        encoding="utf-8"
    )
    expected = {
        "00_quick_start.md": "capability-map-quick-start",
        "01_custom_pipeline.md": "capability-map-cubemap-batch",
        "02_projection_foundations.md": "capability-map-projection-foundations",
        "spherical_image_processing.md": "capability-map-image-processing",
        "03_features_and_matching.md": "capability-map-features",
        "04_two_view_geometry.md": "capability-map-two-view",
        "05_multiview_reconstruction.md": "capability-map-multiview",
        "06_spherical_slam.md": "capability-map-slam",
    }

    for filename, anchor in expected.items():
        tutorial = (ROOT / "docs/tutorials" / filename).read_text(encoding="utf-8")
        assert f"({anchor})=" in capability_map
        assert f"{{ref}}`{anchor}`" in tutorial
        assert "**PanorAi-specific:**" in tutorial

    for required_structure in (
        "Choose a computer-vision task",
        "What you can do",
        "Typical input → result",
        "Start here",
        "Execution domain",
        "Sphere-native",
        "Projection-domain",
        "Hybrid",
        "C++ acceleration is a separate axis",
    ):
        assert required_structure in capability_map

    for theme in (
        "## Projection",
        "## Image processing",
        "## Features and matching",
        "## Two-view geometry",
        "## Multiview reconstruction",
        "## SLAM",
    ):
        assert theme in capability_map


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


def test_flowcharts_are_self_contained_svg_assets() -> None:
    docs = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "docs").rglob("*")
        if path.suffix in {".md", ".rst", ".py"}
    )
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "sphinxcontrib.mermaid" not in docs
    assert "sphinxcontrib-mermaid" not in pyproject
    assert "```{mermaid}" not in docs
    assert ".. mermaid::" not in docs

    for filename in (
        "architecture-flow.svg",
        "projection-workflow.svg",
        "features-matching-flow.svg",
        "reconstruction-graph.svg",
        "known-height-two-view.svg",
        "spherical-stereo-optimization.svg",
    ):
        asset = TUTORIAL_MEDIA / filename
        content = asset.read_text(encoding="utf-8")
        assert content.lstrip().startswith("<svg")
        assert "<title>" in content and "<desc>" in content
        assert "<script" not in content
        assert filename in docs
    assert "include docs/_static/tutorials/*.svg" in (ROOT / "MANIFEST.in").read_text(
        encoding="utf-8"
    )


def test_visual_tutorial_assets_are_public_reproducible_and_not_packaged() -> None:
    from PIL import Image

    attribution = (TUTORIAL_MEDIA / "ATTRIBUTION.md").read_text(encoding="utf-8")
    metadata = json.loads(
        (TUTORIAL_MEDIA / "figure-metadata.json").read_text(encoding="utf-8")
    )
    generator = (ROOT / "scripts/generate_documentation_figures.py").read_text(
        encoding="utf-8"
    )
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert (
        "CC0" in attribution and "polyhaven.com/a/nature_reserve_forest" in attribution
    )
    assert metadata["source_sha256"] == (
        "6c943ddd683de2f3d9aaa62596961dfccdc9cf206adebfc198e70235ae5707cd"
    )
    assert metadata["source_sha256"] in generator
    assert metadata["comparison_transform"] == {
        "kind": "cyclic ERP longitude shift",
        "pixels": 56,
    }
    for filename, expected_shape_wh in (
        ("nature-reserve-forest-erp.jpg", (1024, 512)),
        ("feature-detectors.jpg", (1536, 256)),
        ("feature-matches.jpg", (768, 796)),
        ("spherical-image-processing.jpg", (1536, 768)),
        ("spherical-histogram-equalization.jpg", (1536, 634)),
        ("spherical-dog-sift.jpg", (1024, 512)),
    ):
        path = TUTORIAL_MEDIA / filename
        assert path.stat().st_size > 10_000
        with Image.open(path) as image:
            assert image.size == expected_shape_wh
    assert "global-exclude *.jpg" in manifest
    for filename in (
        "feature-detectors.jpg",
        "feature-matches.jpg",
        "nature-reserve-forest-erp.jpg",
        "spherical-image-processing.jpg",
        "spherical-histogram-equalization.jpg",
        "spherical-dog-sift.jpg",
    ):
        assert f"include docs/_static/tutorials/{filename}" in manifest
    assert hashlib.sha256(
        (TUTORIAL_MEDIA / "feature-detectors.jpg").read_bytes()
    ).hexdigest() == (
        "3acc770c058e171570225b052ff2cb3e02d32ecb43a8b37f0ff560e30ed91a82"
    )
    assert hashlib.sha256(
        (TUTORIAL_MEDIA / "spherical-image-processing.jpg").read_bytes()
    ).hexdigest() == (
        "70136e1076275798175e358677f098305aadb29c4085d943ecd4507a8b5a48c3"
    )
    assert hashlib.sha256(
        (TUTORIAL_MEDIA / "spherical-histogram-equalization.jpg").read_bytes()
    ).hexdigest() == (
        "bc6c04d6592ced4894ca9863b6e07d5aeeeaf63dc8c28c8c369fe8d2fb826b19"
    )
    assert hashlib.sha256(
        (TUTORIAL_MEDIA / "spherical-dog-sift.jpg").read_bytes()
    ).hexdigest() == (
        "2f50b9a0275c519e196c1af9e6fd836885e9df452e360a38fc4b3086cb46eadb"
    )
    assert metadata["spherical_image_processing"]["canny_edge_pixels"] > 0
    assert (
        metadata["spherical_histogram_equalization"]["histogram_weight"]
        == "cos(latitude at ERP row center)"
    )
    assert metadata["spherical_dog_sift"] == {
        "descriptor": "OpenCV SIFT on one tangent patch per keypoint",
        "feature_count": 600,
        "octaves": 3,
        "plotted_count": 450,
    }


def test_readme_is_a_curated_entry_point_with_valid_local_links() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    proposal = (ROOT / "docs/explanation/workflow-evolution.md").read_text(
        encoding="utf-8"
    )
    normalized_proposal = " ".join(proposal.split())

    assert len(readme.splitlines()) <= 110
    for heading in (
        "## Choose a use case",
        "## Install",
        "## Minimal projection",
        "## What PanorAi owns",
        "## Reference",
    ):
        assert readme.count(heading) == 1
    assert "MultiChannelHandler" not in readme
    assert "Zero is data, never an implicit" in readme
    for surface in (
        "panorai.features",
        "panorai.estimators",
        "panorai.reconstruction",
        "panorai.slam",
    ):
        assert surface in readme
    assert "panorai[slam]" in readme
    assert "arbitrary-N" in readme
    assert "public Stable `panorai-object-workflow/v1` contract" in normalized_proposal

    relative_links = re.findall(r"\[[^]]+\]\(([^)]+)\)", readme)
    for target in relative_links:
        if "://" in target or target.startswith("#"):
            continue
        local_path = target.split("#", 1)[0]
        assert (ROOT / local_path).exists(), f"README link target is missing: {target}"


def test_readme_python_examples_are_valid_and_stable_example_executes() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    python_blocks = re.findall(r"```python\n(.*?)```", readme, flags=re.DOTALL)
    assert len(python_blocks) == 1

    for index, block in enumerate(python_blocks):
        compile(block, f"README.md:python-block-{index + 1}", "exec")

    subprocess.run(
        [sys.executable, "-c", python_blocks[0]],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
