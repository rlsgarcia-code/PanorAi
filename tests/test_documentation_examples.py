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
            ROOT / "docs/tutorials/07_spherical_slam.md",
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
        "09_spherical_fcn_cam.md": "capability-map-image-processing",
        "10_spherical_monocular_depth.md": "capability-map-image-processing",
        "03_features_and_matching.md": "capability-map-features",
        "04_two_view_geometry.md": "capability-map-two-view",
        "05_multiview_reconstruction.md": "capability-map-multiview",
        "07_spherical_slam.md": "capability-map-slam",
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
        ("spherical-cam-imagenet-lakeside.jpg", (1024, 512)),
        ("spherical-cam-places365-forest.jpg", (1024, 512)),
        ("spherical-cam-openclip-path.jpg", (1024, 512)),
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
        "spherical-cam-imagenet-lakeside.jpg",
        "spherical-cam-places365-forest.jpg",
        "spherical-cam-openclip-path.jpg",
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
    assert hashlib.sha256(
        (TUTORIAL_MEDIA / "spherical-cam-imagenet-lakeside.jpg").read_bytes()
    ).hexdigest() == (
        "dd0827f6e6cd5301d5ffe14d086cded0ff7ce02e8955d7a95df9b7cd5c65f43c"
    )
    assert hashlib.sha256(
        (TUTORIAL_MEDIA / "spherical-cam-places365-forest.jpg").read_bytes()
    ).hexdigest() == (
        "bcfb1df74b4f57cfacab38d8695015f4eb3ed70d16734b2afe8311ff5d9e9861"
    )
    assert hashlib.sha256(
        (TUTORIAL_MEDIA / "spherical-cam-openclip-path.jpg").read_bytes()
    ).hexdigest() == (
        "baef726a3330669c2d84e711efb8b96e714890d44771f38a14e6d643eff241df"
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


def test_application_gallery_is_multiscene_checksum_pinned_and_metadata_free() -> None:
    from PIL import Image

    metadata_path = TUTORIAL_MEDIA / "application-gallery-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    attribution = (TUTORIAL_MEDIA / "ATTRIBUTION.md").read_text(encoding="utf-8")
    generator = (ROOT / "scripts/generate_application_gallery.py").read_text(
        encoding="utf-8"
    )
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    expected_sources = {
        "boiler_room": "07c353cb275f1e359cdfa1710acb77c5cd075ce7770859ca56641993263d2d19",
        "boma": "9d2e230d5fdd89efd15b6081a3951c90c771687edd81bc1e08cbb66adb98dd8c",
        "mosaic_tunnel": "47f9267f0620158ccdb7fa7bdf25fd3aa4cd9de3fb88a9bfea68b056c739d6d4",
        "sundowner_overlook": "3cdcb8a8f066e02d5554ce2d6a2591c9a47e02959297f4b2649fd91a49014498",
    }
    assert metadata["schema"] == "panorai-application-gallery/v1"
    assert metadata["license"] == "CC0-1.0"
    assert {item["slug"]: item["sha256"] for item in metadata["sources"]} == (
        expected_sources
    )
    assert all(
        digest in generator and digest in attribution
        for digest in expected_sources.values()
    )
    assert "/Users/" not in metadata_path.read_text(encoding="utf-8")
    assert metadata["applications"]["projection"]["working_shape_hw"] == [512, 1024]
    assert len(metadata["applications"]["sphere_native_processing"]["panels"]) == 2
    assert (
        "method diagram" in metadata["applications"]["pose_and_range"]["interpretation"]
    )
    semantic = metadata["applications"]["semantic_localization"]
    assert semantic["class_name"] == "prison"
    assert semantic["proxy_concept"] == "grid-like-structure"
    assert semantic["dense_shape_hw"] == [16, 32]
    assert semantic["global_rank"] == 1
    assert "not a segmentation mask" in semantic["interpretation"]
    assert "method diagram only" in semantic["association_evidence"]

    expected = {
        "application-features-mosaic-tunnel.jpg": (
            (1200, 600),
            "54132e985d473013bfdf03a7767e4ca3aadc6981f83039091886e6a143642f2e",
        ),
        "application-pose-range.jpg": (
            (1200, 600),
            "7a491dc660b359df2f47611411113149aaafeebab91aa652058692909f8803e9",
        ),
        "application-processing-boma.jpg": (
            (1200, 600),
            "b79152afec7776b58cd04c2f9b616c7113333078eb47429c7389b7dc117675b9",
        ),
        "application-projection-sundowner.jpg": (
            (1200, 600),
            "51f48a1030e6fc9a15b7707ce918d487c2ab9d0fd1f422f3dec965ab3152d6b3",
        ),
        "application-semantics-boiler-room.jpg": (
            (1200, 600),
            "48cde8e799d35a55a8b67a9162fdb58262113c8a2bb8f14cfe7c09db6ee25677",
        ),
    }
    for filename, (dimensions, digest) in expected.items():
        path = TUTORIAL_MEDIA / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        with Image.open(path) as image:
            assert image.size == dimensions
            assert not image.getexif()
        assert f"include docs/_static/tutorials/{filename}" in manifest
    assert (
        "include docs/_static/tutorials/application-gallery-metadata.json" in manifest
    )
    strategy_svg = TUTORIAL_MEDIA / "application-strategy-routes.svg"
    assert hashlib.sha256(strategy_svg.read_bytes()).hexdigest() == (
        "07e47e13b03a05b5cabbbafdc79e453dec9255181bc5f61edde39fd57d1bd48a"
    )
    strategy_text = strategy_svg.read_text(encoding="utf-8")
    assert "PROJECTION-DOMAIN" in strategy_text
    assert "SPHERE-NATIVE" in strategy_text
    assert "HYBRID · MULTIVIEW" in strategy_text


def test_application_tutorials_have_consistent_contract_headers() -> None:
    tutorials = (
        "00_quick_start.md",
        "01_custom_pipeline.md",
        "02_projection_foundations.md",
        "03_features_and_matching.md",
        "04_two_view_geometry.md",
        "05_multiview_reconstruction.md",
        "06_spherical_dense_stereo.md",
        "07_spherical_slam.md",
        "08_benchmarking.md",
        "09_spherical_fcn_cam.md",
        "10_spherical_monocular_depth.md",
        "11_cam_guided_object_hypotheses.md",
        "spherical_image_processing.md",
        "spherical_capability_map.md",
    )
    fields = (
        "**Application**",
        "**Input**",
        "**Output**",
        "**Strategy**",
        "**API**",
        "**Stability**",
        "**Dependencies**",
        "**Evidence**",
    )
    for filename in tutorials:
        text = (ROOT / "docs/tutorials" / filename).read_text(encoding="utf-8")
        for field in fields:
            assert text.count(field) >= 1, f"{filename} is missing {field}"


def test_readme_is_a_curated_entry_point_with_valid_local_links() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    normalized_readme = " ".join(readme.split())
    proposal = (ROOT / "docs/explanation/workflow-evolution.md").read_text(
        encoding="utf-8"
    )
    normalized_proposal = " ".join(proposal.split())

    # The README maps applications to concrete calls instead of reproducing
    # tutorials, benchmark reports, or a media gallery.
    assert len(readme.splitlines()) <= 260
    for heading in (
        "## Install",
        "## Choose an application",
        "## 1. Project views or run a pinhole method",
        "## 2. Process directly on the sphere",
        "## 3. Match features and estimate relative pose",
        "## 4. Estimate radial range",
        "## 5. Reconstruct three or more views",
        "## 6. Localize and associate semantic regions",
        "## How the routes fit together",
        "## Data contract",
        "## Evidence and boundaries",
        "## Reference",
    ):
        assert readme.count(heading) == 1
    assert "MultiChannelHandler" not in readme
    assert "Zero is data; it is never an implicit invalid marker" in normalized_readme
    assert "SphericalGlobalMapper" in readme
    assert "PyCOLMAP/COLMAP" in readme
    assert "spherical_equalize_histogram" in readme
    assert "does **not** currently claim a sphere-native CLAHE" in readme
    assert "panorai[slam]" in readme
    assert "arbitrary-N" in readme
    assert "estimate_spherical_range" in readme
    assert "SphericalObjectHypothesisMethod" in readme
    assert "SphericalProxySegmenter" in readme
    assert "public Stable `panorai-object-workflow/v1` contract" in normalized_proposal
    assert "docs/_static" not in readme

    application_entry = readme.split("## Choose an application", 1)[1].split(
        "## 1. Project views or run a pinhole method", 1
    )[0]
    assert "![" not in application_entry
    assert "<img" not in application_entry

    docs_index = (ROOT / "docs/index.rst").read_text(encoding="utf-8")
    docs_application_entry = docs_index.split("Choose an application", 1)[1].split(
        "How PanorAi executes spherical computer vision", 1
    )[0]
    assert ".. image::" not in docs_application_entry
    assert ":img-top:" not in docs_application_entry

    relative_links = re.findall(r"\[[^]]+\]\(([^)]+)\)", readme)
    for target in relative_links:
        if "://" in target or target.startswith("#"):
            continue
        local_path = target.split("#", 1)[0]
        assert (ROOT / local_path).exists(), f"README link target is missing: {target}"


def test_readme_python_examples_are_valid_and_stable_example_executes() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    python_blocks = re.findall(r"```python\n(.*?)```", readme, flags=re.DOTALL)
    assert len(python_blocks) == 3

    for index, block in enumerate(python_blocks):
        compile(block, f"README.md:python-block-{index + 1}", "exec")

    subprocess.run(
        [sys.executable, "-c", "\n".join(python_blocks)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_spherical_fcn_tutorial_documents_acquisition_and_evidence_boundary() -> None:
    tutorial = (ROOT / "docs/tutorials/09_spherical_fcn_cam.md").read_text(
        encoding="utf-8"
    )
    normalized_tutorial = " ".join(tutorial.split())
    benchmark = (ROOT / "benchmarks/spherical_fcn_cam/README.md").read_text(
        encoding="utf-8"
    )
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for required in (
        "panorai[deep-learning]",
        "download_models.py",
        "output stride 32",
        "output stride 8",
        "Independent gnomonic oracle",
        "not yet proven numerically equivalent",
        "not a calibrated per-pixel probability",
    ):
        assert required in normalized_tutorial
    assert "Checkpoints remain in the user-controlled" in benchmark
    assert "09_spherical_fcn_cam.md" in readme
    assert "SphericalProxySegmenter" in normalized_tutorial


def test_spherical_fcn_tutorial_lists_the_complete_imagenet_proxy_vocabulary() -> None:
    from panorai.experimental.deep_learning.segmentation import (
        IMAGENET_PROXY_VOCABULARY,
    )

    tutorial = (ROOT / "docs/tutorials/09_spherical_fcn_cam.md").read_text(
        encoding="utf-8"
    )
    assert "geometric activation proxies" in tutorial
    assert "complete public `IMAGENET_PROXY_VOCABULARY`" in tutorial
    for concept in IMAGENET_PROXY_VOCABULARY:
        assert f"`{concept.concept_id}`" in tutorial
        for class_index, class_name, weight in concept.class_weights:
            documented = f"`{class_name}` ({class_index}, {weight:.2f})"
            assert documented in tutorial


def test_spherical_depth_tutorial_documents_resize_and_license_boundary() -> None:
    tutorial = (ROOT / "docs/tutorials/10_spherical_monocular_depth.md").read_text(
        encoding="utf-8"
    )
    normalized_tutorial = " ".join(tutorial.split())
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for required in (
        "panorai[deep-learning-depth]",
        "accept_upstream_terms=True",
        "does not resize, crop, pad, or prefilter",
        "weights_only=True",
        "radial range",
        "no separate checkpoint/model-card license",
        "CNN and ViT portability",
        "No private evaluation dataset, prediction, metric, or result",
        "prove only mechanical availability",
        "does not prove",
    ):
        assert required in normalized_tutorial
    assert "10_spherical_monocular_depth.md" in readme
