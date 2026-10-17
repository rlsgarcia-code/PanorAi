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
            ROOT / "docs/index.rst",
            ROOT / "docs/getting-started/index.rst",
            ROOT / "docs/guides/api-levels/applications.rst",
            ROOT / "docs/guides/api-levels/pipeline.rst",
            ROOT / "docs/guides/api-levels/spherical_ops.rst",
            ROOT / "docs/guides/api-levels/geometry.rst",
            ROOT / "docs/guides/tutorials/00_quick_start.md",
            ROOT / "docs/guides/tutorials/equirectangular_image.md",
            ROOT / "docs/guides/tutorials/spherical_capability_map.md",
            ROOT / "docs/guides/tutorials/01_custom_pipeline.md",
            ROOT / "docs/guides/tutorials/02_projection_foundations.md",
            ROOT / "docs/guides/tutorials/spherical_image_processing.md",
            ROOT / "docs/guides/tutorials/03_features_and_matching.md",
            ROOT / "docs/guides/tutorials/04_two_view_geometry.md",
            ROOT / "docs/guides/tutorials/05_multiview_reconstruction.md",
            ROOT / "docs/guides/tutorials/06_spherical_dense_stereo.md",
            ROOT / "docs/guides/tutorials/07_spherical_slam.md",
            ROOT / "docs/guides/how-to/data_modalities.rst",
            ROOT / "docs/guides/how-to/spherical_features.rst",
            ROOT / "docs/guides/how-to/spherical_reconstruction.rst",
            ROOT / "docs/guides/how-to/spherical_slam.rst",
        )
    )
    sections = (
        "LEVEL_APPLICATION",
        "LEVEL_APPLICATION_CONFIG",
        "LEVEL_PIPELINE",
        "LEVEL_OPS",
        "LEVEL_GEOMETRY",
        "EQUIRECTANGULAR_IMAGE",
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
    assert "EquirectangularImage.constant(" in runner


def test_primary_documentation_has_four_progressive_api_levels() -> None:
    index = (ROOT / "docs/index.rst").read_text(encoding="utf-8")
    guide_index = (ROOT / "docs/guides/api-levels/index.rst").read_text(
        encoding="utf-8"
    )
    concepts = (ROOT / "docs/concepts/index.rst").read_text(encoding="utf-8")
    reference = (ROOT / "docs/reference/index.rst").read_text(encoding="utf-8")

    ordered_levels = (
        "1. Applications",
        "2. Pipeline",
        "3. Spherical Ops",
        "4. Geometry",
    )
    assert [index.index(level) for level in ordered_levels] == sorted(
        index.index(level) for level in ordered_levels
    )
    assert "Quick start" in index
    assert "less setup to more control" in index
    assert "Tutorial: build it" in index
    assert "Concept: understand it" in index
    assert "Reference: look it up" in index

    for slug in ("applications", "pipeline", "spherical_ops", "geometry"):
        assert slug in guide_index
    for slug in (
        "application-wrappers",
        "pipeline-composition",
        "spherical-operations",
        "geometry-contract",
    ):
        assert slug in concepts
    for slug in ("applications", "pipeline", "spherical_ops", "geometry"):
        assert slug in reference

    for guide_name, concept_name, reference_name in (
        ("applications", "application-wrappers", "applications"),
        ("pipeline", "pipeline-composition", "pipeline"),
        ("spherical_ops", "spherical-operations", "spherical_ops"),
        ("geometry", "geometry-contract", "geometry"),
    ):
        guide = (ROOT / "docs/guides/api-levels" / f"{guide_name}.rst").read_text(
            encoding="utf-8"
        )
        assert f"../../concepts/{concept_name}" in guide
        assert f"../../reference/{reference_name}" in guide


def test_component_catalog_is_role_based_and_exhaustive() -> None:
    manifest = json.loads(
        (ROOT / "contracts/public-api-surfaces-v2.json").read_text(encoding="utf-8")
    )
    reference_index = (ROOT / "docs/reference/index.rst").read_text(encoding="utf-8")
    catalog_index = (ROOT / "docs/reference/catalog/index.rst").read_text(
        encoding="utf-8"
    )
    public_surfaces = (ROOT / "docs/reference/catalog/public_surfaces.rst").read_text(
        encoding="utf-8"
    )
    extension = (ROOT / "docs/_ext/api_inventory.py").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "catalog/index" in reference_index
    for page in (
        "applications",
        "stages",
        "operations",
        "estimators",
        "data_results",
        "geometry",
        "subsystems",
        "public_surfaces",
    ):
        assert page in catalog_index
        assert (ROOT / "docs/reference/catalog" / f"{page}.rst").is_file()

    public_families = {
        family["id"] for family in manifest["families"] if family["id"] != "internal"
    }
    surface_families = {
        surface["family"]
        for surface in manifest["surfaces"]
        if surface["visibility"] != "internal"
    }
    assert surface_families == public_families
    for family in sorted(public_families):
        assert f":family: {family}" in public_surfaces

    assert 'app.add_directive("panorai-api-symbols"' in extension
    assert (
        "https://rlsgarcia-code.github.io/PanorAi/reference/catalog/index.html"
        in readme
    )
    assert (
        "https://rlsgarcia-code.github.io/PanorAi/reference/catalog/public_surfaces.html"
        in readme
    )
    assert "Choose a concrete object in the" in readme
    for key in (
        '"exports_exact"',
        '"classified_exports"',
        '"required_symbols"',
        '"registry_names"',
    ):
        assert key in extension


def test_curated_catalog_maps_applications_stages_and_values() -> None:
    applications = (ROOT / "docs/reference/catalog/applications.rst").read_text(
        encoding="utf-8"
    )
    stages = (ROOT / "docs/reference/catalog/stages.rst").read_text(encoding="utf-8")
    values = (ROOT / "docs/reference/catalog/data_results.rst").read_text(
        encoding="utf-8"
    )

    for name in (
        "SphericalSLIC",
        "SphericalDoGFeatures",
        "SphericalRelativePose",
        "SphericalStereoDepth",
        "SphericalMaskRCNNWorkflow",
    ):
        assert name in applications

    for name in (
        "OnSphere",
        "SampleViews",
        "PerView",
        "ReconstructERP",
        "LiftToBearings",
        "AggregateViews",
        "Parallel",
        "Select",
        "Attach",
        "AssessRelativePoseCapture",
        "ExtractSphericalFeaturePair",
        "MatchSphericalFeatures",
        "ToBearingCorrespondences",
        "EstimateRelativePose",
        "SphericalCAM",
        "CAMRegionProposals",
        "SAMMaskRefinement",
        "SphericalInstanceFusion",
    ):
        assert name in stages

    for name in (
        "EquirectangularImage",
        "RasterGrid",
        "RasterField",
        "ResultBundle",
        "StageResult",
        "PipelineRun",
        "ClassificationResult",
        "EmbeddingResult",
        "SphericalBearingCorrespondences",
        "RelativePoseResult",
    ):
        assert name in values


def test_every_tutorial_links_to_its_user_first_spherical_cv_section() -> None:
    capability_map = (
        ROOT / "docs/guides/tutorials/spherical_capability_map.md"
    ).read_text(encoding="utf-8")
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
        tutorial = (ROOT / "docs/guides/tutorials" / filename).read_text(
            encoding="utf-8"
        )
        assert f"({anchor})=" in capability_map
        assert f"{{ref}}`{anchor}`" in tutorial
        if filename != "00_quick_start.md":
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
    related = (ROOT / "docs/concepts/related_libraries.rst").read_text(encoding="utf-8")

    assert "nitpicky = True" in conf
    assert "suppress_warnings = []" in conf
    for tier in ("Stable", "Compatibility", "Experimental", "Frozen/internal"):
        assert tier in stability
    for project in ("py360convert", "pyequilib", "OpenCV"):
        assert project in related
    assert "No performance superiority is claimed" in related
    assert "my_pano.jpg" not in "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "docs/guides/tutorials").glob("*")
        if path.suffix in {".md", ".rst"}
    )


def test_legacy_documentation_urls_redirect_to_the_new_taxonomy() -> None:
    sys.path.insert(0, str(ROOT / "docs/_ext"))
    from legacy_redirects import legacy_redirects

    redirects = legacy_redirects(
        {
            "concepts/workflow-evolution",
            "development/benchmarks",
            "getting-started/index",
            "guides/index",
            "guides/pose_range/depth_anything_views",
            "guides/tutorials/00_quick_start",
            "reference/capabilities",
            "releases/checklists/3.7.0",
            "research/index",
        }
    )
    assert redirects["applications/index"] == "guides/index"
    assert redirects["applications/pose_range/depth_anything_views"] == (
        "guides/pose_range/depth_anything_views"
    )
    assert redirects["tutorials/00_quick_start"] == ("guides/tutorials/00_quick_start")
    assert redirects["explanation/workflow-evolution"] == (
        "concepts/workflow-evolution"
    )
    assert redirects["capabilities/index"] == "reference/capabilities"
    assert redirects["benchmarks"] == "development/benchmarks"
    assert redirects["start/index"] == "getting-started/index"
    assert redirects["release-3.7.0-checklist"] == ("releases/checklists/3.7.0")
    assert redirects["papers/index"] == "research/index"


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
        "spherical-cnn-porting.svg",
        "metric3d-execution-routes.svg",
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
    core_generator = (ROOT / "scripts/generate_core_workflow_figures.py").read_text(
        encoding="utf-8"
    )
    deep_generator = (ROOT / "scripts/generate_deep_learning_figures.py").read_text(
        encoding="utf-8"
    )
    metric3d_highres_generator = (
        ROOT / "scripts/generate_metric3d_high_resolution_preview.py"
    ).read_text(encoding="utf-8")
    deep_metadata = json.loads(
        (TUTORIAL_MEDIA / "deep-learning-figure-metadata.json").read_text(
            encoding="utf-8"
        )
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
        ("spherical-dog-pyramid.jpg", (1200, 600)),
        ("projected-canny.jpg", (1200, 300)),
        ("spherical-gradient-components.jpg", (1200, 200)),
        ("depth-anything-views.jpg", (1200, 300)),
        ("metric3d-large-spherical-1024x2048.jpg", (2048, 1024)),
        ("return-path-points.jpg", (1200, 300)),
        ("return-path-regions.jpg", (1200, 300)),
        ("return-path-view-evidence.jpg", (1200, 300)),
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
        "spherical-dog-pyramid.jpg",
        "projected-canny.jpg",
        "spherical-gradient-components.jpg",
        "depth-anything-views.jpg",
        "metric3d-large-spherical-1024x2048.jpg",
        "return-path-points.jpg",
        "return-path-regions.jpg",
        "return-path-view-evidence.jpg",
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
        (TUTORIAL_MEDIA / "spherical-dog-pyramid.jpg").read_bytes()
    ).hexdigest() == (
        "41b430745479803c6285071b72a418b7f62012a1fa49c9ed31cbf29fd0391490"
    )
    assert hashlib.sha256(
        (TUTORIAL_MEDIA / "projected-canny.jpg").read_bytes()
    ).hexdigest() == (
        "6f91d7ff0f61672c9769a8ef5ecb03cb8428e9fb88890055e3458c85eb4b9b68"
    )
    assert hashlib.sha256(
        (TUTORIAL_MEDIA / "spherical-gradient-components.jpg").read_bytes()
    ).hexdigest() == (
        "ef6a2eb8472e2e3bacf54b970db74e78bb0a5c83a7a0fa3072ad34b085838ef3"
    )
    assert hashlib.sha256(
        (TUTORIAL_MEDIA / "depth-anything-views.jpg").read_bytes()
    ).hexdigest() == (
        "1349cb77a45128ce8b3c7a9c5c91f075e6ecf4479b144dd3e598b3bb773a83ce"
    )
    assert hashlib.sha256(
        (TUTORIAL_MEDIA / "metric3d-large-spherical-1024x2048.jpg").read_bytes()
    ).hexdigest() == (
        "5770255ad4c072f951a2b7201b407b4e0000ce5642da3f819af7f228a6868b64"
    )
    for filename, expected_hash in (
        (
            "return-path-points.jpg",
            "2637b1269a1e8b4ae18caeb46965509bc7985807de1b6ce3fcdf0adfc17e6b90",
        ),
        (
            "return-path-regions.jpg",
            "6edbe44e27122b94367e1236c0644e64a57a26ff3f7bbf47c66acce5e3f345be",
        ),
        (
            "return-path-view-evidence.jpg",
            "fa026637cf008456abe0684ee51a097a5236f8a87bba6518e8b7611bcfa3df85",
        ),
    ):
        assert hashlib.sha256((TUTORIAL_MEDIA / filename).read_bytes()).hexdigest() == (
            expected_hash
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
    assert metadata["projected_canny"]["edge_pixels"] > 0
    assert metadata["spherical_gradient_components"]["components"] == [
        "east",
        "north",
        "magnitude",
    ]
    assert deep_metadata["schema"] == "panorai-deep-learning-doc-figures/v1"
    assert deep_metadata["depth_anything_3"]["parameter_count"] == 334_171_394
    assert deep_metadata["metric3d"]["parameter_count"] == 203_243_090
    assert deep_metadata["metric3d"]["ported_spatial_operations"] == 66
    assert deep_metadata["metric3d"]["conv_transpose_policy"] == "resize_conv"
    assert deep_metadata["metric3d"]["ported_operation_types"] == {
        "SphericalConv2d": 63,
        "SphericalResizeConv2d": 3,
    }
    assert deep_metadata["metric3d"]["remaining_planar_spatial_layers"] == 0
    assert deep_metadata["metric3d_high_resolution"]["input_shape_hw"] == [
        1024,
        2048,
    ]
    assert deep_metadata["metric3d_high_resolution"]["output_shape_hw"] == [
        1024,
        2048,
    ]
    assert deep_metadata["metric3d_high_resolution"]["output_sha256"] in (
        metric3d_highres_generator
    )
    assert (
        deep_metadata["metric3d"]["native_median_ms"]
        < deep_metadata["metric3d"]["multiface_median_ms"]
    )
    assert "Metric3D preview must remain outside the repository" in deep_generator
    assert "metric3d-execution-routes.svg" in attribution
    assert hashlib.sha256(
        (TUTORIAL_MEDIA / "metric3d-execution-routes.svg").read_bytes()
    ).hexdigest() == (
        "ddfd2658be6be7879dc7e169ed8e49d6133fa46e239d5dce64508cf1b33bb9ba"
    )
    assert deep_metadata["input_sha256"] in deep_generator
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
    assert metadata["spherical_dog_pyramid"] == {
        "display": "absolute DoG normalized at per-octave 99.5 percentile",
        "input_sha256": (
            "5333c2cecc55468fcbc64a252f4bc097fab064cae70bead4f76d6c2e6101c524"
        ),
        "octave_shape_hw": [[256, 512], [128, 256], [64, 128]],
        "output_sha256": (
            "41b430745479803c6285071b72a418b7f62012a1fa49c9ed31cbf29fd0391490"
        ),
        "sigma_pair": [1.0, 2.0],
        "working_shape_hw": [256, 512],
    }
    assert metadata["spherical_dog_pyramid"]["input_sha256"] in core_generator
    assert metadata["spherical_dog_pyramid"]["output_sha256"] in attribution


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
    pinhole_workflow = metadata["applications"]["pinhole_workflow"]
    assert pinhole_workflow["layout"] == "icosahedron"
    assert pinhole_workflow["face_count"] == 12
    assert pinhole_workflow["pinhole_operation"]["name"] == "GaussianBlur"
    assert pinhole_workflow["support_fraction"] == 1.0
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
            "25c949dda66f0a9043ab6cbb80a72308eadd4a8cea82718fee17d6b62ab4b5be",
        ),
        "pinhole-workflow-sundowner.jpg": (
            (1200, 300),
            "dee93ce269ba3f95717cd50a31078f9cc0b7a32bedb38b3b15ac0b75b2834a4d",
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
        "4b6208518d276568946c80d6f0481a93ba1ceb97dca011ca2e24f622c78567ff"
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
        "12_composing_spherical_models.rst",
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
    markdown_tutorials = tuple(
        name
        for name in tutorials
        if name.endswith(".md") and name != "00_quick_start.md"
    )
    for filename in markdown_tutorials:
        text = (ROOT / "docs/guides/tutorials" / filename).read_text(encoding="utf-8")
        for field in fields:
            assert text.count(field) >= 1, f"{filename} is missing {field}"

    for filename in tutorials:
        if filename == "00_quick_start.md":
            continue
        text = (ROOT / "docs/guides/tutorials" / filename).read_text(encoding="utf-8")
        assert "How this application is composed" in text, (
            f"{filename} does not explain its core composition"
        )


def test_quick_start_leads_with_the_two_core_execution_routes() -> None:
    quick = (ROOT / "docs/guides/tutorials/00_quick_start.md").read_text(
        encoding="utf-8"
    )
    assert len(quick.splitlines()) <= 195
    assert "How this application is composed" not in quick
    assert "| Field | Contract |" not in quick
    assert "## 1. Multi-view: sample, process, fuse" in quick
    assert "## 2. Native spherical operations" in quick
    assert "## 3. View output types" in quick
    assert quick.count("```{image}") == 11
    assert quick.count("```python") == 10
    assert "{grid-item-card}" not in quick
    assert "../pose_range/depth_anything_views" in quick
    for public_name in (
        "EquirectangularImage",
        "GnomonicFace",
        "GnomonicFaceSet",
        "process_views()",
        "SphericalGradient",
        "SphericalConv2d",
        "SphericalResize",
        "SphericalMaxPool2d",
        "map_points()",
        "map_regions()",
        "map_views()",
    ):
        assert public_name in quick


def test_application_learning_paths_are_progressive_and_complete() -> None:
    root = ROOT / "docs/guides"
    expected = {
        "projection_processing": (
            "pinhole_pipeline",
            "sampling_and_fusion",
            "outputs_and_observations",
        ),
        "features_matching": (
            "quickstart",
            "projected_features",
            "sphere_native_features",
            "matching_and_diagnostics",
        ),
        "pose_range": (
            "relative_pose",
            "metric_scale_and_triangulation",
            "dense_stereo",
            "monocular_depth",
            "metric3d_sphere_native",
            "depth_anything_views",
            "validate_range",
        ),
        "semantic_regions": (
            "spherical_cam",
            "interpreting_cam",
            "angular_proposals",
            "multiview_association",
            "mask_refinement",
            "evaluation",
        ),
    }
    for application, stages in expected.items():
        index_path = root / application / "index.md"
        index = index_path.read_text(encoding="utf-8")
        assert "{toctree}" not in index
        assert ":doc:" not in index
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", index):
            if "://" in target or target.startswith("#"):
                continue
            local_target = target.split("#", 1)[0]
            assert (index_path.parent / local_target).resolve().is_file(), (
                f"{index_path} has a broken Markdown target: {target}"
            )
        for stage in stages:
            stage_path = root / application / f"{stage}.md"
            assert stage_path.is_file()
            assert stage in index
            stage_text = stage_path.read_text(encoding="utf-8")
            assert len(stage_text.split()) >= 120, (
                f"{stage_path} is a routing stub rather than an explanatory stage"
            )
            assert "## " in stage_text
            assert "{doc}`" not in stage_text
            for target in re.findall(r"(?<!!)\[[^]]+\]\(([^)]+)\)", stage_text):
                if "://" in target or target.startswith("#"):
                    continue
                local_target = target.split("#", 1)[0]
                assert (stage_path.parent / local_target).resolve().is_file(), (
                    f"{stage_path} has a broken Markdown target: {target}"
                )
            for block_index, block in enumerate(
                re.findall(r"```python\n(.*?)```", stage_text, flags=re.DOTALL)
            ):
                compile(block, f"{stage_path}:python-block-{block_index + 1}", "exec")

    pose = (root / "pose_range/index.md").read_text(encoding="utf-8")
    assert "relative pose -> metric scale -> spherical stereo" in pose
    assert "sphere-native CNN or tangent-view depth model" in pose

    semantics = (root / "semantic_regions/index.md").read_text(encoding="utf-8")
    ordered = (
        "spherical_cam",
        "interpreting_cam",
        "angular_proposals",
        "multiview_association",
        "mask_refinement",
        "evaluation",
    )
    assert [semantics.index(stage) for stage in ordered] == sorted(
        semantics.index(stage) for stage in ordered
    )
    da3 = (root / "pose_range/depth_anything_views.md").read_text(encoding="utf-8")
    assert "not converted to a sphere-native network" in da3
    cam = (root / "semantic_regions/spherical_cam.md").read_text(encoding="utf-8")
    assert "port_module_with_report" in cam
    proposals = (root / "semantic_regions/angular_proposals.md").read_text(
        encoding="utf-8"
    )
    assert "R-CNN" in proposals
    assert "angular centre + horizontal/vertical FOV" in proposals


def test_readme_is_a_curated_entry_point_with_valid_local_links() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert len(readme.splitlines()) <= 240
    for heading in (
        "## Stable foundation: three objects",
        "## 1. Ready-made applications",
        "## 2. Compose the lower-level API",
        "## 3. Sphere-native operations and Torch layers",
        "## 4. Mask R-CNN taxonomy and the executable workflow",
        "## Reproducibility and safety",
        "## Go deeper",
    ):
        assert readme.count(heading) == 1
    assert "MultiChannelHandler" not in readme
    assert "panorai.applications" in readme
    assert "panorai.pipeline" in readme
    assert "panorai.ops.spherical" in readme
    assert "mask_rcnn.workflow" in readme
    assert "mask_rcnn.paper" in readme
    assert "SphericalMaskRCNN" in readme
    assert "UnboundMaskRCNNError" in readme
    assert "Experimental" in readme
    assert "prospective scientific gates" in readme
    assert 'panorama.views("fibonacci"' in readme
    assert 'EquirectangularImage.sample("spherical-grid"' in readme
    assert "I/O and safety boundary" in readme
    assert ".map(" in readme
    assert ".reconstruct(" in readme
    assert "PerView.parallel" in readme
    assert "RasterGrid" in readme
    assert "requires_grad" in readme
    for target in (
        "guides/composed_spherical_api.html",
        "reference/samplers_blenders.html",
        "reference/spherical_ops.html",
    ):
        assert f"https://rlsgarcia-code.github.io/PanorAi/{target}" in readme

    # README is the GitHub entry surface. Documentation links leave it for the
    # published HTML site; only repository artifacts such as media, changelog
    # and license remain relative.
    regular_links = re.findall(r"(?<!!)\[[^]]+\]\(([^)]+)\)", readme)
    assert not [
        target
        for target in regular_links
        if target.startswith("docs/") and not target.startswith("docs/_static/")
    ]

    relative_links = re.findall(r"\[[^]]+\]\(([^)]+)\)", readme)
    for target in relative_links:
        if "://" in target or target.startswith("#"):
            continue
        local_path = target.split("#", 1)[0]
        assert (ROOT / local_path).exists(), f"README link target is missing: {target}"

    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "include docs/_static/readme/*.svg" in manifest


def test_readme_python_examples_are_valid_and_stable_example_executes() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    python_blocks = re.findall(r"```python\n(.*?)```", readme, flags=re.DOTALL)
    assert len(python_blocks) == 6

    for index, block in enumerate(python_blocks):
        compile(block, f"README.md:python-block-{index + 1}", "exec")

    # The Stable object-workflow example executes without external callbacks,
    # models, datasets, or optional dependencies.
    subprocess.run(
        [
            sys.executable,
            "-c",
            python_blocks[0],
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_spherical_fcn_tutorial_documents_acquisition_and_evidence_boundary() -> None:
    tutorial = (ROOT / "docs/guides/tutorials/09_spherical_fcn_cam.md").read_text(
        encoding="utf-8"
    )
    normalized_tutorial = " ".join(tutorial.split())
    benchmark = (ROOT / "benchmarks/spherical_fcn_cam/README.md").read_text(
        encoding="utf-8"
    )
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
    assert "SphericalProxySegmenter" in normalized_tutorial


def test_spherical_fcn_tutorial_lists_the_complete_imagenet_proxy_vocabulary() -> None:
    from panorai.experimental.deep_learning.segmentation import (
        IMAGENET_PROXY_VOCABULARY,
    )

    tutorial = (ROOT / "docs/guides/tutorials/09_spherical_fcn_cam.md").read_text(
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
    tutorial = (
        ROOT / "docs/guides/tutorials/10_spherical_monocular_depth.md"
    ).read_text(encoding="utf-8")
    normalized_tutorial = " ".join(tutorial.split())
    for required in (
        "panorai[deep-learning-depth]",
        "accept_upstream_terms=True",
        "does not resize, crop, pad, or prefilter",
        "weights_only=True",
        "radial range",
        "no separate checkpoint/model-card license",
        "CNN and ViT portability",
        "included public prediction is qualitative execution evidence",
        "prove only mechanical availability",
        "does not prove",
    ):
        assert required in normalized_tutorial
