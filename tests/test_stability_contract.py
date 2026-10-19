from __future__ import annotations

import ast
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "contracts" / "public-api-surfaces-v2.json"
SCHEMA_PATH = ROOT / "contracts" / "public-api-surfaces-v2.schema.json"
STABILITY_DOC = ROOT / "docs" / "reference" / "stability.rst"


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _docname_path(docname: str) -> Path:
    candidates = [ROOT / "docs" / f"{docname}.rst", ROOT / "docs" / f"{docname}.md"]
    existing = [candidate for candidate in candidates if candidate.is_file()]
    assert len(existing) == 1, (
        f"documentation target is missing or ambiguous: {docname}"
    )
    return existing[0]


def _modules_declaring_all() -> set[str]:
    modules: set[str] = set()
    package_root = ROOT / "panorai"
    for module_path in package_root.rglob("*.py"):
        relative_parts = module_path.relative_to(package_root).parts
        if any(
            part.startswith("_") and part != "__init__.py" for part in relative_parts
        ):
            continue
        tree = ast.parse(
            module_path.read_text(encoding="utf-8"), filename=str(module_path)
        )
        declares_all = False
        for node in tree.body:
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            if any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in targets
            ):
                declares_all = True
                break
        if not declares_all:
            continue
        relative = module_path.relative_to(package_root).with_suffix("")
        parts = list(relative.parts)
        if parts[-1] == "__init__":
            parts.pop()
        suffix = ".".join(parts)
        modules.add("panorai" + (f".{suffix}" if suffix else ""))
    return modules


def _requirements_available(surface: dict) -> bool:
    return all(
        importlib.util.find_spec(name) is not None
        for name in surface.get("requires", [])
    )


def _importable_exact_surfaces() -> list[dict]:
    return [
        surface
        for surface in _manifest()["surfaces"]
        if surface.get("exports_exact") is not None and _requirements_available(surface)
    ]


def _clean_python(source: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    environment["KMP_INIT_AT_FORK"] = "FALSE"
    environment["KMP_USE_SHM"] = "0"
    environment["OMP_NUM_THREADS"] = "1"
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_stability_manifest_has_unique_complete_surface_records() -> None:
    manifest = _manifest()
    assert manifest["contract_id"] == "panorai-public-api-surfaces/v2"
    assert manifest["release"] == "3.8.0"
    assert manifest["series"] == "3.x"
    assert manifest["tiers"] == [
        "stable",
        "compatibility",
        "experimental",
        "internal",
    ]

    surfaces = manifest["surfaces"]
    ids = [surface["id"] for surface in surfaces]
    assert len(ids) == len(set(ids))
    assert {surface["tier"] for surface in surfaces} == set(manifest["tiers"])

    for surface in surfaces:
        assert surface["domain"]
        assert surface["family"] in {item["id"] for item in manifest["families"]}
        assert surface["kind"] in {
            "product",
            "component",
            "compatibility",
            "internal",
        }
        assert surface["visibility"] in {"primary", "advanced", "internal"}
        assert "canonical_import" in surface
        assert "replacement" in surface
        assert surface["contract"]
        requirements = surface.get("requires", [])
        assert isinstance(requirements, list)
        assert requirements == list(dict.fromkeys(requirements))
        assert all(isinstance(name, str) and name for name in requirements)
        if surface["tier"] == "experimental":
            assert surface["promotion_gates"]


def test_inventory_is_valid_against_its_json_schema() -> None:
    try:
        import jsonschema
    except ImportError:
        # Dependency-boundary jobs intentionally install only runtime bounds.
        # The structural assertions above remain active there; the normal CI
        # dev environment performs the complete JSON Schema validation.
        return
    jsonschema.validate(
        instance=_manifest(),
        schema=json.loads(SCHEMA_PATH.read_text(encoding="utf-8")),
    )


def test_every_public_surface_has_a_documentation_role_and_live_route() -> None:
    manifest = _manifest()
    documentation = manifest["documentation"]
    assert documentation["contract_id"] == "panorai-public-documentation/v2"
    assert documentation["site"] == {
        "provider": "github-pages",
        "base_url": "https://rlsgarcia-code.github.io/PanorAi/",
        "scope": "current-main",
        "versioned": False,
        "home": "index",
        "catalog": "reference/catalog/index",
        "public_surfaces": "reference/catalog/public_surfaces",
        "source_repository": "https://github.com/rlsgarcia-code/PanorAi",
        "workflow": ".github/workflows/docs-pages.yml",
    }

    surfaces = {
        surface["id"]: surface
        for surface in manifest["surfaces"]
        if surface["visibility"] != "internal"
    }
    assigned: set[str] = set()
    for role, surface_ids in documentation["assignments"].items():
        assert role in documentation["roles"]
        assert len(surface_ids) == len(set(surface_ids)), role
        assert set(surface_ids) <= set(surfaces), role
        assigned.update(surface_ids)

        route = documentation["roles"][role]
        for field in ("catalog", "reference", "concept", "guide"):
            routed_document = _docname_path(route[field]).read_text(encoding="utf-8")
            markers = (
                f".. panorai-api-components:: {role}",
                f"{{panorai-api-components}} {role}",
            )
            assert any(marker in routed_document for marker in markers), (
                role,
                field,
            )

    assert assigned == set(surfaces)


def test_application_documentation_and_examples_are_fail_closed() -> None:
    manifest = _manifest()
    documentation = manifest["documentation"]
    components = documentation["application_components"]
    application_ids = set(documentation["assignments"]["application"])

    applications = importlib.import_module("panorai.applications")
    application_base = applications.Application
    framework_types = {"Application", "PipelineApplication"}
    concrete_applications = {
        name
        for name in applications.__all__
        if name not in framework_types
        and inspect.isclass(getattr(applications, name))
        and issubclass(getattr(applications, name), application_base)
    }
    assert set(components) == concrete_applications

    source_runner = (ROOT / "scripts" / "run_documentation_examples.py").read_text(
        encoding="utf-8"
    )
    surfaces = {surface["id"]: surface for surface in manifest["surfaces"]}
    for component_name, example in components.items():
        surface_id = example["surface"]
        assert surface_id in application_ids, component_name
        surface_symbols = {
            name
            for key in ("exports_exact", "classified_exports", "required_symbols")
            for name in surfaces[surface_id].get(key, ())
        }
        assert component_name in surface_symbols, (component_name, surface_id)
        classified_application_surfaces = {
            candidate_id
            for candidate_id in application_ids
            if component_name
            in {
                name
                for key in (
                    "exports_exact",
                    "classified_exports",
                    "required_symbols",
                )
                for name in surfaces[candidate_id].get(key, ())
            }
        }
        assert classified_application_surfaces == {surface_id}, component_name
        page = _docname_path(example["documentation"]).read_text(encoding="utf-8")
        documentation_token = example.get("documentation_token", component_name)
        assert documentation_token in page, component_name
        if runner_token := example.get("source_runner_token"):
            assert runner_token in source_runner, component_name
        if surfaces[surface_id]["tier"] == "stable":
            assert example.get("source_runner_token"), component_name
            source_example = example["source_example"]
            start = f"# {source_example}_START = None"
            end = f"# {source_example}_END = None"
            assert source_runner.count(start) == 1, component_name
            assert source_runner.count(end) == 1, component_name
            example_block = source_runner.split(start, 1)[1].split(end, 1)[0]
            assert component_name in example_block
            consumer = ROOT / example["installed_consumer"]
            assert consumer.is_file(), component_name
            assert documentation_token in consumer.read_text(encoding="utf-8")


def test_readme_feature_policy_is_bound_to_the_catalog_contract() -> None:
    manifest = _manifest()
    readme_policy = manifest["documentation"]["readme"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    public_ids = {
        surface["id"]
        for surface in manifest["surfaces"]
        if surface["visibility"] != "internal"
    }
    assert readme_policy["default"] == "catalog-only"
    assert set(readme_policy["featured"]) <= public_ids
    assert readme_policy["catalog_url"] in readme
    for surface_id, token in readme_policy["featured"].items():
        assert token in readme, surface_id


def test_new_public_modules_cannot_bypass_the_manifest() -> None:
    manifest = _manifest()
    discovered = _modules_declaring_all()
    surfaces = {surface["id"]: surface for surface in manifest["surfaces"]}
    audited = {
        surface["module"]
        for surface in manifest["surfaces"]
        if surface.get("module") in discovered
    }
    exemptions = manifest["policy"]["public_namespace_discovery"]["exemptions"]
    assert set(exemptions) <= discovered
    assert set(exemptions).isdisjoint(audited)
    assert discovered == audited | set(exemptions)
    for module_name, exemption in exemptions.items():
        assert exemption["reason"].strip()
        assert exemption["surfaces"]
        assert set(exemption["surfaces"]) <= set(surfaces)
        expected_requires = sorted(
            {
                requirement
                for surface_id in exemption["surfaces"]
                for requirement in surfaces[surface_id].get("requires", [])
            }
        )
        assert exemption.get("requires", []) == expected_requires, module_name
        if len(exemption["surfaces"]) > 1:
            delegated = exemption.get("surface_exports")
            assert delegated is not None, module_name
            assert set(delegated) == set(exemption["surfaces"]), module_name
            assert all(len(names) == len(set(names)) for names in delegated.values())
            delegated_names = [name for names in delegated.values() for name in names]
            assert len(delegated_names) == len(set(delegated_names)), module_name
            assert set(delegated_names) == set(exemption["exports_exact"]), module_name
        else:
            assert "surface_exports" not in exemption, module_name
        if any(
            importlib.util.find_spec(requirement) is None
            for requirement in exemption.get("requires", [])
        ):
            continue
        module = importlib.import_module(module_name)
        assert list(module.__all__) == exemption["exports_exact"], module_name


def test_declared_symbols_and_methods_exist_on_real_public_modules() -> None:
    # Several legacy tests intentionally install small ``panorai`` stubs at
    # collection time. Contract existence must be checked in a clean process,
    # exactly like a downstream import, rather than against those test doubles.
    completed = _clean_python("""
import importlib
import importlib.util
import json
from pathlib import Path

manifest = json.loads(Path('contracts/public-api-surfaces-v2.json').read_text())
for surface in manifest['surfaces']:
    module_name = surface.get('module')
    if module_name is None:
        continue
    if any(importlib.util.find_spec(name) is None for name in surface.get('requires', [])):
        continue
    module = importlib.import_module(module_name)
    declared = list(surface.get('required_symbols', []))
    declared += surface.get('classified_exports', [])
    declared += surface.get('exports_exact', [])
    for name in declared:
        assert hasattr(module, name), f"{surface['id']}: missing {module_name}.{name}"
    for owner_name, methods in surface.get('required_methods', {}).items():
        owner = getattr(module, owner_name)
        for method_name in methods:
            assert hasattr(owner, method_name), (
                f"{surface['id']}: missing {module_name}.{owner_name}.{method_name}"
            )
""")
    assert completed.returncode == 0, completed.stderr


def test_exact_module_exports_are_frozen_by_the_manifest() -> None:
    for surface in _importable_exact_surfaces():
        expected = surface["exports_exact"]
        module = importlib.import_module(surface["module"])
        assert module.__all__ == expected, surface["id"]


def test_exact_exports_skip_an_unavailable_optional_backend(monkeypatch) -> None:
    real_find_spec = importlib.util.find_spec

    def find_spec_without_torch(name: str):
        if name == "torch":
            return None
        return real_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec_without_torch)
    surface_ids = {surface["id"] for surface in _importable_exact_surfaces()}
    assert "spherical-core-torch" not in surface_ids


def test_every_feature_export_has_one_explicit_stability_classification() -> None:
    manifest = _manifest()
    feature_surfaces = [
        surface
        for surface in manifest["surfaces"]
        if surface.get("module") == "panorai.features"
    ]
    classified = [
        name
        for surface in feature_surfaces
        for name in surface.get("classified_exports", [])
    ]
    assert len(classified) == len(set(classified))

    features = importlib.import_module("panorai.features")
    assert set(classified) == set(features.__all__)


def test_every_audited_public_export_is_classified() -> None:
    manifest = _manifest()
    modules = sorted(
        {surface["module"] for surface in manifest["surfaces"] if "module" in surface}
    )
    for module_name in modules:
        surfaces = [
            item for item in manifest["surfaces"] if item.get("module") == module_name
        ]
        requirements = {
            requirement
            for surface in surfaces
            for requirement in surface.get("requires", [])
        }
        if not _requirements_available({"requires": sorted(requirements)}):
            continue
        module = importlib.import_module(module_name)
        public = set(getattr(module, "__all__", ()))
        classified = {
            name
            for surface in surfaces
            for key in ("required_symbols", "classified_exports", "exports_exact")
            for name in surface.get(key, [])
        }
        assert classified == public, module_name


def test_every_registered_blender_has_one_explicit_stability_classification() -> None:
    manifest = _manifest()
    blender_names = [
        name
        for surface in manifest["surfaces"]
        if surface.get("module") == "panorai.blenders"
        for name in surface.get("registry_names", [])
    ]
    assert len(blender_names) == len(set(blender_names))

    from panorai.blenders import BlenderRegistry

    assert set(blender_names) == set(BlenderRegistry.available_blenders())


def test_public_stability_document_is_generated_from_the_canonical_inventory() -> None:
    text = STABILITY_DOC.read_text(encoding="utf-8")
    manifest = _manifest()
    assert manifest["contract_id"] in text
    assert "contracts/public-api-surfaces-v2.json" in text
    assert ".. panorai-api-inventory::" in text
    extension = (ROOT / "docs" / "_ext" / "api_inventory.py").read_text(
        encoding="utf-8"
    )
    assert "public-api-surfaces-v2.json" in extension
    assert "panorai-api-inventory" in extension


def test_promoted_workflow_records_evidence_and_runtime_provenance() -> None:
    manifest = _manifest()
    workflow = next(
        surface
        for surface in manifest["surfaces"]
        if surface["id"] == "object-workflow"
    )
    assert workflow["tier"] == "stable"
    assert len(workflow["promotion_evidence"]) >= 2

    completed = _clean_python("""
import json
import numpy as np
import panorai

description = panorai.EquirectangularImage(
    np.zeros((4, 8, 3), dtype=np.float32)
).views(size=2).describe()
print(json.dumps(description, sort_keys=True))
""")
    assert completed.returncode == 0, completed.stderr
    description = json.loads(completed.stdout)
    assert description["contract"] == "geometry-v1"
    assert description["interface"] == workflow["contract"]
    assert description["stability"] == "stable"


def test_promoted_features_core_records_evidence_and_runtime_provenance() -> None:
    manifest = _manifest()
    core = next(
        surface
        for surface in manifest["surfaces"]
        if surface["id"] == "spherical-features-core"
    )
    assert core["tier"] == "stable"
    assert len(core["promotion_evidence"]) >= 2

    from panorai.features import SphericalFeaturePipeline

    description = SphericalFeaturePipeline.from_preset(
        "orb-hamming", face_shape_hw=16, max_features=8
    ).describe()
    assert description["interface"] == core["contract"]
    assert description["stability"] == "stable"
    assert description["experimental_extensions"] == [
        "build_virtual_camera_rig",
        "export_pycolmap",
    ]

    pycolmap_surface = next(
        surface
        for surface in manifest["surfaces"]
        if surface["id"] == "spherical-features-pycolmap"
    )
    multiscale_surface = next(
        surface
        for surface in manifest["surfaces"]
        if surface["id"] == "spherical-features-multiscale"
    )
    assert pycolmap_surface["tier"] == "experimental"
    assert multiscale_surface["tier"] == "experimental"


def test_reconstruction_and_slam_remain_explicit_no_promotion_surfaces() -> None:
    manifest = _manifest()
    for surface_id in ("spherical-reconstruction", "spherical-slam"):
        surface = next(
            item for item in manifest["surfaces"] if item["id"] == surface_id
        )
        assert surface["tier"] == "experimental"
        assert surface["promotion_recommendation"] == "do-not-promote-current-method"
        assert len(surface["validation_evidence"]) >= 2
        assert len(surface["promotion_gates"]) >= 4


def test_slic_is_stable_without_promoting_dog_or_other_applications() -> None:
    manifest = _manifest()
    by_id = {surface["id"]: surface for surface in manifest["surfaces"]}

    pipeline = by_id["spherical-pipeline-core"]
    instance_pipeline = by_id["spherical-pipeline-semantic-instances"]
    operations = by_id["spherical-ops-core"]
    morphology = by_id["spherical-ops-morphology"]
    slic_operation = by_id["spherical-ops-slic"]
    dog_operation = by_id["spherical-ops-dog-detector"]
    algorithms_gateway = by_id["spherical-ops-algorithms-gateway"]
    torch_operations = by_id["spherical-ops-torch"]
    general_algorithms = by_id["spherical-algorithms-general"]
    slic_algorithm = by_id["spherical-algorithms-slic"]
    dog_algorithm = by_id["spherical-algorithms-dog-detector"]
    applications = by_id["spherical-applications"]
    slic_application = by_id["spherical-applications-slic"]
    historical_slic = by_id["spherical-image-processing-slic"]

    assert pipeline["tier"] == "stable"
    assert pipeline["contract"] == "panorai-spherical-pipeline/v1"
    assert len(pipeline["promotion_evidence"]) >= 4
    assert operations["tier"] == "stable"
    assert operations["contract"] == "panorai-spherical-ops/v1"
    assert len(operations["promotion_evidence"]) >= 4
    assert morphology["tier"] == "stable"
    assert morphology["contract"] == "panorai-spherical-morphology/v1"
    assert len(morphology["promotion_evidence"]) >= 3

    assert instance_pipeline["tier"] == "experimental"
    assert slic_operation["tier"] == "stable"
    assert slic_operation["contract"] == "panorai-spherical-slic/v1"
    assert dog_operation["tier"] == "experimental"
    assert algorithms_gateway["tier"] == "experimental"
    assert torch_operations["tier"] == "stable"
    assert torch_operations["contract"] == "panorai-spherical-ops-torch/v1"
    assert len(torch_operations["promotion_evidence"]) >= 3
    assert general_algorithms["tier"] == "stable"
    assert slic_algorithm["tier"] == "stable"
    assert dog_algorithm["tier"] == "experimental"
    assert applications["tier"] == "experimental"
    assert slic_application["tier"] == "stable"
    assert historical_slic["tier"] == "stable"
    assert len(slic_application["promotion_evidence"]) >= 4
    assert {"SLIC", "DoGDetector", "algorithms"}.isdisjoint(
        operations["classified_exports"]
    )
    assert slic_operation["classified_exports"] == ["SLIC"]
    assert dog_operation["classified_exports"] == ["DoGDetector"]
    assert algorithms_gateway["classified_exports"] == ["algorithms"]
    assert slic_algorithm["classified_exports"] == ["SLIC"]
    assert dog_algorithm["classified_exports"] == ["DoGDetector"]
    assert {"SLIC", "DoGDetector"}.isdisjoint(general_algorithms["classified_exports"])

    from panorai.applications import SphericalDoGFeatures, SphericalSLIC
    from panorai.ops.spherical import DoGDetector, SLIC

    assert SLIC.interface == slic_operation["contract"]
    assert SLIC.stability == "stable"
    assert SphericalSLIC.interface == slic_application["contract"]
    assert SphericalSLIC.stability == "stable"
    assert DoGDetector.interface == dog_operation["contract"]
    assert DoGDetector.stability == "experimental"
    assert SphericalDoGFeatures.stability == "experimental"

    from panorai.pipeline import (
        CAMEvidenceFrame,
        CAMRegionProposal,
        CAMRegionProposals,
        Pipeline,
        PipelineRun,
        SAMMaskRefinement,
        SphericalCAM,
        SphericalInstanceFusion,
    )

    assert Pipeline.interface == pipeline["contract"]
    assert Pipeline.stability == "stable"
    assert PipelineRun.interface == pipeline["contract"]
    assert PipelineRun.stability == "stable"
    assert CAMEvidenceFrame.stability == "experimental"
    for experimental_type in (
        CAMRegionProposal,
        CAMRegionProposals,
        SAMMaskRefinement,
        SphericalCAM,
        SphericalInstanceFusion,
    ):
        assert experimental_type.interface.endswith("/v1-experimental")
        assert experimental_type.stability == "experimental"


def test_region_analysis_and_evaluation_are_stable_surfaces() -> None:
    manifest = _manifest()
    by_id = {surface["id"]: surface for surface in manifest["surfaces"]}
    surface_ids = (
        "spherical-region-analysis-ops",
        "spherical-region-analysis-data",
        "spherical-region-analysis-pipeline",
        "spherical-evaluation",
    )
    for surface_id in surface_ids:
        surface = by_id[surface_id]
        assert surface["tier"] == "stable"
        assert not surface["contract"].endswith("-experimental")
        assert len(surface["promotion_evidence"]) >= 2

    from panorai.data import RegionAnalysisResult, SphericalContour
    from panorai.evaluation import EvaluationProtocol
    from panorai.ops.spherical import Skeletonize
    from panorai.pipeline import AnalyzeRegions, SphericalWatershed

    for public_type in (RegionAnalysisResult, SphericalContour, EvaluationProtocol):
        assert public_type.stability == "stable"
    for public_type in (Skeletonize, AnalyzeRegions, SphericalWatershed):
        assert public_type.interface.endswith("/v1")
        assert public_type.stability == "stable"


def test_generic_torch_layers_are_stable_without_promoting_applications() -> None:
    manifest = _manifest()
    by_id = {surface["id"]: surface for surface in manifest["surfaces"]}

    assert by_id["spherical-core-torch"]["tier"] == "stable"
    assert by_id["spherical-ops-torch"]["tier"] == "stable"
    assert by_id["spherical-applications"]["tier"] == "experimental"
    assert by_id["spherical-applications-slic"]["tier"] == "stable"
    assert by_id["spherical-ops-dog-detector"]["tier"] == "experimental"
    assert by_id["spherical-pipeline-semantic-instances"]["tier"] == "experimental"


def test_generic_torch_runtime_types_report_stable_when_available() -> None:
    pytest.importorskip("torch")

    from panorai.core.torch import (
        SphericalConv2d,
        SphericalConvTranspose2d,
        SphericalMaxPool2d,
        SphericalResizeConv2d,
    )
    from panorai.ops.spherical import nn as spherical_nn

    for layer_type in (
        SphericalConv2d,
        SphericalConvTranspose2d,
        SphericalMaxPool2d,
        SphericalResizeConv2d,
    ):
        assert layer_type.stability == "stable"
        assert layer_type.interface == "panorai-spherical-torch-convolution/v1"

    for layer_type in (
        spherical_nn.Closing,
        spherical_nn.Conv2d,
        spherical_nn.ConvTranspose2d,
        spherical_nn.Dilate,
        spherical_nn.Erode,
        spherical_nn.MaxPool2d,
        spherical_nn.Opening,
        spherical_nn.AvgPool2d,
        spherical_nn.AdaptiveAvgPool2d,
        spherical_nn.GlobalAveragePool,
        spherical_nn.Resize,
        spherical_nn.Interpolate,
    ):
        assert layer_type.stability == "stable"
        assert layer_type.interface == "panorai-spherical-ops-torch/v1"


def test_experimental_cam_wrapper_does_not_inherit_pipeline_stability() -> None:
    from panorai.pipeline import CAMGuidedSphericalInstances

    def predictor(_value):
        raise AssertionError("describe must not execute the predictor")

    class Segmenter:
        def encode(self, *_args, **_kwargs):
            raise AssertionError("describe must not execute the segmenter")

        def predict(self, *_args, **_kwargs):
            raise AssertionError("describe must not execute the segmenter")

        def release(self, *_args, **_kwargs):
            raise AssertionError("describe must not execute the segmenter")

    description = CAMGuidedSphericalInstances(
        cam=predictor,
        sam=Segmenter(),
        concept_ids=("object",),
    ).describe()
    assert description["interface"].endswith("/v1-experimental")
    assert description["stability"] == "experimental"
