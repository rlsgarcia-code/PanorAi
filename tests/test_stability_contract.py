from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "contracts" / "public-api-surfaces-v2.json"
SCHEMA_PATH = ROOT / "contracts" / "public-api-surfaces-v2.schema.json"
STABILITY_DOC = ROOT / "docs" / "reference" / "stability.rst"


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _clean_python(source: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    environment["KMP_INIT_AT_FORK"] = "FALSE"
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
    assert manifest["release"] == "3.7.0"
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
    for surface in _manifest()["surfaces"]:
        expected = surface.get("exports_exact")
        if expected is None:
            continue
        module = importlib.import_module(surface["module"])
        assert module.__all__ == expected, surface["id"]


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
        if any(importlib.util.find_spec(name) is None for name in requirements):
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
