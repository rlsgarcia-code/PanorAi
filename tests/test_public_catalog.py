from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from panorai.catalog import (
    CatalogSurface,
    DocumentationContext,
    DocumentationLink,
    DocumentationSite,
    PublicCatalog,
    public_catalog,
)


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "contracts" / "public-api-surfaces-v2.json"
PACKAGED = ROOT / "panorai" / "catalog" / "public-api-surfaces-v2.json"
CANONICAL_SCHEMA = ROOT / "contracts" / "public-api-surfaces-v2.schema.json"
PACKAGED_SCHEMA = ROOT / "panorai" / "catalog" / "public-api-surfaces-v2.schema.json"


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_catalog_exposes_immutable_versioned_surface() -> None:
    catalog = public_catalog()
    assert isinstance(catalog, PublicCatalog)
    assert public_catalog() is catalog
    assert catalog.interface == "panorai-catalog/v1"
    assert catalog.contract_id == "panorai-public-api-surfaces/v2"
    assert catalog.release == "3.8.0"
    assert catalog.series == "3.x"
    assert catalog.documentation_site.scope == "current-main"
    assert catalog.documentation_site.versioned is False
    assert catalog.documentation_site.provider == "github-pages"

    exported_types = (
        CatalogSurface,
        DocumentationContext,
        DocumentationLink,
        DocumentationSite,
        PublicCatalog,
    )
    assert all(item.__module__.startswith("panorai.catalog") for item in exported_types)

    with pytest.raises((AttributeError, TypeError)):
        catalog.release = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        catalog.families[0]["id"] = "changed"  # type: ignore[index]

    serialized = catalog.as_dict()
    serialized["release"] = "changed"
    serialized["surfaces"][0]["id"] = "changed"
    assert catalog.release == "3.8.0"
    assert catalog.surface("public-catalog").id == "public-catalog"


def test_surface_lookup_iteration_roles_and_unknown_id() -> None:
    catalog = public_catalog()
    slic = catalog.surface("spherical-applications-slic")
    assert slic.canonical_import == "panorai.applications"
    assert slic.tier == "stable"
    assert slic.roles == ("application",)
    assert slic.public_symbols == ("SphericalSLIC",)
    assert slic.requires == ()

    public_ids = {surface.id for surface in catalog.iter_surfaces()}
    all_ids = {surface.id for surface in catalog.iter_surfaces(include_internal=True)}
    assert "implementation-internals" not in public_ids
    assert "implementation-internals" in all_ids
    assert public_ids < all_ids
    with pytest.raises(KeyError, match="does-not-exist"):
        catalog.surface("does-not-exist")


def test_public_symbols_include_delegated_namespace_exemptions() -> None:
    catalog = public_catalog()
    morphology = catalog.surface("spherical-ops-morphology")
    assert "closing" in morphology.public_symbols
    assert "panorai.ops.spherical.functional.closing" in morphology.public_symbols
    assert len(morphology.public_symbols) == len(set(morphology.public_symbols))


def test_documentation_context_separates_primary_learn_help_and_context() -> None:
    catalog = public_catalog()
    docs = catalog.documentation_for(
        "spherical-applications-slic", component="SphericalSLIC"
    )
    assert docs.primary is not None
    assert docs.primary.docname == "guides/api-levels/applications"
    assert docs.primary.url == (
        "https://rlsgarcia-code.github.io/PanorAi/guides/api-levels/applications.html"
    )
    assert {(link.rel, link.role) for link in docs.learn} == {
        ("guide", "application"),
        ("concept", "application"),
    }
    assert {(link.rel, link.role) for link in docs.help} == {
        ("reference", "application"),
        ("catalog", "application"),
    }
    assert [link.rel for link in docs.context] == [
        "home",
        "catalog",
        "public-surfaces",
    ]
    assert docs.context[0].url.endswith("/index.html")
    assert docs.context[1].url.endswith("/reference/catalog/index.html")
    assert docs.context[2].url.endswith("/reference/catalog/public_surfaces.html")
    assert "#" not in docs.primary.url


def test_documentation_preserves_multiple_roles_and_rejects_bad_components() -> None:
    catalog = public_catalog()
    docs = catalog.documentation_for("core-geometry-preview")
    assert docs.surface.roles == ("geometry", "compatibility")
    assert {link.role for link in docs.learn} == {"geometry", "compatibility"}
    assert {link.role for link in docs.help} == {"geometry", "compatibility"}
    assert docs.primary is None

    with pytest.raises(KeyError, match="UnknownApplication"):
        catalog.documentation_for(
            "spherical-applications-slic", component="UnknownApplication"
        )
    with pytest.raises(ValueError, match="belongs to surface"):
        catalog.documentation_for(
            "spherical-applications-slic", component="SphericalRelativePose"
        )


def test_canonical_and_packaged_contract_resources_are_byte_identical() -> None:
    assert PACKAGED.read_bytes() == CANONICAL.read_bytes()
    assert PACKAGED_SCHEMA.read_bytes() == CANONICAL_SCHEMA.read_bytes()
    completed = subprocess.run(
        [sys.executable, "scripts/sync_public_catalog.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_sync_check_detects_divergence(tmp_path: Path, monkeypatch) -> None:
    sync = _load_script("sync_public_catalog.py")
    contracts = tmp_path / "contracts"
    package = tmp_path / "package"
    contracts.mkdir()
    package.mkdir()
    for name in sync.RESOURCES:
        (contracts / name).write_bytes(b"canonical\n")
        (package / name).write_bytes(b"stale\n")
    monkeypatch.setattr(sync, "CONTRACTS", contracts)
    monkeypatch.setattr(sync, "PACKAGE", package)

    assert sync.synchronize(check=True) == sync.RESOURCES
    assert all((package / name).read_bytes() == b"stale\n" for name in sync.RESOURCES)
    assert sync.synchronize(check=False) == sync.RESOURCES
    assert sync.synchronize(check=True) == ()


def test_import_is_base_only_and_does_not_load_optional_backends() -> None:
    source = (
        f"import sys\nsys.path.insert(0, {str(ROOT)!r})\n"
        + """
import json
from panorai.catalog import public_catalog

catalog = public_catalog()
print(json.dumps({
    'surface': catalog.surface('spherical-applications-slic').id,
    'forbidden': sorted(
        name for name in sys.modules
        if name.split('.', 1)[0] in {'torch', 'torchvision', 'open3d'}
    ),
}))
"""
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", source],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result == {"surface": "spherical-applications-slic", "forbidden": []}


def test_all_declared_documentation_routes_have_source_pages() -> None:
    verifier = _load_script("verify_public_catalog_docs.py")
    manifest = json.loads(CANONICAL.read_text(encoding="utf-8"))
    for docname in verifier.declared_docnames(manifest):
        candidates = (ROOT / "docs" / f"{docname}.rst", ROOT / "docs" / f"{docname}.md")
        assert sum(path.is_file() for path in candidates) == 1, docname
