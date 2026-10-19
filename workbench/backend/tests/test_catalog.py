from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest

from panorai.catalog import CatalogSurface, DocumentationContext, public_catalog
from panorai_workbench import catalog as workbench_catalog
from panorai_workbench.catalog import (
    ADAPTER_BINDINGS,
    ALLOWLISTED_RUNNERS,
    AdapterContractError,
    experiment_catalog,
    runtime_fingerprint,
    validate_adapter_contract,
    workspace_catalog,
)


def test_catalog_separates_installed_surfaces_from_local_experiments() -> None:
    payload = experiment_catalog()
    assert payload["schema"] == "panorai-workbench-catalog/v2"
    assert len(payload["surfaces"]) == len(tuple(public_catalog().iter_surfaces()))
    assert len(payload["experiments"]) == len(ADAPTER_BINDINGS) == 22
    assert {item["id"] for item in payload["surfaces"]} != {
        item["id"] for item in payload["experiments"]
    }
    assert payload["catalog"] == {
        "interface": public_catalog().interface,
        "contract_id": public_catalog().contract_id,
        "release": public_catalog().release,
        "series": public_catalog().series,
    }


def test_runtime_fingerprint_never_describes_editable_as_wheel() -> None:
    payload = runtime_fingerprint()
    install = payload["panorai"]
    direct_url = install["direct_url"]
    if direct_url and direct_url.get("dir_info", {}).get("editable"):
        assert install["installation_type"] == "editable"


def test_catalog_projection_does_not_import_optional_compute_backends() -> None:
    root = Path(__file__).resolve().parents[3]
    source = f"""
import json, sys
sys.path[:0] = [{str(root)!r}, {str(root / "workbench" / "backend")!r}]
from panorai_workbench.catalog import experiment_catalog
payload = experiment_catalog()
print(json.dumps({{
    'surfaces': payload['counts']['surfaces'],
    'forbidden': sorted(name for name in sys.modules if name.split('.', 1)[0] in {{'torch', 'torchvision', 'open3d'}}),
}}))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", source],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["forbidden"] == []


def test_every_adapter_references_a_catalog_surface_and_safe_runner() -> None:
    validate_adapter_contract()
    payload = experiment_catalog()
    surface_ids = {item["id"] for item in payload["surfaces"]}
    for experiment in payload["experiments"]:
        assert experiment["surface_id"] in surface_ids
        assert experiment["input_modes"]
        assert experiment["workspace_targets"]
        assert isinstance(experiment["parameter_schema"], list)
        assert isinstance(experiment["artifact_types"], list)
        if experiment["adapter_status"] == "ready":
            assert experiment["id"] in ALLOWLISTED_RUNNERS


def test_legacy_experiment_ids_and_workspace_filters_are_preserved() -> None:
    ids = {item["id"] for item in experiment_catalog()["experiments"]}
    assert {
        "spherical-slic",
        "slic-op",
        "spherical-dog-features",
        "dog-op",
        "relative-pose",
    } <= ids
    single = workspace_catalog("single")
    two = workspace_catalog("two")
    assert all("single" in item["workspace_targets"] for item in single["experiments"])
    assert all("two" in item["workspace_targets"] for item in two["experiments"])
    assert "relative-pose" not in {item["id"] for item in single["experiments"]}
    assert "relative-pose" in {item["id"] for item in two["experiments"]}


class _CatalogWithSyntheticSurface:
    def __init__(self) -> None:
        self.base = public_catalog()
        self.interface = self.base.interface
        self.contract_id = self.base.contract_id
        self.release = self.base.release
        self.series = self.base.series
        self.documentation_site = self.base.documentation_site
        self.families = self.base.families + (
            {
                "id": "future-family",
                "title": "Future",
                "summary": "Synthetic discovery oracle.",
            },
        )
        self.synthetic = CatalogSurface(
            id="future-surface",
            domain="future",
            family="future-family",
            kind="component",
            tier="experimental",
            visibility="advanced",
            canonical_import="panorai.geometry",
            contract="future/v1",
            replacement=None,
            requires=(),
            roles=("geometry",),
            public_symbols=("GnomonicSpec",),
            _data={
                "id": "future-surface",
                "domain": "future",
                "family": "future-family",
                "kind": "component",
                "tier": "experimental",
                "visibility": "advanced",
                "canonical_import": "panorai.geometry",
                "contract": "future/v1",
                "replacement": None,
            },
        )

    def iter_surfaces(self, include_internal: bool = False):
        yield from self.base.iter_surfaces(include_internal=include_internal)
        yield self.synthetic

    def documentation_for(self, surface_id: str, component: str | None = None):
        if surface_id != self.synthetic.id:
            return self.base.documentation_for(surface_id, component=component)
        geometry = self.base.documentation_for("canonical-geometry")
        return DocumentationContext(
            site=geometry.site,
            primary=None,
            learn=geometry.learn,
            help=geometry.help,
            context=geometry.context,
            surface=self.synthetic,
        )


def test_new_surface_and_unknown_family_appear_without_adapter_changes() -> None:
    payload = experiment_catalog(_CatalogWithSyntheticSurface())  # type: ignore[arg-type]
    synthetic = next(
        item for item in payload["surfaces"] if item["id"] == "future-surface"
    )
    assert synthetic["family"] == "future-family"
    assert synthetic["adapter_ids"] == []
    assert synthetic["availability"] == "available"
    assert all(
        item["surface_id"] != "future-surface" for item in payload["experiments"]
    )


def test_stale_adapter_is_blocked_by_strict_contract(monkeypatch) -> None:
    monkeypatch.setitem(
        workbench_catalog.ADAPTER_BINDINGS,
        "stale-test-adapter",
        ("does-not-exist", "Missing", "Stale"),
    )
    with pytest.raises(AdapterContractError, match="does-not-exist"):
        validate_adapter_contract()


def test_documentation_links_come_from_public_catalog() -> None:
    geometry = next(
        item
        for item in experiment_catalog()["surfaces"]
        if item["id"] == "canonical-geometry"
    )
    docs = geometry["documentation"]
    assert docs["learn"] and docs["help"] and docs["context"]
    assert all(
        link["url"].startswith(public_catalog().documentation_site.base_url)
        for group in docs.values()
        if isinstance(group, list)
        for link in group
    )


def test_workbench_imports_no_private_panorai_modules() -> None:
    package = Path(__file__).parents[1] / "panorai_workbench"
    modules: list[str] = []
    for source in package.glob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        modules.extend(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("panorai")
        )
        modules.extend(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
            if alias.name.startswith("panorai")
        )
    assert modules
    assert all(
        not any(part.startswith("_") for part in module.split(".")[1:])
        for module in modules
    )
