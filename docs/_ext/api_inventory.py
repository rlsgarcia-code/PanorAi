"""Sphinx directives backed by contracts/public-api-surfaces-v2.json.

The contract manifest is the single source of truth for public-surface
coverage and stability.  The documentation may add explanations and usage
routes, but it must not maintain a second, hand-written list of public
symbols.
"""

from __future__ import annotations

import json
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive, directives


def _inventory() -> dict:
    path = (
        Path(__file__).resolve().parents[2]
        / "contracts"
        / "public-api-surfaces-v2.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> nodes.table:
    table = nodes.table()
    group = nodes.tgroup(cols=len(headers))
    table += group
    for _ in headers:
        group += nodes.colspec(colwidth=1)
    head = nodes.thead()
    body = nodes.tbody()
    group += head
    group += body

    def row(values: tuple[str, ...]) -> nodes.row:
        result = nodes.row()
        for value in values:
            entry = nodes.entry()
            entry += nodes.paragraph(text=value)
            result += entry
        return result

    head += row(headers)
    for values in rows:
        body += row(values)
    return table


def _symbols(surface: dict, inventory: dict | None = None) -> tuple[str, ...]:
    """Return the public names represented by one manifest surface."""

    symbols: list[str] = []
    for key in ("exports_exact", "classified_exports", "required_symbols"):
        symbols.extend(surface.get(key, ()))
    symbols.extend(surface.get("registry_names", ()))
    if inventory is not None:
        exemptions = inventory["policy"]["public_namespace_discovery"]["exemptions"]
        for module_name, exemption in exemptions.items():
            if surface["id"] not in exemption["surfaces"]:
                continue
            delegated = exemption.get("surface_exports", {}).get(surface["id"])
            if delegated is None and len(exemption["surfaces"]) == 1:
                delegated = exemption["exports_exact"]
            symbols.extend(f"{module_name}.{name}" for name in (delegated or ()))
    return tuple(dict.fromkeys(symbols))


def _selected_surfaces(
    inventory: dict,
    *,
    role: str | None = None,
    family: str | None = None,
    domain: str | None = None,
    tier: str | None = None,
    visibility: str | None = None,
) -> list[dict]:
    role_ids = None
    if role is not None:
        role_ids = set(inventory["documentation"]["assignments"].get(role, ()))
    return [
        item
        for item in inventory["surfaces"]
        if (role_ids is None or item["id"] in role_ids)
        and (family is None or item["family"] == family)
        and (domain is None or item["domain"] == domain)
        and (tier is None or item["tier"] == tier)
        and (visibility is None or item["visibility"] == visibility)
    ]


class ApiInventoryDirective(Directive):
    has_content = False
    option_spec = {"family": directives.unchanged}

    def run(self):
        inventory = _inventory()
        family = self.options.get("family")
        surfaces = _selected_surfaces(inventory, family=family)
        rows = [
            (
                item["id"],
                item["family"],
                item["tier"],
                item["visibility"],
                item["canonical_import"] or "—",
                item["contract"],
            )
            for item in surfaces
        ]
        return [
            _table(
                (
                    "Surface",
                    "Family",
                    "Tier",
                    "Visibility",
                    "Canonical import",
                    "Contract",
                ),
                rows,
            )
        ]


class ApiSymbolsDirective(Directive):
    """Render the exhaustive symbol catalog for selected public surfaces."""

    has_content = False
    option_spec = {
        "role": directives.unchanged,
        "family": directives.unchanged,
        "domain": directives.unchanged,
        "tier": directives.unchanged,
        "visibility": directives.unchanged,
    }

    def run(self):
        inventory = _inventory()
        surfaces = _selected_surfaces(
            inventory,
            role=self.options.get("role"),
            family=self.options.get("family"),
            domain=self.options.get("domain"),
            tier=self.options.get("tier"),
            visibility=self.options.get("visibility"),
        )
        rows = []
        for item in surfaces:
            symbols = _symbols(item, inventory)
            rows.append(
                (
                    item["id"],
                    item["tier"].title(),
                    item["canonical_import"] or "—",
                    item["contract"],
                    ", ".join(symbols) if symbols else "—",
                )
            )
        return [
            _table(
                (
                    "Surface",
                    "Stability",
                    "Canonical import",
                    "Contract",
                    "Public symbols or registry entries",
                ),
                rows,
            )
        ]


class ApiComponentsDirective(Directive):
    """Render every surface assigned to one public component role."""

    has_content = False
    required_arguments = 1

    def run(self):
        inventory = _inventory()
        role = self.arguments[0]
        roles = inventory["documentation"]["roles"]
        if role not in roles:
            raise self.error(f"unknown PanorAi public component role: {role}")
        surfaces = _selected_surfaces(inventory, role=role)
        rows = [
            (
                item["id"],
                item["tier"].title(),
                item["canonical_import"] or "—",
                item["contract"],
                ", ".join(_symbols(item, inventory)) or "—",
            )
            for item in surfaces
        ]
        return [
            _table(
                (
                    "Surface",
                    "Stability",
                    "Canonical import",
                    "Contract",
                    "Public components or entries",
                ),
                rows,
            )
        ]


class ApiFamiliesDirective(Directive):
    has_content = False

    def run(self):
        inventory = _inventory()
        rows = [
            (
                family["title"],
                family["summary"],
                str(
                    sum(
                        surface["family"] == family["id"]
                        for surface in inventory["surfaces"]
                    )
                ),
            )
            for family in inventory["families"]
            if family["id"] != "internal"
        ]
        return [_table(("Family", "Purpose", "Contracts"), rows)]


def setup(app):
    app.add_directive("panorai-api-components", ApiComponentsDirective)
    app.add_directive("panorai-api-inventory", ApiInventoryDirective)
    app.add_directive("panorai-api-families", ApiFamiliesDirective)
    app.add_directive("panorai-api-symbols", ApiSymbolsDirective)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
