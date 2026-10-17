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


def _symbols(surface: dict) -> tuple[str, ...]:
    """Return the public names represented by one manifest surface."""

    for key in ("exports_exact", "classified_exports", "required_symbols"):
        if key in surface:
            return tuple(surface[key])
    if "registry_names" in surface:
        return tuple(surface["registry_names"])
    return ()


def _selected_surfaces(
    inventory: dict,
    *,
    family: str | None = None,
    domain: str | None = None,
    tier: str | None = None,
    visibility: str | None = None,
) -> list[dict]:
    return [
        item
        for item in inventory["surfaces"]
        if (family is None or item["family"] == family)
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
        "family": directives.unchanged,
        "domain": directives.unchanged,
        "tier": directives.unchanged,
        "visibility": directives.unchanged,
    }

    def run(self):
        inventory = _inventory()
        surfaces = _selected_surfaces(
            inventory,
            family=self.options.get("family"),
            domain=self.options.get("domain"),
            tier=self.options.get("tier"),
            visibility=self.options.get("visibility"),
        )
        rows = []
        for item in surfaces:
            symbols = _symbols(item)
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
    app.add_directive("panorai-api-inventory", ApiInventoryDirective)
    app.add_directive("panorai-api-families", ApiFamiliesDirective)
    app.add_directive("panorai-api-symbols", ApiSymbolsDirective)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
