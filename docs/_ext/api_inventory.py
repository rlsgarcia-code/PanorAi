"""Sphinx directives backed by contracts/public-api-surfaces-v2.json."""

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


class ApiInventoryDirective(Directive):
    has_content = False
    option_spec = {"family": directives.unchanged}

    def run(self):
        inventory = _inventory()
        family = self.options.get("family")
        surfaces = [
            item
            for item in inventory["surfaces"]
            if family is None or item["family"] == family
        ]
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
    return {"parallel_read_safe": True, "parallel_write_safe": True}
