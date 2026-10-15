"""Generate compatibility pages for public documentation URLs that moved."""

from __future__ import annotations

from html import escape
import os
from pathlib import Path


_CONCEPT_PAGES = {
    "architecture",
    "related_libraries",
    "spherical_dense_stereo",
    "spherical_frontend_promotion_plan",
    "workflow-evolution",
}


def legacy_redirects(docnames: set[str]) -> dict[str, str]:
    """Return legacy docname to canonical docname mappings."""

    redirects: dict[str, str] = {}
    for target in docnames:
        if target == "getting-started/index":
            redirects["start/index"] = target
        elif target.startswith("guides/tutorials/"):
            redirects[target.removeprefix("guides/")] = target
        elif target.startswith("guides/how-to/"):
            redirects["how_to/" + target.removeprefix("guides/how-to/")] = target
        elif target == "guides/index" or target.startswith(
            (
                "guides/features_matching/",
                "guides/pose_range/",
                "guides/projection_processing/",
                "guides/semantic_regions/",
            )
        ):
            redirects["applications/" + target.removeprefix("guides/")] = target
        elif target.removeprefix("concepts/") in _CONCEPT_PAGES:
            redirects["explanation/" + target.removeprefix("concepts/")] = target
        elif target == "reference/capabilities":
            redirects["capabilities/index"] = target
        elif target == "development/benchmarks":
            redirects["benchmarks"] = target
        elif target == "development/evidence":
            redirects["evidence/index"] = target
        elif target == "research/index" or target.startswith("research/"):
            redirects["papers/" + target.removeprefix("research/")] = target
        elif target.startswith("releases/checklists/"):
            version = target.removeprefix("releases/checklists/")
            redirects[f"release-{version}-checklist"] = target
    return redirects


def _write_redirects(app, exception) -> None:
    if exception is not None or app.builder.format != "html":
        return

    for legacy, target in legacy_redirects(set(app.env.found_docs)).items():
        old_path = Path(app.builder.get_outfilename(legacy))
        target_path = Path(app.builder.get_outfilename(target))
        relative_target = Path(
            os.path.relpath(target_path, start=old_path.parent)
        ).as_posix()
        old_path.parent.mkdir(parents=True, exist_ok=True)
        old_path.write_text(
            "<!doctype html>\n"
            '<html lang="en"><head><meta charset="utf-8">\n'
            f'<link rel="canonical" href="{escape(relative_target)}">\n'
            f'<meta http-equiv="refresh" content="0; url={escape(relative_target)}">\n'
            "<title>Documentation moved</title></head><body>\n"
            f'<p>This page moved to <a href="{escape(relative_target)}">'
            f"{escape(target)}</a>.</p>\n"
            "</body></html>\n",
            encoding="utf-8",
        )


def setup(app):
    app.connect("build-finished", _write_redirects)
    return {"version": "1", "parallel_read_safe": True}
