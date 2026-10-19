#!/usr/bin/env python3
"""Installed-wheel consumer for the public catalog and documentation routes."""

from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path
import json
import sys


def _assert_installed_origin(package_file: str, source_root: Path) -> Path:
    origin = Path(package_file).resolve()
    try:
        origin.relative_to(source_root.resolve())
    except ValueError:
        return origin
    raise AssertionError(f"panorai resolved inside the source checkout: {origin}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()

    import panorai
    from panorai.catalog import public_catalog

    origin = _assert_installed_origin(panorai.__file__, args.source_root)
    catalog = public_catalog()
    surface = catalog.surface("spherical-applications-slic")
    docs = catalog.documentation_for(
        "spherical-applications-slic", component="SphericalSLIC"
    )

    assert catalog.interface == "panorai-catalog/v1"
    assert surface.public_symbols == ("SphericalSLIC",)
    assert surface.roles == ("application",)
    assert docs.primary is not None
    assert docs.primary.url.endswith("/guides/api-levels/applications.html")
    assert len(docs.learn) == 2
    assert len(docs.help) == 2
    assert len(docs.context) == 3
    assert docs.site.scope == "current-main"
    assert docs.site.versioned is False

    forbidden = {
        name.split(".", 1)[0]
        for name in sys.modules
        if name.split(".", 1)[0] in {"open3d", "torch", "torchvision"}
    }
    assert not forbidden, forbidden

    copy = catalog.as_dict()
    copy["release"] = "mutated consumer copy"
    assert catalog.release != copy["release"]

    print(
        json.dumps(
            {
                "status": "ok",
                "consumer": "public-catalog",
                "version": version("panorai"),
                "origin": str(origin),
                "catalog_release": catalog.release,
                "surface_count": len(tuple(catalog.iter_surfaces())),
                "primary": docs.primary.url,
                "scope": docs.site.scope,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
