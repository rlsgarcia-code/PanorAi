"""Installed PanorAi public-surface and documentation catalog.

The catalog reads only package resources that ship with the installed
distribution.  Importing this module does not import optional numerical or
visual-computing backends.
"""

from ._catalog import (
    CatalogSurface,
    DocumentationContext,
    DocumentationLink,
    DocumentationSite,
    PublicCatalog,
    public_catalog,
)

__all__ = [
    "CatalogSurface",
    "DocumentationContext",
    "DocumentationLink",
    "DocumentationSite",
    "PublicCatalog",
    "public_catalog",
]
