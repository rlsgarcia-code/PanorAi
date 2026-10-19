"""Immutable models for the installed PanorAi public catalog."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
import json
from types import MappingProxyType
from typing import Any
from urllib.parse import urljoin


CATALOG_INTERFACE = "panorai-catalog/v1"
_MANIFEST_RESOURCE = "public-api-surfaces-v2.json"


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"catalog field {name!r} must be an object")
    return value


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("catalog string-list field must be an array")
    if not all(isinstance(item, str) for item in value):
        raise ValueError("catalog string-list entries must be strings")
    return tuple(value)


def public_symbols_for(
    surface: Mapping[str, Any], manifest: Mapping[str, Any]
) -> tuple[str, ...]:
    """Expand every public name represented by a manifest surface.

    This is the one implementation shared by the installed catalog and the
    Sphinx inventory extension.  Delegated namespace exemptions are qualified
    by their declaring module so they cannot collide with ordinary exports.
    """

    symbols: list[str] = []
    for key in ("exports_exact", "classified_exports", "required_symbols"):
        symbols.extend(_string_tuple(surface.get(key)))
    symbols.extend(_string_tuple(surface.get("registry_names")))

    policy = _mapping(manifest.get("policy"), name="policy")
    discovery = _mapping(
        policy.get("public_namespace_discovery"),
        name="policy.public_namespace_discovery",
    )
    exemptions = _mapping(
        discovery.get("exemptions"),
        name="policy.public_namespace_discovery.exemptions",
    )
    surface_id = surface.get("id")
    for module_name, raw_exemption in exemptions.items():
        exemption = _mapping(raw_exemption, name=f"exemption {module_name}")
        exemption_surfaces = _string_tuple(exemption.get("surfaces"))
        if surface_id not in exemption_surfaces:
            continue
        surface_exports = _mapping(
            exemption.get("surface_exports", {}),
            name=f"exemption {module_name}.surface_exports",
        )
        delegated = surface_exports.get(str(surface_id))
        if delegated is None and len(exemption_surfaces) == 1:
            delegated = exemption.get("exports_exact")
        symbols.extend(f"{module_name}.{name}" for name in _string_tuple(delegated))
    return tuple(dict.fromkeys(symbols))


@dataclass(frozen=True, slots=True)
class DocumentationLink:
    """One stable relation from a catalog entry to the published site."""

    rel: str
    title: str
    docname: str
    url: str
    role: str | None = None


@dataclass(frozen=True, slots=True)
class DocumentationSite:
    """Declared publication identity and versioning semantics for the docs."""

    provider: str
    base_url: str
    scope: str
    versioned: bool
    home: str
    catalog: str
    public_surfaces: str
    source_repository: str
    workflow: str
    _data: Mapping[str, Any] = field(repr=False)

    def link(
        self,
        *,
        rel: str,
        title: str,
        docname: str,
        role: str | None = None,
    ) -> DocumentationLink:
        return DocumentationLink(
            rel=rel,
            title=title,
            docname=docname,
            url=urljoin(self.base_url, f"{docname}.html"),
            role=role,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a mutable copy, including fields unknown to this API version."""

        return _thaw(self._data)


@dataclass(frozen=True, slots=True)
class CatalogSurface:
    """One public, compatibility, experimental, or internal API surface."""

    id: str
    domain: str
    family: str
    kind: str
    tier: str
    visibility: str
    canonical_import: str | None
    contract: str
    replacement: str | None
    requires: tuple[str, ...]
    roles: tuple[str, ...]
    public_symbols: tuple[str, ...]
    _data: Mapping[str, Any] = field(repr=False)

    def as_dict(self) -> dict[str, Any]:
        """Return a mutable full-fidelity copy of the manifest record."""

        return _thaw(self._data)


@dataclass(frozen=True, slots=True)
class DocumentationContext:
    """Specific, learning, help, and broad-context links for one surface."""

    site: DocumentationSite
    primary: DocumentationLink | None
    learn: tuple[DocumentationLink, ...]
    help: tuple[DocumentationLink, ...]
    context: tuple[DocumentationLink, ...]
    surface: CatalogSurface
    component: str | None = None


@dataclass(frozen=True, slots=True)
class PublicCatalog:
    """Immutable view of the public manifest shipped in one distribution."""

    interface: str
    contract_id: str
    release: str
    series: str
    families: tuple[Mapping[str, Any], ...]
    documentation_site: DocumentationSite
    _surface_records: tuple[CatalogSurface, ...] = field(repr=False)
    _surface_index: Mapping[str, CatalogSurface] = field(repr=False)
    _documentation: Mapping[str, Any] = field(repr=False)
    _data: Mapping[str, Any] = field(repr=False)

    def surface(self, surface_id: str) -> CatalogSurface:
        """Return a surface or raise ``KeyError`` for an unknown identifier."""

        return self._surface_index[surface_id]

    def iter_surfaces(self, include_internal: bool = False) -> Iterator[CatalogSurface]:
        """Iterate manifest order, excluding internal records by default."""

        for surface in self._surface_records:
            internal = (
                surface.visibility == "internal"
                or surface.tier == "internal"
                or surface.kind == "internal"
            )
            if include_internal or not internal:
                yield surface

    def documentation_for(
        self, surface_id: str, component: str | None = None
    ) -> DocumentationContext:
        """Resolve component, learning, help, and site-context documentation."""

        surface = self.surface(surface_id)
        roles = _mapping(self._documentation.get("roles"), name="documentation.roles")
        learn: list[DocumentationLink] = []
        help_links: list[DocumentationLink] = []
        for role in surface.roles:
            route = _mapping(roles[role], name=f"documentation.roles.{role}")
            role_title = str(route["title"])
            for rel, label in (("guide", "guide"), ("concept", "concept")):
                learn.append(
                    self.documentation_site.link(
                        rel=rel,
                        title=f"{role_title}: {label}",
                        docname=str(route[rel]),
                        role=role,
                    )
                )
            for rel, label in (("reference", "reference"), ("catalog", "catalog")):
                help_links.append(
                    self.documentation_site.link(
                        rel=rel,
                        title=f"{role_title}: {label}",
                        docname=str(route[rel]),
                        role=role,
                    )
                )

        primary = None
        if component is not None:
            components = _mapping(
                self._documentation.get("application_components"),
                name="documentation.application_components",
            )
            if component not in components:
                raise KeyError(component)
            record = _mapping(
                components[component],
                name=f"documentation.application_components.{component}",
            )
            if record.get("surface") != surface_id:
                raise ValueError(
                    f"component {component!r} belongs to surface "
                    f"{record.get('surface')!r}, not {surface_id!r}"
                )
            primary = self.documentation_site.link(
                rel="primary",
                title=f"{component} documentation",
                docname=str(record["documentation"]),
            )

        site = self.documentation_site
        context = (
            site.link(rel="home", title="PanorAi documentation", docname=site.home),
            site.link(
                rel="catalog", title="Public component catalog", docname=site.catalog
            ),
            site.link(
                rel="public-surfaces",
                title="Complete public surface inventory",
                docname=site.public_surfaces,
            ),
        )
        return DocumentationContext(
            site=site,
            primary=primary,
            learn=tuple(learn),
            help=tuple(help_links),
            context=context,
            surface=surface,
            component=component,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a mutable full-fidelity copy suitable for serialization."""

        result = _thaw(self._data)
        result["interface"] = self.interface
        return result


def _catalog_from_data(raw: Mapping[str, Any]) -> PublicCatalog:
    frozen = _mapping(_freeze(dict(raw)), name="catalog")
    documentation = _mapping(frozen.get("documentation"), name="documentation")
    assignments = _mapping(
        documentation.get("assignments"), name="documentation.assignments"
    )
    roles_by_surface: dict[str, list[str]] = {}
    for role, surface_ids in assignments.items():
        for surface_id in _string_tuple(surface_ids):
            roles_by_surface.setdefault(surface_id, []).append(role)

    surfaces: list[CatalogSurface] = []
    for raw_surface in frozen["surfaces"]:
        surface = _mapping(raw_surface, name="surface")
        record = CatalogSurface(
            id=str(surface["id"]),
            domain=str(surface["domain"]),
            family=str(surface["family"]),
            kind=str(surface["kind"]),
            tier=str(surface["tier"]),
            visibility=str(surface["visibility"]),
            canonical_import=(
                None
                if surface.get("canonical_import") is None
                else str(surface["canonical_import"])
            ),
            contract=str(surface["contract"]),
            replacement=(
                None
                if surface.get("replacement") is None
                else str(surface["replacement"])
            ),
            requires=_string_tuple(surface.get("requires")),
            roles=tuple(roles_by_surface.get(str(surface["id"]), ())),
            public_symbols=public_symbols_for(surface, frozen),
            _data=surface,
        )
        surfaces.append(record)

    site_data = _mapping(documentation.get("site"), name="documentation.site")
    site = DocumentationSite(
        provider=str(site_data["provider"]),
        base_url=str(site_data["base_url"]),
        scope=str(site_data["scope"]),
        versioned=bool(site_data["versioned"]),
        home=str(site_data["home"]),
        catalog=str(site_data["catalog"]),
        public_surfaces=str(site_data["public_surfaces"]),
        source_repository=str(site_data["source_repository"]),
        workflow=str(site_data["workflow"]),
        _data=site_data,
    )
    surface_records = tuple(surfaces)
    return PublicCatalog(
        interface=CATALOG_INTERFACE,
        contract_id=str(frozen["contract_id"]),
        release=str(frozen["release"]),
        series=str(frozen["series"]),
        families=tuple(frozen["families"]),
        documentation_site=site,
        _surface_records=surface_records,
        _surface_index=MappingProxyType(
            {surface.id: surface for surface in surface_records}
        ),
        _documentation=documentation,
        _data=frozen,
    )


@lru_cache(maxsize=1)
def public_catalog() -> PublicCatalog:
    """Load and cache the catalog embedded in the installed distribution."""

    resource = resources.files(__package__).joinpath(_MANIFEST_RESOURCE)
    raw = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("installed PanorAi public catalog must be a JSON object")
    return _catalog_from_data(raw)
