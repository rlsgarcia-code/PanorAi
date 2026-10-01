"""PanorAi: spherical image processing and projection.

The canonical :mod:`panorai.geometry` package is intentionally import-light.
The five root-level 3.0 compatibility classes remain available, but their
legacy dependency trees are loaded only when the corresponding name is used.
"""

from __future__ import annotations

from importlib import import_module


try:
    # ``setuptools-scm`` writes this file for source and wheel builds. Reading
    # it first avoids importing the much larger metadata stack on every core
    # import while retaining the exact generated distribution version.
    from ._version import __version__
except ImportError:  # pragma: no cover - fallback for incomplete source trees
    try:
        from importlib.metadata import PackageNotFoundError, version

        __version__ = version("panorai")
    except PackageNotFoundError:
        __version__ = "0+unknown"


__all__ = [
    "EquirectangularImage",
    "GnomonicFace",
    "GnomonicFaceSet",
    "ConfigManager",
    "PanoraiFactory",
    "SphericalFeaturePipeline",
    "__version__",
]


_LAZY_EXPORTS = {
    "EquirectangularImage": (".data", "EquirectangularImage"),
    "GnomonicFace": (".data", "GnomonicFace"),
    "GnomonicFaceSet": (".data", "GnomonicFaceSet"),
    "ConfigManager": (".config.config_manager", "ConfigManager"),
    "PanoraiFactory": (".factory.panorai_factory", "PanoraiFactory"),
    "SphericalFeaturePipeline": (".features", "SphericalFeaturePipeline"),
}


def __getattr__(name: str):
    """Resolve recorded 3.0 root exports without eager compatibility imports."""

    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy public exports in interactive discovery."""

    return sorted(set(globals()) | set(__all__))
