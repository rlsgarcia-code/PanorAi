# panorai/projections/registry.py

from __future__ import annotations

from importlib import import_module


class ProjectionRegistry:
    _projections = {}

    @classmethod
    def _ensure_builtin_projections(cls) -> None:
        """Restore built-ins after a direct import or an isolated module reload.

        Importing ``panorai.projections.registry`` directly does not execute the
        package facade that normally imports the gnomonic implementation.  A
        test harness or plugin loader can also reload this registry while the
        implementation module remains cached.  In both cases the public
        factory must still expose PanorAi's built-in projection.
        """

        if "gnomonic" in cls._projections:
            return
        module = import_module("panorai.projections.gnomonic_projection")
        projection = getattr(module, "GnomonicProjection")
        cls._projections.setdefault("gnomonic", projection)

    @classmethod
    def register(cls, name: str):
        def decorator(projection_cls):
            if name in cls._projections:
                raise KeyError(f"Projection '{name}' already registered.")
            cls._projections[name] = projection_cls
            return projection_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs):
        cls._ensure_builtin_projections()
        if name not in cls._projections:
            raise KeyError(
                f"Projection '{name}' not found. Available: {list(cls._projections.keys())}"
            )
        return cls._projections[name](**kwargs)

    @classmethod
    def available_projections(cls) -> list:
        cls._ensure_builtin_projections()
        return list(cls._projections.keys())
