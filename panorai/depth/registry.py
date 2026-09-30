"""Dependency-free registry for optional depth-model adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar


class ModelRegistry:
    """Discover and invoke optional depth adapters without importing backends."""

    _registry: ClassVar[dict[str, dict[str, Any]]] = {}

    @classmethod
    def register(
        cls, name: str, default_args: dict[str, Any] | None = None
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register one loader and a defensive copy of its default arguments."""

        def decorator(loader_func: Callable[..., Any]) -> Callable[..., Any]:
            cls._registry[name] = {
                "loader_func": loader_func,
                "default_args": dict(default_args or {}),
            }
            return loader_func

        return decorator

    @classmethod
    def load(cls, name: str, **overrides: Any) -> Any:
        """Load a model through its adapter using defaults plus overrides."""

        loader = cls.get_loader(name)
        config = {**cls.get_config(name), **overrides}
        return loader(**config)

    @classmethod
    def list_models(cls) -> list[str]:
        """Return registered adapter keys in deterministic order."""

        return sorted(cls._registry)

    @classmethod
    def get_loader(cls, name: str) -> Callable[..., Any]:
        """Return the registered callable without invoking optional imports."""

        if name not in cls._registry:
            raise ValueError(f"Model {name!r} is not registered.")
        return cls._registry[name]["loader_func"]

    @classmethod
    def get_config(cls, name: str) -> dict[str, Any]:
        """Return a copy of the adapter defaults."""

        if name not in cls._registry:
            raise ValueError(f"Model {name!r} is not registered.")
        return dict(cls._registry[name]["default_args"])
