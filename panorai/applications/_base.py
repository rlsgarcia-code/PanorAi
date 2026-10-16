"""Common immutable contracts for high-level spherical applications."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import fields, is_dataclass, replace
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


APPLICATION_CONFIG_FORMAT = "panorai-application-config/v1-experimental"


class ApplicationConfigurationError(ValueError):
    """An application configuration is unsafe, unknown, or incomplete."""


def _json_safe(value: Any, *, path: str) -> Any:
    if is_dataclass(value):
        return {
            item.name: _json_safe(getattr(value, item.name), path=f"{path}.{item.name}")
            for item in fields(value)
        }
    if isinstance(value, tuple):
        return [_json_safe(item, path=f"{path}[]") for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ApplicationConfigurationError(f"{path}: keys must be strings")
        return {
            key: _json_safe(item, path=f"{path}.{key}") for key, item in value.items()
        }
    if isinstance(value, float) and not math.isfinite(value):
        raise ApplicationConfigurationError(f"{path}: floats must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ApplicationConfigurationError(
        f"{path}: unregistered {type(value).__module__}.{type(value).__qualname__}"
    )


class Application(ABC):
    """Public high-level application contract with immutable configuration."""

    interface = "panorai-application/v1-experimental"
    stability = "experimental"
    application_id: str

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the application."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.run(*args, **kwargs)

    def get_params(self, *, deep: bool = True) -> dict[str, Any]:
        if not is_dataclass(self):
            return {}
        result = {
            item.name: getattr(self, item.name) for item in fields(self) if item.init
        }
        if deep:
            for name, value in tuple(result.items()):
                if is_dataclass(value):
                    for item in fields(value):
                        if item.init:
                            result[f"{name}__{item.name}"] = getattr(value, item.name)
        return result

    def with_params(self, **params: Any) -> "Application":
        if not is_dataclass(self):
            raise TypeError(
                f"{type(self).__name__} does not expose dataclass parameters"
            )
        direct: dict[str, Any] = {}
        nested: dict[str, dict[str, Any]] = {}
        valid = {item.name for item in fields(self) if item.init}
        for name, value in params.items():
            parent, separator, child = name.partition("__")
            if parent not in valid:
                raise KeyError(f"unknown parameter {name!r}")
            if separator:
                if not child:
                    raise KeyError(f"invalid nested parameter {name!r}")
                nested.setdefault(parent, {})[child] = value
            else:
                direct[parent] = value
        for parent, updates in nested.items():
            current = getattr(self, parent)
            if not is_dataclass(current):
                raise KeyError(f"parameter {parent!r} has no nested parameters")
            direct[parent] = replace(current, **updates)
        return replace(self, **direct)

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Return a JSON-friendly public contract description."""

    @abstractmethod
    def to_config(self) -> dict[str, Any]:
        """Return a safe registered application configuration."""

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            self.to_config(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(payload + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, destination)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "Application":
        source = Path(path)
        if source.stat().st_size > 10 * 1024 * 1024:
            raise ApplicationConfigurationError("application config exceeds 10 MiB")
        try:
            config = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ApplicationConfigurationError(
                "application config is unreadable"
            ) from exc
        if not isinstance(config, dict):
            raise ApplicationConfigurationError("unknown application config format")
        from ._features import SphericalDoGFeatures
        from ._slic import SphericalSLIC
        from ._stereo import SphericalStereoDepth

        registry = {
            item.application_id: item
            for item in (SphericalSLIC, SphericalDoGFeatures, SphericalStereoDepth)
        }
        application_type = registry.get(config.get("application"))
        if application_type is None:
            raise ApplicationConfigurationError(
                f"unregistered application {config.get('application')!r}"
            )
        expected_format = getattr(
            application_type, "config_format", APPLICATION_CONFIG_FORMAT
        )
        accepted_formats = {expected_format}
        legacy_formats = getattr(application_type, "legacy_config_formats", ())
        accepted_formats.update(legacy_formats)
        if config.get("format") not in accepted_formats:
            raise ApplicationConfigurationError("unknown application config format")
        return application_type.from_config(config)


class PipelineApplication(Application):
    """Application whose public behavior is expressible as a Pipeline."""

    @abstractmethod
    def as_pipeline(self) -> Any:
        """Build the equivalent low-level pipeline."""

    @property
    def pipeline(self) -> Any:
        return self.as_pipeline()


def application_config(
    application_id: str,
    parameters: Any,
    *,
    format_id: str = APPLICATION_CONFIG_FORMAT,
) -> dict[str, Any]:
    return {
        "format": format_id,
        "application": application_id,
        "parameters": _json_safe(parameters, path="parameters"),
    }


__all__ = [
    "APPLICATION_CONFIG_FORMAT",
    "Application",
    "ApplicationConfigurationError",
    "PipelineApplication",
]
