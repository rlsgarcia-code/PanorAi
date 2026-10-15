"""Immutable execution records for the Experimental pipeline surface."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    module = type(value).__module__
    if module == "numpy" or module.startswith("numpy."):
        if getattr(value, "ndim", None) == 0:
            return value.item()
    return value


@dataclass(frozen=True, slots=True)
class StageRecord:
    """JSON-friendly provenance for one completed pipeline stage."""

    name: str
    stage: str
    input_type: str
    output_type: str
    details: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stage": self.stage,
            "input_type": self.input_type,
            "output_type": self.output_type,
            "details": _json_value(self.details),
        }


class PipelineRun(Mapping[str, Any]):
    """One pipeline output plus explicitly retained intermediate values.

    Use :meth:`Pipeline.run` when execution provenance or intermediates are
    needed. Calling a pipeline directly returns only :attr:`output`.
    """

    interface = "panorai-spherical-pipeline/v1-experimental"
    stability = "experimental"

    def __init__(
        self,
        output: Any,
        *,
        retained: Mapping[str, Any],
        records: tuple[StageRecord, ...],
        execution: Mapping[str, Any] | None = None,
    ) -> None:
        self.output = output
        self._retained = MappingProxyType(dict(retained))
        self._records = tuple(records)
        self._execution = MappingProxyType(dict(execution or {}))

    @property
    def records(self) -> tuple[StageRecord, ...]:
        """Return immutable per-stage execution records."""

        return self._records

    def __getitem__(self, name: str) -> Any:
        return self._retained[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._retained)

    def __len__(self) -> int:
        return len(self._retained)

    def describe(self) -> dict[str, Any]:
        """Return a JSON-friendly description without embedding array data."""

        return {
            "interface": self.interface,
            "stability": self.stability,
            "retained": list(self._retained),
            "steps": [record.to_dict() for record in self._records],
            "output_type": type(self.output).__name__,
            "cache": _json_value(self._execution.get("cache", {})),
        }

    def manifest(self) -> dict[str, Any]:
        """Return the JSON-friendly execution identity and completed records."""

        return {
            "interface": self.interface,
            "stability": self.stability,
            **_json_value(self._execution),
            "steps": [record.to_dict() for record in self._records],
            "output_type": type(self.output).__name__,
        }


def _result_grid(value: Any) -> Any | None:
    grid = getattr(value, "grid", None)
    if grid is not None:
        return grid
    if callable(getattr(value, "_require_workflow", None)):
        try:
            from ._fields import field_from_container

            metadata = (
                value[0]._workflow_metadata
                if hasattr(value, "_faces")
                else value._require_workflow()
            )
            if metadata:
                return field_from_container(value, next(iter(metadata))).grid
        except (AttributeError, KeyError, TypeError, ValueError):
            return None
    return None


@dataclass(frozen=True, slots=True)
class StageResult:
    """Value and immutable execution metadata returned by a public stage.

    ``grid`` is the spatial identity of the returned value when one exists.
    Non-spatial results use ``None``.  Stages may provide it explicitly; the
    constructor otherwise derives it from public raster fields and workflow
    containers.
    """

    value: Any
    details: Mapping[str, Any]
    grid: Any | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.details, Mapping):
            raise TypeError("StageResult details must be a mapping")
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))
        if self.grid is None:
            object.__setattr__(self, "grid", _result_grid(self.value))


# Private name retained only so an in-memory object created by the earlier
# Experimental prototype remains understandable during migration.
_StageOutcome = StageResult


__all__ = ["PipelineRun", "StageRecord", "StageResult"]
