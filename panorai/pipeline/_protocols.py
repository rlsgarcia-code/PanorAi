"""Public extension protocol for spherical pipeline stages."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from ._results import StageResult


StageKind = str | tuple[str, ...]


@runtime_checkable
class Stage(Protocol):
    """A composable pipeline stage implemented through a public method.

    Third-party stages only need to declare their input/output kinds and
    implement :meth:`execute`.  They do not inherit a PanorAi base class and
    never depend on a private executor hook.
    """

    input_kind: StageKind
    output_kind: StageKind

    def execute(self, value: Any) -> StageResult:
        """Process ``value`` and return a typed execution result."""


def normalize_stage_result(result: Any, *, stage: Any) -> StageResult:
    """Validate a public stage result at the executor boundary."""

    if not isinstance(result, StageResult):
        raise TypeError(
            f"stage {type(stage).__name__}.execute() must return StageResult; "
            f"got {type(result).__name__}"
        )
    return result


def stage_details(**values: Any) -> Mapping[str, Any]:
    """Return immutable stage details for small external stages."""

    return MappingProxyType(dict(values))


__all__ = ["Stage", "StageKind", "normalize_stage_result", "stage_details"]
