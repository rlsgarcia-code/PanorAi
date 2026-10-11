"""Provider protocols for explicit graph inputs.

Providers are intentionally outside :class:`SpatialSemanticGraphBuilder`.
Calling code chooses and invokes a provider, then hands its result to the
builder.  This keeps graph construction reproducible and prevents hidden
model/checkpoint selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

from ._models import EvidenceMode, EvidenceSource

InputT = TypeVar("InputT", contravariant=True)
OutputT = TypeVar("OutputT", covariant=True)


@runtime_checkable
class EvidenceProvider(Protocol[InputT, OutputT]):
    source: EvidenceSource

    def provide(self, value: InputT) -> OutputT:
        """Produce one explicit evidence result."""


@dataclass(frozen=True, slots=True)
class ExternalProvider(Generic[OutputT]):
    source: EvidenceSource
    result: OutputT

    def __post_init__(self) -> None:
        if self.source.mode is not EvidenceMode.EXTERNAL:
            raise ValueError("ExternalProvider requires mode='external'")

    def provide(self, value: object = None) -> OutputT:
        return self.result


@dataclass(frozen=True, slots=True)
class OracleProvider(Generic[OutputT]):
    source: EvidenceSource
    result: OutputT

    def __post_init__(self) -> None:
        if self.source.mode is not EvidenceMode.ORACLE:
            raise ValueError("OracleProvider requires mode='oracle'")

    def provide(self, value: object = None) -> OutputT:
        return self.result


@dataclass(frozen=True, slots=True)
class DisabledProvider:
    source: EvidenceSource

    def __post_init__(self) -> None:
        if self.source.mode is not EvidenceMode.DISABLED:
            raise ValueError("DisabledProvider requires mode='disabled'")

    def provide(self, value: object = None) -> None:
        return None


@dataclass(frozen=True, slots=True)
class CallablePanorAiProvider(Generic[InputT, OutputT]):
    """Explicit adapter around a caller-supplied public PanorAi callable."""

    source: EvidenceSource
    function: object

    def provide(self, value: InputT) -> OutputT:
        if not callable(self.function):
            raise TypeError("function must be callable")
        return self.function(value)  # type: ignore[misc, no-any-return]


PoseProvider = EvidenceProvider[object, object]
MultiViewPoseProvider = EvidenceProvider[object, object]
DepthProvider = EvidenceProvider[object, object]
SemanticProvider = EvidenceProvider[object, object]


__all__ = [
    "CallablePanorAiProvider",
    "DepthProvider",
    "DisabledProvider",
    "EvidenceProvider",
    "ExternalProvider",
    "MultiViewPoseProvider",
    "OracleProvider",
    "PoseProvider",
    "SemanticProvider",
]
