"""Typed, deterministic graph lifecycle events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .observations import deterministic_id


@dataclass(frozen=True, slots=True)
class GraphEvent:
    event_id: str
    event_type: str
    subject_ids: tuple[str, ...]
    payload: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        event_type: str,
        subject_ids: tuple[str, ...],
        payload: Mapping[str, Any] | None = None,
    ) -> "GraphEvent":
        normalized_subjects = tuple(sorted(subject_ids))
        normalized_payload = dict(payload or {})
        return cls(
            event_id=deterministic_id(
                "event", event_type, normalized_subjects, normalized_payload
            ),
            event_type=event_type,
            subject_ids=normalized_subjects,
            payload=normalized_payload,
        )


__all__ = ["GraphEvent"]
