"""Logical branching and typed-field routing for spherical pipelines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ._fields import (
    RasterField,
    ResultBundle,
    attach_raster_field,
    field_from_container,
)
from ._results import StageResult


def _valid_branch(value: Any) -> bool:
    return (
        value is None
        or callable(getattr(value, "run", None))
        or callable(getattr(value, "execute", None))
    )


class Parallel:
    """Run independent logical branches sequentially on the same input.

    The name describes topology, not concurrency.  ``None`` is an explicit
    identity branch and is useful as an attachment target.
    """

    input_kind = "any"
    output_kind = "bundle"

    def __init__(self, **branches: Any) -> None:
        if not branches:
            raise ValueError("Parallel requires at least one named branch")
        for name, branch in branches.items():
            if not isinstance(name, str) or not name.strip() or "." in name:
                raise ValueError(
                    "Parallel branch names must be non-empty strings without '.'"
                )
            if not _valid_branch(branch):
                raise TypeError(
                    "Parallel branches must be Pipeline/stage objects or None "
                    "for an identity branch"
                )
        self.branches = MappingProxyType(dict(branches))

    def execute(self, value: Any) -> StageResult:
        outputs: dict[str, Any] = {}
        descriptions: dict[str, Any] = {}
        for name, branch in self.branches.items():
            if branch is None:
                outputs[name] = value
                descriptions[name] = {"kind": "identity"}
            elif callable(getattr(branch, "execute", None)):
                outcome = branch.execute(value)
                outputs[name] = outcome.value
                descriptions[name] = {
                    "kind": "stage",
                    "stage": type(branch).__name__,
                    "details": dict(outcome.details),
                }
            else:
                run = branch.run(value)
                outputs[name] = run.output
                descriptions[name] = run.describe()
        bundle = ResultBundle(outputs, descriptions=descriptions)
        return StageResult(
            bundle,
            {
                "domain": "composition",
                "execution": "sequential-logical-branches",
                "branches": list(self.branches),
            },
        )

    def __repr__(self) -> str:
        content = ", ".join(
            f"{name}={branch!r}" for name, branch in self.branches.items()
        )
        return f"Parallel({content})"


def _select_path(value: Any, path: str) -> Any:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Select path must be a non-empty string")
    head, separator, tail = path.partition(".")
    if isinstance(value, ResultBundle) or isinstance(value, Mapping):
        try:
            selected = value[head]
        except KeyError as exc:
            raise KeyError(f"unknown result {head!r}") from exc
        return _select_path(selected, tail) if separator else selected
    if separator:
        raise KeyError(
            f"cannot continue selection {path!r} through {type(value).__name__}"
        )
    return field_from_container(value, head)


@dataclass(frozen=True, slots=True)
class Select:
    """Select one branch or typed raster field by a dotted path."""

    path: str

    input_kind = "any"
    output_kind = "any"

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError("Select path must be a non-empty string")

    def execute(self, value: Any) -> StageResult:
        selected = _select_path(value, self.path)
        return StageResult(
            selected,
            {
                "domain": "composition",
                "selection": self.path,
                "selected_type": type(selected).__name__,
            },
        )


def _as_field(value: Any, *, attached_name: str) -> RasterField:
    if isinstance(value, RasterField):
        return value
    try:
        return field_from_container(value, attached_name)
    except KeyError:
        if hasattr(value, "_faces"):
            metadata = value[0]._workflow_metadata
        elif callable(getattr(value, "_require_workflow", None)):
            metadata = value._require_workflow()
        else:
            raise TypeError(
                "Attach sources must be RasterField or workflow raster containers"
            ) from None
        if len(metadata) != 1:
            raise KeyError(
                f"source has no field {attached_name!r}; select one explicitly"
            ) from None
        return field_from_container(value, next(iter(metadata)))


class Attach:
    """Attach compatible typed branch fields to one target branch.

    Example::

        Attach(target="base", depth="prediction.depth")
    """

    input_kind = "bundle"
    output_kind = "any"

    def __init__(
        self,
        *,
        target: str,
        replace: bool = False,
        **fields: str,
    ) -> None:
        if not isinstance(target, str) or not target.strip():
            raise ValueError("Attach target must be a non-empty selection path")
        if not fields:
            raise ValueError("Attach requires at least one field mapping")
        if not isinstance(replace, bool):
            raise TypeError("replace must be boolean")
        for name, selector in fields.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("attached field names must be non-empty strings")
            if not isinstance(selector, str) or not selector.strip():
                raise ValueError("Attach selectors must be non-empty strings")
        self.target = target
        self.replace = replace
        self.fields = MappingProxyType(dict(fields))

    def execute(self, value: Any) -> StageResult:
        if not isinstance(value, ResultBundle):
            raise TypeError("Attach requires a ResultBundle input")
        target = _select_path(value, self.target)
        attached = target
        descriptions = []
        for name, selector in self.fields.items():
            source = _select_path(value, selector)
            field = _as_field(source, attached_name=name)
            attached = attach_raster_field(attached, name, field, replace=self.replace)
            descriptions.append(
                {
                    "name": name,
                    "source": selector,
                    "kind": field.kind,
                    "grid": field.grid.describe(),
                }
            )
        return StageResult(
            attached,
            {
                "domain": "composition",
                "target": self.target,
                "attached": descriptions,
            },
        )

    def __repr__(self) -> str:
        fields = ", ".join(f"{name}={path!r}" for name, path in self.fields.items())
        return f"Attach(target={self.target!r}, replace={self.replace!r}, {fields})"


__all__ = ["Attach", "Parallel", "Select"]
