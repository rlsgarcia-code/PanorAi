"""An inspectable linear composition object for spherical processing."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import fields, is_dataclass, replace
import re
from types import MappingProxyType
from typing import Any

from ._protocols import normalize_stage_result
from ._results import PipelineRun, StageRecord, StageResult


def _stage_kinds(value: Any, *, attribute: str) -> tuple[str, ...]:
    declared = getattr(value, attribute, None)
    if isinstance(declared, str):
        result = (declared,)
    elif (
        isinstance(declared, tuple)
        and declared
        and all(isinstance(item, str) and item for item in declared)
    ):
        result = declared
    else:
        raise TypeError(
            f"stage {type(value).__name__} must declare {attribute} as a "
            "non-empty string or tuple of strings"
        )
    return result


def _validate_stage(stage: Any) -> None:
    if not callable(getattr(stage, "execute", None)):
        raise TypeError(
            "pipeline stages must implement public execute(value) -> StageResult"
        )
    _stage_kinds(stage, attribute="input_kind")
    _stage_kinds(stage, attribute="output_kind")


def _compatible(left: Any, right: Any) -> bool:
    outputs = set(_stage_kinds(left, attribute="output_kind"))
    inputs = set(_stage_kinds(right, attribute="input_kind"))
    return "any" in outputs or "any" in inputs or not outputs.isdisjoint(inputs)


def _stage_parameters(stage: Any) -> dict[str, Any]:
    if is_dataclass(stage):
        return {item.name: getattr(stage, item.name) for item in fields(stage)}
    from ._routing import Attach, Parallel

    if isinstance(stage, Parallel):
        return {"branches": stage.branches}
    if isinstance(stage, Attach):
        return {
            "target": stage.target,
            "replace": stage.replace,
            "fields": stage.fields,
        }
    return {}


def _replace_stage(stage: Any, updates: Mapping[str, Any]) -> Any:
    if not updates:
        return stage
    from ._stages import ReconstructERP

    if isinstance(stage, ReconstructERP):
        unknown = set(updates) - {"blend", "modalities", "eq_shape", "field_blends"}
        if unknown:
            raise KeyError(f"unknown ReconstructERP parameters: {sorted(unknown)}")
        if "field_blends" in updates:
            direct = dict(updates["field_blends"])
            return ReconstructERP(
                blend=None,
                modalities="all",
                eq_shape=updates.get("eq_shape", stage.eq_shape),
                **direct,
            )
        return ReconstructERP(
            blend=updates.get("blend", stage.blend),
            modalities=updates.get("modalities", stage.modalities),
            eq_shape=updates.get("eq_shape", stage.eq_shape),
        )
    if is_dataclass(stage):
        return replace(stage, **updates)
    from ._routing import Attach, Parallel

    if isinstance(stage, Parallel):
        unknown = set(updates) - {"branches"}
        if unknown:
            raise KeyError(f"unknown Parallel parameters: {sorted(unknown)}")
        return Parallel(**dict(updates.get("branches", stage.branches)))
    if isinstance(stage, Attach):
        unknown = set(updates) - {"target", "replace", "fields"}
        if unknown:
            raise KeyError(f"unknown Attach parameters: {sorted(unknown)}")
        return Attach(
            target=updates.get("target", stage.target),
            replace=updates.get("replace", stage.replace),
            **dict(updates.get("fields", stage.fields)),
        )
    raise TypeError(f"stage {type(stage).__name__} has no configurable parameters")


def _auto_name(stage: Any) -> str:
    preferred = getattr(stage, "stage_name", None)
    if preferred is not None:
        return _validate_name(preferred)
    raw = type(stage).__name__
    first = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", raw)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first).lower()


def _validate_name(name: Any) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("pipeline step names must be non-empty strings")
    if "__" in name:
        raise ValueError("pipeline step names cannot contain '__'")
    return name


class Pipeline:
    """Compose sphere-native and projected-view stages linearly.

    This Stable layer delegates all projections, per-view mapping and
    reconstruction to the existing Stable object workflow.
    """

    interface = "panorai-spherical-pipeline/v1"
    stability = "stable"

    def __init__(self, *steps: Any, memory: Any = None) -> None:
        if not steps:
            raise ValueError("Pipeline requires at least one stage")
        resolved: list[tuple[str, Any]] = []
        counts: dict[str, int] = {}
        for item in steps:
            if isinstance(item, tuple):
                if len(item) != 2:
                    raise ValueError("named steps must be (name, stage) pairs")
                name, stage = item
                name = _validate_name(name)
            else:
                stage = item
                base = _auto_name(stage)
                counts[base] = counts.get(base, 0) + 1
                name = base if counts[base] == 1 else f"{base}_{counts[base]}"
            _validate_stage(stage)
            if any(existing == name for existing, _ in resolved):
                raise ValueError(f"duplicate pipeline step name {name!r}")
            resolved.append((name, stage))
        self._steps = tuple(resolved)
        self.memory = memory
        self._validate_transitions()

    def _validate_transitions(self) -> None:
        for (left_name, left), (right_name, right) in zip(
            self._steps, self._steps[1:], strict=False
        ):
            if not _compatible(left, right):
                raise ValueError(
                    f"incompatible pipeline transition {left_name!r} "
                    f"({left.output_kind}) -> {right_name!r} ({right.input_kind})"
                )

    @property
    def input_kind(self) -> Any:
        """Return the declared input kind of the first stage."""

        return self._steps[0][1].input_kind

    @property
    def output_kind(self) -> Any:
        """Return the declared output kind of the last stage."""

        return self._steps[-1][1].output_kind

    @property
    def named_steps(self) -> Mapping[str, Any]:
        """Return the immutable ordered stage mapping."""

        return MappingProxyType(dict(self._steps))

    def __call__(self, value: Any) -> Any:
        return self.run(value).output

    def execute(self, value: Any) -> StageResult:
        """Execute this pipeline as one nestable public :class:`Stage`."""

        run = self.run(value)
        return StageResult(
            run.output,
            {
                "domain": "composition",
                "pipeline": run.describe(),
                "manifest": run.manifest(),
            },
        )

    def run(self, value: Any, *, retain: Iterable[str] | str = ()) -> PipelineRun:
        """Execute and return provenance plus selected intermediate values."""

        requested = {retain} if isinstance(retain, str) else set(retain)
        available = {name for name, _ in self._steps}
        unknown = {
            item for item in requested if item.partition(".")[0] not in available
        }
        if unknown:
            raise KeyError(f"unknown retained pipeline steps: {sorted(unknown)}")
        retained: dict[str, Any] = {}
        records: list[StageRecord] = []
        current = value
        cache_identity = None
        if self.memory is not None and not requested:
            from ._cache import _CACHE_MISS
            from ._cache import cache_identity as resolve_cache_identity
            from ._cache import load_cached_result

            cache_identity = resolve_cache_identity(self, value)
            cached = load_cached_result(self.memory, cache_identity, value)
            if cached is not _CACHE_MISS:
                cache_record = StageRecord(
                    name="cache",
                    stage="ContentAddressedCache",
                    input_type=type(value).__name__,
                    output_type=type(cached).__name__,
                    details=MappingProxyType(
                        {"hit": True, "key": cache_identity["key"]}
                    ),
                )
                return PipelineRun(
                    cached,
                    retained={},
                    records=(cache_record,),
                    execution={
                        "pipeline_interface": self.interface,
                        "cache": {"enabled": True, "hit": True, **cache_identity},
                    },
                )
        for name, stage in self._steps:
            input_type = type(current).__name__
            outcome = normalize_stage_result(stage.execute(current), stage=stage)
            current = outcome.value
            if name in requested:
                retained[name] = current
            nested = sorted(item for item in requested if item.startswith(f"{name}."))
            if nested:
                from ._routing import Select

                for path in nested:
                    retained[path] = (
                        Select(path.split(".", 1)[1]).execute(current).value
                    )
            records.append(
                StageRecord(
                    name=name,
                    stage=type(stage).__name__,
                    input_type=input_type,
                    output_type=type(current).__name__,
                    details=MappingProxyType(dict(outcome.details)),
                )
            )
        cache = {
            "enabled": self.memory is not None,
            "hit": False,
            "bypassed": bool(self.memory is not None and requested),
        }
        if cache_identity is not None:
            from ._cache import store_cached_result

            store_cached_result(self.memory, cache_identity, current, value)
            cache.update(cache_identity)
        return PipelineRun(
            current,
            retained=retained,
            records=tuple(records),
            execution={"pipeline_interface": self.interface, "cache": cache},
        )

    def get_params(self, *, deep: bool = True) -> dict[str, Any]:
        """Return stage objects and sklearn-style ``step__parameter`` values."""

        result: dict[str, Any] = {}
        for name, stage in self._steps:
            result[name] = stage
            if deep:
                for parameter, value in _stage_parameters(stage).items():
                    result[f"{name}__{parameter}"] = value
        return result

    def with_params(self, **params: Any) -> "Pipeline":
        """Return a new pipeline with selected stage parameters replaced."""

        updates: dict[str, dict[str, Any]] = {}
        stages = dict(self._steps)
        for key, value in params.items():
            if "__" not in key:
                raise ValueError("parameters must use the 'step__parameter' form")
            name, parameter = key.split("__", 1)
            if name not in stages:
                raise KeyError(f"unknown pipeline step {name!r}")
            if not parameter:
                raise ValueError("stage parameter name cannot be empty")
            valid = set(_stage_parameters(stages[name]))
            if parameter not in valid:
                raise KeyError(f"unknown parameter {parameter!r} for step {name!r}")
            updates.setdefault(name, {})[parameter] = value
        rebuilt = [
            (name, _replace_stage(stage, updates.get(name, {})))
            for name, stage in self._steps
        ]
        return Pipeline(*rebuilt, memory=self.memory)

    def describe(self) -> dict[str, Any]:
        """Return the static, JSON-friendly pipeline topology."""

        return {
            "interface": self.interface,
            "stability": self.stability,
            "steps": [
                {
                    "name": name,
                    "stage": type(stage).__name__,
                    "input_kind": stage.input_kind,
                    "output_kind": stage.output_kind,
                    "parameters": list(_stage_parameters(stage)),
                }
                for name, stage in self._steps
            ],
            "memory": None if self.memory is None else str(self.memory),
        }

    def to_config(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible registered-component config.

        Arbitrary callables remain executable in memory but fail here with the
        exact parameter path. No pickle or dynamic import is used.
        """

        from ._serialization import pipeline_to_config

        return pipeline_to_config(self)

    @classmethod
    def from_config(cls, config: Any) -> "Pipeline":
        """Construct a pipeline from a validated registered-component config."""

        from ._serialization import pipeline_from_config

        result = pipeline_from_config(config)
        if not isinstance(result, cls):  # pragma: no cover - defensive boundary
            raise TypeError("decoded pipeline has an incompatible implementation")
        return result

    def save(self, directory: Any, *, overwrite: bool = False) -> Any:
        """Write a checksum-verified JSON archive for registered components."""

        from ._serialization import save_pipeline

        return save_pipeline(self, directory, overwrite=overwrite)

    @classmethod
    def load(cls, directory: Any) -> "Pipeline":
        """Load and verify a registered-component pipeline archive."""

        from ._serialization import load_pipeline

        result = load_pipeline(directory)
        if not isinstance(result, cls):  # pragma: no cover - defensive boundary
            raise TypeError("archive pipeline has an incompatible implementation")
        return result

    def __repr__(self) -> str:
        contents = ", ".join(f"({name!r}, {stage!r})" for name, stage in self._steps)
        suffix = "" if self.memory is None else f", memory={self.memory!r}"
        return f"Pipeline({contents}{suffix})"
