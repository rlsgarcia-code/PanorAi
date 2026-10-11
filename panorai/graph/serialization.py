"""Portable JSONL/NPZ persistence for graph snapshots."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from hashlib import sha256
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from ._events import GraphEvent
from ._models import (
    EvidenceSource,
    ObjectHypothesisNode,
    PairFeatureEvidence,
    RegionCorrespondenceCandidate,
    RegionCorrespondenceEdge,
    RelativePoseEdge,
    SemanticRegionNode,
    SpatialSemanticGraph,
    SpatialSemanticPosterior,
    SphericalViewNode,
)

_TYPE_REGISTRY = {
    item.__name__: item
    for item in (
        EvidenceSource,
        GraphEvent,
        ObjectHypothesisNode,
        PairFeatureEvidence,
        RegionCorrespondenceCandidate,
        RegionCorrespondenceEdge,
        RelativePoseEdge,
        SemanticRegionNode,
        SpatialSemanticGraph,
        SpatialSemanticPosterior,
        SphericalViewNode,
    )
}
_ENUM_REGISTRY = {
    "EvidenceMode": __import__(
        "panorai.graph._models", fromlist=["EvidenceMode"]
    ).EvidenceMode
}


class GraphArchiveError(ValueError):
    """Raised when a graph archive is incomplete, incompatible, or corrupt."""


class _ArchiveCodec:
    def __init__(self, arrays_dir: Path, *, write: bool) -> None:
        self.arrays_dir = arrays_dir
        self.write = write

    def encode(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            buffer = io.BytesIO()
            np.savez_compressed(buffer, value=value)
            payload = buffer.getvalue()
            digest = sha256(payload).hexdigest()
            if self.write:
                self.arrays_dir.mkdir(parents=True, exist_ok=True)
                target = self.arrays_dir / f"{digest}.npz"
                if not target.exists():
                    target.write_bytes(payload)
            return {
                "__array__": digest,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            }
        if isinstance(value, Enum):
            return {"__enum__": type(value).__name__, "value": value.value}
        if is_dataclass(value):
            return {
                "__dataclass__": type(value).__name__,
                "fields": {
                    item.name: self.encode(getattr(value, item.name))
                    for item in fields(value)
                },
            }
        if isinstance(value, tuple):
            return {"__tuple__": [self.encode(item) for item in value]}
        if isinstance(value, list):
            return [self.encode(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self.encode(item) for key, item in value.items()}
        if isinstance(value, np.generic):
            return value.item()
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(f"unsupported graph archive value: {type(value).__name__}")

    def decode(self, value: Any) -> Any:
        if isinstance(value, list):
            return [self.decode(item) for item in value]
        if not isinstance(value, dict):
            return value
        if "__array__" in value:
            digest = value["__array__"]
            target = self.arrays_dir / f"{digest}.npz"
            if not target.is_file():
                raise GraphArchiveError(f"missing array sidecar: {digest}.npz")
            payload = target.read_bytes()
            if sha256(payload).hexdigest() != digest:
                raise GraphArchiveError(f"array checksum mismatch: {digest}.npz")
            with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                result = np.array(archive["value"], copy=True)
            if (
                list(result.shape) != value["shape"]
                or str(result.dtype) != value["dtype"]
            ):
                raise GraphArchiveError(f"array metadata mismatch: {digest}.npz")
            return result
        if "__enum__" in value:
            enum_type = _ENUM_REGISTRY.get(value["__enum__"])
            if enum_type is None:
                raise GraphArchiveError(f"unknown enum type: {value['__enum__']}")
            return enum_type(value["value"])
        if "__tuple__" in value:
            return tuple(self.decode(item) for item in value["__tuple__"])
        if "__dataclass__" in value:
            class_type = _TYPE_REGISTRY.get(value["__dataclass__"])
            if class_type is None:
                raise GraphArchiveError(
                    f"unknown graph record type: {value['__dataclass__']}"
                )
            kwargs = {name: self.decode(item) for name, item in value["fields"].items()}
            return class_type(**kwargs)
        return {key: self.decode(item) for key, item in value.items()}


def save_graph_archive(
    graph: SpatialSemanticGraph,
    directory: str | Path,
    *,
    events: tuple[GraphEvent, ...] = (),
) -> Path:
    """Write a canonical portable archive and return ``graph.jsonl``."""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    codec = _ArchiveCodec(root / "arrays", write=True)
    records = [
        {
            "record_type": "metadata",
            "format": "panorai-graph-archive/v1",
            "graph_interface": graph.interface,
            "graph_id": graph.graph_id,
        },
        *(
            {
                "record_type": "event",
                "event": codec.encode(event),
            }
            for event in sorted(events, key=lambda item: item.event_id)
        ),
        {"record_type": "snapshot", "graph": codec.encode(graph)},
    ]
    target = root / "graph.jsonl"
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for record in records
    )
    target.write_text(payload, encoding="utf-8")
    return target


def load_graph_archive(directory: str | Path) -> SpatialSemanticGraph:
    """Load and checksum-verify a graph archive snapshot."""

    root = Path(directory)
    manifest = root / "graph.jsonl"
    if not manifest.is_file():
        raise GraphArchiveError("graph archive is missing graph.jsonl")
    try:
        records = [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise GraphArchiveError("graph.jsonl is not readable canonical JSONL") from exc
    if not records or records[0].get("format") != "panorai-graph-archive/v1":
        raise GraphArchiveError("unsupported graph archive format")
    snapshots = [item for item in records if item.get("record_type") == "snapshot"]
    if len(snapshots) != 1:
        raise GraphArchiveError("graph archive must contain exactly one snapshot")
    graph = _ArchiveCodec(root / "arrays", write=False).decode(snapshots[0]["graph"])
    if not isinstance(graph, SpatialSemanticGraph):
        raise GraphArchiveError("snapshot is not a SpatialSemanticGraph")
    if graph.graph_id != records[0].get("graph_id"):
        raise GraphArchiveError("snapshot graph ID does not match archive metadata")
    return graph


def replay_graph_archive(directory: str | Path) -> SpatialSemanticGraph:
    """Replay the archive's canonical records into the immutable snapshot."""

    return load_graph_archive(directory)


__all__ = [
    "GraphArchiveError",
    "load_graph_archive",
    "replay_graph_archive",
    "save_graph_archive",
]
