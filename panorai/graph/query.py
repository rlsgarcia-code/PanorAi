"""Transparent queries over an immutable spatial-semantic graph snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable

import numpy as np

from ._models import ObjectHypothesisNode, SpatialSemanticGraph


@dataclass(frozen=True, slots=True)
class GraphQueryFilters:
    splits: tuple[str, ...] = ()
    frames: tuple[str, ...] = ()
    source_modalities: tuple[str, ...] = ()
    hypothesis_states: tuple[str, ...] = ()
    min_confidence: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must lie within [0, 1]")


@dataclass(frozen=True, slots=True)
class GraphQueryResult:
    hypothesis_id: str
    score: float
    score_components: Mapping[str, float]
    uncertainty: float | None
    evidence_ids: tuple[str, ...]
    state: str
    class_name: str


@runtime_checkable
class TextImageEncoder(Protocol):
    encoder_id: str
    text_image_aligned: bool

    def encode_text(self, text: str) -> np.ndarray: ...


class SpatialSemanticGraphQuery:
    def __init__(self, graph: SpatialSemanticGraph) -> None:
        self.graph = graph
        self._regions = {item.region_id: item for item in graph.regions}
        self._posteriors = {item.posterior_id: item for item in graph.posteriors}
        self._hypotheses = {item.hypothesis_id: item for item in graph.hypotheses}

    def _eligible(
        self, hypothesis: ObjectHypothesisNode, filters: GraphQueryFilters
    ) -> bool:
        if hypothesis.confidence < filters.min_confidence:
            return False
        if (
            filters.hypothesis_states
            and hypothesis.state not in filters.hypothesis_states
        ):
            return False
        regions = [self._regions[item] for item in hypothesis.region_ids]
        if filters.splits and not any(
            item.source.split in filters.splits for item in regions
        ):
            return False
        if filters.frames and not any(
            item.source.frame in filters.frames for item in regions
        ):
            return False
        if filters.source_modalities and not any(
            item.source.modality in filters.source_modalities for item in regions
        ):
            return False
        return True

    def _result(
        self,
        hypothesis: ObjectHypothesisNode,
        *,
        similarity: float,
        spatial: float = 0.0,
    ) -> GraphQueryResult:
        posterior_items = [
            self._posteriors[item]
            for item in hypothesis.posterior_ids
            if item in self._posteriors
        ]
        uncertainty = None
        if posterior_items:
            radii = []
            for posterior in posterior_items:
                if posterior.covariance_xyz is not None:
                    radii.append(
                        float(
                            np.sqrt(
                                max(
                                    float(
                                        np.linalg.eigvalsh(
                                            posterior.covariance_xyz
                                        ).max()
                                    ),
                                    0.0,
                                )
                            )
                        )
                    )
            uncertainty = min(radii) if radii else None
        score = float(
            np.clip(
                0.55 * similarity + 0.30 * hypothesis.confidence + 0.15 * spatial,
                -1.0,
                1.0,
            )
        )
        evidence_ids = tuple(
            sorted(
                {
                    evidence_id
                    for posterior in posterior_items
                    for evidence_id in posterior.source_evidence_ids
                }
            )
        )
        return GraphQueryResult(
            hypothesis_id=hypothesis.hypothesis_id,
            score=score,
            score_components={
                "embedding": similarity,
                "identity_confidence": hypothesis.confidence,
                "spatial": spatial,
            },
            uncertainty=uncertainty,
            evidence_ids=evidence_ids,
            state=hypothesis.state,
            class_name=hypothesis.class_name,
        )

    def query_embedding(
        self,
        embedding: np.ndarray,
        *,
        encoder_id: str,
        filters: GraphQueryFilters | None = None,
        limit: int = 10,
    ) -> tuple[GraphQueryResult, ...]:
        query = np.asarray(embedding, dtype=np.float64)
        if (
            query.ndim != 1
            or not np.all(np.isfinite(query))
            or np.linalg.norm(query) <= 1e-12
        ):
            raise ValueError("embedding must be a finite non-zero vector")
        query = query / np.linalg.norm(query)
        active_filters = filters or GraphQueryFilters()
        results: list[GraphQueryResult] = []
        found_compatible_encoder = False
        for hypothesis in self.graph.hypotheses:
            if not self._eligible(hypothesis, active_filters):
                continue
            vectors = []
            for region_id in hypothesis.region_ids:
                region = self._regions[region_id]
                if region.encoder_id == encoder_id and region.embedding is not None:
                    found_compatible_encoder = True
                    if region.embedding.shape != query.shape:
                        raise ValueError(
                            "query and indexed embedding dimensions differ"
                        )
                    vector = region.embedding / np.linalg.norm(region.embedding)
                    vectors.append(float(np.dot(query, vector)))
            if vectors:
                results.append(self._result(hypothesis, similarity=max(vectors)))
        if not found_compatible_encoder:
            raise ValueError(f"graph contains no embeddings for encoder {encoder_id!r}")
        return tuple(
            sorted(results, key=lambda item: (-item.score, item.hypothesis_id))[:limit]
        )

    def query_text(
        self,
        text: str,
        *,
        encoder: TextImageEncoder,
        filters: GraphQueryFilters | None = None,
        limit: int = 10,
    ) -> tuple[GraphQueryResult, ...]:
        if not text.strip():
            raise ValueError("text query must be non-empty")
        if not encoder.text_image_aligned:
            raise ValueError(
                "query_text requires a declared text-image aligned encoder"
            )
        aligned_regions = [
            item
            for item in self.graph.regions
            if item.encoder_id == encoder.encoder_id and item.embedding is not None
        ]
        if aligned_regions and not all(
            item.text_image_aligned for item in aligned_regions
        ):
            raise ValueError("indexed embeddings do not declare text-image alignment")
        return self.query_embedding(
            encoder.encode_text(text),
            encoder_id=encoder.encoder_id,
            filters=filters,
            limit=limit,
        )

    def query_spatial(
        self,
        bounds_xyz: np.ndarray,
        *,
        frame: str,
        units: str,
        filters: GraphQueryFilters | None = None,
        limit: int = 10,
    ) -> tuple[GraphQueryResult, ...]:
        bounds = np.asarray(bounds_xyz, dtype=np.float64)
        if bounds.shape != (2, 3) or np.any(bounds[1] <= bounds[0]):
            raise ValueError("bounds_xyz must have shape (2, 3) with increasing bounds")
        active_filters = filters or GraphQueryFilters()
        results: list[GraphQueryResult] = []
        for hypothesis in self.graph.hypotheses:
            if not self._eligible(hypothesis, active_filters):
                continue
            matches = []
            for posterior_id in hypothesis.posterior_ids:
                posterior = self._posteriors.get(posterior_id)
                if posterior is None or posterior.mode != "metric":
                    continue
                if posterior.frame != frame or posterior.units != units:
                    raise ValueError("spatial query frame or units are incompatible")
                point = posterior.centroid_xyz
                if point is not None:
                    matches.append(
                        bool(np.all(point >= bounds[0]) and np.all(point <= bounds[1]))
                    )
            if any(matches):
                results.append(self._result(hypothesis, similarity=0.0, spatial=1.0))
        return tuple(
            sorted(results, key=lambda item: (-item.score, item.hypothesis_id))[:limit]
        )

    def neighbors(self, node_id: str) -> tuple[str, ...]:
        if node_id in self._regions:
            neighbors = {
                edge.region_id_b if edge.region_id_a == node_id else edge.region_id_a
                for edge in self.graph.correspondence_edges
                if node_id in {edge.region_id_a, edge.region_id_b}
            }
            neighbors.update(
                item.hypothesis_id
                for item in self.graph.hypotheses
                if node_id in item.region_ids
            )
            return tuple(sorted(neighbors))
        if node_id in self._hypotheses:
            hypothesis = self._hypotheses[node_id]
            return tuple(sorted((*hypothesis.region_ids, *hypothesis.posterior_ids)))
        raise KeyError(node_id)


def query_embedding(
    graph: SpatialSemanticGraph,
    embedding: np.ndarray,
    *,
    encoder_id: str,
    **kwargs: object,
) -> tuple[GraphQueryResult, ...]:
    return SpatialSemanticGraphQuery(graph).query_embedding(
        embedding, encoder_id=encoder_id, **kwargs  # type: ignore[arg-type]
    )


def query_text(
    graph: SpatialSemanticGraph,
    text: str,
    *,
    encoder: TextImageEncoder,
    **kwargs: object,
) -> tuple[GraphQueryResult, ...]:
    return SpatialSemanticGraphQuery(graph).query_text(
        text, encoder=encoder, **kwargs  # type: ignore[arg-type]
    )


def query_spatial(
    graph: SpatialSemanticGraph,
    bounds_xyz: np.ndarray,
    *,
    frame: str,
    units: str,
    **kwargs: object,
) -> tuple[GraphQueryResult, ...]:
    return SpatialSemanticGraphQuery(graph).query_spatial(
        bounds_xyz, frame=frame, units=units, **kwargs  # type: ignore[arg-type]
    )


def neighbors(graph: SpatialSemanticGraph, node_id: str) -> tuple[str, ...]:
    return SpatialSemanticGraphQuery(graph).neighbors(node_id)


__all__ = [
    "GraphQueryFilters",
    "GraphQueryResult",
    "SpatialSemanticGraphQuery",
    "TextImageEncoder",
    "neighbors",
    "query_embedding",
    "query_spatial",
    "query_text",
]
