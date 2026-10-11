"""Incremental, replayable spatial-semantic graph builder."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, replace
from typing import Iterable

from ._events import GraphEvent
from ._models import (
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
from ._policies import PromotionPolicy
from .observations import deterministic_id


@dataclass(frozen=True, slots=True)
class GraphBuilderConfig:
    graph_name: str
    promotion_policy: PromotionPolicy

    @classmethod
    def conservative_v1(cls, graph_name: str) -> "GraphBuilderConfig":
        return cls(graph_name, PromotionPolicy.conservative_v1())

    def __post_init__(self) -> None:
        if not self.graph_name.strip():
            raise ValueError("graph_name must be non-empty")


class SpatialSemanticGraphBuilder:
    """Accumulate prepared evidence and materialize deterministic snapshots.

    The builder never selects or calls a provider.  All perception results are
    passed explicitly through ``add_*`` methods.
    """

    def __init__(self, config: GraphBuilderConfig) -> None:
        if not isinstance(config, GraphBuilderConfig):
            raise TypeError("an explicit GraphBuilderConfig is required")
        self.config = config
        self._views: dict[str, SphericalViewNode] = {}
        self._feature_evidence: dict[str, PairFeatureEvidence] = {}
        self._poses: dict[str, RelativePoseEdge] = {}
        self._regions: dict[str, SemanticRegionNode] = {}
        self._candidates: dict[str, RegionCorrespondenceCandidate] = {}
        self._edges: dict[str, RegionCorrespondenceEdge] = {}
        self._posteriors: dict[str, SpatialSemanticPosterior] = {}
        self._events: dict[str, GraphEvent] = {}
        self._suppressed_edges: set[str] = set()
        self._lineage_by_regions: dict[tuple[str, ...], tuple[str, ...]] = {}
        self._forced_merges: list[frozenset[str]] = []
        self._superseded: dict[str, ObjectHypothesisNode] = {}

    @property
    def events(self) -> tuple[GraphEvent, ...]:
        return tuple(self._events[key] for key in sorted(self._events))

    def _record(self, event_type: str, *subject_ids: str, **payload: object) -> None:
        event = GraphEvent.create(event_type, tuple(subject_ids), payload)
        self._events[event.event_id] = event

    @staticmethod
    def _insert(target: dict[str, object], key: str, value: object) -> None:
        current = target.get(key)
        if current is not None and current is not value:
            raise ValueError(f"conflicting graph record for deterministic ID {key}")
        target[key] = value

    @staticmethod
    def _require_method_evidence(source: object) -> None:
        if not getattr(source, "allowed_as_method_evidence", False):
            raise ValueError(
                "evaluation_only or disabled evidence cannot enter method state"
            )

    def add_view(self, view: SphericalViewNode) -> None:
        if view.access_role == "evaluation_only":
            raise ValueError("evaluation_only views cannot enter graph state")
        self._insert(self._views, view.view_id, view)
        self._record("view-added", view.view_id)

    def add_feature_evidence(self, evidence: PairFeatureEvidence) -> None:
        self._require_method_evidence(evidence.source)
        self._insert(self._feature_evidence, evidence.evidence_id, evidence)
        self._record("feature-evidence-added", evidence.evidence_id)

    def add_pose(self, pose: RelativePoseEdge) -> None:
        self._require_method_evidence(pose.source)
        if not pose.accepted or pose.degenerate:
            raise ValueError("rejected or degenerate pose fails closed")
        self._insert(self._poses, pose.edge_id, pose)
        self._record("pose-added", pose.edge_id)

    def add_region(self, region: SemanticRegionNode) -> None:
        self._require_method_evidence(region.source)
        self._insert(self._regions, region.region_id, region)
        self._record("region-added", region.region_id)

    def add_candidate(self, candidate: RegionCorrespondenceCandidate) -> None:
        self._insert(self._candidates, candidate.candidate_id, candidate)
        self._record("correspondence-candidate-added", candidate.candidate_id)

    def add_correspondence(self, edge: RegionCorrespondenceEdge) -> None:
        self._insert(self._edges, edge.edge_id, edge)
        self._record("correspondence-added", edge.edge_id)

    def add_posterior(self, posterior: SpatialSemanticPosterior) -> None:
        self._insert(self._posteriors, posterior.posterior_id, posterior)
        self._record("posterior-added", posterior.posterior_id)

    def add_many(self, records: Iterable[object]) -> None:
        dispatch = {
            SphericalViewNode: self.add_view,
            PairFeatureEvidence: self.add_feature_evidence,
            RelativePoseEdge: self.add_pose,
            SemanticRegionNode: self.add_region,
            RegionCorrespondenceCandidate: self.add_candidate,
            RegionCorrespondenceEdge: self.add_correspondence,
            SpatialSemanticPosterior: self.add_posterior,
        }
        for record in records:
            handler = dispatch.get(type(record))
            if handler is None:
                raise TypeError(f"unsupported graph record: {type(record).__name__}")
            handler(record)  # type: ignore[arg-type]

    def _validate_references(self) -> None:
        for evidence in self._feature_evidence.values():
            for view_id in (evidence.view_id_a, evidence.view_id_b):
                if view_id not in self._views:
                    raise ValueError(
                        f"feature evidence references missing view {view_id}"
                    )
        for pose in self._poses.values():
            if pose.feature_evidence_id not in self._feature_evidence:
                raise ValueError("pose references missing feature evidence")
            if pose.view_id_a not in self._views or pose.view_id_b not in self._views:
                raise ValueError("pose references a missing view")
            first_view = self._views[pose.view_id_a]
            second_view = self._views[pose.view_id_b]
            if first_view.split != second_view.split:
                raise ValueError("relative pose cannot cross dataset splits")
            if pose.source.split is not None and pose.source.split != first_view.split:
                raise ValueError("pose source and view split are incompatible")
        for region in self._regions.values():
            if region.view_id not in self._views:
                raise ValueError(f"region references missing view {region.view_id}")
            view = self._views[region.view_id]
            if region.source.split is not None and region.source.split != view.split:
                raise ValueError("region and view split are incompatible")
            if region.source.frame is not None and region.source.frame != view.frame:
                raise ValueError("region and view frame are incompatible")
        for edge in self._edges.values():
            if edge.candidate_id not in self._candidates:
                raise ValueError("correspondence edge references missing candidate")
            if (
                edge.region_id_a not in self._regions
                or edge.region_id_b not in self._regions
            ):
                raise ValueError("correspondence edge references missing region")
            first = self._regions[edge.region_id_a]
            second = self._regions[edge.region_id_b]
            if first.vocabulary != second.vocabulary:
                raise ValueError("correspondence joins incompatible vocabularies")

    def _components(self) -> list[tuple[str, ...]]:
        adjacency: dict[str, set[str]] = defaultdict(set)
        for region_id in self._regions:
            adjacency[region_id]
        for edge_id, edge in self._edges.items():
            if edge_id in self._suppressed_edges or edge.polarity != "positive":
                continue
            adjacency[edge.region_id_a].add(edge.region_id_b)
            adjacency[edge.region_id_b].add(edge.region_id_a)
        components: list[tuple[str, ...]] = []
        pending = set(adjacency)
        while pending:
            start = min(pending)
            queue = deque([start])
            reached: set[str] = set()
            while queue:
                current = queue.popleft()
                if current in reached:
                    continue
                reached.add(current)
                queue.extend(sorted(adjacency[current] - reached))
            pending -= reached
            components.append(tuple(sorted(reached)))
        for forced in self._forced_merges:
            touched = [item for item in components if set(item).intersection(forced)]
            if len(touched) < 2:
                continue
            joined = tuple(sorted({region for item in touched for region in item}))
            components = [item for item in components if item not in touched]
            components.append(joined)
        return sorted(components)

    def _component_confidence(self, region_ids: tuple[str, ...]) -> float:
        region_set = set(region_ids)
        positive: dict[str, float] = {}
        negative: dict[str, float] = {}
        for edge_id, edge in self._edges.items():
            if edge_id in self._suppressed_edges:
                continue
            if {edge.region_id_a, edge.region_id_b}.issubset(region_set):
                groups = edge.correlation_groups or (edge.edge_id,)
                key = "|".join(groups)
                target = positive if edge.polarity == "positive" else negative
                target[key] = max(target.get(key, 0.0), edge.confidence)
        if not positive:
            return max(self._regions[item].semantic_score for item in region_ids) * 0.5
        positive_score = sum(positive.values()) / len(positive)
        negative_score = max(negative.values(), default=0.0)
        return max(0.0, positive_score * (1.0 - negative_score))

    def _build_hypotheses(self) -> tuple[ObjectHypothesisNode, ...]:
        result: list[ObjectHypothesisNode] = []
        for region_ids in self._components():
            regions = [self._regions[item] for item in region_ids]
            class_ids = {item.class_id for item in regions}
            if len(class_ids) != 1:
                raise ValueError("one entity component cannot contain multiple classes")
            view_ids = tuple(sorted({item.view_id for item in regions}))
            confidence = self._component_confidence(region_ids)
            policy = self.config.promotion_policy
            if len(view_ids) < policy.min_independent_views:
                state = "proposed"
            elif confidence >= policy.min_confidence:
                state = "confirmed"
            elif confidence < policy.dormant_below_confidence:
                state = "dormant"
            else:
                state = "ambiguous"
            edge_ids = tuple(
                sorted(
                    edge.edge_id
                    for edge in self._edges.values()
                    if edge.edge_id not in self._suppressed_edges
                    and {edge.region_id_a, edge.region_id_b}.issubset(region_ids)
                )
            )
            source_ids = {
                source_id
                for edge_id in edge_ids
                for source_id in self._edges[edge_id].source_evidence_ids
            }
            posterior_ids = tuple(
                sorted(
                    item.posterior_id
                    for item in self._posteriors.values()
                    if source_ids.intersection(item.source_evidence_ids)
                )
            )
            hypothesis_id = deterministic_id("entity", region_ids)
            result.append(
                ObjectHypothesisNode(
                    hypothesis_id=hypothesis_id,
                    class_id=regions[0].class_id,
                    class_name=regions[0].class_name,
                    state=state,  # type: ignore[arg-type]
                    region_ids=region_ids,
                    view_ids=view_ids,
                    correspondence_edge_ids=edge_ids,
                    posterior_ids=posterior_ids,
                    confidence=confidence,
                    lineage_parent_ids=self._lineage_by_regions.get(region_ids, ()),
                    independent_view_count=len(view_ids),
                )
            )
        result.extend(
            item
            for hypothesis_id, item in self._superseded.items()
            if hypothesis_id not in {current.hypothesis_id for current in result}
        )
        return tuple(sorted(result, key=lambda item: item.hypothesis_id))

    def merge_hypotheses(self, hypothesis_ids: tuple[str, ...]) -> str:
        current = {
            item.hypothesis_id: item
            for item in self._build_hypotheses()
            if item.state != "superseded"
        }
        if len(hypothesis_ids) < 2 or any(
            item not in current for item in hypothesis_ids
        ):
            raise ValueError("merge requires at least two current hypotheses")
        regions = tuple(
            sorted(
                {
                    region
                    for item in hypothesis_ids
                    for region in current[item].region_ids
                }
            )
        )
        merged_id = deterministic_id("entity", regions)
        self._forced_merges.append(frozenset(regions))
        self._lineage_by_regions[regions] = tuple(sorted(hypothesis_ids))
        for hypothesis_id in hypothesis_ids:
            self._superseded[hypothesis_id] = replace(
                current[hypothesis_id],
                state="superseded",
                superseded_by_ids=(merged_id,),
            )
        self._record("hypotheses-merged", *hypothesis_ids, merged_id=merged_id)
        return merged_id

    def split_hypothesis(
        self, hypothesis_id: str, region_groups: tuple[tuple[str, ...], ...]
    ) -> tuple[str, ...]:
        current = {
            item.hypothesis_id: item
            for item in self._build_hypotheses()
            if item.state != "superseded"
        }
        if hypothesis_id not in current:
            raise ValueError("unknown hypothesis")
        original = set(current[hypothesis_id].region_ids)
        flattened = [item for group in region_groups for item in group]
        if set(flattened) != original or len(flattened) != len(set(flattened)):
            raise ValueError("split groups must partition the original regions")
        group_by_region = {
            region_id: index
            for index, group in enumerate(region_groups)
            for region_id in group
        }
        for edge_id, edge in self._edges.items():
            if group_by_region.get(edge.region_id_a) != group_by_region.get(
                edge.region_id_b
            ):
                self._suppressed_edges.add(edge_id)
        children = tuple(
            deterministic_id("entity", tuple(sorted(group))) for group in region_groups
        )
        self._forced_merges = [
            item for item in self._forced_merges if not item.intersection(original)
        ]
        for group in region_groups:
            self._lineage_by_regions[tuple(sorted(group))] = (hypothesis_id,)
        self._superseded[hypothesis_id] = replace(
            current[hypothesis_id],
            state="superseded",
            superseded_by_ids=children,
        )
        self._record("hypothesis-split", hypothesis_id, *children)
        return children

    def snapshot(self) -> SpatialSemanticGraph:
        self._validate_references()
        hypotheses = self._build_hypotheses()
        graph_id = deterministic_id(
            "graph",
            self.config.graph_name,
            tuple(sorted(self._events)),
            tuple(sorted(self._suppressed_edges)),
        )
        return SpatialSemanticGraph(
            graph_id=graph_id,
            views=tuple(self._views[key] for key in sorted(self._views)),
            feature_evidence=tuple(
                self._feature_evidence[key] for key in sorted(self._feature_evidence)
            ),
            pose_edges=tuple(self._poses[key] for key in sorted(self._poses)),
            regions=tuple(self._regions[key] for key in sorted(self._regions)),
            correspondence_candidates=tuple(
                self._candidates[key] for key in sorted(self._candidates)
            ),
            correspondence_edges=tuple(
                self._edges[key]
                for key in sorted(self._edges)
                if key not in self._suppressed_edges
            ),
            posteriors=tuple(self._posteriors[key] for key in sorted(self._posteriors)),
            hypotheses=hypotheses,
            event_ids=tuple(sorted(self._events)),
        )


__all__ = ["GraphBuilderConfig", "SpatialSemanticGraphBuilder"]
