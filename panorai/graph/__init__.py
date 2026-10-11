"""Experimental spatial-semantic graph contracts and lifecycle APIs.

Importing :mod:`panorai.graph` loads no neural-network, reconstruction, or
database backend.  Perception providers are selected and called explicitly by
the application; the graph builder only consumes their prepared results.
"""

from ._events import GraphEvent
from ._models import (
    GRAPH_INTERFACE,
    GRAPH_STABILITY,
    EvidenceMode,
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
from ._policies import PromotionPolicy
from .association import (
    RegionAssociationConfig,
    RegionAssociationInput,
    RegionAssociationResult,
    assign_region_correspondences,
    associate_regions,
    propose_region_correspondences,
)
from .builder import GraphBuilderConfig, SpatialSemanticGraphBuilder
from .localization import (
    PairwiseLocalizationConfig,
    PairwiseLocalizationInput,
    PairwiseLocalizationResult,
    localize_region_pair,
)
from .providers import (
    CallablePanorAiProvider,
    DepthProvider,
    DisabledProvider,
    EvidenceProvider,
    ExternalProvider,
    MultiViewPoseProvider,
    OracleProvider,
    PoseProvider,
    SemanticProvider,
)
from .query import (
    GraphQueryFilters,
    GraphQueryResult,
    SpatialSemanticGraphQuery,
    TextImageEncoder,
    neighbors,
    query_embedding,
    query_spatial,
    query_text,
)
from .serialization import (
    GraphArchiveError,
    load_graph_archive,
    replay_graph_archive,
    save_graph_archive,
)

__all__ = [
    "GRAPH_INTERFACE",
    "GRAPH_STABILITY",
    "CallablePanorAiProvider",
    "DepthProvider",
    "DisabledProvider",
    "EvidenceMode",
    "EvidenceProvider",
    "EvidenceSource",
    "ExternalProvider",
    "GraphArchiveError",
    "GraphBuilderConfig",
    "GraphEvent",
    "GraphQueryFilters",
    "GraphQueryResult",
    "MultiViewPoseProvider",
    "ObjectHypothesisNode",
    "OracleProvider",
    "PairFeatureEvidence",
    "PairwiseLocalizationConfig",
    "PairwiseLocalizationInput",
    "PairwiseLocalizationResult",
    "PoseProvider",
    "PromotionPolicy",
    "RegionAssociationConfig",
    "RegionAssociationInput",
    "RegionAssociationResult",
    "RegionCorrespondenceCandidate",
    "RegionCorrespondenceEdge",
    "RelativePoseEdge",
    "SemanticProvider",
    "SemanticRegionNode",
    "SpatialSemanticGraph",
    "SpatialSemanticGraphBuilder",
    "SpatialSemanticGraphQuery",
    "SpatialSemanticPosterior",
    "SphericalViewNode",
    "TextImageEncoder",
    "assign_region_correspondences",
    "associate_regions",
    "load_graph_archive",
    "localize_region_pair",
    "neighbors",
    "propose_region_correspondences",
    "query_embedding",
    "query_spatial",
    "query_text",
    "replay_graph_archive",
    "save_graph_archive",
]
