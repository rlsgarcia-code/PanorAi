"""Experimental semantic association and spatial localization for two views.

This package composes maps, spherical features, matches, and relative pose.
It deliberately does not construct or depend on a scene graph.
"""

from ._association import (
    assign_region_associations,
    object_hypothesis_id,
    propose_region_associations,
)
from ._localization import localize_region_association
from ._match_clusters import (
    JOINT_MATCH_CLUSTERS_INTERFACE,
    JointMatchCluster,
    JointMatchClusterConfig,
    JointMatchClusterResult,
    cluster_region_association_matches,
)
from ._maps import (
    SEMANTIC_REGION_PROPOSALS_INTERFACE,
    SemanticRegionProposalConfig,
    SemanticRegionProposalResult,
    semantic_region_from_map,
    semantic_regions_from_map,
)
from ._models import (
    OBJECT_LOCALIZATION_INTERFACE,
    OBJECT_LOCALIZATION_STABILITY,
    ObjectHypothesis,
    ObjectLocalizationConfig,
    ObjectLocalizationResult,
    PairObjectLocalizationInput,
    RegionAssociation,
    SemanticQuery,
    SemanticRegionObservation,
    SpatialLocationHypothesis,
)
from ._pipeline import ObjectLocalizationPipeline
from ._semantic_prior import (
    SEMANTIC_MATCH_PRIOR_INTERFACE,
    SemanticMatchPrior,
    SemanticMatchPriorConfig,
    build_semantic_match_prior,
)

__all__ = [
    "JOINT_MATCH_CLUSTERS_INTERFACE",
    "JointMatchCluster",
    "JointMatchClusterConfig",
    "JointMatchClusterResult",
    "OBJECT_LOCALIZATION_INTERFACE",
    "OBJECT_LOCALIZATION_STABILITY",
    "ObjectHypothesis",
    "ObjectLocalizationConfig",
    "ObjectLocalizationPipeline",
    "ObjectLocalizationResult",
    "PairObjectLocalizationInput",
    "RegionAssociation",
    "SEMANTIC_MATCH_PRIOR_INTERFACE",
    "SEMANTIC_REGION_PROPOSALS_INTERFACE",
    "SemanticMatchPrior",
    "SemanticMatchPriorConfig",
    "SemanticQuery",
    "SemanticRegionProposalConfig",
    "SemanticRegionProposalResult",
    "SemanticRegionObservation",
    "SpatialLocationHypothesis",
    "assign_region_associations",
    "build_semantic_match_prior",
    "cluster_region_association_matches",
    "localize_region_association",
    "object_hypothesis_id",
    "propose_region_associations",
    "semantic_region_from_map",
    "semantic_regions_from_map",
]
