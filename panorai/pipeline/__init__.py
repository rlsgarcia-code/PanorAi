"""Experimental composition of sphere-native and projected-view processing."""

from ._pipeline import Pipeline
from ._protocols import Stage
from ._results import PipelineRun, StageRecord, StageResult
from ._stages import (
    AggregateViews,
    LiftToBearings,
    OnSphere,
    PerView,
    ReconstructERP,
    SampleViews,
)
from ._fields import Output, RasterField, RasterGrid, ResultBundle
from ._observations import (
    ClassificationResult,
    EmbeddingResult,
    PerViewResults,
    SphericalObservations,
    ViewObservations,
    ViewResult,
)
from ._routing import Attach, Parallel, Select
from ._serialization import PipelineSerializationError, registered_pipeline_components
from ._cache import PipelineCacheError
from ._torch_serialization import register_torch_module, registered_torch_modules
from ._instances import (
    CAMEvidenceFrame,
    CAMGuidedSphericalInstances,
    CAMRegionProposal,
    CAMRegionProposalSet,
    CAMRegionProposals,
    SAMMaskProposalBatch,
    SAMMaskRefinement,
    SphericalCAM,
    SphericalInstanceFusion,
)

__all__ = [
    "AggregateViews",
    "Attach",
    "CAMEvidenceFrame",
    "CAMGuidedSphericalInstances",
    "CAMRegionProposal",
    "CAMRegionProposalSet",
    "CAMRegionProposals",
    "ClassificationResult",
    "EmbeddingResult",
    "LiftToBearings",
    "OnSphere",
    "Output",
    "Parallel",
    "PerView",
    "PerViewResults",
    "Pipeline",
    "PipelineCacheError",
    "PipelineRun",
    "PipelineSerializationError",
    "RasterField",
    "RasterGrid",
    "ReconstructERP",
    "ResultBundle",
    "SampleViews",
    "SAMMaskProposalBatch",
    "SAMMaskRefinement",
    "Select",
    "SphericalObservations",
    "SphericalCAM",
    "SphericalInstanceFusion",
    "StageRecord",
    "Stage",
    "StageResult",
    "ViewObservations",
    "ViewResult",
    "registered_pipeline_components",
    "register_torch_module",
    "registered_torch_modules",
]
