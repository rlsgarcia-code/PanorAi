# panorai/__init__.py

from .pipeline.pipeline import ProjectionPipeline, PipelineConfig, Pipeline
from .pipeline.pipeline_data import PipelineData
from .pipeline.utils.resizer import ResizerConfig

from .sampler.registry import SamplerRegistry
from spherical_projections import ProjectionRegistry

__version__ = "v1.0-beta"

__all__ = [
    # Pipeline
    "ProjectionPipeline",
    "PipelineConfig",
    "PipelineData",
    "ResizerConfig",
    # Sampler
    "SamplerRegistry",
    # Projection
    "ProjectionRegistry",
    "Pipeline"
]