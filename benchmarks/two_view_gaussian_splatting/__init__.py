"""Development-only two-view Gaussian depth-prior experiment."""

from .gaussian_depth import (
    GaussianDepthOptions,
    GaussianDepthResult,
    GaussianDepthStage,
    align_depth_scale_from_landmarks,
    optimize_hierarchical_gaussian_depth,
    optimize_gaussian_depth,
    project_erp,
    recommended_full_factor_gaussian_depth_stages,
    recommended_gaussian_depth_stages,
    render_spherical_gaussians,
)
from .gaussian_depth_feedback import (
    GaussianDepthAnchors,
    GaussianDepthFeedbackOptions,
    GaussianDepthFeedbackResult,
    rasterize_gaussian_depth_anchors,
    read_gaussian_centres_ply,
    refine_depth_from_gaussian_anchors,
)
from .surface_densification import (
    SurfaceDensificationOptions,
    densify_stereo_surface,
)

__all__ = [
    "GaussianDepthOptions",
    "GaussianDepthResult",
    "GaussianDepthStage",
    "align_depth_scale_from_landmarks",
    "optimize_hierarchical_gaussian_depth",
    "optimize_gaussian_depth",
    "project_erp",
    "recommended_full_factor_gaussian_depth_stages",
    "recommended_gaussian_depth_stages",
    "render_spherical_gaussians",
    "GaussianDepthAnchors",
    "GaussianDepthFeedbackOptions",
    "GaussianDepthFeedbackResult",
    "rasterize_gaussian_depth_anchors",
    "read_gaussian_centres_ply",
    "refine_depth_from_gaussian_anchors",
    "SurfaceDensificationOptions",
    "densify_stereo_surface",
]
