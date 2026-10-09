"""Experimental deep-learning portability for spherical PanorAi operators.

The namespace is explicit opt-in and requires Torch. It combines reusable ERP
spherical operators with adapters that preserve pretrained classifier
parameters while changing their topology and sampling domain.
"""

from panorai.image_processing.torch import (
    SPHERICAL_TORCH_CONVOLUTION_INTERFACE,
    PortedLayer,
    SphericalConv2d,
    SphericalConvTranspose2d,
    SphericalMaxPool2d,
    SphericalPortReport,
    port_module,
    port_module_with_report,
    spherical_area_average,
)

from .fcn import DenseFCNOutput, ImageNetFCN, class_activation_map
from .depth import (
    SPHERICAL_METRIC_DEPTH_INTERFACE,
    METRIC3D_CHECKPOINT_REVISION,
    METRIC3D_CONVNEXT_TINY_V1,
    METRIC3D_SOURCE,
    METRIC3D_SOURCE_COMMIT,
    ExternalArtifactRecord,
    ExternalArtifactSpec,
    LoadedSphericalDepthModel,
    Metric3DAssets,
    SphericalMetric3D,
    UpstreamTermsNotAcceptedError,
    acquire_metric3d_convnext_tiny_v1,
    load_spherical_metric3d_convnext_tiny_v1,
)
from .pretrained import (
    SUPPORTED_IMAGENET_MODELS,
    ImageNetWeightRecord,
    LoadedImageNetModel,
    load_pretrained_imagenet_model,
    prefetch_imagenet_weights,
)

sphericalize = port_module

__all__ = [
    "SPHERICAL_TORCH_CONVOLUTION_INTERFACE",
    "SPHERICAL_METRIC_DEPTH_INTERFACE",
    "DenseFCNOutput",
    "ImageNetFCN",
    "ImageNetWeightRecord",
    "ExternalArtifactRecord",
    "ExternalArtifactSpec",
    "LoadedSphericalDepthModel",
    "LoadedImageNetModel",
    "PortedLayer",
    "METRIC3D_CHECKPOINT_REVISION",
    "METRIC3D_CONVNEXT_TINY_V1",
    "METRIC3D_SOURCE",
    "METRIC3D_SOURCE_COMMIT",
    "Metric3DAssets",
    "SphericalConv2d",
    "SphericalConvTranspose2d",
    "SphericalMetric3D",
    "SphericalMaxPool2d",
    "SphericalPortReport",
    "SUPPORTED_IMAGENET_MODELS",
    "UpstreamTermsNotAcceptedError",
    "acquire_metric3d_convnext_tiny_v1",
    "class_activation_map",
    "load_pretrained_imagenet_model",
    "load_spherical_metric3d_convnext_tiny_v1",
    "port_module",
    "port_module_with_report",
    "prefetch_imagenet_weights",
    "spherical_area_average",
    "sphericalize",
]
