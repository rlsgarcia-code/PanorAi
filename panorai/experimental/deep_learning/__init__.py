"""Experimental deep-learning portability for spherical PanorAi operators.

The namespace is explicit opt-in and requires Torch. It combines reusable ERP
spherical operators with adapters that preserve pretrained classifier
parameters while changing their topology and sampling domain.
"""

from panorai.image_processing.torch import (
    SPHERICAL_TORCH_CONVOLUTION_INTERFACE,
    PortedLayer,
    SphericalConv2d,
    SphericalMaxPool2d,
    SphericalPortReport,
    port_module,
    port_module_with_report,
    spherical_area_average,
)

from .fcn import DenseFCNOutput, ImageNetFCN, class_activation_map
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
    "DenseFCNOutput",
    "ImageNetFCN",
    "ImageNetWeightRecord",
    "LoadedImageNetModel",
    "PortedLayer",
    "SphericalConv2d",
    "SphericalMaxPool2d",
    "SphericalPortReport",
    "SUPPORTED_IMAGENET_MODELS",
    "class_activation_map",
    "load_pretrained_imagenet_model",
    "port_module",
    "port_module_with_report",
    "prefetch_imagenet_weights",
    "spherical_area_average",
    "sphericalize",
]
