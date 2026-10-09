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

sphericalize = port_module

__all__ = [
    "SPHERICAL_TORCH_CONVOLUTION_INTERFACE",
    "DenseFCNOutput",
    "ImageNetFCN",
    "PortedLayer",
    "SphericalConv2d",
    "SphericalMaxPool2d",
    "SphericalPortReport",
    "class_activation_map",
    "port_module",
    "port_module_with_report",
    "spherical_area_average",
    "sphericalize",
]
