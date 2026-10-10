"""Experimental deep-learning portability for spherical PanorAi operators.

The namespace is explicit opt-in. Public objects are resolved lazily so the
pure NumPy segmentation contracts remain importable without installing Torch,
while model and convolution objects load their optional backend on demand.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    # Spherical Torch operators.
    "SPHERICAL_TORCH_CONVOLUTION_INTERFACE": (
        "panorai.image_processing.torch",
        "SPHERICAL_TORCH_CONVOLUTION_INTERFACE",
    ),
    "PortedLayer": ("panorai.image_processing.torch", "PortedLayer"),
    "SphericalConv2d": ("panorai.image_processing.torch", "SphericalConv2d"),
    "SphericalConvTranspose2d": (
        "panorai.image_processing.torch",
        "SphericalConvTranspose2d",
    ),
    "SphericalMaxPool2d": (
        "panorai.image_processing.torch",
        "SphericalMaxPool2d",
    ),
    "SphericalPortReport": (
        "panorai.image_processing.torch",
        "SphericalPortReport",
    ),
    "port_module": ("panorai.image_processing.torch", "port_module"),
    "port_module_with_report": (
        "panorai.image_processing.torch",
        "port_module_with_report",
    ),
    "spherical_area_average": (
        "panorai.image_processing.torch",
        "spherical_area_average",
    ),
    "sphericalize": ("panorai.image_processing.torch", "port_module"),
    # Fully convolutional classifiers.
    "DenseFCNOutput": (f"{__name__}.fcn", "DenseFCNOutput"),
    "ImageNetFCN": (f"{__name__}.fcn", "ImageNetFCN"),
    "class_activation_map": (f"{__name__}.fcn", "class_activation_map"),
    # Metric3D.
    "SPHERICAL_METRIC_DEPTH_INTERFACE": (
        f"{__name__}.depth",
        "SPHERICAL_METRIC_DEPTH_INTERFACE",
    ),
    "METRIC3D_CHECKPOINT_REVISION": (
        f"{__name__}.depth",
        "METRIC3D_CHECKPOINT_REVISION",
    ),
    "METRIC3D_CONVNEXT_TINY_V1": (
        f"{__name__}.depth",
        "METRIC3D_CONVNEXT_TINY_V1",
    ),
    "METRIC3D_SOURCE": (f"{__name__}.depth", "METRIC3D_SOURCE"),
    "METRIC3D_SOURCE_COMMIT": (f"{__name__}.depth", "METRIC3D_SOURCE_COMMIT"),
    "ExternalArtifactRecord": (f"{__name__}.depth", "ExternalArtifactRecord"),
    "ExternalArtifactSpec": (f"{__name__}.depth", "ExternalArtifactSpec"),
    "LoadedSphericalDepthModel": (
        f"{__name__}.depth",
        "LoadedSphericalDepthModel",
    ),
    "Metric3DAssets": (f"{__name__}.depth", "Metric3DAssets"),
    "SphericalMetric3D": (f"{__name__}.depth", "SphericalMetric3D"),
    "UpstreamTermsNotAcceptedError": (
        f"{__name__}.depth",
        "UpstreamTermsNotAcceptedError",
    ),
    "acquire_metric3d_convnext_tiny_v1": (
        f"{__name__}.depth",
        "acquire_metric3d_convnext_tiny_v1",
    ),
    "load_spherical_metric3d_convnext_tiny_v1": (
        f"{__name__}.depth",
        "load_spherical_metric3d_convnext_tiny_v1",
    ),
    # Depth Anything 3.
    "SPHERICAL_DA3_METRIC_DEPTH_INTERFACE": (
        f"{__name__}.da3",
        "SPHERICAL_DA3_METRIC_DEPTH_INTERFACE",
    ),
    "DA3_CANONICAL_FOCAL_PX": (f"{__name__}.da3", "DA3_CANONICAL_FOCAL_PX"),
    "DA3_CHECKPOINT_REVISION": (f"{__name__}.da3", "DA3_CHECKPOINT_REVISION"),
    "DA3_MODEL_ID": (f"{__name__}.da3", "DA3_MODEL_ID"),
    "DA3_PATCH_SIZE": (f"{__name__}.da3", "DA3_PATCH_SIZE"),
    "DA3_SOURCE": (f"{__name__}.da3", "DA3_SOURCE"),
    "DA3_SOURCE_COMMIT": (f"{__name__}.da3", "DA3_SOURCE_COMMIT"),
    "DA3METRIC_LARGE": (f"{__name__}.da3", "DA3METRIC_LARGE"),
    "DA3Assets": (f"{__name__}.da3", "DA3Assets"),
    "DA3MetricTangent": (f"{__name__}.da3", "DA3MetricTangent"),
    "DA3TermsNotAcceptedError": (f"{__name__}.da3", "DA3TermsNotAcceptedError"),
    "LoadedDA3MetricModel": (f"{__name__}.da3", "LoadedDA3MetricModel"),
    "acquire_da3metric_large": (f"{__name__}.da3", "acquire_da3metric_large"),
    "load_da3metric_large": (f"{__name__}.da3", "load_da3metric_large"),
    # Torchvision ImageNet models.
    "SUPPORTED_IMAGENET_MODELS": (
        f"{__name__}.pretrained",
        "SUPPORTED_IMAGENET_MODELS",
    ),
    "ImageNetWeightRecord": (f"{__name__}.pretrained", "ImageNetWeightRecord"),
    "LoadedImageNetModel": (f"{__name__}.pretrained", "LoadedImageNetModel"),
    "load_pretrained_imagenet_model": (
        f"{__name__}.pretrained",
        "load_pretrained_imagenet_model",
    ),
    "prefetch_imagenet_weights": (
        f"{__name__}.pretrained",
        "prefetch_imagenet_weights",
    ),
    # InternImage-G.
    "SPHERICAL_INTERNIMAGE_INTERFACE": (
        f"{__name__}.internimage",
        "SPHERICAL_INTERNIMAGE_INTERFACE",
    ),
    "INTERNIMAGE_G_REPOSITORY": (
        f"{__name__}.internimage",
        "INTERNIMAGE_G_REPOSITORY",
    ),
    "INTERNIMAGE_G_REVISION": (f"{__name__}.internimage", "INTERNIMAGE_G_REVISION"),
    "INTERNIMAGE_G_SHARDS": (f"{__name__}.internimage", "INTERNIMAGE_G_SHARDS"),
    "DenseInternImageOutput": (f"{__name__}.internimage", "DenseInternImageOutput"),
    "InternImageGAssets": (f"{__name__}.internimage", "InternImageGAssets"),
    "InternImageGDenseClassifier": (
        f"{__name__}.internimage",
        "InternImageGDenseClassifier",
    ),
    "InternImageShardSpec": (f"{__name__}.internimage", "InternImageShardSpec"),
    "InternImageTermsNotAcceptedError": (
        f"{__name__}.internimage",
        "InternImageTermsNotAcceptedError",
    ),
    "LoadedInternImageG": (f"{__name__}.internimage", "LoadedInternImageG"),
    "SphericalDCNv3": (f"{__name__}.internimage", "SphericalDCNv3"),
    "SphericalDCNv3Record": (f"{__name__}.internimage", "SphericalDCNv3Record"),
    "SphericalInternImagePortReport": (
        f"{__name__}.internimage",
        "SphericalInternImagePortReport",
    ),
    "acquire_internimage_g": (f"{__name__}.internimage", "acquire_internimage_g"),
    "attention_pool_class_contributions": (
        f"{__name__}.internimage",
        "attention_pool_class_contributions",
    ),
    "load_pretrained_internimage_g": (
        f"{__name__}.internimage",
        "load_pretrained_internimage_g",
    ),
    "port_internimage_to_spherical": (
        f"{__name__}.internimage",
        "port_internimage_to_spherical",
    ),
    "spherical_dcnv3_core": (f"{__name__}.internimage", "spherical_dcnv3_core"),
}

for _name in (
    "SPHERICAL_benchmark_SEGMENTATION_INTERFACE",
    "SPHERICAL_SEMANTIC_SEGMENTATION_INTERFACE",
    "benchmark_IMAGENET_PROXY_VOCABULARY",
    "DenseSemanticEvidence",
    "Sam21Assets",
    "Sam21HieraLargeSegmenter",
    "Sam21TermsNotAcceptedError",
    "SemanticProxyConcept",
    "SphericalBinaryMask",
    "SphericalbenchmarkSegmenter",
    "SphericalMaskProposal",
    "SphericalSeed",
    "SphericalSegment",
    "SphericalSegmentationConfig",
    "SphericalSegmentationResult",
    "SphericalSemanticSegmenter",
    "acquire_sam21_hiera_large",
    "concept_evidence_from_imagenet",
    "fibonacci_coverage_seeds",
    "load_sam21_hiera_large",
    "merge_proposals",
    "resolve_panoptic_map",
    "select_semantic_seeds",
):
    _EXPORTS[_name] = (f"{__name__}.segmentation", _name)

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve one public experimental object without eager backend imports."""

    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
