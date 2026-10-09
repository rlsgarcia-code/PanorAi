"""Compatibility imports for the original experiment-local adapter path.

New experiments should import :mod:`panorai.experimental.deep_learning`.
"""

from panorai.experimental.deep_learning import (
    DenseFCNOutput,
    ImageNetFCN,
    class_activation_map,
    sphericalize,
)

__all__ = [
    "DenseFCNOutput",
    "ImageNetFCN",
    "class_activation_map",
    "sphericalize",
]
