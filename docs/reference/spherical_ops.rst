Spherical operations API
========================

Use the :doc:`Spherical Ops catalog <catalog/operations>` to compare function,
configured-object, algorithm and Torch-layer forms.  This page records the
namespace-level contracts and important validation rules.

NumPy operations
----------------

``panorai.ops.spherical.functional`` contains direct functions. Matching
immutable objects are exported from ``panorai.ops.spherical``. Algorithms are
``SLIC``, ``DoGDetector``, ``SphericalNMS``, ``Canny``, ``GaussianPyramid`` and
``LaplacianPyramid``.

The general NumPy operations are Stable as ``panorai-spherical-ops/v1``.
``GeodesicDisk``, ``Dilate``, ``Erode``, ``Opening``, ``Closing`` and their
lower-case function forms are separately Stable as
``panorai-spherical-morphology/v1``.  They accept boolean or scalar floating
ERP rasters, use nearest sampling, preserve the grid and propagate support and
validity strictly.  Integer labels and untyped NumPy ``HWC`` vectors fail
explicitly because neither has a canonical max/min ordering.  The Torch
``NCHW`` form treats channels as independent scalar fields.
``SLIC`` is separately Stable as ``panorai-spherical-slic/v1``.
``DoGDetector`` and the mixed ``algorithms`` gateway remain Experimental; the
direct imports of the other listed general operations retain the Stable
contract.

Torch operations
----------------

The optional, Stable ``panorai.ops.spherical.nn`` namespace provides ``Conv2d``,
``ConvTranspose2d``, ``MaxPool2d``, ``AvgPool2d``, ``AdaptiveAvgPool2d``,
``GlobalAveragePool``, ``Resize``, ``Rotate``, ``Interpolate``, ``from_planar`` and
``validate``.  It also provides ``Dilate``, ``Erode``, ``Opening`` and
``Closing`` for floating ``NCHW`` tensors. Importing the NumPy or Pipeline namespaces does not require
Torch. The Stable device contract covers CPU ``float32`` and ``float64``;
CUDA and MPS may work through Torch but remain outside the compatibility
guarantee until each has an independently reproduced matrix.

See :doc:`Spherical morphology <../concepts/spherical-morphology>` for the
geodesic footprint and algebraic contract, then run the :doc:`mask-cleaning
tutorial <../guides/tutorials/spherical_morphology>`.

Validation is deliberately fail-closed. Conventional planar spatial layers,
unknown leaves, custom containers with opaque ``forward`` semantics, and
functional spatial calls that cannot be inspected are rejected. Use
``nn.Sequential`` with declared spherical layers, or implement explicit grid
and mask transforms at the Pipeline boundary. Custom pointwise modules require
the trusted ``spherical_semantics="pointwise"`` declaration during Pipeline
registration. Forward hooks are rejected, and runtime mask-shape checks prevent
an explicitly declared module from silently changing the spatial lattice.

Mask R-CNN roles
----------------

``panorai.experimental.deep_learning.mask_rcnn`` exposes Backbone,
ProposalGenerator/RPN, RoIAlign, BoxHead, MaskHead and RoIHeads records,
``paper()`` for the executable CAM/SAM method, and the deliberately unbound
``SphericalMaskRCNN`` trainable topology.
