Spherical operations and composition
====================================

Applications
------------

``panorai.applications`` exposes ``Application``, ``PipelineApplication``,
``SphericalSLIC``, ``SphericalDoGFeatures``, ``SphericalStereoDepth``, and
``SphericalMaskRCNNPaper``. These public wrappers are Experimental; see
:doc:`../guides/composed_spherical_api` for inputs, outputs and promotion
blockers.

Pipeline
--------

``panorai.pipeline`` exposes ``Pipeline`` and the public ``Stage`` protocol;
execution records ``StageResult`` and ``PipelineRun``; typed ``RasterGrid`` and
``RasterField`` values; non-raster results; and the ``OnSphere``,
``SampleViews``, ``PerView``, ``ReconstructERP``, ``Parallel``, ``Select``,
``Attach``, ``LiftToBearings``, and ``AggregateViews`` stages.
Registered optional-Torch architectures use ``register_torch_module`` and
``registered_torch_modules``; their archives store weights with
safetensors rather than pickle.

This generic core is Stable as ``panorai-spherical-pipeline/v1``. CAM/SAM
proposal and instance-fusion stages exported from the same namespace remain
Experimental.

NumPy operations
----------------

``panorai.ops.spherical.functional`` contains direct functions. Matching
immutable objects are exported from ``panorai.ops.spherical``. Algorithms are
``SLIC``, ``DoGDetector``, ``SphericalNMS``, ``Canny``, ``GaussianPyramid`` and
``LaplacianPyramid``.

The general NumPy operations are Stable as ``panorai-spherical-ops/v1``.
``SLIC``, ``DoGDetector`` and the mixed ``algorithms`` gateway remain
Experimental; the direct imports of the other listed general operations retain
the Stable contract.

Torch operations
----------------

The optional, Experimental ``panorai.ops.spherical.nn`` namespace provides ``Conv2d``,
``ConvTranspose2d``, ``MaxPool2d``, ``AvgPool2d``, ``AdaptiveAvgPool2d``,
``GlobalAveragePool``, ``Resize``, ``Interpolate``, ``from_planar`` and
``validate``. Importing the NumPy or Pipeline namespaces does not require
Torch.

Mask R-CNN roles
----------------

``panorai.experimental.deep_learning.mask_rcnn`` exposes Backbone,
ProposalGenerator/RPN, RoIAlign, BoxHead, MaskHead and RoIHeads records,
``paper()`` for the executable CAM/SAM method, and the deliberately unbound
``SphericalMaskRCNN`` trainable topology.
