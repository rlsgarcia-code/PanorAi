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

NumPy operations
----------------

``panorai.ops.spherical.functional`` contains direct functions. Matching
immutable objects are exported from ``panorai.ops.spherical``. Algorithms are
``SLIC``, ``DoGDetector``, ``SphericalNMS``, ``Canny``, ``GaussianPyramid`` and
``LaplacianPyramid``.

Torch operations
----------------

The optional ``panorai.ops.spherical.nn`` namespace provides ``Conv2d``,
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
