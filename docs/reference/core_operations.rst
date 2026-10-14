Composable spherical operations
===============================

``panorai.core`` is the small vocabulary from which sphere-native image
processing and learned models can be composed. It separates three concerns:

* immutable mathematical specifications in ``panorai.core``;
* explicitly selected implementations in ``panorai.core.numpy`` and
  ``panorai.core.torch``;
* the optional ERP-container bridge in ``panorai.workflow``.

Importing ``panorai.core`` or ``panorai.core.numpy`` does not import Torch.
The two backends use the same ERP pixel-centre anchors, tangent basis,
exponential-map offsets, longitude wrap, and solid-angle reductions. NumPy is
required. Torch is optional and preserves autograd.

The compositional API is Experimental in the 3.x series. Existing
``panorai.geometry``, ``panorai.image_processing``, and object-workflow imports
remain available. Fixed channel-wise filters and general channel-mixing NumPy
convolution can use the exact native C++17 backend. ``backend="auto"`` selects
it for compatible float32/float64 input when the extension is installed;
``backend="numpy"`` preserves the reference and ``backend="native"`` fails
explicitly rather than silently falling back.

Fundamental vocabulary
----------------------

Neighbourhood and learned operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``SphericalFilter2d``, ``SphericalConv2d``,
``SphericalConvTranspose2d``, ``SphericalGaussianBlur2d``,
``SphericalMaxPool2d``, and ``SphericalAvgPool2d`` sample a local tangent
neighbourhood around ERP lattice anchors. Pointwise activation, normalization,
addition, subtraction, concatenation, and branching remain ordinary NumPy or
Torch operations.

Grid, reduction, and spatial operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``SphericalResize``, ``SphericalAreaSum``, ``SphericalAreaMean``,
``SphericalGradient``, ``SphericalLaplacian``,
``SphericalNeighborhoodMax``, ``SphericalNeighborhoodMin``, and
``SphericalScaleExtrema`` cover the reusable geometric work. Area reductions
integrate exact ERP pixel-cell solid angles rather than treating rows as equal
area.

NumPy composition
-----------------

NumPy modules accept ``CHW`` or ``NCHW`` floating arrays. They can be combined
without an application-specific pipeline class:

.. code-block:: python

   import numpy as np
   from panorai.core.numpy import (
       Sequential,
       SphericalGaussianBlur2d,
       SphericalLaplacian,
   )

   model = Sequential(
       SphericalGaussianBlur2d(1.2),
       SphericalLaplacian(),
   )
   response_nchw = model(np.zeros((1, 1, 128, 256), np.float32))

Fixed filters may request the native implementation explicitly:

.. code-block:: python

   from panorai.core import SphericalExecutionSpec
   from panorai.core.numpy import SphericalFilter2d

   scharr = SphericalFilter2d(
       np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], np.float32) / 32,
       execution=SphericalExecutionSpec(backend="native"),
   )

The same execution contract applies to learned channel-mixing weights,
including grouped and depthwise convolution:

.. code-block:: python

   from panorai.core.numpy import SphericalConv2d

   convolution = SphericalConv2d(
       np.zeros((64, 32, 3, 3), np.float32),
       padding=1,
       execution=SphericalExecutionSpec(backend="auto"),
   )
   features = convolution(np.zeros((1, 32, 128, 256), np.float32))

The fused implementation computes each tangent sampling location once per
output row and tap, reuses it across channels, and bounds auxiliary storage to
per-worker row metadata and one local sampled vector. It does not materialize
the global ``N x C x K x Hout x Wout`` reference tensor. The reproducible
NumPy/native comparison is in
``benchmarks/core_operations/benchmark_spherical_conv.py``.

Torch composition
-----------------

Torch layers follow the normal ``nn.Module`` grammar. A layer can be created
directly or can share the exact ``Parameter`` objects of a pretrained planar
layer:

.. code-block:: python

   import torch
   from torch import nn
   from panorai.core.torch import SphericalConv2d, SphericalMaxPool2d

   encoder = nn.Sequential(
       SphericalConv2d(
           in_channels=3,
           out_channels=32,
           kernel_size=3,
           padding=1,
       ),
       nn.ReLU(),
       SphericalMaxPool2d(kernel_size=2, stride=2),
   )
   features = encoder(torch.zeros(1, 3, 128, 256))

   planar = nn.Conv2d(32, 64, 3, padding=1)
   spherical = SphericalConv2d.from_torch(planar, share_parameters=True)
   assert spherical.weight is planar.weight

Container bridge
----------------

``SphericalImageModule`` is deliberately backend-specific. The user writes a
normal ``forward`` method and may call either ``model(array)`` or
``model.predict(panorama)``. The latter interprets declared inputs and outputs,
then returns an immutable ``SphericalOutputTree`` of
``EquirectangularImage`` containers.

.. code-block:: python

   from panorai.core.numpy import SphericalGaussianBlur2d
   from panorai.workflow import SphericalInputSpec, SphericalOutputSpec
   from panorai.workflow.numpy import SphericalImageModule

   class DifferenceOfGaussians(SphericalImageModule):
       def __init__(self):
           super().__init__(
               input_spec=SphericalInputSpec(
                   field="image", grayscale=True, scale=1 / 255
               ),
               output_specs={
                   "dog": SphericalOutputSpec.continuous(
                       component_ids=("sigma_2_minus_sigma_1",)
                   )
               },
           )
           self.small = SphericalGaussianBlur2d(1.0)
           self.large = SphericalGaussianBlur2d(2.0)

       def forward(self, image):
           return {"dog": self.large(image) - self.small(image)}

   output = DifferenceOfGaussians().predict(panorama)
   dog_hw = output["dog"].field("dog")

Outputs may be nested dictionaries and may have different resolutions. Each
leaf has an explicit ``SphericalOutputSpec``. The default
``support_policy="require_full"`` rejects partially supported or invalid
inputs rather than silently convolving invalid samples. Advanced models may
choose ``support_policy="explicit"`` and return ``RasterFieldResult`` leaves
with explicit validity and support masks.

Application composition
-----------------------

The same primitives support several applications without introducing one
framework per application:

* DoG and scale-space features: Gaussian layers, subtraction, resize, then
  ``SphericalScaleExtrema``;
* CNN/CAM: port or construct convolution/pooling layers, retain ordinary Torch
  activations, then use ``SphericalAreaMean`` or dense class channels;
* segmentation and depth: compose an encoder/decoder and declare categorical
  or continuous output fields;
* dense matching: create multiscale feature tensors and lift selected outputs
  through the observation workflow;
* classical filtering: use fixed kernels through ``SphericalFilter2d`` and
  the native backend where available.

``EquirectangularImage.apply_spherical`` remains useful for one-off
array-to-array functions. A composed model should instead be called directly;
it needs ``predict`` only when entering or leaving the typed ERP container
boundary.

Core versus application wrappers
--------------------------------

The core is the right level when defining a new neighbourhood, scale-space,
CNN block or reduction. An application wrapper is the right level when the
method also requires calibrated defaults, provenance, robust estimation,
state, failure classification or several typed outputs. The wrapper should be
understood as a tested composition of core representations, not as an
alternative mathematical stack.

Examples include ``SphericalFeaturePipeline`` over projection and bearing
lifting, ``SphericalRelativePoseEstimator`` over matched bearings and robust
spherical residuals, and ``estimate_spherical_range`` over pose-driven ray
warping and matching-cost reduction. Their tutorials include a “How this
application is composed” section and then lead with the wrapper that users
should normally call.
