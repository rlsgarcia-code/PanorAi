Spherical Ops: control the operation
====================================

Choose Spherical Ops when the computation itself must be sphere-aware. These
operations work on the ERP lattice using longitude wrapping, tangent-frame
neighbourhoods, explicit masks and grid transforms rather than treating the ERP
as an ordinary flat image.

.. panorai-api-components:: operation

.. panorai-api-components:: torch-layer

1. Start with a function
------------------------

Functional calls are concise for one operation. Configured immutable objects
carry the same parameters into Pipeline and safe topology persistence.

.. literalinclude:: ../../../scripts/run_documentation_examples.py
   :language: python
   :start-after: DOCS_LEVEL_OPS_START = None
   :end-before: DOCS_LEVEL_OPS_END = None
   :dedent: 4

``sobel(..., direction="east")`` differentiates in the local tangent frame,
not along a planar image axis. ``area_average`` weights the ERP by spherical
surface area rather than raw pixel count.

2. Put an operation in a Pipeline
---------------------------------

Pass an immutable operation such as ``GaussianBlur`` to ``OnSphere``. The
operation transforms values, support, validity and ``RasterGrid`` together.
Shape-changing operations record anchor and stride history; a reduced lattice
cannot masquerade as canonical ERP without explicit regridding.

For binary-mask cleanup, compose ``GeodesicDisk`` with ``Dilate``, ``Erode``,
``Opening`` or ``Closing``.  Their radius is a great-circle angle and the
finite neighbour relation preserves the standard morphology laws.  Continue
with the :doc:`spherical morphology tutorial
<../tutorials/spherical_morphology>`.

3. Build Torch networks with spherical layers
----------------------------------------------

``panorai.ops.spherical.nn`` provides convolution, transposed convolution,
pooling, resize and interpolation for ``torch.nn.Sequential``. Run
``validate(model, input_shape=...)`` before execution: planar spatial layers,
opaque containers and unverifiable functional spatial calls fail closed.

The Stable device contract covers CPU ``float32`` and ``float64``. CUDA and
MPS may work through Torch but are not yet compatibility guarantees. Learned
convolution propagates validity strictly; arbitrary learned weights are never
renormalized as though they were an averaging kernel.

4. Distinguish general operations from algorithms
-------------------------------------------------

General filters, transforms and reductions are Stable. SLIC has its own Stable
contract. DoG and the mixed ``algorithms`` gateway remain Experimental. Always
check the imported symbol's stability instead of inferring it from the module.

Read :doc:`Sphere-aware operations <../../concepts/spherical-operations>` for
the mathematical distinction from planar ERP processing and
:doc:`Operations API <../../reference/spherical_ops>` for the full inventory.
The detailed :doc:`image-processing tutorial
<../tutorials/spherical_image_processing>` includes Canny, pyramids and
equal-area histogram processing.
