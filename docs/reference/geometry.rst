Canonical geometry API
======================

``panorai.geometry`` is the stable mathematical API for PanorAi 3.x. Shapes,
coordinate conventions, dtype rules, and depth meaning are defined in
:doc:`../geometry-v1`.

Composable projectors
---------------------

Configured projectors accept their own result objects during back-projection;
manual extraction of ``.data`` is optional:

.. code-block:: python

   import numpy as np

   from panorai.geometry import GnomonicProjector, GnomonicSpec

   erp = np.zeros((512, 1024), dtype=np.float32)
   projector = GnomonicProjector(
       GnomonicSpec(output_shape_hw=(320, 640)), interpolation="bilinear"
   )
   view = projector.project(erp)
   restored = projector.back_project(view, output_shape_hw=erp.shape[:2])

``CubemapProjector.back_project`` likewise accepts the complete mapping
returned by ``project``. Both paths preserve NumPy/Torch backend, layout,
dtype, device, batch, channels, and input-value autograd according to the
geometry-v1 contract.

Invalid-aware bilinear sampling
-------------------------------

``invalid_policy="propagate"`` is the strict default. To interpolate only
valid floating neighbours, construct a projector with
``invalid_policy="renormalize"`` and an explicit ``min_valid_weight``, then
pass a boolean ``validity_mask`` to ``project``. Its result includes both the
output ``validity_mask`` and the sampled ``valid_weight``; passing that result
directly to ``back_project`` carries its validity mask forward. Integer labels
and boolean masks remain nearest-only.

Specification validation
------------------------

Canonical specifications are immutable and normalized. Latitude is finite and
within ``[-90, 90]`` degrees. Longitude and roll are stored in
``[-180, 180)``. Fields of view are finite and strictly within ``(0, 180)``.
Shapes contain exactly two positive integers; booleans, numeric strings, and
floating dimensions are rejected rather than coerced or truncated. An invalid
interpolation name fails when a projector is constructed.

Typing and accepted arrays
--------------------------

The distribution includes ``py.typed``. Public functions preserve their
input array type through generic ``ProjectionResult`` and
``ERPPointProjection`` annotations. NumPy accepts ``HW`` and ``HWC``
``numpy.ndarray`` values. Optional Torch accepts ``HW``, ``CHW``, and ``NCHW``
``torch.Tensor`` values. ``ArrayLike`` and ``TensorLike`` provide public
annotations without importing Torch at runtime; importing
``panorai.geometry`` therefore keeps the optional backend unloaded.

.. automodule:: panorai.geometry
   :members:
   :member-order: bysource
   :undoc-members:
