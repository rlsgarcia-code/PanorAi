Clean spherical masks with morphology
=====================================

This tutorial closes a small hole in a binary region that crosses the ERP
longitude seam.  The structuring radius is four degrees on the sphere, not
four image pixels.

.. literalinclude:: ../../../scripts/run_documentation_examples.py
   :language: python
   :start-after: DOCS_SPHERICAL_MORPHOLOGY_START = None
   :end-before: DOCS_SPHERICAL_MORPHOLOGY_END = None
   :dedent: 4

Choose the operation
--------------------

``Dilate`` expands bright/true regions and ``Erode`` shrinks them. ``Opening``
removes isolated bright regions; ``Closing`` fills small dark holes and joins
nearby regions.  All four use the same ``GeodesicDisk`` contract and preserve
the canonical ERP grid.

The configured object is reusable and safely serializable inside a Pipeline.
The equivalent direct functions are ``dilate``, ``erode``, ``opening`` and
``closing``.

Compose a Torch model
---------------------

With the optional Torch extra, the same roles are available as
``panorai.ops.spherical.nn.Dilate``, ``Erode``, ``Opening`` and ``Closing``.
They accept floating ``NCHW`` tensors, preserve the spatial grid, propagate
autograd to selected extrema, and compose in ``torch.nn.Sequential``.  The
Stable device contract is CPU ``float32``/``float64``; this does not imply a
CUDA or MPS guarantee.

Read :doc:`the concept <../../concepts/spherical-morphology>` before choosing
a radius or sampling step, especially when comparing multiple ERP
resolutions.
