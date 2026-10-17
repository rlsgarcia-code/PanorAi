Geometry: control rays and projections
======================================

Choose Geometry when the application must own coordinates, view placement,
projection shape, FOV, interpolation or back-projection. ``panorai.geometry``
is the mathematical source of truth used by every higher level.

1. Convert ERP pixels to bearings
---------------------------------

ERP pixel coordinates address pixel centres. Conversion returns unit bearings
in the canonical panorama frame; the inverse returns pixels plus explicit
projection information.

.. literalinclude:: ../../../scripts/run_documentation_examples.py
   :language: python
   :start-after: DOCS_LEVEL_GEOMETRY_START = None
   :end-before: DOCS_LEVEL_GEOMETRY_END = None
   :dedent: 4

2. Make a virtual camera explicit
---------------------------------

``GnomonicSpec`` fixes centre longitude/latitude, horizontal and vertical FOV,
roll and output shape. ``GnomonicProjector`` adds interpolation and reusable
projection planning. The returned ``ProjectionResult`` keeps data and support
together for natural back-projection.

3. Select interpolation by modality
------------------------------------

Floating RGB and continuous fields may use bilinear or nearest sampling.
Integer labels and boolean masks require nearest sampling. Radial-range depth
keeps validity separate from numeric zero and may use strict propagation or an
explicit valid-weight renormalization policy.

4. Respect seams and poles
--------------------------

Longitude wraps horizontally in ``[-pi, pi)``. Exact poles lie on raster
boundaries rather than pixel centres and have no unique longitude. Cubemap
face order, tie rules and gnomonic roll/FOV conventions are fixed by
``geometry-v1`` rather than by an application wrapper.

Read :doc:`The geometry contract <../../concepts/geometry-contract>` for the
mental model, :doc:`Geometry API <../../reference/geometry>` for signatures and
:doc:`the normative specification <../../geometry-v1>` for exact formulas and
boundary behavior. The long-form :doc:`projection tutorial
<../tutorials/02_projection_foundations>` continues into samplers and blenders.
