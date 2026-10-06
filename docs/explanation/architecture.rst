Architecture
============

PanorAi separates mathematical geometry from compatibility containers and
application-specific processing.

The user-facing spherical computer-vision guide is maintained in
:doc:`../tutorials/spherical_capability_map`. It starts from what users can do
across projection, image processing, features and matching, two-view geometry,
multiview reconstruction, and SLAM. Within each theme it identifies
sphere-native, projection-domain, and hybrid routes, independently from C++
acceleration.

.. mermaid::

   flowchart LR
      A[ERP array] --> B[Canonical geometry]
      B --> C[Projected data]
      B --> D[Geometric support mask]
      C --> E[User processing]
      D --> F[Mask-aware reconstruction]
      E --> F
      F --> G[ERP result plus support]

Canonical geometry
------------------

``panorai.geometry`` owns the coordinate frame, projection formulas, sampling,
layouts, dtype rules, and support masks. Functional calls and immutable
projector objects delegate to one engine so NumPy, Torch, cubemap, and
gnomonic paths do not redefine the mathematics independently.

Compatibility containers
------------------------

``EquirectangularImage``, ``GnomonicFace``, and ``GnomonicFaceSet`` retain the
3.0 object-oriented workflow. In 3.1 they can accept canonical geometry specs,
but they remain compatibility APIs rather than the source of new geometry
semantics.

Validity and blending
---------------------

Projection returns data together with geometric support. A valid black RGB
pixel, label zero, or radial range zero remains valid when its mask is true.
An unsupported non-zero value remains unsupported. Blenders consume those
explicit masks and never infer validity from numeric values.

Scope boundary
--------------

Training pipelines, datasets, checkpoints, research metrics, and vendored
model implementations are outside the projection core and are excluded from
the 3.1 wheel and sdist. ``panorai.depth`` contains only lazy compatibility
adapters to separately installed upstream projects; PCD remains an optional
compatibility surface. See
:doc:`../reference/stability` for the supported surface and
:doc:`../geometry-v1` for the mathematical contract.
