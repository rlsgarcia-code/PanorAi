.. ───────────────────────────────────────────────────────────────
.. 🗂  docs/reference/samplers_blenders.rst  ── dropdown tables
.. ───────────────────────────────────────────────────────────────
<!-- cut:start -->

Samplers
========


.. dropdown:: Available samplers

   .. list-table::
      :widths: 20 80
      :header-rows: 1

      * - Name
        - Description
      * - ``cube``
        - 6 orthogonal faces.
      * - ``icosahedron``
        - Vertices of an icosahedron; hierarchical density.
      * - ``fibonacci``
        - Nearly uniform distribution via golden-angle spiral.
      * - ``spiral``
        - Simple incremental spiral over sphere.
      * - ``blue_noise``
        - Random yet well-spaced points.


Blenders
========

.. dropdown:: Available blenders

   .. list-table::
      :widths: 20 80
      :header-rows: 1

      * - Name
        - Description
      * - ``average``
        - Uniform mean of overlaps.
      * - ``gaussian``
        - Distance-based Gaussian weights.
      * - ``feathering``
        - Smooth cosine fall-off (feather).
      * - ``closest``
        - Pixel from nearest face centre.
      * - ``huber``
        - Robust mean that down-weights outliers.

<!-- cut:end -->
