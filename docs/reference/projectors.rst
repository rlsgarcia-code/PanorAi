.. ───────────────────────────────────────────────────────────────
.. 🗂  docs/reference/projectors.rst  ── cards via sphinx-design
.. ───────────────────────────────────────────────────────────────
<!-- cut:start -->

Projectors
==========

.. list-table::
   :widths: 20 80

   * - **gnomonic**
     - Projects latitude/longitude to a tangent plane. Used by default.

.. admonition:: Why it matters

   The projector determines how spherical coordinates map to a plane.
   ``gnomonic`` preserves angles near the centre but distorts size at
   the edges. Alternative projections trade distortion for area or
   distance accuracy. Pick the one that matches your algorithm's
   assumptions.

<!-- cut:end -->
