.. ───────────────────────────────────────────────────────────────
.. 🗂  docs/reference/projectors.rst  ── cards via sphinx-design
.. ───────────────────────────────────────────────────────────────
<!-- cut:start -->

Projectors
==========

.. grid:: 1 1 2 2
   :gutter: 1

   .. grid-item-card:: **gnomonic**
      :class-card: sd-rounded-md

      Projects latitude/longitude to a tangent plane. Used by default.

.. admonition:: Why it matters

   The projector determines how spherical coordinates map to a plane.
   ``gnomonic`` preserves angles near the centre but distorts size at
   the edges. Alternative projections trade distortion for area or
   distance accuracy. Pick the one that matches your algorithm's
   assumptions.

<!-- cut:end -->
