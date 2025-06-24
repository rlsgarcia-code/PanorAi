.. ───────────────────────────────────────────────────────────────
.. 🗂  docs/reference/samplers_blenders.rst  ── cards via sphinx-design
.. ───────────────────────────────────────────────────────────────
<!-- cut:start -->

Samplers
========

.. grid:: 2 2 3 3
   :gutter: 1

   .. grid-item-card:: **cube**
      :class-card: sd-rounded-md

      6 orthogonal faces.

   .. grid-item-card:: **icosahedron**
      Vertices of an icosahedron; hierarchical density.

   .. grid-item-card:: **fibonacci**
      Nearly uniform distribution via golden-angle spiral.

   .. grid-item-card:: **spiral**
      Simple incremental spiral over sphere.

   .. grid-item-card:: **blue_noise**
      Random yet well-spaced points.

Blenders
========

.. grid:: 2 2 3 3
   :gutter: 1

   .. grid-item-card:: **average**
      Uniform mean of overlaps.

   .. grid-item-card:: **gaussian**
      Distance-based Gaussian weights.

   .. grid-item-card:: **feathering**
      Smooth cosine fall-off (feather).

   .. grid-item-card:: **closest**
      Pixel from nearest face centre.

   .. grid-item-card:: **huber**
      Robust mean that down-weights outliers.

<!-- cut:end -->