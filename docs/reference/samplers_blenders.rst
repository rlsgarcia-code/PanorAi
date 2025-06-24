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

.. admonition:: Why it matters

   Choosing a sampler affects how evenly faces cover the sphere.
   ``cube`` is quick but uneven near the edges, whereas strategies
   like ``fibonacci`` or ``blue_noise`` provide smoother coverage at
   the cost of extra computation. Pick the one that balances
   efficiency with your desired detail distribution.

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

.. admonition:: Why it matters

   Blenders control how overlapping faces merge when projecting
   back to a panorama. ``average`` is fast but can blur seams,
   ``gaussian`` and ``feathering`` give smoother transitions by
   weighting centre pixels more, while robust options like
   ``huber`` resist outliers. Choose the trade-off between speed
   and visual quality that fits your task.

<!-- cut:end -->