.. ───────────────────────────────────────────────────────────────
.. 🗂  docs/reference/samplers_blenders.rst  ── cards via sphinx-design
.. ───────────────────────────────────────────────────────────────
<!-- cut:start -->

Samplers
========

.. list-table::
   :widths: 20 80

   * - **cube**
     - 6 orthogonal faces.
   * - **icosahedron**
     - Vertices of an icosahedron; hierarchical density.
   * - **fibonacci**
     - Nearly uniform distribution via golden-angle spiral.
   * - **spiral**
     - Simple incremental spiral over sphere.
   * - **blue_noise**
     - Random yet well-spaced points.

.. admonition:: Why it matters

   Choosing a sampler affects how evenly faces cover the sphere.
   ``cube`` is quick but uneven near the edges, whereas strategies
   like ``fibonacci`` or ``blue_noise`` provide smoother coverage at
   the cost of extra computation. Pick the one that balances
   efficiency with your desired detail distribution.

Blenders
========

.. list-table::
   :widths: 20 80

   * - **average**
     - Uniform mean of overlaps.
   * - **gaussian**
     - Distance-based Gaussian weights.
   * - **feathering**
     - Smooth cosine fall-off (feather).
   * - **closest**
     - Pixel from nearest face centre.
   * - **huber**
     - Robust mean that down-weights outliers.

.. admonition:: Why it matters

   Blenders control how overlapping faces merge when projecting
   back to a panorama. ``average`` is fast but can blur seams,
   ``gaussian`` and ``feathering`` give smoother transitions by
   weighting centre pixels more, while robust options like
   ``huber`` resist outliers. Choose the trade-off between speed
   and visual quality that fits your task.

<!-- cut:end -->