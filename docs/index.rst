PanorAi documentation
=====================

PanorAi is a focused Python library for spherical image projection, sampling,
and mask-aware reconstruction. The stable 3.x center is
``panorai.geometry``: explicit pixel-center coordinates, radial range, NumPy
and optional Torch backends, and validity that never depends on pixel value.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Geometry contract
      :link: geometry-v1.html

      Coordinate frame, pixel centers, seams, depth meaning, layouts, and
      interpolation rules.

   .. grid-item-card:: Executable tutorials
      :link: tutorials/index.html

      Self-contained NumPy, container, cubemap, mask, label, depth, and Torch
      examples tested by CI.

   .. grid-item-card:: Modality guide
      :link: how_to/data_modalities.html

      Choose dtype, layout, interpolation, fill, support, and validity
      semantics deliberately.

   .. grid-item-card:: API stability
      :link: reference/stability.html

      Stable, compatibility, experimental, and frozen surfaces for 3.1.

   .. grid-item-card:: Workflow evolution
      :link: explanation/workflow-evolution.html

      A non-binding post-3.1 design for typed channels, plans, and plugins.

.. toctree::
   :hidden:
   :maxdepth: 2

   geometry-v1
   explanation/architecture
   explanation/workflow-evolution
   explanation/related_libraries
   tutorials/index
   how_to/index
   reference/index
   release-3.1.0-checklist
