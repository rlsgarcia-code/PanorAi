PanorAi documentation
=====================

PanorAi is a focused Python library for spherical image projection and
computer vision. Choose a task below, then use the reference pages when exact
coordinates, options, or stability boundaries matter. The stable 3.x center is
``panorai.geometry``: explicit pixel-center coordinates, radial range, NumPy
and optional Torch backends, and validity that never depends on pixel value.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Learn projections and rays
      :link: tutorials/02_projection_foundations.html

      Sphere geometry, ERP, gnomonic and cubemap surfaces, samplers, blenders,
      interpolation, support, and validity.

   .. grid-item-card:: Extract and match features
      :link: tutorials/03_features_and_matching.html

      SIFT, ORB, AKAZE, BF and FLANN on panorama-aware virtual cameras, with
      real plotted keypoints and matches.

   .. grid-item-card:: Solve two-view geometry
      :link: tutorials/04_two_view_geometry.html

      Spherical Essential matrix, robust pose diagnostics, scale ambiguity,
      and educational triangulation.

   .. grid-item-card:: Reconstruct multiple panoramas
      :link: tutorials/05_multiview_reconstruction.html

      Pair graphs, tracks, camera/point positioning, spherical bundle
      adjustment, filtering, and failure handling.

   .. grid-item-card:: Run a model over views
      :link: tutorials/01_custom_pipeline.html

      Stable six-view or arbitrary-N projection, processing, and
      modality-aware reconstruction.

   .. grid-item-card:: Read the exact contract
      :link: geometry-v1.html

      Coordinate frames, seams, poles, cubemap ties, interpolation, dtype,
      layouts, and numerical semantics.

   .. grid-item-card:: Check API stability
      :link: reference/stability.html

      Stable, compatibility, Experimental, and internal surfaces for 3.x.

   .. grid-item-card:: Visual SLAM
      :link: how_to/spherical_slam.html

      Track central ERP sequences or one calibrated central fisheye lens.

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
   release-3.3.0-checklist
   release-3.2.0-checklist
   release-3.1.0-checklist
