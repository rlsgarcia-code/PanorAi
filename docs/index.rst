PanorAi documentation
=====================

PanorAi is a product-oriented Python library for convention-safe spherical
computer vision. Its six capability families connect geometry and visual
evidence to pose, depth, mapping, semantics, and an Experimental replayable
spatial-semantic graph. Choose a task below, then use the reference pages when
exact coordinates, options, or stability boundaries matter.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Learn projections and rays
      :link: tutorials/02_projection_foundations.html

      Sphere geometry, ERP, gnomonic and cubemap surfaces, samplers, blenders,
      interpolation, support, and validity.

   .. grid-item-card:: Explore spherical computer vision
      :link: tutorials/spherical_capability_map.html

      Start from projection, image processing, features and matching, two-view
      geometry, multiview reconstruction, or SLAM; then see how each workflow
      combines sphere-native and projection-domain operations.

   .. grid-item-card:: Extract and match features
      :link: tutorials/03_features_and_matching.html

      SIFT, ORB, AKAZE, BF and FLANN on panorama-aware virtual cameras, with
      real plotted keypoints and matches.

   .. grid-item-card:: Filter and enhance panoramas
      :link: tutorials/spherical_image_processing.html

      Illustrated tangent convolution, smoothing, gradients, Canny, pyramids,
      rotation, resizing, and solid-angle histogram equalization.

   .. grid-item-card:: Port pretrained CNNs to the sphere
      :link: tutorials/09_spherical_fcn_cam.html

      Experimental AlexNet, VGG16, and ResNet18 FCN/CAM inference with exact
      weight reuse, automatic external checkpoint caching, and explicit limits.

   .. grid-item-card:: Estimate monocular spherical depth
      :link: tutorials/10_spherical_monocular_depth.html

      Experimental resize-free Metric3D CNN acquisition and spherical port,
      with checksum provenance, radial range, and no bundled weights.

   .. grid-item-card:: Solve two-view geometry
      :link: tutorials/04_two_view_geometry.html

      Spherical Essential matrix, robust pose diagnostics, scale ambiguity,
      and educational triangulation.

   .. grid-item-card:: Estimate spherical dense range
      :link: tutorials/06_spherical_dense_stereo.html

      Direct ERP plane sweep constrained by a two-view pose, with radial-range,
      confidence, validity, and bidirectional consistency.

   .. grid-item-card:: Reconstruct multiple panoramas
      :link: tutorials/05_multiview_reconstruction.html

      Pair graphs, tracks, camera/point positioning, spherical bundle
      adjustment, filtering, and failure handling.

   .. grid-item-card:: Understand dense stereo internals
      :link: explanation/spherical_dense_stereo.html

      Derive the spherical epipolar search, inverse-range cost volume,
      four-path optimization, refinement, rejection, and scaling behavior.

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

   .. grid-item-card:: Build a spatial-semantic graph
      :link: how_to/spatial_semantic_graph.html

      Associate prepared observations, preserve spatial uncertainty, maintain
      entity lineage, archive deterministically, and query with evidence.

   .. grid-item-card:: Visual SLAM
      :link: tutorials/07_spherical_slam.html

      Learn the spherical frontend and tracking route, then continue to the
      operational how-to and API reference.

   .. grid-item-card:: Benchmarks and performance
      :link: benchmarks.html

      Installed-artifact timings, public-dataset accuracy, evidence limits,
      and a reproducible benchmarking protocol.

.. toctree::
   :hidden:
   :maxdepth: 2

   start/index
   capabilities/index
   concepts/index
   reference/index
   evidence/index
   development/index
   releases/index
