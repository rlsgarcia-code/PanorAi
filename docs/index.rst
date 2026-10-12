PanorAi documentation
=====================

PanorAi is a product-oriented Python library for convention-safe spherical
image processing and projection. Its Stable contract covers geometry,
sampling, modalities, projection, and back-projection. Experimental
research workflows compose that foundation with learned evidence, pose, depth,
object hypotheses, segmentation, and graphs. Choose a task below, then check
the stated stability and evidence limits before depending on an advanced path.

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

   .. grid-item-card:: Generate multiview object hypotheses
      :link: tutorials/11_cam_guided_object_hypotheses.html

      Convert CAM components and registered tracks into detector-style
      spherical FOV anchors, auditable cross-view associations, and compact
      view-specific consensus regions.

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
      entity lineage, archive deterministically, and query already accepted
      hypotheses. The graph is a downstream persistence layer.

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
   papers/index
   reference/index
   evidence/index
   development/index
   releases/index
