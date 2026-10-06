Tutorials
=========

Start with projection geometry, enhance the signal with spherical image
processing, then move from local image evidence to two-view and multiview
geometry. Core contract snippets are included from one executable runner and
exercised in CI against the installed wheel. The image-processing and visual
feature examples are reproducibly generated from a checksum-pinned public CC0
panorama; no industrial or private data is used.

Start with :doc:`spherical_capability_map` to discover what PanorAi can do for
spherical computer vision. It organizes the library by end-user themes and,
inside each theme, identifies sphere-native, projection-domain, and hybrid
operations. It also treats first-party C++ acceleration as a separate axis.

The dense-stereo tutorial assumes relative pose is already available and links
to a separate derivation of the spherical search and numerical optimization.

.. toctree::
   :maxdepth: 1

   00_quick_start
   spherical_capability_map
   01_custom_pipeline
   02_projection_foundations
   spherical_image_processing
   03_features_and_matching
   04_two_view_geometry
   05_multiview_reconstruction
   06_spherical_dense_stereo
   06_spherical_slam
