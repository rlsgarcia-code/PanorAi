Tutorials
=========

The tutorials follow the same workflow order as the README: define spherical
geometry, process image evidence, extract and match local features, then solve
two-view, dense, and multiview geometry. Core snippets are exercised against
the installed wheel. Illustrated image-processing and feature examples use a
checksum-pinned public CC0 panorama; no private dataset is included.

Orientation and quick start
---------------------------

Start with the capability map when choosing between sphere-native,
projection-domain, and hybrid workflows. C++ acceleration is reported as an
implementation property, not confused with the mathematical route.

.. toctree::
   :maxdepth: 1

   00_quick_start
   spherical_capability_map

Projection
----------

.. toctree::
   :maxdepth: 1

   02_projection_foundations
   01_custom_pipeline

Image processing
----------------

.. toctree::
   :maxdepth: 1

   spherical_image_processing
   09_spherical_fcn_cam
   10_spherical_monocular_depth

Detection and feature extraction
--------------------------------

.. toctree::
   :maxdepth: 1

   03_features_and_matching

Two-view geometry
-----------------

.. toctree::
   :maxdepth: 1

   04_two_view_geometry

Sparse metric landmark refinement
---------------------------------

Metric spherical bundle adjustment refines only supplied, multiply observed
landmarks. Its sparse accuracy must not be interpreted as a dense depth-map
result.

.. toctree::
   :maxdepth: 1

   11_metric_landmark_ba

Stereo dense reconstruction
---------------------------

Dense stereo assumes a known relative pose and links to the derivation of the
spherical search and numerical optimization.

.. toctree::
   :maxdepth: 1

   06_spherical_dense_stereo
   11_two_view_gaussian_depth

Multi-view geometry
-------------------

.. toctree::
   :maxdepth: 1

   05_multiview_reconstruction

Sequential tracking and mapping
-------------------------------

.. toctree::
   :maxdepth: 1

   07_spherical_slam

Benchmarks and performance
--------------------------

.. toctree::
   :maxdepth: 1

   08_benchmarking
