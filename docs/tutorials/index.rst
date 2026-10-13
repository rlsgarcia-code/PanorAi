Tutorials by application
========================

Choose the output you need first. Each tutorial begins with its input, output,
execution strategy, public API, stability, optional dependencies, and evidence
type. The :doc:`spherical_capability_map` provides the complementary view by
execution strategy.

Project and reconstruct panoramic views
---------------------------------------

.. toctree::
   :maxdepth: 1

   00_quick_start
   02_projection_foundations
   01_custom_pipeline

Process directly on the sphere
------------------------------

.. toctree::
   :maxdepth: 1

   spherical_image_processing
   12_composing_spherical_models

Extract and match visual features
---------------------------------

.. toctree::
   :maxdepth: 1

   03_features_and_matching

Estimate pose and radial range
------------------------------

.. toctree::
   :maxdepth: 1

   04_two_view_geometry
   06_spherical_dense_stereo
   10_spherical_monocular_depth

Reconstruct and navigate
------------------------

.. toctree::
   :maxdepth: 1

   05_multiview_reconstruction
   07_spherical_slam

Localize and associate semantic regions
---------------------------------------

.. toctree::
   :maxdepth: 1

   09_spherical_fcn_cam
   11_cam_guided_object_hypotheses

CAM channels are directional learned evidence. They are not presented as
instance masks or calibrated pixel probabilities; the association tutorial
shows how registered views and tracks can turn broad regions into auditable
spatial hypotheses.

Understand strategies and evidence
----------------------------------

.. toctree::
   :maxdepth: 1

   spherical_capability_map
   08_benchmarking
