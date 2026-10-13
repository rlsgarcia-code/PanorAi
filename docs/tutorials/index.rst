Detailed tutorials
==================

Start with the progressive :doc:`../applications/index` paths when solving an
application. This section retains the deeper tutorials, derivations, complete
configuration references and reproducibility protocols to which those paths
lead.

Use :doc:`12_composing_spherical_models` when inventing a new sphere-native
method. The :doc:`spherical_capability_map` provides the complementary view by
execution strategy.

Compose a new sphere-native method
----------------------------------

.. toctree::
   :maxdepth: 1

   12_composing_spherical_models

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
