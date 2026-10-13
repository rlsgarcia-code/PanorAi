PanorAi documentation
=====================

PanorAi provides geometry-safe spherical image processing and projection for
computer-vision applications. Its applications share one foundation:
canonical projection geometry, composable spherical operations, and typed
bridges between ERP fields, projected views and spherical observations.

Build from the core or use a wrapper
------------------------------------

Use :doc:`geometry-v1` for coordinate and projection mathematics. Use
:doc:`reference/core_operations` to compose a new sphere-native NumPy or Torch
method. Use the typed :doc:`explanation/workflow-evolution` boundary to carry
fields, points, regions and view evidence between methods and representations.

Application wrappers package a tested composition, defaults and diagnostics.
They still return the same explicit fields, bearings, poses, range maps or
hypotheses, so using a wrapper does not create a second geometry stack.

Choose an application
---------------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Process and project panoramic images
      :link: applications/projection_processing/index
      :link-type: doc

      Reuse pinhole methods through sampling and fusion, then choose the
      correct spherical output contract.

   .. grid-item-card:: Extract and match visual features
      :link: applications/features_matching/index
      :link-type: doc

      Start with the Stable projected route, then compare direct spherical
      detection and matching diagnostics.

   .. grid-item-card:: Estimate pose and radial range
      :link: applications/pose_range/index
      :link-type: doc

      Separate relative pose, metric scale, stereo, sphere-native monocular
      CNNs and tangent-view depth models.

   .. grid-item-card:: Localize and associate semantic regions
      :link: applications/semantic_regions/index
      :link-type: doc

      Progress from a pretrained spherical CAM to angular proposals,
      cross-view association and optional mask refinement.

How PanorAi executes spherical computer vision
----------------------------------------------

Use the :doc:`tutorials/spherical_capability_map` to compare sphere-native,
projection-domain, and hybrid execution without introducing a second API
hierarchy. Then use the exact :doc:`geometry-v1`, generated
:doc:`reference/stability`, and :doc:`evidence/index` contracts to decide what
is appropriate for production or research.

.. grid:: 1 1 3 3
   :gutter: 2

   .. grid-item-card:: Start quickly
      :link: start/index
      :link-type: doc

      Install the package, project a first view, and follow runnable examples.

   .. grid-item-card:: Browse all tutorials
      :link: tutorials/index
      :link-type: doc

      The same six application groups, followed by advanced and explanatory
      material.

   .. grid-item-card:: Reproduce the evidence
      :link: benchmarks
      :link-type: doc

      Public inputs, fixed protocols, limitations, and installed-artifact
      measurements.

.. toctree::
   :hidden:
   :maxdepth: 2

   start/index
   applications/index
   capabilities/index
   concepts/index
   papers/index
   reference/index
   evidence/index
   development/index
   releases/index
