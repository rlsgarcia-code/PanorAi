PanorAi documentation
=====================

PanorAi provides geometry-safe spherical image processing and projection for
computer-vision applications. Start from the result you need; each route names
its execution strategy, public entry point, stability, and evidence type.

Choose an application
---------------------

.. image:: _static/tutorials/application-strategy-routes.svg
   :alt: Projection-domain, sphere-native, and hybrid multiview routes for spherical computer vision
   :width: 100%

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Project and reconstruct panoramic views
      :link: tutorials/02_projection_foundations.html
      :img-top: _static/tutorials/application-projection-sundowner.jpg

      **Projection-domain · Stable · Real output.** ERP footprints, projected
      faces, explicit support, reconstruction, and round-trip error.

   .. grid-item-card:: Process directly on the sphere
      :link: tutorials/spherical_image_processing.html
      :img-top: _static/tutorials/application-processing-boma.jpg

      **Sphere-native · Experimental · Real output.** Constant-angle Gaussian
      blur and seam-aware geodesic Canny on one ERP.

   .. grid-item-card:: Extract and match visual features
      :link: tutorials/03_features_and_matching.html
      :img-top: _static/tutorials/application-features-mosaic-tunnel.jpg

      **Projection-domain + spherical mapping · Stable core · Real output.**
      Mosaic Tunnel descriptors mapped back to ERP pixels and unit bearings.

   .. grid-item-card:: Estimate pose and radial range
      :link: tutorials/06_spherical_dense_stereo.html
      :img-top: _static/tutorials/application-pose-range.jpg

      **Sphere-native/Hybrid · Experimental · Method + analytic fixture.**
      Two-view bearing geometry beside estimated radial range and error.

   .. grid-item-card:: Reconstruct and navigate
      :link: tutorials/05_multiview_reconstruction.html
      :img-top: _static/tutorials/reconstruction-graph.svg

      **Hybrid · Experimental · Method diagram.** Tracks, poses, sparse points,
      spherical bundle adjustment, failure handling, and SLAM.

   .. grid-item-card:: Localize and associate semantic regions
      :link: tutorials/09_spherical_fcn_cam.html
      :img-top: _static/tutorials/application-semantics-boiler-room.jpg

      **Sphere-native evidence + hybrid association · Experimental · Real CAM
      + method diagram.** A CAM proposes a region; registered views can support
      one shared spatial hypothesis. It is not a segmentation mask.

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
      :link: start/index.html

      Install the package, project a first view, and follow runnable examples.

   .. grid-item-card:: Browse all tutorials
      :link: tutorials/index.html

      The same six application groups, followed by advanced and explanatory
      material.

   .. grid-item-card:: Reproduce the evidence
      :link: benchmarks.html

      Public inputs, fixed protocols, limitations, and installed-artifact
      measurements.

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
