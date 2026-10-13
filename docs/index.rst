PanorAi documentation
=====================

PanorAi provides geometry-safe spherical image processing and projection for
computer-vision applications. Start from the result you need; each route names
its execution strategy, public entry point, stability, and evidence type.

Choose an application
---------------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Project and reconstruct panoramic views
      :link: tutorials/02_projection_foundations
      :link-type: doc

      **Projection-domain · Stable.** Sample explicit views, preserve support,
      reconstruct an ERP, and measure round-trip error.

   .. grid-item-card:: Process directly on the sphere
      :link: tutorials/spherical_image_processing
      :link-type: doc

      **Sphere-native · Experimental.** Apply constant-angle filters and
      seam-aware edge operators directly to an ERP.

   .. grid-item-card:: Extract and match visual features
      :link: tutorials/03_features_and_matching
      :link-type: doc

      **Projection-domain + spherical mapping · Stable core.** Detect in local
      views and return ERP pixels, bearings, and descriptor matches.

   .. grid-item-card:: Estimate pose and radial range
      :link: tutorials/06_spherical_dense_stereo
      :link-type: doc

      **Sphere-native/Hybrid · Experimental.** Relate registered bearings and
      recover radial range with explicit confidence, support, and error.

   .. grid-item-card:: Reconstruct and navigate
      :link: tutorials/05_multiview_reconstruction
      :link-type: doc

      **Hybrid · Experimental.** Build tracks, poses, sparse structure, and
      navigation state with explicit failure and provenance information.

   .. grid-item-card:: Localize and associate semantic regions
      :link: tutorials/09_spherical_fcn_cam
      :link-type: doc

      **Sphere-native evidence + hybrid association · Experimental.** Use
      learned directional evidence to propose regions and test shared
      hypotheses across registered views. CAM is not a segmentation mask.

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
   capabilities/index
   concepts/index
   papers/index
   reference/index
   evidence/index
   development/index
   releases/index
