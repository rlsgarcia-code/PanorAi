Guides
======

Learn the API progressively first, then choose a computer-vision outcome. The
four API levels explain increasing control; capability guides show how those
levels combine in complete workflows.

For lookup rather than a walkthrough, open the :doc:`component catalog
<../reference/catalog/index>` and choose an Application, Stage, operation,
estimator, typed result or Geometry primitive by role.

Learn by API level
------------------

.. grid:: 1 2 2 4
   :gutter: 2

   .. grid-item-card:: Applications
      :link: api-levels/applications
      :link-type: doc

      Run, reconfigure, inspect and persist a task wrapper.

   .. grid-item-card:: Pipeline
      :link: api-levels/pipeline
      :link-type: doc

      Compose direct-sphere and projected-view stages.

   .. grid-item-card:: Spherical Ops
      :link: api-levels/spherical_ops
      :link-type: doc

      Use functions, immutable operations and Torch layers.

   .. grid-item-card:: Geometry
      :link: api-levels/geometry
      :link-type: doc

      Control pixels, rays, views, interpolation and masks.

.. toctree::
   :maxdepth: 2

   api-levels/index

Build by outcome
----------------

These guides start from a result and connect the relevant API levels,
mathematical explanation, extensions and evidence.

.. toctree::
   :maxdepth: 2

   projection_processing/index
   features_matching/index
   pose_range/index
   semantic_regions/index
   composed_spherical_api
   tutorials/index
   how-to/index

.. toctree::
   :hidden:

   projection_processing/pinhole_pipeline
   projection_processing/sampling_and_fusion
   projection_processing/outputs_and_observations
   features_matching/quickstart
   features_matching/projected_features
   features_matching/sphere_native_features
   features_matching/matching_and_diagnostics
   pose_range/relative_pose
   pose_range/metric_scale_and_triangulation
   pose_range/panoramic_dense_stereo
   pose_range/dense_stereo
   pose_range/classical_stereo
   pose_range/monocular_depth
   pose_range/metric3d_sphere_native
   pose_range/depth_anything_views
   pose_range/validate_range
   semantic_regions/spherical_cam
   semantic_regions/interpreting_cam
   semantic_regions/angular_proposals
   semantic_regions/multiview_association
   semantic_regions/mask_refinement
   semantic_regions/evaluation
