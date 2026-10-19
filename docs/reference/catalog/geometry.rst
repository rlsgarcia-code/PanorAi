Geometry catalog
================

``panorai.geometry`` is the Stable mathematical foundation.  Use it directly
when you need explicit pixels, rays, projection specs or back-projection.
Higher-level objects, Stages and operations delegate to the same
``geometry-v1`` convention.

Registered Geometry surfaces
----------------------------

.. panorai-api-components:: geometry

Coordinate conversions
----------------------

.. list-table:: Stable functional geometry
   :header-rows: 1
   :widths: 31 27 42

   * - Function
     - Transformation
     - Result
   * - ``erp_pixels_to_rays``
     - continuous ERP pixels → unit rays
     - array preserving supported backend/layout semantics
   * - ``rays_to_erp_pixels``
     - unit rays → continuous ERP pixels
     - ``ERPPointProjection`` with support/validity information
   * - ``gnomonic_pixels_to_rays``
     - face pixels → unit rays
     - rays in the declared face orientation
   * - ``rays_to_gnomonic_pixels``
     - unit rays → face pixels
     - ``GnomonicRayProjection``
   * - ``equirectangular_to_gnomonic``
     - ERP raster → one tangent view
     - ``ProjectionResult`` with support and validity
   * - ``gnomonic_to_equirectangular``
     - tangent view → ERP contribution
     - values and masks on the requested ERP grid
   * - ``equirectangular_to_cubemap``
     - ERP → six canonical faces
     - face mapping in ``CUBE_FACE_ORDER``
   * - ``cubemap_to_equirectangular``
     - six faces → ERP
     - fused ERP under the declared sampling policy

Configured projectors and specs
-------------------------------

.. list-table:: Stable geometry objects
   :header-rows: 1
   :widths: 28 25 47

   * - Component
     - Configures or returns
     - Responsibility
   * - ``GnomonicSpec``
     - one tangent projection
     - Shape, longitude, latitude, roll and field of view with normalized validation.
   * - ``CubemapSpec``
     - canonical cube projection
     - Face size and projection settings.
   * - ``GnomonicProjector``
     - reusable project/back-project pair
     - Preserve backend, dtype, layout, batch, channels and autograd where documented.
   * - ``CubemapProjector``
     - reusable cubemap conversion
     - Accept its complete projection result during back-projection.
   * - ``ProjectionResult``
     - sampled raster
     - Data, geometric support, data validity and valid interpolation weight.
   * - ``GnomonicFaceGeometry``
     - face lattice geometry
     - Intrinsics, rotation, rays and ERP sampling map.
   * - ``ERPPointProjection`` / ``GnomonicPointProjection`` / ``GnomonicRayProjection``
     - projected continuous coordinates
     - Coordinates plus explicit visibility/support state.

Face construction helpers
-------------------------

.. list-table:: Stable gnomonic lattice helpers
   :header-rows: 1
   :widths: 31 69

   * - Function
     - Responsibility
   * - ``gnomonic_rotation``
     - Build the canonical active 3D orientation from longitude, latitude and roll.
   * - ``gnomonic_intrinsics``
     - Derive the pixel-centre pinhole intrinsics for one gnomonic specification.
   * - ``gnomonic_pixels_to_rays``
     - Lift continuous face pixels into canonical unit rays.
   * - ``gnomonic_pixel_map``
     - Produce the ERP sampling map and support for a declared face.
   * - ``gnomonic_face_geometry``
     - Return the complete, reusable face geometry in one typed object.

Experimental registered-camera transforms
-----------------------------------------

``camera_rays_to_world_points``, ``world_points_to_camera_rays`` and
``relative_camera_pose`` apply the canonical
``x_camera = R_world_to_camera @ (X - C)`` convention. They preserve NumPy or
Torch dtype/device and Torch autograd, but form a separately classified
Experimental surface while installed-wheel and independent frame review are
open. Their formulas are shared by registered MVS and dense reconstruction.

Policies and constants
----------------------

``Interpolation`` and ``InvalidPolicy`` name the allowed sampling behaviors.
Labels and boolean masks are nearest-only.  ``invalid_policy="propagate"`` is
strict; ``"renormalize"`` requires a meaningful floating-data formula and an
explicit minimum valid weight.  ``CUBE_FACE_BASES`` and ``CUBE_FACE_ORDER``
define the canonical cube convention.

Experimental planar pinhole geometry
-------------------------------------

``panorai.calibration.PinholeIntrinsics`` and its pixel/ray functions use the
same pixel-centre, ``+X`` right, ``+Y`` up and ``+Z`` forward conventions for
ordinary planar pinhole images.  The current model fixes square pixels, zero
skew and a centred principal point and does not model lens distortion.  It is
an Experimental foundation for registration, not an alternative to
``geometry-v1`` ERP and gnomonic geometry.

See :doc:`the calibration API <../calibration>` for exact signatures.

Read :doc:`geometry-v1 <../../geometry-v1>` before implementing custom
geometry, then use :doc:`the Geometry API <../geometry>` for signatures.
