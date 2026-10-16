The spherical image object
==========================

``EquirectangularImage`` deliberately feels familiar to users of Pillow:
images can be opened, created, inspected, transformed and saved through one
object.  The resemblance ends where planar image semantics would become
misleading.  An ERP is a sampling lattice for a sphere, so its horizontal seam
is periodic, its rows cover unequal surface area and its orientation lives in
three dimensions.

Why this is more than a Pillow wrapper
--------------------------------------

The object owns the same four invariants used by Geometry, Spherical Ops and
Pipeline: raster values, geometric support, data validity and the spherical
coordinate convention.  A convenience method is useful only if it transforms
those invariants together.

.. list-table:: Familiar name, spherical contract
   :header-rows: 1
   :widths: 19 36 45

   * - Method
     - Planar image expectation
     - PanorAi meaning
   * - ``resize(size)``
     - Interpolate a rectangle
     - Resample canonical ERP pixel-centre rays, wrap longitude, transform
       support and validity, and force nearest sampling for categories.
   * - ``crop(spec)``
     - Return another rectangular image
     - Project an angular tangent window and return ``GnomonicFace`` with the
       exact inverse projection.  A crop is not presented as a complete sphere.
   * - ``rotate(...)``
     - Rotate a canvas by one angle
     - Apply an active 3D ray rotation.  Yaw, pitch and roll name the axes; a
       bare ``rotate(45)`` is rejected as geometrically ambiguous.
   * - ``transpose(method)``
     - Swap or flip array axes
     - Apply one named longitude, latitude or antipodal reflection.  Height and
       width are never exchanged silently.
   * - ``thumbnail(size)``
     - Mutate an image in place
     - Return a smaller immutable ERP while preserving aspect ratio and never
       enlarging the input.
   * - ``paste(face)``
     - Copy pixels into a rectangular box
     - Reproject a typed ``GnomonicFace`` through its ``GnomonicSpec`` and use
       explicit replace or alpha semantics at support and validity boundaries.
   * - ``filter(operation)``
     - Apply a planar kernel
     - Require a registered sphere-aware operation with its own grid and mask
       transforms.

This is not a claim that every spherical operation is numerically preferable
to every projected-view method.  It is a stronger statement about meaning:
the object cannot silently substitute planar geometry for spherical geometry.
Use ``views().map().reconstruct()`` when a mature pinhole algorithm is the
right tool; use ``filter()`` or Pipeline ``OnSphere`` when the operation itself
is sphere-native.

Ownership and immutability
--------------------------

``open()`` decodes eagerly and the returned object owns its array, so it does
not inherit an open file's lifetime.  ``constant(value, height=..., width=...)``
creates a numerical ``HW`` field or ``HWC`` channel image without exposing
array construction in tutorials; height and width are named to keep geometry
order explicit.  ``sample()`` supplies deterministic visual fixtures, while
``fromarray()`` and ``from_pil()`` copy integration inputs.  ``resize()``,
``rotate()``, ``transpose()``, ``thumbnail()``, ``convert()``, ``filter()`` and
``paste()`` return new containers.  ``copy()`` is the image-style spelling of
the existing deep ``clone()`` operation.

One container, several API levels
---------------------------------

The object is not a fifth API level.  It is the common spherical input carried
through the four documented levels:

* Applications accept it and return typed task results.
* Pipeline routes its fields through direct and projected stages.
* Spherical Ops transform its fields through ``filter()`` or ``OnSphere``.
* Geometry creates its tangent ``GnomonicFace`` crops and inverse mappings.

Multimodal transforms keep co-registered image, radial range, labels and custom
raster fields aligned.  Continuous fields use their declared interpolation;
categorical fields remain nearest-only.  Numeric zero is never used to infer
support or validity.

I/O is intentionally narrower than workflow persistence
--------------------------------------------------------

``save()`` encodes one selected raster field through Pillow.  PNG or JPEG
cannot preserve PanorAi grids, units, validity or attached modalities, so the
method does not pretend to serialize the complete workflow.  Use registered
Pipeline/Application configuration and typed cache/persistence contracts when
those semantics must survive a process boundary.

Continue with the :doc:`object tutorial
<../guides/tutorials/equirectangular_image>` and the
:doc:`exact API reference <../reference/equirectangular_image>`.
