Spherical single-view metrology
================================

``panorai.experimental.single_view_metrology`` is a NumPy-only Experimental
surface for leveling one equirectangular panorama and making metric
measurements on an inferred floor.  It does **not** estimate translation,
register two panoramas, or recover arbitrary scene depth.

The input is two independently observable families of image segments whose
3D directions are parallel and lie in the floor plane.  Segment endpoints use
the canonical ERP pixel-center convention.  Each endpoint is lifted to a unit
bearing and each segment becomes the normal of its spherical great circle.
The family direction is the robust null direction of those normals.  The
cross product of the two fitted family directions provides the floor normal;
a user-supplied floor observation resolves its antipodal sign.

The returned ``rotation_panorama_to_world`` maps panorama bearings to a frame
with ``+Z`` up.  With camera center ``(0, 0, camera_height_m)`` and floor
``z = 0``, ``measure_floor_point`` intersects an observed bearing with that
plane.  ``measure_vertical`` intersects the top bearing with the vertical
through a measured floor base and reports the horizontal inconsistency as an
observable residual.

The engine reports family membership, per-observation angular residuals,
P90 residuals, direction-family separation, and a first-order floor-normal
uncertainty indicator.  Rays on or above the horizon, nearly parallel family
directions, degenerate segments, non-finite inputs, and inconsistent verticals
fail explicitly.

The API remains Experimental.  Its promotion gates require independently
surveyed real-scene measurements, calibrated uncertainty under annotation
noise and difficult ERP geometry, an installed-wheel consumer, and independent
numerical/API review.

.. automodule:: panorai.experimental.single_view_metrology
   :members:
   :member-order: bysource
   :undoc-members:
