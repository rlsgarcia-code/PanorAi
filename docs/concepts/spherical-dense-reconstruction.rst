Spherical dense reconstruction
==============================

Dense reconstruction is not a larger depth map. It combines several radial
range observations in a shared metric world frame and asks where a surface is
supported by more than one camera.

PanorAi separates the route into six responsibilities:

.. code-block:: text

   registered ERPs + metric poses
       -> reference-centred MVS sets
       -> radial range per reference
       -> world-frame TSDF
       -> zero-level triangle surface
       -> range-consistent color
       -> accepted or rejected result

Frames and radial range
-----------------------

Every input pose follows

.. math::

   x_i = R_i (X - C_i).

For reference camera :math:`r` and source camera :math:`s`, the MVS relation is

.. math::

   R_{sr} = R_s R_r^T,
   \qquad
   t_{sr} = R_s(C_r-C_s).

The public geometry helpers implement these equations once. MVS estimates
radial range along a camera bearing, not optical-axis depth. Backprojection is
therefore :math:`X=C_i+R_i^T(r b)`.

TSDF evidence
-------------

For a world voxel centre projected into camera :math:`i`, PanorAi evaluates

.. math::

   d_i(X) = r_i(b_i(X)) - \|x_i\|,

and integrates the clipped value :math:`d_i/\tau`. A voxel is integrated only
when the sampled range, support and validity satisfy the declared policy.
Weight zero means unknown. It is not silently interpreted as empty space.
Free space in front of an accepted range is observed, while integration behind
the surface stops after one truncation band; occluded space stays unknown.

The candidate reviewed profile uses uniform per-observation weights because
the current MVS confidence is a within-reference ranking score, not a
calibrated probability shared by cameras. Confidence-weighted fusion remains
Experimental.

Surface and color
-----------------

Marching Cubes extracts the zero level only where the required number of
views observed the TSDF. The internal array is ZYX, but public vertices and
normals are converted explicitly to world XYZ. Normals point toward positive
TSDF, the observed free-space side.

A vertex receives color from a camera only when its projected range agrees
with that camera's range map within the truncation distance. Accepted colors
are fused by the per-channel median. A valid black sample remains black;
numeric zero is never used as a validity test.

Failure and maturity
--------------------

Invalid input or configuration raises an exception before expensive work.
Scientific insufficiency—no accepted ranges, no supported zero crossing, or
no surface—becomes a ``rejected`` result with reasons. The entire application
is Experimental. ``reviewed_profile`` freezes the intended future Stable
subset but does not promote it without prospective external evidence,
installed-wheel consumers and independent review.
The 2026-10-19 promotion attempt is NO-GO because every locally available
Matterport3D/Stanford2D3D cohort had already been used for Stereo development;
none is eligible as a new scene-disjoint Dense Reconstruction held-out set.

See :doc:`the tutorial <../guides/pose_range/surface_reconstruction>` and
:doc:`the reconstruction reference <../reference/reconstruction>`.
