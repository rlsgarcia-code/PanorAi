Global spherical reconstruction
===============================

``panorai.reconstruction`` is an Experimental, arbitrary-scale sparse
reconstruction surface for three or more central panoramas. It is PanorAi
NumPy/SciPy code: OpenCV remains responsible for optional feature extraction
and matching, and this mapper does not call OpenCV, PyCOLMAP, COLMAP, or Torch
for geometry.

The high-level input is a sequence of
``panorai.features.SphericalFeatureMatches`` objects. Pairwise poses may be
estimated internally with ``SphericalRelativePoseEstimator`` or computed once
and supplied as ``SphericalPairwisePoseEdge`` objects::

   from panorai.reconstruction import SphericalGlobalMapper

   mapper = SphericalGlobalMapper(relative_pose_estimator=relative_estimator)

   result = mapper.reconstruct(matches=pairwise_matches)

   # Equivalent route when pairwise estimation is intentionally reused.
   edges = mapper.estimate_pairwise(pairwise_matches)
   reused = mapper.reconstruct(edges=edges)

   if result.success:
       pose = result.pose("panorama-003")
       R_world_to_panorama = pose.R
       center_world = pose.center
       points_world = result.points_xyz
   else:
       inspect(result.failure_reasons)

Exactly one of ``matches`` and ``edges`` is accepted. Invalid contracts raise
``TypeError`` or ``ValueError``. Insufficient geometry returns an unsuccessful
result with no fabricated partial poses or points.

Frames and scale
----------------

The public pose convention is

.. math::

   x_i = R_i (X - C_i),

where ``R_i`` is ``rotation_world_to_panorama`` and ``C_i`` is
``center_world``. A relative-pose edge follows

.. math::

   R_{ba} = R_b R_a^T,
   \qquad
   C_b-C_a \parallel -R_b^T t_{ba}.

The reference panorama has identity rotation and zero center. Monocular
bearings do not contain metric scale; ``result.scale`` is therefore always
``"arbitrary"``. The BATA depth anchor and bundle-adjustment scale anchor are
recorded in diagnostics rather than presented as physical measurements.

GlobalMapper-aligned stages
---------------------------

The mapper follows the current global-SfM decomposition without copying
COLMAP code or claiming numerical identity:

1. admit pairwise poses with ``quality_report.accepted`` by default;
2. retain the deterministic largest connected component;
3. initialize through a maximum-support spanning tree and robustly average
   rotations on :math:`SO(3)`;
4. filter rotation-inconsistent edges and resolve the surviving component;
5. build conflict-free tracks by unioning only components with disjoint
   panorama IDs;
6. initialize centers from relative translation directions;
7. jointly position cameras and points using

   .. math::

      X_k - C_i - s_{ik}R_i^T b_{ik} \simeq 0,
      \qquad s_{ik}>0;

8. run fixed-rotation and joint spherical bundle adjustment;
9. filter angular reprojection outliers and weak triangulation, retriangulate,
   and perform a final joint refinement.

Bundle adjustment compares the measured bearing with

.. math::

   \hat b_{ik} =
   \frac{R_i(X_k-C_i)}{\|X_k-C_i\|}

using a two-dimensional log-map residual in the tangent plane of the measured
bearing. Descriptor distance is not treated as a calibrated geometric weight.

Admission and failure policy
----------------------------

``edge_admission="accepted"`` is the default. The explicit
``"successful"`` override admits every numerical pairwise result, including
one rejected by its quality policy. Rejected edges, rotation-filtered edges,
excluded panoramas, track conflicts, costs, gauges, and stage names remain in
``SphericalReconstructionDiagnostics``.

The mapper requires at least three connected panoramas. Pure rotation,
insufficient tracks, low triangulation angle, disconnected reference cameras,
or complete filtering return ``success=False`` and an explicit reason.

Limitations
-----------

This first version assumes calibrated central panoramas and NumPy bearings. It
does not select image pairs, discover loops, estimate metric scale, adjust
intrinsics, or replace the existing PyCOLMAP database exporter. Promotion from
Experimental requires independent real multiview consumers and broader
degeneracy/performance evidence.

API
---

.. automodule:: panorai.reconstruction
   :members:
   :member-order: bysource
   :undoc-members:
