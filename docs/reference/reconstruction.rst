Global spherical reconstruction
===============================

``panorai.reconstruction`` is an Experimental, arbitrary-scale sparse
reconstruction surface for three or more central panoramas. Its equations,
policies and orchestration are PanorAi NumPy/SciPy code, with optional
first-party C++ acceleration for repeated numerical kernels. OpenCV remains
responsible for optional feature extraction and matching, and this mapper does
not call OpenCV, PyCOLMAP, COLMAP, or Torch for geometry.

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

#. admit pairwise poses with ``quality_report.accepted`` by default;
#. retain the deterministic largest connected component;
#. initialize through a maximum-support spanning tree and robustly average
   rotations on :math:`SO(3)`;
#. filter rotation-inconsistent edges and resolve the surviving component;
#. build conflict-free tracks by unioning only components with disjoint
   panorama IDs;
#. solve a translation-independent linear camera/point initialization from
   track bearings using several deterministic positive-depth scale anchors;
   select first by the weakest per-camera cheirality support, then total
   support and scale-normalized robust residual;
#. compare every relative translation as an *unsigned axis* against those
   centers, resolve its sign jointly, and remove the worst
   direction-inconsistent edge before rebuilding tracks and re-estimating;
#. refine centers from the surviving, consistently oriented relative
   translations;
#. jointly position cameras and points using

   .. math::

      X_k - C_i - s_{ik}R_i^T b_{ik} \simeq 0,
      \qquad s_{ik}>0;

#. run fixed-rotation and joint spherical bundle adjustment;
#. filter angular reprojection outliers and weak triangulation, retriangulate,
   and perform a final joint refinement;
#. independently repeat the position solve using only tracks observed by at
   least three panoramas, then require its pairwise camera-center directions
   to agree with the primary reconstruction.

Bundle adjustment compares the measured bearing with

.. math::

   \hat b_{ik} =
   \frac{R_i(X_k-C_i)}{\|X_k-C_i\|}

using a two-dimensional log-map residual in the tangent plane of the measured
bearing. The BATA and bundle residuals are evaluated in vectorized batches;
this is an execution change, not a different objective. Descriptor distance is
not treated as a calibrated geometric weight.

Native bundle computation
-------------------------

``SphericalGlobalMapperOptions(bundle_compute_backend="auto")`` uses the
optional first-party C++17 kernel when it is present and otherwise retains the
NumPy reference path. ``"numpy"`` forces the readable reference with SciPy's
sparse finite-difference Jacobian; ``"native"`` requires the compiled kernel
and fails explicitly when it is unavailable.

The native kernel evaluates the same tangent-plane log-map residual and its
analytic local Jacobian blocks with respect to the rotation increment, camera
center and world point. Python still owns graph admission, gauges, robust loss,
SciPy optimization, filtering, provenance and result construction. The
resolved route is recorded as ``diagnostics.bundle_compute_backend``. Native
and NumPy paths are required to agree against independent finite differences
and complete reconstruction fixtures; the native path is acceleration, not a
second reconstruction method.

Admission and failure policy
----------------------------

``edge_admission="accepted"`` is the default. The explicit
``"successful"`` override admits every numerical pairwise result, including
one rejected by its quality policy. Relative-pose quality now records all four
Essential decompositions, the best and alternative positive-depth counts, and
their normalized cheirality margin. A small margin is reported as
``ambiguous-translation-orientation``.

The global mapper treats pairwise translation as an axis before assigning a
sign. ``translation_max_error_deg`` controls removal against the independent
track-bearing initialization; the default is 20 degrees. Inconsistent edges
are removed worst-first for at most ``translation_consistency_rounds`` solves,
rather than all at once. ``bearing_position_anchor_trials`` bounds the
deterministic scale-anchor candidates. Rejected edges,
rotation- and translation-filtered edges, sign flips, per-edge axis errors,
positive-depth coverage, final reprojection percentiles, excluded panoramas,
active-track counts per panorama, multiview-corroboration status and angular
disagreement, track conflicts, costs, gauges, and stage names remain in
``SphericalReconstructionDiagnostics``.

The default ``require_multiview_corroboration=True`` prevents a map supported
only by independent two-view tracks from being reported as trustworthy. The
independent solve uses tracks of length three or greater and rejects the
result if it fails or if the P90 disagreement between pairwise center
directions exceeds 15 degrees. These controls are exposed as
``multiview_corroboration_min_track_length`` and
``multiview_corroboration_max_position_error_deg``. Disabling corroboration
is an explicit permissive override; it increases coverage but removes this
independent check against repeated-texture and weak-translation solutions.

The mapper requires at least three connected panoramas. Pure rotation,
insufficient tracks, low triangulation angle, disconnected reference cameras,
complete filtering, or a panorama with fewer than
``min_active_tracks_per_panorama`` surviving tracks return ``success=False``
and an explicit reason. A camera is therefore never reported as registered
solely because it survived graph connectivity while all of its observations
were rejected. Failed independent multiview corroboration likewise returns no
partial geometry and starts its reason list with
``multiview-corroboration-failed``.

A ``reference_id`` that does not identify any input panorama remains invalid
and raises ``ValueError``. If it identifies an input panorama that is excluded
from the selected connected component, that is geometric insufficiency: the
mapper returns ``success=False`` with
``reference-not-in-selected-component`` instead of raising during a batch.

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
