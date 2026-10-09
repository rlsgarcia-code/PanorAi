Spherical relative-pose API
===========================

``panorai.estimators`` is an isolated Experimental surface. It consumes
panorama-frame bearings and does not import or call OpenCV, PyCOLMAP or Torch. See
:doc:`../how_to/spherical_features` for the end-to-end feature workflow,
coordinate convention and limitations.

The higher-level, probability-calibrated composition from two EQR images is
documented in :doc:`../how_to/probabilistic_two_view`. It remains Experimental
and leaves this estimator's geometric result unchanged.

.. automodule:: panorai.estimators
   :members:
   :member-order: bysource
   :undoc-members:

Polynomial five-correspondence kernel
--------------------------------------

The PanorAi-owned solver parameterizes ``E`` in the four-dimensional nullspace
of five epipolar equations, builds the ten calibrated cubic constraints, and
uses a 10-by-10 action matrix to enumerate real solutions. All four projective
coefficient charts are attempted and sign-equivalent roots are deduplicated.
A separately identified deterministic numerical search remains a fallback for
singular polynomial charts. This is PanorAi code and does not call OpenCV or
COLMAP's minimal solver.

Native compute kernels
----------------------

The distribution can build a first-party C++17 extension for the repeated
fixed-size polynomial construction and spherical tangent-Sampson residual
kernels. This is an acceleration of the same PanorAi mathematics, not a call
to OpenCV or COLMAP. ``RelativePoseOptions(compute_backend="auto")`` selects
the native implementation when it is installed and otherwise uses the NumPy
reference. The caller can force either route for reproducibility::

   from panorai.estimators import (
       RelativePoseOptions,
       SphericalRelativePoseEstimator,
       native_kernels_available,
   )

   print(native_kernels_available())

   reference = SphericalRelativePoseEstimator(
       RelativePoseOptions(compute_backend="numpy")
   )
   accelerated = SphericalRelativePoseEstimator(
       RelativePoseOptions(compute_backend="native")
   )

Explicit ``native`` selection fails when the compiled extension is absent;
it never silently falls back. Every successful pose records the resolved
``compute_backend`` in ``describe()``. Because the global mapper and
incremental SLAM both consume ``SphericalRelativePoseEstimator``, they reuse
this kernel automatically instead of carrying separate essential-matrix
implementations.

The NumPy implementation remains the normative readable oracle. Native and
NumPy paths are required to agree on essential solution sets, residuals,
inlier masks, rotations, translation directions and acceptance decisions.

Minimal-set sampling inside RANSAC
----------------------------------

The default ``SpatiallyWeightedFivePointSampler`` is a proposal component
*inside* RANSAC. On each trial it chooses only the five correspondences used
to generate a hypothesis. It never removes correspondences from full-set
scoring, inlier classification, cheirality checks, or local refinement.

The proposal combines optional caller weights with angular diversity in both
panoramas, spherical-cell coverage, and a conditioning check on the five
epipolar equations. It first enforces the configured angular separation,
then retries with half that separation, and finally falls back to a uniform
five-point draw. A uniform-trial mixture, enabled by default, ensures that a
localized but valid overlap region retains a direct proposal route.

The sampler is injectable::

   from panorai.estimators import (
       SpatiallyWeightedFivePointSampler,
       SphericalRelativePoseEstimator,
       UniformFivePointSampler,
   )

   spatial = SpatiallyWeightedFivePointSampler(
       min_angular_separation_deg=8.0,
       min_unique_cells=4,
   )
   estimator = SphericalRelativePoseEstimator(sampler=spatial)

   # Exact compatibility route for classic uniform proposals.
   uniform_estimator = SphericalRelativePoseEstimator(
       sampler=UniformFivePointSampler()
   )

The classic adaptive ``w**5`` RANSAC trial bound is valid only for independent
uniform minimal-set draws and is therefore used only by
``UniformFivePointSampler``. The spatial sampler runs the configured
conservative trial budget. The returned pose records proposal counts,
relaxations, sample separation and conditioning in ``sampling_diagnostics``.

Quality, stability and model competition
-----------------------------------------

Every successful result includes ``quality_report``. It records inlier count
and ratio, spherical-cell occupancy and normalized entropy in both panoramas,
residual quantiles, parallax, cheirality, independent re-estimation from
subsets of the discovered consensus (scored against all observations), and
competition among Essential, rotation-only, and spherical
homography explanations. The default ``RelativePoseAcceptancePolicy`` turns
that evidence into ``accepted`` and explicit ``rejection_reasons`` without
hiding or replacing the estimated ``R`` and ``t``::

   pose = estimator.estimate(correspondences)
   if pose is not None and pose.quality_report.accepted:
       use_pose(pose.R, pose.t)
   elif pose is not None:
       inspect(pose.quality_report.rejection_reasons)

Competing models within ``model_competition_tie_margin`` of the best normalized
score are treated conservatively as an ambiguity, with rotation-only and then
spherical homography taking precedence over Essential in a near-tie.

Hypotheses are ranked by inlier support and then a scale-marginal continuous
residual score, rather than inlier count alone. Local refinement uses
iterative weights marginalized
over a declared range of angular noise scales. This is an independently named
PanorAi scoring policy; it is not advertised as the MAGSAC++ implementation.

Experimental scoring and consensus-refit controls
--------------------------------------------------

``RelativePoseOptions.hypothesis_ranking`` defaults to ``"msac-first"``. It
minimizes ``sum(min((residual / threshold)**2, 1))`` before applying the other
tie breakers. ``"count-first"`` preserves the former maximum-consensus
ordering, while ``"scale-marginal-first"`` promotes the existing multi-scale
continuous score ahead of hard inlier count. These controls change model
selection and remain Experimental.

``nonminimal_refit_max_steps`` defaults to 100. A positive value enables an
unweighted all-inlier linear Essential refit whenever at least eight current
inliers are available. The implementation takes the last right singular
vector of the spherical epipolar design matrix, projects its singular values
to ``(s, s, 0)``, decomposes and globally rescores the pose, and repeats until
the inlier mask stabilizes, cycles, becomes rank deficient, or reaches the
explicit cap. It retains the best hypothesis observed under the selected
ranking. ``RelativePoseResult.consensus_refit_steps`` reports the executed
steps. No Hartley affine recentering is applied to calibrated unit bearings.

For real feature pairs with adequate support, the evidence-backed default is::

   options = RelativePoseOptions()

The former compatibility route remains explicit::

   options = RelativePoseOptions(
       hypothesis_ranking="count-first",
       nonminimal_refit_max_steps=0,
   )

The limit is a cap; the loop normally stops on a stable or repeated inlier
mask. A frozen 12-pair calibrated-fisheye phase reduced median oriented
translation error from 12.93 to 5.93 degrees and increased strict successes
from 4/12 to 6/12. Median rotation improved from 1.01 to 0.90 degrees, which
did not meet the preregistered 25 percent reduction target. The estimator
therefore remains Experimental despite this default selection. One 33-match
pair produced no candidate, and near-zero translation remained unobservable;
callers must retain the quality decision rather than assuming every returned
or requested pose is trustworthy.

Experimental decoupled pose refinement
---------------------------------------

``pose_refinement_method="decoupled"`` enables a two-stage estimator for
far-background-dominated pairs. A three-ray Wahba RANSAC uses angular MSAC to
estimate rotation. Correspondences rejected by that rotation-only model form
the translation pool; pairs of their epipolar-plane normals propose
translation axes, which are ranked by multiscale epipolar, cheirality and
parallax support. Both consensus fits repeat to a bounded stable mask.

``decoupled_rotation_trials`` controls the rotation proposal budget and
``decoupled_refit_max_steps`` bounds each stabilization loop. A translation
candidate is accepted only when its relative margin over every direction more
than ten degrees away reaches
``decoupled_translation_min_score_margin``. The default margin is 0.15.
Unobservable and ambiguous translation pools are explicit rejection reasons.
``RelativePoseResult.decoupled_pose_report`` exposes the decision evidence.
The default ``pose_refinement_method="joint"`` preserves existing behavior.

Translation-orientation evidence
--------------------------------

Every Essential hypothesis has four decompositions. The default
``translation_orientation_method="parallax-weighted"`` chooses among them by
positive-depth support weighted by bounded triangulation strength::

   options = RelativePoseOptions(
       translation_orientation_method="parallax-weighted",
       translation_orientation_parallax_scale_deg=1.0,
   )

For triangulation angle ``theta`` and configured scale ``theta0``, the weight
is ``sin(theta)**2 / (sin(theta)**2 + sin(theta0)**2)``. Nearly parallel rays
therefore cannot dominate the orientation merely because they are numerous.
The compatibility experiment
``translation_orientation_method="positive-depth-count"`` reproduces the
historical binary vote.

At least five rays must have weight greater than or equal to 0.5 before the
weighted orientation is considered observable. Otherwise the result records
``selection_method="positive-depth-count-fallback"`` and a zero decision
margin. The historical axis representative remains available for diagnostics,
but the default quality policy rejects its oriented translation.

``TranslationOrientationReport`` retains the raw positive-depth counts and
raw margin alongside weighted supports, effective correspondence weight,
weighted margin, method, and scale. ``cheirality_margin`` always denotes the
margin used by the selected method, so the existing acceptance policy remains
explicit. This changes only four-way decomposition and confidence evidence;
it does not make translation scale observable and cannot rescue pure rotation
or uniformly weak parallax.

Calibrated confidence
---------------------

``raw_quality_score`` is a bounded ranking score, **not** a probability. A
probability is available only after fitting
``RelativePoseConfidenceCalibrator`` on labeled calibration examples. The
calibrator uses isotonic regression and stores every calibration sample ID.
Evaluation fails if any ID overlaps, making calibration/evaluation leakage an
explicit error::

   calibrator = RelativePoseConfidenceCalibrator.fit(
       calibration_reports,
       calibration_successes,
       sample_ids=calibration_ids,
   )
   probability = calibrator.predict_proba(new_pose.quality_report)
   metrics = calibrator.evaluate(
       held_out_reports,
       held_out_successes,
       sample_ids=held_out_ids,
   )

The default acceptance thresholds are deliberately conservative and remain
Experimental; they are not a calibrated operating point. The 15-pair
development fixture is not large enough to calibrate a production
probability. A separately frozen and labeled set is required before confidence
values can be interpreted statistically.
