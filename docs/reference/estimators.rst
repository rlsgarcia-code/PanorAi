Spherical relative-pose API
===========================

``panorai.estimators`` is an isolated Experimental surface. It consumes
panorama-frame bearings and does not import or call OpenCV, PyCOLMAP or Torch. See
:doc:`../how_to/spherical_features` for the end-to-end feature workflow,
coordinate convention and limitations.

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
