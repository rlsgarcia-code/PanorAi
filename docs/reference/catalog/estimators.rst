Estimators and policies catalog
===============================

Estimators consume observations and infer a model.  They are not image
filters and they are not necessarily Pipeline Stages.  PanorAi exposes a
direct estimator for independent use and a thin Stage adapter when the same
computation belongs in a Pipeline.

Registered estimator and policy surfaces
----------------------------------------

.. panorai-api-components:: estimator-policy

Relative pose
-------------

The current public estimator family is Experimental and imported from
``panorai.estimators``.

.. list-table:: Relative-pose computation components
   :header-rows: 1
   :widths: 27 22 22 29

   * - Component
     - Input
     - Output
     - Role
   * - ``SphericalRelativePoseEstimator``
     - weighted reference/source unit bearings
     - ``RelativePoseResult`` or ``None``
     - Direct configurable Essential/R/t estimator; usable without Pipeline.
   * - ``estimate_relative_pose``
     - bearing arrays and options
     - ``RelativePoseResult`` or ``None``
     - Functional entry point to the same mathematical implementation.
   * - ``EstimateRelativePose``
     - ``SphericalFeatureMatches`` or ``SphericalBearingCorrespondences``
     - ``RelativePoseResult`` or ``None``
     - Stage adapter; it does not duplicate the estimator.
   * - ``RelativePoseOptions``
     - configuration values
     - immutable options
     - Solver, MSAC ranking, refit, refinement, stability and backend controls.
   * - ``RelativePoseAcceptancePolicy``
     - ``RelativePoseQualityReport``
     - accepted flag and rejection reasons
     - Post-estimation quality decision, separate from capture eligibility.
   * - ``RelativePoseConfidenceCalibrator``
     - labeled quality reports with disjoint IDs
     - calibrated probabilities and evaluation
     - Optional calibration; raw quality is not presented as probability.

Sampling strategies
-------------------

.. list-table:: Injectable minimal-set samplers
   :header-rows: 1
   :widths: 30 20 50

   * - Component
     - Stability
     - Behavior
   * - ``FivePointSampler``
     - Experimental
     - Protocol implemented by RANSAC proposal strategies.
   * - ``SpatiallyWeightedFivePointSampler``
     - Experimental
     - Default diversity-aware proposals with a uniform mixture; full-set scoring is unchanged.
   * - ``UniformFivePointSampler``
     - Experimental
     - Classic uniform minimal-set proposals and the valid adaptive ``w**5`` trial bound.
   * - ``FivePointSample`` / ``FivePointSamplingDiagnostics``
     - Experimental
     - Auditable proposal and sampling evidence retained by the result.

Policies are intentionally separate
-----------------------------------

``RelativePoseCapturePolicy`` evaluates operator-visible capture facts before
feature extraction.  ``RelativePoseAcceptancePolicy`` evaluates the inferred
pose after robust estimation.  A capture can be eligible and still fail pose
estimation; a returned pose can still be rejected by its quality policy.

``RelativePoseProfile`` freezes the selected frontend, options, sampler and
post-estimation policy under one named identity.  It does not absorb capture
policy because capture evidence has different inputs and provenance.

Result and diagnostic objects are cataloged under :doc:`typed data and
results <data_results>`.  Mathematical details and exact members remain in
:doc:`the estimator API <../estimators>`.

Reusable robust model fitting
-----------------------------

The Experimental :mod:`panorai.model_fitting` namespace is independent from
any one Application:

.. list-table:: Robust fitting components
   :header-rows: 1
   :widths: 28 22 50

   * - Component
     - Input / output
     - Responsibility
   * - ``robust_fit`` / ``ModelFamily``
     - aligned observation arrays → ``RobustFitResult``
     - Deterministic RANSAC/MSAC sampling, full-set scoring, adaptive trial
       bound and local all-inlier refitting for a caller-defined model family.
   * - ``RotationModel`` / ``fit_rotation``
     - paired unit bearings → target-from-source SO(3)
     - Wahba/Kabsch fit with angular residuals in radians.
   * - ``HomographyModel`` / ``fit_homography``
     - paired planar pixels → projective transform
     - Normalized DLT with symmetric transfer residuals in pixels.

Residual units are part of the model contract and must match the configured
threshold.  These components do not select which physical scene model is
appropriate.

Rotation-pair calibration
-------------------------

The Experimental :mod:`panorai.calibration` namespace converts pinhole pixel
centres to ``+Z``-forward camera rays and estimates a central rotation pair.
``calibrate_rotation_pair`` accepts matched pixels, image sizes and optional
horizontal FOVs and returns ``RotationCalibrationResult`` with intrinsics,
SO(3), inliers, spatial coverage and observability reasons.

The current model fixes a centred principal point, square pixels, zero skew
and no lens distortion.  When both FOVs are missing it estimates one shared
horizontal FOV; absence of metadata is not absence of assumptions.  See the
:doc:`calibration reference <../calibration>` and
:doc:`stitching guide <../../guides/registration_stitching/index>`.
