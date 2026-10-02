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

Numerical five-correspondence kernel
------------------------------------

The first implementation parameterizes ``E`` in the four-dimensional
nullspace of five epipolar equations and solves the calibrated essential
constraints numerically from deterministic starts. It follows the same outer
five-sample, robust-consensus, local-refinement and cheirality structure as
established relative-pose systems, but it is not the closed-form Nister
elimination implementation. That distinction, the solver identifier and all
threshold units are part of the returned provenance.

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
