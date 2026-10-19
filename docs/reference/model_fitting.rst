Model-fitting API
=================

The Experimental :mod:`panorai.model_fitting` namespace separates geometric
model families from robust sampling and scoring.  It is not coupled to the
stitching Application.

.. automodule:: panorai.model_fitting

.. autoclass:: panorai.model_fitting.ModelFamily
   :members: fit, residuals

.. autoclass:: panorai.model_fitting.RobustFitOptions

.. autoclass:: panorai.model_fitting.RobustFitResult
   :members: num_inliers, inlier_ratio

.. autofunction:: panorai.model_fitting.robust_fit

.. autoclass:: panorai.model_fitting.RotationModel
   :members: fit, residuals

.. autofunction:: panorai.model_fitting.fit_rotation

.. autoclass:: panorai.model_fitting.HomographyModel
   :members: fit, residuals

.. autofunction:: panorai.model_fitting.fit_homography

RANSAC and MSAC use the same full-set residuals and deterministic seed.  A
``ModelFamily`` must declare the residual unit; thresholds are never silently
converted between pixels, radians or another domain.

See :doc:`the estimator catalog <catalog/estimators>` and the
:doc:`stitching guide <../guides/registration_stitching/index>`.
