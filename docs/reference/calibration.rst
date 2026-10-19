Calibration API
===============

The Experimental :mod:`panorai.calibration` namespace provides the pinhole
pixel/ray convention and bounded pair calibration used by stitching.  It is
independently usable and does not perform feature extraction or matching.

.. automodule:: panorai.calibration

.. autoclass:: panorai.calibration.PinholeIntrinsics
   :members: from_horizontal_fov, horizontal_fov_deg

.. autofunction:: panorai.calibration.pinhole_pixels_to_rays

.. autofunction:: panorai.calibration.pinhole_rays_to_pixels

.. autoclass:: panorai.calibration.RotationCalibrationOptions

.. autoclass:: panorai.calibration.RotationCalibrationResult
   :members: num_inliers, median_angular_error_deg

.. autofunction:: panorai.calibration.calibrate_rotation_pair

The current estimator assumes square pixels, zero skew and a centred principal
point.  Unknown pairs share one horizontal field of view.  Lens distortion and
principal-point estimation are not implied by this surface.

Continue with :doc:`the stitching guide
<../guides/registration_stitching/index>` or compare this estimator in the
:doc:`catalog <catalog/estimators>`.
