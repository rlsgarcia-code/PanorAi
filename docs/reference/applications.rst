Applications API
================

Application stability is assigned per concrete wrapper. ``SphericalSLIC`` is
Stable as ``panorai-spherical-slic/v1``. The common base and DoG, Stereo,
Relative Pose and Mask R-CNN Paper wrappers remain Experimental.

Stable application
------------------

.. autoclass:: panorai.applications.SphericalSLIC
   :members: run, get_params, with_params, describe, as_pipeline, to_config, save, load

Common Experimental contracts
-----------------------------

.. autoclass:: panorai.applications.Application
   :members: run, get_params, with_params, describe, to_config, save, load

.. autoclass:: panorai.applications.PipelineApplication
   :members: as_pipeline, pipeline

Experimental wrappers
---------------------

.. autoclass:: panorai.applications.SphericalDoGFeatures
   :members: run, describe, as_pipeline

.. autoclass:: panorai.applications.SphericalStereoDepth
   :members: run, describe, as_pipeline

.. autoclass:: panorai.applications.SphericalRelativePose
   :members: reference, from_profile, run, assess_capture, estimate_matches, estimate_correspondences, describe, as_pipeline

.. autoclass:: panorai.applications.RelativePoseProfile
   :members: evidence_backed_v1, resolved_frontend_config, to_dict

.. autoclass:: panorai.applications.RelativePoseCaptureMetadata
   :members: baseline_depth_ratio, to_dict

.. autoclass:: panorai.applications.RelativePoseCapturePolicy
   :members: scanner_assisted_reference, hilti_fisheye_development, assess, to_dict

.. autoclass:: panorai.applications.RelativePoseCaptureAssessment
   :members: eligible, to_dict

.. autoclass:: panorai.applications.RelativePoseCaptureInput

.. autoclass:: panorai.applications.RelativePoseCaptureError

.. autoclass:: panorai.applications.AssessRelativePoseCapture
   :members: execute

.. autoclass:: panorai.applications.SphericalMaskRCNNPaper
   :members: run, describe, as_pipeline

Start with :doc:`the Applications tutorial
<../guides/api-levels/applications>` and read :doc:`Application wrappers
<../concepts/application-wrappers>` for the contract rationale.
