Registration and stitching API
==============================

The Experimental ``panorai.stitching`` namespace contains the typed data
and composable Stages used by
:class:`panorai.applications.PanoramaStitcher`.

Input and configuration
-----------------------

.. autoclass:: panorai.stitching.StitchingOptions

.. autoclass:: panorai.stitching.StitchingImage
   :members: shape_hw

.. autoclass:: panorai.stitching.StitchingInput
   :members: from_arrays

Stages
------

.. autoclass:: panorai.stitching.DetectStitchingFeatures
   :members: execute

.. autoclass:: panorai.stitching.DiscoverOverlaps
   :members: execute

.. autoclass:: panorai.stitching.CalibrateAndRegister
   :members: execute

.. autoclass:: panorai.stitching.WarpRegisteredImages
   :members: execute

.. autoclass:: panorai.stitching.CompensateExposure
   :members: execute

.. autoclass:: panorai.stitching.BlendStitchingPanorama
   :members: execute

.. autoexception:: panorai.stitching.StitchingRegistrationError

Intermediate and result values
------------------------------

.. autoclass:: panorai.stitching.StitchingFeatureSet

.. autoclass:: panorai.stitching.StitchingFeatures

.. autoclass:: panorai.stitching.ImagePairMatches
   :members: count

.. autoclass:: panorai.stitching.StitchingMatches

.. autoclass:: panorai.stitching.ImagePairRegistration
   :members: num_inliers

.. autoclass:: panorai.stitching.RegisteredImageSet

.. autoclass:: panorai.stitching.WarpedImageSet

.. autoclass:: panorai.stitching.StitchingDiagnostics

.. autoclass:: panorai.stitching.PanoramaStitchingResult

The output panorama may cover only part of the ERP lattice.  Inspect support,
validity, ``support_count``, ``contribution_map`` and diagnostics rather than
assuming rectangular coverage.

Start with :doc:`the progressive guide
<../guides/registration_stitching/index>` and read the
:doc:`registration concept <../concepts/registration-stitching>` before using
the lower-level Stages.
