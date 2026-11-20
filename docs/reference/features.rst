Spherical feature API
=====================

The Stable ``panorai-spherical-features/v1`` core orchestrates OpenCV feature
algorithms over PanorAi's canonical gnomonic geometry. Direct spherical DoG
and Harris detection, multiscale routing, and virtual-rig/PyCOLMAP export are
explicitly Experimental extensions. The resolution-selection report is also
Experimental and never resizes caller data implicitly. Detector-only spherical DoG,
coarse-to-fine proposals, tangent patches, and tangent OpenCV descriptors are
Experimental composition surfaces. The typed image-pair, feature-pair,
matching and bearing-conversion stages used by ``SphericalRelativePose`` are
also Experimental; their default configuration delegates to the frozen
relative-pose feature profile. See
:doc:`../guides/how-to/spherical_features` for the workflow and integration rules.

Spherical Harris corners
------------------------

``SphericalHarrisDetector`` computes east/north derivatives and smooths the
structure tensor in local tangent planes. It therefore treats the ERP as a
spherical sampling grid rather than applying a planar Harris window directly
to the raster. Numeric response, support fraction, and validity are returned
separately; invalid response samples are ``NaN`` and black pixels remain valid
when the supplied masks say they are valid.

.. code-block:: python

   from panorai.features import SphericalHarrisConfig, SphericalHarrisDetector

   detector = SphericalHarrisDetector(
       SphericalHarrisConfig(threshold_rel=0.01, max_keypoints=500)
   )
   corners = detector.detect(erp, panorama_id="room")
   print(corners.source_erp_xy, corners.bearings, corners.responses)

Use ``spherical_harris_response(erp)`` when the dense response, support, and
validity maps are needed without keypoint selection. The API is descriptor-free;
descriptor extraction remains an explicit later composition step.

.. automodule:: panorai.features
   :members:
   :member-order: bysource
   :undoc-members:

Advanced OpenCV backend
-----------------------

.. automodule:: panorai.features.backends.opencv
   :members:
   :member-order: bysource
