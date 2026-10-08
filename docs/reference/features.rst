Spherical feature API
=====================

The Stable ``panorai-spherical-features/v1`` core orchestrates OpenCV feature
algorithms over PanorAi's canonical gnomonic geometry. Direct spherical DoG
detection, multiscale routing, and virtual-rig/PyCOLMAP export are explicitly
Experimental extensions. The resolution-selection report is also Experimental
and never resizes caller data implicitly. Detector-only spherical DoG,
coarse-to-fine proposals, tangent patches, and tangent OpenCV descriptors are
Experimental composition surfaces. See
:doc:`../how_to/spherical_features` for the workflow and integration rules.

.. automodule:: panorai.features
   :members:
   :member-order: bysource
   :undoc-members:

Advanced OpenCV backend
-----------------------

.. automodule:: panorai.features.backends.opencv
   :members:
   :member-order: bysource
