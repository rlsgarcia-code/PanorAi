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
