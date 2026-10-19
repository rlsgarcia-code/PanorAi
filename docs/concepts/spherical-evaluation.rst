Spherical research evaluation
=============================

.. panorai-api-components:: evaluator

Evaluation is a separate product role
-------------------------------------

An operation transforms data, an estimator infers a model, and an Application
solves a configured task.  An evaluator does none of those things: it compares
an already produced result with independently defined reference evidence.
Keeping that role separate prevents an algorithm from redefining its own
success criterion.

The Stable region, feature, pose and radial-range surface is
``panorai.evaluation``.  It is deliberately
small and NumPy-canonical.  It provides geometry-aware metrics, immutable
typed results, a fingerprinted ``EvaluationProtocol``, and a JSON-safe
``EvaluationRecord``.  Dataset acquisition, experiment orchestration,
training, checkpoints and promotion decisions remain outside the library.

Spherical measurement contract
------------------------------

ERP pixels do not cover equal areas.  A pixel in row :math:`r` receives the
exact area of its latitude cell,

.. math::

   \Delta\Omega_r = \Delta\lambda\left(\sin\phi_{north} -
   \sin\phi_{south}\right).

The sum over a complete ERP is :math:`4\pi`.  Region overlap and radial-range
errors therefore use solid-angle weights rather than a flat pixel mean.
Feature localization and region boundaries use angular distance between
unit bearings.  Rotation error is the geodesic distance on SO(3), while
translation direction is an angle on the unit sphere.

Support, validity and value stay distinct
-----------------------------------------

``support`` says where the geometry covers the sphere.  Prediction and
reference validity say where each datum is usable.  The numeric value carries
the measurement.  A zero range may be valid; a false mask value is valid
background; NaN is not automatically converted to zero.  Evaluation occurs
only on the explicit intersection appropriate to the metric, and coverage is
reported instead of hidden.

Protocols before results
------------------------

An ``EvaluationProtocol`` names the evaluator, exact metric set and all
thresholds.  Its canonical JSON fingerprint changes when any of those choices
changes.  Freeze that protocol and the split identities before opening
held-out results.  ``EvaluationRecord`` then binds metrics to the protocol,
algorithm configuration, package version, artifact checksum, backend and
environment.

The record proves what was measured; it does not prove that a split is truly
held out.  Split provenance and independent review remain governance evidence.
Similarly, a metric implementation passing analytic tests is not by itself a
scientific validation of SLIC, DoG, Stereo or instance segmentation.

Empty-set conventions
---------------------

Two empty binary regions have perfect overlap; a non-empty region compared
with an empty one has zero precision or recall as applicable.  Empty feature
sets use the same set-theoretic convention.  Undefined depth ratios and
confidence correlations are represented by ``None`` rather than fabricated
zeros.  These decisions are part of the Stable interface and must be
recorded in the protocol used by a publication.
