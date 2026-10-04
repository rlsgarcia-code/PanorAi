Estimate spherical pose and reconstruct multiple panoramas
===========================================================

This guide connects the Stable spherical-feature core to two Experimental
public surfaces without exposing OpenCV objects:

.. code-block:: text

   central ERP arrays
   → panorai.features spherical matches
   → panorai.estimators pairwise R and unit t direction
   → panorai.reconstruction arbitrary-scale cameras and points

Every input panorama must represent a calibrated central camera. Translation
scale is not observable from monocular bearings. A numerical result is not an
automatic acceptance decision: inspect the pairwise quality report and the
global reconstruction diagnostics.

Pairwise pose
-------------

Start with two decoded ERP arrays named ``panorama_a`` and ``panorama_b``.
They may be NumPy ``HW``/``HWC`` arrays or supported Torch layouts; OpenCV
feature work is performed on CPU and the public feature results are NumPy.

.. code-block:: python

   from panorai.estimators import SphericalRelativePoseEstimator
   from panorai.features import SphericalFeaturePipeline

   pipeline = SphericalFeaturePipeline.from_preset(
       "sift-flann",
       face_sampler="icosahedron",
       face_fov_deg=80.0,
       face_shape_hw=(512, 512),
   )
   matches = pipeline.extract_and_match(
       panorama_a,
       panorama_b,
       panorama_id_a="pano-a",
       panorama_id_b="pano-b",
   )

   estimator = SphericalRelativePoseEstimator()
   pose = estimator.estimate(matches.to_bearing_correspondences())
   if pose is None or not pose.quality_report.accepted:
       raise RuntimeError("pairwise geometry was not trustworthy")

   R_b_from_a = pose.R
   t_b_from_a_direction = pose.t
   inliers = pose.inlier_mask

The convention is

.. math::

   x_b = R_{ba}x_a + t_{ba},
   \qquad
   b_b^T[t_{ba}]_\times R_{ba}b_a = 0.

``pose.t`` has unit norm and no metric magnitude. Low parallax, poor angular
coverage, unstable consensus, rotation-only evidence, or ambiguous translation
orientation remain visible in ``pose.quality_report`` and
``pose.degeneracy_reasons``.

Multiview reconstruction
------------------------

For three or more panoramas, extract the pairs that are expected to overlap.
The IDs supplied during extraction identify observations when tracks are built.

.. code-block:: python

   from panorai.reconstruction import SphericalGlobalMapper

   panoramas = {"p0": pano0, "p1": pano1, "p2": pano2}
   pairs = (("p0", "p1"), ("p1", "p2"), ("p0", "p2"))
   pairwise_matches = [
       pipeline.extract_and_match(
           panoramas[a],
           panoramas[b],
           panorama_id_a=a,
           panorama_id_b=b,
       )
       for a, b in pairs
   ]

   mapper = SphericalGlobalMapper()
   result = mapper.reconstruct(matches=pairwise_matches)
   if not result.success:
       raise RuntimeError(result.failure_reasons)

   pose_p1 = result.pose("p1")
   R_world_to_p1 = pose_p1.R
   center_p1_world = pose_p1.center
   points_world = result.points_xyz
   assert result.scale == "arbitrary"

The mapper estimates each pair once, admits quality-accepted edges, averages
rotations, builds conflict-free tracks, positions cameras and points, runs
spherical bundle adjustment, filters weak geometry, and independently checks
tracks observed by at least three panoramas. Insufficient geometry returns
``success=False`` with explicit reasons and no fabricated partial poses.

If pairwise estimation is expensive and must be reused, make that stage
explicit:

.. code-block:: python

   edges = mapper.estimate_pairwise(pairwise_matches)
   result = mapper.reconstruct(edges=edges)

Exactly one of ``matches=`` and ``edges=`` is accepted.

Executable contract checks
--------------------------

The documentation runner executes the estimator boundary against real
``SphericalFeatureMatches`` produced by the feature façade. A rejected or
degenerate pair may correctly return ``None``; a returned pose must satisfy
the public rotation, translation-direction, and interface contracts:

.. literalinclude:: ../../scripts/run_documentation_examples.py
   :language: python
   :start-after: DOCS_RELATIVE_POSE_START = None
   :end-before: DOCS_RELATIVE_POSE_END = None
   :dedent: 4

It also verifies that insufficient multiview geometry is an explicit result
with no fabricated points:

.. literalinclude:: ../../scripts/run_documentation_examples.py
   :language: python
   :start-after: DOCS_RECONSTRUCTION_START = None
   :end-before: DOCS_RECONSTRUCTION_END = None
   :dedent: 4

What to inspect
---------------

Before consuming a map, inspect at least:

* pairwise ``quality_report.accepted`` and rejection reasons;
* admitted, rotation-filtered, and translation-filtered edges;
* active tracks per panorama and multiview corroboration;
* angular reprojection percentiles and triangulation support;
* ``failure_reasons`` and the arbitrary-scale contract.

See :doc:`../reference/estimators` for pairwise policy and
:doc:`../reference/reconstruction` for frames, gauges, numerical stages,
native acceleration, and current validation limits. For online processing,
continue with :doc:`spherical_slam`.
