Semantic object association and localization
============================================

``panorai.object_localization`` is the Experimental
``panorai-object-localization/v1`` API. It is the first of two intentionally
separate stages: this module estimates semantic object hypotheses from two
views; a future graph module may persist and connect those hypotheses. This
module does not import, mutate, or construct a graph.

Simple v1 flow
--------------

The caller supplies:

* a required :class:`~panorai.object_localization.SemanticQuery`, preserving
  the original text and the class IDs resolved in a declared vocabulary;
* one or more CAM/semantic regions per panorama;
* :class:`panorai.features.SphericalFeatureSet` objects and their
  :class:`panorai.features.SphericalFeatureMatches`;
* an accepted :class:`panorai.estimators.RelativePoseResult`; and
* optionally, a positive translation scale.

The query defaults to the ``imagenet-1k`` vocabulary. V1 deliberately does
not parse unrestricted language: a detector, classifier, synonym table, or UI
resolves the text to explicit vocabulary IDs and records that decision in the
query. Region proposals outside those IDs are ignored.

The pipeline is deliberately small::

   textual query + regions/maps + spherical features + matches + relative pose
       -> propose class-compatible region pairs
       -> one-to-one assignment with an ambiguity gate
       -> deterministic ObjectHypothesis ID
       -> SpatialLocationHypothesis

The same pair of ``(view_id, region_id)`` observations always produces the
same direction-independent object ID. It identifies the object hypothesis
shared by those two images. Persistent identity across three or more views,
track merging, split/merge resolution, and lifecycle management belong to the
later graph stage.

Spatial meaning
---------------

``SpatialLocationHypothesis`` has three honest output modes:

``metric_3d``
   A translation scale was supplied. Position and covariance use the caller's
   declared units in the first panorama frame.

``scale_free_3d``
   Translation direction is known but scale is not. Position and covariance
   use a normalized unit baseline in the first panorama frame.

``bearing_only``
   Triangulation did not meet the parallax, positive-depth, support, or
   reprojection gates. The result retains a unit viewing direction and does
   not invent a 3D point.

The 3D position is the robust center of triangulated regional feature points;
it is not claimed to be the physical center of the object. ``identity_score``
and ``location_score`` are transparent ranking scores, not calibrated
probabilities.

Map adapter
-----------

:func:`~panorai.object_localization.semantic_region_from_map` converts one
ERP-aligned CAM or semantic score map into a
:class:`~panorai.object_localization.SemanticRegionObservation`. It samples
the map at existing feature coordinates and stores feature indices rather than
copying descriptors. The caller remains responsible for splitting a class
activation map into distinct candidate instances.

:func:`~panorai.object_localization.semantic_regions_from_map` provides the
simple built-in alternative when no detector or instance splitter is
available. It thresholds one class CAM, labels connected components with
periodic horizontal ERP connectivity, and returns deterministic
:class:`~panorai.object_localization.SemanticRegionObservation` candidates.
Vertical rows do not wrap. Four- and eight-neighbour connectivity are explicit
configuration choices.

The accompanying
:class:`~panorai.object_localization.SemanticRegionProposalResult` reports raw,
retained, too-small, featureless, truncated, and seam-crossing component
counts. V1 requires each retained component to contain at least one existing
feature. A component is a CAM support island, not a guaranteed object
instance: two nearby objects may merge, and one object may fragment into
multiple components.

Joint match clusters
--------------------

When one CAM component contains multiple physical objects,
:func:`~panorai.object_localization.cluster_region_association_matches` can
refine an already accepted broad region association. Two matches are
neighbours only when their bearings are within the configured angular radius
in both panoramas. Bearing-space distances cross the ERP longitude seam
without a pixel-coordinate special case.

The returned :class:`~panorai.object_localization.JointMatchClusterResult`
contains paired feature-indexed subregions. They are ordinary
:class:`~panorai.object_localization.SemanticRegionObservation` objects and
can be passed through :class:`~panorai.object_localization.ObjectLocalizationPipeline`
again::

   from dataclasses import replace
   from panorai.object_localization import cluster_region_association_matches

   clustered = cluster_region_association_matches(
       accepted_broad_association,
       evidence,
   )
   refined_evidence = replace(
       evidence,
       regions_a=clustered.regions_a,
       regions_b=clustered.regions_b,
   )
   refined_result = ObjectLocalizationPipeline().estimate(refined_evidence)

By default, only pose inliers participate. Small components are discarded and
reported rather than promoted. Cluster IDs depend on the matched feature pairs,
not match row order, so repeated execution is deterministic.

This is deliberately a candidate-instance heuristic, not instance
segmentation. It uses connected components in a joint angular neighbourhood;
dense background bridges can chain two objects together, while a radius that
is too small can fragment one object. V1 keeps the radius explicit and does
not claim it is calibrated across scenes. On the frozen real office pair, a
1-12 degree sweep created up to 26 localized hypotheses, recovered at most two
of three target identities, and never recovered the second repeated desk.
Therefore this helper is not enabled automatically by the pipeline and its
clusters require an additional instance-aware semantic cue before object-ID
promotion.

Query-conditioned pose sampling
--------------------------------

:func:`~panorai.object_localization.build_semantic_match_prior` turns those
feature-indexed regions into a full-length RANSAC proposal prior. For each
class explicitly named by the :class:`~panorai.object_localization.SemanticQuery`,
it takes the geometric mean of the feature membership evidence in the two
views. Evidence is combined by a maximum, so duplicating a class or region
cannot increase its mass.

The returned :class:`~panorai.object_localization.SemanticMatchPrior` has one
``sampling_weights`` row per original match. Invalid rows remain zero and
every valid row remains strictly positive. Consequently, a CAM can make a
minimal set more likely but cannot remove a correspondence from robust
scoring or pose refinement. If the query has no shared support, the result is
an exact uniform fallback with an explicit ``fallback_reason``::

   from panorai.object_localization import build_semantic_match_prior

   prior = build_semantic_match_prior(
       query,
       regions_a,
       regions_b,
       matches,
   )
   correspondences = prior.to_bearing_correspondences(matches)
   pose = relative_pose_estimator.estimate(correspondences)

The default convex mixture uses a 0.25 uniform component before normalization
and caps the maximum valid-row weight ratio at four. These are conservative
uncalibrated defaults, not learned probabilities. A weight ratio of one is
exactly the uniform-weight baseline.

Semantic support and geometric suitability are separate. Features on a
compact object can have excellent CAM agreement but insufficient angular
spread for a conditioned five-point sample. Conversely, a false CAM can
concentrate sampling on outliers. Callers must compare the candidate against
the ordinary geometric estimate and must not accept it merely because it used
semantic evidence or found more inliers.

Minimal composition
-------------------

After a detector or CAM has created the two region observations::

   from panorai.object_localization import (
       ObjectLocalizationPipeline,
       PairObjectLocalizationInput,
       SemanticQuery,
   )

   evidence = PairObjectLocalizationInput(
       view_id_a=features_a.panorama_id,
       view_id_b=features_b.panorama_id,
       query=SemanticQuery(text="find a chair", class_ids=(chair_class_id,)),
       regions_a=(chair_region_a,),
       regions_b=(chair_region_b,),
       features_a=features_a,
       features_b=features_b,
       matches=matches,
       pose=relative_pose,
       translation_scale=None,
   )
   result = ObjectLocalizationPipeline().estimate(evidence)

   for hypothesis in result.hypotheses:
       print(hypothesis.hypothesis_id, hypothesis.location.mode)

By default, a relative pose rejected by its own quality policy fails closed.
Ambiguous assignments remain visible in ``result.associations`` but are not
promoted into confirmed object hypotheses.

Practical validation
--------------------

``benchmarks/object_localization`` contains a reproducible development
validation on a real Stanford2D3D office pair. It estimates features, matches,
relative pose, object identity, and scale-free spatial hypotheses from RGB and
frozen manual regions before opening the reference pose or depth.

For the frozen pair, the pipeline associated two distinct desk instances and
one bookcase with precision and recall of 1.0. The three hypothesis IDs were
unique and repeated identically. Against registered reference data, relative
rotation and translation-direction errors were 0.268 and 0.283 degrees; the
maximum evaluation-only metric error of the regional 3D centers was 0.014 m.
All identities survived a 16-pixel expansion and contraction of every region.
At the 32-pixel stress level, insufficient or overlapping region evidence was
rejected or marked ambiguous, with no incorrect identity promoted.

This is evidence for association and geometry after candidate regions exist;
it is not evidence for detector or CAM quality, and one room is not enough to
claim broad dataset generalization.

``run_semantic_prior_limits.py`` is a separate causal synthetic benchmark.
Each baseline/guided pair keeps correspondences, validity, estimator options,
and random seed fixed and changes only minimal-set proposal weights. It covers
correct diffuse support, correct compact support, mixed evidence, a misleading
prior, and absent support. This benchmark maps an operating envelope; it is
not evidence that ImageNet CAMs achieve the simulated precision or recall on
real panoramas.

Public API
----------

.. automodule:: panorai.object_localization
   :members:
