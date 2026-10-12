Semantic object association and localization
============================================

``panorai.object_localization`` is the Experimental
``panorai-object-localization/v1`` API. Its compatibility path estimates
semantic object hypotheses from exactly two views. The additive
``panorai-multiview-object-hypotheses/v1`` path consumes an already registered
spherical reconstruction and dense semantic evidence from two or more views.
``panorai.graph`` may persist and connect accepted evidence, but this module
does not mutate or construct a graph.

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

Registered multiview flow
-------------------------

:class:`~panorai.object_localization.SphericalObjectHypothesisMethod` is the
high-level Experimental facade for the complete implemented methodology. It
does not download or run a classifier, depth estimator, feature encoder, or
segmenter. The caller supplies those evidence records explicitly::

   from panorai.object_localization import SphericalObjectHypothesisMethod

   method_result = SphericalObjectHypothesisMethod().run(
       reconstruction=registered_reconstruction,
       semantic_evidence=evidence,
       concepts=concepts,
       radial_depth=registered_radial_depth,  # optional radial range
       independent_region_provider=infer_each_view_region_independently,
   )

The result retains generation, supported regional utility, track/reprojection
and optional embedding consistency, compact registered-transfer summaries,
the exact pair cues used for reduction, and the reduction result. Transfer
masks are transient and are not retained.

Automatic reduction transfers are evaluated only between hypotheses with the
same exact vocabulary and concept. Incompatible concepts already fail the
hard affinity gate, so rasterizing their regions would add cost without
creating admissible evidence. All hypothesis pairs still appear in the final
auditable affinity table.

Every generated hypothesis exposes a ``provenance`` mapping containing the
geometry contract, spatial-point source, reconstruction scale, exact concept
and proxy weights, semantic evidence sources and their upstream provenance,
the frozen generator configuration, and the clustering/state trace.
Point-free results additionally retain per-track triangulation quality.

Registered transfer is deliberately disabled unless the caller supplies
per-view regions through ``independent_regions_by_hypothesis`` or
``independent_region_provider``. The ordinary generated
``view_regions`` share one reconstructed support cluster and therefore
cannot be used to claim independent transfer agreement. This gate prevents
the tautological comparison prohibited by the methodology.

:class:`~panorai.object_localization.SphericalObjectHypothesisGenerator`
treats a CAM or similarity map as directional semantic evidence, not as an
object mask. Geometry and correspondence identity come from either accepted
sparse landmarks in a successful refined
:class:`panorai.reconstruction.SphericalReconstructionResult`, or from known
camera poses plus cross-view bearing tracks that contain no input 3-D points.
For every active track observation, the generator samples the semantic field
at the observation bearing. Tracks with compatible evidence are clustered in
the registered world frame and promoted conservatively only after the
configured number of independent semantic views and supporting tracks is
reached::

   from panorai.object_localization import (
       RegisteredSemanticEvidence,
       SemanticConcept,
       SphericalObjectHypothesisGenerator,
   )

   evidence = tuple(
       RegisteredSemanticEvidence.from_dense(view_id, dense_cam)
       for view_id, dense_cam in dense_cams_by_view.items()
   )
   concepts = (
       SemanticConcept(
           concept_id="piping",
           display_name="Piping",
           proxy_weights=(("coil", 1.0), ("reel", 1.0)),
           vocabulary="industrial-proxy/v1",
       ),
   )
   result = SphericalObjectHypothesisGenerator().generate(
       registered_reconstruction,
       evidence,
       concepts,
   )
   for hypothesis in result.confirmed:
       print(hypothesis.concept_name, hypothesis.spatial.center_xyz)
       for region in hypothesis.view_regions:
           print(region.view_id, region.spec, region.sparse_support_coverage)

If refined sparse landmarks are not available, use the no-point entry point::

   import numpy as np
   from panorai.object_localization import (
       RegisteredBearingObservation,
       RegisteredBearingTrack,
   )
   from panorai.reconstruction import SphericalCameraPose

   # Given x_v = R_v @ X + t_v, PanorAi's equivalent camera centre is
   # C_v = -R_v.T @ t_v and its convention is x_v = R_v @ (X - C_v).
   poses = tuple(
       SphericalCameraPose(view_id, R_v, -R_v.T @ t_v)
       for view_id, R_v, t_v in registered_rt
   )
   tracks = tuple(
       RegisteredBearingTrack(
           track_id,
           tuple(
               RegisteredBearingObservation(view_id, feature_index, bearing)
               for view_id, feature_index, bearing in observations
           ),
       )
       for track_id, observations in matched_bearing_tracks
   )
   result = SphericalObjectHypothesisGenerator().generate_from_bearings(
       poses=poses,
       tracks=tracks,
       semantic_evidence=evidence,
       concepts=concepts,
       scale="arbitrary",  # use a declared metric scale only when t is metric
   )

``generate_from_bearings`` estimates each latent track point by weighted
multiray intersection. ``bearing_track_triangulations`` reports parallax,
normal-matrix conditioning, positive-depth fraction, angular reprojection
error, and ray-distance residual for every track. An accepted ``point_xyz`` is
an estimate produced by this operation, not ground truth.

Low parallax or an ill-conditioned intersection is a local track degeneracy.
The track is excluded without invalidating otherwise coherent poses. In
contrast, when the configured fraction of non-degenerate tracks fails
cheirality or angular reprojection, the generator raises a pose-consistency
error. Specifically,
:class:`~panorai.object_localization.RegisteredPoseInconsistencyError` exposes
the input, non-degenerate, and rejected track counts, rejected fraction, and
reason histogram. Callers can therefore record ``pose_inconsistent`` without
parsing text. The method does not hide systematic disagreement by returning a
collection of ``bearing_only`` hypotheses. This is the explicit signal to
inspect ``R,t``.

Do not pass points assembled from a depth raster, scanner XYZ, or a one-shot
reprojection filter to ``generate()`` and call them refined landmarks. Either
the points and active observations survived a declared geometric-refinement
stage, or the caller must discard their coordinates and use the point-free
bearing path above.

``RegisteredSemanticEvidence`` is NumPy-only and keeps ``scores`` (``CHW``)
separate from a geometric ``support`` mask (``HW``). Its ``from_dense``
adapter accepts the Experimental ``DenseSemanticEvidence`` contract by
attributes, so importing ``panorai.object_localization`` does not import Torch,
SAM, or the segmentation package.

One canonical concept may list several proxy channels. Missing proxies
contribute zero rather than having their weights silently redistributed, and
the hypothesis retains every proxy and source identifier as provenance. This
supports, for example, a domain concept such as piping backed by explicit
ImageNet ``coil`` and ``reel`` channels without claiming that either class is
an industrial ground-truth label.

The returned spatial support is one of:

``metric_3d``
   Weighted reconstructed-track support in the registered world frame when
   reconstruction declares metric scale. The configured units are explicit.

``scale_free_3d``
   The same finite support in the reconstruction's arbitrary coordinate scale.
   It must not be interpreted as metres.

``bearing_only``
   A proposed direction in one panorama when dense evidence exists but no
   registered semantic track supports it. Bearing-only results are never
   automatically confirmed or triangulated from unrelated CAM peaks.

``extent_covariance_xyz`` and ``extent_radius`` describe the spatial spread of
supporting refined or internally triangulated track points. They are not
calibrated estimator uncertainty. Per-track triangulation conditioning and
residuals remain separate in ``bearing_track_triangulations``.
Every finite hypothesis also exposes one ``HypothesisGnomonicRegion`` per
registered camera. Its centre is the weighted mean direction of the projected
sparse support; its rectangular field of view contains that support plus the
configured angular margin, up to the configured maximum FOV. The region is a
reproducible crop proposal, not an object mask.

``hypothesis_view_consistency`` compares every pair of these view regions. It
reports shared registered tracks, feature-index pairs, Jaccard and smaller-view
coverage. When the source tracks carry geometric residuals, it also reports
the median pair residual and the robust exponential reprojection agreement.
Optional caller-owned ``HypothesisViewEmbedding`` vectors add cosine
similarity without selecting or importing an embedding model. Set
``preprocessing_id`` on those embeddings to make the frozen encoder contract
auditable; mixed preprocessing IDs fail explicitly. These quantities remain
separate rather than being hidden in one uncalibrated confidence score.

Regional utility and registered transfer
----------------------------------------

:func:`~panorai.object_localization.evaluate_spherical_region` evaluates one
independently inferred :class:`~panorai.object_localization.HypothesisGnomonicRegion`
against a :class:`~panorai.object_localization.RegisteredSemanticEvidence`.
It reports solid-angle-weighted semantic-mass recall, observed-sphere
fraction, and mean-score enrichment. Zero semantic mass is valid evidence:
recall and enrichment are then ``None`` rather than being manufactured with
an epsilon. The observed angular fraction remains defined whenever geometric
support is non-empty.

:class:`~panorai.object_localization.RegisteredRadialDepth` stores radial
range, never camera-axis depth. Its ``support`` mask is authoritative and
supported values must be finite and strictly positive. One directional
transfer is::

   from panorai.object_localization import transfer_spherical_region

   a_to_b = transfer_spherical_region(
       region_a,
       region_b,              # inferred independently in B
       radial_depth_a,
       pose_a,
       pose_b,
       target_support=radial_depth_b.support,
   )
   print(a_to_b.solid_angle_iou)

Use :func:`~panorai.object_localization.symmetric_spherical_region_transfer`
when radial depth is available in both views. It transfers A to B and B to A
separately and returns their mean solid-angle IoU. The transfer rasterizes
boolean membership with nearest pixel-centre assignment and wraps longitude;
it never bilinearly interpolates a mask. Independently inferred regions are a
precondition. Projecting one common 3-D box into both views and comparing
those projections is tautological and is not this metric.

Spherical FOV anchors and multiview consensus
---------------------------------------------

``panorai-spherical-anchor-fusion/v1`` is the detector-style region stage for
registered panoramas. A :class:`~panorai.object_localization.SphericalFOVAnchor`
is a local gnomonic field-of-view proposal, not an ERP rectangle or an object
mask. :func:`~panorai.object_localization.generate_spherical_fov_anchors`
produces deterministic scale/aspect alternatives around one direction.

The one-call :class:`~panorai.object_localization.SphericalObjectHypothesisMethod`
converts independently inferred per-view regions into anchors and returns an
optional ``anchor_fusion`` result. The result applies spherical intra-view NMS,
registered pair scoring, deterministic complete-link clustering with at most
one anchor per view, cluster-level NMS for duplicate scale/aspect families, and
a per-view consensus FOV refit. Suppressed anchor and cluster IDs remain
auditable in the result. The method never forces two
cameras to use the same angular box: each view retains the extent supported by
its local evidence and by transferred support from the other views.

The complete audit trail is typed and mask-free. Every generated anchor has
exactly one :class:`~panorai.object_localization.SphericalAnchorDecision` with
action ``retained``, ``suppressed-overlap``, or ``suppressed-capacity``. An
overlap suppression identifies its representative and solid-angle IoU; a
capacity suppression deliberately has neither. Every raw complete-link cluster
similarly has one
:class:`~panorai.object_localization.SphericalAnchorClusterDecision` with
action ``retained`` or ``suppressed-family-overlap``. A cluster is suppressed
only when it has the same explicit local-component proposal-family IDs as a
stronger cluster *and* its per-view consensus regions meet the configured
solid-angle overlap threshold. Sharing an accumulated upstream hypothesis ID
is not sufficient: one hypothesis may contain multiple distinct instances or
local components. These records preserve compact FOV geometry and scores,
never transient ERP masks.

Association cues are typed by
:class:`~panorai.object_localization.SphericalAnchorPairCues`. Registered
region-transfer IoU, shared-track consistency, and optional appearance cosine
similarity are alternative identity evidence. Geometric compatibility is a
multiplier, while concept/vocabulary mismatch, same-view alternatives,
geometric contradiction, depth separation, and track contradiction are hard
gates. ``None`` means a cue was not observed; numeric zero means it was
observed and found no agreement. These states are never conflated.

Clusters use complete-link acceptance: a new anchor must exceed the frozen
affinity threshold against every existing member. This prevents the
single-link ``A--B--C`` bridge failure in which mutually inconsistent A and C
would otherwise be joined through B. A cluster is ``confirmed`` only after it
contains the configured minimum number of distinct registered views.

Consensus masks are transient. For target view :math:`v`, the local anchor
mask and every available registered transfer into :math:`v` are accumulated
with the corresponding pair affinities. The thresholded spherical support is
then refitted to the smallest configured-coverage gnomonic FOV plus an angular
margin. Only the compact region, provenance, and scores are retained; the
temporary ERP masks are released.

Three regions therefore have different meanings:

``search region``
   The projection of the current spatial hypothesis used to locate nearby
   local CAM components. It is not the reported object extent.

``local anchor``
   The FOV fitted from one view's independently thresholded CAM component.

``consensus region``
   The final per-view FOV after cross-view association and registered support
   fusion. This is the recommended gnomonic crop for downstream feature or
   segmentation processing.

Affinity, fusion, and Soft-NMS
------------------------------

:func:`~panorai.object_localization.hypothesis_pair_affinity` keeps the
semantic, geometric, track, transfer, and optional appearance factors
separate. Unavailable optional cues are omitted from the product; an observed
factor equal to zero remains decisive. Vocabulary, concept, spatial-mode,
frame, unit, caller-declared geometric, and depth-separation contradictions
are hard gates and force zero affinity.

Use :func:`~panorai.object_localization.reduce_object_hypotheses` with typed
:class:`~panorai.object_localization.HypothesisReductionCandidate` records.
Edges above ``affinity_threshold`` form deterministic duplicate graph
components. A finite, mutually hard-compatible component is fused in
registered 3-D using
``semantic_score * geometric_quality * independent_view_factor``. The
result retains the representative hypothesis, every source hypothesis ID,
every original score, and a separate ``fused_spatial`` record. It does not
pretend that one source view region was re-inferred from the fused centre.

Unfused alternatives are never deleted. Gaussian spherical Soft-NMS decays
only their ranking score and records every suppressor ID. Incompatible
concepts, frames, units, spatial modes, or explicitly depth-separated supports
remain concurrent. The input hypotheses and their semantic scores are
immutable.

The deterministic hypothesis ID depends on the canonical concept and track
IDs, not input order or a segmentation boundary. Segmentation may refine a
confirmed hypothesis later, but it is not required by this workflow.

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
