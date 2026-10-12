# From spherical CAMs to spatial object hypotheses

This workflow uses a class-activation map as **directional semantic evidence**
in registered panoramas. Its target is an object hypothesis in the shared
reconstruction frame, not a pixel-perfect mask.

```text
spherical CAMs ── sample at registered observations ──► semantic tracks
       │                                                    │
       │ no registered support                              ├─ spatial clustering
       ▼                                                    ▼
bearing-only proposal                              3D object hypothesis
                                                            │
                                      optional SAM refinement or graph persistence
```

This ordering matters. A red CAM region says that a class or proxy responds in
a direction. It does not establish that two red regions in different images
belong to the same physical object. Registered feature tracks provide that
identity evidence.

## 1. Inputs

The public Experimental facade needs:

1. either a successful refined `SphericalReconstructionResult` containing
   accepted sparse landmarks, or registered camera poses plus multiview
   bearing tracks with no input 3-D points;
2. one ERP-aligned score tensor and explicit support mask for each semantic
   view; and
3. canonical concepts whose proxy channels are fixed before evaluation.

At least two registered poses and two semantic views are required. The score
tensor is `CHW` in `[0, 1]`; support is a separate `HW` boolean array. A black
pixel, zero depth, or zero CAM value is never inferred to mean invalid data.

```python
from panorai.object_localization import (
    HypothesisViewEmbedding,
    RegisteredSemanticEvidence,
    SemanticConcept,
    SphericalObjectHypothesisConfig,
    SphericalObjectHypothesisGenerator,
    hypothesis_view_consistency,
)

evidence = tuple(
    RegisteredSemanticEvidence.from_dense(view_id, dense)
    for view_id, dense in dense_cams_by_view.items()
)

concepts = (
    SemanticConcept(
        concept_id="piping",
        display_name="Piping candidate",
        proxy_weights=(("coil", 0.5), ("reel", 0.5)),
        vocabulary="industrial-proxy/v1",
    ),
    SemanticConcept(
        concept_id="vessel",
        display_name="Cylindrical vessel candidate",
        proxy_weights=(("barrel", 0.5), ("milk can", 0.5)),
        vocabulary="industrial-proxy/v1",
    ),
)
```

`from_dense()` accepts the Experimental `DenseSemanticEvidence` contract by
attributes. This keeps `panorai.object_localization` NumPy-only: importing it
does not import Torch, SAM, or the segmentation package.

If an upstream model has already pooled exact ImageNet classes into a canonical
concept channel, use that channel as the sole proxy. Use multiple proxy names
only when the supplied evidence exposes those exact channels. Missing proxies
contribute zero; their weights are not silently redistributed.

For the complete implemented method, use the high-level facade:

```python
from panorai.object_localization import SphericalObjectHypothesisMethod

method_result = SphericalObjectHypothesisMethod().run(
    reconstruction=registered_reconstruction,
    semantic_evidence=evidence,
    concepts=concepts,
    radial_depth=registered_radial_depth,  # optional radial range
    independent_region_provider=infer_each_view_region_independently,
)
```

This composes hypothesis generation, regional utility, pairwise
track/reprojection evidence, optional registered transfer, pair cues, fusion,
and Soft-NMS. It does not run a CNN, depth model, embedding encoder, or SAM.
Those evidence producers remain explicit and reproducible.

Each output hypothesis keeps a structured `provenance` record with the
geometry convention, exact proxy weights, semantic source IDs and upstream
metadata, generator thresholds, spatial-point source, and reduction trace.
The point-free path also records the triangulation quality of every supporting
track.

The independent-region provider is an important validity gate. Generated
`view_regions` share one reconstructed sparse cluster; transferring them
against one another would be tautological. Physical transfer scores are
computed only from regions that the caller declares as independently inferred
per view. Without those regions, the facade correctly leaves transfer
evaluations empty.

## 2. Generate hypotheses

```python
generator = SphericalObjectHypothesisGenerator(
    SphericalObjectHypothesisConfig(
        minimum_observation_score=0.25,
        minimum_track_score=0.25,
        minimum_independent_views=2,
        minimum_tracks_per_hypothesis=2,
    )
)

result = generator.generate(
    reconstruction=registered_reconstruction,
    semantic_evidence=evidence,
    concepts=concepts,
)

for hypothesis in result.hypotheses:
    print(
        hypothesis.state,
        hypothesis.concept_name,
        hypothesis.view_ids,
        hypothesis.spatial.mode,
    )
```

For each active track observation, PanorAi samples the semantic field at the
registered bearing. It combines the proxies with the declared fixed weights,
uses a geometric mean across positive views, and compares competing concepts.
It then applies deterministic complete-link spatial clustering. Two pipe banks
may therefore become distinct `piping` hypotheses even though both use the same
semantic proxies.

A hypothesis is confirmed only when it has the configured minimum number of
semantic tracks and independent semantic views and does not exceed the allowed
competing-concept ambiguity. Otherwise it remains `proposed` or `ambiguous`,
with explicit reasons. A semantic peak without a supporting registered track
stays a per-view `bearing_only` proposal; PanorAi does not triangulate unrelated
CAM maxima.

### When the track has no input point

The method does not require a precise input (X_t). If the reconstruction
already contains sparse landmarks accepted after geometric refinement, the
regular `generate()` path reuses them and records
`spatial_point_source="refined-sparse-reconstruction"`.

“Accepted after refinement” is a provenance requirement, not merely a small
reprojection number. A point sampled from depth or scanner XYZ and inserted
into a reconstruction-shaped container is not a refined visual landmark. In
that case discard the coordinate, keep only matched bearings, and use the
point-free path. If those bearings disagree systematically with the supplied
poses, `RegisteredPoseInconsistencyError` reports structured counts and the
run must abstain as `pose_inconsistent`.

Otherwise construct `RegisteredBearingTrack` objects and call
`generate_from_bearings()`. The only geometric inputs are the registered poses
and corresponding unit bearings. If poses are supplied in the conventional
form

\[
\mathbf x_v=\mathbf R_v\mathbf X+\mathbf t_v,
\]

convert translation to the PanorAi camera centre

\[
\mathbf C_v=-\mathbf R_v^\mathsf T\mathbf t_v,
\qquad
\mathbf x_v=\mathbf R_v(\mathbf X-\mathbf C_v).
\]

For each track, PanorAi estimates the latent point by weighted multiray least
squares and reports parallax, conditioning, positive depth, angular
reprojection error, and ray-distance residual. An accepted point is an
estimated sparse support point, not a precise object centre and not ground
truth.

The failure interpretation is deliberately asymmetric:

- isolated low-parallax or ill-conditioned tracks are local degeneracies and
  are discarded;
- systematic non-degenerate cheirality or reprojection failures indicate that
  the registered `R,t` are inconsistent with the bearing tracks, so generation
  fails closed with `RegisteredPoseInconsistencyError`;
- CAM evidence with no corresponding track may still produce an explicitly
  per-view `bearing_only` proposal, but it cannot conceal a global pose error.

## 3. Interpret the spatial result

Three output modes are explicit:

| mode | meaning | safe interpretation |
| --- | --- | --- |
| `metric_3d` | centre and sparse support in a metric registered frame | units are those declared by the configuration |
| `scale_free_3d` | finite centre and support in reconstruction coordinates | compare topology and relative position, not metres |
| `bearing_only` | one direction in one panorama | candidate search direction, not a triangulated object |

`extent_covariance_xyz`, the bounding box, and `extent_radius` describe the
spread of supporting reconstructed points. They are not calibrated statistical
uncertainty and do not imply the object's complete physical boundary.

The hypothesis retains its concept, exact proxy IDs, source IDs, sampled
observations, view IDs, and track IDs. Its deterministic ID depends on the
canonical concept and supporting tracks, rather than a segmentation contour or
input order.

## 4. Open a region, not just a point

Each finite hypothesis contains a `view_regions` entry for every registered
camera, including a camera whose CAM was not supplied. A region contains a
canonical `GnomonicSpec` centred on the weighted mean projected direction. Its
horizontal and vertical FOV expand to contain the projected sparse support plus
an explicit margin, subject to the configured maximum FOV.

```python
from panorai.geometry import GnomonicProjector

for hypothesis in result.confirmed:
    for region in hypothesis.view_regions:
        face = GnomonicProjector(region.spec).project(
            panoramas_by_view[region.view_id]
        ).data
        print(
            hypothesis.hypothesis_id,
            region.view_id,
            region.spec.hfov_deg,
            region.spec.vfov_deg,
            region.sparse_support_coverage,
        )
```

`sparse_support_coverage` checks only the reconstructed tracks. It does not
claim that the crop contains the complete physical boundary of the object.

Registered tracks provide a first, model-independent consistency measure:

```python
consistency = hypothesis_view_consistency(hypothesis)
for pair in consistency:
    print(
        pair.view_id_a,
        pair.view_id_b,
        pair.shared_track_count,
        pair.track_jaccard,
        pair.smaller_view_track_coverage,
    )
```

After extracting the transient faces, a caller may compute one descriptor per
face and provide it without coupling the facade to an encoder:

```python
embeddings = tuple(
    HypothesisViewEmbedding(
        view_id,
        vector,
        source_id=f"my-face-encoder/v1:{view_id}",
        preprocessing_id="my-face-encoder/v1:preprocess-sha256",
    )
    for view_id, vector in face_vectors.items()
)
consistency = hypothesis_view_consistency(hypothesis, embeddings)
```

Cosine similarity complements rather than replaces keypoint/track evidence.
Only compare vectors produced by the same frozen encoder and preprocessing.
When `preprocessing_id` is provided, mixed contracts fail explicitly. Track
observations also carry their angular geometric residual, so each pair reports
the median residual and its robust exponential agreement separately from
appearance.

## 5. Fuse detector-style FOV anchors across views

A projected sparse hypothesis is a search direction, not the final crop.
Within that neighborhood, infer CAM components independently in each view and
fit each component with a gnomonic FOV. The facade converts these local regions
to `SphericalFOVAnchor` records, suppresses near-duplicate alternatives on the
sphere, associates compatible anchors across cameras, and refits one consensus
FOV per contributing view.

```python
from panorai.object_localization import (
    SphericalAnchorFusionConfig,
    SphericalObjectHypothesisMethod,
    SphericalObjectHypothesisMethodConfig,
)

method = SphericalObjectHypothesisMethod(
    SphericalObjectHypothesisMethodConfig(
        anchor_fusion=SphericalAnchorFusionConfig(
            intra_view_nms_iou=0.85,
            complete_link_threshold=0.20,
            consensus_coverage=0.90,
            consensus_margin_deg=3.0,
            minimum_final_fov_deg=8.0,
            maximum_final_fov_deg=120.0,
        )
    )
)
result = method.run(
    reconstruction=registered_reconstruction,
    semantic_evidence=evidence,
    concepts=concepts,
    radial_depth=registered_radial_depth,
    independent_region_provider=infer_each_view_region_independently,
)

for cluster in result.anchor_fusion.clusters:
    if cluster.state != "confirmed":
        continue
    for region in cluster.consensus_regions:
        print(cluster.concept_id, region.view_id, region.spec)
```

The three extents must not be confused:

1. the generated hypothesis region is the search window predicted from sparse
   geometry;
2. the local anchor is fitted only from that view's CAM component; and
3. the consensus region is the final per-view gnomonic crop after multiview
   association and registered transfer.

Different cameras generally produce different final horizontal and vertical
FOVs for the same object. This is expected perspective geometry, not an error.
Complete-link clustering requires every new view to agree with every existing
member, avoiding the transitive `A--B--C` bridge failure of single-link
grouping. `None` means a transfer, track, or appearance cue was unavailable;
an observed value of `0.0` remains explicit negative evidence.

The result stores compact FOVs and diagnostics only. ERP masks used for NMS,
registered transfer, and consensus are transient. You can also call
`generate_spherical_fov_anchors()` and
`fuse_registered_spherical_anchors()` directly when an upstream detector
already owns the local proposals or their appearance cues.

For diagnosis, inspect `anchor_fusion.anchor_decisions` and
`anchor_fusion.cluster_decisions`. Together they answer four different
questions without reconstructing hidden state:

1. Which scale/aspect FOV proposals were generated around each local CAM
   component?
2. Which proposals survived spherical NMS, which overlapped a stronger
   representative, and which were removed by the frozen capacity limit?
3. Which retained anchors passed complete-link association across views, with
   what geometry, transfer, track, or appearance factors?
4. Which raw clusters survived cluster-level NMS and how their selected local
   anchors changed during consensus refitting?

Cluster-level NMS uses explicit `proposal_family_ids` tied to the original
local CAM component. It suppresses only a geometrically overlapping
scale/aspect alternative from the same family. Common
`source_hypothesis_ids` alone never remove a cluster, because one upstream
hypothesis can contain multiple object instances or disconnected components.

The multiview benchmark writes an `anchor-fusion-audit.jpg` panel with these
four columns. Cyan shows generated proposals; green shows retained anchors;
red and orange show overlap and capacity suppression; gold is the selected
cross-view input; magenta is the final view-specific consensus FOV.

## 6. Segmentation is optional

Use SAM or another segmenter only when a later operation needs a boundary,
area, cutout, or panoptic display. A confirmed spatial hypothesis can already
support association, indexing, navigation, inspection planning, and graph
persistence. Mask refinement should attach to that hypothesis; it should not
retroactively invent its cross-view identity.

This also makes failure easier to diagnose:

- broad CAM but coherent geometry: retain a low-confidence semantic
  hypothesis;
- strong CAM without tracks: retain bearing-only proposals;
- tracks without stable semantics: keep geometric structure unlabeled;
- competing concepts: mark ambiguity rather than choosing a convenient label;
- failed registration: stop before semantic fusion.

## 7. Evaluate association and regional transfer

For three or more registered panoramas, leave one semantic view out, generate
hypotheses from the remaining views, project each finite hypothesis into the
held-out camera, and sample the held-out CAM at that predicted direction.
The central point percentile is useful as a diagnostic, but the primary target
is a stable region. Useful measures are:

- semantic-mass or annotated-region coverage inside the proposed gnomonic
  window, reported together with its solid-angle cost;
- symmetric regional transfer IoU: infer a region independently in A and B,
  transfer A to B and B to A using registered depth/3D support, then average
  the two spherical IoUs;
- shared-track count, track Jaccard, smaller-view coverage, and reprojection
  residual inside the paired regions;
- optional face-embedding cosine similarity from a fixed external encoder;
- held-out centre score or rank as a secondary diagnostic;
- hypothesis repeatability across folds;
- confirmed/proposed/ambiguous and abstention rates;
- duplicate rate for repeated instances of the same concept; and
- position error when independent metric reference geometry exists.

The registered geometry may remain fixed when the experiment is explicitly
testing semantic leave-one-view-out generalization. A stricter end-to-end study
must also rebuild registration without the held-out image and report that as a
different protocol.

Do not tune thresholds on the held-out panorama. Freeze the model, input
angular density, proxy vocabulary, weights, track-quality limits, and spatial
cluster radius before scoring. Report Stanford2D3D and private industrial data
separately; neither dataset nor generated evidence belongs in the PanorAi
distribution.

The transferred region and the target region must be estimated independently.
Projecting one common 3D bounding region into both views and measuring their
agreement would be tautological. When depth is unavailable, report angular
window overlap and feature consistency separately; do not call it physical
region IoU.

The regional equations are available directly from the Experimental facade:

```python
from panorai.object_localization import (
    RegisteredRadialDepth,
    evaluate_spherical_region,
    symmetric_spherical_region_transfer,
)

utility = evaluate_spherical_region(
    evidence_by_view[region_a.view_id],
    piping_concept,
    region_a,
)

depth_a = RegisteredRadialDepth(
    view_id=region_a.view_id,
    values=radial_range_a,
    support=radial_support_a,
    units="m",
    source_id="scanner-depth:a",
)
depth_b = RegisteredRadialDepth(
    view_id=region_b.view_id,
    values=radial_range_b,
    support=radial_support_b,
    units="m",
    source_id="scanner-depth:b",
)
transfer = symmetric_spherical_region_transfer(
    region_a,
    region_b,
    depth_a,
    depth_b,
    poses[region_a.view_id],
    poses[region_b.view_id],
)
```

`utility.semantic_mass_recall` and `utility.observed_sphere_fraction`
form the coverage-cost pair. `transfer.symmetric_solid_angle_iou` is the
mean of the independently rasterized A-to-B and B-to-A overlaps. Depth means
radial range in the declared units; unsupported or non-positive samples are
never silently converted into valid zero depth.

The final proposal reduction is also explicit:

```python
from panorai.object_localization import (
    HypothesisPairCues,
    HypothesisReductionCandidate,
    reduce_object_hypotheses,
)

pair_cues = (
    HypothesisPairCues(
        hypothesis_id_a=min(hypothesis_a.hypothesis_id, hypothesis_b.hypothesis_id),
        hypothesis_id_b=max(hypothesis_a.hypothesis_id, hypothesis_b.hypothesis_id),
        transfer_iou=transfer.symmetric_solid_angle_iou,
        appearance_cosine_similarity=face_cosine,
        soft_nms_overlap=transfer.symmetric_solid_angle_iou,
        provenance={"transfer": "registered-radial-depth/v1"},
    ),
)
reduction = reduce_object_hypotheses(
    tuple(HypothesisReductionCandidate(item) for item in hypotheses),
    pair_cues=pair_cues,
)
```

Inspect `pair_affinities` before trusting a reduction. Missing appearance or
transfer evidence is omitted rather than converted to zero. Hard semantic,
frame, unit, geometry, or depth contradictions force zero affinity. Compatible
finite components expose a separate fused 3D support with complete lineage;
unfused alternatives remain in the result with an immutable original score
and, when applicable, a decayed Soft-NMS ranking score.

The source-checkout Stanford validation follows this contract with 158 tracks
that survived the global mapper's final refinement. Its detector-style anchor
stage expanded independently fitted CAM regions into 602 scale/aspect anchors,
retained 48 after spherical intra-view NMS, formed 42 raw complete-link
clusters, and retained 7 after cluster-level NMS. Exactly one `bookcase`
cluster was confirmed in two views; 35 redundant cluster alternatives were
recorded as suppressed. The final
FOVs were `56.2° × 77.5°` and `49.4° × 45.1°`; both surround the same visible
bookshelf, while the two unsupported views correctly abstained. The earlier
hypothesis-level diagnostic still reports median symmetric transfer IoU 0.239
and P90 0.471. P74 failed earlier at pairwise edge admission and therefore
produced no semantic, transfer, or anchor-fusion claim. These are development
results from one reconstructable room, not a general performance claim.

## 8. Relationship to the rest of PanorAi

- [Spherical FCN/CAM](09_spherical_fcn_cam.md) creates the dense directional
  evidence.
- [Features and matching](03_features_and_matching.md) creates local spherical
  observations.
- [Multiview reconstruction](05_multiview_reconstruction.md) registers poses
  and tracks.
- [`panorai.object_localization`](../reference/object_localization.rst) defines
  the exact hypothesis contract.
- Proxy-guided segmentation may refine a hypothesis after association.
- `panorai.graph` may persist accepted evidence and lifecycle events; the
  generator itself does not mutate a graph.

The complete path is therefore association-first: learned evidence proposes
*what direction may matter*, registered geometry establishes *which physical
support repeats*, and only then do optional mask and persistence stages add
detail.
