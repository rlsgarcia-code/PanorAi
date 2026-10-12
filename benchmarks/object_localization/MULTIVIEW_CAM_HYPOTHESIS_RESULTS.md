# Multiview CAM-to-spatial-hypothesis validation

This benchmark tests whether direct-spherical CAM evidence can support a
multiview spatial search region. CAM remains directional evidence rather than
a segmentation mask, and finite 3-D support is admitted only after geometric
refinement.

## Geometry policy

The default `v5` protocol extracts spherical features, matches every panorama
pair, and runs `SphericalGlobalMapper`. A cohort may enter semantic evaluation
only when the mapper returns success and records
`filter-retriangulate-final-refinement`. Only valid tracks and active inlier
observations then participate.

The external dataset poses, radial depth, and scanner XYZ do not construct the
mapper result. The older `v2` depth/XYZ-built points remain withdrawn. The
`v3` point-free external-pose diagnostic remains available with
`--geometry-source point-free-given-poses`; it fails the frozen pose gate for
both current cohorts and is not silently substituted for the refined path.

## Refined reconstruction result

| Cohort | Mapper result | Refined tracks | Active / inactive observations | Reprojection median / P90 / max | Multiview corroboration |
| --- | --- | ---: | ---: | ---: | --- |
| Stanford2D3D `area_2/office_3` | success | 158 | 340 / 1 | 0.083° / 0.306° / 0.846° | passed |
| P74 W120-W122 | `no-admitted-pairwise-edges` | 0 | 0 / 0 | n/a | not reached |

Stanford therefore satisfies the user's sparse-point condition. P74 correctly
abstains before loading or interpreting either semantic model.

## Stanford semantic leave-one-view-out

For each of four folds, the refined reconstruction stays fixed, one
panorama's CAM is removed, hypotheses are generated from the other three
semantic views, and the held-out CAM is opened only for evaluation.

| Measure | Result |
| --- | ---: |
| all-view hypotheses / confirmed | 50 / 20 |
| held-out projected hypotheses | 183 |
| held-out concept top-1 rate | 62.8% |
| same-class spherical percentile, median | 0.819 |
| same-class top-quartile / top-decile rate | 63.9% / 35.0% |
| confirmed same-class percentile, median | 0.830 |
| confirmed top-quartile / top-decile rate | 69.1% / 40.0% |
| gnomonic semantic-mass recall, median | 18.0% |
| supported sphere consumed, median | 10.7% |
| mean CAM enrichment | 1.69× |
| repeatability against all-view hypotheses | 100% |

The result supports the narrow claim that the proposed direction and region
are semantically enriched. It does not establish a complete object extent or
durable instance identity.

### Frozen manual-region audit

The two annotated held-out views contain frozen RGB-only desk and bookcase
rectangles. Across all eligible projections, centre hit rate was 32.6%, median
annotation recall was 32.2%, precision was 22.7%, and spherical IoU was 0.169.
For confirmed hypotheses, centre hit rate rose to 46.7%, median recall to
35.5%, precision to 31.8%, and IoU to 0.194. These coarse rectangles are an
evaluation aid, not training data or ground-truth masks.

## Independent transfer and hypothesis-level reduction diagnostic

The complete facade was then run on the all-view Stanford evidence. Each
view/concept CAM was thresholded independently at 0.60 on a `128×256`
solid-angle evaluation lattice with horizontal seam connectivity. Component
extent came only from that local CAM; registered track bearings associated a
component to a hypothesis but did not define its boundary. Common 3-D projected
`view_regions` were explicitly excluded from transfer.

The arbitrary-scale mapper frame was aligned once to the metric evaluation
poses so that Stanford radial range could be used. This evaluation-only
similarity produced camera-centre RMSE 0.030 m, maximum centre error 0.038 m,
median rotation error 1.77°, and maximum rotation error 2.18°. It was not used
for feature matching, pairwise pose estimation, bundle adjustment, CAM
generation, or component extraction.

| Complete-method measure | Result |
| --- | ---: |
| generated / confirmed hypotheses | 29 / 15 |
| independent-region hypotheses | 29 |
| within-hypothesis transfer evaluations | 33 |
| positive symmetric-transfer rate | 100% |
| symmetric transfer IoU, median / P90 | 0.239 / 0.471 |
| cross-hypothesis reduction transfers | 120 |
| cross-hypothesis transfer IoU, median | 0.102 |
| pair cues / pair affinities | 37 / 406 |
| selected / Soft-NMS / fused records | 13 / 16 / 0 |

No pair exceeded the older hypothesis-level fusion affinity because the
candidate clusters had disjoint registered track sets; under that declared
multiplicative affinity, an observed track-consistency score of zero is
decisive rather than treated as a missing cue. Transfer overlap still drove
spherical Soft-NMS for 16 of 29 records. This remains an auditable diagnostic,
but it is not the final detector-style region association rule.

## Spherical FOV-anchor fusion

Version 5 adds the region stage required by the intended downstream use. A
projected 3-D hypothesis supplies only a search centre. In every registered
semantic view, the nearest independently thresholded CAM component within the
frozen 35-degree fallback gate becomes a local FOV anchor. A view without a
semantic track observation may still propose an anchor around the projected
centre; it does not inherit a fabricated track-consistency score.

The fusion stage then applies solid-angle intra-view NMS, computes registered
depth transfer and centre-geometry cues, and clusters with deterministic
complete link. A new anchor must agree with every cluster member, and one
cluster may contain at most one anchor from each view. Transfer, track, and
appearance cues are alternative identity evidence; geometric compatibility is
a multiplier and declared contradictions are hard gates. Missing cues and
observed zeros remain distinct.

| Anchor-fusion measure | Result |
| --- | ---: |
| input scale/aspect anchors | 602 |
| retained after spherical intra-view NMS | 48 |
| suppressed near-duplicates / capped alternatives | 554 |
| evaluated anchor pairs | 1,128 |
| pairs above the frozen affinity gate | 36 |
| raw complete-link clusters | 42 |
| duplicate clusters suppressed | 35 |
| retained clusters / confirmed multiview clusters | 7 / 1 |
| confirmed concept | `bookcase` |
| confirmed views | 2 of 4 |
| transient masks retained | 0 |

The two final consensus crops surround the same physical bookshelf. Their
view-specific FOVs are `56.2° × 77.5°` and `49.4° × 45.1°`; requiring equal
angular extents would be geometrically unjustified. The remaining two views
retain local proposals but are not added to the confirmed cluster because no
pairwise registered association passes the frozen gate. This is the intended
abstention behavior, not a recall claim.

### Decision audit

The v7 audit extension accounts for every intermediate decision:

| Decision type | Count |
| --- | ---: |
| anchors retained by intra-view NMS | 48 |
| anchors suppressed by solid-angle overlap | 231 |
| anchors suppressed by the per-view/concept capacity | 323 |
| raw clusters retained by cluster NMS | 7 |
| raw clusters suppressed by matching proposal family and overlap | 35 |

`anchor-fusion-audit.jpg` visualizes the winning proposal families in four columns:
CAM/local components, generated scale/aspect FOVs, green/red/orange NMS
decisions, and gold selected inputs versus magenta consensus outputs. For the
confirmed pair, geometric compatibility was 0.530, symmetric registered
transfer support was 0.371, track consistency was 0.700, and the combined
affinity was 0.430. These are separate auditable factors, not a claim that the
CAM is a segmentation mask.

## Interpretation and remaining method gate

The corrected experiment now proves that a real post-refinement sparse
reconstruction can feed direct-spherical CAM sampling, clustering, adaptive
gnomonic regions, and held-out semantic evaluation without an input object
centre. It also proves fail-closed behavior when a cohort cannot be
reconstructed.

The real-data path now exercises independent symmetric transfer, spherical
FOV-anchor generation, NMS, complete-link association, view-specific consensus
refitting, and the older hypothesis-level reduction diagnostic. The remaining
empirical expansion is scene-level: more reconstructable public rooms,
pose/track perturbation ablations, scale/aspect-anchor ablations, optional
frozen face embeddings, and confidence calibration. P74 still requires a
geometric reconstruction that passes pairwise admission before it can join
that comparison.

## Reproduction and provenance

- schema: `panorai-multiview-cam-hypothesis-loo/v7`;
- runner SHA-256:
  `b977fe41fa781a30a3eb02061939fec302d989ba40183c802e1a141d8540f953`;
- external `result.json` SHA-256:
  `7712589ebc345e2b5018ce83106490a935e4c8bb2bffbc7e2508c0594982b992`;
- external anchor-fusion panel SHA-256:
  `afc6684f225dd26ac97fa4ae1e74bd820ab2e3525ba636f88c019975afa5bf8d`;
- external anchor-fusion audit panel SHA-256:
  `f1cc5f1a95dba0f885893dd1b9f7240e6ed23e5d854999c50746134a7e14c88b`;
- official ImageNet ResNet18 checkpoint SHA-256:
  `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`;
- private P74 checkpoint present in preflight:
  `5e464762f0f9284b03eecb75bd5a68ad62c6ff70faa54889f8112bf307cc44a7`;
- external result directory:
  `/private/tmp/panorai-doc016-method-narrative`.

```bash
python benchmarks/object_localization/run_multiview_cam_hypotheses.py \
  --preflight

python benchmarks/object_localization/run_multiview_cam_hypotheses.py \
  --geometry-source refined-mapper \
  --output-dir /private/tmp/panorai-doc016-method-narrative
```

Stanford2D3D remains governed by its dataset agreement. P74 images, scanner
arrays, poses, private checkpoint, CAM tensors, and generated panels remain
external and are not part of the PanorAi source tree, wheel, or sdist.
