# Practical object-localization validation

This development benchmark exercises `panorai.object_localization` on two real
Stanford2D3D panoramas of `area_2/office_3`. The scene contains two desk
instances and one bookcase visible from both camera positions.

The prediction path uses only RGB, manually frozen RGB rectangles, spherical
features, descriptor matching, and estimated relative pose. It produces
scale-free object hypotheses and writes a read-only `prediction.json`. Only
after that file is frozen does the evaluator open:

- expected region-to-region identities;
- the registered metric relative pose and baseline; and
- Stanford radial depth for the first panorama.

The two desk regions share the same ImageNet class ID. Correct performance
therefore requires feature/geometric evidence to select the two correct links
and reject the two cross-instance alternatives.

## Run

The source dataset remains external. With the locally prepared dataset used by
the existing PanorAi public-pair studies:

```bash
python benchmarks/object_localization/run_practical_validation.py --preflight

python benchmarks/object_localization/run_practical_validation.py \
  --output-dir /private/tmp/panorai-object-localization-practical
```

Use `--dataset-root` to point to another prepared copy with the exact frozen
input checksums.

Outputs are `prediction.json`, `summary.json`, `hypotheses.csv`, and an RGB-only
`annotated-regions.png`. Dataset images, depth, and poses are never copied into
the repository.

## What this validates

- real spherical feature extraction and matching;
- accepted estimated relative pose without reference-pose input;
- one-to-one discrimination between two instances of the same class;
- stable unique IDs for the three accepted hypotheses;
- scale-free triangulation followed by evaluation-only metric scaling; and
- spatial center error against independent radial depth at the supporting
  matched features.
- exact identity stability when every manual region is contracted or expanded
  by 16 pixels, approximating ordinary boundary noise; and
- fail-closed behavior under a 32-pixel stress perturbation: missing support or
  overlapping same-class regions may reduce recall, but must not create a
  wrong object identity.

This is post-hoc development evidence from one room, not a promotion result.
The rectangles are intentionally simple and coarse; they validate association
and geometry after regions exist, not detector, CAM, or semantic-segmentation
quality.

## Multipair association and abstention

`run_multipair_association_validation.py` expands the same isolated
association test to six real panorama pairs from three Stanford2D3D areas. It
freezes 14 visible object identities and applies three negative or ambiguity
controls to every pair. Manual RGB regions intentionally remove detector/CAM
quality from this test; spherical features, matches, and relative pose remain
real estimated inputs.

```bash
python benchmarks/object_localization/run_multipair_association_validation.py \
  --preflight
python benchmarks/object_localization/run_multipair_association_validation.py \
  --output-dir /private/tmp/panorai-object-association-multipair-final
```

See [MULTIPAIR_ASSOCIATION_RESULTS.md](MULTIPAIR_ASSOCIATION_RESULTS.md) for
the frozen result. The current end-to-end gate intentionally fails: the
associator promoted all 8 identities that had eligible geometric evidence and
no false identities, but two rejected poses and one pair with insufficient
regional support limited base recall to 8/14. This distinction prevents a
safe fail-closed association result from being misreported as adequate
end-to-end recall.

## Semantic-prior limit study

`run_semantic_prior_limits.py` isolates the proposal-weight hypothesis on
known-SE(3) synthetic correspondences. Baseline and guided runs use the exact
same matches, validity mask, pose options, and random seed. The only changed
input is the minimal-set sampling weight vector.

The five scenarios deliberately include positive and negative controls:

- correct, angularly distributed semantic support;
- correct support concentrated on one compact object;
- equally mixed correct and false semantic support;
- support concentrated entirely on outliers; and
- absent support, which must reproduce the uniform baseline exactly.

Run the short development sweep with:

```bash
python benchmarks/object_localization/run_semantic_prior_limits.py \
  --quick \
  --output-dir /private/tmp/panorai-semantic-prior-limits
```

For a harder pilot with a stronger prior:

```bash
python benchmarks/object_localization/run_semantic_prior_limits.py \
  --seeds 6 --inliers 24 --outliers 96 --max-trials 48 \
  --uniform-mix 0.1 --max-weight-ratio 10 \
  --output-dir /private/tmp/panorai-semantic-prior-limits-hard
```

The synthetic semantic precision is controlled by the benchmark. It must not
be presented as measured CAM performance.

## Real ImageNet CAM pose-prior pilot

`run_real_cam_pose_prior.py` applies the same proposal-only API to the frozen
Stanford office pair using cached official ResNet18 weights and native
spherical CAM lattices. It compares the requested `desk + bookcase` query with
an unrelated `flamingo + volcano` textual control, then opens the registered
pose and manual regions only after prediction is frozen.

```bash
python benchmarks/object_localization/run_real_cam_pose_prior.py --preflight
python benchmarks/object_localization/run_real_cam_pose_prior.py \
  --output-dir /private/tmp/panorai-real-cam-pose-prior-final
```

See [REAL_CAM_PRIOR_RESULTS.md](REAL_CAM_PRIOR_RESULTS.md) for quantitative
results and limitations. Dataset and checkpoint bytes remain external.

## Real CAM component-to-object pilot

`run_real_cam_object_components.py` evaluates the next seam in the same frozen
pair: native class CAMs are converted into spherical connected components,
indexed by real features, associated geometrically, and promoted to
deterministic object IDs with scale-free spatial hypotheses. The prediction is
again frozen before manual instance regions and expected identities are opened.

```bash
python benchmarks/object_localization/run_real_cam_object_components.py \
  --preflight
python benchmarks/object_localization/run_real_cam_object_components.py \
  --output-dir /private/tmp/panorai-real-cam-object-components
```

See [REAL_CAM_COMPONENT_RESULTS.md](REAL_CAM_COMPONENT_RESULTS.md). The pilot
shows that the end-to-end API path works, but also that native `7x14` class CAM
components are too coarse and threshold-sensitive to serve as instance masks
alone. The proposed V1 boundary is CAM for semantic relevance, followed by
feature/geometric clustering for instance separation.

## Real joint match-cluster pilot

`run_real_joint_match_clusters.py` tests that proposed geometric splitting
step rather than assuming it works. It clusters the pose inliers of each broad
accepted CAM association by angular proximity in both panoramas and feeds the
paired subregions back through the existing association/localization pipeline.

```bash
python benchmarks/object_localization/run_real_joint_match_clusters.py \
  --preflight
python benchmarks/object_localization/run_real_joint_match_clusters.py \
  --output-dir /private/tmp/panorai-real-joint-match-clusters
```

See
[REAL_JOINT_MATCH_CLUSTER_RESULTS.md](REAL_JOINT_MATCH_CLUSTER_RESULTS.md).
The result is negative for automatic ID promotion: the sweep returned up to 26
localized hypotheses but never recovered the second desk, and strict
hypothesis precision stayed below 0.17. Joint clusters remain useful candidate
geometry, but V1 still needs an instance-aware semantic cue before assigning a
unique object ID.

## Multiview CAM-to-spatial-hypothesis validation

`run_multiview_cam_hypotheses.py` evaluates the intended CAM role directly:
directional semantic evidence for generating spatial candidates, not a
surrogate segmentation mask. It uses four registered Stanford2D3D panoramas
and three P74 panoramas. Version 6 first runs PanorAi's global spherical mapper;
only tracks surviving bundle adjustment, filtering, retriangulation, and
multiview corroboration may enter CAM sampling. Stanford succeeds with 158
refined tracks and completes semantic leave-one-view-out. P74 fails closed at
pairwise edge admission and produces no semantic claim. The older positive
P74 table used externally assembled depth/XYZ points that had not survived the
mapper's refinement and remains withdrawn.

On the successful Stanford reconstruction, the runner also executes the full
method facade. A projected 3-D region is used only to search for a nearby CAM
component; the local component itself defines a compact spherical FOV anchor.
Invalid-aware radial depth supplies symmetric registered transfer, complete
link prevents transitive cross-view merges, and the final per-view FOV is
refitted from consensus support. The common projected 3-D windows are never
reused as independent transfer evidence.

The frozen v7 run expanded local CAM regions into 602 scale/aspect anchors,
retained 48 after spherical intra-view NMS, and reduced 42 raw complete-link
clusters to 7 through cluster-level NMS. It confirmed one two-view `bookcase`
cluster and recorded 35 geometrically overlapping alternatives from the same
explicit proposal families as suppressed. Shared upstream hypothesis lineage
alone is not a suppression condition. Its final FOVs were
`56.2° × 77.5°` and `49.4° × 45.1°`, both centred on the same shelf. The other
two views abstained instead of accepting a nearby but geometrically
unsupported CAM component. The older hypothesis-level reduction remains an
auditable diagnostic; its multiplicative zero-track result does not define the
new anchor association rule.

The runner also writes `stanford2d3d-office3-anchor-fusion-audit.jpg`. Its four
columns show the local CAM components, every scale/aspect proposal in the
winning proposal families, the per-anchor NMS decisions, and the selected cross-view
pair versus its final consensus FOV. The JSON contains one compact decision
record for every input anchor and raw cluster; it retains no ERP mask.

```bash
python benchmarks/object_localization/run_multiview_cam_hypotheses.py \
  --preflight
python benchmarks/object_localization/run_multiview_cam_hypotheses.py \
  --geometry-source refined-mapper \
  --output-dir /private/tmp/panorai-doc016-method-narrative
```

The P74 raster adapter is benchmark-only: it explicitly converts the native
polar scan raster to the canonical PanorAi sphere. Dataset and checkpoint bytes
are checked in place and are never copied into the repository. See
[MULTIVIEW_CAM_HYPOTHESIS_RESULTS.md](MULTIVIEW_CAM_HYPOTHESIS_RESULTS.md) for
the corrected protocol, Stanford result, P74 abstention, model hashes, and next
independent-transfer gate.
