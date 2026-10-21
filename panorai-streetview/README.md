# PanorAi Street View

`panorai-streetview` is an isolated local application for turning image sets
into auditable, navigable Street View projects. It is not part of the
`panorai` wheel or sdist. The application owns uploads, projects, manual
registration/dense evidence, build planning, visualization, and export. Computer-vision
engines must come from documented public PanorAi APIs.

## Current vertical slice

The MVP implements:

- multi-image upload with SHA-256, dimensions, media type, and local object
  storage;
- four explicit routes: registered ERPs, unregistered ERPs, calibrated
  perspective groups, and uncalibrated perspective groups;
- deterministic build plans that name the exact PanorAi public engine for
  every stage;
- pairwise spherical feature detection, matching, Essential estimation and
  review through the public `SphericalRelativePose.reference()` application
  profile;
- global composition of accepted, current pair revisions through the public
  `SphericalGlobalMapper`, with persisted cameras, tracks, diagnostics and a
  recommended pair to inspect;
- a goal-oriented scene setup that distinguishes unrun, weak, stale, and
  accepted pairs and reports the actual prerequisites for global mapping,
  pair-dense preview, metric scale, and scene-dense reconstruction;
- operator-approved/rejected matches, manually placed correspondences and
  cross-view object regions that can prioritize or restrict re-estimation;
- synchronized 1×–8× zoom, pan and reset controls plus a transient ERP ruler
  that reports seam-safe great-circle angle and source-pixel separation
  without presenting either as metric scene distance;
- a goal-oriented ``Piso & escala`` workspace backed by the public
  Experimental PanorAi single-view metrology engine: two robust floor
  direction families and a floor-side observation estimate a leveled frame,
  camera height establishes scale, and floor points or vertical heights are
  measured with explicit residuals and immutable calibration revisions;
- immutable estimation snapshots, before/after revision deltas, public-policy
  quality gates, and a corrective action suggested from the first failed gate;
- dense-workspace annotations with an explicit boundary between bilateral
  include/exclude masks consumed by pair stereo, staged surface traces, and
  feature-bound radial measurements consumed as public `SphericalRangePrior`
  values;
- a persisted bidirectional pair-dense diagnostic that calls public
  `PanoramicDenseStereo`, consumes independent include/exclude regions from
  both panoramas as boolean validity masks, and visualizes radial range without
  confusing a scale-free preview with metric reconstruction;
- a persisted metric scene reconstruction that calls public
  `SphericalDenseReconstruction`, records every per-view mask and range-map
  summary, preserves accepted/rejected provenance, and exposes a PLY only for
  an accepted MVS → TSDF → surface result;
- an executable conventional-photo route through public `PanoramaStitcher`,
  using supplied horizontal FOV/fx or estimating missing shared FOV, SO(3)
  orientations, valid support, exposure gains and a partial ERP per capture
  group; accepted ERPs can be promoted to an unregistered spherical project;
- a downloadable navigable-scene manifest for already registered ERPs.
- explicit registered pair graphs with baseline/overlap provenance;
- a local-only Stanford2D3D importer whose default contract reads pixels but
  not camera poses or ground-truth pair metrics.

Pairwise Essential estimation and graph-scale pose recovery are implemented.
Generic structural annotation tools remain absent.  The narrower floor and
metric workflow is exposed only through
``panorai.experimental.single_view_metrology`` and never changes pairwise
Essential ``R`` or translation direction.  Dense surface-trace constraints
remain staged until a compatible public engine contract exists.
Perspective groups are executable for the public central rotation-only pinhole model;
translation, strong parallax, rolling shutter, distortion, disconnected
overlap and unobservable FOV fail explicitly. Include/exclude regions already
constrain pair stereo and metric multiview reconstruction through explicit
boolean validity masks.

## Conventional-photo workspace

`Fotos calibradas` accepts `horizontal_fov_deg` or `intrinsics.fx` per source
image. `Fotos sem calibração` leaves FOV unset and lets public
`calibrate_rotation_pair`, composed inside `PanoramaStitcher`, estimate a
shared pinhole FOV only when the matched evidence is observable. Both routes
show an editable group assignment for every selected file; the generated
per-file `capture_group` metadata remains available in the advanced JSON and
all files default to `capture-1`. Each group is executed and audited
independently. The stage labels every thumbnail by group, and the result
interface displays calibration provenance, accepted image
pairs, inliers, angular error, coverage and overlap; an accepted group exposes
a 512×256 partial ERP PNG, while a rejected group keeps its reason and exposes
no output image. With at least two accepted groups, promotion creates a new
`spherical_unregistered` project containing only those ERPs and an explicit
all-pairs candidate graph. It does not create camera poses or reinterpret
capture order as geometry. Support and validity remain distinct even though
the downloadable RGB preview renders invalid pixels as black.

## Registration workspace

The primary inspection unit is an explicit candidate image pair, not a
standalone panorama or an annotation primitive. Unknown pose is the default
application assumption. The workspace lets the operator:

1. choose a candidate pair and inspect which pose, overlap and scale evidence
   is still missing;
2. compare reference and neighbor side by side, as a raw ERP overlay, or by
   blinking between them;
3. zoom and pan both views in sync, or measure the great-circle angle between
   two ERP observations while retaining their source-pixel coordinates;
4. run PanorAi feature detection, matching and robust Essential estimation,
   then inspect detections, matches, inliers, residuals, stability, model
   competition and every acceptance gate from the reference profile;
5. accept or reject automatic correspondences, add a visual correspondence,
   or bind two image-space regions to the same object identity;
6. re-estimate `R` and the direction of `t` from all, prioritized, or strictly
   region-constrained correspondences;
7. apply include/exclude masks to both directions of pair stereo, stage surface
   traces, and attach radial measurements to nearby accepted inlier features;
   three distinct bound measurements satisfy the public mapper minimum and can
   establish metric scale;
8. return to the build plan and generate the same auditable scene manifest.

The Setup tab is the default entry point. It does not infer that uploaded
panoramas are registered: every candidate edge starts as `not-run`, becomes
`review` or `accepted` only after estimation, and becomes `stale` when new
human evidence has not yet been incorporated into a new pose revision. The
scene becomes eligible for global mapping only when accepted edges connect the
minimum number of panoramas required by `SphericalGlobalMapperOptions`.

The Mapa tab rebuilds typed `SphericalPairwisePoseEdge` values from the exact
accepted revisions and calls `SphericalGlobalMapper`. It displays only a
successful result: rejected composition persists failure reasons and an edit
recommendation, but exposes no partial cameras, tracks, or point geometry.
Without a baseline or range prior, the map is explicitly labeled as having
arbitrary scale. New pair evidence makes the persisted map stale until it is
run again.

A radial anchor is not automatically depth. The backend performs a seam-aware
nearest search over accepted inlier observations and records the exact panorama
feature index, pair and pixel distance. Only bound, distinct features can become
`SphericalRangePrior`; fewer than `range_prior_min_count` remain unapplied and
the UI reports the deficit. Include/exclude regions have a narrower executable
route: the pair preview rasterizes the boxes for each panorama independently.
Because the public sphere-native route accepts partial validity only for its
reference input, the application runs A→B with A's boolean mask and B→A with
B's mask and the exact inverse relative pose. Both results preserve invalid
pixels as invalid, store the exact consumed annotation IDs, and remain in
`baseline-units` without metric evidence. Surface hints remain staged because
the public dense application does not yet accept that evidence type.

The dense-scene readiness assistant audits three requirements independently: a
current successful global map, a physical scale, and at least three registered
views. Metric scale can come from either a measured `SphericalBaselinePrior` or
an accepted global map scaled by the public minimum of bound
`SphericalRangePrior` values. An arbitrary-scale map therefore routes the
operator directly to the exact number of missing radial measurements, not to a
generic map-inspection action. Passing readiness enables an interactive or
reviewed public `SphericalDenseReconstruction` MVS → TSDF → surface run. The
adapter downscales each ERP to an explicit working shape, applies that view's
accepted include/exclude regions as validity, persists the exact map revision
and annotation signature, and marks the result stale when either changes. A
scale-free map cannot start this route.

`t` is always presented as a direction with unobservable pairwise scale. A
reviewed match is evidence for the next estimation revision; rejecting it does
not rewrite the original detector/matcher snapshot. Review edits update the
working record, while every estimation/re-estimation writes a separate
immutable revision file. Object boxes are image-space
feature-selection constraints: in `prioritize` mode their matches receive
larger sampling weight, while `restrict` mode excludes automatic matches that
do not lie inside the paired boxes.

The estimator always consumes the full match set. To keep the browser payload
bounded, detection visualization contains at most 900 ranked/source-required
features per image; object detection counts in the interface are therefore
explicitly labeled as visible/displayed counts rather than total detections.

The overlay is deliberately labeled as an image-space diagnostic. It does not
claim pose-compensated reprojection; that operation must be implemented by a
documented public PanorAi engine before the UI may present it as geometric
alignment.

## Run locally

Backend:

```bash
cd /path/to/PanorAi
uv run --with-editable . --with-editable './panorai-streetview[dev]' \
  uvicorn panorai_streetview.api:app --reload --port 8010
```

The repository-root editable install is intentional: the application requires
the `3.8.1.dev0` API line that introduces single-view metrology. A standalone
PanorAi 3.8.0 environment is not compatible.

Frontend:

```bash
cd panorai-streetview
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

## Import a real multi-pair Stanford scene

The importer reads an authorized external dataset and writes image bytes only
to ignored local application storage. It never copies dataset media into
source, wheel, or sdist. By default it does not read the Stanford pose files
and does not copy baseline or overlap truth into the project. Four selected
views become six unordered candidate pairs using a complete-graph policy.

```bash
cd panorai-streetview
PYTHONPATH=backend:.. python -m panorai_streetview.datasets \
  --dataset-root /path/to/stanford2d3d \
  --selection-manifest ../benchmarks/panoramic_stereo/public_three_source_manifest.json \
  --set-id stanford-area3-lounge1-4a7bfe-three-source \
  --data-root .local \
  --name "Stanford Area 3 · Lounge 1"
```

The official Stanford poses remain an evaluation oracle. They can be imported
only by an explicit validation-only opt-in:

```bash
panorai-streetview-import-stanford \
  --dataset-root /path/to/stanford2d3d \
  --selection-manifest ../benchmarks/panoramic_stereo/public_three_source_manifest.json \
  --set-id stanford-area3-lounge1-4a7bfe-three-source \
  --data-root .local-evaluation \
  --input-contract registered-ground-truth
```

That opt-in oracle has three reference-centred pairs with declared baselines
of 2.0704 m, 2.8296 m, and 4.3203 m and directional cloud overlaps of 0.9419,
0.8944, and 0.6934. Only this validation mode reads Stanford pose files,
applies the established Stanford-to-PanorAi basis conversion, and validates
each result as finite SO(3).

## Test data boundary

P74 and Stanford2D3D are not bundled. Validation may point this application at
an authorized local P74 subset or a provenance-reviewed Stanford2D3D subset.
Dataset bytes, derived panoramas, checkpoints, and run artifacts remain
external to both application and PanorAi distribution.
