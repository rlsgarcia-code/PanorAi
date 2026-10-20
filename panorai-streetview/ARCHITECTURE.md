# Architecture and engine boundary

## Human-in-the-loop pair registration

The application treats every candidate pair as unregistered until a persisted
registration run exists. The backend adapter calls only public PanorAi APIs:
`SphericalFeaturePipeline.for_relative_pose()` for detections and descriptor
matching, `SphericalRelativePose.reference()` with typed
`SphericalBearingCorrespondences` for robust Essential estimation, and
`erp_pixels_to_rays` for manual ERP-pixel correspondences. The frontend never
implements a second geometry formula or acceptance policy.

An initial run persists immutable detector/matcher evidence plus a versioned
pose estimate. Match review state, manual correspondences, and paired object
regions are application evidence. Re-estimation creates a new revision from
that evidence and reports the rotation, Essential matrix, unit translation
direction, inlier mask, angular residuals, degeneracy checks, and quality
score. Pairwise translation scale remains explicitly unobservable.

The application exposes the estimator's quality diagnostics as an editing
decision system. Gate thresholds come from the result's public
`RelativePoseAcceptancePolicy`; coverage, residual, parallax, cheirality,
rotation/translation stability, and model competition are not redefined by the
UI. A failed gate maps to an application tool such as outlier review, manual
correspondence, paired object regions, or choosing another pair. Each new
estimate creates an immutable snapshot and a comparison with its predecessor.

Scene setup is a separate application contract. It derives edge state from
persisted registrations and flags an accepted pose as stale when match reviews
or human evidence are newer than the estimate. Global-map eligibility follows
`SphericalGlobalMapperOptions.min_panoramas`. Pair dense preview can be
eligible in explicit baseline units after one accepted relative pose; metric
scene reconstruction remains blocked until both a connected global scene and
an independent metric scale source exist. A free-form image-space range anchor
is not silently promoted to `SphericalRangePrior`.

Scene-dense readiness is derived from the persisted mapper result rather than
upload metadata: the map must be successful, current, metric, and contain at
least three poses. A `metric-range-prior` mapper result is a valid scale source
even when no imported baseline exists. The setup response exposes stable
blocker IDs and bound/minimum prior counts so the interface can route the
operator to mapping, radial anchors, or mask review without duplicating engine
geometry.

The global-map adapter deterministically rebuilds each eligible
`SphericalFeatureMatches` set from persisted PanorAi feature indices, bearings,
review state, region weights, and accepted manual correspondences. It
re-estimates the public pair pose and admits the edge only if the accepted
result reproduces. `SphericalGlobalMapper` then owns track formation, rotation
and translation consistency, triangulation, bundle adjustment, multiview
corroboration, and scale semantics. The application persists one map revision,
projects successful cameras and points for display, and never serializes
partial geometry from a failed result.

Object regions share an `object_id` across the two images. They are not object
detections invented by the engine: they are user-authored image-space regions.
The adapter counts PanorAi detections and correspondences inside each paired
region, then either increases their robust-sampling weight (`prioritize`) or
limits automatic correspondences to them (`restrict`). Explicitly rejected
matches never enter estimation; accepted matches and manual correspondences
receive higher sampling weight without changing the original descriptor
distance.

Dense annotations are deliberately a different contract. A radial measurement
has one implemented effect: when a seam-aware search binds it to an accepted
inlier feature, the adapter may construct a public `SphericalRangePrior`. The
mapper applies these only when the distinct bound-feature count reaches its
public minimum and makes the prior set part of map staleness.

Include/exclude regions have a separate pair-preview effect. The application
rasterizes them into independent boolean masks at 128x256. It passes the first
as `reference_validity` to public
`PanoramicDenseStereo(route="sphere-native", pose_mode="provided")` for A→B,
then runs the same public application for B→A with the second mask and the
exact inverse pose `R_inv = R.T`, `t_inv = -R.T @ t`. This is necessary because
the sphere-native pair route deliberately does not accept partial source
validity. The application persists separate public
range/confidence/validity summaries and a two-direction display PNG, including
the exact annotation IDs consumed on each side. The preview is not a final
surface and remains explicitly in baseline units when the pair has no metric
baseline. These records use the `pair_validity` effect; legacy staged
include/exclude records normalize to that truthful effect when loaded.

The scene-dense adapter starts only from a current metric global map with at
least three poses. It rebuilds a public `RegisteredSphericalScene`, downscales
each ERP to an explicit working shape, and applies the same accepted per-image
include/exclude regions as typed boolean validity. Public
`SphericalDenseReconstruction` owns multireference range estimation, TSDF
fusion, marching-cubes extraction, colorization, and final validation. The
application persists accepted and rejected runs with the exact map revision,
annotation signature, per-view mask provenance, per-reference range summary,
and engine diagnostics. It emits a PLY only for an accepted surface and marks
the record stale after a map or mask edit. Surface traces remain staged because
the public application has no typed surface-hint input; they are not silently
reinterpreted as depth.

## Layers

1. **Interface** — TypeScript/Vite project intake, pair/image inspection,
   reviewed correspondence/object/dense evidence, build-plan review, and
   scene-manifest export.
2. **Application backend** — FastAPI validation, checksum-bound local storage,
   deterministic routing, job adapters, and provenance.
3. **PanorAi engines** — projection geometry, spherical features, Essential
   estimation, global mapping, stereo, calibration, and stitching. Engines are
   called only through documented public APIs.

Browser input cannot choose imports or arbitrary callables. `engines.py`
contains the allow-listed public symbols. Missing symbols remain blocked.

Perspective projects partition images by `capture_group`; absent metadata
means one default `capture-1` group. The adapter loads checksum-bound images
and calls public `PanoramaStitcher` with public `StitchingOptions` independently
for each group. A calibrated group must provide `horizontal_fov_deg` or
`intrinsics.fx` for every source; an uncalibrated group may omit them and rely
on the engine's robust pair calibration. Each result persists supplied versus
estimated FOV, SO(3) orientations, pair inliers/angular residuals, support,
validity, overlap and engine diagnostics. A downloadable partial ERP exists
only for an accepted group; rejected groups remain isolated and auditable.
Once two or more groups are accepted, the application may create a separate
`spherical_unregistered` project from those outputs with explicit candidate
pairs and `pose_status=unknown`. No pose is inferred from stitching, upload
order, or group identity. The model is deliberately central rotation-only;
the adapter does not reinterpret a parallax rejection as a usable panorama.

## Route matrix

| Input | Implemented application route | Engine state |
| --- | --- | --- |
| Registered ERP | Validate supplied `R_world_to_panorama`, `center_world`, and explicit pair graph; emit nodes and navigation edges | usable now |
| Unregistered ERP | Spherical features → reviewable pairwise Essential poses → global map | pair and public global-map adapters implemented; arbitrary scale remains explicit without priors |
| Calibrated perspective group | Validate FOV/fx → overlap → SO(3) registration → partial ERP | public `PanoramaStitcher` adapter implemented |
| Uncalibrated perspective group | Feature overlap → observable shared-FOV calibration → SO(3) registration → partial ERP | public `calibrate_rotation_pair` composed by `PanoramaStitcher` |

## Mathematical and semantic boundary

- ERP coordinates follow pixel centres with a top-left origin.
- Poses use `x_panorama = R_world_to_panorama @ (X_world - center_world)`.
- Two-view translation has direction only until an independent scale source is
  supplied.
- A returned pose and an accepted pose remain distinct.
- Upload order never defines adjacency. Navigation edges are explicit pairs
  with optional metric baseline and directional-overlap provenance.
- Navigation can exist without dense geometry. Dense stereo/reconstruction is
  an optional downstream stage requiring compatible poses and scale.

## Deferred engine capability

Structural line and vanishing-direction priors are not application features.
Their typed estimation, uncertainty and pose-adjustment contract is tracked in
the PanorAi computer-vision research roadmap and must be implemented, audited
and exposed through a public engine before any UI or persistence surface is
reintroduced here.
