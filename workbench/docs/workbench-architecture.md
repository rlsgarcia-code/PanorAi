# Workbench architecture

This document separates implemented behavior from the intended pipeline.

## Product boundary and hierarchy

The workbench is a local client of PanorAi. PanorAi owns geometry, spherical
sampling, feature extraction/matching, estimators, and Experimental object
hypotheses. The workbench owns file/session handling, stage selection,
visualization, manual review, and export.

The product has one discovery entrance and cardinality-specific execution
surfaces:

1. **Catalog** describes every registered public capability and its status.
2. **Single View** executes one-ERP experiments.
3. **Two View** executes pair-native experiments and can apply a single-view
   operation to A, B or both.
4. **Multi View** is visible but intentionally has no runner yet.

Segmentation and pair geometry are tools inside these benches, not peers of the
Catalog. The frontend never recomputes a pose or changes an inlier decision. It may
project returned bearings into a gnomonic viewport for display. The Python
backend calls PanorAi and serializes its evidence without collapsing these
distinct states:

- numeric value versus validity;
- estimated pose versus accepted pose;
- descriptor match versus geometric inlier;
- epipolar compatibility versus a triangulated point;
- semantic evidence versus a confirmed object identity.

## Installed public catalog and local adapters

The backend imports `panorai.catalog.public_catalog()` from the same installed
distribution identified by the runtime fingerprint. Its public, non-internal
surfaces are serialized as `surfaces` without a Workbench-maintained component
list. Families, roles, stability, requirements, public symbols and Learn/Help/
Reference/context links therefore follow the installed package automatically.

Execution has two explicit layers. Automatic components are derived from
catalog surfaces, public `__all__` exports and concrete type annotations; the
generic v2 endpoint accepts only server-issued component IDs. A single legacy
compatibility map retains stable experiment IDs, old parameter forms and
workspace routes for saved URLs and v1 histories. Where parity is proven, that
record also points at the automatic component. A missing surface or symbol
blocks the compatibility record and the strict contract test fails. There is
no second binding registry or runner allow-list.

`GET /api/v1/runtime` fingerprints the imported package. `GET /api/v1/catalog`
returns installed `surfaces` and local `experiments` separately, while
`/api/v1/workspaces/single` and `/api/v1/workspaces/two` filter only local
experiments by cardinality. `/api/v1/experiments` remains a compatibility alias
for the complete payload.

Automatic Runner dispatch is allow-listed by the installed catalog itself.
Parameters are validated from the public signature on the backend;
browser-provided imports, symbols and callables are not accepted. Legacy
branches remain only behind the v1 compatibility endpoint until their input
and visualization parity is demonstrated.

## Single View tools

The Academy is an experimental learning surface, not an alternate computer
vision backend. Its first case adapts the projective-plane and single-view
constructions from Hartley and Zisserman to ERP imagery:

The manual annotation editor is a local tool reached from Single View. Its
existing surface is embedded to preserve editing behavior, but it is labeled
`workbench.tool` and never counted as a PanorAi Application, Pipeline or Op.
The earlier Academy route remains a compatibility learning surface outside the
main product navigation.

- a reading-only projective prelude introduces homogeneous equivalence,
  incidence, points at infinity, and `x' ~ Hx` as the bridge to later
  stitching, without yet introducing a camera projection matrix;
- a selected canonical pixel centre is lifted only through the public
  `panorai.geometry.erp_pixels_to_rays` API;
- the browser may visualize a manual two-point interpretation plane, its great
  circle, and intersections of manually selected planes as a didactic
  construction;
- manual constructions are labeled as such and are not reported as robust
  PanorAI estimators;
- robust great-circle fitting, vanishing-direction estimation, and
  calibration from one panorama remain explicit PanorAI Main library gaps. A
  PanorAI library agent must implement and validate those public APIs before
  the Workbench exposes them as backend analysis.

## Two View workspace

### 1. Keypoints

For each panorama, inspect bearing, ERP source coordinate, detector response,
scale, orientation, validity, and source gnomonic face. The overview may show a
top-response subset, but the total count and full backend result remain
available.

### 2. Descriptor matches

Inspect A/B feature indices, descriptor distance, ratio score, bilateral match
status, and explicit validity. Selecting one match centres both gnomonic views.

### 3. Relative pose and geometric consistency

The backend estimates panorama-B-from-panorama-A rotation `R` and only the unit
direction of translation `t`. It exposes the PanorAi quality decision,
rejection reasons, model competition, inlier mask, parallax, cheirality, and
per-match angular residual.

Before a 3-D point is supported, the B-side display shows the epipolar great
circle generated by an A bearing. It must not call one location on that circle
a reprojection. If a match can be triangulated with positive range and adequate
parallax, a later stage may show the triangulated reprojection and its residual.

### 4. Semantic evidence and regions

Planned, not implemented in the current backend. Each view will retain dense
CAM channels and explicit support. Components become spherical FOV-anchor
families rather than fixed planar boxes.

### 5. Cross-view hypotheses

Planned, not implemented in the current backend. The review surface will show:

- transferred region A→B and independent proposal B;
- transferred region B→A and independent proposal A;
- solid-angle overlap, stable track support, appearance/embedding cue, and
  semantic compatibility as separate measurements;
- local anchors retained or removed, with the exact NMS/fusion reason;
- the final per-view FOVs without forcing the same angular box in both views.

For three or more images, the pair table should become a registered view graph
and hypotheses should be supported by tracks. A precise `X_t` is optional: only
tracks that pass geometric refinement may support triangulation; otherwise the
hypothesis stays bearing-only or scale-free.

## Segmentation and annotation tool

Mask review remains a specialized editing surface inside Single View because it
is a different task from running a PanorAi component. A selected semantic
hypothesis may later be opened here to:

- inspect all SAM mask alternatives, not only the chosen mask;
- review face expansion and spherical fusion;
- compare consensus mask and uncertainty envelope;
- correct the region label or geometry manually;
- export accepted annotations with model provenance kept separate.

Gnomonic images are transient editing views. The source annotation remains a
spherical region or ERP mask with explicit nearest-neighbour semantics.

## Current backend endpoints

- `GET /api/v1/health`
- `GET /api/v1/capabilities`
- `GET /api/v1/runtime`
- `GET /api/v1/catalog`
- `GET /api/v1/experiments` (compatibility alias)
- `GET /api/v1/workspaces/{single|two}`
- `POST /api/v1/runs`
- `GET /api/v1/runs`
- `GET /api/v1/runs/{id}`
- `GET /api/v1/runs/{id}/events`
- `POST /api/v1/runs/{id}/cancel`
- `GET /api/v1/runs/{id}/artifacts/{artifact_id}`
- `GET /api/v1/academy/single-view/status`
- `POST /api/v1/academy/single-view/rays`
- `POST /api/v1/pairs/analyze`
- `GET /api/v1/semantics/status`

The semantic status endpoint returns `available: false`; it exists so the UI
can disable unavailable stages rather than fabricate placeholder results.

`POST /api/v1/runs` queues work on one local device worker. Run manifests are
immutable experiment records in the sense that a replay creates a new ID;
their status and append-only progress events advance until a terminal state.
Inputs are deduplicated by checksum and artifacts are retrieved only through
their run descriptor.
