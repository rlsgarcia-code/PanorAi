# Workbench architecture

This document separates implemented behavior from the intended pipeline.

## Product boundary and hierarchy

The workbench is a local client of PanorAi. PanorAi owns geometry, spherical
sampling, feature extraction/matching, estimators, and Experimental object
hypotheses. The workbench owns file/session handling, stage selection,
visualization, manual review, and export.

The product has one goal-oriented discovery entrance, a separate technical
inventory, and cardinality-specific execution surfaces:

1. **O que posso fazer?** (“What can I do?”) maps typed input context to
   deterministic goals and recipes without inspecting pixels or starting a
   run.
2. **Catálogo técnico** (“Technical catalog”) describes every registered public
   capability and its status.
3. **Single View** executes one-ERP experiments.
4. **Two View** executes pair-native experiments and can apply a single-view
   operation to A, B or both.
5. **Multi View** executes registered-scene workflows when their contracts are
   available.

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

The discovery projection is an internal, additive contract:
`panorai-workbench-discovery/v1`. `GET /api/v1/discovery` returns named goals,
problem-language synonyms, recipes, automatic classification and the default
sample. `POST /api/v1/discovery/recommendations` accepts only a closed
`DiscoveryContext` made of ERP roles and modality, masks/labels/support, pose,
camera height, registered-scene state and prior typed artifacts. An RGB ERP and
a scalar raster are distinct capabilities: an operation that requires a
one-channel raster cannot become `ready` merely because an RGB upload exists.
It ranks at most one
recommended path and three alternatives per goal. States are `ready`,
`needs-input` or `unavailable`, with explicit missing requirements or runtime
diagnostics.

The classifier uses public input/output schemas, semantic annotations and
catalog roles. Curated data is limited to goal language, ordering, composed
recipes and specialized workspace routes. Unknown or ambiguous outputs do not
gain a visual or scientific promise. A ready component must be discoverable or
explicitly classified as technical-only; this is enforced by backend tests.

Recipe drafts and their browser-selected files are kept in session storage and
IndexedDB on the same origin. Single View, Two View and the universal Runner
consume those prepared inputs, but none executes on navigation. Python imports
remain server-controlled catalog data.

The backend imports `panorai.catalog.public_catalog()` from the same installed
distribution identified by the runtime fingerprint. Its public, non-internal
surfaces are serialized as `surfaces` without a Workbench-maintained component
list. Families, roles, stability, requirements, public symbols and Learn/Help/
Reference/context links therefore follow the installed package automatically.

Execution has one scientific engine and one finite compatibility vocabulary.
Automatic components are derived from catalog surfaces, public `__all__`
exports and concrete type annotations; the generic v2 endpoint accepts only
server-issued component IDs. The compatibility map retains stable experiment
IDs, old parameter forms and workspace routes, but every ready record points
at an automatic component. The v1 endpoint translates the old form into a v5
component/plan run before scheduling it. Saved v2 records are translated in
memory when re-executed. A missing surface or symbol blocks the compatibility
record and the strict contract test fails. There is no second scientific
runner, binding registry or browser-controlled import allow-list.

`GET /api/v1/runtime` fingerprints the imported package. `GET /api/v1/catalog`
returns installed `surfaces` and local `experiments` separately, while
`/api/v1/workspaces/single` and `/api/v1/workspaces/two` filter only local
experiments by cardinality. `/api/v1/experiments` remains a compatibility alias
for the complete payload.

Automatic Runner dispatch is allow-listed by the installed catalog itself.
Parameters are validated from the public signature on the backend;
browser-provided imports, symbols and callables are not accepted. The v2
endpoint also accepts an ordered plan of catalog component IDs. A step may bind
a typed input to a prior step result (or a declared public result field), but
not to a forward reference, private attribute or arbitrary object path.
Components whose only inspection reason is a required upstream typed result
are runnable exclusively in such a validated plan. UI-only normalization —
RGB scaling, explicit threshold masks, polar-support policy and working
resolution — is recorded as a server-authored input transform. Scientific
execution remains inside the selected public component.

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
- robust great-circle fitting and vanishing-direction estimation are provided
  by the public Experimental `panorai.reconstruction.planar` pipeline
  (`SphericalPlanarDiscovery`, `DetectAndFitGnomonicLines` and
  `EstimateVanishingBundles`) and are invoked only by the backend;
- a general, reviewed camera-calibration API from one arbitrary panorama is
  still not claimed. The known-height workspace resolves only its documented
  constrained reference frame and keeps human review explicit.

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

The FastAPI application currently exposes these route groups:

- health: `GET /api/v1/health`, `/health/live` and `/health/ready`;
- discovery and inventory: `GET /api/v1/capabilities`, `/runtime`, `/catalog`,
  `/discovery`, `/experiments` (compatibility alias) and
  `/workspaces/{workspace}`, plus
  `POST /api/v1/discovery/recommendations`;
- public examples: `GET /api/v1/samples` and the server-issued sample image,
  pair, known-height scene and planar-scene routes below `/api/v1/samples/...`;
- optional resources: `GET /api/v1/resources/sam21` and the explicitly
  confirmed `POST /api/v1/resources/sam21/install`;
- review interactions: `POST /api/v1/interactions/refine-line`,
  `/discover-planar-orientations` and `/preview-planar-relation`;
- immutable planar scenes: `POST /api/v1/planar/scenes`, `GET` a scene revision
  or one of its images, and `POST` a new revision or run below
  `/api/v1/planar/scenes/{scene_id}/{revision}`;
- execution: `GET/POST /api/v1/runs`, `POST /api/v2/runs`, `GET` a run or its
  events, inputs and artifacts, and `POST /api/v1/runs/{run_id}/cancel`;
- focused tools: `GET /api/v1/academy/single-view/status`,
  `POST /api/v1/academy/single-view/rays`, `POST /api/v1/pairs/analyze` and
  `GET /api/v1/semantics/status`.

The semantic status endpoint returns `available: false`; it exists so the UI
can disable unavailable stages rather than fabricate placeholder results.

`POST /api/v1/runs` queues work on one local device worker. Run manifests are
immutable experiment records in the sense that a replay creates a new ID;
their status and append-only progress events advance until a terminal state.
Inputs are deduplicated by checksum and artifacts are retrieved only through
their run descriptor.
