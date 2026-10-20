# Initial roadmap

The roadmap is intentionally small. It does not imply that the listed features
already exist.

## Implemented foundation: Catalog + Single View + Two View

- the installed versioned PanorAi catalog, generic typed Runner and one
  compatibility map for 22 historical experiment IDs;
- cardinality-aware routing and direct links;
- declarative parameter and artifact contracts;
- local asynchronous run queue, content-addressed inputs and persistent
  `scratch` history;
- Single View execution and type-compatible run comparison;
- Two View relative-pose inspection plus pair/single target selection;
- compatibility redirects for the previous geometry and segmentation URLs.

## 0. Academic geometry cases

Keep each Academy case as a small, versioned learning contract with: source
sections, learning goals, formulas, image/value parameters, required manual
tools, PanorAI public APIs, explicit library gaps, and a reproducible demo.

The notebook remains a local Workbench tool exposed from Single View. Its
manual constructions must not be presented as PanorAi estimators.

The path starts one step earlier with homogeneous coordinates, ideal points,
incidence, and homographies as a camera-model-free bridge to stitching. The
first interactive case is single-view geometry over ERP: canonical rays,
manual interpretation planes, great circles, antipodal vanishing directions,
and the orthogonality constraints that motivate calibration. Continue only as
matching PanorAI Main capabilities exist or are added and validated:

1. planar projective transformations and rectification;
2. calibrated camera models and metric measurements from one panorama;
3. two-view epipolar geometry and relative pose;
4. homographies, trifocal geometry, and multi-view reconstruction;
5. uncertainty, robust estimation, degeneracy, and numerical diagnostics.

The Workbench may visualize a manual derivation, but it must never present that
derivation as a PanorAI estimator. Missing robust computer-vision operations
belong to a PanorAI library agent and become executable here only through a
documented public API.

## 1. Pair inspection foundation

- call PanorAi for spherical features, matches, and relative pose;
- inspect KPs individually or as a response-ranked subset;
- expose inlier/outlier decisions, `R`, unit-`t`, quality rejection reasons,
  and per-match epipolar residual;
- distinguish epipolar locus from a triangulated reprojection.

## 2. Contract fixture

- choose a versioned JSON record;
- create synthetic seam, pole, and multi-view examples;
- verify spherical-to-gnomonic-to-spherical round trips independently.

## 3. Single-panorama segmentation/annotation editor

- load an ERP and optional support mask;
- navigate with a transient gnomonic viewport;
- create, edit, delete, and review one region type;
- save and reopen the same annotation without geometric drift.

## 4. Registered multi-view detection

- show corresponding registered panoramas;
- link per-view regions to one instance hypothesis;
- display reprojection or track support without claiming a precise 3-D point;
- preserve ambiguity instead of forcing a match.

Add CAM-region stages only after the pairwise geometric evidence can be
inspected reliably:

- dense semantic channels with explicit support;
- FOV-anchor generation and local NMS audit;
- A→B and B→A region transfer;
- track, angular-overlap, appearance, and semantic cues shown separately;
- retained/suppressed/fused hypothesis decisions.

## 5. Dataset export

- export accepted annotations and provenance;
- derive planar training views only as a reproducible export step;
- keep train/validation/test splits and source licenses explicit.

Model-assisted proposals, collaborative review, and large-dataset operations
are later decisions, contingent on evidence from the small workflow.
