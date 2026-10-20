# PanorAi Workbench

Local inspection environment for the public PanorAi API. The **Catalog** is the
discovery entrance; the universal **Runner** derives inputs and configuration
from public type annotations and executes catalog-issued component IDs. **Single
View** and **Two View** remain task-focused shortcuts, and their legacy adapter
IDs remain valid for saved URLs and run histories.

## Status

This directory contains an early local prototype. It is not part of PanorAi's
Stable API and is not yet a production evaluation or annotation system.

The navigation hierarchy is:

- **Catalog:** consumes `panorai.catalog.public_catalog()` from the exact
  PanorAi installation imported by the backend. New public surfaces and new
  families appear automatically, even when the Workbench has no adapter for
  them. Learn, Help, Reference and broader site-context links come from the
  same installed catalog.

- **Single View:** executes compatible Applications, Pipeline templates,
  Spherical Ops and Geometry over one ERP. It keeps a central image canvas,
  declarative parameters, effective configuration, stage events, immutable
  run history and type-compatible A/B comparison. Manual annotation is a local
  Workbench tool inside this bench; it is not presented as a PanorAi API.

- **Two View:** accepts an ERP pair. Pair-native components include relative
  pose and stereo; single-view components may be applied to A, B or both. The
  detailed relative-pose surface preserves keypoints, matches, `R`, unit-`t`,
  residuals, acceptance diagnostics and the selected epipolar great circle.

- **Multi View — em breve:** remains visible as a future destination, but has
  no runner in this phase.

The CAM, region-transfer, hypothesis-fusion, SAM, Mask R-CNN and optional Torch
entries remain explicit as dependency/input gaps when their public runner is
not usable in the active installation.

The same fail-closed rule applies across PanorAi versions. A generic run is
resolved only from a component ID emitted by the installed catalog; the browser
cannot provide a module or symbol path. Concrete public Applications,
operations, Stages and estimators become runnable when their required inputs
and result are typed with supported public contracts. Config/result/protocol
types remain visible as inspect-only. Missing optional dependencies produce an
explicit unavailable state and are never installed or downloaded by the
Workbench. No Workbench-side scientific substitute is used.

## Reproducible runs

The browser sends only a catalog-issued component ID, typed configuration/input
values and an arbitrary ordered upload list. It cannot request an arbitrary
Python import. Every new run is stored under the local `scratch` project with
its `component_id` (or legacy `experiment_id`/`adapter_id`), `surface_id`,
catalog contract/release, input SHA-256 checksums, resolved values, runtime
fingerprint, events, metrics, warnings and versioned artifacts. Exact NumPy
outputs are kept separately from derived previews. Repeating a component
creates a new run; it never overwrites the previous one.

The default store is `outputs/workbench/projects/scratch`. Set
`PANORAI_WORKBENCH_HOME` to use another local root. Dataset inputs remain
content-addressed references in that store and are not added to the repository.

## Installed catalog and adapter overlay

`GET /api/v1/catalog` returns three separate collections. `surfaces` comes from
the installed `PublicCatalog`; `components` is the type-derived automatic
projection; `experiments` is the compatibility adapter overlay. `POST
/api/v2/runs` executes an automatic component, while `POST /api/v1/runs`
preserves old adapters. `GET /api/v1/workspaces/{single|two}` continues to
filter the legacy overlay.

### Rule for new PanorAi features

1. Export the feature through the namespace `__all__` and give concrete
   executable inputs/results unambiguous public types. `Annotated[np.ndarray,
   "panorai:..."]` is reserved for array semantics not expressed by a public
   container.
2. Run `python scripts/sync_public_catalog.py`. Existing explicit
   classifications win; a new symbol enters the namespace's Experimental
   catchall and never receives Stable status by inference. The policy covers
   Applications, Ops, Geometry, Features, Estimators, Pipeline, Stereo,
   Reconstruction, SLAM, Workflow and the other audited public namespaces;
   no per-symbol catalog list is edited.
3. Run `python workbench/scripts/check_automatic_components.py`. A new concrete
   executable with `Any`, an ambiguous required array, or an unsupported input
   blocks CI. Auxiliary/config/result types are inspect-only. The inherited
   burn-down baseline and its exact update rules are documented in
   [`docs/typing-debt.md`](docs/typing-debt.md); `--update` can only remove
   resolved entries.
4. Open `/runner.html`: no frontend list, dispatch branch or scientific adapter
   is required. Add a visualization override only when the shared semantic
   families cannot faithfully present the typed result.
5. Keep manual annotation/geometry tools in the Workbench. Old experiment IDs
   live in one compatibility map and point to the automatic component when
   parity is proven; saved v1 runs remain readable. Never bridge a catalog gap
   with a private import.

## Geometry and data boundary

The canonical record belongs to the sphere. Gnomonic crops are editing views,
not independent source images. Model suggestions may accelerate annotation,
but are not ground truth until a person accepts or edits them.

[PanorAi](https://github.com/rlsgarcia-code/PanorAi) supplies projection
geometry, spherical features, estimators, and Experimental object-hypothesis
methods. The `workbench/` application owns inspection workflow, annotation state,
experiment sessions, and the user interface. It must not move training data,
dataset-specific policy, or UI dependencies into the PanorAi projection core.

No dataset images, private panoramas, checkpoints, or generated labels are
stored in the tracked Workbench source.

## Entry points

- `/` — Catalog
- `/runner.html` — universal type-derived Runner
- `/single-view.html` — Single View
- `/two-view.html` — Two View
- `/geometry.html` — compatibility redirect to Relative Pose in Two View
- `/segmentation.html` — compatibility redirect to annotation in Single View

## Design notes

- [Workbench architecture](docs/workbench-architecture.md)
- [Annotation model](docs/annotation-model.md)
- [Small implementation roadmap](docs/roadmap.md)

## Run locally

The browser frontend and Python backend run separately. Run the commands from
this `workbench/` directory. During development, install the parent PanorAi
checkout so the backend exercises the exact source under review:

```bash
npm install
npm run dev

python -m pip install -e ..
python -m pip install -e '.[dev]'
uvicorn panorai_workbench.api:app --app-dir backend --host 127.0.0.1 --port 8000
```

The development proxy sends `/api` to `http://127.0.0.1:8000` by default. To
use another backend port without changing files, set the proxy target when
starting Vite:

```bash
PANORAI_WORKBENCH_API_URL=http://127.0.0.1:8001 npm run dev -- --port 5174
```

To test the latest remote `main` instead of a neighbouring checkout, replace
the first Python install command with:

```bash
python -m pip install --force-reinstall \
  'panorai @ git+https://github.com/rlsgarcia-code/PanorAi.git@main'
```

The Workbench requires the installed-catalog API introduced after `v3.8.0`, so
that published release is intentionally rejected by its dependency floor.
Once a release containing `panorai.catalog` is published, pin that exact
release for distribution validation. The runtime panel distinguishes a VCS
installation, an editable checkout, and an installed distribution and shows
the resolved commit whenever package metadata provides it.

Validation commands:

```bash
npm run check
npm test
npm run build
pytest -q backend/tests
python scripts/check_public_imports.py
python scripts/check_automatic_components.py
python scripts/smoke_scientific.py
```

The Catalog header and runtime panel display `panorai.__file__`, version,
installation type, Python, optional dependency versions and available devices.
An editable checkout is explicitly labeled as editable and is never presented
as a published wheel.

### Validate an installed PanorAi wheel in isolation

Build the PanorAi wheel from the repository root, then create a temporary
environment outside the checkout. Install that exact wheel and the Workbench
without resolving PanorAi again:

```bash
python -m build --wheel --outdir /tmp/panorai-workbench-wheel ..
python -m venv /tmp/panorai-workbench-wheel-env
/tmp/panorai-workbench-wheel-env/bin/python -m pip install \
  /tmp/panorai-workbench-wheel/panorai-*.whl
/tmp/panorai-workbench-wheel-env/bin/python -m pip install -e '.[dev]' --no-deps
/tmp/panorai-workbench-wheel-env/bin/python scripts/validate_installed_panorai.py \
  --forbid-source-root ..
PANORAI_WORKBENCH_HOME=/tmp/panorai-workbench-runs \
  /tmp/panorai-workbench-wheel-env/bin/python -m pytest -q backend/tests
```

The validator fails if `panorai` resolves inside the forbidden source root,
prints the exact module path/version, resolves the public catalog, and executes
a real public gnomonic projection. Install the Workbench runtime dependencies
in the temporary environment when they are not already available; `--no-deps`
is intentional so it cannot replace the wheel under test with the checkout.

## Current limitations

- Multi View and tracks across three or more views are not implemented; the
  universal Runner already accepts a dynamic number of typed inputs;
- cancellation is explicit and prevents queued/final artifact publication,
  but an already executing native call can only release its memory when that
  public call returns;
- segmentation/annotation remains a manual Workbench tool for one panorama;
- the canonical manual region is centre bearing plus horizontal/vertical FOV
  with zero roll;
- CAM, semantic region transfer, fused object hypotheses, SAM masks, polygons,
  collaborative review, and dataset split management are not implemented yet;
- imported annotations are not silently treated as reviewed ground truth.
