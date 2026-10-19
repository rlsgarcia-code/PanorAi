# PanorAi Workbench

Local inspection environment for the public PanorAi API. The **Catalog** is the
discovery entrance; **Single View** and **Two View** are the execution benches.
The Catalog mirrors the families and roles published by the exact installed
PanorAi catalog; Single View and Two View show only the separate, locally
registered experiment adapters compatible with each bench.

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

The same fail-closed rule applies across PanorAi versions. Legacy experiment
IDs remain stable, but each adapter is bound to a catalog `surface_id` and a
declared public symbol. Missing surfaces or symbols are blocked rather than
redirected to private code. The main-line adapters bind SLIC and DoG both at
the Application level (`panorai.applications`) and at their lower public
operation/detector surfaces. Relative Pose binds to
`panorai.estimators.estimate_relative_pose`; stereo adapters likewise use only
cataloged public Applications. No Workbench-side scientific substitute is
used.

## Reproducible runs

The browser sends only a registered experiment identifier and parameters from
its declarative schema. It cannot request an arbitrary Python import. Every new
run is stored under the local `scratch` project with its legacy
`experiment_id`, explicit `adapter_id`, `surface_id`, catalog contract/release,
input SHA-256 checksums, resolved parameters, PanorAi runtime fingerprint and source commit, events,
metrics, warnings and versioned artifact descriptors. Repeating an experiment
creates a new run; it never overwrites the previous one.

The default store is `outputs/workbench/projects/scratch`. Set
`PANORAI_WORKBENCH_HOME` to use another local root. Dataset inputs remain
content-addressed references in that store and are not added to the repository.

## Installed catalog and adapter overlay

`GET /api/v1/catalog` returns two different collections. `surfaces` is derived
only from the installed `PublicCatalog`; `experiments` is the local execution
overlay. `GET /api/v1/workspaces/{single|two}` filters only `experiments`.
The browser may submit a legacy adapter ID and validated parameters, never an
import path or arbitrary callable. New runners require an explicit backend
registration and contract test.

### Rule for new PanorAi features

1. Publish the feature in the versioned PanorAi public-surface contract, with
   its stable `surface_id`, family, public import, requirements and
   documentation links, then synchronize the packaged catalog resource.
2. The Catalog will discover and display it from the installed distribution;
   no Workbench frontend list or routing change is allowed or required.
3. If interactive execution is useful, add a Workbench adapter keyed by that
   `surface_id`, bind only public symbols, register a narrowly scoped runner in
   `ALLOWLISTED_RUNNERS`, and cover the binding and dispatch with contract
   tests. Discovery does not grant execution automatically.
4. Keep manual UI tools in the Workbench and preserve existing adapter IDs for
   saved URLs and run histories. Never bridge a catalog gap with a private
   PanorAi import.

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

- Multi View, tracks across three or more views and a multi-image runner are
  not implemented;
- cancellation is explicit and prevents queued/final artifact publication,
  but an already executing native call can only release its memory when that
  public call returns;
- segmentation/annotation remains a manual Workbench tool for one panorama;
- the canonical manual region is centre bearing plus horizontal/vertical FOV
  with zero roll;
- CAM, semantic region transfer, fused object hypotheses, SAM masks, polygons,
  collaborative review, and dataset split management are not implemented yet;
- imported annotations are not silently treated as reviewed ground truth.
