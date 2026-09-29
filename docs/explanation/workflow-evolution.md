# Workflow API evolution

Status: design proposal for post-3.1 validation. The names and signatures on
this page are illustrative and are not part of the PanorAi 3.x compatibility
contract.

## Purpose

PanorAi's panorama, face, sampler, projector, and blender abstractions express
a useful workflow:

```text
Panorama → sample views → process faces → reconstruct panorama
```

The proposal preserves that model while moving new behavior onto the canonical
`panorai.geometry` engine. It does not replace the array-first API and does not
remove the 3.0 containers.

## Design principles

1. Geometry has one owner. Workflow objects compose `panorai.geometry`; they do
   not reimplement projection formulas or conventions.
2. Configuration is explicit and immutable. Running one workflow must not
   mutate global defaults or change a later workflow implicitly.
3. Modality is part of the contract. RGB, features, labels, masks, and radial
   range may require different interpolation and validity policies.
4. Support and validity remain distinct. A workflow must preserve both through
   projection, user processing, and reconstruction.
5. NumPy and Torch share the public contract. Batch, dtype, layout, device, and
   input-value autograd behavior must be stated and tested.
6. Extension points fail visibly. An unavailable or invalid plugin produces an
   actionable error rather than an unattached component or silent fallback.
7. Convenience must be reproducible. A complete plan can be inspected,
   serialized, compared, and included in provenance.

## Proposed concepts

### Channel

A channel binds data to semantics rather than relying on its name or numeric
values. An illustrative channel description could contain:

- `data`: NumPy array or Torch tensor;
- `modality`: continuous, labels, mask, or radial range;
- `validity`: optional explicit validity mask;
- `interpolation`: inferred safely from modality or supplied explicitly;
- `invalid_policy` and `min_valid_weight`: explicit continuous-data behavior;
- `units`: optional metadata such as metres for radial range.

The implementation must reject unsafe combinations, including bilinear
interpolation of boolean or integer categorical data.

### View plan

A view plan combines a sampler with a gnomonic view template. The sampler owns
only view placement; the template owns FOV, roll, and output shape. Each sampled
view becomes a normal canonical `GnomonicSpec`.

### Projection plan

A projection plan composes:

- a view plan;
- per-modality projection policies;
- an optional reconstruction blender;
- an output panorama shape;
- optional execution controls such as chunk size or grid caching.

The plan is immutable. Projecting the same compatible input twice must not
change its configuration or depend on another plan's execution.

### Projected batch

Projection returns a structured batch rather than a bare list. It retains:

- ordered view specifications;
- channel data for every view;
- geometric support;
- validity and valid weight where applicable;
- source layout, dtype, and device metadata;
- the plan needed for reconstruction.

Users can map a function or model across views without discarding the metadata
required to reconstruct safely.

## Illustrative user experience

The intended shape is deliberately close to the 3.0 workflow:

```python
plan = ProjectionPlan(
    views=FibonacciViews(count=20, fov_deg=90, shape_hw=(512, 512)),
    blender=GaussianBlender(sigma=1.2),
)

panorama = SphericalChannels(
    rgb=Channel(rgb, modality="continuous"),
    labels=Channel(labels, modality="labels"),
    depth=Channel(
        depth_m,
        modality="radial-range",
        validity=depth_is_valid,
        units="m",
    ),
)

views = plan.project(panorama)
processed = views.map(model)
result = plan.reconstruct(processed)
```

These names are placeholders. A public design must first validate whether
existing users prefer methods on `EquirectangularImage`, a standalone plan, or
both through a thin adapter.

## Extension contract

String registries remain useful for configuration files and command-line
applications, but strings should resolve to typed protocols:

- a sampler returns deterministic, ordered view placements and records any
  random seed;
- a projector consumes canonical specs and returns canonical results;
- a blender consumes equal-shaped values plus explicit masks and reports output
  support;
- a plugin publishes its name, version, supported modalities, backend support,
  and stability tier.

Direct object composition must work without global registration. Package entry
points may provide optional discovery, but importing `panorai.geometry` must
remain free of plugin and optional-backend imports.

## Compatibility path

PanorAi 3.x keeps `EquirectangularImage`, `GnomonicFace`, `GnomonicFaceSet`,
`DataFactory`, `PanoraiFactory`, `ConfigManager`, and the `attach_*` methods.
Evolution should proceed in this order:

1. keep current names and document their stability tier;
2. route their geometry through canonical specs and results where behavior is
   equivalent;
3. let containers accept explicit component objects as well as registry names;
4. add opt-in adapters from containers to a validated projection plan;
5. consider deprecation only in a 4.0 process with migration evidence.

No compatibility adapter may silently choose a lossy modality policy.

## Evidence required before a 3.2 commitment

The roadmap intentionally uses adoption evidence after 3.1 to decide 3.2
scope. A typed workflow becomes a release candidate only after:

- at least two realistic consumer workflows are recorded;
- RGB, labels, masks, radial range, and mixed-modality cases have executable
  acceptance examples;
- NumPy and Torch `HW`, `HWC`/`CHW`, and batched layouts are defined;
- seams, poles, support boundaries, and invalid regions are tested;
- plugins are exercised through real implementations rather than test-only
  stubs;
- serialization and provenance round-trip without global state;
- cold and cached performance are measured before any speed claim;
- 3.0 container adapters pass the existing compatibility suite;
- installed-wheel examples run outside the repository.

## Non-goals

- A second geometry engine.
- Training pipelines, datasets, checkpoints, or model-specific orchestration.
- Inferring masks from values, channel names, or dtypes alone.
- Making Torch or Open3D mandatory.
- Removing public 3.0 names during the 3.x line.
- Advertising performance without reproducible measurements.

## Phased plan

### Phase 0 — 3.1 communication

Curate the README, retain the existing abstractions, remove unsafe
mixed-modality guidance, and describe stable versus compatibility surfaces.

### Phase 1 — consumer discovery

Collect small, reproducible examples from panorama inference, semantic labels,
radial depth, and differentiable Torch use. Record which current abstractions
help and where hidden state or manual mask handling causes errors.

### Phase 2 — contract prototype

Write public type contracts and failing acceptance tests before implementation.
Resolve names, serialization, plugin discovery, modality defaults, and batch
semantics with no new projection mathematics.

### Phase 3 — canonical implementation

Implement the smallest plan and channel layer on top of `panorai.geometry`,
then add container adapters. Verify real NumPy/Torch integration and installed
artifacts.

### Phase 4 — performance and ecosystem

Only after correctness, evaluate reusable grid caches, device-local grids,
chunked batches, and third-party plugins using published benchmark methodology.
