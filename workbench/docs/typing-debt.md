# Workbench typing debt

The machine-readable source of truth is
`workbench/contracts/typing-debt-v1.json`. The file records executable public
symbols that predate the automatic Runner but still have an ambiguous or
unbound contract. It is a burn-down baseline, not an allow-list:

- `python workbench/scripts/check_automatic_components.py` rejects every new
  blocked executable on an existing explicit surface;
- debt is never allowed on a namespace's generated Experimental catchall;
- a changed diagnostic fails, so an existing entry cannot silently cover a
  different defect;
- `--update` may remove resolved entries but refuses to add entries.

## Current map

The baseline is empty: 0 blocked surface entries and 0 approved debt entries.
Every installed symbol is now either automatically runnable or deliberately
inspect-only with a machine-readable reason. Opaque model backends, callbacks,
protocols, abstract bases, and inputs that must come from an upstream typed
result are inspect-only rather than false-ready.

The empty JSON baseline remains tracked. Any future blocked executable is
therefore a new regression and fails CI; there is no historical allowance to
hide behind.

## Resolved in the burn-down

Typing-only changes made the following public families automatically runnable
without changing scientific behavior:

- spherical region analysis and morphology;
- `Resample`, `AreaAverage`, `Filter2D`, `Rotate`, `SphericalNMS`, and the
  functional `area_average`;
- camera/world frame transforms through both canonical and compatibility
  surfaces;
- relative-pose estimation and tangent Sampson error;
- spherical keypoint deduplication;
- multi-image stitching, panoramic stereo, and the legacy DoG facade;
- pipeline Stage inputs, routing containers, and region/instance stages;
- structured sequences of public dataclasses and named image mappings;
- the abstract NumPy `Module` base.

Arrays use public containers where one exists and `Annotated[...,
"panorai:..."]` only for otherwise ambiguous bearings, masks, kernels, ranges,
rotations, labels, and rasters.

## Merge rule

A new public executable must be either:

1. `ready`, with every required input and output described by a supported
   public type; or
2. deliberately `inspect`, because it is abstract, stateful, externally
   acquired, or requires an opaque callback/session/backend.

It may not enter as `blocked`. Configuration/result/helper symbols remain
inspect-only and do not create false typing debt.
