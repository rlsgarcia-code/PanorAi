# Multimodal arbitrary-N consumer

This standalone consumer exercises the public PanorAi object workflow from an
installed wheel. It intentionally lives outside the `panorai` package and does
not import private modules.

The flow uses fourteen deterministic Fibonacci views rather than assuming a
six-face cubemap:

```text
ERP RGB + radial depth + labels + validity
  -> 14 rectangular gnomonic views
  -> stateless image model
  -> Gaussian/average/closest ERP reconstruction
```

It checks NumPy `HWC`/`HW` and Torch `NCHW`, explicit depth validity,
categorical label preservation, source immutability, the JSON-friendly workflow
description, and exact agreement between the expanded workflow and
`process_views()`.

Run it through `scripts/run_installed_consumer.py` with an exact local wheel;
the runner installs that wheel into a temporary target and executes from a
temporary directory so the checkout cannot satisfy `import panorai`.
