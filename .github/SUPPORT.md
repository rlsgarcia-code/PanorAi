# Support

Use GitHub Issues for reproducible defects, documentation gaps, and narrowly
scoped feature requests. Include the PanorAi version, Python version, platform,
installation method, public contract, minimal code, and complete error text.

Questions about research conclusions should identify the benchmark catalog
entry, dataset facet, split, and generating commit. Project-specific dataset
preparation, private checkpoints, unpublished dissertation results, and custom
deployment support are outside the public product contract.

Before reporting an import problem, verify whether the capability requires an
optional extra. `panorai.graph` and the base package must import without Torch,
Open3D, PyCOLMAP, checkpoints, or database services.

Security reports belong in the private channel described in `SECURITY.md`, not
in public support threads.
