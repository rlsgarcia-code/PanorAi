# Quick start

This example is self-contained: it synthesizes a small floating RGB panorama,
projects one view, and keeps geometric support explicit. Input layout is
`HWC`, dtype is `float32`, angles are degrees, and bilinear interpolation is
appropriate because RGB is continuous here.

```{literalinclude} ../../scripts/run_documentation_examples.py
:language: python
:dedent: 4
:start-after: DOCS_FUNCTIONAL_START = None
:end-before: DOCS_FUNCTIONAL_END = None
```

The immutable projector form uses the same engine and contract:

```{literalinclude} ../../scripts/run_documentation_examples.py
:language: python
:dedent: 4
:start-after: DOCS_PROJECTOR_START = None
:end-before: DOCS_PROJECTOR_END = None
```

The 3.0 container workflow remains available as a compatibility adapter:

```{literalinclude} ../../scripts/run_documentation_examples.py
:language: python
:dedent: 4
:start-after: DOCS_CONTAINER_START = None
:end-before: DOCS_CONTAINER_END = None
```

The preserved high-level workflow can sample several faces, apply user
processing, and reconstruct the panorama with explicit support:

```{literalinclude} ../../scripts/run_documentation_examples.py
:language: python
:dedent: 4
:start-after: DOCS_WORKFLOW_START = None
:end-before: DOCS_WORKFLOW_END = None
```

Run all canonical NumPy examples with:

```console
python scripts/run_documentation_examples.py --source-checkout
```
