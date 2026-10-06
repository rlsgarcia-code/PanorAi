# Quick start

This example is self-contained: it synthesizes a small floating RGB panorama,
projects one view, and keeps geometric support explicit. Input layout is
`HWC`, dtype is `float32`, angles are degrees, and bilinear interpolation is
appropriate because RGB is continuous here.

See the {ref}`capability-map-quick-start` section of the spherical
computer-vision guide for the distinction between a
single ERP→gnomonic projection, inverse projection, and the complete
views→processing→ERP workflow used below.

**PanorAi-specific:** canonical sphere rays, periodic ERP sampling, projection
support and validity, and the inverse geometry used to reconstruct ERP.

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

The Stable object workflow can sample several faces, apply user processing,
reconstruct the panorama, and expose its resolved choices:

```{literalinclude} ../../scripts/run_documentation_examples.py
:language: python
:dedent: 4
:start-after: DOCS_WORKFLOW_START = None
:end-before: DOCS_WORKFLOW_END = None
```

## Scanner shadow caps

Some scanners omit a south-polar cap. Declare both its angular size and
whether its black rows already exist; PanorAi never guesses from pixel color:

```python
from panorai.data import EquirectangularImage

# Cropped raster: the 30° cap is missing and must be materialized once.
pano = EquirectangularImage(
    observed_rgb,
    shadow_angle=30.0,
    shadow_padded=False,
)
pano.preprocess()

# If the file already includes the black band, declare it explicitly instead.
padded = EquirectangularImage(
    padded_rgb,
    shadow_angle=30.0,
    shadow_padded=True,
)
```

Both objects now carry explicit geometric support that excludes the
south-polar cap while preserving valid black pixels elsewhere. Calling
`preprocess()` again cannot add the band twice. See
{doc}`../how_to/preprocess_containers` for the state table and row-count rule.

Run all canonical NumPy examples with:

```console
python scripts/run_documentation_examples.py --source-checkout
```
