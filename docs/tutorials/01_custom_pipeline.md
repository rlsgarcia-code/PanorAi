# Cubemap, batch, and autograd

See the {ref}`capability-map-cubemap-batch` section of the spherical
computer-vision guide for what PanorAi projects, when an ERP is reconstructed,
and why neither cubemap conversion nor batched projection is a spherical
convolution.

**PanorAi-specific:** fixed cubemap face order/orientation, canonical ray
sampling, batch-preserving projection, explicit support and inverse ERP
reconstruction. The operation or model applied to each face remains caller-
owned.

## Cubemap round trip

Cubemap faces use the immutable order `front`, `right`, `back`, `left`, `up`,
`down`. This continuous RGB example uses bilinear interpolation and reconstructs
an `HWC` panorama.

```{literalinclude} ../../scripts/run_documentation_examples.py
:language: python
:dedent: 4
:start-after: DOCS_CUBEMAP_START = None
:end-before: DOCS_CUBEMAP_END = None
```

## Torch batch and input gradients

Torch is optional. Batched image tensors use `NCHW`; the result preserves
batch, channels, dtype, and device. Autograd is supported for input values, not
for geometric parameters.

```{literalinclude} ../../scripts/run_documentation_examples.py
:language: python
:dedent: 8
:start-after: DOCS_TORCH_START = None
:end-before: DOCS_TORCH_END = None
```

Execute every example, including Torch, with:

```console
python scripts/run_documentation_examples.py --source-checkout --torch
```
