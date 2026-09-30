# Cubemap, batch, and autograd

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
