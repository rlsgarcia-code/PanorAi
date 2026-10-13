# Spherical core-operation benchmark

This benchmark compares the exact NumPy reference and the optional fused C++
backend for `panorai.core.numpy.SphericalConv2d`. It records every timing
sample, median, nearest-rank P95, Python-traced peak memory, process maximum-RSS
delta, workload, package origin, versions, and hardware metadata.

Run it against an installed wheel from outside the source checkout:

```bash
python benchmark_spherical_conv.py --repeats 15 --warmup 10 --output result.json
```

The result describes one artifact and machine; it is not a runtime guarantee.
Numerical equivalence is established separately by the native convolution
tests.
