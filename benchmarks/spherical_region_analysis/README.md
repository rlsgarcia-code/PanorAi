# Spherical region-analysis performance

This deterministic synthetic benchmark records raw timings, median, P95 and
process peak RSS for connected components, physical measurements, ordered
contours and sampled medial axes at 256×512 and 512×1024. It is a bounded
performance gate, not a scientific accuracy evaluation.

Run from a source checkout:

```bash
python benchmarks/spherical_region_analysis/benchmark.py \
  --output /tmp/panorai-region-performance.json
```

For release evidence, install the exact candidate wheel outside the checkout,
invoke the same script with `--require-installed --source-root PATH`, and retain
the output beside the wheel checksum and environment manifest.
