# Spherical observation workflow benchmark

This benchmark measures lifting and reduction separately for 96 Fibonacci
faces. It reports every elapsed-time sample, median, P95, and Python-tracked
peak memory together with PanorAi/Python/platform versions and the complete
workload configuration.

```bash
python benchmarks/spherical_observations/benchmark.py \
  --faces 96 --warmup 1 --repeats 5 --output result.json
```

The result is performance evidence for the recorded commit and hardware, not a
runtime guarantee. Correctness is established by the geometry and workflow
tests; this script does not relax them.
