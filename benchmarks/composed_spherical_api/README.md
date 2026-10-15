# Composed spherical API benchmark

This benchmark separates four NumPy CPU paths: construction plus execution,
warm execution of one pipeline, cache miss plus atomic write, and verified cache
hit. It records every timing sample, median, P95, process peak RSS, environment,
source commit, installed-package origin, and exact wheel SHA-256.

Run it through the installed-consumer harness so the import cannot resolve to
the checkout:

```bash
python scripts/run_installed_consumer.py \
  --wheel /absolute/path/to/panorai.whl \
  --consumer benchmarks/composed_spherical_api/benchmark.py \
  --source-root "$PWD" -- \
  --wheel /absolute/path/to/panorai.whl \
  --source-commit "$(git rev-parse HEAD)" \
  --output /absolute/path/to/composed-pipeline-benchmark.json
```

The runner output and JSON file are development evidence. They do not establish
scientific quality for SLIC, DoG, stereo, CAM, or SAM, and they do not promote
an Experimental API.
