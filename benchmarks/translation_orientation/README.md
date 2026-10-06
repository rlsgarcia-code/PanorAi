# Translation-orientation benchmark

This directory contains the reproducible GEO-012 comparison between the
historical positive-depth-count decomposition of an Essential matrix and the
parallax-weighted policy.

The corpus is generated analytically from known 3D points and an independently
recorded SE(3) transform. No PanorAi estimator is used to construct the ground
truth. Every method receives the same unit-bearing arrays for a given
condition and seed.

Run from the repository root:

```console
python benchmarks/translation_orientation/run_benchmark.py \
  --output benchmarks/translation_orientation/results
```

The runner writes:

- `raw.csv`: one paired observation per method, condition, and seed;
- `summary.json`: aggregate sign accuracy, selective precision, coverage,
  pose errors, runtime, source identity, environment, and configuration;
- `TECHNICAL_REPORT.md`: an article-style report generated from the exact
  summary.

Tracked CSV output uses LF line endings on every platform so byte-level hashes
remain stable in Git checkouts.

[`ALGORITHM_PAPER.md`](ALGORITHM_PAPER.md) is the self-contained methods paper
for the complete current $R,t$ estimator: five-point hypotheses, spherical
RANSAC scoring, nonlinear LO/IRLS refinement, Essential decomposition,
parallax-weighted orientation, reported costs, and limitations. The generated
technical report remains the source of measured tables and run identity.

The orientation-only experiment isolates the four-way decomposition using the
known Essential matrix but noisy observations. The end-to-end experiment also
runs five-point hypothesis generation, robust scoring, and nonlinear
refinement. The `unobservable` condition is a negative control: a method should
abstain rather than manufacture confidence when baseline/depth is too small.

These synthetic results are development evidence. They do not replace a
metadata-blind replay of the frozen VAL-002 real-image corpus.
