# Two-view pose probability study

This directory defines the VAL-018 academic validation program for calibrated
acceptance and precision probabilities in PanorAi's spherical two-view R,t
pipeline.

The study keeps two models separate:

- capture probability from operator-controllable or operator-visible inputs;
- post-processing precision from frontend and estimator diagnostics.

See [STUDY_PROTOCOL.md](STUDY_PROTOCOL.md) for the preregistered experiments,
outcomes, permitted predictors, leakage rules, validation design and academic
deliverables.

Run E0 with identity-bearing JSONL sources:

```bash
python benchmarks/two_view_pose_probability/run_census.py \
  --source /path/to/public-predictions.jsonl \
  --source /path/to/p74-pairs-method-inputs.jsonl \
  --output-dir /private/tmp/panorai-val018-e0
```

The runner inventories canonical Matterport360, Stanford2D3D and P74 pairs at
image, pair and group levels, rejects duplicate or reversed pairs, merges
groups connected by a reused image, and freezes deterministic component-safe
development/calibration/evaluation splits before any model is fitted. It reads
identity fields only and does not copy dataset bytes or ground-truth poses.

## Paper narrative figures

The [`figures`](figures) directory contains three conceptual, generated assets:

- `graphical-abstract-two-view.png`: the two-view geometry and the two evidence
  stages;
- `capture-boundary-conditions.png`: qualitative capture regimes;
- `post-processing-evidence.png`: matches, robust geometry and post-estimation
  diagnostics.

Their provenance, intended roles and complete prompts are recorded in
[`figures/FIGURES.md`](figures/FIGURES.md). They are illustrative only; measured
plots and thresholds must be produced from the frozen experiment outputs.
