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

## Measured pair table and probability models

The measured workflow intentionally keeps predictors and outcomes in separate
files. In particular, the fitting commands never receive evaluation outcomes.
The full command lines and immutable input hashes belong in the run report.

```bash
python benchmarks/two_view_pose_probability/build_pair_table.py \
  --help

python benchmarks/two_view_pose_probability/run_probability_models.py \
  fit-predict \
  --features /path/to/features.jsonl \
  --training-outcomes /path/to/outcomes-development-calibration.jsonl \
  --output-dir /private/tmp/panorai-val018-models

python benchmarks/two_view_pose_probability/run_probability_models.py \
  fit-predict-lodo \
  --features /path/to/features.jsonl \
  --training-outcomes /path/to/outcomes-development-calibration.jsonl \
  --output-dir /private/tmp/panorai-val018-models-lodo
```

`fit-predict-lodo` fits the common capture and post-processing models three
times. Each fit excludes every outcome from one target dataset. If fewer than
five independent source components remain, the runner records that
cross-validation is unsupported and uses the fixed regularization declared in
the source instead of producing a misleading tuned estimate.

Evaluation is a separate operation which opens the sealed evaluation outcomes:

```bash
python benchmarks/two_view_pose_probability/run_probability_models.py \
  evaluate \
  --predictions /path/to/predictions.jsonl \
  --model-card /path/to/model-card.json \
  --evaluation-outcomes /path/to/outcomes-evaluation.jsonl \
  --output-dir /private/tmp/panorai-val018-evaluation
```

The quantitative paper figures are reproducible rather than hand-edited:

```bash
python benchmarks/two_view_pose_probability/render_paper_results.py \
  --analysis-table /path/to/analysis-table.jsonl \
  --evaluation /path/to/evaluation.json \
  --lodo-evaluation /path/to/lodo-evaluation.json \
  --release-rule /path/to/release-rule.json \
  --release-evaluation /path/to/release-rule-evaluation.json \
  --output-dir /private/tmp/panorai-val018-paper-results
```

E7 keeps threshold selection separate from evaluation-outcome access:

```bash
python benchmarks/two_view_pose_probability/select_release_rule.py freeze \
  --features /path/to/features.jsonl \
  --predictions /path/to/predictions.jsonl \
  --training-outcomes /path/to/outcomes-development-calibration.jsonl \
  --output /private/tmp/panorai-val018-e7/release-rule.json

python benchmarks/two_view_pose_probability/select_release_rule.py evaluate \
  --rule /private/tmp/panorai-val018-e7/release-rule.json \
  --features /path/to/features.jsonl \
  --predictions /path/to/predictions.jsonl \
  --evaluation-outcomes /path/to/outcomes-evaluation.jsonl \
  --output-dir /private/tmp/panorai-val018-e7
```

The E8 power and screening-burden plan is generated with:

```bash
python benchmarks/two_view_pose_probability/plan_prospective_confirmation.py \
  --release-evaluation /path/to/release-rule-evaluation.json \
  --output-dir /private/tmp/panorai-val018-e8-plan
```

See [`PAPER_RESULTS.md`](PAPER_RESULTS.md) for the paper-ready interpretation
and [`PROSPECTIVE_CONFIRMATION_PROTOCOL.md`](PROSPECTIVE_CONFIRMATION_PROTOCOL.md)
for the frozen acquisition and unsealing order.

The present results are post-hoc evidence. They do not by themselves establish
a release threshold; that requires a rule frozen on development/calibration
groups and confirmation on new independent capture groups.
