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

[`CAPTURE_STANDARD.md`](CAPTURE_STANDARD.md) translates the frozen variables
and evidence boundaries into a scanner-assisted and RGB-only acquisition
workflow, while keeping capture eligibility, estimator acceptance, and
selective release as separate gates.

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

It also contains nine deterministic narrative and quantitative figures covering
the two-model design, evidence census, overlap response, calibration, transfer,
post-model ablation, selective-rule verdict, optimized-main mechanism replay,
and prospective confirmation plan. Their recommended paper order, provenance,
captions, and checksums are recorded in
[`figures/FIGURES.md`](figures/FIGURES.md). The three generated assets are
illustrative only; measured plots and thresholds come from frozen experiment
outputs.

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
  --model-card /path/to/model-card.json \
  --lodo-evaluation /path/to/lodo-evaluation.json \
  --release-rule /path/to/release-rule.json \
  --release-evaluation /path/to/release-rule-evaluation.json \
  --output-dir /private/tmp/panorai-val018-paper-results
```

The paper-level evidence census, model story, optimized-main replay, and
prospective-design figures are rendered with:

```bash
python benchmarks/two_view_pose_probability/render_narrative_figures.py \
  --census /path/to/census.json \
  --replay-summary /path/to/unified-replay-summary.json \
  --prospective-plan /path/to/prospective-power-plan.json \
  --output-dir /private/tmp/panorai-val018-narrative-figures
```

E7 keeps threshold selection separate from evaluation-outcome access:

```bash
python benchmarks/two_view_pose_probability/select_release_rule.py freeze \
  --features /path/to/features.jsonl \
  --predictions /path/to/predictions.jsonl \
  --training-outcomes /path/to/outcomes-development-calibration.jsonl \
  --post-model post-precise-raw-score \
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

The real-pair taxonomy and local, non-redistributable contact sheet are built
from the frozen sources with:

```bash
python benchmarks/two_view_pose_probability/build_failure_taxonomy.py \
  --analysis-table /path/to/analysis-table.jsonl \
  --probability-predictions /path/to/predictions.jsonl \
  --public-predictions /path/to/public-predictions.jsonl \
  --public-views /path/to/views-evaluation.jsonl \
  --p74-pairs /path/to/pairs-method-inputs.jsonl \
  --output-dir /private/tmp/panorai-val018-failure-taxonomy
```

Representative mechanisms can be rerun under the exact optimized wheel route
without importing PanorAi from the checkout:

```bash
python benchmarks/two_view_pose_probability/prepare_unified_taxonomy_replay.py \
  --taxonomy /path/to/failure-taxonomy.json \
  --public-predictions /path/to/public-predictions.jsonl \
  --public-evaluation /path/to/public-evaluation.jsonl \
  --public-views /path/to/public-views.jsonl \
  --p74-inputs /path/to/p74-inputs.jsonl \
  --p74-evaluation /path/to/p74-evaluation.jsonl \
  --output-dir /private/tmp/panorai-val018-unified-replay

# Copy this runner and run_optimized_public_pair.py outside the checkout first.
/path/to/wheel-venv/bin/python run_unified_optimized_pair.py \
  --inputs /path/to/inputs.jsonl \
  --evaluation /path/to/evaluation.jsonl \
  --pair-id PAIR_ID --height 1024 \
  --expected-source-commit ORIGIN_MAIN_COMMIT \
  --forbidden-checkout /path/to/source-checkout \
  --output /path/to/result.json
```

## Resumable aligned-frontend population replay

The full 2,385-pair replay uses the current release wheel, PanorAi `3.5.0`,
whose source tree is identical to the fetched `origin/main` commit `03c5b36`.
The runner rejects source-checkout imports, NumPy convolution fallback,
sequential detection, or any result without the complete translation-
orientation diagnostics required by the post-processing model.

Prepare the complete population in a deterministic order that alternates
datasets while each dataset is independently shuffled by a frozen hash:

```bash
python benchmarks/two_view_pose_probability/prepare_unified_population_replay.py \
  --split-manifest /path/to/standardized-pairs.jsonl \
  --public-predictions /path/to/public-predictions.jsonl \
  --public-evaluation /path/to/public-pairs-evaluation.jsonl \
  --public-views /path/to/public-views-evaluation.jsonl \
  --p74-inputs /path/to/p74-pairs-method-inputs.jsonl \
  --p74-evaluation /path/to/p74-pairs-evaluation.jsonl \
  --output-dir /private/tmp/panorai-val018-population-replay
```

Run or resume with the same command. Each pair is committed by atomic rename;
valid existing results are checked and skipped. A file named `STOP` asks the
runner to finish the current pair and pause, and `--max-pairs` provides a
bounded pilot or checkpoint run.

```bash
python benchmarks/two_view_pose_probability/run_resumable_population_replay.py \
  --inputs /private/tmp/panorai-val018-population-replay/inputs.jsonl \
  --evaluation /private/tmp/panorai-val018-population-replay/evaluation.jsonl \
  --runner /private/tmp/panorai-val018-main-runner/run_unified_optimized_pair.py \
  --python /private/tmp/panorai-val018-main-venv/bin/python \
  --output-dir /private/tmp/panorai-val018-population-replay/run \
  --expected-package-version 3.5.0 \
  --expected-source-commit 03c5b36b28225b24d3909286bf53250d7b532aa3 \
  --forbidden-checkout /path/to/PanorAi-source-checkout \
  --height 1024
```

Only after all 2,385 pairs are present can the aligned analysis table be built:

```bash
python benchmarks/two_view_pose_probability/build_aligned_population_table.py \
  --base-analysis-table /path/to/frozen-analysis-table.jsonl \
  --results-dir /private/tmp/panorai-val018-population-replay/run/results \
  --output-dir /private/tmp/panorai-val018-aligned-table \
  --expected-package-version 3.5.0 \
  --expected-source-commit 03c5b36b28225b24d3909286bf53250d7b532aa3
```

The table builder fails closed if even one pair is missing, duplicated, has an
older result schema, reports a package version or source commit other than the
frozen PanorAi 3.5.0 artifact, or contradicts the independently recomputed
pose-error thresholds.

The complete sealed post-replay sequence can then be run in a new, empty
output directory. It fits both aligned model families before opening the
evaluation outcomes, evaluates transfer, freezes and evaluates the aligned
selective rule, preserves a no-qualifying-rule result when appropriate, plans
E8, and regenerates the quantitative paper figures:

```bash
python benchmarks/two_view_pose_probability/run_aligned_analysis.py \
  --base-analysis-table /path/to/frozen-analysis-table.jsonl \
  --census /path/to/frozen-census.json \
  --analysis-code-lock benchmarks/two_view_pose_probability/ALIGNED_ANALYSIS_CODE_LOCK.json \
  --results-dir /private/tmp/panorai-val018-population-replay/run/results \
  --output-dir /private/tmp/panorai-val018-aligned-analysis \
  --expected-package-version 3.5.0 \
  --expected-source-commit 03c5b36b28225b24d3909286bf53250d7b532aa3
```

After the runner reaches `complete`, independently reopen the raw results and
all derived artifacts:

```bash
python benchmarks/two_view_pose_probability/verify_aligned_analysis.py \
  --analysis-dir /private/tmp/panorai-val018-aligned-analysis \
  --census /path/to/frozen-census.json \
  --expected-package-version 3.5.0 \
  --expected-source-commit 03c5b36b28225b24d3909286bf53250d7b532aa3 \
  --output /private/tmp/panorai-val018-aligned-analysis/verification.json
```

The verifier recomputes artifact and raw-result hashes, route identity,
population denominators, split partitions, model profiles, LODO exclusions,
selective-rule identity, and paper-figure/document presence. A failed check is
reported as `FAIL` and exits nonzero.

The post-replay model specification was frozen before any aligned population
aggregate was computed. It is recorded in
[`ALIGNED_FRONTEND_ANALYSIS_PLAN.md`](ALIGNED_FRONTEND_ANALYSIS_PLAN.md). Use
`--model-profile aligned` for both fitting commands to add the declared
translation-orientation model while preserving the historical ablations.
Freeze the aligned replacement rule with
`--post-model post-precise-aligned-orientation`; rule evaluation reads and uses
that exact serialized model identity.

See [`PAPER_RESULTS.md`](PAPER_RESULTS.md) for the paper-ready interpretation
and [`PROSPECTIVE_CONFIRMATION_PROTOCOL.md`](PROSPECTIVE_CONFIRMATION_PROTOCOL.md)
for the frozen acquisition and unsealing order. Representative failure modes
and their scientific interpretation are in
[`FAILURE_TAXONOMY.md`](FAILURE_TAXONOMY.md).

The present results are post-hoc evidence. They do not by themselves establish
a release threshold; that requires a rule frozen on development/calibration
groups and confirmation on new independent capture groups.
