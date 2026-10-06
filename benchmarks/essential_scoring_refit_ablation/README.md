# Essential scoring and all-inlier refit ablation

This source-checkout experiment compares three hypothesis rankings under
identical deterministic five-point proposals:

- `count-first`: the current maximum-consensus ordering;
- `msac-first`: normalized truncated-quadratic MSAC cost first;
- `scale-marginal-first`: the existing multi-scale robust score first.

Each ranking is evaluated with and without an unweighted non-minimal Essential
refit over the current inlier set. The refit iterates until the inlier mask is
stable, a mask cycle is detected, the fit becomes rank deficient, or the
explicit 100-step cap is reached. The best hypothesis observed along that path
is retained according to the ranking under test.

All variants use the same scene seed, RANSAC seed, sampler, trial count,
residual threshold, cheirality policy and nonlinear-refinement budget. The
default public estimator remains `count-first` with refit disabled.

Development seeds in the `810000` series were observed before freezing the
primary seed base `1100000`.

```console
python benchmarks/essential_scoring_refit_ablation/run_ablation.py \
  --output benchmarks/essential_scoring_refit_ablation/results \
  --cases 20 \
  --seed-base 1100000
```

This is targeted synthetic source-checkout evidence, not an installed-wheel or
real-feature accuracy claim.
