# Query-conditioned sampling-prior pilot

Status: **mixed development result; not real-CAM or release evidence**

This synthetic causal study held the 120 correspondences, validity mask,
relative-pose options, 48-trial budget, and random seed fixed. Each scene had
24 true matches and 96 outliers. Only the five-point proposal weights changed.
Success means rotation error at most 5 degrees and translation-direction error
at most 10 degrees. Twelve deterministic seeds were evaluated per scenario.

## Results

| Semantic support | Uniform-weight baseline | Guided, ratio 4 | Guided, ratio 10 |
| --- | ---: | ---: | ---: |
| correct and angularly distributed | 5/12 | 7/12 | 11/12 |
| correct but compact | 3/12 | 2/12 | 5/12 |
| half correct, half false | 5/12 | 6/12 | 5/12 |
| entirely on outliers | 5/12 | 2/12 | 1/12 |
| absent | 5/12 | 5/12 | 5/12 |

The absent-support control reproduced every baseline record exactly. No match
was removed from scoring or refinement. With an aggressive prior, correct
distributed support added six successes without losing a baseline success.
The same setting lost four net successes when the semantic evidence selected
outliers. The conservative prior reduced both effects but did not eliminate
the risk.

Correct compact support is not equivalent to a good pose prior. Its 15
high-weight matches lie on one small object and conflict with the estimator's
angular-separation and spherical-cell diversity requirements. The conservative
prior was slightly worse than baseline, while the stronger prior was only
modestly better. None of the compact-scene estimates passed the estimator's
full quality acceptance policy in either guided configuration, even when the
pose met the error thresholds.

These results establish an operating principle, not a tuned default:

> semantic evidence may prioritize proposals, but geometric diversity and an
> independently validated pose must retain final authority.

The next validation must replace controlled semantic membership with actual
ImageNet CAM values on frozen real panorama pairs. Until then, the benchmark
does not measure CAM precision, recall, instance separation, or cross-domain
behavior.

## Reproduction

Conservative prior:

```bash
python benchmarks/object_localization/run_semantic_prior_limits.py \
  --seeds 12 --inliers 24 --outliers 96 --max-trials 48 \
  --uniform-mix 0.25 --max-weight-ratio 4 \
  --output-dir /private/tmp/panorai-semantic-prior-limits-hard-conservative-12
```

Aggressive diagnostic prior:

```bash
python benchmarks/object_localization/run_semantic_prior_limits.py \
  --seeds 12 --inliers 24 --outliers 96 --max-trials 48 \
  --uniform-mix 0.1 --max-weight-ratio 10 \
  --output-dir /private/tmp/panorai-semantic-prior-limits-hard-aggressive-12
```

Result checksums from the recorded runs:

- conservative `results.json`:
  `47278300d723d16bb701de86541b8a3763bf41b10cc4b378ec0de2064be9f7c7`
- conservative `runs.csv`:
  `ac3bac2b8da56fc0c0a91d46ca8fa50c7ad426d79175ee70440abece61efb212`
- aggressive `results.json`:
  `1434551d4f783aaafc2e3b6c0c9bb98ae6f44e6ba66106929aa1768c2df29405`
- aggressive `runs.csv`:
  `0fca8694a5158143f0940e020f36f3da059b2992867583d80eed63764bc32f02`
