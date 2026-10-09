# Prospective E8 acquisition plan

Status: **PLANNING_ONLY_NO_COLLECTION_AUTHORIZED**.

## Why 40 groups are not an acquisition plan

The confirmatory gate requires 120 primary selected pairs, at least 40
independent groups, and at most three selected pairs per group. With exactly 40
groups, all 40 groups must contribute exactly three selected pairs. The
retrospective selection rate of the stricter candidate is 34/435, or 7.82%,
with an exact one-sided 95% lower bound of 5.80%. Planning only from the mean
`120 / 0.0782 = 1,536` therefore understates the burden created by the
per-group cap and domain balance.

`plan_prospective_acquisition.py` uses the exact distribution of
`min(Binomial(candidate pairs, selection probability), 3)` in each independent
group. This is an accrual model, not a model of pose accuracy. It never reads a
future reference pose.

## Recommended balanced design

Use three prospective capture strata corresponding to the deployment regimes
represented by the retrospective datasets:

- Matterport-like structured indoor capture;
- Stanford-like large or repetitive indoor capture;
- P74-like industrial capture.

Domain identity is evaluation metadata and must never enter either probability
model. The recommended plan is conservative with respect to selection rate:

| Quantity | Plan |
| --- | ---: |
| Domains | 3 |
| New independent groups per domain | 20 |
| Total new groups | 60 |
| Target selected pairs per domain | 40 |
| Total primary selected target | 120 |
| Maximum primary selected per group | 3 |
| Candidate pairs per group | 52 |
| Total candidate pairs processed | 3,120 |
| Proposed unique panoramas per group | 16 |
| Total proposed unique panoramas | 960 |
| Unordered pairs possible from 16 panoramas | 120/group |
| Conservative selection probability | 5.80% |
| Probability each domain reaches 40 selected | 96.64% |
| Probability all three domains reach 40 selected | 90.25% |

Every R,t estimate still uses exactly two panoramas. Panoramas may participate
in more than one candidate pair within the same group, but no panorama may
cross groups. Group-level uncertainty and the three-pair primary cap prevent
that reuse from inflating the confirmatory sample size.

Sixteen panoramas are an operational proposal rather than a mathematical
minimum. Eleven images can form 52 distinct unordered pairs, but 16 provide 120
possible pairs from which 52 capture-eligible pairs can be registered before
prediction. Pair registration must use capture variables and the frozen
capture envelope, never pose errors or post-processing outcomes.

## Group-count trade-off

The following plans all use the conservative 5.80% selection rate and reach at
least 90% joint probability of 40 selected pairs in every domain:

| Groups/domain | Total groups | Candidates/group | Total candidates | Mathematical minimum images/group |
| ---: | ---: | ---: | ---: | ---: |
| 14 | 42 | 120 | 5,040 | 16 |
| 20 | 60 | 52 | 3,120 | 11 |
| 24 | 72 | 41 | 2,952 | 10 |
| 25 | 75 | 39 | 2,925 | 10 |
| 30 | 90 | 32 | 2,880 | 9 |
| 40 | 120 | 23 | 2,760 | 8 |

The 60-group design balances acquisition-site cost against pair-processing
cost. If acquiring new independent sites is cheap, 90–120 groups reduce the
number of pairs required per site. If sites are expensive, the 42-group plan is
possible but requires nearly exhaustive eligible pairing and is more sensitive
to within-group dependence.

## Stopping and registration rules

1. Register all candidate image IDs, group IDs, validity-mask identities,
   capture variables, and pair eligibility before post-processing outcomes or
   reference poses are visible.
2. Continue collection according to the registered accrual plan, not observed
   R/t accuracy. Selection counts may be used to manage accrual because they are
   outcome-blind; reference errors may not.
3. Retain every failed, returned, accepted, withheld, and selected candidate in
   coverage denominators.
4. Choose at most three primary selected pairs per group by a frozen
   outcome-blind ordering. Extra selected pairs remain diagnostic.
5. Seal all 120 primary decisions before opening reference geometry.

## Reproducibility

The permanent acquisition plan has SHA-256
`4fb6b71e46dfe6d51aca8bda09e23b9f72c937d9e47c65406894512f23d56133`;
the complete scenario CSV has SHA-256
`dc3fcb06d8d29950c2eb3c33f2573540fa0a4a37605861ea4766d12dfb568510`.
A second run in a different output directory was byte-identical.

```bash
python benchmarks/two_view_pose_probability/plan_prospective_acquisition.py \
  --readiness-report /path/to/readiness-report.json \
  --selected-target 120 \
  --groups-per-domain 20 \
  --cap-per-group 3 \
  --accrual-probability 0.90 \
  --proposed-images-per-group 16 \
  --output-dir /path/to/new-acquisition-plan
```
