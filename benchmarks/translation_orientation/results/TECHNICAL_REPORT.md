# Parallax-weighted Essential translation orientation

## Abstract

This report compares the historical unweighted positive-depth count with
a bounded parallax-weighted cheirality score on identical seeded spherical
correspondences. Results are source-checkout development evidence, not a
claim about the published package or the frozen VAL-002 real-image corpus.

The candidate improved the isolated weak-parallax stress condition by
3.7 percentage points with 37 paired gains and 0 losses. Across the end-to-end conditions there were 0 paired sign changes: this run does not demonstrate an end-to-end accuracy gain.
The result therefore supports the decomposition change while identifying
Essential estimation/refinement as the next measured bottleneck.

## 1. Problem and research question

The Essential constraint identifies a translation axis but its SVD has four
pose decompositions. The historical rule selected the decomposition with the
largest raw count of positive-depth triangulations. The preregistered question
for this benchmark is whether bounded parallax weighting improves the
orientation of translation without degrading well-conditioned scenes.

## 2. Candidate method

For triangulation angle $\theta_i$, the candidate policy uses
$w_i=\sin^2\theta_i/(\sin^2\theta_i+\sin^2\theta_0)$ with
$\theta_0=1$ degree, then maximizes the weighted positive-depth support
over the four Essential decompositions. The baseline maximizes the raw
positive-depth count. Ground truth is the SE(3) transform used to generate
the 3D points; tangent-plane noise and outliers are added afterward.
If fewer than five rays reach weight 0.5, the candidate retains the
historical axis representative but forces the orientation margin to zero;
this makes the orientation decision abstain rather than invent confidence.

The orientation-only acceptance column applies only the common 0.05
orientation-margin threshold. End-to-end acceptance applies the complete
`RelativePoseAcceptancePolicy`, including residual, coverage, parallax,
stability configuration and competing-model evidence.

## 3. Experimental design

Each row is paired by condition and seed: baseline and candidate receive
byte-identical bearing arrays. The orientation-only experiment supplies the
known Essential matrix and isolates decomposition. The end-to-end experiment
runs five-point generation, robust consensus and nonlinear refinement. The
four conditions cover ordinary geometry, a sub-metre mixed-depth scene, an
extreme weak-parallax stress case, and a deliberately unobservable negative
control. No evaluation seed is selected after observing its outcome.

Primary metrics are sign accuracy (`angle(t_hat,t_gt) < 90 degrees`), strict
pose accuracy (`R < 5 degrees` and `t < 10 degrees` end to end), accepted-sign
precision, and acceptance coverage. Runtime uses one warm-up per method and
experiment, `perf_counter_ns`, median and nearest-rank P95. Peak RSS is the
whole-process upper bound rather than an isolated per-method measurement.

## 4. Results

| Experiment / condition | Method | N | Sign accuracy | Strict accuracy | Accepted sign precision | Coverage | t median / P95 (deg) | Runtime median / P95 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| end-to-end / sub-meter-mixed-depth | parallax-weighted | 20 | 100.0% | 100.0% | — | 0.0% | 0.248 / 1.245 | 545.875 / 1241.221 |
| end-to-end / sub-meter-mixed-depth | positive-depth-count | 20 | 100.0% | 100.0% | — | 0.0% | 0.248 / 1.245 | 543.375 / 877.378 |
| end-to-end / unobservable | parallax-weighted | 20 | 85.0% | 0.0% | — | 0.0% | 58.526 / 159.759 | 610.780 / 1119.701 |
| end-to-end / unobservable | positive-depth-count | 20 | 85.0% | 0.0% | — | 0.0% | 58.526 / 159.759 | 595.316 / 1299.178 |
| end-to-end / weak-parallax-stress | parallax-weighted | 20 | 90.0% | 5.0% | — | 0.0% | 45.815 / 104.598 | 564.836 / 972.658 |
| end-to-end / weak-parallax-stress | positive-depth-count | 20 | 90.0% | 5.0% | — | 0.0% | 45.815 / 104.598 | 555.930 / 742.237 |
| end-to-end / well-conditioned | parallax-weighted | 20 | 100.0% | 100.0% | 100.0% | 100.0% | 0.039 / 0.082 | 484.132 / 978.248 |
| end-to-end / well-conditioned | positive-depth-count | 20 | 100.0% | 100.0% | 100.0% | 100.0% | 0.039 / 0.082 | 487.845 / 603.282 |
| orientation-only / sub-meter-mixed-depth | parallax-weighted | 1000 | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 / 0.000 | 0.131 / 0.150 |
| orientation-only / sub-meter-mixed-depth | positive-depth-count | 1000 | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 / 0.000 | 0.131 / 0.151 |
| orientation-only / unobservable | parallax-weighted | 1000 | 92.8% | 92.8% | — | 0.0% | 0.000 / 180.000 | 0.161 / 0.207 |
| orientation-only / unobservable | positive-depth-count | 1000 | 92.8% | 92.8% | 98.2% | 67.9% | 0.000 / 180.000 | 0.162 / 0.215 |
| orientation-only / weak-parallax-stress | parallax-weighted | 1000 | 99.7% | 99.7% | 100.0% | 95.1% | 0.000 / 0.000 | 0.157 / 0.183 |
| orientation-only / weak-parallax-stress | positive-depth-count | 1000 | 96.0% | 96.0% | 99.6% | 75.2% | 0.000 / 0.000 | 0.159 / 0.185 |
| orientation-only / well-conditioned | parallax-weighted | 1000 | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 / 0.000 | 0.129 / 0.180 |
| orientation-only / well-conditioned | positive-depth-count | 1000 | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 / 0.000 | 0.130 / 0.188 |

### Paired sign changes

| Experiment / condition | Pairs | Candidate gains | Candidate losses | Net change | Exact McNemar p | Acceptance gains / losses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| end-to-end/sub-meter-mixed-depth | 20 | 0 | 0 | +0.0 pp | 1 | 0 / 0 |
| end-to-end/unobservable | 20 | 0 | 0 | +0.0 pp | 1 | 0 / 0 |
| end-to-end/weak-parallax-stress | 20 | 0 | 0 | +0.0 pp | 1 | 0 / 0 |
| end-to-end/well-conditioned | 20 | 0 | 0 | +0.0 pp | 1 | 0 / 0 |
| orientation-only/sub-meter-mixed-depth | 1000 | 0 | 0 | +0.0 pp | 1 | 0 / 0 |
| orientation-only/unobservable | 1000 | 0 | 0 | +0.0 pp | 1 | 0 / 679 |
| orientation-only/weak-parallax-stress | 1000 | 37 | 0 | +3.7 pp | 1.45519e-11 | 234 / 35 |
| orientation-only/well-conditioned | 1000 | 0 | 0 | +0.0 pp | 1 | 0 / 0 |

## 5. Interpretation

The candidate is neutral on well-conditioned and ordinary sub-metre
decomposition cases, while the stress corpus measures the intended gain:
far, noise-dominated rays no longer outvote the few useful-parallax rays.
The end-to-end rows are unchanged because their dominant error occurs
before four-way decomposition: the estimated Essential matrix is already
too inaccurate in the weak-parallax regimes. Both methods correctly
receive zero full-policy coverage there, so no inaccurate pose is promoted.
In the orientation-only unobservable control, the candidate's five-ray
minimum converts noise-driven raw-count decisions into explicit
abstentions; its retained axis representative is not an accepted sign.

This is a bounded positive result, not evidence that the entire E estimator
has improved on real panoramas. The next experiment must address robust E
hypothesis/refinement quality and then replay a frozen real-image corpus.

## 6. Reproducibility

- Schema: `panorai-translation-orientation-benchmark/v1`
- Source commit: `4066c76d7f10e76b481a97bde595b992cb524125`
- Dirty source: `True`
- Python: `3.12.4`
- NumPy: `1.26.4`
- SciPy: `1.14.0`
- Platform: `macOS-26.6.2-arm64-arm-64bit`
- Whole-run peak RSS: `95617024` bytes
- Raw CSV SHA-256: `5eb475568bb34589a810ba00c7e0aeeb69a0edf4ecd992ca7e93f306e463f462`
- Orientation cases per condition: `1000`
- End-to-end cases per condition: `20`

## 7. Threats to validity and limitations

This benchmark isolates a known failure mechanism and includes an
end-to-end synthetic check, but it does not measure feature-matching
domain shift, dynamic scenes, rolling shutter, non-central panoramas, or
real reconstruction prevalence. Any promotion or default confidence
claim requires a new outcome-blind real-pair replay.

## 8. Conclusion

Parallax weighting fixes the targeted decomposition failure without a
measurable runtime penalty or a regression in the two ordinary synthetic
conditions. It does not improve the present end-to-end weak-parallax E
estimate. The implementation should therefore remain Experimental and its
real-data gain should not be claimed until the frozen replay is completed.
