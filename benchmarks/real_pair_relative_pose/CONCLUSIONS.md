# Conclusions from the real calibrated-pair benchmark

## Decision

Use MSAC-first hypothesis ranking followed by bounded all-inlier Essential
refit as the recommended **opt-in real-pair refinement**:

```python
RelativePoseOptions(
    hypothesis_ranking="msac-first",
    nonminimal_refit_max_steps=100,
)
```

Keep count-first/no-refit as the compatibility default. The real evidence is
one calibrated fisheye lens from one sequence, one held-out criterion was not
met, and the candidate returned no pose for one 33-match pair. The guarded
GEO-015 decoupled method remains a targeted synthetic weak-parallax option;
its high precision but low real-pair coverage does not justify selecting it as
the general real-data path.

## Metadata separation and coordinate oracle

The estimator processes only the ROS bag and Kalibr calibration. Predictions
are atomically frozen read-only before a separate evaluator opens the
LiDAR-derived trajectory. The two frozen prediction hashes are
`852e19e63cb07938f1be6890efa810b1e5449ce7d31c74d40f7293f71e6484d2`
and
`4a6e44c027bd54192eeef4703cb220c5600c17a3b6724be617a6c80bd2d9d582`.

Hilti poses describe `cam0 -> map` in the OpenCV/Kalibr optical basis. PanorAi
fisheye bearings use `+Y` up, so the evaluator explicitly applies
`P=diag(1,-1,1)`:

```text
R_2_from_1_PanorAi = P (R_world_from_2.T R_world_from_1) P
t_2_from_1_PanorAi = normalize(P R_world_from_2.T (C_1 - C_2))
```

An analytic point-transport test checks this composition independently. The
uncorrected basis interpretation produced apparent 5--25 degree rotation
errors in the smoke; the declared transformation reduced those same two
errors to 0.17 and 0.31 degrees.

## Development evidence

The descriptive phase contained 18 adjacent pairs from six disjoint windows
and compared five variants. MSAC+refit improved median rotation from 0.658 to
0.351 degrees and median oriented translation from 6.852 to 5.110 degrees.
Translation P95 fell from 82.861 to 20.554 degrees. Strict success rose from
13/18 to 14/18 and high precision from 6/18 to 9/18. Count-first refit alone
did not help, confirming that ranking and refit act together.

## Frozen held-out evidence

`HELDOUT_PROTOCOL.md` was hashed before prediction and recorded inside the
frozen file. It selected six new windows, 12 pairs and one candidate. Four of
five preregistered criteria passed:

| Metric | Count-first | MSAC + refit | Relative change | Criterion |
| --- | ---: | ---: | ---: | --- |
| R median | 1.014 deg | 0.901 deg | -11.1% | **fail**: required -25% |
| t median | 12.933 deg | 5.933 deg | -54.1% | pass: required at least -15% |
| t P95 | 65.903 deg | 58.451 deg | -11.3% | pass: required lower |
| Strict | 4/12 | 6/12 | +2 | pass: required non-decreasing |
| High precision | 2/12 | 3/12 | +1 | pass: required non-decreasing |

The candidate improved R on 7 and lost on 4 common-return pairs; it improved t
on 8 and lost on 3. It abstained on the only 33-match pair, whereas count-first
returned a rejected 10.17-degree-R/11.43-degree-t pose. The held-out result
therefore validates a strong translation improvement and a smaller rotation
improvement, not the preregistered 25% rotation target.

## Combined descriptive evidence

Combining both frozen phases only as a descriptive 30-pair summary gives:

| Metric | Count-first | MSAC + refit | Relative change |
| --- | ---: | ---: | ---: |
| Returned | 30/30 | 29/30 | -1 pair |
| R median | 0.785 deg | 0.398 deg | -49.2% |
| R P95 | 8.467 deg | 1.773 deg | -79.1% |
| t median | 8.396 deg | 5.933 deg | -29.3% |
| t P95 | 65.903 deg | 29.241 deg | -55.6% |
| Strict | 17 | 20 | +3 |
| High precision | 8 | 12 | +4 |

Across the 29 common-return pairs, MSAC+refit improved R in 20 versus 9 losses
and improved t in 22 versus 7 losses. This combined table is not a second
held-out claim because the first 18 pairs selected the candidate.

## Limits and operational guidance

- Translation direction remains poorly observable for baselines near zero.
  The benchmark includes five pairs below 0.1 m; rejection is preferred over
  presenting their direction as precise.
- Fewer than roughly 100 descriptor matches is an observed risk region, not a
  universal threshold. The candidate explicitly failed to return for the
  33-match pair instead of fabricating confidence.
- The public quality decision must still be checked. The option changes the
  estimator, not the meaning of `quality_report.accepted`.
- The sequence uses a fitted pinhole-equidistant model, rolling shutter and
  temporally correlated pairs. Cross-sequence, cross-camera and ERP evidence
  remains future work.
- A symmetric angular reprojection prototype was tested on already opened
  development pairs and rejected: it increased error in the first smoke pair
  and added substantial cost. It was not integrated into public source.

Original Hilti dataset bytes remain ignored under `.datasets/`. Tracked files
contain only derived numeric measurements, hashes and prose.
