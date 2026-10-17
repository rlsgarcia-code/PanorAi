# Run panoramic dense stereo from two ERPs

**Level:** Start/compose · **Input:** two canonical ERPs, with optional relative
pose and baseline · **Output:** radial range, validity, confidence,
cross-view disagreement, pose and diagnostics

`PanoramicDenseStereo` is the A-to-Z application. It does not treat a
rectified pinhole matcher as if it understood an ERP. It resolves pose and
scale, creates paired rectified views from spherical geometry, runs one dense
matcher at a time, converts disparity to radial range, and fuses range in the
reference ERP.

```text
ERP A + ERP B
      |
      +-- use supplied R,t ---------+
      |                             |
      +-- spherical features -> relative pose
                                    |
                 baseline / explicit scale-free status
                                    |
              paired rectified gnomonic views
                                    |
          classical dense stereo, one pair at a time
                                    |
                 disparity -> radial range
                                    |
            backproject -> confidence-weighted fusion
                                    |
       ERP range + validity + confidence + disagreement
```

## Shortest call with known pose

```python
from panorai.applications import PanoramicDenseStereo

stereo = PanoramicDenseStereo(pose_mode="provided")
result = stereo(
    erp_a,
    erp_b,
    rotation_source_from_reference=R_b_from_a,
    translation_source_from_reference=t_b_from_a_m,
    units="m",
)

range_m = result.radial_range
valid = result.validity
confidence = result.confidence
matching_confidence = result.matching_confidence
consistency_confidence = result.consistency_confidence
disagreement_m = result.disagreement
```

The pose convention is
`X_source = R_source_from_reference @ X_reference + t_source_from_reference`.
The norm of supplied `t` is used as the baseline unless `baseline=` explicitly
rescales its direction.

## Let PanorAi estimate relative pose

```python
stereo = PanoramicDenseStereo(pose_mode="auto")
result = stereo(erp_a, erp_b, baseline=baseline_m, units="m")
```

This route reuses `SphericalFeaturePipeline.for_relative_pose()` and
`SphericalRelativePoseEstimator`. The estimator's existing quality policy is
an enforced gate by default; rejection reasons are not converted into a dense
result. Two central images determine translation
direction, not metric scale. If `baseline` is omitted, the result is labelled
`scale-free` and its range unit is `baseline-units`; it must not be described
as metres.

## Exchange the pinhole matcher

```python
from panorai.applications import ADCensusStereoDepth, PanoramicDenseStereo

stereo = PanoramicDenseStereo(
    matcher=ADCensusStereoDepth(),
    route="rectified-views",
)
```

The six compatible matchers are documented in
[non-learned rectified stereo](classical_stereo.md). They operate only after
spherical planning and projection have produced rectified pinhole pairs.

## Choose the execution route

```python
direct = PanoramicDenseStereo(route="sphere-native")
result = direct(erp_a, erp_b, pose=known_pose)
```

| Route | Geometry | Dense method | Return contract |
|---|---|---|---|
| `rectified-views` | paired rolled `GnomonicSpec` views | interchangeable classical matcher | `PanoramicStereoResult` |
| `sphere-native` | direct ERP epipolar curves | inverse-range plane sweep | `PanoramicStereoResult` |

The routes are alternatives, not silently mixed estimates.
The current sphere-native estimator has no source-validity argument, so this
facade rejects partial source support on that route instead of treating invalid
source pixels as evidence. The rectified-view route propagates validity for
both panoramas through every projected pair.

## Why the views already match

After pose resolution, the reference virtual camera's local x-axis is aligned
with the physical A-to-B baseline. Its source counterpart is the same axis
rotated into camera B. Corresponding epipolar great circles therefore have the
same raster row in both faces. Pair identity is geometric and deterministic;
appearance does not decide which planned reference face belongs to which
planned source face.

`SphericalStereoViewPairer` is the separate many-to-many utility for views that
were sampled independently. Its score exposes angular overlap, sparse-inlier
coverage, epipolar consistency and geometric support instead of hiding them in
one opaque association.

Outside the application, arbitrary rolled views use the same object workflow:

```python
views = panorama.project_views(specs)

for view in panorama.iter_views(specs):
    consume_then_release(view)
```

`project_views()` returns a `GnomonicFaceSet`; `iter_views()` projects only the
next spec and is the appropriate route when view tensors must not coexist in
memory.

## Streaming and fusion

The projected route materializes one RGB pair and one cost volume at a time.
Before allocating that volume, `SelectStereoViewPairs` records and gates joint
support, texture, pose-inlier coverage, rectified-row residual and disparity
interval viability. Accepted and rejected decisions remain in
`result.view_selection`.
The cost volume is discarded after disparity selection. Pair-local disparity
is converted to radial range before backprojection; disparities from different
pinhole lattices are never averaged. ERP fusion weights accepted range by
matcher confidence and a face-edge taper, and reports weighted disagreement
and contribution count.

The taper is only a geometric range-fusion weight. It is divided back out of
the reported `matching_confidence`, so a correct observation does not appear
less certain merely because it lies near a projected face boundary.
`consistency_confidence` is
`exp(-disagreement / radial_range)`, and `confidence` is exactly the product
of those two explicit components. None is a calibrated probability.

Finite pinhole views cannot cover the two epipolar singularities on the
baseline axis. `result.view_pairs` and stage diagnostics expose the chosen FOV,
pair count and omitted cone rather than claiming complete spherical coverage.

## Cover the full sphere with multiple baselines

A cubemap fixes **sampling coverage**, but it cannot make the two epipoles of
one camera pair observable. `FullSphereDenseStereo` therefore combines six
overlapping cube-oriented tiles with two or more non-collinear registered
source panoramas:

```python
from panorai.applications import FullSphereDenseStereo
from panorai.stereo.panoramic import (
    PanoramicStereoPose,
    RegisteredPanoramaSet,
    StereoSourceView,
)

registered = RegisteredPanoramaSet(
    reference_id="A",
    reference=erp_a,
    sources=(
        StereoSourceView(source_id="B", panorama=erp_b, pose=pose_b_from_a),
        StereoSourceView(source_id="C", panorama=erp_c, pose=pose_c_from_a),
    ),
)

result = FullSphereDenseStereo()(registered)
range_m = result.radial_range
```

Registration belongs to the set: each source transform is interpreted only as
``T_source_from_A`` for this explicit reference. A source panorama is never
described as independently registered.

For every face pixel and radial hypothesis, PanorAi forms the 3-D point in the
reference frame, transforms it into each source camera, samples that source
ERP, and aggregates only sources with sufficient parallax. Faces overlap by
using a FOV greater than 90 degrees and are fused back into the reference ERP.
Only one face and one hypothesis-sized source stack are retained at a time.
The default per-hypothesis source aggregation is the median, which rejects one
occluded or appearance-inconsistent source when at least three sources are
available. `weighted-mean` and `best` remain explicit diagnostic alternatives.

### Choose the cost pipeline explicitly

Full-sphere matching exposes three separate stages instead of hiding them
behind one method name:

```python
from panorai.applications import FullSphereDenseStereo
from panorai.stereo.panoramic import FullSphereStereoOptions

stereo = FullSphereDenseStereo(
    FullSphereStereoOptions(
        matching_cost="ad-census",  # raw unary evidence
        local_aggregation="box",    # one validity-normalized window
        regularizer="sgm4",         # four-path spatial regularization
    )
)
result = stereo(registered)
```

`matching_cost` is `intensity-gradient`, `ad-census`, or
`multiscale-ad-census`;
`local_aggregation` is `none` or `box`; and `regularizer` is `none`, `sgm4`,
or `tmgm`. The stages are composed exactly once in that order. The selected
configuration and its parameters are preserved in `result.diagnostics` and in
the application's saved JSON configuration.

The reviewed default is `ad-census` + `box` + `sgm4`, with angular ERP and face
taps. Explicit `matching_cost="intensity-gradient"`,
`erp_feature_tap_deg=None`, and `face_cost_tap_deg=None` retain the historical
profile. This selects the strongest frozen cross-domain candidate; it does not
make the method Stable. The application remains **Experimental**.

Use the two-scale cost without changing the downstream aggregation:

```python
multiscale = FullSphereDenseStereo(
    FullSphereStereoOptions(matching_cost="multiscale-ad-census")
)
```

This averages AD-Census evidence at the configured angular tap spacing and at
twice that spacing. A frozen twelve-pair, view-disjoint audit excluded
sky/no-return and non-observable directions before scoring. Relative to the
single-scale route, fixed δ<1.25 changed from 0.345 to 0.349 on Stanford and
from 0.440 to 0.445 on Matterport; conditional gross error fell by 0.008 and
0.009, while coverage fell by 0.003 and 0.005. Median paired runtime was
1.11x. The complete run reproduced bit for bit. The option is useful when the
small accuracy/outlier gain justifies the additional work; it is not a
calibrated confidence or a Stable-method claim.

A subsequent outcome-blind public multipair audit froze five Stanford 2D-3D
pairs and six Matterport3D pairs before dense inference. It used the unchanged
default above, a fixed evaluation population, explicit no-return exclusion,
same-surface visibility, and solid-angle weighting:

| Dataset | Pairs | Micro coverage | Micro δ<1.25 | Median pair median error | Worst pair coverage / δ<1.25 |
|---|---:|---:|---:|---:|---:|
| Stanford 2D-3D | 5 | 0.568 | 0.301 | 0.336 m | 0.332 / 0.160 |
| Matterport3D | 6 | 0.536 | 0.339 | 0.192 m | 0.388 / 0.195 |

Invalid predictions count as failures in δ<1.25; coverage is not removed from
the denominator. A complete replay reproduced every metric and all eleven NPZ
payloads bit for bit. Reproducibility is therefore established for this source
environment, but accuracy and worst-case coverage still reject Stable
promotion.

The same frozen audit tested whether the current confidence margin at least
ranks predictions usefully. Retaining a predetermined fraction of each pair's
valid predictions gave:

| Dataset | Retained predictions | Fixed-population coverage | Conditional δ<1.25 |
|---|---:|---:|---:|
| Stanford 2D-3D | 25% | 0.142 | 0.827 |
| Stanford 2D-3D | 50% | 0.284 | 0.723 |
| Stanford 2D-3D | 100% | 0.568 | 0.529 |
| Matterport3D | 25% | 0.134 | 0.835 |
| Matterport3D | 50% | 0.268 | 0.764 |
| Matterport3D | 100% | 0.536 | 0.632 |

All eleven pairs were monotonic: the highest-confidence subset was more
accurate than larger retained subsets. This supports confidence as a ranking
diagnostic, but not as a calibrated probability or a portable threshold. The
25% subset covers only about 14% of the fixed observable population. A Stable
abstention policy still requires a threshold learned without held-out outcomes
and then tested unchanged on separate pairs.

That absolute-threshold test was then run at a frozen operating target:
conditional δ<1.25 of at least 0.75 and fixed-population coverage of at least
0.10. For each fold, the most permissive absolute score satisfying both
constraints was selected from calibration pairs only and applied unchanged to
the target:

| Calibration route | Joint passes | Main failure |
|---|---:|---|
| same-domain leave-one-pair-out | 4 / 11 | seven target pairs fell below 0.75 accuracy or 0.10 coverage |
| all-other-pairs leave-one-pair-out | 4 / 11 | the pooled threshold did not preserve target accuracy |
| Stanford → Matterport | pass | 0.837 conditional accuracy at 0.226 fixed coverage |
| Matterport → Stanford | fail | 0.655 conditional accuracy at 0.377 fixed coverage |

The two source replays produced identical thresholds and evaluations. The
failure is therefore not nondeterminism: the numeric confidence scale is not
portable between all pairs or both domain directions. Use it to rank evidence
within a result, but do not expose a universal threshold or call it calibrated.
Promotion requires a revised score or an independently frozen calibration
model tested prospectively on unseen scenes.

The taper correction preserved radial range, validity, disagreement, support,
observability and all source-count fields bit for bit on all eleven pairs. It
improved within-pair ranking, but did not improve absolute-threshold transfer.
Geometric blending contamination was therefore a real defect, not the main
calibration failure.

A separate prospective multiview audit then froze three two-source references
per public domain using only metric baseline and registered-cloud overlap.
No dense output entered selection, sky/no-return directions were excluded,
and the single- and two-source routes were scored on the same observable-union
population:

| Dataset / route | Micro coverage | Micro δ<1.25 | Conditional gross error (>2 inverse-range labels) |
|---|---:|---:|---:|
| Stanford / primary source | 0.500 | 0.208 | 0.499 |
| Stanford / two sources | 0.528 | 0.233 | 0.482 |
| Stanford / strict two-source consensus | 0.041 | 0.028 | 0.213 |
| Matterport / primary source | 0.637 | 0.362 | 0.338 |
| Matterport / two sources | 0.655 | 0.439 | 0.253 |
| Matterport / strict two-source consensus | 0.100 | 0.082 | 0.102 |

Two sources improved fixed-population δ<1.25 in five of six triplets and
reduced the conditional gross-outlier rate in both domain aggregates. This is
real evidence for multibaseline aggregation, but the remaining Stanford error
and low absolute accuracy still reject Stable promotion. Requiring both source
views to choose a nearby lattice label produces a cleaner subset but collapses
coverage to 4--10%; it is therefore an audit/abstention mode, not the default
visibility solution. Both complete executions reproduced the deterministic
metrics and all eighteen array payload hashes exactly.

Replacing the two-source lower median with parallax-weighted mean was tested
unchanged on the same frozen triplets as a subsequent ablation. It reduced
observable-union micro δ<1.25 from `0.233` to `0.197` on Stanford and from
`0.439` to `0.388` on Matterport; Stanford coverage also fell from `0.528` to
`0.482`. Requiring every visible source to contribute therefore does not solve
occlusion. The default is unchanged: a future policy must reject an
occluded/incompatible source while still penalizing unsupported minima from a
single ambiguous source.

A second parameter-free ablation multiplied the existing confidence by
`(1 + source_consensus_count / source_count) / 2`. It changed neither range nor
validity. On the same six triplets, however, median top-quartile conditional
δ<1.25 decreased from `0.744` to `0.740` on Stanford and from `0.835` to
`0.825` on Matterport. Absolute-threshold transfer was unchanged: only one of
six all-other-pair folds and one of two cross-domain directions passed the
frozen accuracy-and-coverage gate. Source-label agreement is therefore useful
as a separate diagnostic, but this simple multiplication is rejected as a
confidence improvement and is not part of the runtime score.

Selecting one fixed source group from complete per-source cost curves was also
tested without using evaluation range. The most reliable source anchored the
group, independently preferred labels had to agree within the existing lattice
tolerance, and a second streaming pass formed a parallax-weighted mean. This
prevented source switching between labels and reduced Stanford's conditional
gross-error fraction from `0.482` to `0.460`. It nevertheless reduced Stanford
coverage/δ<1.25 from `0.528/0.233` to `0.496/0.222`, and worsened Matterport
from `0.655/0.439/0.253` to `0.638/0.417/0.267` for
coverage/δ<1.25/gross-error. The policy was therefore rejected and was not
added to the runtime API. Visibility cannot be inferred reliably from unary
curve sharpness alone.

A prospective three-source protocol then selected six new four-view sets from
pose and registered-cloud overlap before inference. The unchanged lower median
became a true median when three costs were available:

All values below exclude sky and no-return directions before scoring. The
fixed denominator is the prediction-independent intersection of positive,
finite dataset range, geometric sampling support and the stated observability
population. A missing or rejected prediction inside that denominator still
counts against coverage and fixed-population accuracy.

| Dataset / sources | Coverage | Fixed-population δ<1.25 | Conditional gross >2-label error |
|---|---:|---:|---:|
| Stanford / one | 0.422 | 0.150 | 0.533 |
| Stanford / two | 0.468 | 0.195 | 0.490 |
| Stanford / three | 0.430 | 0.178 | 0.464 |
| Matterport / one | 0.704 | 0.413 | 0.341 |
| Matterport / two | 0.728 | 0.515 | 0.238 |
| Matterport / three | 0.714 | 0.514 | 0.224 |

The third source suppresses gross outliers in both domains. It does not yet
improve the fixed-denominator result because accepted coverage falls. On the
population visible in all three sources, Matterport δ<1.25 increases from
`0.537` to `0.542`, while Stanford decreases from `0.225` to `0.213`.
Requiring three valid source costs per hypothesis produces almost the same
result as the ordinary three-source route, so missing source support is not the
main explanation.

A frozen gate-decomposition then disabled one existing acceptance condition at
a time without choosing replacement thresholds from outcomes. The table shows
the three-source result minus the two-source result on the same observable-union
population:

| Domain / policy | Δ coverage | Δ fixed δ<1.25 | Δ gross fixed fraction |
|---|---:|---:|---:|
| Stanford / baseline | -0.0385 | -0.0172 | -0.0299 |
| Stanford / no cost gate | -0.0385 | -0.0172 | -0.0299 |
| Stanford / no confidence gate | -0.0274 | -0.0160 | -0.0227 |
| Stanford / no texture gate | -0.0502 | -0.0387 | -0.0253 |
| Matterport / baseline | -0.0144 | -0.0017 | -0.0131 |
| Matterport / no cost gate | -0.0144 | -0.0017 | -0.0131 |
| Matterport / no confidence gate | -0.0054 | +0.0002 | -0.0079 |
| Matterport / no texture gate | -0.0387 | -0.0122 | -0.0269 |

The absolute matching-cost gate is inactive on all 60 evaluated conditions:
setting it from `0.65` to `1.0` changes no output. Confidence explains part of
the source-count coverage loss, especially on Matterport, but not Stanford.
Texture controls much of the absolute coverage; it does not cause the
two-to-three-source regression. The next correction must therefore instrument
rejection reasons after cost fusion—especially confidence and boundary-label
rejection—before defining a source-count-aware score prospectively. Relaxing
the current cost threshold cannot solve the observed problem.

The estimator now exposes the cumulative face-pixel counts that produced this
decision in `result.diagnostics["face_acceptance_cumulative_counts"]`. These
counts follow the declared `face_acceptance_order`; overlapping faces are
counted separately, so they diagnose the algorithm but are not ERP accuracy
metrics. Replaying the unchanged VAL-112 protocol preserved all 24 numerical
array hashes bit for bit and produced:

| Domain / cumulative stage | Two sources | Three sources | Difference |
|---|---:|---:|---:|
| Stanford / interior hypothesis | 266996 | 253836 | -13160 |
| Stanford / after texture | 153346 | 144737 | -8609 |
| Stanford / after confidence = accepted | 135211 | 123309 | -11902 |
| Matterport / interior hypothesis | 284052 | 281350 | -2702 |
| Matterport / after texture | 224525 | 224294 | -231 |
| Matterport / after confidence = accepted | 212276 | 208775 | -3501 |

Stanford's regression is primarily associated with the median choosing an
endpoint of the finite inverse-range lattice; such endpoints cannot support
the current parabolic sub-hypothesis refinement and are conservatively
rejected. Matterport's remaining regression is primarily the normalized
best-versus-second-best confidence gate. Matching cost, minimum source count,
range consensus and the final parallax predicate remove no additional face
samples after confidence in this frozen configuration. This does **not**
justify accepting endpoint labels or dropping confidence: both alternatives
need independent correctness oracles and prospective public validation.

That validation was then run with the exact discrete endpoint label accepted
without parabolic refinement. The candidate reduced the two-to-three-source
coverage gap, but failed every frozen promotion criterion:

| Domain / three-source policy | Coverage | Fixed δ<1.25 | Conditional gross error |
|---|---:|---:|---:|
| Stanford / reject endpoint | 0.4296 | 0.1782 | 0.4639 |
| Stanford / accept discrete endpoint | 0.4810 | 0.1772 | 0.5135 |
| Matterport / reject endpoint | 0.7139 | 0.5136 | 0.2242 |
| Matterport / accept discrete endpoint | 0.7495 | 0.5136 | 0.2566 |

Coverage increased, but fixed-population accuracy decreased slightly in both
domains and conditional gross error increased by `0.0496` on Stanford and
`0.0323` on Matterport, above the frozen `0.02` limit. Two full executions
reproduced the evidence and all 24 payload hashes exactly. The candidate option
was removed: a boundary winner must remain rejected until the estimator can
distinguish a true endpoint from a censored out-of-range or ambiguous match.
The principled next routes are range-interval expansion with a second search,
or an explicit bound/censored output contract—not silently treating the bound
as a point range.

Confidence was then separated prospectively from hard validity on twelve new
public pairs: six Stanford pairs balanced across areas 2, 3 and 4, and six
Matterport pairs from distinct spatial groups. The manifest was frozen before
inference, excluded every pair and view used by the earlier protocols, and
used only pose, registered-cloud overlap and asset availability. Evaluation
again removed sky/no-return with positive finite dataset range before applying
prediction validity.

| Domain / confidence policy | Coverage | Fixed δ<1.25 | Conditional gross error |
|---|---:|---:|---:|
| Stanford / threshold | 0.5045 | 0.3032 | 0.3666 |
| Stanford / report only | 0.5630 | 0.3140 | 0.4105 |
| Matterport / threshold | 0.6412 | 0.4881 | 0.2165 |
| Matterport / report only | 0.6754 | 0.4995 | 0.2380 |

Report-only confidence increased coverage and fixed-population accuracy in
both domains, and confidence selectivity remained monotonic. It nevertheless
increased Stanford conditional gross error by `0.04386`, exceeding the frozen
`0.04` tolerance; Matterport increased by `0.02150`. Two complete executions
produced identical evidence and all 24 result payload hashes. The candidate
failed as specified, was removed, and the threshold default is unchanged.
This isolates the next requirement: improve the score or its independently
validated acceptance rule rather than making confidence advisory everywhere.

A separate cost-default protocol then froze twelve further view-disjoint pairs
before inference and held box aggregation, SGM4, geometry, range lattice and
all acceptance predicates constant. It compared only the unary matching cost:

| Domain / unary cost | Coverage | Fixed δ<1.25 | Conditional gross error |
|---|---:|---:|---:|
| Stanford / intensity-gradient | 0.5888 | 0.2849 | 0.4963 |
| Stanford / AD-Census | 0.6164 | 0.3395 | 0.4379 |
| Matterport / intensity-gradient | 0.6312 | 0.4196 | 0.2795 |
| Matterport / AD-Census | 0.6353 | 0.4228 | 0.2803 |

AD-Census passed every frozen scientific criterion: fixed-population accuracy
improved in both domains, coverage did not regress, and conditional gross error
stayed within the `0.02` tolerance. Median runtime was `0.923×` the existing
cost in the first run and `0.904×` in the replay, below the `1.25×` limit. The
deterministic evidence and all 24 result payloads reproduced exactly. A fresh
review reproduced all payloads, populations, metrics, and aggregates. The raw
review JSON additionally contains the later angular-neighbourhood diagnostic;
removing only that field reconstructs the original evidence byte for byte.
AD-Census is consequently the default for new option objects. Old saved
configurations without the field still deserialize to the historical cost.

A resolution/lattice diagnostic then reused the already observed VAL-098
development pairs. It is not held-out promotion evidence. A controlled
``512x1024`` ERP factorial kept AD-Census, box aggregation, SGM4, poses,
populations and every acceptance gate fixed while varying face size and the
inverse-range lattice:

| Controlled cell | Stanford coverage / fixed δ<1.25 | Matterport coverage / fixed δ<1.25 |
|---|---:|---:|
| face 128 / 48 hypotheses | 0.4361 / 0.2525 | 0.4197 / 0.2775 |
| face 128 / 96 hypotheses | 0.4446 / 0.2837 | 0.4121 / 0.2785 |
| face 256 / 48 hypotheses | 0.4424 / 0.2176 | 0.4210 / 0.2440 |
| face 256 / 96 hypotheses | 0.4543 / 0.2568 | 0.4186 / 0.2559 |

Doubling the range lattice improves fixed accuracy by ``0.0312`` on Stanford
but only ``0.0010`` on Matterport. Doubling face resolution with the unchanged
pixel-window parameters lowers fixed accuracy by about ``0.034`` in both
domains. The result reproduces exactly for the best controlled cell and rules
out a simple "use more pixels" remedy. It also exposes a contract issue: the
same pixel-valued ``window_size`` currently controls an ERP feature
neighbourhood and projected-face cost aggregation, so its angular support
changes when ERP or face resolution changes. Angularly invariant feature,
Census and aggregation support must be evaluated before a higher-resolution
configuration can be a Stable default.

The follow-up implementation separates those roles without changing legacy
execution when the new options are omitted:

```python
from panorai.stereo.panoramic import FullSphereStereoOptions

angular = FullSphereStereoOptions(
    matching_cost="ad-census",
    erp_feature_tap_deg=0.703125,
    face_cost_tap_deg=1.0667900073,
)
```

`erp_feature_tap_deg` is passed to the tangent-plane spherical feature filter.
`face_cost_tap_deg` is converted from actual adjacent centre-ray spacing to a
deterministic integer stride and is shared by Census and valid-normalized
local aggregation. The number of Census bits and aggregation taps stays
constant; only their represented angle is preserved. `None` retains the
legacy one-pixel tap exactly.

On the same eleven development pairs at `512x1024`, face 256 and 96 range
hypotheses, preserving both angular supports improved every aggregate domain:

| Neighbourhood | Stanford coverage / fixed δ<1.25 | Matterport coverage / fixed δ<1.25 |
|---|---:|---:|
| legacy pixel taps | 0.4543 / 0.2568 | 0.4186 / 0.2559 |
| angular ERP only | 0.5973 / 0.3273 | 0.5506 / 0.3301 |
| angular face only | 0.4490 / 0.2883 | 0.4150 / 0.2789 |
| angular ERP + face | 0.5910 / 0.3694 | 0.5456 / 0.3647 |

The complete route gains about `0.11` fixed δ in both domains and reproduces
all eleven candidate payload hashes exactly. A prospectively frozen follow-up
then selected twelve new view-disjoint pairs without inspecting RGB matches,
dense output, or range-reference outcomes. The fixed population excluded
sky/no-return through the dataset's positive finite range return, then required
geometric support and independent same-surface observability. Prediction
validity never changed this denominator.

| Held-out domain / neighbourhood | Coverage | Fixed δ<1.25 | Conditional error >2x |
|---|---:|---:|---:|
| Stanford / legacy pixels | 0.4351 | 0.2281 | 0.2810 |
| Stanford / angular ERP + face | 0.5723 | 0.3321 | 0.2391 |
| Matterport / legacy pixels | 0.5301 | 0.4130 | 0.1095 |
| Matterport / angular ERP + face | 0.6420 | 0.5130 | 0.0940 |

The angular route passed every frozen scientific gate, improved all three
reported measures in both domains, and took `1.226×` the median runtime in the
first execution and `1.212×` in the replay, below the `1.35×` ceiling. The
complete evidence file, all 24 output payloads, and all 24 population masks
reproduced exactly. A separate reviewer then reproduced the same evidence SHA
and every payload/population hash. The angular values are now defaults for new
option objects, while explicit `None` and old saved configurations preserve the
legacy one-pixel behavior. This does **not** promote panoramic dense stereo to
Stable: pose-estimation validation, transferable uncertainty/abstention
evidence, and installed-wheel streaming evidence remain open.

The three masks answer different questions:

- `sampling_support`: did at least one cube tile cover this direction?
- `observable`: did at least one registered baseline provide enough parallax
  and valid source support?
- `validity`: did the matching cost, texture, confidence, and range interval
  also pass?

`source_consensus_count` answers a fourth question: how many source views
independently preferred an inverse-range label within the configured lattice
tolerance of the fused label? It is diagnostic by default. Set
`min_range_consensus_sources=2` only when a conservative two-source subset is
more valuable than dense coverage; the rejected pixels remain invalid rather
than being filled from another prior.

Thus full-sphere sampling does not imply fabricated full-sphere depth.
Occlusion, missing source support, textureless regions, and inconsistent
appearance can still leave valid directions empty. With only one source,
directions around its baseline axis remain explicitly unobservable.

## Inspect or replace stages

```python
pipeline = stereo.as_pipeline()
print(tuple(pipeline.named_steps))
```

The projected route exposes:

1. `ResolveStereoPose`;
2. `ResolveStereoScale`;
3. `PlanRectifiedStereoViews`;
4. `ProjectStereoViewPairs`;
5. `SelectStereoViewPairs`;
6. `DenseStereoPerPair`;
7. `BackProjectStereoRange`;
8. `FuseStereoRange`;
9. `ValidatePanoramicStereo`.

Use the low-level APIs when developing a new matcher or pairing policy. Use the
application façade when the desired output is a complete reference-ERP range
product.
