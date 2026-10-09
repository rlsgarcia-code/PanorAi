# Dense / pose / match refinement experiments

This benchmark is the frozen evaluation harness for the six progressive
experiments that connect PanorAi spherical feature matches, relative pose and
dense radial range. Each stage must retain the preceding stage as an ablation.

Stage 1 asks only whether dense range is useful as an independent match
filter. It compares:

1. the unchanged sparse matches;
2. filtering with the reference pose (an upper bound); and
3. filtering with the pose estimated from the same sparse matches (the usable
   pipeline, where circular confirmation is a known risk).

The scenes are deterministic analytic textured spheres rendered from declared
central-camera poses. Sparse source pixels, exact intersections, angular
noise, descriptor outlier labels and dense reference ranges are known before
the filter runs. Sixty percent of the outliers are deliberately difficult:
they lie on the correct spherical epipolar curve but correspond to the wrong
range. The remainder are random target bearings. Development and held-out
seeds are disjoint. No generated
image or result is distributed with the package.

Run:

```bash
python benchmarks/dense_pose_match_refinement/run_experiment.py \
  --stage 1 --split development
python benchmarks/dense_pose_match_refinement/run_experiment.py \
  --stage 1 --split heldout
```

The command prints one JSON report. It evaluates source-checkout behavior; it
is not installed-wheel or release evidence.

Stage 2 increases the controlled target-bearing localization noise, retains
the stage-1 filter, and adds a dense-center-range, pose-warped tangent-patch
ZNCC search around the predicted target bearing. It reports original/refined
angular error, cost-gated application count and a fresh pose estimate:

```bash
python benchmarks/dense_pose_match_refinement/run_experiment.py \
  --stage 2 --split development
python benchmarks/dense_pose_match_refinement/run_experiment.py \
  --stage 2 --split heldout
```

## Stage 3 gate: negative

Stage 3 tested whether the refined matches could safely improve rotation and
translation direction while keeping radial range fixed and preserving the
translation norm. Three independently frozen designs were evaluated after the
initial same-data gate failed held-out:

1. spatial cross-fit with an essential-matrix candidate;
2. fixed-range nonlinear pose correction with one and three spatial folds; and
3. an independent third-view gate using A-C triangulated ranges, including
   joint, rotation-only and translation-direction-only candidates.

None met the progression contract. Two-view gates either failed to improve
held-out medians or moved exact-pose controls. Noise-calibrated third-view
evidence rejected all exact and insufficient-track controls, but no ordinary
candidate produced a significant held-out per-track gain even in development.
The correction candidates used a smaller filtered subset than the already
accurate initial estimator and supplied no reproducible signal above match and
triangulation noise.

A subsequent real-scene confirmation used the best direct spherical-convolution
P74 frontend on the 18 already-opened pairs with at least 50% registered-cloud
overlap. Metric translation norm was supplied externally and held fixed; dense
range was estimated from RGB at 256x512 under the initial pose. Dense filtering
and filtering plus subpixel refinement both reduced 14 strict initial poses to
13, recovered none of the four initial failures, and caused the initially
strict M-052 -> M-053 pair to return no candidate pose. Median dense time was
0.57 seconds, but the complete added path was 11.26 seconds per evaluated pair
because two full pose re-estimations dominated the cost. This development-only
result is another negative gate; no untouched P74 confirmation set was opened.

Consequently no stage-3 implementation is retained, and stages 4-6 are not
entered automatically. In particular, a global photometric pose optimizer or
an alternating range/matches/pose loop must not be justified by lower residual
on range estimated under that same pose. A future attempt needs a materially
different candidate with independent confirmation; the reserved analytic
confirmation seeds must not be used for threshold retuning.
