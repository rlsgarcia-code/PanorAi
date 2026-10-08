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
