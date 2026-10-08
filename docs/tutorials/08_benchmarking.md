# Tutorial: benchmark speed and geometric accuracy

This tutorial turns a PanorAi experiment into evidence that someone else can
interpret and reproduce. It distinguishes microbenchmarks, stage metrics, and
end-to-end success; none can substitute for the others.

## 1. Freeze the evaluation unit

The independent unit is a physical scene group, not an image, crop, face,
patch, pair, or multiview set.

- keep every Matterport3D panorama from one building in one split;
- keep every Stanford2D3D panorama from one area in one split;
- freeze selection rules before reading test outcomes;
- report Matterport3D and Stanford2D3D separately before any aggregate;
- record whether ground truth was available to selection, tuning, or only
  final scoring.

PanorAi does not distribute either dataset. Follow the dataset owners'
licenses and point benchmark manifests at your local copies.

## 2. Benchmark the installed artifact

Running inside a checkout can import local source instead of the built wheel.
Create a clean environment, install the wheel under test, change to a directory
outside the repository, and record:

- `panorai.__version__` and `panorai.__file__`;
- Python, NumPy, OpenCV, OS, architecture, and CPU/GPU;
- input shape, dtype, layout, FOV, face resolution, interpolation, and blender;
- first-call time separately from reused calls;
- warm-up count, repetitions, median, P95, and peak memory when relevant.

For the two image-processing routes:

```bash
python /path/to/PanorAi/scripts/benchmark_image_processing_routes.py \
  --require-installed \
  --backend native \
  --output image-processing-routes.json
```

The runner starts each route in a fresh process. It times direct spherical
Gaussian filtering and the complete project/filter/reconstruct route, then
evaluates both with the same yaw-equivariance probe.

## 3. Measure each stage, then the full pipeline

For detection and relative pose, store at least:

| Stage | Runtime | Accuracy / quality |
| --- | --- | --- |
| preprocessing | wall time and peak memory | dynamic range, valid support |
| detection | pyramid, extrema, localization, selection | keypoint count, angular coverage, repeatability |
| description | patch materialization and descriptor compute | valid descriptor rate, duplicate/ambiguity rate |
| matching | search and filtering | match count, precision against GT epipolar compatibility, spatial coverage |
| robust pose | hypotheses, scoring, refit, quality gate | inlier precision/recall, residual, accepted/rejected reason |
| end to end | total cold and reused time | rotation error, translation-direction error, success and abstention |

Do not infer that a faster detector gives better pose, or that more matches
give better pose. Preserve intermediate artifacts so a failure can be assigned
to detection, description, matching, robust consensus, decomposition, or the
acceptance gate.

## 4. Report selective estimators honestly

A quality gate creates three different quantities:

1. **coverage** — fraction of cases for which a result is accepted;
2. **selective precision** — fraction of accepted results that satisfy the
   declared accuracy criterion;
3. **recall** — fraction of target-solvable/correct cases recovered.

Always report them together. “99% precision” without coverage can describe a
system that almost always abstains.

For continuous errors, report distributions—not only means:

- rotation geodesic error in degrees;
- translation-axis and oriented-direction error in degrees;
- spherical epipolar residual in degrees;
- radial-range AbsRel and threshold accuracies;
- median, P90/P95, and clearly declared failure handling.

## 5. Separate tuning from evidence

Use development groups to choose patch FOV, descriptor resolution, Lowe ratio,
RANSAC thresholds, and gate parameters. The frozen test split may be scored
once for the registered configuration; it must not guide a second choice.

Every result record should contain:

- exact command and resolved configuration;
- generating commit and installed version;
- dataset, split, group IDs, and oracle-access declaration;
- coordinate frame, units, pose direction, and scale convention;
- counts before and after every filter/gate;
- machine-readable metrics plus a short limitations section;
- checksums or stable URIs for large external artifacts.

## 6. Read the current evidence

The project-wide {doc}`../benchmarks` page records the current public-dataset
accuracy and installed-artifact performance evidence, including limitations
and outstanding promotion gates. Continue with:

- {doc}`03_features_and_matching` for frontend choices;
- {doc}`04_two_view_geometry` for relative-pose diagnostics;
- {doc}`06_spherical_dense_stereo` for dense metrics;
- {doc}`05_multiview_reconstruction` for native/PyCOLMAP comparison.
