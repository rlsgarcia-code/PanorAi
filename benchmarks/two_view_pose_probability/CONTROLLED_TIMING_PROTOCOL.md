# Controlled timing protocol for the aligned spherical two-view route

Status: frozen after detecting shared-host contention and before opening any
aligned population accuracy aggregate. Population-replay wall times remain
diagnostic and must not be substituted for this benchmark.

## Objective

Measure PanorAi 3.5.0 frontend, matching, pose, total time, and peak memory as
a function of spatial-overlap stratum without contaminating the probability
models or selecting pairs by pose outcome.

## Artifact and route

- PanorAi 3.5.0 wheel SHA-256
  `e861dafbaa5991aef77dd512b3ef1bf6fdc7967d10fbd236d850bab5c1a5f8a7`.
- Source commit `03c5b36b28225b24d3909286bf53250d7b532aa3`, tree
  `c0a7d8bbf7ed1f29ff449e77d7e4b12afda40043`.
- Installed-wheel import outside the repository.
- 1024×2048 panoramas, explicit masks, native convolution, batch-two
  `SphericalDoGDetector.detect_batch()`, 4,096-keypoint capacity, four tangent
  patch workers, 48×48 upright patches, and calibrated one-scale RootSIFT.
- The same frozen matcher, R,t estimator, quality policy, and
  translation-orientation diagnostics as the population replay.

Any source-tree import, NumPy convolution fallback, sequential detector path,
configuration drift, or missing mask invalidates the run.

## Outcome-blind sample

Stratify each dataset by the frozen overlap bands `<10%`, `10–25%`, `25–50%`,
`50–70%`, and `≥70%`. Within each dataset × band cell, order independent
components by a frozen SHA-256 key and traverse them round-robin, ordering
pairs inside each component by a second frozen SHA-256 key. Select up to five
pairs. This maximizes group diversity before returning to the same component.
Selection may use dataset, pair ID, independence component, overlap band,
resolution, and validity availability only. It may not use returned/accepted
state, match count, pose error, runtime, or any other algorithm outcome.

The frozen selection manifest must be written and hashed before the timing
process reads any result object. Empty or underfilled cells remain explicit.

Prepare it with:

```bash
python benchmarks/two_view_pose_probability/prepare_controlled_timing_manifest.py \
  --features /path/to/frozen/features.jsonl \
  --pairs-per-cell 5 \
  --output-dir /private/tmp/panorai-val018-controlled-timing-selection
```

## Host-control gate

Record CPU model, physical/logical cores, RAM, macOS version, Python, NumPy,
OpenCV, compiler/runtime libraries, power source/mode, thermal state, and all
thread-related environment variables.

Start only when:

- no unrelated CPU-, GPU-, or memory-intensive experiment is running;
- the host is on stable power and not thermally throttled;
- free memory comfortably exceeds the observed peak requirement;
- the timing process has exclusive ownership of its output directory;
- one untimed route-validation pair confirms the exact wheel and native path.

If the gate fails during a repetition, record the interruption and rerun the
entire affected repetition; do not delete the failed diagnostic.

## Repetitions

1. Run one untimed warm-up pair per dataset.
2. Execute three complete repetitions of the frozen pair manifest.
3. Randomize order deterministically per repetition using a recorded seed.
4. Use one pair process at a time. Do not parallelize pairs.
5. Preserve raw per-pair/per-repetition JSON and peak RSS.

Report cold process-start cost separately if each pair launches a fresh
process. The primary comparison uses the same process-lifecycle definition as
the existing 1024×2048 reference.

## Metrics

For each dataset, overlap band, and complete benchmark:

- detection per pair and per image;
- patch materialization per pair;
- descriptor per pair;
- matching;
- pose estimation;
- total two-panorama time;
- peak RSS;
- panorama resolution and keypoint counts.

Report raw observations, median, P95, repetitions, and cell sample size. Do not
average pair medians as if they were independent groups. Runtime is an
engineering outcome and never enters a probability model.

## Reference comparison and interpretation

At 1024×2048 the route reference is 4.22 s detection per cold pair and 8.99 s
complete cold pair, with 2.12 s detection and 3.44 s image-ready time per
image. Hardware differs, so first compare stage order of magnitude. Any stage
above 1.5× the corresponding reference must be diagnosed for fallback,
sequential execution, thread oversubscription, thermal throttling, I/O, or host
contention before a performance conclusion is accepted.

Overlap is not expected to materially change convolution cost. A measured
trend may reflect keypoint/patch counts, matching size, robust-estimation work,
or scene-dependent stability trials; report stage decomposition rather than
attributing the trend causally to overlap.
