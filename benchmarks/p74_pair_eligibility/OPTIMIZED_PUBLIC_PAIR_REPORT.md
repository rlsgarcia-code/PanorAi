# Optimized public spherical pair route on P74

## Scope

This record verifies the optimized pair route requested for PanorAi `origin/main` without dense stereo. It is a cold-process benchmark of one P74 pair with independently measured 71.08% minimum directional cloud overlap. It verifies implementation and timing; one pair is not enough to revise the release overlap threshold.

The exact package under test was a wheel built from `origin/main` commit `a00b74f3833b92f47c1e7537e91c9161350f0ecc` and installed outside the repository:

- resolved version: `panorai 3.4.1.dev17+ga00b74f38`;
- wheel: `panorai-3.4.1.dev17+ga00b74f38-cp312-cp312-macosx_14_0_arm64.whl`;
- wheel SHA-256: `6ef12a685d9ca4ef0f6dce8f0fa711b51b15b58540b6cd0ab4e3f1ca80eeb38e`;
- resolved import: `/private/tmp/panorai-val016-venv/lib/python3.12/site-packages/panorai/__init__.py`;
- detector provenance in both outputs: `ga00b74f38`.

No editable install, source-checkout import, multiface route, private PanorAi import, sequential detector call, or NumPy convolution fallback was used.

## Confirmed route

- `SphericalDoGDetector.detect_batch()` processed exactly two equal-resolution panoramas.
- `convolution_backend="native"`; native filter support was present and NumPy fallback was not permitted.
- `max_keypoints=4096`.
- Explicit boolean P74 geometric-support masks were passed to detection and patch materialization; validity was not inferred from black pixels.
- `TangentPatchProvider(max_workers=4)` materialized 48x48 upright tangent patches with radius six scales.
- `OpenCVTangentDescriptorV2` used the calibrated 1.25-diameter, one-scale, fixed-zero, locally standardized RootSIFT profile.
- Matching used FLANN with Lowe ratio 0.72 and angular deduplication.
- Pose used the frozen spatially weighted, MSAC-first, 1-degree profile with 1,000 trials, six stability trials, and model competition.

The complete resolved configuration is serialized in each raw result JSON.

## Environment

| Item | Value |
|---|---|
| CPU | Apple M3 Max, 16 logical cores |
| Memory | 64 GB |
| OS | macOS 26.6.2 arm64 |
| Python | 3.12.4 |
| OpenCV | 4.11.0, GCD parallel framework |
| Patch workers | 4 |
| BLAS/OpenMP environment | OMP, OpenBLAS, MKL, vecLib, NumExpr all fixed to 1 |
| OpenCV thread request | 1, but GCD continued to report 16; see limitation below |

## Accuracy on the 71.08% overlap pair

| Resolution | Keypoints A/B | Descriptors A/B | Matches | Inliers | Rotation error | Translation-direction error | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| 1024x2048 | 3,235 / 3,741 | 3,216 / 3,720 | 43 | 31 | 0.705° | 0.395° | precise valid |
| 2048x4096 | 4,096 / 4,096 | 4,080 / 4,077 | 47 | 31 | 0.258° | 0.589° | precise valid |

The pose result is expressed by PanorAi in its canonical frame. Errors above were computed only after the known P74-to-PanorAi frame transform was applied to the estimate; directly comparing matrices in unlike frames produces a false approximately 11°/98° failure.

## Stage timing

Input adaptation is reported separately and excluded from the comparable pair total.

| Resolution | Detect pair | Detect/image | Patches pair | Descriptor pair | Matching | Pose | Ready/image | Pair total | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024x2048 | 5.890 s | 2.945 s | 5.978 s | 1.228 s | 0.039 s | 1.999 s | 6.548 s | 15.134 s | 1,708 MiB |
| 2048x4096 | 20.082 s | 10.041 s | 10.430 s | 1.439 s | 0.048 s | 2.070 s | 15.976 s | 34.069 s | 2,890 MiB |

| Resolution | Metric | Measured | Reference | Ratio | 1.5x gate |
|---|---|---:|---:|---:|---|
| 1024x2048 | detection pair | 5.890 s | 4.220 s | 1.40x | pass |
| 1024x2048 | ready/image | 6.548 s | 3.440 s | 1.90x | fail |
| 1024x2048 | full pair | 15.134 s | 8.990 s | 1.68x | fail |
| 2048x4096 | detection pair | 20.082 s | 18.390 s | 1.09x | pass |
| 2048x4096 | ready/image | 15.976 s | 14.350 s | 1.11x | pass |
| 2048x4096 | full pair | 34.069 s | 30.960 s | 1.10x | pass |

At 2048x4096, every primary comparison is within 1.11x of the supplied reference. At 1024x2048, detection itself stays below 1.5x, while patch materialization consumes 5.98 seconds per pair and drives ready/image and full-pair totals above the gate. Matching and pose do not explain the gap.

## Threading conclusion

There was no NumPy convolution fallback and no sequential detection. The BLAS/OpenMP thread pools were explicitly limited to one and patch materialization used exactly four workers. OpenCV's macOS GCD build ignored `cv2.setNumThreads(1)` and continued to report 16 threads, so strict absence of every form of OpenCV oversubscription cannot be claimed. This does not match the slow-stage signature: descriptor plus matching took only 1.27 seconds at 1024, while the 5.98-second patch stage uses PanorAi geometry rather than the OpenCV descriptor pool.

## Interpretation

The optimized public route from `main` is reproduced and operational. On this high-overlap P74 pair it returns a precise pose at both resolutions, and the canonical 2048x4096 runtime is in the expected range. The 1024x2048 path misses the performance gate because fixed-count tangent patch materialization does not fall proportionally with raster resolution.

This run must not be used by itself to recommend a minimum spatial overlap for release. That recommendation still requires the frozen multi-pair overlap evaluation; this record changes the implementation/timing evidence, not the statistical overlap evidence.

## Raw evidence

- `.agents/results/VAL-016-optimized-main-pair-route/20261008T214500-0300/panorai-val016-result-1024.json`
- `.agents/results/VAL-016-optimized-main-pair-route/20261008T214500-0300/panorai-val016-result-2048.json`

Raw JSON SHA-256 values are `5596558c7a2876a9cc580ad9a435d28cbdc25d9b8f478ae126d37918b3259d26` for 1024 and `6e298139d89b1072b29f9fab48631452b764a4cf70e9c94c10e299aca9432018` for 2048.
