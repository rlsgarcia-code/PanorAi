Benchmarks and performance
==========================

This page records what PanorAi has actually measured, separates runtime from
accuracy, and makes the evidence boundary explicit. A benchmark result is not
an API-stability promise.

Evidence rules
--------------

* Public real-scene results use Matterport3D and Stanford2D3D only. Their data
  is not distributed with PanorAi; obtain it under the dataset owners' terms.
* Corpora are reported separately. A macro-average would hide their different
  acquisition geometry, texture, and overlap.
* Buildings (Matterport3D) and areas (Stanford2D3D) are indivisible groups.
  Frames, faces, patches, or pairs from one group must not be split across
  development and test.
* Runtime measurements identify the installed artifact, machine, input shape,
  first-call policy, and repetition statistic.
* Selective estimators report both accepted-result precision and coverage.
  High precision at low coverage is useful, but it is not high recall.
* ``Experimental`` evidence may have been gathered from a source checkout.
  Such a result is labeled and must not be represented as installed-release
  evidence.

Image-processing routes
-----------------------

The reproducible runner
``scripts/benchmark_image_processing_routes.py`` compares:

1. direct ``spherical_gaussian_blur`` over an ERP; and
2. ERP → six cube faces → OpenCV Gaussian → back-projection → Gaussian blend.

Both use the bundled checksum-pinned CC0 panorama resized to 512×1024, a 5×5
kernel with sigma 1.2, and a 37° yaw-equivariance probe. Cube faces are 256²
with 95° FOV. Fifteen reused calls follow one separately measured first call,
and the spherical backend is required explicitly as ``native``.

Environment: installed PanorAi ``3.4.1.dev14+gfae9478a1``, Apple M3 Max,
macOS arm64, CPython 3.12.4, NumPy 1.26.4, and OpenCV 4.11.0.

.. list-table::
   :header-rows: 1

   * - Route
     - First call
     - Reused median
     - Reused P95
     - Yaw-equivariance MAE
     - P95 absolute error
   * - Direct spherical/native
     - 7.99 ms
     - 7.99 ms
     - 8.27 ms
     - 0.00113
     - 0.00456
   * - Cube/project/filter/reconstruct
     - 268.16 ms
     - 104.25 ms
     - 303.78 ms
     - 0.00328
     - 0.01349

For this configuration, the reused direct route was 13.0× faster. This is one
operator on one public image, not a universal ranking. The routes have matched
local raster scales but are not bit-identical: one uses constant-angle tangent
samples and the other uses face-local pixels plus overlap blending. The quality
probe measures approximate rotation equivariance; neither route is the other's
oracle. The rotation/resampling floor for the probe was MAE 0.00685.

Reproduce from an installed artifact, outside the source checkout:

.. code-block:: bash

   python scripts/benchmark_image_processing_routes.py \
     --require-installed \
     --backend native \
     --output image-processing-routes.json

Projection/process/reconstruction microbenchmark
------------------------------------------------

A separate installed-wheel microbenchmark measures an arbitrary-N Fibonacci
face set (N=42), ERP 512×1024, and 256² faces. It compares scalar projection
loops with batched projection and cached geometry on the same Apple M3 Max
environment.

.. list-table::
   :header-rows: 1

   * - Operation
     - First-use speedup
     - Reused speedup
     - Peak-RSS change
   * - Face generation
     - 2.48×
     - 2.59×
     - +3.2%
   * - Gaussian reconstruction
     - 1.12×
     - 17.70×
     - −49.5%
   * - End to end
     - 1.61×
     - 2.33×
     - −26.3%

The RSS increase for first-use face generation is a trade-off, not a failure:
cached geometry is retained so subsequent projection/reconstruction avoids
rebuilding it. Run ``scripts/benchmark_geometry.py`` and
``scripts/benchmark_native_multiface.py`` for current-machine measurements.

Two-view relative pose
----------------------

The public real-scene campaign contains 2,340 preselected pairs across 66
indivisible spatial groups. Predictions were frozen before ground-truth access.
The evidence was gathered from an Experimental source checkout, not promoted
as an installed-release acceptance result.

.. list-table::
   :header-rows: 1

   * - Corpus
     - Pairs
     - Primary success
     - Strict success
   * - Matterport3D
     - 1,890
     - 740 (39.2%)
     - 661 (35.0%)
   * - Stanford2D3D
     - 450
     - 84 (18.7%)
     - 67 (14.9%)

The conservative acceptance gate illustrates the precision/coverage trade-off:
on Matterport3D, 617 of 618 accepted results were correct (99.84% selective
precision) and recall was 83.4% relative to the campaign's correct-result
target. On Stanford2D3D, 61 of 72 accepted results were correct (84.7%
selective precision). Stanford evidence covers only three areas and is therefore
descriptive, not a broad domain claim.

Read :doc:`tutorials/04_two_view_geometry` for pose conventions, thresholds,
degeneracy checks, and the distinction between translation direction and
metric translation.

Dense spherical stereo
----------------------

The current public development evidence uses ten Matterport3D/Stanford2D3D
pairs with known metric pose. Pixel-weighted AbsRel is 0.180,
``delta < 1.25`` is 0.879, and median accepted coverage is 0.295. This is a
small, post-hoc development set; it is useful for regression detection but not
sufficient for promotion. A preregistered, group-independent public evaluation
is still required.

Read :doc:`tutorials/06_spherical_dense_stereo` for the search interval,
radial-range convention, confidence, consistency, and failure modes.

Two-view Gaussian depth feedback
--------------------------------

A separate source-checkout experiment used two posed benchmark panoramas, a native
Metric3D prior, 69 sparse bundle-adjusted landmarks, and 824,040 fused
visible-surface Gaussian centres. The Gaussian correction was solved on a
1024x2048 ERP lattice and applied to the original 4128x8256 prior. Prediction
arrays and hashes were frozen before registered depth was opened.

On 25,162,111 common evaluation pixels, AbsRel improved from 0.244529 to
0.196196, RMSE from 1.472432 m to 1.284763 m, and ``delta < 1.25`` from
0.489551 to 0.619498. Of the 17,613,016 pixels changed by at least one percent,
86.34% moved closer to registered depth. A fresh full rerun reproduced all
nine prediction arrays byte for byte.

This result covers one external benchmark pair and includes a monocular prior
in the construction path. It is evidence for the bounded feedback mechanism,
not an independent Gaussian-depth oracle, general scene accuracy claim, or
installed-wheel result. No dataset image, checkpoint, PLY, depth array, or
generated panel is distributed with PanorAi.

Read :doc:`tutorials/12_two_view_gaussian_depth` for the PLY schema,
visibility/confidence rules, edge-aware solve, commands, outputs, and failure
boundaries.

Multi-view reconstruction
-------------------------

The source-checkout campaign contains all 423 selected multiview sets:
378 Matterport3D and 45 Stanford2D3D sets across 66 groups.

.. list-table::
   :header-rows: 1

   * - Route / slice
     - Full
     - Primary
     - Strict
     - Important qualification
   * - Native PanorAi mapper, all sets
     - 131/423
     - 131/423 (31.0%)
     - 131/423 (31.0%)
     - all 131 complete outputs were strict-correct
   * - Native Matterport held-out gate
     - —
     - 18/22 (81.8%)
     - 18/22 (81.8%)
     - thresholds were development-tuned
   * - Native Stanford descriptive slice
     - —
     - —
     - 8/45
     - only three Stanford areas
   * - PyCOLMAP mapper-core A/B, all sets
     - 172/423
     - 158/423
     - 137/423
     - same PanorAi frontend; not a full GLOMAP pipeline

For native primary-complete cases, median rotation error was 0.331° and median
translation-direction error was 0.460°. The native route demonstrates strong
selective accuracy but limited coverage. The PyCOLMAP row is an interoperability
A/B of the mapper core, not evidence that every PyCOLMAP/COLMAP configuration
has the same behavior.

Read :doc:`tutorials/05_multiview_reconstruction` for native and PyCOLMAP
routes, gauge conventions, and diagnostics.

Current evidence gaps
---------------------

.. list-table::
   :header-rows: 1

   * - Area
     - What is established
     - What remains
   * - Direct spherical feature frontend
     - unit, parity, rotation, and synthetic localization tests; C++/NumPy extrema parity
     - rerun detection, descriptor, matching, and pose metrics on public Matterport3D and Stanford2D3D splits
   * - CLAHE
     - planar CLAHE can be caller-supplied per projected view
     - first-party seam/pole-safe spherical contextual equalization and downstream evaluation
   * - Image-processing route comparison
     - reproducible Gaussian timing and yaw-equivariance probe on one CC0 ERP
     - multiple resolutions, kernels, rotations, devices, and public real-scene images
   * - Dense stereo
     - small public development regression
     - preregistered, group-independent accuracy and runtime evaluation
   * - Native multiview
     - high strict accuracy among complete results
     - improve coverage and validate on more independent buildings/areas

See :doc:`tutorials/08_benchmarking` to run and interpret the benchmark
suite without leaking test data into method selection.
