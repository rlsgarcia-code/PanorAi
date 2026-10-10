# Two-view Gaussian depth-prior feasibility experiment

This development-only benchmark asks a narrow question: can a dense monocular
radial-range prior, robustly scaled by sparse bundle-adjusted landmarks, be
represented and locally refined as visible-surface Gaussians using one second
posed panorama?

The frozen P74 experiment trains on G046 and G047. G048 and the registered
G046 depth are not opened until the optimized range map and its SHA-256 have
been written. One isotropic Gaussian is initialized per supported target pixel.
RGB, opacity and camera pose remain fixed; only a smooth log-range correction
field is optimized. No Gaussian is created outside the target panorama's
declared support.

The plain-PyTorch renderer is intentionally small and auditable. It runs on
CPU or Apple MPS, selecting MPS automatically when the process can access it.
It uses Gaussian image-space accumulation and a nearest-depth visibility gate.
It is a feasibility control for the geometry/densification hypothesis, not a
claim of feature parity with a canonical 3DGS rasterizer or its adaptive
split/prune policy.

```bash
python benchmarks/two_view_gaussian_splatting/run_p74_experiment.py \
  --p74-root /path/to/P74/eq \
  --prior /path/to/G046-monocular-radial-m.npy \
  --landmarks /path/to/G046-G047-ba-landmarks-m.npy \
  --refined-pose /path/to/VAL-042-refined-pose.json \
  --ground-truth /path/to/G046-registered-radial-m.npy \
  --output /path/to/output \
  --device auto
```

Run the command from the repository root, or use the script's absolute path.
For a visual-quality render rather than the fast feasibility default, use
`--height 1024 --width 2048 --render-gaussian-sigma-px 1.15
--render-gaussian-radius-px 3`. These evaluation-only options preserve the
optimized geometry while avoiding the hard one-pixel kernel truncation in the
final render. They do not invent geometry in disoccluded regions.

The primary comparisons are unaligned monocular depth, landmark-scaled depth,
and two-view Gaussian-optimized depth. An oracle-depth render is reported only
to expose the renderer/photometric floor; it is never used for optimization.

## Native-resolution two-surface path

The useful two-image result requires running the perspective depth model at
its trained spatial scale. Inferring a 1024x2048 ERP through 196x336 tangent
windows retained the angular field of view but collapsed Metric3Dv2's depth
variation: landmark depth correlation was only 0.14. Inferring the original
4128x8256 ERP through 812x1400 windows raised that correlation to 0.69 and the
registered log-depth correlation to 0.91. Downsampling happens only after
native inference.

Supply one native radial-range NPY for each observed panorama. Model inference
is deliberately outside this benchmark: external Metric3D source, checkpoints,
and their license terms remain caller-managed and are never redistributed by
PanorAi. The arrays used in the frozen run were generated at 4128x8256 through
812x1400 tangent windows and were not resized before model inference.

Then build both observed surfaces. `consistent` keeps only samples corroborated
by both depth and RGB in the other view. `all-seen` retains every sample seen
by either input and gives much denser bounded novel views; it still creates no
surface outside the two source panoramas' support.

```bash
python benchmarks/two_view_gaussian_splatting/run_stereo_surface_experiment.py \
  --p74-root /path/to/P74/eq \
  --landmarks /path/to/G046-G047-ba-landmarks-m.npy \
  --refined-pose /path/to/VAL-042-refined-pose.json \
  --ground-truth /path/to/G046-registered-radial-m.npy \
  --target-prior /path/to/G046-prior/metric3d-radial-m.npy \
  --source-prior /path/to/G047-prior/metric3d-radial-m.npy \
  --prior-mode all-seen \
  --interpolation-alpha 0.10 \
  --height 1024 --width 2048 \
  --device mps \
  --output /path/to/output
```

For G046/G047, the visually stable range is deliberately bounded near an
observed camera (approximately alpha 0.00-0.20 or 0.80-1.00). G048 is not an
interpolation test: its camera centre is 1.59 m from G046 and 2.78 m from G047,
on the opposite side of G046 from G047. Rendering it is therefore severe
extrapolation, and its holes or floaters must not be presented as a failure of
two-view interpolation. The primary/fallback compositor preserves the nearer
training surface and uses the other only to fill disocclusions.

## Dense RGB point-cloud export

Export the Gaussian centres from both observed surfaces into the G046 frame:

```bash
python benchmarks/two_view_gaussian_splatting/export_gaussian_point_cloud.py \
  --p74-root /path/to/P74/eq \
  --surface-run /path/to/two-view-surface-run \
  --refined-pose /path/to/VAL-042-refined-pose.json \
  --voxel-size-m 0.01 \
  --output /path/to/gaussian-centres-1cm.ply \
  --preview /path/to/gaussian-centres-preview.png
```

The binary PLY contains XYZ, RGB, the number of centres fused into each voxel,
and a view bitmask (`1` for G046, `2` for G047, `3` for both). The exporter
uses the Gaussian centres themselves; it does not fabricate density by drawing
extra samples inside each covariance ellipsoid.

## Gaussian-to-depth feedback

The exported visible Gaussian surface can be projected back into G046 to
refine the native Metric3D radial-range map. This closes the experimental loop
without treating the cloud as independent ground truth: target-only centres
receive very low weight, source-only centres require RGB agreement, and the
strongest anchors are visible-surface splats supported by both views or the BA
landmarks. A spherical nearest-depth gate removes occluded contributions.

The correction is solved in log radial range on the Gaussian lattice with
longitude wrapping and RGB edge-aware regularization, then bilinearly applied
to the original native map. Invalid prior support remains invalid; the method
does not create unseen surfaces.

```bash
python benchmarks/two_view_gaussian_splatting/run_gaussian_depth_feedback.py \
  --p74-root /path/to/P74/eq \
  --prior /path/to/G046-prior/metric3d-radial-m.npy \
  --gaussian-cloud /path/to/gaussian-centres-consistent-1cm.ply \
  --landmarks /path/to/G046-G047-ba-landmarks-m.npy \
  --ground-truth /path/to/G046-registered-radial-m.npy \
  --solve-height 1024 --solve-width 2048 \
  --output /path/to/depth-feedback
```

Prediction arrays and checksums are frozen before the evaluation-only ground
truth is opened. The principal outputs are native-resolution
`refined-radial-m.npy` and `refined-confidence.npy`; the solve-lattice Gaussian
depth, confidence, provenance, correction and acceptance maps remain available
for diagnosis. On the frozen G046 run, full-support AbsRel changed from
`0.24453` to `0.19620`, delta-1 from `0.48955` to `0.61950`, and 86.34% of the
17.61 million pixels changed by at least one percent moved closer to registered
depth. These are development-only P74 results, not a general accuracy claim.
