# Human-guided planar reconstruction

`panorai.reconstruction.planar` is an Experimental, NumPy application over
registered central panoramas. It estimates one world-space plane per accepted
surface and generates native-grid **radial ranges in metres** by intersecting
unit rays with that plane. It does not assume Manhattan directions.

The human identifies surface membership; registered image observations provide
metric geometry. A mask alone cannot determine a plane's distance. Completely
textureless regions need sufficient correspondences on identifiable features,
or manual point pairs distributed over the same physical surface.

## Workbench

Open `/multi-view.html`. Import a `panorai-planar-scene/v1` JSON and its images,
then create planes and draw in either gnomonic viewport. A and B can select any
of the registered views. The small ERP images reposition the editing view.

The fixed toolbar above the images exposes **Desfazer / Refazer**, **Mão** and
**Zoom** independently of inspector scrolling. Ctrl/⌘ Z undoes annotations;
Ctrl/⌘ Shift Z or Ctrl Y redoes them. While drawing a polygon, these actions
remove/restore draft vertices before affecting the completed annotation history.
History is local to the editing session; reopening a revision resets it.

Click **Mão** (H or V) and drag to rotate the sphere, or hold **Space** for a
temporary hand and release to return to the selected brush. The cursor changes
from an open to a closed hand while dragging. **Zoom** (Z) uses vertical drag:
up zooms in, down zooms out. The wheel, +/− buttons and **FOV** slider also
adjust magnification, bounded to 15–120 horizontal degrees. Choose **Vista
ativa A/B** or interact with a viewport to target those controls; each view
keeps its own orientation and FOV. These are editing-view changes, not physical
camera pose changes, and do not paint, invalidate masks or consume undo steps.
Finish/cancel a pending polygon before navigating so its frozen projection
remains coherent. Editing/navigation is locked while a numerical run executes.

The inspector is divided into four numbered sections with separate actions:

1. **Marcar plano:** paint a rough estimate or draw a polygon (Enter closes
   it). Use negative brush strokes to exclude foreground objects.
   Brush + / − show distinct cursors and the projected spherical footprint;
   its size follows the angular-radius slider. With no plane selected, the
   first brush stroke creates a new plane in the painted view. **New plane**
   remains available to explicitly start another surface.
   Enable **Preencher interior ao fechar (Brush +)** to fill enclosed regions
   when releasing a positive stroke. It is off by default. Separate strokes
   can complete the boundary; open gaps are not automatically bridged.
   **Preencher contornos existentes** applies the same operation to the current
   region without redrawing. Explicit negative seeds and unsupported pixels
   remain excluded. A stroke and its automatic fill are one undo operation;
   manual fill has its own undo entry. Filling never accepts a mask for you.
2. **Segmentar:** click **Segmentar região**. The default is GrabCut with an
   automatic native-grid ROI around the rough estimate (eight angular degrees
   or 15% padding, whichever is larger). The exact ROI is persisted in the run's
   input revision. Only the active plane/view is segmented; other drafts remain
   untouched. The proposed mask is immediately displayed over the image.
3. **Revisar e confirmar:** correct the proposal if needed and click
   **Aceitar máscara**. A generated mask remains a draft until this explicit
   confirmation. Repeat to mark more surfaces.
4. **Reconstruir:** estimate the planes, load proposals in auxiliary views,
   correct occlusions and accept those observations. Rerun for final ranges.

**Avançado**, collapsed by default, contains the method selector, manual ROI,
seed/final interpretation, manual point pairs, SIFT options, resolution,
tolerances, diagnostics and alternative masks. SAM uses positive/negative
points, an ROI box and propagated mask prompts; loading still requires explicit
upstream terms acceptance. Choosing manual ROI preserves the original polygon
workflow. All discrete masks retain nearest-neighbour semantics.

### Seeded watershed and a session-local classifier

Two additional choices are available under **Avançado → Segmentação**:

- **Watershed · sementes e bordas:** foreground strokes are markers in a
  smoothed Lab-color gradient relief. Negative strokes, outside-ROI pixels and
  chart borders compete as background. Flooding is computed in overlapping
  gnomonic charts, and labels return to ERP with nearest sampling. A weak or
  missing image edge may leak; an internal texture edge may stop expansion too
  early. A stroke is a foreground seed, not a closed contour interpreted as a
  barrier. Use the separate ROI contour to bound expansion, or the existing
  closed-fill control if the interior should be positive too.
- **Classificador local · positivos / negativos:** click **Desenhar limite da
  região**, close its polygon with Enter, then paint positive surface samples
  and negative samples of objects/background **inside** that contour. Click
  **Segmentar região**. Selecting this method requires a manual ROI; it never
  silently substitutes the default automatic rectangle. Imported explicit ROIs
  are also supported. The blue outline/dimmed exterior indicates the limit,
  gold the current region and red the negative examples. Drawing the limit
  does not label its entire interior positive.

The classifier is deterministic, balanced L2 logistic regression over nine
Lab/quadratic color features, trained from at most 4096 positive and 4096
negative pixels selected in raster order. Prediction evaluates only the
supported native-grid ROI, in row blocks; it does not upscale a small mask.
It requires at least two supported pixels of each class. Indistinguishable
positive/negative colors or failed fitting produce an actionable error, not a
silently accepted mask. Coordinates, depth, semantic class names, external
training data and pretrained weights are not features. No learned model is
persisted or shared across scenes; the saved prompts, immutable inputs and
algorithm configuration reproduce the fitting. History records the algorithm,
sample counts, penalty and fitting iterations. Decision scores are not
calibrated confidence or measured segmentation accuracy.

Both methods return **drafts**, preserve explicit positives/negatives and
publish no selected pixel outside ROI/support. Correct and accept the mask
before reconstruction. Their appearance assumptions do not establish
planarity: a same-colored pipe or differently illuminated floor can still be
misclassified. The methods need real-scene comparison with frozen human masks
before any quality/generalization claim. They reuse existing NumPy, SciPy,
OpenCV and scikit-image dependencies; SAM remains separate and optional.

The default minimum is six usable tracks, a 0.5-degree minimum parallax,
0.5-degree reprojection tolerance, 3-cm plane consensus tolerance and 20%
reserved validation tracks. These are inspectable experimental defaults,
not calibrated guarantees for P74 or other scenes. Conflicting masks suppress
the affected planar values until the human repairs their overlap.

Every run captures the exact input revision. Changing a mask creates another
revision and requires recomputation; earlier results remain attributable to
their original revision. There is no hidden latest-result cache. The first
implementation reruns the complete numerical pipeline, including unaffected
planes; incremental per-plane computation is a later performance improvement.

## Scene input

Each view names an uploaded RGB image and supplies a proper rotation, camera
centre, explicit pose convention and support. The scene requires `units="m"`
and `geometry_convention="panorai-geometry-v1"`.

```json
{
  "schema": "panorai-planar-scene/v1",
  "geometry_convention": "panorai-geometry-v1",
  "units": "m",
  "views": [
    {"view_id":"a", "image_key":"a.png", "pose_convention":"world_to_camera",
     "rotation":[[1,0,0],[0,1,0],[0,0,1]], "center_world":[0,0,0], "support":"full"},
    {"view_id":"b", "image_key":"b.png", "pose_convention":"world_to_camera",
     "rotation":[[1,0,0],[0,1,0],[0,0,1]], "center_world":[0.7,0,0], "support":"full"}
  ],
  "regions": [],
  "correspondences": []
}
```

`world_to_camera` means `x = R @ (X-C)`. It may supply `translation` instead
of `center_world`, in which case `C=-R.T@translation`. `camera_to_world`
requires `center_world` and its rotation is explicitly transposed on import.
External camera-axis or raster conventions must be adapted before import.
There is no automatic guess of the pose convention or metric scale.

Masks are `{shape_hw:[H,W], runs:[start,length,...]}`, with ordered, disjoint
row-major foreground runs. Use `encode_planar_mask`/`decode_planar_mask` in
Python. A view's `support` can be this mask or the explicit string `"full"`.
An optional `baseline_range` requires a separate `baseline_validity` mask;
JSON null can represent invalid numeric data. Regions record `plane_id`,
`view_id`, `mask`, review `state`, `interpretation`, label and drawing history.

Generate an independently rendered demo, with exact registered point tracks:

```bash
python workbench/scripts/create_planar_demo.py --output /tmp/panorai-planar-demo
```

Open its `scene.json` and three PNGs. Only the first mask is accepted initially;
the first reconstruction supplies auxiliary proposals. The scene is synthetic
and must not be reported as measured P74 performance.

For private P74 files, the source-checkout importer adapts the already verified
P74 raster and camera basis without accessing scanner XYZ:

```bash
python workbench/scripts/import_p74_planar_scene.py \
  --npz /external/p74/first.npz --npz /external/p74/second.npz \
  --output /tmp/p74-planar-input
```

The canonical width defaults to the original source width; the output height
is half that width. The manifest records the P74 endpoint-inclusive,
150-degree-source remapping and shifted world origin. “Native range” here
means the loaded canonical ERP grid, not the original scanner raster.

## Python and pipeline

```python
from pathlib import Path
import json
import numpy as np
from PIL import Image
from panorai.reconstruction.planar import (
    HumanGuidedPlanarReconstruction, PlanarReconstructionOptions,
    planar_scene_from_document, export_planar_result,
)

root = Path("/tmp/panorai-planar-demo")
document = json.loads((root / "scene.json").read_text())
images = {v["image_key"]: np.asarray(Image.open(root / v["image_key"]))
          for v in document["views"]}
scene = planar_scene_from_document(document, images)
application = HumanGuidedPlanarReconstruction(
    PlanarReconstructionOptions(automatic_matching=False)
)
run = application.run(scene, retain=("match", "triangulate", "fit"))
print(run.output.planes[0].state)  # provisional until auxiliary masks accepted
print(application.describe())
export_planar_result(run.output, root / "first-result",
                     pipeline_manifest=run.manifest())
```

The wrapper owns a real `Pipeline` with named stages:
`validate → segment → match → triangulate → fit → propagate → range`.
Stages and options are available from `panorai.reconstruction.planar` and
registered with the existing safe pipeline codec. The manual/simple/watershed/classifier configuration
round-trips to JSON. Injected SAM model objects remain runtime resources and
are deliberately not serialized as Python callables or checkpoint bytes.

## Geometry, validation and limitations

The estimator uses fixed metric poses, registered bearing triangulation,
RANSAC, robust weighted TLS and a fixed-pose bearing reprojection refinement.
When two or more masks are accepted, the final fit uses observations only in
accepted views. Reserved tracks never contribute to the fit. Insufficient,
collinear, low-parallax or rejected support produces a needs-review result.

For `n·X+d=0`, unit camera bearing `b` and `v=R.T@b`, radial range is
`-(n·C+d)/(n·v)`. Native pixels are evaluated in blocks. Tangential, nonpositive,
out-of-bounds and excessively uncertain intersections remain invalid NaN.

The estimated standard deviation propagates triangulation noise through a
linearized plane covariance **with fixed poses**. It excludes calibration,
segmentation and systematic correspondence errors. It is not a calibrated
correctness probability. A low fitting residual is not independent accuracy
evidence: a coherently wrong correspondence set can fit the wrong plane.

NPZs preserve native support, validity, range, uncertainty, plane index,
conflict, composed range and per-pixel source (0 invalid, 1 baseline, 2 plane).
The composed result preserves baseline values outside published planar pixels.
Binary PLYs contain inferred world-space surface points and original colors;
triangulated measurements are identified separately in `result.json`.
The export manifest records checksums and execution provenance.

## Frozen experiments

`benchmarks/planar_reconstruction/run_evaluation.py` accepts externally reviewed
variant scenes and reference range NPZs. It freezes all predictions and their
hashes before opening any oracle. It reports per-plane/per-distance coverage,
MAE, RMSE, P95 and metric/relative hit rates; rejected predictions remain in
coverage and full-region hit-rate denominators. Independent masks enable
leakage and IoU measurements. Annotation time can be supplied as `human_effort`
in each frozen document. Missing measurements remain unavailable.

The real P74 comparison requires the human's accepted, frozen annotations.
Synthetic correctness tests and a working editor do not establish that the
method improves real-scene accuracy or generalizes.

Create a case JSON using absolute external paths, for example:

```json
{
  "options": {"automatic_matching": true, "feature_face_size": 1024},
  "variants": [
    {"name": "manual", "scene": "/external/manual/scene.json", "images_dir": "/external/images"},
    {"name": "simple", "scene": "/external/simple/scene.json", "images_dir": "/external/images"},
    {"name": "sam", "scene": "/external/sam/scene.json", "images_dir": "/external/images"}
  ],
  "oracles": {"a": "/external/reference/a.npz", "b": "/external/reference/b.npz"},
  "leave_one_view_out": true
}
```

Each variant must already contain accepted final masks, not unreviewed seeds.
Each reference NPZ contains native canonical `range` and boolean `validity`;
optional `plane_mask_<plane_id>` supplies an independently annotated mask.
An optional baseline is carried by the registered scene, with separate validity.

```bash
python benchmarks/planar_reconstruction/run_evaluation.py \
  --case /external/case.json --output /tmp/planar-comparison-new
```

The new output directory contains `freeze.json`, `metrics.json`, numerical
exports and comparison figures. Three or more views additionally enable
leave-one-view-out stability diagnostics. Memory is reported as the OS process
RSS high-water mark, explicitly cumulative across earlier variants, not as
isolated per-variant or GPU peak memory. Human effort is supplied by the operator;
it is not inferred from numerical runtime.
