Registration and panorama stitching
====================================

``PanoramaStitcher`` is the shortest route from two or more overlapping
planar images to one typed, partial ERP panorama.  It is an Experimental
application rather than a flagship API: use it when its central rotation-only
capture model matches the data, and inspect its diagnostics before accepting
the output.

1. Load real images and run the Application
--------------------------------------------

Pillow owns planar image file decoding.  PanorAi owns calibration,
registration, spherical warping and the output ``EquirectangularImage``::

   from pathlib import Path

   import numpy as np
   from PIL import Image

   from panorai.applications import PanoramaStitcher
   from panorai.stitching import StitchingOptions

   paths = [Path("left.jpg"), Path("centre.jpg"), Path("right.jpg")]
   images = tuple(np.asarray(Image.open(path).convert("RGB")) for path in paths)

   stitcher = PanoramaStitcher(
       StitchingOptions(
           feature_method="sift",
           output_shape_hw=(1024, 2048),
           exposure_compensation="gain",
           blend="feather",
       )
   )
   result = stitcher(images, image_ids=tuple(path.stem for path in paths))
   result.panorama.save("partial-panorama.png")

   if result.diagnostics.coverage < 0.5:
       print("The capture covers less than half of the sphere")
   print(result.diagnostics.rejected_pairs)

When every horizontal field of view is known, pass one value per image::

   result = stitcher(
       images,
       image_ids=("left", "centre", "right"),
       horizontal_fov_deg=(78.0, 78.0, 78.0),
   )

Omit ``horizontal_fov_deg`` only when the shared-FOV assumption is justified.
The estimator may reject weak or boundary solutions as unobservable.
``max_median_angular_error_deg`` additionally bounds the accepted
central-rotation residual, so a low-error planar homography cannot by itself
justify a panorama camera model.

2. Preserve explicit support and validity
------------------------------------------

Masks are independent from RGB values and use ``(H, W)`` boolean arrays::

   result = stitcher(
       images,
       support_masks=(support_left, support_centre, support_right),
       validity_masks=(valid_left, valid_centre, valid_right),
   )

``validity`` must never extend outside ``support``.  Zero-valued RGB is not a
missing-data marker.  The returned ERP exposes its support and validity while
``result.support_count`` and ``result.contribution_map`` explain which inputs
contributed to each output sample.

3. Inspect and reconfigure the Pipeline
----------------------------------------

The wrapper exposes the exact lower-level composition::

   pipeline = stitcher.as_pipeline()
   print(tuple(pipeline.named_steps))
   # DetectStitchingFeatures, DiscoverOverlaps, CalibrateAndRegister,
   # WarpRegisteredImages, CompensateExposure, BlendStitchingPanorama

Each public Stage implements ``execute(value) -> StageResult`` and may be used
alone or replaced in another Pipeline::

   from panorai.pipeline import Pipeline
   from panorai.stitching import (
       BlendStitchingPanorama,
       CalibrateAndRegister,
       CompensateExposure,
       DetectStitchingFeatures,
       DiscoverOverlaps,
       StitchingInput,
       WarpRegisteredImages,
   )

   options = stitcher.options
   custom = Pipeline(
       DetectStitchingFeatures(options),
       DiscoverOverlaps(options),
       CalibrateAndRegister(options),
       WarpRegisteredImages(options),
       CompensateExposure(options),
       BlendStitchingPanorama(options),
   )
   typed_input = StitchingInput.from_arrays(images)
   result = custom(typed_input)

Pipeline serialization uses the explicit PanorAi registry; it never imports a
class named by an input JSON document.

4. Use calibration independently
---------------------------------

Calibration consumes already matched planar pixel centres and returns
intrinsics, one target-from-source rotation, inliers and observability
diagnostics::

   from panorai.calibration import (
       RotationCalibrationOptions,
       calibrate_rotation_pair,
   )

   calibration = calibrate_rotation_pair(
       pixels_a_xy,
       pixels_b_xy,
       image_a.shape[:2],
       image_b.shape[:2],
       horizontal_fov_a_deg=None,
       horizontal_fov_b_deg=None,
       options=RotationCalibrationOptions(min_inliers=30),
   )
   if not calibration.observable:
       raise RuntimeError(calibration.degeneracy_reasons)

This function does not detect or match features.  Keeping those decisions out
of calibration makes the geometry reusable by other Applications.

5. Use robust model fitting independently
------------------------------------------

``panorai.model_fitting`` owns deterministic sampling, RANSAC/MSAC scoring and
local all-inlier refitting::

   import math

   from panorai.model_fitting import RobustFitOptions, fit_rotation

   fit = fit_rotation(
       source_bearings,
       target_bearings,
       options=RobustFitOptions(
           residual_threshold=math.radians(1.0),
           method="msac",
           min_inliers=30,
           random_seed=0,
       ),
   )
   rotation_target_from_source = fit.model

Custom model families implement ``fit`` and ``residuals`` through the public
``ModelFamily`` protocol.  Residual units must match ``residual_threshold``.

Failure boundary
----------------

Do not accept the result only because an image was produced.  The current
profile is not suitable for translated cameras, strong parallax, rolling
shutter, fisheye images without a prior rectification, unrelated image groups
or unobservable focal length.  It provides average/feather fusion, not seam
optimization or multi-band blending.  See the
:doc:`registration concept <../../concepts/registration-stitching>` and the
:doc:`public roadmap <../../development/computer_vision_research_roadmap>` for
the promotion gates.
