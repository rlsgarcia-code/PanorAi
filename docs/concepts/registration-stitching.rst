Registration and stitching
===========================

Panorama stitching is not a planar ``paste`` operation.  It is an inference
problem followed by a resampling and fusion problem.  PanorAi therefore keeps
five responsibilities separate:

* :mod:`panorai.calibration` defines the pinhole pixel/ray convention and
  estimates observable intrinsics;
* :mod:`panorai.model_fitting` robustly fits geometric models to aligned
  observations;
* ``panorai.stitching`` exposes typed evidence and composable Pipeline
  Stages;
* :class:`panorai.applications.PanoramaStitcher` packages one complete,
  deliberately bounded workflow; and
* ``panorai.evaluation`` measures registration and panorama results without
  changing the estimator.

The first public profile is Experimental and assumes a **central,
rotation-only capture**: all images are pinhole views whose optical centres
coincide and whose orientations differ.  The first image fixes the otherwise
arbitrary world gauge.  Camera rays use ``+X`` right, ``+Y`` up and ``+Z``
forward, and pixel coordinates refer to pixel centres.

No metadata does not mean no assumptions
-----------------------------------------

The application accepts images without EXIF, orientation or field-of-view
metadata.  Missing horizontal field of view is estimated from pairwise
correspondences only when the evidence makes the parameter observable.  If
both images of a pair have unknown field of view, the current profile assumes
a shared value.  If one is known, that value is retained while the other is
estimated.

The implementation does **not** currently estimate principal point, skew,
radial distortion or tangential distortion.  Translation, strong parallax,
rolling shutter, independently moving content, insufficient texture and a
disconnected overlap graph are outside the model.  These conditions must
produce diagnostics or :class:`panorai.stitching.StitchingRegistrationError`,
not invented camera geometry.

The pair gate checks both an absolute inlier threshold and a bounded median
central-rotation angular residual.  This rejects a planar translation that a
homography alone could explain.  It is still a conservative incompatibility
gate, not proof that every accepted real capture had exactly zero translation.

Support, validity and values remain distinct
--------------------------------------------

Every source has a geometric support mask and a data-validity mask.  Warping
transforms both masks independently.  A black pixel can be valid; an invalid
sample cannot become valid because its numeric value is zero.  The result is a
partial :class:`panorai.EquirectangularImage`: unsupported directions remain
unsupported rather than being filled to make a visually complete rectangle.

The result also retains per-image rotations and intrinsics, accepted pairwise
registrations, gains, a contribution map, support counts, rejected pairs and
optimization diagnostics.  This evidence is part of the result contract, not
debug logging.

Current fusion boundary
-----------------------

The Experimental profile provides optional global scalar gain compensation
and ``average`` or ``feather`` blending.  It does not yet provide a seam
finder, multi-band blending, local parallax warps or joint camera/lens bundle
adjustment.  A visually plausible output is therefore not sufficient evidence
for promotion.  Real-image, shuffled-order, missing-metadata, parallax and
failure-envelope evaluations remain required before any Stable claim.

Use :doc:`the progressive guide <../guides/registration_stitching/index>` for
the Application and equivalent Pipeline, and the
:doc:`component catalog <../reference/catalog/index>` to inspect each reusable
piece.
