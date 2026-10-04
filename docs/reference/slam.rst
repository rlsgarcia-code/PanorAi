Spherical visual SLAM
=====================

``panorai.slam`` is an Experimental, visual-only, arbitrary-scale SLAM
surface. It provides two deliberately separate workflows:

* ``SphericalIncrementalSLAM`` consumes central equirectangular images and
  returns one tracking result immediately per frame. It owns automatic
  keyframes, a local landmark map, spherical 2D--3D pose refinement, local
  bundle adjustment, relocalization, loop discovery, and global correction.
* ``SphericalVisualSLAM`` consumes one calibrated central equidistant fisheye
  lens. It constructs the temporal graph incrementally but performs global
  reconstruction only when ``finish()`` freezes that graph.

Both workflows use PanorAi spherical features, relative pose, and global
reconstruction. The implementation is NumPy/SciPy/OpenCV based; ``rosbags``
is needed only by dataset adapters.

Incremental central-ERP workflow
--------------------------------

::

   from panorai.slam import SphericalIncrementalSLAM

   slam = SphericalIncrementalSLAM.from_preset("sift-flann")
   for frame_id, timestamp_s, panorama_rgb in frames:
       tracking = slam.add_frame(
           panorama_rgb,
           timestamp_s=timestamp_s,
           frame_id=frame_id,
       )
       if tracking.pose is None:
           print(frame_id, tracking.reasons)
       else:
           use_online_pose(tracking.pose)

   result = slam.finish()

The input must be a central ERP panorama. Each accepted relative pose is
quality-gated. Landmark associations use only descriptor matches that are
also geometric inliers. ``tracking.state`` is one of ``initializing``,
``tracking``, ``keyframe``, ``relocalized``, or ``lost``. A lost frame has no
pose; PanorAi does not fill trajectory gaps with fabricated estimates.

The local map triangulates unit bearings and maintains at most one active
observation per frame and landmark. Local BA minimizes spherical log-map
residuals, fixes the two oldest local poses to retain the monocular gauge, and
removes observations above the configured angular limit. Accepted old/new
keyframe links trigger loop correction through the global spherical mapper.

Calibrated-fisheye graph workflow
---------------------------------

The first supported physical camera is one calibrated central equidistant
fisheye lens::

   from panorai.slam import EquidistantFisheyeCamera, SphericalVisualSLAM

   camera = EquidistantFisheyeCamera.from_kalibr_yaml(
       "kalibr_imucam_chain.yaml",
       camera_id="cam0",
   )
   slam = SphericalVisualSLAM(camera)

   for frame_id, timestamp_s, image in frames:
       slam.add_frame(image, timestamp_s=timestamp_s, frame_id=frame_id)

   result = slam.finish()
   if result.success:
       centers = result.centers_world
   else:
       print(result.failure_reasons)

In this workflow every supplied frame is treated as a keyframe.
``temporal_window`` controls
which recent keyframes are matched. The front end is incremental, while
rotation averaging, track construction, positioning, triangulation, and
spherical bundle adjustment run when ``finish()`` freezes the graph.

Coordinate and scale contract
-----------------------------

Input pixels use ``(x, y)`` with a top-left origin. Kalibr/OpenCV equidistant
intrinsics are inverted before estimation, and returned unit bearings use the
PanorAi ``+X`` right, ``+Y`` up, ``+Z`` forward convention. Camera poses obey

.. math::

   x_i = R_i(X-C_i).

Monocular translation scale is unobservable. ``result.scale`` is therefore
``"arbitrary"``; metric trajectories must be aligned with a similarity
transform for evaluation.

Dual-fisheye limit
------------------

A dual-fisheye camera normally has two distinct optical centres. This version
does **not** concatenate both lenses and pretend that they form one central
camera. Use one calibrated lens, or supply a pre-stitched ERP only when its
central-camera approximation is justified. A future generalized-camera back
end is required for exact joint use of both lenses.

Current limits shared by both workflows include rolling-shutter correction,
IMU fusion, metric scale, dynamic-scene modeling, and bounded-memory
marginalization. The incremental ERP workflow has automatic keyframes,
relocalization and loop discovery, but its current local BA window drops old
optimization variables rather than marginalizing them into a prior. Failures
are explicit. A final incremental result may contain a valid partial map while
``complete_trajectory`` remains false in evaluation tooling.

API
---

.. automodule:: panorai.slam
   :members:
   :member-order: bysource
   :undoc-members:
