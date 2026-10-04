Run spherical visual SLAM
=========================

Incremental central-ERP tracking
--------------------------------

For already stitched central panoramas, use the incremental API. It returns a
status and, when tracking succeeds, a pose before the next frame is supplied::

   from panorai.slam import SphericalIncrementalSLAM

   session = SphericalIncrementalSLAM.from_preset(
       "sift-flann",
       face_sampler="cube",
       face_fov_deg=100.0,
       face_shape_hw=(320, 320),
       max_features=2048,
   )

   for frame_id, timestamp_s, panorama_rgb in decoded_erp_frames:
       tracking = session.add_frame(
           panorama_rgb,
           timestamp_s=timestamp_s,
           frame_id=frame_id,
       )
       if tracking.pose is not None:
           R_world_to_panorama = tracking.pose.R
           center_world = tracking.pose.center
       else:
           log_tracking_failure(frame_id, tracking.reasons)

   result = session.finish()

Do not infer a pose for ``state == "lost"``. Inspect
``map_correspondence_count``, ``median_parallax_deg``,
``local_map_point_count`` and ``reasons`` for every frame. The final result
contains immutable keyframe, map-point, observation, local-BA and loop
diagnostics. The trajectory scale is arbitrary.

``add_features()`` is the advanced route when the caller already owns a
``SphericalFeatureSet``. Inputs are copied; neither feature objects nor ERP
arrays are modified. ``reset()`` starts a new session with the same
configuration.

The installed documentation smoke also exercises the minimum lifecycle with
real extracted features: the first frame initializes a pose, while
``finish()`` explicitly refuses to call a one-frame session a successful
trajectory.

.. literalinclude:: ../../scripts/run_documentation_examples.py
   :language: python
   :start-after: DOCS_SLAM_START = None
   :end-before: DOCS_SLAM_END = None
   :dedent: 4

The default API targets correctness rather than real-time throughput. A
controlled development benchmark used three original-resolution Matterport360
ERPs, SIFT/FLANN, icosahedral 512-pixel faces, 4096 features and 500 relative
pose trials. On an Apple M3 Max with 64 GB, Python 3.12.4, NumPy 1.26.4,
SciPy 1.14.0 and OpenCV 4.11.0, five measured sessions after one warm-up had a
237.58 second median and 238.66 second nearest-rank P95, or 79.19 seconds per
registered frame at the median. Process peak RSS had a 760 MiB median.

Every measured session registered all three frames and passed the post-hoc
strict accuracy guard (median rotation error at most 0.562 degrees and median
translation-direction error at most 0.465 degrees). Inclusive stage timing
identified relative-pose estimation as the primary bottleneck: 192.99 seconds
at the median, about 81 percent of total SLAM time. Feature extraction used
14.96 seconds, final global correction 24.80 seconds, local BA 3.19 seconds and
descriptor matching 0.32 seconds. Inclusive stages can overlap and must not be
summed.

This is a development baseline for one three-frame set, not a performance
promise or a long-video throughput result. The source checkout was dirty and
is identified by its source fingerprint in the PERF-003 evidence. The five
samples are too few for a mature tail estimate, so the reported P95 is the
worst measured warm sample. Reduce trials, face size or feature count only
after re-running accuracy and tracking acceptance on the intended capture
domain.

Calibrated fisheye graph session
--------------------------------

Install the optional ROS bag reader only when a dataset adapter needs it::

   pip install "panorai[slam]"

The core API consumes decoded arrays and timestamps, so it is independent of
ROS and of any particular dataset::

   from panorai.slam import EquidistantFisheyeCamera, SphericalVisualSLAM

   camera = EquidistantFisheyeCamera.from_kalibr_yaml(
       calibration_path,
       camera_id="cam0",
   )
   session = SphericalVisualSLAM(camera)

   for frame_id, timestamp_s, bgr_image in decoded_frames:
       session.add_frame(
           bgr_image,
           timestamp_s=timestamp_s,
           frame_id=frame_id,
       )

   result = session.finish()

Inspect ``result.diagnostics`` even after success. It records feature and pair
counts; the nested reconstruction diagnostics record edge admission,
translation consistency, active tracks, reprojection errors, and refinement
costs. A returned trajectory has arbitrary scale.

For the Hilti-Trimble-Oxford 2026 dataset, use ``cam0`` for the first central
baseline because the ground truth is expressed for that camera and the two
fisheye lenses have different optical centres. Reference poses must remain
closed until the estimated trajectory is frozen.

Preliminary fisheye capture envelope
------------------------------------

The first metadata-blind replay on
``floor_2_2025-10-28_run_2`` compared two temporal samplings. At 1 Hz, the
reference motion later measured a 1.62 m median and 38.2 degree median step
(maxima 2.36 m and 143.7 degrees); only one of 13 numerical edges passed the
default admission policy and no trajectory was returned. At 5 Hz, a 16-frame
segment measured 0.37 m and 13.6 degrees median motion per step (maxima 0.57 m
and 28.9 degrees). Six accepted consecutive edges yielded seven registered
poses, with Sim(3)-aligned ATE RMSE 0.249 m over that short component.

This is a single-sequence development result, not a general benchmark. Until
broader validation exists, capture at **at least 5 Hz** for motion of this
magnitude and target adjacent keyframes below roughly 0.4 m / 15 degrees in
typical motion, while avoiding steps above about 0.6 m / 30 degrees. Higher
rates are appropriate for faster motion, rolling-shutter exposure, weak
texture, or narrow usable support. Always retain the conservative accepted
edge policy and inspect actual overlap, inliers, angular coverage, parallax,
translation orientation, and active tracks; frequency alone cannot guarantee
success.
