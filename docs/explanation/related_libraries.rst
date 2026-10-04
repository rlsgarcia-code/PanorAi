Related libraries and PanorAi's role
====================================

PanorAi does not claim that spherical image conversion is unique. Mature
libraries already provide excellent transformation kernels:

`py360convert <https://github.com/sunset1995/py360convert>`_
   A focused NumPy library for equirectangular, cubemap, and perspective
   conversion. Its public ``e2p`` API supports horizontal/vertical FOV,
   in-plane rotation, output shape, and nearest or bilinear sampling.

`pyequilib <https://github.com/haruishi43/equilib>`_
   A modular NumPy/Torch library for equirectangular, cubemap, perspective, and
   rotation transforms, with functional and class APIs, batching, and CPU/CUDA
   execution.

`OpenCV omnidirectional calibration <https://docs.opencv.org/4.x/dd/d12/tutorial_omnidir_calib_main.html>`_
   Camera calibration, rectification, and stereo reconstruction for
   omnidirectional camera models. Its problem is camera calibration rather
   than a typed ERP/cubemap processing contract.

Why PanorAi exists
------------------

PanorAi's intended contribution is an explicit, testable contract around the
conversion kernels:

- pixel-center coordinates and one named Cartesian frame;
- radial range rather than ambiguous ``depth``;
- geometric support separated from data validity and numeric value;
- nearest-only labels and masks, with invalid interpolation rejected;
- matching NumPy/Torch behavior and input-value autograd;
- mask-aware reconstruction and compatibility adapters for existing 3.0 code;
- installable-artifact, license, documentation, and conformance evidence.

Choose py360convert when a compact NumPy converter matches the task. Choose
pyequilib when its broad NumPy/Torch transform and acceleration surface is the
best fit. Choose OpenCV when calibration is the actual problem. Choose PanorAi
when downstream correctness depends on the stated coordinate, modality,
validity, compatibility, and release-evidence contract.

No performance superiority is implied by functional conformance. The
reproducible comparison is run with ``scripts/benchmark_geometry.py`` against
an installed wheel outside the checkout. It pins ``py360convert==1.0.4`` and
``pyequilib==0.6.0`` and records first call separately from reused calls,
median, nearest-rank P95, peak RSS, hardware, versions, and raw samples.

Results whose frame, pixel lattice, rotation, or cubemap layout differs are
labelled timing-only. In particular, equal output dimensions are not treated
as numerical equivalence, and missing public operations are reported as
unsupported instead of being approximated by a different transform.
