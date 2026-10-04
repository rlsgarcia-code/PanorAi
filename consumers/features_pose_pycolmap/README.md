# Features, pose, and PyCOLMAP consumer

This standalone consumer starts from two deterministic equirectangular images,
runs real OpenCV SIFT extraction and matching through PanorAi's public
spherical-feature façade, converts matches to panorama-frame unit bearings,
evaluates them with PanorAi's relative-pose estimator, and exports the visual
evidence to a real PyCOLMAP database.

Extraction, matching, result objects, and bearing conversion exercise the
Stable ``panorai-spherical-features/v1`` core. Relative pose and the
virtual-camera rig/PyCOLMAP exporter remain Experimental extensions and are
checked here without being promoted with the core.

The second panorama is a horizontal cyclic shift of the first. That is a
rotation-only scene relation, so the consumer requires the quality policy to
reject a trustworthy translation claim even when an Essential pose object can
be estimated. This checks public failure semantics instead of treating every
returned pose as accepted.

PyCOLMAP read-back independently verifies cameras, rigs, frames, images,
keypoints, descriptors, matches, the COLMAP half-pixel convention, and
reconstructed panorama-frame rays. The optional PyCOLMAP installation remains
outside PanorAi's wheel.
