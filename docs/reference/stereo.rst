Spherical dense-stereo API
==========================

``panorai.stereo`` is an Experimental direct matching surface for two
same-shape central ERPs with known relative pose. It returns radial range in
the translation unit together with explicit validity, cost, confidence, and
the inverse-range hypothesis lattice.

Start with :doc:`../tutorials/06_spherical_dense_stereo`. The full geometry,
objective, four-path aggregation, refinement, consistency policy, complexity,
and public validation limitations are derived in
:doc:`../explanation/spherical_dense_stereo`.

The pose convention is

.. math::

   X_B = R_{BA} X_A + t_{BA}.

Supplying only a unit translation direction produces range in baseline units,
not metres. Confidence is a best-versus-second-best separation on the current
cost volume and is not a calibrated probability.

The additive Experimental dense-guided match filter validates existing
``SphericalFeatureMatches`` by sampling radial range at each source bearing and
measuring the angular error of its 3D reprojection in the target panorama. It
returns separate accepted, rejected, and unsupported masks. It never mutates
the matches or refines the supplied pose; unsupported dense pixels are not
treated as rejected matches.

``refine_matches_on_sphere`` is the next optional Experimental stage. It uses
the fixed pose and dense center range to search a seam-safe angular subpixel
grid with transported local patches. Original bearings always remain in the
result; refined bearings are applied only after explicit support, texture,
photometric-improvement and maximum-motion gates. The operation does not
update ``R,t`` or overwrite the input match object.

API
---

.. automodule:: panorai.stereo
   :members:
   :member-order: bysource
   :undoc-members:
