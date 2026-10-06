Spherical dense-stereo API
==========================

``panorai.stereo`` is an Experimental direct matching surface for two
same-shape central ERPs with known relative pose. It returns radial range in
the translation unit together with explicit validity, cost, confidence, and
the inverse-range hypothesis lattice.

Start with :doc:`../tutorials/06_spherical_dense_stereo`. The full geometry,
objective, four-path aggregation, refinement, consistency policy, complexity,
and P-74 limitation evidence are derived in
:doc:`../explanation/spherical_dense_stereo`.

The pose convention is

.. math::

   X_B = R_{BA} X_A + t_{BA}.

Supplying only a unit translation direction produces range in baseline units,
not metres. Confidence is a best-versus-second-best separation on the current
cost volume and is not a calibrated probability.

API
---

.. automodule:: panorai.stereo
   :members:
   :member-order: bysource
   :undoc-members:
