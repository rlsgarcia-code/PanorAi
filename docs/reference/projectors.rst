Projectors
==========

Canonical projectors
--------------------

``GnomonicProjector`` and ``CubemapProjector`` are immutable reusable objects
from ``panorai.geometry``. They retain only their spec, interpolation mode, and
fill value; all computation delegates to the canonical functional engine.

Use ``GnomonicProjector`` for a rectilinear tangent view. Use
``CubemapProjector`` for the fixed face order ``front``, ``right``, ``back``,
``left``, ``up``, ``down``. Functional and object-oriented calls have the same
coordinate contract.

Legacy registry
---------------

The name ``gnomonic`` remains available through the 3.0 compatibility
registry. It does not define new canonical geometry. Prefer
``panorai.geometry.GnomonicSpec`` in new code.
