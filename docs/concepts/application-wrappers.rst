Application wrappers
====================

An Application is a task-level contract. It answers “what result do I want?”
before asking the user to select stages, grids or fusion policies. Its value is
not hidden magic: a good wrapper makes one supported method easy to configure,
inspect and reproduce.

Responsibilities
----------------

An Application owns input validation, immutable parameters, the call signature
and the public result type. ``run`` and ``__call__`` execute the method;
``get_params`` and ``with_params`` support configuration tools; ``describe``
exposes the method boundary. A ``PipelineApplication`` additionally exposes
``as_pipeline`` so advanced users can see or adapt the exact composition.

The wrapper delegates mathematics to existing domains. ``SphericalSLIC`` uses
the SLIC spherical operation; it does not maintain a second implementation.
Likewise, feature, stereo and instance wrappers adapt their established
algorithms and result types.

Configuration is not execution state
------------------------------------

Application objects are immutable descriptions. Safe persistence is available
only for registered components and canonical JSON-compatible parameters.
Checkpoints, datasets, lambdas and arbitrary Python imports are not hidden in a
configuration file.

Stability belongs to the concrete application
---------------------------------------------

The common shape of wrappers is ergonomic, not a blanket stability promise.
``SphericalSLIC`` is Stable; DoG, Stereo and Mask R-CNN Paper wrappers remain
Experimental while their individual evidence gates are open. A Stable
operation inside an Experimental application does not promote the complete
method.

See :doc:`the Applications tutorial <../guides/api-levels/applications>` and
:doc:`Applications API <../reference/applications>`.
