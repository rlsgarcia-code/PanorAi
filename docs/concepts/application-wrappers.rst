Application wrappers
====================

An Application is a task-level contract. It answers “what result do I want?”
before asking the user to select stages, grids or fusion policies. Its value is
not hidden magic: a good wrapper makes one supported method easy to configure,
inspect and reproduce.

.. panorai-api-components:: application

Responsibilities
----------------

An Application owns input validation, immutable parameters, the call signature
and the public result type. ``run`` and ``__call__`` execute the method;
``get_params`` and ``with_params`` support configuration tools; ``describe``
exposes the method boundary. A ``PipelineApplication`` additionally exposes
``as_pipeline`` so advanced users can see or adapt the exact composition.

The wrapper delegates mathematics to existing domains. ``SphericalSLIC`` uses
the SLIC spherical operation; it does not maintain a second implementation.
Likewise, feature, pose, stereo and instance wrappers adapt their established
algorithms and result types. ``SphericalRelativePose.reference()`` names the
complete evidence-backed frontend, sampler, MSAC/refit estimator and acceptance
configuration. It exposes four named computational stages: feature extraction,
matching, bearing conversion and robust pose estimation. A custom frontend or
pose strategy must be selected explicitly.

Capture eligibility is not hidden inside estimator options.
``RelativePoseCapturePolicy`` consumes independent physical metadata before
feature extraction, while ``RelativePoseAcceptancePolicy`` judges evidence
after estimation. A configured capture policy adds a fifth, leading
``AssessRelativePoseCapture`` Stage. Keeping these decisions separate prevents
registered-cloud overlap or a dataset-specific baseline range from being
misrepresented as an image-derived confidence score.

Operations, stages and layers are different roles
-------------------------------------------------

An operation transforms a spherical raster, a Stage transforms a typed value,
and a Torch layer participates in a differentiable tensor network. Applications
may use any of those pieces, but do not pretend they are interchangeable.
Relative pose is a useful boundary case: it projects ERPs for feature
extraction, then operates on sparse bearings. Its public composition therefore
uses feature and estimator Stages, not ``OnSphere`` raster operations or a
``torch.nn.Module``. The returned translation is direction-only because metric
scale is unobservable from two calibrated central views.

Configuration is not execution state
------------------------------------

Application objects are immutable descriptions. Safe persistence is available
only for registered components and canonical JSON-compatible parameters.
Checkpoints, datasets, lambdas and arbitrary Python imports are not hidden in a
configuration file.

Stability belongs to the concrete application
---------------------------------------------

The common shape of wrappers is ergonomic, not a blanket stability promise.
``SphericalSLIC`` is Stable; DoG, Stereo, Relative Pose and Mask R-CNN Workflow
wrappers remain Experimental while their individual evidence gates are open.
A Stable operation inside an Experimental application does not promote the
complete method.

See :doc:`the Applications tutorial <../guides/api-levels/applications>` and
:doc:`Applications API <../reference/applications>`.
