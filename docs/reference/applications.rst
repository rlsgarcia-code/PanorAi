Applications API
================

Application stability is assigned per concrete wrapper. ``SphericalSLIC`` is
Stable as ``panorai-spherical-slic/v1``. The common base and DoG, Stereo and
Mask R-CNN Paper wrappers remain Experimental.

Stable application
------------------

.. autoclass:: panorai.applications.SphericalSLIC
   :members: run, get_params, with_params, describe, as_pipeline, to_config, save, load

Common Experimental contracts
-----------------------------

.. autoclass:: panorai.applications.Application
   :members: run, get_params, with_params, describe, to_config, save, load

.. autoclass:: panorai.applications.PipelineApplication
   :members: as_pipeline, pipeline

Experimental wrappers
---------------------

.. autoclass:: panorai.applications.SphericalDoGFeatures
   :members: run, describe, as_pipeline

.. autoclass:: panorai.applications.SphericalStereoDepth
   :members: run, describe, as_pipeline

.. autoclass:: panorai.applications.SphericalMaskRCNNPaper
   :members: run, describe, as_pipeline

Start with :doc:`the Applications tutorial
<../guides/api-levels/applications>` and read :doc:`Application wrappers
<../concepts/application-wrappers>` for the contract rationale.
