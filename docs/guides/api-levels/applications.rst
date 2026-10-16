Applications: solve the task first
==================================

Choose an Application when the method is already known and only its parameters
or inputs change. A wrapper validates configuration, invokes existing PanorAi
algorithms and returns a public typed result; it does not duplicate the
algorithm.

1. Run the smallest Stable application
---------------------------------------

The following example is self-contained and executed in CI. It uses intrinsic
spherical SLIC, so no projected faces are created.

.. literalinclude:: ../../../scripts/run_documentation_examples.py
   :language: python
   :start-after: DOCS_LEVEL_APPLICATION_START = None
   :end-before: DOCS_LEVEL_APPLICATION_END = None
   :dedent: 4

2. Reconfigure immutably
------------------------

Application instances are configurations. ``with_params`` returns a new
instance and leaves the original untouched:

.. literalinclude:: ../../../scripts/run_documentation_examples.py
   :language: python
   :start-after: DOCS_LEVEL_APPLICATION_CONFIG_START = None
   :end-before: DOCS_LEVEL_APPLICATION_CONFIG_END = None
   :dedent: 4

Use ``get_params()`` for tuning tools and ``describe()`` for an auditable view
of input, output, execution domain and lower-level composition.

3. Persist only registered configurations
------------------------------------------

``SphericalSLIC.save("slic.json")`` writes canonical JSON through an explicit
registry. Loading never imports a class named by the file, and arbitrary
callables are not serialized. This makes the saved configuration portable
without using pickle.

4. Descend when the method changes
----------------------------------

``application.as_pipeline()`` exposes the equivalent composition. Descend to
:doc:`Pipeline <pipeline>` when you need a different stage, branch, batch or
fusion policy. Descend to :doc:`Spherical Ops <spherical_ops>` when the
operation itself changes.

Current application boundary
----------------------------

``SphericalSLIC`` is Stable. ``SphericalDoGFeatures``,
``SphericalStereoDepth`` and ``SphericalMaskRCNNPaper`` are executable
Experimental wrappers with separate scientific and dependency gates. A common
wrapper shape does not imply common stability.

Understand the design in :doc:`Application wrappers
<../../concepts/application-wrappers>` and look up signatures in
:doc:`Applications API <../../reference/applications>`.
