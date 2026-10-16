Adding a composed application
=============================

This page is the contributor contract for adding a user-facing application to
PanorAi.  It defines which public object to add, how it composes with
``panorai.pipeline``, when an operation or Torch layer belongs in
``panorai.ops.spherical``, and which semantics and evidence must accompany the
change.

The short rule is: **one algorithm, several deliberate adapters**.  A wrapper
must reuse the implementation owned by its domain package; it must not copy an
algorithm into ``panorai.applications``.  Projection geometry continues to be
owned by ``panorai.geometry`` and the Stable object workflow.

Stability is attached to each public symbol, not to its namespace. The generic
Pipeline, general NumPy spherical operations, and the SLIC vertical are Stable.
The generic application framework, DoG, Stereo, Mask R-CNN Paper, and CAM/SAM
stages remain Experimental until their individual promotion gates pass. The
optional generic Torch layer grammar is Stable on CPU; other device backends
retain separate evidence requirements.

Choose the public artifact
--------------------------

An application may expose more than one artifact, but each has a different
responsibility.

.. list-table:: Extension decision table
   :header-rows: 1
   :widths: 18 30 25 27

   * - Artifact
     - Add it when
     - Required public shape
     - Composes through
   * - Domain algorithm
     - The mathematics already has an owning package such as
       ``image_processing``, ``features`` or ``stereo``.
     - Existing domain API and result type.
     - Called by adapters; never copied.
   * - Functional operation
     - Stateless NumPy users need the primitive directly.
     - ``operation(values, *, parameters) -> value``.
     - ``panorai.ops.spherical.functional``.
   * - Configured operation
     - The primitive needs immutable parameters, grid/mask semantics, or use
       inside ``OnSphere``.
     - Callable frozen object with ``describe()``, ``transform_grid()`` and
       mask propagation.
     - ``OnSphere`` and safe pipeline persistence when registered.
   * - ``Stage``
     - A step changes execution domain, consumes a typed request, or returns a
       structured/non-raster result.
     - ``execute(value) -> StageResult`` plus ``input_kind`` and
       ``output_kind``.
     - ``Pipeline``; no PanorAi base class is required.
   * - ``Pipeline``
     - Existing stages express the complete low-level route.
     - ``Pipeline(*stages)``; the pipeline is itself a ``Stage``.
     - Nesting, branching, provenance, persistence and cache.
   * - ``Application``
     - Users need a task-level call with domain language and a typed result.
     - Immutable wrapper implementing ``run``, ``describe`` and
       ``to_config``.
     - Direct high-level use.
   * - ``PipelineApplication``
     - The task-level call has an equivalent one-input composition.
     - ``Application`` plus ``as_pipeline() -> Pipeline``.
     - High-level use and inspectable low-level composition.
   * - Torch layer/module
     - The operation is differentiable, learned, or belongs in
       ``nn.Sequential``.
     - ``forward(values)`` using validated spherical or pointwise layers.
     - ``OnSphere``; optional registered safetensors persistence.

Do not make one object change category according to a parameter.  For example,
an application must not sometimes return a Pipeline and sometimes execute it.
The application always returns the documented result; ``as_pipeline()``
always returns the inspectable composition.

An application is not automatically a Torch layer.  When both are useful,
publish the layer or layer factory separately under
``panorai.ops.spherical.nn`` and let the application compose it.  Only
``as_pipeline()`` is standardized today.  A public ``build_module()`` or
``layer`` convenience may be proposed, but it is application-specific until a
separate contract is accepted; it must return a module that passes
``panorai.ops.spherical.nn.validate`` and it must not bypass grid or mask
propagation.

Application contract
--------------------

Use a frozen dataclass for built-in applications.  This makes
``get_params()`` and immutable ``with_params()`` useful without a second
configuration system.

The general contract is:

.. code-block:: python

   from typing import Any

   class Application:
       application_id: str

       def run(self, *args: Any, **kwargs: Any) -> Any: ...
       def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
       def get_params(self, *, deep: bool = True) -> dict[str, Any]: ...
       def with_params(self, **params: Any) -> "Application": ...
       def describe(self) -> dict[str, Any]: ...
       def to_config(self) -> dict[str, Any]: ...
       def save(self, path) -> path: ...

   class PipelineApplication(Application):
       def as_pipeline(self) -> Pipeline: ...

``run`` is intentionally variadic because task-level inputs differ.  New
applications should select one of these signatures and document it exactly:

``__call__`` delegates to ``run``.  ``with_params(name=value)`` returns a new
instance of the same concrete frozen dataclass; it never mutates the original.
Nested dataclass configuration uses ``parent__child=value`` and unknown paths
fail explicitly.  Do not add a second mutable ``set_params`` surface.

Unary ERP
~~~~~~~~~

Use for one-panorama processing.  An array may accept an explicit validity
mask; an ``EquirectangularImage`` already owns its support and validity.

.. code-block:: python

   def run(
       self,
       panorama: ArrayOrEquirectangularImage,
       *,
       validity_mask: BooleanHW | None = None,
   ) -> PublicResult: ...

Unary ERP with external support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use only when the source itself can be geometrically unsupported, as in an
adapter that receives a cropped or sensor-limited raster.

.. code-block:: python

   def run(
       self,
       panorama: ArrayOrEquirectangularImage,
       *,
       support_mask: BooleanHW | None = None,
       validity_mask: BooleanHW | None = None,
   ) -> PublicResult: ...

Never accept masks alongside a container and silently override the container.
Fail explicitly, as the existing wrappers do.

Calibrated multi-input
~~~~~~~~~~~~~~~~~~~~~~

Use for stereo or another task whose public inputs are naturally separate.
State frame and units in the signature documentation.

.. code-block:: python

   def run(
       self,
       reference: Array,
       source: Array,
       rotation_source_from_reference: Array,
       translation_source_from_reference: Array,
   ) -> PublicResult: ...

A Pipeline consumes one value.  Bridge a multi-argument wrapper to the
pipeline with an immutable typed request:

.. code-block:: python

   @dataclass(frozen=True, slots=True)
   class StereoInput:
       reference: Array
       source: Array
       rotation_source_from_reference: Array
       translation_source_from_reference: Array

   class StereoStage:
       input_kind = "stereo_pair"
       output_kind = "result"

       def execute(self, value: StereoInput) -> StageResult:
           if not isinstance(value, StereoInput):
               raise TypeError("StereoStage requires StereoInput")
           result = estimator(value)
           return StageResult(result, {"domain": "sphere", "quantity": "radial_range"})

The high-level ``run`` creates ``StereoInput`` and calls ``as_pipeline()``.
This preserves an ergonomic user signature without adding variadic semantics
to Pipeline.

Stateful or streaming application
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Do not force a mutable session into ``PipelineApplication``.  Keep immutable
configuration in an application/factory and return a separately named session
object with explicit lifecycle methods.  State, reset behavior, thread safety,
serialization, and bounded-memory behavior require their own contract.

Wrapper skeleton
~~~~~~~~~~~~~~~~

.. code-block:: python

   from dataclasses import dataclass
   from typing import Any

   from panorai.applications import PipelineApplication
   from panorai.applications._base import application_config  # built-ins only
   from panorai.pipeline import OnSphere, Output, Pipeline, RasterField

   @dataclass(frozen=True, slots=True)
   class SphericalExample(PipelineApplication):
       strength: float = 1.0

       application_id = "panorai.applications.SphericalExample"

       def _operation(self):
           # Return the domain-owned configured operation. Do not reimplement it.
           return ExampleOperation(strength=self.strength)

       def as_pipeline(self) -> Pipeline:
           return Pipeline(
               OnSphere(
                   Output(self._operation(), kind="confidence"),
                   output="confidence",
               )
           )

       def run(self, panorama: Any) -> RasterField:
           result = self.as_pipeline()(panorama)
           if not isinstance(result, RasterField):
               raise RuntimeError("unexpected SphericalExample result")
           return result

       def describe(self) -> dict[str, Any]:
           return {
               "interface": self.interface,
               "stability": self.stability,
               "application": self.application_id,
               "input": "canonical ERP with explicit support/validity",
               "output": "RasterField(kind='confidence')",
               "domain": "direct-spherical",
               "pipeline": self.as_pipeline().describe(),
           }

       def to_config(self) -> dict[str, Any]:
           return application_config(self.application_id, self.get_params(deep=False))

The example omits ``from_config`` and registry changes only for brevity.  A
serializable built-in application must implement them as described below.

Stage contract
--------------

Use a Stage when the work is not just a raster-to-raster spherical operation.
The complete public signature is:

.. code-block:: python

   class ExampleStage:
       input_kind = "erp"
       output_kind = "result"

       def execute(self, value: Any) -> StageResult:
           result = domain_algorithm(value)
           return StageResult(
               result,
               {
                   "domain": "sphere",
                   "algorithm": "example",
                   "result_count": len(result),
               },
           )

``input_kind`` and ``output_kind`` may be a string or tuple of strings.  Use
the established kinds (``erp``, ``views``, ``view_results``, ``bundle``,
``result`` and ``any``) when they fit.  Introduce a new kind only with tests
for every legal and illegal transition.  ``execute`` must return
``StageResult``; returning the raw value is an error.  Details must be an
immutable-friendly, JSON-explainable mapping.  Supply ``grid`` explicitly for
spatial values that do not carry a public grid themselves.

A Pipeline is already a Stage.  Prefer nesting ``Pipeline`` over a private
executor or a second orchestration loop.  Use ``Parallel`` for logical
branches, ``Select`` for a dotted result path, and ``Attach`` only when raster
grids are compatible.  ``Parallel`` and ``PerView.parallel`` describe
topology; they do not promise threads.

Sphere-native NumPy operation contract
--------------------------------------

Put the stateless primitive in ``panorai.ops.spherical.functional`` when it is
useful by itself.  Put the immutable configured adapter in
``panorai.ops.spherical`` when it must run through ``OnSphere``.  The adapter
delegates to the primitive or an existing domain implementation.

For a spatial raster result, the configured object provides:

.. code-block:: python

   @dataclass(frozen=True, slots=True)
   class ExampleOperation:
       parameter: float

       spherical_operation = True
       preserves_shadow_state = True

       def __call__(self, values: np.ndarray) -> np.ndarray: ...

       def transform_grid(self, input_grid: RasterGrid) -> RasterGrid: ...

       def transform_mask(self, mask: BooleanHW) -> BooleanHW: ...

       def describe(self) -> dict[str, Any]: ...

Shape-preserving operations still append an auditable grid transform.  A
shape-changing operation must compute the exact first sampled centre, stride,
resolution, periodicity, and transform history.  It may return to canonical
``erp-pixel-centres`` only after explicit regridding at those rays.

If an operation supports validity renormalization, implement:

.. code-block:: python

   def apply_with_masks(
       self,
       values,
       support,
       validity,
       *,
       invalid_policy: str,
       min_valid_weight: float | None,
   ) -> tuple[value, support, validity]: ...

Renormalization is valid only when its mathematics is defined.  General
learned convolution must use strict propagation.  Labels and masks are
nearest-only; integer or boolean bilinear interpolation must fail.

Projected-view callable contract
--------------------------------

Use ``SampleViews`` plus ``PerView`` for an ordinary pinhole/model callable.
The callable receives one raw face array/tensor and may return:

* an array/tensor;
* ``(array_or_tensor, boolean_validity_mask)``; or
* ``panorai.data.RasterFieldResult(data, validity, support)``.

Declare output semantics with ``Output``.  Raster kinds are ``image``,
``depth``, ``labels``, ``mask``, ``logits``, ``features`` and ``confidence``.
Non-raster kinds are ``classification``, ``embedding``, ``detections`` and
``keypoints``.  Depth always means radial range and requires units.  A
non-raster per-view result retains face identity, projection, support and
validity.

For ``batch="auto"``, a callable must either be a Torch module or declare
``supports_batch = True``.  Batched output must preserve a leading batch
dimension or return one item per input.  There is no silent retry after OOM.

Torch layer contract
--------------------

Torch is optional and must remain lazy.  New direct-spherical layers belong in
``panorai.ops.spherical.nn`` and use NCHW/CHW conventions documented by the
existing layer family.  A model intended for ``OnSphere`` must satisfy:

.. code-block:: python

   def forward(self, values): ...

and must be composed from validated spherical spatial layers and permitted
pointwise layers.  In the Stable contract, custom container ``forward``
methods fail closed because hidden functional spatial calls cannot be proven
safe.  Use ``torch.nn.Sequential`` or propose explicit
``transform_grid``, ``transform_mask`` and ``forward_with_masks`` support with
tests before exposing a custom container.

Before returning or accepting a module, run:

.. code-block:: python

   from panorai.ops.spherical import nn as spherical_nn

   report = spherical_nn.validate(module, input_shape=(height, width))

The report must contain no forbidden planar spatial layer.  Grid and mask
transforms must agree with the actual output shape, and device, dtype, layout,
batch order and autograd must be preserved.

An application that uses a layer should normally accept the configured module
as a component or construct it through a clearly named, deterministic factory.
Do not make model acquisition, checkpoint download or license acceptance a
side effect of importing or constructing the application.

Result and modality contract
----------------------------

Choose the narrowest public result that preserves semantics:

* use ``RasterField`` for a pipeline raster, including values, kind, units,
  support, validity and ``RasterGrid``;
* use ``ClassificationResult`` or ``EmbeddingResult`` for those global or
  per-view results;
* use ``PerViewResults``/``ViewResult`` for detections or keypoints that must
  retain view geometry;
* use ``SphericalObservations`` after lifting observations into panorama-frame
  bearings;
* use ``ResultBundle`` for named mixed outputs;
* reuse a domain result type when it already expresses the complete contract.

Never return a bare spatial tensor when its grid differs from the input.  Do
not infer support or validity from the numeric value.  Black RGB, zero labels
and zero radial range can all be valid values.  Floating non-finite data may
not silently become valid zero.

Every application specification must state:

* input domain, shape/layout, dtype and backend;
* coordinate frame and pose direction;
* output type, grid and units;
* radial range versus any explicitly named alternative depth quantity;
* support and validity propagation;
* interpolation/fusion for every raster kind;
* batch, device and autograd behavior;
* deterministic versus stochastic behavior and seed ownership;
* optional dependencies, model/data licenses and absence behavior;
* external weights or resources and their identity/checksum policy;
* stability tier, evidence and known failure modes.

Configuration, persistence and cache
------------------------------------

``describe()`` is explanatory and must be JSON-friendly.  ``to_config()`` is
an executable persistence contract and is stricter.

For a serializable built-in application:

1. use a globally unique, stable ``application_id``;
2. emit ``application_config(application_id, parameters)``;
3. implement ``from_config`` with type and value validation;
4. add the class to the explicit allowlist in ``Application.load``;
5. add round-trip, unknown-id, non-finite, oversized-file and atomic-write
   tests.

Do not place an import path from the file into a dynamic importer.  An
application containing a lambda, arbitrary callable, unregistered model or
external adapter may execute in memory, but ``save()`` must fail with the exact
configuration path.  This is the expected behavior for model-driven wrappers
until their components have safe registrations.

A new built-in Stage or configured operation is not automatically accepted by
Pipeline persistence.  Add it to the explicit registry in
``panorai.pipeline._serialization`` only after defining a safe parameter codec
and round-trip tests.  Never use pickle.

For a custom Torch architecture, trusted application code may call
``register_torch_module(identifier, module_type, encode=..., decode=...)``.
Registration is process-local and must occur before loading.  A custom module
used directly by ``OnSphere`` is rejected unless it is built entirely from
validated layers or its registration explicitly declares
``spherical_semantics="pointwise"``.  That declaration is a trusted assertion
by application code, not an inference by PanorAi; it is only valid for modules
that preserve every spatial sample independently.  Forward hooks are rejected
because they can change output semantics after validation.  Topology remains
canonical JSON and weights use checksum-bound safetensors.  Loading never
imports a class named by the archive.

Cache identity must cover values, support, validity, grid, dtype, layout,
backend, device, version, configuration and registered weight identity.  Do
not cache a training module or a tensor requiring gradients; cache must never
detach autograd silently.

Files, exports and documentation
--------------------------------

For a normal built-in application, the change usually touches:

.. code-block:: text

   panorai/applications/_example.py          high-level wrapper
   panorai/applications/__init__.py          explicit public export
   panorai/<domain>/...                      one algorithm implementation
   panorai/ops/spherical/...                 optional operation/layer adapter
   panorai/pipeline/...                      only if a reusable Stage is needed
   tests/test_example_application.py        wrapper and real composition tests
   docs/guides/...                           user route and limitations
   docs/reference/...                        exact public names/signatures
   contracts/public-api-surfaces-v2.json     stability inventory

Also update the application load registry and Pipeline/Torch registries only
when persistence is intentionally supported.  Do not add a root
``panorai`` export merely for convenience; preserve import isolation and use
the narrowest owning namespace.

The wrapper must show both forms in documentation:

.. code-block:: python

   result = SphericalExample(...)(panorama)
   result = SphericalExample(...).as_pipeline()(panorama)

The two forms must be equivalent for the wrapper's documented inputs.  The
low-level form may require a typed request for a multi-input application.

Required tests and evidence
---------------------------

Before review, cover all applicable rows:

* the wrapper uses the real domain implementation and returns the declared
  public result;
* ``wrapper(...)`` and ``wrapper.as_pipeline()(...)`` agree;
* ``with_params`` is immutable and preserves the concrete wrapper type;
* every legal/illegal Stage transition fails or succeeds before inference;
* support, validity and numeric values are tested independently;
* seam, poles, partial support and all-invalid regions are considered;
* black RGB, label zero, valid zero range and invalid ``NaN`` are distinct;
* NumPy ``HW/HWC`` and applicable Torch ``CHW/NCHW`` cases are covered;
* labels/masks remain nearest-only and depth has explicit radial units;
* batching preserves order, dtype, device and autograd;
* optional imports stay lazy and missing dependencies fail with actionable
  guidance;
* configuration/pipeline round trips pass for registered components and fail
  explicitly for unregistered components;
* public exports and stability inventory agree;
* examples execute, strict Sphinx has zero warnings, and an installed-wheel
  consumer runs outside the checkout before any Stable promotion.

Algorithmic correctness requires an independent oracle.  Scientific
applications also need frozen prospective criteria and eligible held-out data;
unit tests and successful adapter execution are not sufficient promotion
evidence.  Model weights, datasets and training orchestration remain external
unless redistribution and scope are separately approved.

Review checklist
----------------

A reviewer should be able to answer yes to each question:

1. Is there one mathematical implementation rather than a copy in the
   wrapper?
2. Is the chosen artifact level the smallest one that expresses the behavior?
3. Are the high-level and low-level routes equivalent and inspectable?
4. Are grid, support, validity, units and interpolation explicit?
5. Does every shape-changing spatial operation transform the grid correctly?
6. Are optional dependencies lazy and license/resource failures early?
7. Does persistence use explicit registrations and safe formats only?
8. Are public exports, stability tier, migration impact and limitations
   documented?
9. Do tests exercise real public objects and an independent oracle where
   correctness is claimed?
10. Is the evidence target named correctly: source, wheel, sdist, installed
    wheel or external service?

If any answer is no, keep the surface internal or Experimental and record the
missing contract or evidence instead of widening the public promise.
