Compose a sphere-native model
=============================

.. list-table:: Tutorial contract
   :widths: 28 72

   * - **Application**
     - Custom filtering, scale space, CNN, CAM, depth, or segmentation model
   * - **Input**
     - Floating ``CHW``/``NCHW`` ERP tensor, or ``EquirectangularImage``
   * - **Output**
     - Tensor tree or typed ERP field tree
   * - **Strategy**
     - Sphere-native composition
   * - **API**
     - ``panorai.core.numpy`` / ``panorai.core.torch`` and
       ``panorai.workflow``
   * - **Stability**
     - Experimental
   * - **Dependencies**
     - NumPy required; Torch optional
   * - **Evidence type**
     - Analytic geometry, backend parity, autograd, and integration regression

The composition rule is intentionally the same as the selected numerical
backend: construct small spherical operations, write normal control flow, and
call the model. ``apply_spherical`` is not required for a composed model.

How this application is composed
--------------------------------

This tutorial operates at the core layer itself: immutable operation specs,
one explicit numerical backend, ordinary backend control flow, and an optional
typed-container bridge. It intentionally does not introduce a DoG, CNN or
segmentation wrapper. Those can be built from the same vocabulary; established
PanorAi application wrappers add defaults and diagnostics only where those are
part of the method.

.. code-block:: python

   from panorai.core.numpy import Module, SphericalGaussianBlur2d

   class DifferenceOfGaussians(Module):
       def __init__(self, sigma_a=1.0, sigma_b=2.0):
           self.a = SphericalGaussianBlur2d(sigma_a)
           self.b = SphericalGaussianBlur2d(sigma_b)

       def forward(self, image):
           return self.b(image) - self.a(image)

   dog = DifferenceOfGaussians()
   response_nchw = dog(image_nchw)

To accept a typed ERP container as well, inherit the backend-specific
``SphericalImageModule`` and declare the input and output field semantics:

.. code-block:: python

   from panorai.workflow import SphericalInputSpec, SphericalOutputSpec
   from panorai.workflow.numpy import SphericalImageModule

   class ContainerDoG(SphericalImageModule):
       def __init__(self):
           super().__init__(
               input_spec=SphericalInputSpec(
                   field="image", grayscale=True, scale=1 / 255
               ),
               output_specs={"dog": SphericalOutputSpec.continuous()},
           )
           self.dog = DifferenceOfGaussians()

       def forward(self, image):
           return {"dog": self.dog(image)}

   fields = ContainerDoG().predict(panorama)
   response_hw = fields["dog"].field("dog")

The default policy accepts only fully supported, valid inputs. A method that
supports partial observations must opt into the explicit policy and return
``RasterFieldResult`` masks; numeric zero is never interpreted as invalid.

See :doc:`../../reference/core_operations` for the complete primitive vocabulary,
Torch parameter sharing, native-backend boundary, multiresolution outputs, and
application composition patterns.
