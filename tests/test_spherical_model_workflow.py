from __future__ import annotations

import numpy as np
import pytest

from panorai.core.numpy import SphericalGaussianBlur2d
from panorai.data import EquirectangularImage, RasterFieldResult
from panorai.workflow import SphericalInputSpec, SphericalOutputSpec
from panorai.workflow.numpy import SphericalImageModule


class UserDefinedDoG(SphericalImageModule):
    def __init__(self) -> None:
        super().__init__(
            input_spec=SphericalInputSpec(
                field="image", grayscale=True, scale=1.0 / 255.0
            ),
            output_specs={
                "dog": SphericalOutputSpec.continuous(
                    component_ids=("sigma_2_minus_sigma_1",)
                )
            },
        )
        self.small = SphericalGaussianBlur2d(1.0, kernel_size=3)
        self.large = SphericalGaussianBlur2d(2.0, kernel_size=7)

    def forward(self, image: np.ndarray):
        return {"dog": self.large(image) - self.small(image)}


def test_user_defined_numpy_model_runs_on_tensor_and_container() -> None:
    y, x = np.indices((16, 32))
    image = np.stack((x * 4, y * 8, (x + y) * 2), axis=-1).astype(np.uint8)
    panorama = EquirectangularImage(image)
    model = UserDefinedDoG()
    raw = model(model._prepare(panorama))
    tree = model.predict(panorama)
    assert set(tree) == {"dog"}
    output = tree["dog"]
    assert output.field("dog").shape == (16, 32)
    np.testing.assert_allclose(output.field("dog"), raw["dog"][0, 0])
    assert output.field_spec("dog").component_ids == ("sigma_2_minus_sigma_1",)


def test_model_predict_rejects_partial_support_by_default() -> None:
    image = np.ones((12, 24, 3), dtype=np.uint8)
    support = np.ones((12, 24), dtype=bool)
    support[0] = False
    panorama = EquirectangularImage(image, support_mask=support)
    with pytest.raises(ValueError, match="complete support"):
        UserDefinedDoG().predict(panorama)


def test_explicit_policy_transports_single_batch_output_masks() -> None:
    class MaskAwareModel(SphericalImageModule):
        def __init__(self) -> None:
            super().__init__(
                input_spec=SphericalInputSpec(field="image"),
                output_specs={"response": SphericalOutputSpec.continuous()},
                support_policy="explicit",
            )

        def forward(self, image: np.ndarray):
            mask = np.ones((1, 1, *image.shape[-2:]), dtype=bool)
            mask[..., 0, :] = False
            return {"response": RasterFieldResult(image[:, :1], mask, mask)}

    image = np.ones((12, 24, 3), dtype=np.uint8)
    output = MaskAwareModel().predict(EquirectangularImage(image))["response"]
    assert output.field("response").shape == (12, 24)
    assert output.support("response").shape == (12, 24)
    assert not output.validity("response")[0].any()


def test_torch_model_bridge_preserves_tensor_and_autograd() -> None:
    torch = pytest.importorskip("torch")
    from panorai.core.torch import SphericalGaussianBlur2d as TorchGaussian
    from panorai.workflow.torch import SphericalImageModule as TorchImageModule

    class TorchDoG(TorchImageModule):
        def __init__(self) -> None:
            super().__init__(
                input_spec=SphericalInputSpec(field="image"),
                output_specs={"dog": SphericalOutputSpec.continuous()},
            )
            self.small = TorchGaussian(1.0, kernel_size=3)
            self.large = TorchGaussian(2.0, kernel_size=7)

        def forward(self, image):
            return {"dog": self.large(image) - self.small(image)}

    source = torch.randn(1, 1, 12, 24, dtype=torch.float32, requires_grad=True)
    panorama = EquirectangularImage(source)
    output = TorchDoG().predict(panorama)["dog"].field("dog")
    assert isinstance(output, torch.Tensor)
    assert output.grad_fn is not None
    output.square().mean().backward()
    assert source.grad is not None
