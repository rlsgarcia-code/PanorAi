"""Fully convolutional adapters for pretrained ImageNet classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from panorai.image_processing.torch import spherical_area_average


def _linear_as_conv(
    layer: nn.Linear,
    *,
    input_channels: int,
    kernel_size: tuple[int, int],
) -> nn.Conv2d:
    expected = input_channels * kernel_size[0] * kernel_size[1]
    if layer.in_features != expected:
        raise ValueError(
            f"linear layer expects {layer.in_features} values, not {expected}"
        )
    converted = nn.Conv2d(
        input_channels,
        layer.out_features,
        kernel_size=kernel_size,
        bias=layer.bias is not None,
        device=layer.weight.device,
        dtype=layer.weight.dtype,
    )
    converted.weight = layer.weight
    converted.weight.data = layer.weight.data.reshape(
        layer.out_features, input_channels, *kernel_size
    )
    converted.bias = layer.bias
    return converted


def _mlp_as_convolution(
    classifier: nn.Sequential,
    *,
    input_channels: int,
    spatial_shape: tuple[int, int],
) -> nn.Sequential:
    converted: list[nn.Module] = []
    next_channels = input_channels
    first_linear = True
    for layer in classifier:
        if isinstance(layer, nn.Linear):
            kernel = spatial_shape if first_linear else (1, 1)
            converted.append(
                _linear_as_conv(layer, input_channels=next_channels, kernel_size=kernel)
            )
            next_channels = layer.out_features
            first_linear = False
        elif isinstance(layer, nn.Dropout):
            converted.append(nn.Dropout2d(layer.p, inplace=layer.inplace))
        else:
            converted.append(layer)
    if first_linear:
        raise ValueError("classifier does not contain a linear layer")
    return nn.Sequential(*converted)


@dataclass(frozen=True, slots=True)
class DenseFCNOutput:
    """Dense backbone features and ImageNet class-score lattice."""

    features: Tensor
    logits: Tensor


class ImageNetFCN(nn.Module):
    """Convert an AlexNet, VGG16, or ResNet18 classifier to dense inference.

    The conversion reuses the exact classifier ``Parameter`` objects. Call
    :func:`port_module_with_report` afterwards to replace all conventional
    spatial layers with spherical ERP equivalents.
    """

    SUPPORTED = ("alexnet", "vgg16", "resnet18")

    def __init__(self, model: nn.Module, architecture: str) -> None:
        super().__init__()
        architecture = architecture.strip().lower()
        if architecture not in self.SUPPORTED:
            raise ValueError(f"architecture must be one of {self.SUPPORTED}")
        self.architecture = architecture
        if architecture == "alexnet":
            self.features = model.features
            self.classifier = _mlp_as_convolution(
                model.classifier, input_channels=256, spatial_shape=(6, 6)
            )
        elif architecture == "vgg16":
            self.features = model.features
            self.classifier = _mlp_as_convolution(
                model.classifier, input_channels=512, spatial_shape=(7, 7)
            )
        else:
            self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
            self.layer1 = model.layer1
            self.layer2 = model.layer2
            self.layer3 = model.layer3
            self.layer4 = model.layer4
            self.classifier = _linear_as_conv(
                model.fc, input_channels=model.fc.in_features, kernel_size=(1, 1)
            )

    def forward_dense(self, values: Tensor) -> DenseFCNOutput:
        """Return the final feature lattice and dense class-score lattice."""

        if self.architecture in {"alexnet", "vgg16"}:
            features = self.features(values)
            logits = self.classifier(features)
            return DenseFCNOutput(features=features, logits=logits)
        features = self.stem(values)
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        features = self.layer4(features)
        return DenseFCNOutput(features=features, logits=self.classifier(features))

    def forward(self, values: Tensor, *, spherical_average: bool = True) -> Tensor:
        logits = self.forward_dense(values).logits
        if spherical_average:
            return spherical_area_average(logits)
        return logits.mean(dim=(-2, -1))


def class_activation_map(
    logits: Tensor,
    class_index: int | Tensor,
    *,
    output_shape: tuple[int, int] | None = None,
    positive_only: bool = True,
) -> Tensor:
    """Return normalized dense evidence for one class per batch item."""

    if logits.ndim != 4:
        raise ValueError("logits must use NCHW layout")
    batch, classes = logits.shape[:2]
    if isinstance(class_index, Integral):
        indices = torch.full(
            (batch,), int(class_index), device=logits.device, dtype=torch.long
        )
    else:
        indices = class_index.to(device=logits.device, dtype=torch.long)
        if indices.shape != (batch,):
            raise ValueError("class_index tensor must have shape (N,)")
    if torch.any((indices < 0) | (indices >= classes)):
        raise ValueError("class index is outside the logits channel range")
    maps = logits[torch.arange(batch, device=logits.device), indices]
    if positive_only:
        maps = maps.relu()
    if output_shape is not None and maps.shape[-2:] != output_shape:
        maps = F.interpolate(
            maps[:, None], size=output_shape, mode="bilinear", align_corners=False
        )[:, 0]
    minimum = maps.amin(dim=(-2, -1), keepdim=True)
    maximum = maps.amax(dim=(-2, -1), keepdim=True)
    scale = (maximum - minimum).clamp_min(torch.finfo(maps.dtype).eps)
    return (maps - minimum) / scale
