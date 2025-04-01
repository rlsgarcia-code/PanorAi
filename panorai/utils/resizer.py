from skimage.transform import resize
import logging
import sys
import cv2
import numpy as np
from typing import Union

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.handlers = [stream_handler]


class ImageResizer:
    """Handles image resizing with explicit configuration (NumPy only)."""

    def __init__(
        self,
        resize_factor: float = 1.0,
        method: str = "skimage",
        mode: str = "reflect",
        anti_aliasing: bool = True,
        interpolation: int = cv2.INTER_LINEAR
    ) -> None:
        self.resize_factor = resize_factor
        self.method = method
        self.mode = mode
        self.anti_aliasing = anti_aliasing
        self.interpolation = interpolation

        logger.info(f"Initialized ImageResizer with resize_factor={resize_factor}, method={method}, "
                    f"mode={mode}, anti_aliasing={anti_aliasing}, interpolation={interpolation}")

    def resize_image(self, img: np.ndarray, upsample: bool = True) -> np.ndarray:
        resize_factor = self.resize_factor
        if not upsample:
            resize_factor = 1 / resize_factor

        if resize_factor != 1.0:
            new_shape = (
                int(img.shape[0] * resize_factor),
                int(img.shape[1] * resize_factor),
            )
            logger.info(f"Resizing image with factor={resize_factor}.")
            logger.debug(f"Original shape: {img.shape}, New shape: {new_shape}.")

            if self.method == "skimage":
                if len(img.shape) == 3:
                    resized_img = resize(
                        img, (*new_shape, img.shape[2]),
                        mode=self.mode,
                        anti_aliasing=self.anti_aliasing
                    )
                else:
                    resized_img = resize(
                        img, new_shape,
                        mode=self.mode,
                        anti_aliasing=self.anti_aliasing
                    )
                logger.info("Image resizing completed using skimage.")
                return resized_img
            elif self.method == "cv2":
                resized_img = cv2.resize(
                    img,
                    (new_shape[1], new_shape[0]),
                    interpolation=self.interpolation
                )
                logger.info("Image resizing completed using cv2.")
                return resized_img
            else:
                raise ValueError(f"Unknown resizing method: {self.method}")

        logger.debug("No resizing applied; resize_factor is 1.0.")
        return img


class ResizerConfig:
    """Configuration for the resizer."""

    def __init__(
        self,
        resizer_cls: type = ImageResizer,
        resize_factor: float = 1.0,
        method: str = "skimage",
        mode: str = "reflect",
        anti_aliasing: bool = True,
        interpolation: int = cv2.INTER_LINEAR
    ) -> None:
        self.resize_factor = resize_factor
        self.method = method
        self.mode = mode
        self.anti_aliasing = anti_aliasing
        self.interpolation = interpolation
        self.resizer_cls = resizer_cls

    def __repr__(self) -> str:
        return (f"ResizerConfig(resize_factor={self.resize_factor}, method='{self.method}', "
                f"mode='{self.mode}', anti_aliasing={self.anti_aliasing}, interpolation={self.interpolation})")

    def create_resizer(self) -> ImageResizer:
        return self.resizer_cls(
            resize_factor=self.resize_factor,
            method=self.method,
            mode=self.mode,
            anti_aliasing=self.anti_aliasing,
            interpolation=self.interpolation
        )