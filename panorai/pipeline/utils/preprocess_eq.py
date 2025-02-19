import cv2
import numpy as np
import logging
from typing import Tuple, Union
from skimage.transform import resize

class ImageResizer:
    """Handles image resizing with explicit configuration."""

    def __init__(
        self,
        resize_factor: float = 1.0,
        method: str = "skimage",
        mode: str = "reflect",
        anti_aliasing: bool = True,
        interpolation: int = cv2.INTER_LINEAR
    ) -> None:
        """
        Initialize the ImageResizer with explicit attributes.

        Args:
            resize_factor (float): Factor by which to resize the image. >1 for upsampling, <1 for downsampling.
            method (str): Resizing method ('skimage' or 'cv2'). Default is 'skimage'.
            mode (str): Mode parameter for skimage resize. Default is "reflect".
            anti_aliasing (bool): Whether to apply anti-aliasing (only for skimage).
            interpolation (int): Interpolation method for cv2.resize. Default is cv2.INTER_LINEAR.
        """
        self.resize_factor = resize_factor
        self.method = method
        self.mode = mode
        self.anti_aliasing = anti_aliasing
        self.interpolation = interpolation

    def resize_image(self, img: np.ndarray, upsample: bool = True) -> np.ndarray:
        """
        Resize the input image based on the stored configuration.

        Args:
            img (np.ndarray): Input image.
            upsample (bool): Whether we are upsampling or downsampling (affects effective resize_factor).

        Returns:
            np.ndarray: Resized image.
        """
        resize_factor = self.resize_factor # if upsample else 1 / self.resize_factor

        if resize_factor != 1.0:
            new_shape = (int(img.shape[0] * resize_factor), int(img.shape[1] * resize_factor))

            if self.method == "skimage":
                return resize(
                    img, (*new_shape, img.shape[2]) if img.ndim == 3 else new_shape,
                    mode=self.mode,
                    anti_aliasing=self.anti_aliasing,
                    preserve_range=True

                )
            elif self.method == "cv2":
                return cv2.resize(
                    img,
                    (new_shape[1], new_shape[0]),
                    interpolation=self.interpolation
                )
            else:
                raise ValueError(f"Unknown resizing method: {self.method}")

        return img


class PreprocessEquirectangularImage:
    """
    Provides methods for extending, rotating, and resizing equirectangular images (360° panoramas).
    """

    logger = logging.getLogger("EquirectangularImage")
    logger.setLevel(logging.DEBUG)

    @classmethod
    def extend_height(cls, image: np.ndarray, shadow_angle: float) -> np.ndarray:
        """
        Extends the height of an equirectangular image based on the given additional FOV.

        Args:
            image (np.ndarray): Input equirectangular image.
            shadow_angle (float): Additional field of view in degrees to extend vertically.

        Returns:
            np.ndarray: Image with extended bottom region.
        """
        if shadow_angle <= 0:
            return image  # No extension needed

        fov_original = 180.0
        height = image.shape[0]
        h_prime = int((shadow_angle / fov_original) * height)

        black_extension = np.zeros((h_prime, image.shape[1], image.shape[2] if image.ndim == 3 else 1), dtype=image.dtype)
        return np.vstack((image, black_extension))

    @classmethod
    def undo_extend_height(cls, extended_image: np.ndarray, shadow_angle: float) -> np.ndarray:
        """
        Removes the extra bottom rows that were added by 'extend_height'.

        Args:
            extended_image (np.ndarray): The extended equirectangular image.
            shadow_angle (float): Additional field of view in degrees that was used to extend.

        Returns:
            np.ndarray: The image with the extended part removed.
        """
        fov_original = 180.0
        estimated_original_height = int(
            round(extended_image.shape[0] / (1.0 + shadow_angle / fov_original))
        )
        return extended_image[:estimated_original_height, :, :]

    @classmethod
    def rotate(cls, image: np.ndarray, delta_lat: float, delta_lon: float) -> np.ndarray:
        """
        Rotates an equirectangular image based on latitude (delta_lat) and longitude (delta_lon) shifts.

        Args:
            image (np.ndarray): Input equirectangular image.
            delta_lat (float): Latitude rotation in degrees.
            delta_lon (float): Longitude rotation in degrees.

        Returns:
            np.ndarray: Rotated equirectangular image.
        """
        H, W = image.shape[:2]
        x = np.linspace(0, W - 1, W)
        y = np.linspace(0, H - 1, H)
        xv, yv = np.meshgrid(x, y)

        lon = (xv / (W - 1)) * 360.0 - 180.0
        lat = 90.0 - (yv / (H - 1)) * 180.0

        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        x_sphere = np.cos(lat_rad) * np.cos(lon_rad)
        y_sphere = np.cos(lat_rad) * np.sin(lon_rad)
        z_sphere = np.sin(lat_rad)

        delta_lat_rad = np.radians(delta_lat)
        delta_lon_rad = np.radians(delta_lon)

        # Rotate around latitude axis
        x_rot = x_sphere
        y_rot = y_sphere * np.cos(delta_lat_rad) - z_sphere * np.sin(delta_lat_rad)
        z_rot = y_sphere * np.sin(delta_lat_rad) + z_sphere * np.cos(delta_lat_rad)

        # Rotate around longitude axis
        x_final = x_rot * np.cos(delta_lon_rad) - y_rot * np.sin(delta_lon_rad)
        y_final = x_rot * np.sin(delta_lon_rad) + y_rot * np.cos(delta_lon_rad)
        z_final = z_rot

        lon_final = np.degrees(np.arctan2(y_final, x_final))
        lat_final = np.degrees(np.arcsin(z_final))

        x_rot_map = ((lon_final + 180.0) / 360.0) * (W - 1)
        y_rot_map = ((90.0 - lat_final) / 180.0) * (H - 1)

        map_x = x_rot_map.astype(np.float32)
        map_y = y_rot_map.astype(np.float32)

        return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    @classmethod
    def preprocess(cls, image: np.ndarray, shadow_angle: float = 0, delta_lat: float = 0, delta_lon: float = 0, resize_factor: float = 1.0, resize_method: str = 'skimage') -> np.ndarray:
        """
        Preprocess an equirectangular image by extending its height, rotating, and resizing.

        Args:
            image (np.ndarray): Input equirectangular image.
            shadow_angle (float): Additional FOV to extend.
            delta_lat (float): Latitude rotation.
            delta_lon (float): Longitude rotation.
            resize_factor (float): Resize factor for upsampling/downsampling.
            resize_method (float): Resize method for upsampling/downsampling (skimage or cv2).

        Returns:
            np.ndarray: Preprocessed image.
        """
        if shadow_angle >= 0:
            processed_image = cls.extend_height(image, shadow_angle)
        else:
            processed_image = cls.undo_extend_height(image, shadow_angle)

        # processed_image = cls.rotate(processed_image, delta_lat, delta_lon) ---> Rotation is now done at the Sampler level

        if resize_factor != 1.0:
            resizer = ImageResizer(resize_factor=resize_factor, method=resize_method)
            processed_image = resizer.resize_image(processed_image)

        return processed_image

    @classmethod
    def save_image(cls, image: np.ndarray, file_path: str) -> None:
        """
        Save the given image to the specified file path.

        Args:
            image (np.ndarray): Image to save.
            file_path (str): Output path.
        """
        cv2.imwrite(file_path, image)