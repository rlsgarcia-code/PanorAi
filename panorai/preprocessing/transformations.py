import cv2
import numpy as np
from skimage.transform import resize
from typing import Union

class ImageResizer:
    """
    Handles image resizing for NumPy-based data only.
    """
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

    def resize_image(self, img: np.ndarray) -> np.ndarray:
        if self.resize_factor == 1.0:
            return img

        new_shape = (int(img.shape[0] * self.resize_factor), int(img.shape[1] * self.resize_factor))
        if self.method == "skimage":
            if img.ndim == 3:
                return resize(
                    img, (*new_shape, img.shape[2]),
                    mode=self.mode,
                    anti_aliasing=self.anti_aliasing,
                    preserve_range=True
                )
            else:
                return resize(
                    img, new_shape,
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


class PreprocessEquirectangularImage:
    """
    Provides methods for extending, rotating, and resizing equirectangular images (NumPy-based).
    """

    @classmethod
    def extend_height(cls, image: np.ndarray, shadow_angle: float) -> np.ndarray:
        if shadow_angle <= 0:
            return image

        fov_original = 180.0
        height, width = image.shape[:2]
        h_prime = int(round(height / (1 - (shadow_angle / fov_original)))) - height
        extension_shape = (h_prime, width) if image.ndim == 2 else (h_prime, width, image.shape[2])
        extension = np.zeros(extension_shape, dtype=image.dtype)
        return np.vstack((image, extension))

    @classmethod
    def undo_extend_height(cls, extended_image: np.ndarray, shadow_angle: float) -> np.ndarray:
        fov_original = 180.0
        estimated_original_height = int(round(extended_image.shape[0] / (1.0 + shadow_angle / fov_original)))
        return extended_image[:estimated_original_height, :, ...]

    @classmethod
    def rotate(cls, image: np.ndarray, delta_lat: float, delta_lon: float) -> np.ndarray:
        if image.ndim == 2:
            image = image[..., np.newaxis]

        if (delta_lat == 0) & (delta_lon == 0):
            return image

        H, W = image.shape[:2]
        lat_vals = np.linspace(-90, 90, H)
        lon_vals = np.linspace(-180, 180, W)
        lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing="ij")

        lat_rad = np.radians(lat_grid)
        lon_rad = np.radians(lon_grid)
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)

        rot_lat = np.radians(delta_lat)
        rot_lon = np.radians(delta_lon)

        R_y = np.array([
            [ np.cos(rot_lat), 0, np.sin(rot_lat)],
            [             0,   1,             0],
            [-np.sin(rot_lat), 0, np.cos(rot_lat)]
        ])
        R_z = np.array([
            [np.cos(rot_lon), -np.sin(rot_lon), 0],
            [np.sin(rot_lon),  np.cos(rot_lon), 0],
            [           0,                0,    1]
        ])
        R = R_z @ R_y
        xyz = np.stack([x, y, z], axis=-1)
        xyz_rotated = np.einsum("ij,hwj->hwi", R, xyz)
        x_r = xyz_rotated[..., 0]
        y_r = xyz_rotated[..., 1]
        z_r = xyz_rotated[..., 2]

        lat_rot = np.degrees(np.arcsin(z_r))
        lon_rot = np.degrees(np.arctan2(y_r, x_r))
        lon_rot = (lon_rot + 180) % 360 - 180

        map_x = (lon_rot + 180) / 360 * W
        map_y = (lat_rot + 90) / 180 * H

        # We'll do a simple cv2.remap for each channel
        rotated = np.zeros_like(image)
        for c in range(image.shape[2]):
            rotated[..., c] = cv2.remap(
                image[..., c],
                map_x.astype(np.float32),
                map_y.astype(np.float32),
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP
            )
        if rotated.shape[2] == 1:
            rotated = rotated[..., 0]
        return rotated

    @classmethod
    def preprocess(cls, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply height extension, rotation and resize in sequence."""
        processed = image
        if kwargs.get("shadow_angle"):
            processed = cls.extend_height(processed, kwargs.get("shadow_angle", 0))

        processed = cls.rotate(
            processed,
            kwargs.get("delta_lat", 0),
            kwargs.get("delta_lon", 0)
        )

        if kwargs.get("resize_factor", 1) != 1:
            processed = ImageResizer(
                resize_factor=kwargs.get("resize_factor", 1.0),
                method=kwargs.get("resize_method", "skimage")
            ).resize_image(processed)

        return processed