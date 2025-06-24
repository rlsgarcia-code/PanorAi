# Blending Strategies: README

Below are three blending strategies: **Average**, **Feathering**, and **Gaussian**.
Each strategy implements the `blend` method differently, resulting in distinct blending behaviors.
After the code snippets, there is an explanation of how each strategy works, the key differences between them, and what outputs to expect.

---

## 1. AverageBlender

```python
import numpy as np
from .base_blenders import BaseBlender
from typing import Any

import numpy as np
import matplotlib.pyplot as plt


class AverageBlender(BaseBlender):

    def blend(self, images, masks, **kwargs):
        """
        Blends images by performing a simple average in overlapping areas.

        :param images: List of image arrays.
        :param masks: List of corresponding masks for weighting.
        :return: Blended image.
        """
        if not images or not masks or len(images) != len(masks):
            raise ValueError("Images and masks must have the same non-zero length.")

        img_shape = images[0].shape
        combined = np.zeros(img_shape, dtype=np.float32)
        weight_map = np.zeros(img_shape[:2], dtype=np.float32)

        for img, mask in zip(images, masks):
            # A simple binary mask check: pixels with mean > 0 are considered valid.
            equirect_mask = (np.mean(img, axis=-1) > 0).astype(np.float32)

            # Apply blending by adding each image to the combined output
            combined += img

            # Track how many images contribute to each pixel
            weight_map += equirect_mask

        # Normalize the blended image where we have valid pixels
        valid_weights = weight_map > 0
        combined[valid_weights] /= weight_map[valid_weights, None]

        # Ensure that pixels with zero weight remain zero
        combined[~valid_weights] = 0
        return combined
```

### How It Works
- **Averaging**: For each overlapping pixel, it sums up the values from all contributing images, then divides by the total number of valid contributions (based on non-zero mask areas).
- **Mask Check**: In this snippet, the mask effectively comes from checking which pixels have non-zero content (`mean(img, axis=-1) > 0`), then counting how many times each pixel is covered.

### Expected Output
- **Uniformly Weighted Blend**: Overlapping areas from multiple images end up being averaged with equal weight. This is the simplest approach and can produce sharp transitions at image boundaries if exposure or color differences exist.

---

## 2. FeatheringBlender

```python
import numpy as np
from scipy.ndimage import distance_transform_edt
from .base_blenders import BaseBlender
from typing import Any

class FeatheringBlender(BaseBlender):

    def blend(self, images, masks, **kwargs):
        """
        Blends images using a distance-transform-based feathering approach.

        :param images: List of image arrays.
        :param masks: List of corresponding masks for weighting.
        :return: Blended image.
        """
        if not images or not masks or len(images) != len(masks):
            raise ValueError("Images and masks must have the same non-zero length.")

        img_shape = images[0].shape
        combined = np.zeros(img_shape, dtype=np.float32)
        weight_map = np.zeros(img_shape[:2], dtype=np.float32)

        for img, mask in zip(images, masks):
            # valid_mask identifies which pixels are non-zero (valid in the image)
            valid_mask = np.max(img > 0, axis=-1).astype(np.float32)

            # Compute distance from zero/non-valid regions
            distance = distance_transform_edt(valid_mask)

            # Normalize distance values to the [0, 1] range
            if distance.max() != 0:
                feathered_mask = distance / distance.max()
            else:
                feathered_mask = distance

            # Multiply the image by its feathered weight before adding
            combined += img * feathered_mask[..., None]
            weight_map += feathered_mask

        # Normalize only the valid pixels
        valid_weights = weight_map > 0
        combined[valid_weights] /= weight_map[valid_weights, None]

        # Preserve zeros where there was no contribution
        combined[~valid_weights] = 0
        return combined
```

### How It Works
- **Feathering**: Each image is converted into a distance map (using `distance_transform_edt`) that measures how far each valid pixel is from an invalid (zero) pixel.
- **Smooth Transitions**: Pixels near the center of a valid region have higher weights, and those near edges have lower weights. This blending creates softer seams in overlapping regions.

### Expected Output
- **Smooth Boundary Transitions**: Overlapping areas are blended seamlessly with gradual fade-in/out, reducing visible seams compared to a simple average.

---

## 3. GaussianBlender

```python
import numpy as np
from .base_blenders import BaseBlender
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

def multivariate_gaussian_2d(x, mean, cov):
    """
    Computes the 2D multivariate Gaussian (normal) probability density function.
    ...
    """
    mean = np.asarray(mean).reshape(-1)
    x = np.atleast_2d(x)
    inv_cov = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)
    if det_cov <= 0:
        raise ValueError("Covariance matrix must be positive definite (det > 0).")
    norm_factor = 1.0 / (2.0 * np.pi * np.sqrt(det_cov))
    diff = x - mean
    exponent = -0.5 * np.einsum('...i,ij,...j', diff, inv_cov, diff)
    pdf_vals = norm_factor * np.exp(exponent)
    if pdf_vals.shape[0] == 1:
        return pdf_vals[0]
    return pdf_vals

def get_distribution(fov_deg, H, W, mu=0, sig=1):
    v_max = u_max = np.tan(np.deg2rad(fov_deg / 2))
    v_min = u_min = -u_max
    grid = np.stack(np.meshgrid(
        np.linspace(u_min, u_max, W),
        np.linspace(v_min, v_max, H)
    ))
    probs = multivariate_gaussian_2d(
        grid.reshape(2, -1).T,
        mean=np.array([mu, mu]),
        cov=np.diag(np.array([sig, sig]))
    ).reshape((H, W))
    return probs

class GaussianBlender(BaseBlender):

    def blend(self, images, masks, **kwargs):
        """
        Blends images using a Gaussian-based weighting approach.

        :param images: List of image arrays.
        :param masks: List of corresponding masks for weighting.
        :return: Blended image.
        """
        if not images or not masks or len(images) != len(masks):
            raise ValueError("Images and masks must have the same non-zero length.")

        # Initialize the output
        img_shape = images[0].shape
        combined = np.zeros(img_shape, dtype=np.float32)
        weight_map = np.zeros(img_shape[:2], dtype=np.float32)

        # Check for required params
        required_keys = ['fov_deg', 'projector', 'tangent_points']
        missing_keys = [key for key in required_keys if key not in self.params]
        if missing_keys:
            raise ValueError(f"Error: Missing required parameters: {', '.join(missing_keys)}")

        fov_deg = self.params.get('fov_deg')
        tangent_points = self.params.get('tangent_points')
        projector = self.params.get('projector')
        mu = self.params.get('mu', 0)
        sig = self.params.get('sig', 1)

        for img, mask, (lat_deg, lon_deg) in zip(images, masks, tangent_points):
            # Configure the projector for the current image
            projector.config.update(
                phi1_deg=lat_deg,
                lam0_deg=lon_deg,
            )

            # Create a 2D Gaussian distribution on the tangent plane
            distance = get_distribution(
                fov_deg,
                projector.config.x_points,
                projector.config.y_points,
                mu=mu,
                sig=sig
            )

            # Stack to get shape (H, W, 3) if needed
            distance_3d = np.dstack([distance, distance, distance])

            # Project the Gaussian weights back to the equirectangular space
            equirect_weights = projector.backward(distance_3d, return_mask=False)[:, :, 0]

            # Normalize weights to [0, 1]
            max_val = distance.max()
            if max_val > 0:
                equirect_mask = equirect_weights / max_val
            else:
                equirect_mask = equirect_weights

            # Blend
            combined += img * equirect_mask[..., None]
            weight_map += equirect_mask

        # Normalize output
        valid_weights = weight_map > 0
        combined[valid_weights] /= weight_map[valid_weights, None]
        combined[~valid_weights] = 0
        return combined
```

### How It Works
- **Gaussian Weight Distribution**: For each image, a 2D Gaussian distribution is computed in the tangent plane (using `fov_deg`, `mu`, `sig`). The highest weight is typically at the “center” of the FOV, gradually decreasing towards the edges.
- **Projection**: The distribution is projected (via the `projector`) to match the final coordinate system (e.g., equirectangular). This yields a smoothly varying weight map.
- **Blending**: Each pixel is weighted by the Gaussian distribution, and the sum is normalized by the total contributed weights from all images.

### Expected Output
- **Center-Weighted Blending**: Overlaps are handled with a smooth Gaussian fade, usually emphasizing the image center. This can reduce seams and provides a high degree of control over how images combine.

---



## Key Differences and Summary

1.
**AverageBlender**
   - **Strategy**: Straightforward average of overlapping areas.
   - **Transitions**: Can be abrupt if there are brightness or color mismatches.
   - **Use Case**: Fast prototyping, uniform merges, and when images are already consistent.

2.
**FeatheringBlender**
   - **Strategy**: Distance transform to create soft edges.
   - **Transitions**: Smooth fade at boundaries, reducing harsh seams.
   - **Use Case**: Panorama stitching where edges overlap and need gentle blending.

3.
**GaussianBlender**
   - **Strategy**: Gaussian distribution in a tangent plane, then project to the final coordinate system.
   - **Transitions**: Soft blending with adjustable center emphasis via Gaussian parameters.
   - **Use Case**: Complex multi-view or panoramic stitching, where center weighting and adjustable fade are desirable.

All three approaches normalize the result by a weight map, ensuring that overlapping areas are properly scaled and non-overlapping areas remain zero.

---
