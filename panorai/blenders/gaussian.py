import numpy as np
from .base_blenders import BaseBlender
from .registry import BlenderRegistry
from typing import Any
from scipy.ndimage import gaussian_filter

def multivariate_gaussian_2d(x, mean, cov):
    """
    2D multivariate Gaussian PDF.
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
    v_max = u_max = np.tan(np.deg2rad(fov_deg/2))
    grid = np.stack(np.meshgrid(np.linspace(-u_max, u_max, W), np.linspace(-v_max, v_max, H)))
    coords = grid.reshape(2, -1).T
    probs = multivariate_gaussian_2d(coords, mean=np.array([mu,mu]), cov=np.diag([sig,sig]))
    return probs.reshape(H, W)

@BlenderRegistry.register("gaussian")
class GaussianBlender(BaseBlender):
    def blend(self, images, masks, **kwargs):
        """
        Blends images using a Gaussian weighting approach.
        """
        if not images or not masks or len(images) != len(masks):
            raise ValueError("Images and masks must have the same non-zero length.")

        img_shape = images[0].shape
        combined = np.zeros(img_shape, dtype=np.float32)
        weight_map = np.zeros(img_shape[:2], dtype=np.float32)

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
            projector.config.update(phi1_deg=lat_deg, lam0_deg=lon_deg)
            distance = get_distribution(fov_deg, projector.config.y_points, projector.config.x_points, mu=mu, sig=sig)
            distance_3d = np.dstack([distance]*3)
            equirect_weights = projector.backward(distance_3d, return_mask=False)[:, :, 0]
            equirect_mask = equirect_weights / distance.max() if distance.max() > 0 else equirect_weights
            combined += img * equirect_mask[..., None]
            weight_map += equirect_mask

        valid_weights = weight_map > 0
        combined[valid_weights] /= weight_map[valid_weights, None]
        combined[~valid_weights] = 0
        return combined