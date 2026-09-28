import numpy as np
from .base_blenders import BaseBlender
from .registry import BlenderRegistry
from typing import Any
from scipy.ndimage import gaussian_filter
from ._inputs import expand_mask, finish_blend, masked_values, prepare_blend_inputs

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
    def blend(self, images, masks, return_mask=False, **kwargs):
        """
        Blends images using a Gaussian weighting approach.
        """
        images, masks = prepare_blend_inputs(images, masks)

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

        if len(tangent_points) != len(images):
            raise ValueError("tangent_points must contain one entry per image.")

        effective_masks = []
        for img, mask, (lat_deg, lon_deg) in zip(images, masks, tangent_points):
            spec = getattr(projector, "spec", None)
            if spec is not None:
                face_height, face_width = spec.output_shape_hw
            else:
                face_height = projector.config.y_points
                face_width = projector.config.x_points
            distance = get_distribution(
                fov_deg, face_height, face_width, mu=mu, sig=sig
            )
            distance_3d = np.dstack([distance]*3)
            if spec is not None:
                from panorai.geometry import GnomonicProjector, GnomonicSpec

                local_spec = GnomonicSpec(
                    center_lat_deg=lat_deg,
                    center_lon_deg=lon_deg,
                    hfov_deg=fov_deg,
                    vfov_deg=fov_deg,
                    roll_deg=spec.roll_deg,
                    output_shape_hw=spec.output_shape_hw,
                )
                projected = GnomonicProjector(
                    local_spec, interpolation="bilinear"
                ).back_project(distance_3d, img.shape[:2])
                equirect_weights = projected.data
                projector_mask = projected.support_mask
            else:
                projector.config.update(phi1_deg=lat_deg, lam0_deg=lon_deg)
                equirect_weights, projector_mask = projector.back_project(
                    distance_3d, img.shape[:2], return_mask=True
                )
            equirect_weights = np.asarray(equirect_weights)[..., 0]
            projector_mask = np.asarray(projector_mask, dtype=bool)
            equirect_mask = equirect_weights / distance.max() if distance.max() > 0 else equirect_weights
            effective_mask = mask & projector_mask
            weights = np.where(effective_mask, equirect_mask, 0.0)
            combined += masked_values(img, effective_mask) * expand_mask(weights, img)
            weight_map += weights
            effective_masks.append(effective_mask)

        valid_weights = weight_map > 0
        denominator = expand_mask(weight_map, combined)
        np.divide(combined, denominator, out=combined, where=denominator > 0)
        combined[~valid_weights] = 0
        return finish_blend(combined, effective_masks, return_mask)
