from typing import Any, Tuple
import numpy as np
import logging
import math
from ..base.strategy import BaseProjectionStrategy
from .config import GnomonicConfig
from ...utils.exceptions import ProcessingError

logger = logging.getLogger('spherical_projections.gnomonic_projection.gnomonic.strategy')

class GnomonicProjectionStrategy(BaseProjectionStrategy):
    """
    Strategy for Gnomonic Projection.
    Implements both forward (geographic → planar) and inverse (planar → geographic) projections.
    """

    def __init__(self, config: GnomonicConfig) -> None:
        logger.debug("Initializing GnomonicProjectionStrategy.")
        if not isinstance(config, GnomonicConfig):
            raise TypeError(f"Expected GnomonicConfig, got {type(config)} instead.")
        self.config = config
        logger.info("GnomonicProjectionStrategy initialized.")

    def from_projection_to_spherical(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        logger.debug("Starting inverse projection (planar → geographic).")
        try:
            phi1_rad = self.config.phi1_deg * math.pi / 180.0
            lam0_rad = self.config.lam0_deg * math.pi / 180.0
            rho = np.sqrt(x**2 + y**2)
            c = np.arctan2(rho, self.config.R)
            sin_c, cos_c = np.sin(c), np.cos(c)
            phi = np.arcsin(cos_c * np.sin(phi1_rad) - (y * sin_c * np.cos(phi1_rad)) / rho)
            lam = lam0_rad + np.arctan2(x * sin_c, rho * np.cos(phi1_rad) * cos_c + y * np.sin(phi1_rad) * sin_c)
            return np.rad2deg(phi), np.rad2deg(lam)
        except Exception as e:
            raise ProcessingError(f"Inverse projection error: {e}") from e

    def from_spherical_to_projection(self, lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        logger.debug("Starting forward projection (geographic → planar).")
        try:
            phi1_rad = self.config.phi1_deg * math.pi / 180.0
            lam0_rad = self.config.lam0_deg * math.pi / 180.0
            phi_rad = lat * math.pi / 180.0
            lam_rad = lon * math.pi / 180.0
            cos_c = (np.sin(phi1_rad) * np.sin(phi_rad) +
                     np.cos(phi1_rad) * np.cos(phi_rad) * np.cos(lam_rad - lam0_rad))
            cos_c = np.where(cos_c == 0, 1e-10, cos_c)
            x = self.config.R * np.cos(phi_rad) * np.sin(lam_rad - lam0_rad) / cos_c
            y = self.config.R * (np.cos(phi1_rad) * np.sin(phi_rad) - np.sin(phi1_rad) * np.cos(phi_rad) * np.cos(lam_rad - lam0_rad)) / cos_c
            mask = cos_c > 0
            return x, y, mask
        except Exception as e:
            raise ProcessingError(f"Forward projection error: {e}") from e