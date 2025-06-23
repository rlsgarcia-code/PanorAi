import logging
import numpy as np
from typing import Any, Dict, List, Union, Literal, Tuple, Optional

from PIL import Image

from ..config.config_manager import ConfigManager
from ..samplers.registry import SamplerRegistry
from ..blenders.registry import BlenderRegistry
from ..projections.registry import ProjectionRegistry
from .exceptions import ProjectionNotFoundError, \
    BlenderNotFoundError, \
    SamplerNotFoundError, \
    ConfigNotFoundError
from ..data.factory import DataFactory
from ..data.equirectangular_image import EquirectangularImage
from ..data.gnomonic_image import GnomonicFace
from ..data.gnomonic_imageset import GnomonicFaceSet #==> TODO, implement method to create facesets out of faces or list of arrays
#  from ..models.panorai_pipeline import EquirectangularProcessingPipeline -> Deprecated

logger = logging.getLogger("panorai.factory")


class PanoraiFactory:
    """
    High-level factory for managing and creating objects in PanorAi.

    - **Data Objects**: `EquirectangularImage`, `GnomonicFace`, `GnomonicFaceSet`
    - **Samplers, Blenders, Projections**: Retrieve ready-to-use objects and attach them.
    - **Configuration Management**: Retrieve and modify configurations.
    - **Pipeline Creation**: Generate `PanoraiPipeline` instances.

    📌 **Key Features:**
    - Directly **attaches** samplers, blenders, and projections to objects upon creation.
    - Eliminates manual configuration steps for projections and samplers.
    - Ensures consistency in transformation workflows.

    📌 **Usage Examples**
    ```python
    # Load an EquirectangularImage
    eq_img = PanoraiFactory.load_image("panorama.jpg")

    # Attach a sampler
    eq_img.attach_sampler("fibonacci", n_points=100)

    # Attach a projection
    eq_img.attach_projection("gnomonic", fov=90)

    # Convert to Gnomonic
    gnomonic_face = eq_img.to_gnomonic(lat=30, lon=45, fov=90)

    # Convert back using an attached blender
    gfs = eq_img.to_gnomonic_face_set(fov=90)
    gfs.attach_blender("average")
    reconstructed_eq = gfs.to_equirectangular((1024, 2048))
    ```
    """

    ### 📌 CONFIGURATION MANAGEMENT ###

    @classmethod
    def modify_config(cls, name: str, **kwargs) -> None:
        """
        Modify an existing configuration dynamically.
        """
        try:
            ConfigManager.modify_config(name, **kwargs)
            logger.info(f"✅ Updated config '{name}' with {kwargs}")
        except KeyError as e:
            available = ConfigManager.available_configs()
            raise ConfigNotFoundError(name, available) from e

    @classmethod
    def describe_config(cls, name: str) -> None:
        """Print details of a configuration."""
        try:
            ConfigManager.describe_config(name)
        except KeyError as e:
            raise ConfigNotFoundError(name, ConfigManager.available_configs()) from e

    ### 📌 DATA CREATION & ATTACHMENT ###

    @classmethod
    def load_image(cls, file_path: str, data_type: Literal["equirectangular", "gnomonic_face"] = "equirectangular") -> EquirectangularImage:
        """
        Loads an image and returns an **EquirectangularImage** or **GnomonicFace**.
        
        Automatically attaches a **default projection** for gnomonic images.

        Args:
            file_path (str): Path to the image file.
            data_type (str): "equirectangular" or "gnomonic_face".

        Returns:
            EquirectangularImage | GnomonicFace
        """
        if not isinstance(file_path, str) or not file_path.endswith((".png", ".jpg", ".jpeg")):
            raise ValueError("❌ Invalid file path. Please provide a valid image file (PNG, JPG, JPEG).")
        
        img = DataFactory.from_file(file_path, data_type)
        
        # Attach default projection for Gnomonic images
        if isinstance(img, GnomonicFace):
            img.attach_projection("gnomonic")
        
        return img

    @classmethod
    def create_data_from_array(cls, data: np.ndarray, data_type: str, **kwargs) -> Union[EquirectangularImage, GnomonicFace]:
        """
        Create a spherical data object from a NumPy array.
        """
        img = DataFactory.from_array(data, data_type, **kwargs)

        # Attach default projection for Gnomonic images
        if isinstance(img, GnomonicFace):
            img.attach_projection("gnomonic")

        return img

    ### 📌 SAMPLER, BLENDER, AND PROJECTION MANAGEMENT ###

    @classmethod
    def get_sampler(cls, name: str, **kwargs):
        """Retrieve a sampler, raising an error if not found."""
        available = SamplerRegistry.available_samplers()
        if name not in available:
            raise SamplerNotFoundError(name, available)
        return SamplerRegistry.create(name, **kwargs)

    @classmethod
    def get_blender(cls, name: str, **kwargs):
        """Retrieve a blender, raising an error if not found."""
        available = BlenderRegistry.available_blenders()
        if name not in available:
            raise BlenderNotFoundError(name, available)
        return BlenderRegistry.create(name, **kwargs)

    @classmethod
    def get_projection(cls, name: str, lat: float, lon: float, fov: float, **kwargs):
        """Retrieve a projection, raising an error if not found."""
        available = ProjectionRegistry.available_projections()
        kwargs['phi1_deg'] = lat
        kwargs['lam0_deg'] = lon
        kwargs['fov_deg'] = fov
        if name not in available:
            raise ProjectionNotFoundError(name, available)
        return ProjectionRegistry.create(name, **kwargs)

    ### 📌 PIPELINE CREATION (deprecated)###

    # @classmethod
    # def create_pipeline(cls, sampler_name: Optional[str] = None, blender_name: Optional[str] = None) -> EquirectangularProcessingPipeline:
    #     """
    #     Creates a `PanoraiPipeline` instance.

    #     📌 **Example Usage**
    #     ```python
    #     pipeline = PanoraiFactory.create_pipeline(sampler_name="fibonacci", blender_name="average")

    #     faces = pipeline.forward_pass(eq_image, fov=90)
    #     reconstructed_eq = pipeline.backward_pass(faces, eq_shape=(1024, 2048))
    #     ```
    #     """
    #     return EquirectangularProcessingPipeline(sampler_name=sampler_name, blender_name=blender_name)

    ### 📌 SUMMARY & RESET METHODS ###

    @classmethod
    def reset_all(cls):
        """Reset cached configuration instances.

        This method calls :func:`ConfigManager.reset` and only clears
        configuration objects currently stored by the manager. Sampler,
        blender and projection registries remain untouched. Extend this
        method if global cleanup of those registries is desired.
        """
        ConfigManager.reset()
        logger.info("🔄 Reset all configurations.")

    @classmethod
    def list_available(cls):
        """List all available configurations, samplers, blenders, and projections."""
        #print("✅ Available Configs:", ConfigManager.describe_config())
        logger.info("✅ Available Samplers: %s", SamplerRegistry.available_samplers())
        logger.info("✅ Available Blenders: %s", BlenderRegistry.available_blenders())
        #print("✅ Available Projections:", ProjectionRegistry.available())

