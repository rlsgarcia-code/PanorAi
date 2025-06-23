import logging
from ..samplers.registry import SamplerRegistry
from ..blenders.registry import BlenderRegistry
from ..projections.registry import ProjectionRegistry

logger = logging.getLogger("panorai.registry")

class PanoraiRegistry:
    """
    Unified interface to access registered samplers, blenders, and projections in PanorAi.

    This class provides a simplified way for users to:
    - List available sphere samplers, blending methods, and projection types.
    - Create instances of registered samplers, blenders, and projections dynamically.

    Example:
        from panorai.utils.registry import PanoraiRegistry
        
        # List available components
        print(PanoraiRegistry.available_samplers())
        print(PanoraiRegistry.available_blenders())
        print(PanoraiRegistry.available_projections())

        # Create objects
        sampler = PanoraiRegistry.create_sampler("fibonacci", n_points=10)
        blender = PanoraiRegistry.create_blender("average")
        projection = PanoraiRegistry.create_projection("gnomonic", fov=60)
    """

    @staticmethod
    def available_samplers():
        """
        Get a list of all registered sphere samplers.

        Returns:
            list: A list of registered sampler names.

        Example:
            samplers = PanoraiRegistry.available_samplers()
            print(samplers)  # Output: ['cube', 'fibonacci', 'spiral', ...]
        """
        return SamplerRegistry.available_samplers()

    @staticmethod
    def available_blenders():
        """
        Get a list of all registered blending methods.

        Returns:
            list: A list of registered blender names.

        Example:
            blenders = PanoraiRegistry.available_blenders()
            print(blenders)  # Output: ['average', 'closest', 'weighted', ...]
        """
        return BlenderRegistry.available_blenders()

    @staticmethod
    def available_projections():
        """
        Get a list of all registered projection methods.

        Returns:
            list: A list of registered projection names.

        Example:
            projections = PanoraiRegistry.available_projections()
            print(projections)  # Output: ['gnomonic', 'stereographic', 'cylindrical', ...]
        """
        return ProjectionRegistry.available_projections()

    @staticmethod
    def create_sampler(name: str, **kwargs):
        """
        Create an instance of a registered sphere sampler.

        Args:
            name (str): The name of the sampler to create.
            **kwargs: Additional parameters for the sampler.

        Returns:
            Sampler: An instance of the requested sampler.

        Raises:
            SamplerNotFoundError: If the requested sampler does not exist.

        Example:
            sampler = PanoraiRegistry.create_sampler("fibonacci", n_points=10)
        """
        return SamplerRegistry.create(name, **kwargs)

    @staticmethod
    def create_blender(name: str, **kwargs):
        """
        Create an instance of a registered blending method.

        Args:
            name (str): The name of the blender to create.
            **kwargs: Additional parameters for the blender.

        Returns:
            Blender: An instance of the requested blender.

        Raises:
            BlenderNotFoundError: If the requested blender does not exist.

        Example:
            blender = PanoraiRegistry.create_blender("average")
        """
        return BlenderRegistry.create(name, **kwargs)

    @staticmethod
    def create_projection(name: str, **kwargs):
        """
        Create an instance of a registered projection method.

        Args:
            name (str): The name of the projection to create.
            **kwargs: Additional parameters for the projection.

        Returns:
            Projection: An instance of the requested projection.

        Raises:
            ProjectionNotFoundError: If the requested projection does not exist.

        Example:
            projection = PanoraiRegistry.create_projection("gnomonic", fov=60)
        """
        return ProjectionRegistry.create(name, **kwargs)

# Example Usage
if __name__ == "__main__":
    logger.info("🔍 Available Samplers: %s", PanoraiRegistry.available_samplers())
    logger.info("🔍 Available Blenders: %s", PanoraiRegistry.available_blenders())
    logger.info(
        "🔍 Available Projections: %s",
        PanoraiRegistry.available_projections(),
    )

    # Create objects for testing
    try:
        sampler = PanoraiRegistry.create_sampler("fibonacci", n_points=10)
        logger.info("✅ Created Sampler: %s", sampler)

        blender = PanoraiRegistry.create_blender("average")
        logger.info("✅ Created Blender: %s", blender)

        projection = PanoraiRegistry.create_projection("gnomonic", fov=60)
        logger.info("✅ Created Projection: %s", projection)

    except Exception as e:
        logger.error("❌ Error: %s", str(e))
