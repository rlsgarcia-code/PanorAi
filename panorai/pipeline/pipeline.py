import numpy as np
import logging
import os
import sys
import math
from typing import Any, Dict, List, Optional, Tuple, Union

from joblib import Parallel, delayed

from .pipeline_data import PipelineData
from .utils.resizer import ResizerConfig

from ..sampler import SamplerRegistry
from ..sampler.base_samplers import Sampler  # For type hints
from spherical_projections import ProjectionRegistry
from ..blender.registry import BlenderRegistry  # Importing BlenderRegistry

# Pipeline dependencies
from copy import deepcopy
import cv2

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if os.environ.get('DEBUG', 'False').lower() in ('true', '1') else logging.INFO)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logger.level)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.handlers = [stream_handler]  # Replace existing handlers


def deg_to_rad(degrees: float) -> float:
    """
    Convert degrees to radians.

    Args:
        degrees (float): Angle in degrees.

    Returns:
        float: Angle in radians.
    """
    return degrees * math.pi / 180.0


def rad_to_deg(radians: float) -> float:
    """
    Convert radians to degrees.

    Args:
        radians (float): Angle in radians.

    Returns:
        float: Angle in degrees.
    """
    return radians * 180.0 / math.pi


class PipelineConfig:
    """
    Configuration class for the ProjectionPipeline.
    """

    def __init__(
        self,
        resizer_cfg: Optional[ResizerConfig] = None,
        resize_factor: float = 1.0,
        n_jobs: int = 1
    ) -> None:
        """
        Initialize pipeline-level configuration.

        Args:
            resizer_cfg (Optional[ResizerConfig]): Configuration for the image resizer.
            resize_factor (float): Factor by which to resize input images before projection.
            n_jobs (int): Number of parallel jobs to use.
        """
        self.resizer_cfg = resizer_cfg or ResizerConfig(resize_factor=resize_factor)
        self.n_jobs = n_jobs

    def update(self, **kwargs: Any) -> None:
        """
        Update configuration using keyword arguments.

        Args:
            **kwargs (Any): Dictionary of attributes to update.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class ProjectionPipeline:
    """
    Manages sampling and projection strategies using modular configuration objects.
    Stacks all data channels into one multi-channel array for forward/backward operations,
    automatically un-stacks after backward if input was PipelineData.
    """

    def __init__(
        self,
        projection_name: str,
        sampler_name: Optional[str] = None,
        blender_name: Optional[str] = "ClosestBlender",
        pipeline_cfg: Optional[PipelineConfig] = None,
    ) -> None:
        """
        Initialize the ProjectionPipeline.

        Args:
            projection_name (str): Name of the projection to be used.
            sampler_name (Optional[str]): Name of the sampler to be used. If None, pipeline only does single projections.
            blender_name (Optional[str]): Name of the blender to be used. If None, pipeline only does single projections.
            pipeline_cfg (Optional[PipelineConfig]): Pipeline configuration object.
        """
        if projection_name is None:
            raise ValueError(
                "A 'projection_name' must be specified when creating a ProjectionPipeline instance. "
                f"Available options: {ProjectionRegistry.list_projections()}."
            )

        self.pipeline_cfg = pipeline_cfg or PipelineConfig(resize_factor=1.0)
        self.projection_name = projection_name
        self.sampler_name = sampler_name
        self.blender_name = blender_name 

        self.sampler: Optional[Sampler] = None
        if self.sampler_name:
            # Retrieve the sampler instance
            self.sampler = SamplerRegistry.get_sampler(sampler_name)

        # Retrieve the projector
        self.projector = ProjectionRegistry.get_projection(projection_name, return_processor=True)
        
        # Create the resizer
        self.resizer = self.pipeline_cfg.resizer_cfg.create_resizer()

        # Parallel jobs
        self.n_jobs = self.pipeline_cfg.n_jobs

        # Internal references for un-stacking after backward
        self._original_data: Optional[PipelineData] = None
        self._keys_order: Optional[List[str]] = None
        self._stacked_shape: Optional[Tuple[int, int, int]] = None

        # Retrieve blender instance
        self.blender = BlenderRegistry.get_blender(self.blender_name)

    @classmethod
    def list_samplers(cls) -> List[str]:
        """
        List all registered samplers.

        Returns:
            List[str]: Names of available samplers.
        """
        return SamplerRegistry.list_samplers()

    @classmethod
    def list_blenders(cls) -> List[str]:
        """
        List all registered blending strategies.

        Returns:
            List[str]: Names of available blending strategies.
        """
                
        return BlenderRegistry.list_blenders()

    @classmethod
    def list_projections(cls) -> List[str]:
        """
        List all registered projections.

        Returns:
            List[str]: Names of available projections.
        """
        return ProjectionRegistry.list_projections()

    def __repr__(self) -> str:
        projection_config = self.projector.config.config_object.config.model_dump()
        sampler_config = self.sampler.params if self.sampler else {"default": "No sampler selected"}
        blender_config = self.blender.params if self.blender else {"default": "No sampler selected"}

        projection_config_str = "\n".join(f"{key}: {value}" for key, value in projection_config.items())
        sampler_config_str = (
            "\n".join(f"{key}: {value}" for key, value in sampler_config.items()) if sampler_config else "    No parameters"
        )
        blender_config_str = (
            "\n".join(f"{key}: {value}" for key, value in blender_config.items()) if blender_config else "    No parameters"
        )

        return f"""
{self.projection_name.capitalize()} Projection - Configurations:
{projection_config_str}

{self.sampler_name} Sampler - Configurations:
{sampler_config_str}

{self.blender_name} Blender - Configurations:
{blender_config_str}

Note: You can pass any updates to these configurations via kwargs.
"""

    def update(self, **kwargs: Any) -> None:
        """
        Update the pipeline configuration, projector config, and sampler parameters.

        Args:
            **kwargs (Any): Arbitrary keyword arguments that override existing configurations.
        """
        self.projector.config.update(**kwargs)
        if self.sampler:
            self.sampler.update(**kwargs)
        if self.blender:
            self.blender.update(**kwargs)
        self.pipeline_cfg.update(**kwargs)
        self.resizer = self.pipeline_cfg.resizer_cfg.create_resizer()
        self.n_jobs = self.pipeline_cfg.n_jobs

    def _resize_image(self, img: np.ndarray, upsample: bool = True) -> np.ndarray:
        """
        Resize the input image using the ImageResizer.

        Args:
            img (np.ndarray): Input image.
            upsample (bool): Whether to apply upsampling or not.

        Returns:
            np.ndarray: Resized image.
        """
        return self.resizer.resize_image(img, upsample)

    def _prepare_data(self, data: Union[PipelineData, np.ndarray]) -> Tuple[np.ndarray, Optional[List[str]]]:
        """
        Prepare the data for processing. If it's PipelineData, stack all channels; if it's a NumPy array, use as is.

        Args:
            data (Union[PipelineData, np.ndarray]): The input data.

        Returns:
            Tuple[np.ndarray, Optional[List[str]]]: (stacked_array, keys_order_if_any).
        """
        if isinstance(data, PipelineData):
            stacked, keys_order = data.stack_all()
            self._original_data = data
            self._keys_order = keys_order
            return stacked, keys_order
        elif isinstance(data, np.ndarray):
            self._original_data = None
            self._keys_order = None
            return data, None
        else:
            raise TypeError("Data must be either PipelineData or np.ndarray.")

    def project_with_sampler(
        self,
        data: Union[PipelineData, np.ndarray],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Forward projection on a single stacked array for all tangent points (from the sampler).

        Args:
            data (Union[PipelineData, np.ndarray]): Input data for projection.
            **kwargs (Any): Additional overrides for projector or sampler.

        Returns:
            Dict[str, Any]: Dictionary with key "stacked" containing tangent-point projections.
                            If original data was PipelineData, also includes unstacked versions.
        """
        if not self.sampler:
            raise ValueError("Sampler is not set. Provide 'sampler_name' or use single_projection().")

        # Update pipeline with any additional kwargs
        self.update(**kwargs)

        tangent_points = self.sampler.get_tangent_points()
        prepared_data, _ = self._prepare_data(data)

        if isinstance(prepared_data, np.ndarray):
            self._stacked_shape = prepared_data.shape

        projections: Dict[str, Any] = {"stacked": {}}

        for idx, (lat_deg, lon_deg) in enumerate(tangent_points, start=1):
            lat = deg_to_rad(lat_deg)
            lon = deg_to_rad(lon_deg)
            logger.debug(f"Forward projecting for point {idx}, lat={lat_deg}, lon={lon_deg}.")

            # Update projector config for each tangent point
            self.projector.config.update(phi1_deg=rad_to_deg(lat), lam0_deg=rad_to_deg(lon))

            out_img = self.projector.forward(prepared_data)
            projections["stacked"][f"point_{idx}"] = out_img

            if self._original_data:
                # Also provide unstacked version
                unstacked = self._original_data.unstack_new_instance(out_img, self._keys_order).as_dict()
                projections[f"point_{idx}"] = unstacked

        return projections

    def single_projection(
        self,
        data: Union[PipelineData, np.ndarray],
        **kwargs: Any
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Single forward projection without using a sampler.

        Args:
            data (Union[PipelineData, np.ndarray]): Input data for projection.
            **kwargs (Any): Additional overrides for the projector config.

        Returns:
            Union[np.ndarray, Dict[str, Any]]: Projected image (stacked array), or if input was PipelineData,
                                              a dict with both "stacked" and unstacked components.
        """
        self.update(**kwargs)

        prepared_data, _ = self._prepare_data(data)
        if isinstance(prepared_data, np.ndarray):
            self._stacked_shape = prepared_data.shape

        out_img = self.projector.forward(prepared_data)

        if self._original_data:
            unstacked = self._original_data.unstack_new_instance(out_img, self._keys_order)
            output = {'stacked': out_img}
            output.update(unstacked.as_dict())
            return output
        else:
            return out_img


    def backward_with_sampler(
        self,
        rect_data: Dict[str, Any],
        img_shape: Optional[Tuple[int, int, int]] = None,
        **kwargs: Any
    ) -> Dict[str, np.ndarray]:
        """
        Handles backward projection and blends multiple equirectangular images into one
        using feathered blending, applied separately per data type (RGB, depth, etc.).

        Args:
            rect_data (Dict[str, Any]): Dictionary containing "stacked" key with tangent-point images.
            img_shape (Optional[Tuple[int,int,int]]): Desired final shape. Overridden if pipeline had a forward pass.
            **kwargs (Any): Additional overrides for the projector config.

        Returns:
            Dict[str, np.ndarray]: Dictionary with blended outputs for each data type.
        """
        self.update(**kwargs)

        if not self.sampler:
            raise ValueError("Sampler is not set. Provide 'sampler_name' or use single_backward().")

        # Override img_shape with the shape from the forward pass if available
        if self._stacked_shape is not None:
            if img_shape is not None and img_shape != self._stacked_shape:
                logger.warning(
                    f"Overriding user-supplied img_shape={img_shape} with stacked_shape={self._stacked_shape} "
                    "to ensure consistent channel dimensions."
                )
            img_shape = self._stacked_shape

        if img_shape is None:
            raise ValueError("img_shape must be provided if no prior forward shape is available.")

        tangent_points = self.sampler.get_tangent_points()

        stacked_dict = rect_data.get("stacked")
        if stacked_dict is None:
            raise ValueError("rect_data must have a 'stacked' key with tangent-point images.")

        # Update projector config for final shape
        self.projector.config.update(
            lon_points=img_shape[1],
            lat_points=img_shape[0]
        )

        # Unstack data before processing
        unstacked_data = {}
        for idx, (lat_deg, lon_deg) in enumerate(tangent_points, start=1):
            stacked_img = stacked_dict.get(f"point_{idx}")
            if stacked_img is None:
                raise ValueError(f"Missing 'point_{idx}' in rect_data['stacked'].")

            if self._original_data is None:
                raise ValueError("Original data structure is required to unstack.")

            # Unstack the stacked image into separate data types
            unstacked_data[f"point_{idx}"] = self._original_data.unstack_all(stacked_img, self._keys_order)

        blended_results = {}

        # Perform separate backward projection & blending for each data type
        for data_type in self._keys_order:
            images = []
            masks = []

            for idx, (lat_deg, lon_deg) in enumerate(tangent_points, start=1):
                if f"point_{idx}" not in unstacked_data:
                    raise ValueError(f"Missing 'point_{idx}' in unstacked data.")

                rect_img = unstacked_data[f"point_{idx}"].get(data_type)
                if rect_img is None:
                    raise ValueError(f"Missing '{data_type}' for 'point_{idx}' in rect_data.")

                # Perform backward projection for each data type separately
                self.projector.config.update(phi1_deg=lat_deg, lam0_deg=lon_deg)
                equirect_img, mask = self.projector.backward(rect_img, return_mask=True)

                images.append(equirect_img)
                masks.append(mask)
            
            self._backward_cache = {'images': images, 'masks': masks}
            # Blend the images of the same type separately
            self.blender.update(**{
                "projector": self.projector,
                "tangent_points": tangent_points
            })
            blended_results[data_type] = self.blender.blend(images, masks)

        # If original data exists, return in the PipelineData structure
        if self._original_data is not None:
            new_data = PipelineData.from_dict(blended_results)
            output: Dict[str, Any] = new_data.as_dict()
            return output
        else:
            return blended_results


    def _backward_with_sampler(
        self,
        rect_data: Dict[str, Any],
        img_shape: Optional[Tuple[int, int, int]] = None,
        **kwargs: Any
    ) -> Dict[str, np.ndarray]:
        """
        Handles backward projection and blends multiple equirectangular images into one
        using feathered blending to reduce visible edges.

        Args:
            rect_data (Dict[str, Any]): Dictionary containing "stacked" key with tangent-point images.
            img_shape (Optional[Tuple[int,int,int]]): Desired final shape. Overridden if pipeline had a forward pass.
            **kwargs (Any): Additional overrides for the projector config.

        Returns:
            Dict[str, np.ndarray]: Dictionary with key "stacked" for the blended equirectangular output,
                                   plus unstacked components if original data was used.
        """
        self.update(**kwargs)

        if not self.sampler:
            raise ValueError("Sampler is not set. Provide 'sampler_name' or use single_backward().")

        # Override img_shape with the shape from the forward pass if available
        if self._stacked_shape is not None:
            if img_shape is not None and img_shape != self._stacked_shape:
                logger.warning(
                    f"Overriding user-supplied img_shape={img_shape} with stacked_shape={self._stacked_shape} "
                    "to ensure consistent channel dimensions."
                )
            img_shape = self._stacked_shape

        if img_shape is None:
            raise ValueError("img_shape must be provided if no prior forward shape is available.")

        tangent_points = self.sampler.get_tangent_points()

        stacked_dict = rect_data.get("stacked")
        if stacked_dict is None:
            raise ValueError("rect_data must have a 'stacked' key with tangent-point images.")

        # Update projector config for final shape
        self.projector.config.update(
            lon_points=img_shape[1],
            lat_points=img_shape[0]
        )

        tasks = []
        for idx, (lat_deg, lon_deg) in enumerate(tangent_points, start=1):
            rect_img = stacked_dict.get(f"point_{idx}")
            if rect_img is None:
                raise ValueError(f"Missing 'point_{idx}' in rect_data['stacked'].")

            if rect_img.shape[-1] != img_shape[-1]:
                raise ValueError(
                    f"rect_img for point_{idx} has {rect_img.shape[-1]} channels, "
                    f"but final shape indicates {img_shape[-1]} channels. Check your data."
                )

            tasks.append((idx, lat_deg, lon_deg, rect_img))

        def _backward_task(
            idx: int,
            lat_deg_: float,
            lon_deg_: float,
            rect_img_: np.ndarray
        ) -> Tuple[int, np.ndarray, np.ndarray]:
            """
            Worker function for parallel backward projection.
            """
            logger.debug(f"[Parallel] Backward projecting point_{idx}, lat={lat_deg_}, lon={lon_deg_}...")

            self.projector.config.update(
                phi1_deg=lat_deg_,
                lam0_deg=lon_deg_,
            )
            equirect_img, mask = self.projector.backward(rect_img_, return_mask=True)
            return idx, equirect_img, mask

        logger.info(f"Starting backward with n_jobs={self.n_jobs} on {len(tasks)} tasks.")
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_backward_task)(*task) for task in tasks
        )
        logger.info("All backward tasks completed.")

        idxs, images, masks = zip(*results)
        self._cached_images_masks = {'images': images, 'idxs': idxs, 'masks': masks, 'tangent_points': tangent_points}

        # Apply blending using the registered blender
        self.blender.update(**{
            'projector': self.projector,
            'tangent_points': tangent_points

        })
        
        combined = self.blender.blend(images, masks,)

        if self._original_data is not None and self._keys_order is not None:
            new_data = self._original_data.unstack_new_instance(combined, self._keys_order)
            output: Dict[str, Any] = {"stacked": combined}
            output.update(new_data.as_dict())
            return output
        else:
            return {"stacked": combined}

    def single_backward(
        self,
        rect_data: Union[np.ndarray, Dict[str, Any]],
        img_shape: Optional[Tuple[int, int, int]] = None,
        **kwargs: Any
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Perform a single backward projection without sampler-based blending.

        Args:
            rect_data (Union[np.ndarray, Dict[str, Any]]): Either a NumPy array or a dict with a 'stacked' key.
            img_shape (Optional[Tuple[int,int,int]]): Shape for the output (overridden if pipeline had a forward pass).
            **kwargs (Any): Additional overrides for the projector config.

        Returns:
            Union[np.ndarray, Dict[str, Any]]: Backprojected image (stacked array), or dict with unstacked components if original data was used.
        """
        self.update(**kwargs)

        if self._stacked_shape is not None and img_shape != self._stacked_shape:
            logger.warning(
                f"Overriding user-supplied img_shape={img_shape} with stacked_shape={self._stacked_shape} "
                "for single_backward."
            )
            img_shape = self._stacked_shape

        # If rect_data is directly a NumPy array
        if isinstance(rect_data, np.ndarray):
            out_img, _ = self.projector.backward(rect_data, return_mask=True)
            if self._original_data and self._keys_order:
                new_data = self._original_data.unstack_new_instance(out_img, self._keys_order)
                return new_data.as_dict()
            else:
                return out_img

        # If rect_data is a dictionary
        stacked_arr = rect_data.get("stacked")
        if stacked_arr is None:
            raise ValueError("Expecting key 'stacked' in rect_data for single_backward.")

        if img_shape and stacked_arr.shape[-1] != img_shape[-1]:
            raise ValueError(
                f"Stacked array has {stacked_arr.shape[-1]} channels, but final shape indicates {img_shape[-1]} channels."
            )

        out_img, _ = self.projector.backward(stacked_arr, return_mask=True)
        if self._original_data and self._keys_order:
            new_data = self._original_data.unstack_new_instance(out_img, self._keys_order)
            return new_data.as_dict()
        else:
            return out_img

    def project(self, data: Union[PipelineData, np.ndarray], **kwargs: Any) -> Dict[str, Any]:
        """
        Top-level forward projection interface. Chooses sampler-based or single projection.

        Args:
            data (Union[PipelineData, np.ndarray]): Input data for projection.
            **kwargs (Any): Additional overrides.

        Returns:
            Dict[str, Any]: A dictionary containing projection results.
        """
        if isinstance(data, PipelineData):
            img_shape = data.H, data.W
        elif isinstance(data, np.ndarray):
            img_shape = data.shape[:2]
        else:
            raise Exception("Input data must be an instance of PipelineData or a Numpy Array")
        
        shape = {'lon_points': img_shape[0], 'lat_points': img_shape[1]}
        logger.debug(f"Updating x_points and y_points to {shape}.")
        self.update(**shape)
       
        if self.sampler:
            return self.project_with_sampler(data, **kwargs)
        else:
            out = self.single_projection(data, **kwargs)
            if isinstance(out, dict):
                return out
            return {"stacked": out}

    def backward(
        self,
        data: Union[Dict[str, Any], np.ndarray],
        img_shape: Optional[Tuple[int, int, int]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Top-level backward projection interface. Chooses sampler-based or single backward approach.

        Args:
            data (Union[Dict[str, Any], np.ndarray]): Equirectangular or rectified input data.
            img_shape (Optional[Tuple[int,int,int]]): Desired output shape (overridden if pipeline had a forward pass).
            **kwargs (Any): Additional overrides.

        Returns:
            Dict[str, Any]: A dictionary containing backward projection (blended) results.
        """
        if self.sampler:
            return self.backward_with_sampler(data, img_shape=img_shape, **kwargs)
        else:
            out = self.single_backward(data, img_shape=img_shape, **kwargs)
            if isinstance(out, dict):
                return out
            return {"stacked": out}


class Pipeline(ProjectionPipeline):

    @classmethod
    def get_inference_results(cls, faces, model_fn):
        new_faces = deepcopy(faces)
        for k in new_faces['stacked'].keys():
            assert new_faces['stacked'][k].shape[-1] == 3
            new_faces['stacked'][k] = new_faces['stacked'][k].astype(np.float32)
            depthmap = model_fn(new_faces['stacked'][k])
            new_faces['stacked'][k][:, :, 0] = depthmap
            new_faces['stacked'][k][:, :, 1] = depthmap
            new_faces['stacked'][k][:, :, 2] = depthmap
        return new_faces

    def inference(self, data, model_fn, **kwargs):
        faces = self.project(data, **kwargs)
        infered_faces = Pipeline.get_inference_results(
            faces,
            model_fn=model_fn
        )

        interpolation = kwargs.get("interpolation", cv2.INTER_NEAREST)
        if "interpolation" in kwargs:
            kwargs.pop("interpolation")
        borderMode = kwargs.get("borderMode", cv2.BORDER_CONSTANT)
        if "borderMode" in kwargs:
            kwargs.pop("borderMode")

        equirect_inference_result = self.backward(infered_faces,
                                                  interpolation=interpolation,
                                                  borderMode=borderMode,
                                                  **kwargs
                                                  )
        
        return {'inference_result': equirect_inference_result['rgb']}
