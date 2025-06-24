import sys
import logging
from pathlib import Path
from PIL import Image
import numpy as np
import os

import torch
from .registry import ModelRegistry


@ModelRegistry.register("zoe", default_args={
    "return_model": False
})
def load_zoe_model(return_model: bool = False,
                   **kwargs):
    """
    Dynamically loads ZoeDepth model using HuggingFace Transformers.
    """
    try:
        from transformers import AutoImageProcessor, ZoeDepthForDepthEstimation
        import torch
    except Exception as e:
        raise ValueError(e)

    try:
        image_processor = AutoImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti")
        model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti")
    except Exception as e:
        raise ValueError(e)
    
    #image_processor = AutoImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti")
    model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti")
    
    if return_model:
        return model
    

    class ZoeModel:
        _image_processor = image_processor
        _model = model

        @staticmethod
        def inference(arr: np.ndarray):
            image = Image.fromarray(arr)
            inputs = ZoeModel._image_processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = ZoeModel._model(**inputs)

            post_processed = ZoeModel._image_processor.post_process_depth_estimation(
                outputs,
                source_sizes=[(image.height, image.width)],
            )
            return post_processed[0]["predicted_depth"].detach().cpu().numpy()

    def fn(image_array: np.ndarray) -> np.ndarray:
        """
        Run inference on an input image array.

        Args:
            image_array (np.ndarray): RGB image array (H, W, 3)

        Returns:
            np.ndarray: Depth map (H, W)
        """
        try:
            depth = ZoeModel.inference(image_array)

            return depth
        except Exception as e:

            raise

    return fn

