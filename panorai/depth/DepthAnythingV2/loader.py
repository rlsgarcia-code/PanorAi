"""
dav2.py
=======

A lazy loader for DepthAnythingV2. No top-level Torch or device init.
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from ..registry import ModelRegistry
from panorai.path_config import get_path


# Example: local import if you placed Depth-Anything-V2 code in a subfolder
# from .depth_anything_v2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
# But in your snippet, you had:
#import sys
#sys.path.insert(1,'./panorai_models/Depth-Anything-V2')

from .metric_depth.depth_anything_v2.dpt import DepthAnythingV2

def _rgb_to_bgr(rgb_image: np.ndarray) -> np.ndarray:
    # Reverses channels to get BGR from RGB
    return rgb_image[..., ::-1]


@ModelRegistry.register("dav2", default_args={
    "checkpoint_path": None,
    "encoder": "vits",
    "dataset": "vkitti",  # hypersim
    "max_depth": 80,
    "device": "mps",
    "return_model": False
})
def load_dav2_model(
    checkpoint_path,
    encoder,
    dataset,
    max_depth,
    device,
    return_model
):
    """
    Builds DepthAnythingV2 with provided config, loads weights, moves to device.
    Returns an inference(filename) function that returns a numpy depth map.
    """

    # Example config
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }

    #max_depth = 80 if dataset == 'vkitti' else 20 # 20 for indoor model, 80 for outdoor model

    cfg = {**model_configs[encoder], 'max_depth': max_depth}
    print(cfg)

    if not checkpoint_path:
        root = get_path("depth_anything_v2", "checkpoint_dir")
        if root:
            root = Path(root)
            if not dataset:
                checkpoint_path = str(root / f"depth_anything_v2_{encoder}.pth")
            else:
                ckpt_name = f"depth_anything_v2_metric_{dataset}_{encoder}.pth"
                checkpoint_path = str(root / ckpt_name)
        else:
            raise FileNotFoundError(
                "Checkpoint path not provided and 'depth_anything_v2.checkpoint_dir' not set in paths.yaml"
            )
    
    model = DepthAnythingV2(**cfg)
    print(f"[dav2] Loading from {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)

    #model.load_state_dict({k: v for k, v in torch.load(checkpoint_path).items() if 'pretrained' in k}, strict=False)

    model = model.to(device).eval()
    if return_model:
        
        return model

    def inference(arr: np.ndarray) -> np.ndarray:
        rgb_image = arr
        bgr_image = _rgb_to_bgr(rgb_image)
        depth_map = model.infer_image(bgr_image)  # (H, W) depth
        return depth_map

    return inference