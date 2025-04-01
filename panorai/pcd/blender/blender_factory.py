# pcd/blender/blender_factory.py

from .simple_equirect import SimpleEquirectBlender
from .consensus_blender import EquirectangularConsensusBlender
from .kdtree_blender import KDTreeBlender
from .voxel_blender import VoxelBlender
from .scale_ba_blender import ScaleBundleAdjustmentBlender


class PCDBlenderFactory:
    """
    Provides a single entry point for creating the five 
    main blender strategies by name.
    """

    @staticmethod
    def get_blender(name: str, **kwargs):
        name = name.lower()
        if name == "simple":
            return SimpleEquirectBlender(**kwargs)
        elif name == "consensus":
            return EquirectangularConsensusBlender(**kwargs)
        elif name == "kdtree":
            return KDTreeBlender(**kwargs)
        elif name == "voxel":
            return VoxelBlender(**kwargs)
        elif name == "scale_ba":
            return ScaleBundleAdjustmentBlender(**kwargs)
        else:
            raise ValueError(f"Unknown blender type: {name}")