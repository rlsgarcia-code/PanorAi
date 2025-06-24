from torchvision.transforms import Compose
from panorai.depth.trainers.transforms import NormalizeImage, DeterministicProjectionTransform

def build_gnomonic_projection_transform(size=518, n_angles=8):
    # Returns a Compose pipeline that expects input in dict format.
    return ComposeWithKey([
        DeterministicProjectionTransform(size=size, n_angles=n_angles),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

class ComposeWithKey:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample_with_key):
        # For each transform, update sample_with_key["data"]
        for t in self.transforms:
            # Some transforms (like NormalizeImage) expect just the sample dict,
            # so if t has an attribute __keyless__ set to True, we pass sample_with_key["data"].
            if hasattr(t, "__keyless__") and t.__keyless__:
                sample_with_key["data"] = t(sample_with_key["data"])
            else:
                sample_with_key["data"] = t(sample_with_key)
        return sample_with_key["data"]