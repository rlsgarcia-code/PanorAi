"""Application wrapper owning a concrete, inspectable Pipeline."""

from ._models import INTERFACE, PlanarReconstructionOptions
from ._stages import (
    ValidatePlanarScene,
    SegmentPlaneRegions,
    MatchPlaneRegions,
    TriangulatePlaneSupport,
    FitPlanes,
    PropagatePlaneRegions,
    RasterizePlaneRanges,
)


class HumanGuidedPlanarReconstruction:
    interface = INTERFACE
    stability = "experimental"

    def __init__(self, options=None, *, segmentation="manual", sam_backend=None):
        from panorai.pipeline import Pipeline

        self.options = options or PlanarReconstructionOptions()
        self.pipeline = Pipeline(
            ("validate", ValidatePlanarScene()),
            ("segment", SegmentPlaneRegions(segmentation, sam_backend=sam_backend)),
            ("match", MatchPlaneRegions(self.options)),
            ("triangulate", TriangulatePlaneSupport(self.options)),
            ("fit", FitPlanes(self.options)),
            ("propagate", PropagatePlaneRegions(self.options)),
            ("range", RasterizePlaneRanges(self.options)),
        )

    def __call__(self, scene):
        return self.pipeline(scene)

    def run(self, scene, *, retain=()):
        return self.pipeline.run(scene, retain=retain)

    @property
    def named_steps(self):
        return self.pipeline.named_steps

    def get_params(self, *, deep=True):
        return self.pipeline.get_params(deep=deep)

    def with_params(self, **params):
        result = object.__new__(type(self))
        result.options = self.options
        result.pipeline = self.pipeline.with_params(**params)
        return result

    def describe(self):
        return {
            **self.pipeline.describe(),
            "interface": self.interface,
            "stability": self.stability,
            "review": "immutable input revision; segmentation and propagation produce drafts",
            "uncertainty": "fixed-pose linearization, excludes systematic errors",
        }
