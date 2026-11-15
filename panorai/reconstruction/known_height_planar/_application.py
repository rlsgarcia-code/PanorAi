"""Application wrapper for the known-height planar mapping pipeline."""

from __future__ import annotations

from ._models import INTERFACE, KnownHeightPlanarOptions
from ._stages import (
    AnchorFloorPlane,
    AssembleKnownHeightPlanarMap,
    RasterizeKnownHeightRanges,
    ResolveConnectedPlanes,
    ResolvePlanarPreviews,
    ResolvePlanarReferenceFrame,
    SegmentKnownHeightRegions,
    ValidateKnownHeightPlanarScene,
)


class KnownHeightPlanarMapping:
    """Map reviewed planar regions from one ERP and a metric floor anchor."""

    interface = INTERFACE
    stability = "experimental"

    def __init__(self, options=None, *, sam_backend=None):
        from panorai.pipeline import Pipeline

        self.options = options or KnownHeightPlanarOptions()
        self.pipeline = Pipeline(
            ("validate", ValidateKnownHeightPlanarScene()),
            ("reference_frame", ResolvePlanarReferenceFrame(self.options)),
            (
                "segment",
                SegmentKnownHeightRegions(self.options, sam_backend=sam_backend),
            ),
            ("anchor_floor", AnchorFloorPlane(self.options)),
            ("resolve_connected", ResolveConnectedPlanes(self.options)),
            ("preview", ResolvePlanarPreviews(self.options)),
            ("rasterize", RasterizeKnownHeightRanges(self.options)),
            ("assemble", AssembleKnownHeightPlanarMap()),
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
            "frame": "+Y up; camera centre [0, camera_height_m, 0]",
            "review": "draft masks and proposed constraints never publish geometry",
            "uncertainty": "height-linearized when camera_height_sigma_m is supplied",
        }
