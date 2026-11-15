"""Application wrapper for Experimental spherical planar discovery."""

from __future__ import annotations

from ._discovery_models import (
    PLANAR_DISCOVERY_INTERFACE,
    SphericalPlanarDiscoveryOptions,
)
from ._discovery_stages import (
    AssemblePlanarDiscoveryResult,
    BuildOverlappingGnomonicAtlas,
    DetectAndFitGnomonicLines,
    EstimateVanishingBundles,
    MergeSphericalLineObservations,
    ProposePlanarOrientations,
    SuggestVanishingBundleLines,
    ValidatePlanarDiscoveryRequest,
)


class SphericalPlanarDiscovery:
    """Discover reviewable line, vanishing and planar-orientation evidence."""

    interface = PLANAR_DISCOVERY_INTERFACE
    stability = "experimental"

    def __init__(self, options: SphericalPlanarDiscoveryOptions | None = None):
        from panorai.pipeline import Pipeline

        self.options = options or SphericalPlanarDiscoveryOptions()
        self.pipeline = Pipeline(
            ("validate", ValidatePlanarDiscoveryRequest(self.options)),
            ("atlas", BuildOverlappingGnomonicAtlas(self.options)),
            ("detect_lines", DetectAndFitGnomonicLines(self.options)),
            ("merge_lines", MergeSphericalLineObservations(self.options)),
            ("vanishing_bundles", EstimateVanishingBundles(self.options)),
            ("line_suggestions", SuggestVanishingBundleLines(self.options)),
            ("planar_orientations", ProposePlanarOrientations(self.options)),
            ("assemble", AssemblePlanarDiscoveryResult(self.options)),
        )

    def __call__(self, request):
        return self.pipeline(request)

    def run(self, request, *, retain=()):
        return self.pipeline.run(request, retain=retain)

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
            "line_fit_domain": "overlapping gnomonic charts",
            "vanishing_space": "projective sphere; v and -v are equivalent",
            "plane_output": "orientation-only; metric offset unavailable",
            "review": "automatic evidence is never accepted silently",
        }
