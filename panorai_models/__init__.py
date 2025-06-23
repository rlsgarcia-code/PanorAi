# panorai_models/__init__.py
from .DepthAnythingV2.loader import load_dav2_model
from .Metric3D.loader import load_m3dv2_model
from .zoe import load_zoe_model
try:
    from .Dust3r.loader import load_dust3r_model
except Exception as e:
    raise ValueError(e)



from .registry import ModelRegistry


__all__ = [
'ModelRegistry'
]