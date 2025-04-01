# panorai/projections/registry.py

class ProjectionRegistry:
    _projections = {}

    @classmethod
    def register(cls, name: str):
        def decorator(projection_cls):
            if name in cls._projections:
                raise KeyError(f"Projection '{name}' already registered.")
            cls._projections[name] = projection_cls
            return projection_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs):
        if name not in cls._projections:
            raise KeyError(f"Projection '{name}' not found. Available: {list(cls._projections.keys())}")
        return cls._projections[name](**kwargs)

    @classmethod
    def available_projections(cls) -> list:
        return list(cls._projections.keys())