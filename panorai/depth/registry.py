# panorai/depth/registry.py

class ModelRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str, default_args: dict = None):
        def decorator(loader_func):
            if name in cls._registry:
                print(f"Model '{name}' is already registered.")
                pass
            try:
                cls._registry[name] = {
                    "loader_func": loader_func,
                    "default_args": default_args or {},
                }
                return loader_func
            except Exception as e:
                raise ValueError(e)
        return decorator

    @classmethod
    def load(cls, name: str, **overrides):
        if name not in cls._registry:
            raise ValueError(f"Model '{name}' is not registered.")
        entry = cls._registry[name]
        
        config = {**entry["default_args"], **overrides}
        return entry["loader_func"](**config)

    @classmethod
    def list_models(cls):
        return list(cls._registry.keys())

    @classmethod
    def get_config(cls, name: str):
        if name not in cls._registry:
            raise ValueError(f"Model '{name}' is not registered.")
        return cls._registry[name]["default_args"]