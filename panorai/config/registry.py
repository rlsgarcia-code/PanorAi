# panorai/config/registry.py

class ConfigRegistry:
    """Central registry for projection configurations."""
    _configs = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register configuration classes.
        """
        def decorator(config_cls):
            if name in cls._configs:
                raise KeyError(f"Configuration '{name}' already registered.")
            cls._configs[name] = config_cls
            return config_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs):
        """
        Create an instance of a registered configuration.
        """
        if name not in cls._configs:
            raise KeyError(f"Configuration '{name}' not found. Available: {list(cls._configs.keys())}")
        return cls._configs[name](**kwargs)

    @classmethod
    def available_configs(cls):
        """List all registered configuration names."""
        return list(cls._configs.keys())