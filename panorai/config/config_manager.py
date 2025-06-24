import json
import yaml
import os
import logging
import importlib
from typing import Dict, Any, Optional, List
from .registry import ConfigRegistry

logger = logging.getLogger("panorai.config.manager")

class ConfigManager:
    """
    :no-index:
    High-level manager for handling projection configurations in PanorAi.

    - Dynamically discovers and registers configuration objects.
    - Caches instantiated configs for reuse.
    - Provides methods to create, retrieve, update, reset, and describe configs.

    Example:
        from panorai.config.config_manager import ConfigManager
        
        # Create a new config
        gn_config = ConfigManager.create("gnomonic_config", fov_deg=90)

        # Modify an existing config
        ConfigManager.modify_config("gnomonic_config", fov_deg=75)

        # Retrieve all available configurations
        configs = ConfigManager.get_all_configs()

        # Describe a config in YAML format
        ConfigManager.describe_config("gnomonic_config", output_format="yaml")
    """

    _instances: Dict[str, Any] = {}
    _loaded_configs: bool = False

    @classmethod
    def _auto_discover_configs(cls):
        """
        Automatically discovers and registers all available configurations.
        """
        if cls._loaded_configs:
            return

        logger.info("Auto-discovering registered configurations...")

        config_modules = [
            #"panorai.projections.gnomonic.config",
            "panorai.pipelines.sampler.config",
            "panorai.preprocessing.config"
        ]

        for module in config_modules:
            try:
                importlib.import_module(module)
                logger.info(f"Loaded config module: {module}")
            except ModuleNotFoundError:
                logger.warning(f"Config module {module} not found, skipping.")

        cls._loaded_configs = True

    @classmethod
    def create(cls, name: str, **kwargs) -> Any:
        """
        Create a new configuration or retrieve an existing one.
        """
        cls._auto_discover_configs()
        if name in cls._instances:
            return cls._instances[name]

        cls._instances[name] = ConfigRegistry.create(name, **kwargs)
        return cls._instances[name]

    @classmethod
    def get(cls, name: str) -> Any:
        """
        Retrieve an existing configuration instance, or create it if it exists in the registry.
        """
        cls._auto_discover_configs()
        if name not in cls._instances:
            return cls.create(name)
        return cls._instances[name]

    @classmethod
    def modify_config(cls, name: str, **kwargs):
        """
        Modify an existing configuration dynamically.
        """
        if name not in cls._instances:
            raise KeyError(f"Cannot modify '{name}': Configuration not found.")

        config_obj = cls._instances[name]

        if hasattr(config_obj, "update"):
            config_obj.update(kwargs)
        elif isinstance(config_obj, dict):
            config_obj.update(kwargs)
        else:
            raise TypeError(f"Config '{name}' is not modifiable.")

        logger.info(f"Updated config '{name}' with {kwargs}")
        return config_obj

    @classmethod
    def get_config_parameters(cls, name: str) -> Dict[str, Any]:
        """Return a dictionary of public parameters for a configuration."""
        cfg = cls.get(name)
        if isinstance(cfg, dict):
            return dict(cfg)
        if hasattr(cfg, "__dict__"):
            return {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
        raise TypeError(f"Config '{name}' has no accessible parameters")

    @classmethod
    def describe_config(cls, name: str, output_format: str = "pretty") -> None:
        """
        Pretty-print configuration details.
        """
        try:
            params = cls.get_config_parameters(name)
            if output_format == "json":
                print(json.dumps(params, indent=4))
            elif output_format == "yaml":
                print(yaml.dump(params, default_flow_style=False))
            else:
                print(f"\n🔹 Configuration: {name}")
                for k, v in params.items():
                    print(f"   - {k}: {v}")
        except Exception as e:
            print(f"⚠️ Error retrieving config '{name}': {e}")

    @classmethod
    def reset(cls, name: Optional[str] = None):
        """
        Reset (delete) configuration instances.
        """
        if name:
            cls._instances.pop(name, None)
            logger.info(f"Reset configuration '{name}'.")
        else:
            cls._instances.clear()
            logger.info("Reset all configurations.")

    @classmethod
    def get_all_configs(cls) -> Dict[str, Any]:
        """
        Retrieve all instantiated configurations.
        """
        return dict(cls._instances)