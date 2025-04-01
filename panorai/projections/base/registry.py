from typing import Any, Dict, Type
import logging

logger = logging.getLogger('spherical_projections.registry')

class RegistryBase(type):
    """
    Metaclass to automatically register classes.
    """
    REGISTRY: Dict[str, Type] = {}

    def __new__(cls, name, bases, attrs):
        new_cls = type.__new__(cls, name, bases, attrs)
        cls.REGISTRY[new_cls.__name__] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls) -> Dict[str, Type]:
        return dict(cls.REGISTRY)

class BaseRegisteredClass(metaclass=RegistryBase):
    """Base class for automatically registered classes."""
    pass