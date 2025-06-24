from functools import lru_cache

class CachedTransform:
    def __init__(self, transform_fn, maxsize=256):
        self._transform_fn = transform_fn
        self._cache = lru_cache(maxsize=maxsize)(self._wrapped)
        self._data_dict = None

    def attach_data(self, data_dict):
        """Provide a mapping from keys to raw image data."""
        self._data_dict = data_dict

    def _wrapped(self, key):
        return self._transform_fn(self._data_dict[key])

    def __call__(self, key):
        return self._cache(key)