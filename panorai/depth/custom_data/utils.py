# custom_data/utils.py
import random
import numpy as np

import random
import numpy as np

def sample_n_points(sample, N, p_high=0.8):
    """
    Sample N points, prioritizing those with more valid_mask values but allowing some randomness.
    If not enough valid samples exist, it uses every available sample before repeating.

    Args:
        sample (dict): Dictionary with 'image', 'depth', 'valid_mask', and 'image_path'.
        N (int): Number of samples to select.
        p_high (float): Probability of selecting from top-ranked samples (default: 0.8).

    Returns:
        dict: Sampled subset of the dataset.
    """
    total_samples = len(sample['image'])
    N = min(N, total_samples)  # Ensure we don't sample more than available

    # Ensure valid_mask is an array and compute sum correctly
    valid_counts = [np.sum(sample['valid_mask'][i]) for i in range(total_samples)]

    # Rank samples by valid_mask count (descending)
    sorted_indices = sorted(range(total_samples), key=lambda i: valid_counts[i], reverse=True)

    # Define high-priority selection group
    split_idx = max(1, int(p_high * total_samples))
    top_indices = sorted_indices[:split_idx]
    low_indices = sorted_indices[split_idx:]

    # Randomly sample from high-priority group first
    selected_indices = random.sample(top_indices, min(N, len(top_indices)))

    # If not enough, fill with low-priority images
    remaining_needed = N - len(selected_indices)
    if remaining_needed > 0 and low_indices:
        selected_indices += random.sample(low_indices, min(remaining_needed, len(low_indices)))

    # If still not enough, reuse available images but ensure all unique ones are used first
    if len(selected_indices) < N:
        print("Warning: Not enough valid images. Reusing available samples to fill the batch.")
        extra_samples = (selected_indices * ((N // len(selected_indices)) + 1))[:N]
        selected_indices = extra_samples  # Fill up to N with repeats if necessary

    # Return the final sampled dataset
    return {key: [sample[key][i] for i in selected_indices] for key in sample}


from functools import lru_cache

class CachedTransform:
    def __init__(self, transform_fn, maxsize=256):
        self._transform_fn = transform_fn
        self._cache = lru_cache(maxsize=maxsize)(self._wrapped)
        self._data_dict = {}

    def attach_data(self, data_dict):
        self._data_dict = data_dict

    def _wrapped(self, key: str):
        return self._transform_fn(self._data_dict[key])

    def __call__(self, key: str):
        if not isinstance(key, str):
            raise ValueError(f"CachedTransform expects string keys, got: {type(key)}")
        return self._cache(key)