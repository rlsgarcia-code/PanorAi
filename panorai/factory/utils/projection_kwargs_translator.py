
def translate_kwargs(**original_kwargs):
  """
  Helper function to translate kwargs keys, handling both notations.

  Args:
    original_kwargs: The dictionary of kwargs provided by the user.

  Returns:
    A dictionary with the translated kwargs, or None if an invalid key is found.
  """

  key_mapping = {
      "lat": "phi1_deg",
      "lon": "lam0_deg",
      "latitude": "phi1_deg",  # Add more synonyms as needed
      "longitude": "lam0_deg"
  }

  translated_kwargs = {}

  for key, value in original_kwargs.items():
    if key in key_mapping:
      print(f'[Projection __init__] translating from {key}={value} -> {key_mapping[key]}={value}')
      translated_kwargs[key_mapping[key]] = value
    else:
      continue

  return translated_kwargs

