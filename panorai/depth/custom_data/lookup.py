def data_lookup(dataset, key):
    idx = int(key.split('_')[0])
    return dataset._load_sample(idx, apply_transform=False)