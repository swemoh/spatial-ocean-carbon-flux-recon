import os
from datetime import datetime
import numpy as np

def load_dataset(data_path, start_year=1958, end_year=2018, target_index=3):
    """
    Loads features and targets from a single dataset folder.
    """

    start_index = (start_year-1958)*12+1
    end_index = (end_year-1957)*12+1

    feature_files = [os.path.join(data_path, f"{i}_features.npy") for i in range(start_index, end_index)]
    target_files = [os.path.join(data_path, f"{i}_targets.npy") for i in range(start_index, end_index)]

    features = np.stack([np.load(fp) for fp in feature_files])
    targets = np.stack([np.load(fp) for fp in target_files])
    
    targets = targets[..., target_index]

    return features, targets

def load_and_interleave(start_year=1958, end_year=2018, datasets=["exp1", "exp3", "exp5"], target_index=3):
    """
    Loads multiple datasets and interleaves them in a round-robin fashion.
    """

    data_path_prefix = "../../data/"
    data_paths = [data_path_prefix + d for d in datasets]

    features_list, targets_list = zip(*(load_dataset(path, start_year=start_year, end_year=end_year, target_index=target_index  ) for path in data_paths))
    num_datasets = len(features_list)
    num_timesteps = features_list[0].shape[0]

    features = np.empty((num_timesteps * num_datasets, *features_list[0].shape[1:]), dtype=features_list[0].dtype)
    targets = np.empty((num_timesteps * num_datasets, *targets_list[0].shape[1:]), dtype=targets_list[0].dtype)

    for i, (f, t) in enumerate(zip(features_list, targets_list)):
        features[i::num_datasets] = f
        targets[i::num_datasets] = t

    return features, targets

def load_for_mlp(start_year=1958, end_year=2018, datasets=[], target_index=3):
    """
    Loads data and preprocesses it for MLP models.

    Parameters
    ----------
    start_year : int
        Starting year of the data to load (inclusive).
    end_year : int
        Ending year of the data to load (inclusive).
    datasets : list of str    
        List of dataset identifiers to load.
    target_index : int
        Index of the target variable to extract (default is 3 for 'co2flux_pre')

    Returns
    -------
    features : np.ndarray
        Feature array of shape (num_samples, num_features)
    targets : np.ndarray
        Target array of shape (num_samples, 1)
    """
    features, targets = load_and_interleave(start_year=start_year, end_year=end_year, datasets=datasets, target_index=target_index)

    features = features.reshape(-1, features.shape[-1])
    targets = targets.reshape(-1, 1)

    months = features[:, 10]
    lon = features[:, 14]

    # remove unnecessary columns: global_co2 (15), lon (14), year (12), month (10)
    features = np.delete(features, [15, 14, 12, 10], axis=1)

    # convert degrees to radians
    lon_rad = np.radians(lon)

    # sine/cosine encoding
    month_sin = np.sin(2 * np.pi * months / 12)
    month_cos = np.cos(2 * np.pi * months / 12)
    lon_sin = np.sin(lon_rad)
    lon_cos = np.cos(lon_rad)

    # append to features (or replace original lat/lon)
    features = np.column_stack([features, lon_sin, lon_cos, month_sin[:, None], month_cos[:, None]])

    return features, targets

def load_as_maps(start_year=1958, end_year=2018, datasets=[], target_index=3):
    """
    Loads data as maps.

    Parameters
    ----------
    start_year : int
        Starting year of the data to load (inclusive).
    end_year : int
        Ending year of the data to load (inclusive).
    datasets : list of str    
        List of dataset identifiers to load.
    target_index : int
        Index of the target variable to extract (default is 3 for 'co2flux_pre')
    Returns
    -------
    features : np.ndarray
        Feature array of shape (num_samples, 167, 360, num_features)
    targets : np.ndarray
        Target array of shape (num_samples, 167, 360)
    """
    features, targets = load_and_interleave(start_year=start_year, end_year=end_year, datasets=datasets, target_index=target_index)

    mask = features[..., 11] == 0 
    # setting this feature to zero for land points because it had some random values
    features[..., 5][mask] = 0 

    months = features[..., 10] 

    # remove unnecessary columns: global_co2 (15), lon (14), lat (13) year (12), month (10)
    features = np.delete(features, [15, 14, 13, 12, 10], axis=-1)

    month_sin = np.sin(2 * np.pi * months / 12)[..., None]  
    month_cos = np.cos(2 * np.pi * months / 12)[..., None]

    # stack along the last dimension
    features = np.concatenate([features, month_sin, month_cos], axis=-1)

    return features, targets