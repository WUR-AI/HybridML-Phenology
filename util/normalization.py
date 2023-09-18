import numpy as np

import torch


def normalize_latitude(lat, revert=False):
    if isinstance(lat, torch.Tensor):
        return torch.deg2rad(lat) if not revert else torch.rad2deg(lat)
    else:
        return np.radians(lat) if not revert else np.degrees(lat)
    # # SCALE = 60
    # SCALE = 5
    # SHIFT = 35
    # return lat * SCALE + SHIFT if revert else (lat - SHIFT) / SCALE  # latitude scale from -90 to 90


def normalize_longitude(lon, revert=False):
    if isinstance(lon, torch.Tensor):
        return torch.deg2rad(lon) / 2 if not revert else torch.rad2deg(lon) * 2
    else:
        return np.radians(lon) / 2 if not revert else np.degrees(lon) * 2
    # # SCALE = 120
    #
    # # SCALE = 4.5
    # # SHIFT = 135
    #
    # SCALE = 60
    # SHIFT = 70
    # return lon * SCALE + SHIFT if revert else (lon - SHIFT) / SCALE  # longitude ranges from -180 to 180


def normalize_altitude(alt, revert=False) -> float:  # TODO
    # SCALE = 1000  # 1500

    SCALE = 370
    SHIFT = 370
    return alt * SCALE + SHIFT if revert else (alt - SHIFT) / SCALE
    # return alt * SCALE if revert else alt / SCALE  # Based on max value found in dataset


def min_max_normalize(data, vmin: float, vmax: float, revert: bool = False):
    if not revert:
        return (data - vmin) / (vmax - vmin)
    else:
        return data * (vmax - vmin) + vmin


def mean_std_normalize(data, mean: float, std: float, revert: bool = False):
    if not revert:
        return (data - mean) / std
    else:
        return data * std + mean

