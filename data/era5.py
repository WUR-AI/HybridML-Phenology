import os
from functools import reduce

import numpy as np
import pandas as pd

import tables

from tqdm import tqdm

import xarray as xr

from config import PATH_DATA_DIR
from data.bloom_doy import get_locations_coordinates_japan

"""

    Code for loading the ERA5 data

"""

MIN_YEAR = 1980
MAX_YEAR = 2021
YEAR_RANGE = tuple(range(MIN_YEAR, MAX_YEAR + 1))

# YEAR_RANGE = tuple(range(1940, 2021 + 1))

DATA_PATH_ERA5 = os.path.join(PATH_DATA_DIR, 'era5')

FN_DATA_STORE = 'store_era5.h5'
PATH_DATA_STORE = os.path.join(PATH_DATA_DIR, FN_DATA_STORE)
KEY_DATA_STORE = 'df_era5'


def get_data_temperature_map_in_year(year: int, unit: str = 'C') -> pd.DataFrame:
    assert year in YEAR_RANGE[1:]

    data = xr.merge([
        xr.open_dataset(os.path.join(DATA_PATH_ERA5, f'temperature_{year - 1}.nc')),
        xr.open_dataset(os.path.join(DATA_PATH_ERA5, f'temperature_{year}.nc')),
    ])

    df = data['t2m'].to_dataframe()

    df = df[['t2m']]

    if unit == 'C':
        # Convert temperature unit from Kelvin to Celsius
        df = df.map(lambda x: x - 273.15)
    elif unit == 'K':
        pass  # No conversion needed
    else:
        raise Exception('Unknown temperature unit selection for ERA5 data')

    df = df.groupby(level=['latitude', 'longitude']).resample('D', level='time').apply(list)

    return df


def _round_partial(value, resolution):
    # Thanks to https://stackoverflow.com/questions/8118679/python-rounding-by-quarter-intervals
    return round(value / resolution) * resolution


def _preprocess_era5_temperature_filtered_to_locations(unit: str = 'C') -> pd.DataFrame:
    """
    Preprocess the ERA5 data by only selecting the locations for which blooming dates are present
    """
    locations_coordinates = get_locations_coordinates_japan()

    year_iter = tqdm(YEAR_RANGE, desc='Loading ERA5 data')

    dfs = []

    for year in year_iter:

        data_year = xr.open_dataset(os.path.join(DATA_PATH_ERA5, f'temperature_{year}.nc'))

        df = data_year['t2m'].to_dataframe()

        df = df[['t2m']]

        if unit == 'C':
            # Convert temperature unit from Kelvin to Celsius
            df = df.map(lambda x: x - 273.15)
        elif unit == 'K':
            pass  # No conversion needed
        else:
            raise Exception('Unknown temperature unit selection for ERA5 data')

        dfs_locations = []

        for location in locations_coordinates.index.values:
            lat, lon = locations_coordinates.loc[location][['lat', 'long']]

            lat_rounded = _round_partial(lat, 0.25)
            lon_rounded = _round_partial(lon, 0.25)

            # A mistake was made when downloading the era5 data since some locations in okinawa are not covered by the
            # dataset. These few locations therefore have to be skipped. This does not affect the results however, since
            # these locations were excluded from the training/evaluation data anyway.
            if not 124 <= lon_rounded <= 146:   # Data collection used these limits for lat/lon
                continue
            if not 25 <= lat_rounded <= 46:
                continue

            df_temperature = df.loc[:, lat_rounded, lon_rounded]

            df_temperature = df_temperature.resample('D', level='time').apply(list)

            df_temperature.rename(columns={'t2m': location}, inplace=True)

            dfs_locations.append(df_temperature)

        df_year = pd.concat(dfs_locations, axis=1)

        dfs.append(df_year)

    df = pd.concat(dfs, axis=0)

    store = pd.HDFStore(PATH_DATA_STORE)

    store[KEY_DATA_STORE] = df
    # df = store[KEY_DATA_STORE]

    store.close()

    return df


def get_era5_temperature_data() -> pd.DataFrame:
    if os.path.exists(PATH_DATA_STORE):
        store = pd.HDFStore(PATH_DATA_STORE)
        df = store[KEY_DATA_STORE]
        store.close()
        return df
    else:
        df = _preprocess_era5_temperature_filtered_to_locations(unit='C')
        return df


if __name__ == '__main__':

    # _data = get_data_temperature_map_in_year(2000)
    #
    # print(_data)

    _data = _preprocess_era5_temperature_filtered_to_locations()

    print(_data)

    store = pd.HDFStore(PATH_DATA_STORE)

    print(store[KEY_DATA_STORE])
    store.close()
