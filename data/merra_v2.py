import os

import numpy as np
import pandas as pd

from config import PATH_DATA_DIR

"""

    Code for loading the MERRA v2 data

"""


DATA_PATH_MERRA = os.path.join(PATH_DATA_DIR, 'merra_v2')
DATA_FILENAME_TEMPERATURE = 'daily_temperature.csv.gz'

FN_DATA_STORE = 'store_merrav2.h5'
PATH_DATA_STORE = os.path.join(PATH_DATA_DIR, FN_DATA_STORE)
KEY_DATA_STORE = 'df_merrav2'


def get_data_temperature(unit: str = 'C') -> pd.DataFrame:
    if os.path.exists(PATH_DATA_STORE):
        store = pd.HDFStore(PATH_DATA_STORE)
        df = store[KEY_DATA_STORE]
        store.close()
        return df
    else:
        df = _preprocess_merrav2_temperature_data(unit='C')
        return df

    # df = pd.read_csv(os.path.join(DATA_PATH_MERRA, DATA_FILENAME_TEMPERATURE),
    #                  compression='gzip',
    #                  index_col=0,  # Use dates as index
    #                  parse_dates=True,  # Parse them to numpy.datetime64 objects
    #                  )
    #
    # # Convert all arrays (from a string representation) to numpy.ndarray objects
    # # The outermost '[]' characters are removed
    # df = df.map(lambda x: np.fromstring(x[1:-1], sep=' '))
    #
    # if unit == 'C':
    #     # Convert temperature unit from Kelvin to Celsius
    #     df = df.map(lambda x: x - 273.15)
    # elif unit == 'K':
    #     pass  # No conversion needed
    # else:
    #     raise Exception('Unknown temperature unit selection for MERRA v2 data')
    #
    # return df


def _preprocess_merrav2_temperature_data(unit: str = 'C') -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_PATH_MERRA, DATA_FILENAME_TEMPERATURE),
                     compression='gzip',
                     index_col=0,  # Use dates as index
                     parse_dates=True,  # Parse them to numpy.datetime64 objects
                     )

    # Convert all arrays (from a string representation) to numpy.ndarray objects
    # The outermost '[]' characters are removed
    df = df.map(lambda x: np.fromstring(x[1:-1], sep=' '))

    if unit == 'C':  # TODO -- do this after loading from store -- unit conversion doesnt work properly now
        # Convert temperature unit from Kelvin to Celsius
        df = df.map(lambda x: x - 273.15)
    elif unit == 'K':
        pass  # No conversion needed
    else:
        raise Exception('Unknown temperature unit selection for MERRA v2 data')

    store = pd.HDFStore(PATH_DATA_STORE)

    store[KEY_DATA_STORE] = df
    # df = store[KEY_DATA_STORE]

    store.close()

    return df


if __name__ == '__main__':

    # print(_data.iloc[0]['Japan/Abashiri'])
    # print(type(_data.iloc[0]['Japan/Abashiri']))

    print(get_data_temperature())

    # print((get_data()))
    # print(type(get_data().iloc[0]['Japan/Abashiri']))
