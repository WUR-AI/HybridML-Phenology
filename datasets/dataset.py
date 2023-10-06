import os

import numpy as np
import pandas as pd

import pickle

import config
import data.bloom_doy
import data.merra_v2
from util.photoperiod import photoperiod


class Dataset:
    """
        Cherry Blossom DOY dataset
    """

    # Earliest year that will be encountered in the dataset
    MIN_YEAR = 1981
    # Latest year that will be encountered in the dataset
    MAX_YEAR = 2021
    YEAR_RANGE = list(range(MIN_YEAR, MAX_YEAR + 1))

    """
        Days within the season are labeled by three units, namely
            - date
            - index
            - doy
            whose timelines are related as follows:

        date:  +----------------------+-------------------------------------------------------------+
       SEASON_START_DATE          1st of Jan.                                           (end date depends on year)

        index: +----------------------+-------------------------------------------------------------+
               0                 DOY_SHIFT + 1                                               SEASON_LENGTH - 1

        doy:                          +-------------------------------------------------------------+
                                      1                                                      SEASON_END_DOY 

    """
    # Start date (in the previous year) from which temperature data is included
    SEASON_START_DATE = '10-01'
    # Number of days of temperature data that will be included
    SEASON_LENGTH = 274  # Season end is roughly at the start of July (includes 1st of July in leap years)

    # Difference between 1st DOY and start of season
    DOY_SHIFT = (np.datetime64(f'1980-12-31') - np.datetime64(f'1980-{SEASON_START_DATE}')) // np.timedelta64(1, 'D')
    # All DOYS in the season (incl negatives)
    DOYS = (np.arange(SEASON_LENGTH) - DOY_SHIFT).astype(int)
    SEASON_END_DOY = DOYS[-1]

    def __init__(self,
                 year_split: tuple,
                 locations_train: list = None,
                 locations_test: list = None,
                 min_num_local_samples_train: int = 2,
                 min_num_local_samples_test: int = 2,
                 include_temperature: bool = False,
                 include_photoperiod: bool = False,
                 temperature_unit: str = 'C',
                 ):
        """
        Temperature dataset class

        :param year_split: two-tuple of lists that contains (train years, test years)
        :param locations_train: (optional) list of locations to include in the train data. Default is None, in which
                                case all locations are chosen
        :param locations_test: (optional) list of locations to include in the test data. Default is None, in which
                               case all locations are chosen
        :param min_num_local_samples_train: (optional) Minimum number of train samples that should be available for a
                                            location to be included in the dataset (both train and test)
        :param min_num_local_samples_test: (optional) Minimum number of test samples that should be available for a
                                            location to be included in the dataset (both train and test)
        :param temperature_unit: (optional) Temperature unit in which the data is expressed. Default is 'C' (Celsius)
        """

        # Validate input arguments
        assert len(year_split) == 2
        assert min_num_local_samples_train >= 0
        assert min_num_local_samples_test >= 0

        """
            Loading the data
        """

        # Obtain the blooming DOY label dataset
        # The dataset consists of a DataFrame with blooming day-of-year statistics per region
        # The DataFrame is indexed by (year (int), location (str)) and has the following columns:
        #   - lat (latitude, float)
        #   - long (longitude, float)
        #   - alt (altitude, float)
        #   - bloom_date (blooming date)
        #   - bloom_doy (blooming day-of-year, int)
        self._data_doy = data.bloom_doy.get_data(set_index=True)
        # Filter the data based on the years that should be included
        # Since years are part of the index, this is done through grouping the data for each year and concatenating
        self._data_doy = pd.concat(
            [self._data_doy.xs(year, drop_level=False) for year in Dataset.YEAR_RANGE]
        )

        """
            Splitting the data based on years
        """
        # Validate the input split
        years_train, years_test = year_split
        assert len(set(years_train).intersection(set(years_test))) == 0  # make sure there is no overlap
        assert all([year in self.YEAR_RANGE for year in years_train + years_test])

        # Set the year split
        self._years_train = years_train
        self._years_test = years_test

        # Split the data based on the year split
        self._ixs_train = [(year, loc) for year, loc in self._data_doy.index if year in self._years_train]
        self._ixs_test = [(year, loc) for year, loc in self._data_doy.index if year in self._years_test]

        """
            Filter based on nr. of samples available for training/testing
        """
        locations = set([loc for _, loc in self._data_doy.index.values])  # Get all locations

        location_selection = set()
        for loc in locations:
            # Get the indices (of samples) that correspond to this location
            ixs_loc_train = [(year, location) for year, location in self._ixs_train if location == loc]
            ixs_loc_test = [(year, location) for year, location in self._ixs_test if location == loc]

            enough_samples_train = len(ixs_loc_train) >= min_num_local_samples_train
            enough_samples_test = len(ixs_loc_test) >= min_num_local_samples_test

            # If all requirements are met the location will be included in the dataset
            if enough_samples_train and enough_samples_test:
                location_selection.add(loc)

        """
            Optionally filter based on location if train/test locations were explicitly given
        """

        # Validate input
        # assert locations_train is None or all([loc in locations for loc in locations_train])
        # assert locations_test is None or all([loc in locations for loc in locations_test])

        # If train locations were explicitly given, filter index based on locations
        if locations_train is None:
            locations_train = location_selection  # All valid locations are used
            # No filtering required
        else:
            # Only the valid locations in the provided locations are used
            locations_train = location_selection.intersection(locations_train)

        # If test locations were explicitly given, filter index based on locations
        if locations_test is None:
            locations_test = location_selection  # All valid locations are used
            # No filtering required
        else:
            # Only the valid locations in the provided locations are used
            locations_test = location_selection.intersection(locations_test)

        # Filter indices based on the location selection
        self._ixs_train = [(year, location) for year, location in self._ixs_train if location in locations_train]
        self._ixs_test = [(year, location) for year, location in self._ixs_test if location in locations_test]

        # Filter based on selected indices
        self._data_doy = self._data_doy.loc[self._ixs_train + self._ixs_test]

        """
            Obtain optional data
        """

        self._includes_temperature = include_temperature

        if self._includes_temperature:
            # Obtain the temperature dataset
            # The dataset consists of a DataFrame with hourly temperature statistics for the relevant locations
            # The DataFrame is indexed by the date (numpy.datetime64). Each location corresponds to one column
            # Each entry is a numpy.ndarray containing 24 float32 values corresponding to hourly temperature estimates
            self._data_t = data.merra_v2.get_data_temperature(unit=temperature_unit)
        else:
            self._data_t = None

        self._includes_photoperiod = include_photoperiod

        pass  # TODO -- soil moisture

    def __len__(self):
        return len(self._data_doy)

    def __getitem__(self, index) -> dict:
        # Index can be specified as integer or multi-index values as used in the DataFrame
        if isinstance(index, int):
            sample_doy = self._data_doy.iloc[index]
            year, location = sample_doy.name

        elif isinstance(index, tuple):
            sample_doy = self._data_doy.loc[index]
            year, location = index

        else:
            raise Exception(f'Unsupported index type {type(index)}')

        date_end = self.end_date_in_year(year)
        date_start = self.start_date_in_year(year)

        data_opt = dict()

        if self._includes_temperature:
            # Get temperature data per day within season
            temperature_data = self._data_t[date_start:date_end][location]
            # As numpy array
            temperature_data = temperature_data.values
            # Concatenate all temperature arrays to one matrix
            temperature_data = np.concatenate([np.expand_dims(a, axis=0) for a in temperature_data], axis=0)
            # Add to data
            data_opt['temperature'] = temperature_data

        if self._includes_photoperiod:
            data_opt['photoperiod'] = [photoperiod(
                phi=sample_doy.lat,
                doy=doy,
                verbose=False,
            ) for doy in Dataset.DOYS]

        # TODO -- soil moisture

        return {
            'lat': sample_doy.lat,
            'lon': sample_doy.long,
            'alt': sample_doy.alt,
            'location': location,
            'year': year,
            'bloom_date': sample_doy.bloom_date,
            'bloom_doy': sample_doy.bloom_doy,
            'bloom_ix': self.doy_to_index(sample_doy.bloom_doy),
            **data_opt,
        }

    @property
    def includes_temperature(self) -> bool:
        return self._includes_temperature

    @property
    def includes_photoperiod(self) -> bool:
        return self._includes_photoperiod

    @property
    def locations(self):
        return list(set([loc for _, loc in self._data_doy.index.values]))

    @property
    def locations_train(self):
        return list(set([loc for _, loc in self._ixs_train]))

    @property
    def locations_test(self):
        return list(set([loc for _, loc in self._ixs_test]))

    @property
    def years(self):
        return list(set([year for year, _ in self._data_doy.index.values]))

    @property
    def countries(self):
        return list(set([split_location_token(loc)[0] for _, loc in self._data_doy.index.values]))

    @property
    def countries_train(self):
        return list(set([split_location_token(loc)[0] for _, loc in self.get_train_indices()]))

    @property
    def countries_test(self):
        return list(set([split_location_token(loc)[0] for _, loc in self.get_test_indices()]))

    @staticmethod
    def start_date_in_year(year: int) -> np.datetime64:
        """
        Get the start date of the season for the specified year as a np.datetime64 object
        Note: The season starts in the year before
        """
        return np.datetime64(f'{year - 1}-{Dataset.SEASON_START_DATE}')

    @staticmethod
    def end_date_in_year(year: int) -> np.datetime64:
        """
        Get the end date of the season for the specified year as a np.datetime64 object

        This is the last date at which temperature data is included
        So this date in included in the dataset
        """
        start_date = Dataset.start_date_in_year(year)
        return start_date + np.timedelta64(Dataset.SEASON_LENGTH - 1, 'D')

    @staticmethod
    def doy_to_date_in_year(year: int, doy: int) -> np.datetime64:
        assert 0 < doy <= 365
        return np.datetime64(f'{year}-01-01') + np.timedelta64(doy - 1, 'D')

    @staticmethod
    def index_to_doy(index: int) -> int:
        assert 0 <= index < Dataset.SEASON_LENGTH
        return Dataset.DOYS[index]

    @staticmethod
    def doy_to_index(doy: int, assert_positive: bool = True) -> int:
        if assert_positive:
            assert 0 < doy <= Dataset.DOYS[-1], f'invalid DOY ({doy})'
        return doy + Dataset.DOY_SHIFT

    @staticmethod
    def index_to_date_in_year(year: int, index: int) -> np.datetime64:
        doy = Dataset.index_to_doy(index)
        year_start = np.datetime64(f'{year}-01-01')
        return year_start + np.timedelta64(doy - 1, 'D')

    @staticmethod
    def dates_in_year(year: int):
        assert year in Dataset.YEAR_RANGE
        start_date = Dataset.start_date_in_year(year)
        end_date = Dataset.end_date_in_year(year)
        return pd.date_range(
            start=start_date,
            end=end_date,
        ).values

    def get_train_indices(self) -> list:
        return list(self._ixs_train)

    def get_test_indices(self) -> list:
        return list(self._ixs_test)

    def get_local_train_indices(self, location: str) -> list:
        return [(year, loc) for year, loc in self.get_train_indices() if location == loc]

    def get_local_test_indices(self, location: str) -> list:
        return [(year, loc) for year, loc in self.get_test_indices() if location == loc]

    def get_train_data(self) -> list:
        samples = [self[index] for index in self.get_train_indices()]
        return samples

    def get_test_data(self) -> list:
        samples = [self[index] for index in self.get_test_indices()]
        return samples

    def get_local_train_data(self, location: str,) -> list:
        samples = [self[index] for index in self.get_local_train_indices(location)]
        return samples

    def get_local_test_data(self, location: str,) -> list:
        samples = [self[index] for index in self.get_local_test_indices(location)]
        return samples

    def get_train_data_in_years(self, years: list):
        ixs = [(year, loc) for year, loc in self.get_train_indices() if year in years]
        return [self[index] for index in ixs]

    def get_test_data_in_years(self, years: list):
        ixs = [(year, loc) for year, loc in self.get_test_indices() if year in years]
        return [self[index] for index in ixs]

    def save(self, fn: str):
        path = os.path.join(config.PATH_DATASET_DIR, type(self).__name__)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, fn), 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, fn: str) -> 'Dataset':
        path = os.path.join(config.PATH_DATASET_DIR, cls.__name__)
        assert os.path.isfile(os.path.join(path, fn))
        with open(os.path.join(path, fn), 'rb') as f:
            dataset = pickle.load(f)
        return dataset

    def _get_min_max_temperatures(self):  # For normalization purposes
        t_min = np.inf
        t_max = -np.inf
        for i in range(len(self)):
            x = self[i]
            ts = x['temperature']
            if np.min(ts) < t_min:
                t_min = np.min(ts)
            if np.max(ts) > t_max:
                t_max = np.max(ts)
        return t_min, t_max

    def _get_min_max_val(self, key: str):  # For normalization purposes
        v_min = np.inf
        v_max = -np.inf
        vals = []
        for x in self.get_train_data():
        # for i in range(len(self)):
        #     x = self[i]
            v = x[key]
            if v < v_min:
                v_min = v
            if v > v_max:
                v_max = v
            vals.append(v)

        print(key)
        print(v_min, v_max)
        print(np.mean(vals))
        print(np.std(vals))

        return v_min, v_max


def split_location_token(location: str):
    """
    Split a location token string into (country, site)
    :param location: the location token
    :return: a two-tuple of (location (str), site (str))
    """
    country, site = location.split('/')
    return country, site


def _save_debug_dataset():
    from sklearn.model_selection import train_test_split
    # Create a small subset of the data for testing and save it
    dataset = Dataset(
        year_split=train_test_split(Dataset.YEAR_RANGE),
        locations_train=['Japan/Abashiri'],
        locations_test=['Japan/Abashiri'],
        include_temperature=True,
    )
    dataset.save('dataset_temperature_debug.pickle')


def load_debug_dataset():
    return Dataset.load('dataset_temperature_debug.pickle')


if __name__ == '__main__':
    print(Dataset.DOY_SHIFT)
    print(Dataset.DOYS[Dataset.DOY_SHIFT])
    print(Dataset.DOYS)
    print(Dataset.SEASON_LENGTH - 1 - Dataset.DOY_SHIFT)
    print(Dataset.doy_to_index(1))

    _save_debug_dataset()
