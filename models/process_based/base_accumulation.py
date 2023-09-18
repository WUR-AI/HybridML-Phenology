import os
from itertools import product

import numpy as np
import pandas as pd

import multiprocessing as mp

from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

import config
from datasets.dataset import Dataset
from models.base import BaseModel


class BaseAccumulationModel(BaseModel):

    def __init__(self,
                 threshold_chill: float,
                 threshold_growth: float,
                 t_base: float,
                 ):
        super().__init__()
        self._threshold_chill = threshold_chill
        self._threshold_growth = threshold_growth
        self._t_base = t_base

    def chill_units(self, ts: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def growth_units(self, ts: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _predict_ix_alt(self, x: dict) -> tuple:
        ts = x['temperature']

        cu = self.chill_units(ts)
        cus = cu.cumsum(axis=-1)
        r_c = np.where(cus >= self._threshold_chill, 1, 0)

        gu = self.growth_units(ts) * r_c
        gus = gu.cumsum(axis=-1)
        r_g = np.where(gus >= self._threshold_growth, 1, 0)

        ix_bloom = (1-r_g).sum()

        ix_bloom = np.minimum(ix_bloom, Dataset.SEASON_LENGTH - 1)

        return ix_bloom, True, {}

    def predict_ix(self, x: dict) -> tuple:
        return self._predict_ix_alt(x)

        ts = x['temperature']

        # Temperatures np.ndarray ts is assumed to have shape (t, n), where
        #   t is the number of days
        #   n is the number of temperature measurements per day

        # Compute the chill units that are acquired for each day
        cu = self.chill_units(ts)

        # Compute the cumulative chill units over time
        cus = cu.cumsum(axis=-1)

        # Obtain the first index where the cumulative chill units exceeds some threshold
        ixs, = np.where(cus >= self._threshold_chill)
        if len(ixs) == 0:  # If the threshold is never reached, return -1
            info = {
                'chill_units': cu,
                'bloom': False,
            }
            return BaseModel.NO_FIT_IX, False, info

        # If we reached this point -> chill requirement has been met

        # Get the first index where the chill threshold is exceeded
        ix_cr = ixs.min(axis=-1)

        # Select the remaining temperature time series over which the gdd units should be computed
        # That is, all days after meeting the chill requirement
        ts = ts[ix_cr:]
        # Chill units are accumulated until the requirement has been met
        # Omit the other dates
        cu = cu[:ix_cr]

        # Compute the growth units that are acquired for each day
        gu = self.growth_units(ts)

        # Compute the cumulative growth units over time
        gus = gu.cumsum(axis=-1)

        # Get the first index where the cumulative growth units exceeds some threshold
        ixs, = np.where(gus >= self._threshold_growth)
        if len(ixs) == 0:  # If the threshold is never reached, return -1
            info = {
                'ix_cr': ix_cr,
                'chill_units': cu,
                'growth_units': gu,
                'bloom': False,
            }
            return BaseModel.NO_FIT_IX, False, info

        # If we reached this point -> both chill and growth requirement have been met

        # Get the first index at which the threshold was reached
        ix_gr = ixs.min(axis=-1)

        # The bloom index is when both chill and growth requirements have been met
        ix_bloom = ix_gr + ix_cr

        info = {
            'ix_bloom': ix_bloom,
            'ix_cr': ix_cr,
            'chill_units': cu,
            'growth_units': gu,
            'bloom': True,
        }

        return ix_bloom, True, info

    @property
    def growth_requirement(self) -> float:
        return self._threshold_growth

    @property
    def chill_requirement(self) -> float:
        return self._threshold_chill

    @property
    def t_base(self):
        return self._t_base

    @classmethod
    def _fit_grid_local(cls,
                        location: str,
                        xs: list,
                        ) -> pd.DataFrame:

        pool = mp.Pool(processes=mp.cpu_count() - 1)

        def _param_iter() -> iter:
            for params in cls._parameter_grid():
                yield xs, params

        local_grid = pool.starmap(cls._eval_samples, _param_iter())
        for entry in local_grid:
            entry['location'] = location

        # Store all entries in a DataFrame
        df = pd.DataFrame(local_grid)
        # Set the index
        df.set_index(['location', 'i_t_base', 'i_chill_req', 'i_growth_req'], inplace=True)

        return df

    @classmethod
    def _eval_samples(cls, xs: dict, params: tuple) -> dict:

        # Obtain the data that is required to fit the model
        ix_true = [x['bloom_ix'] for x in xs]

        # Unpack parameter values
        (i_tb, tb), (i_cr, cr), (i_gr, gr) = params

        # Initialize a model using parameters
        model = cls(
            threshold_chill=cr,
            threshold_growth=gr,
            t_base=tb,
        )

        # Make predictions
        results = [model.predict_ix(x) for x in xs]
        ix_pred = [doy for doy, _, _ in results]

        # Compute metrics
        mse = mean_squared_error(ix_true, ix_pred)
        r2 = r2_score(ix_true, ix_pred)

        return {
            'i_t_base': i_tb,
            't_base': tb,
            'i_chill_req': i_cr,
            'chill_req': cr,
            'i_growth_req': i_gr,
            'growth_req': gr,
            'mse': mse,
            'r2': r2,
            'n': len(xs),
        }

    @classmethod
    def _grid_tbs(cls):
        """
        :return: a list of base temperatures that will be evaluated during grid search
        """
        raise NotImplementedError

    @classmethod
    def _grid_crs(cls) -> np.ndarray:
        """
        :return: a list of chill requirements that will be evaluated during grid search
        """
        raise NotImplementedError

    @classmethod
    def _grid_grs(cls) -> np.ndarray:
        """
        :return: a list of growth requirements that will be evaluated during grid search
        """
        raise NotImplementedError

    @classmethod
    def _parameter_grid(cls) -> iter:
        """
        Iterate through all possible combinations of parameters
        """
        return product(
            enumerate(cls._grid_tbs()),
            enumerate(cls._grid_crs()),
            enumerate(cls._grid_grs()),
        )


class BaseLocalAccumulationModel(BaseModel):

    def __init__(self, local_models: dict):
        super().__init__()
        self._local_models = local_models

    @classmethod
    def model_cls(cls) -> callable:
        raise NotImplementedError

    def predict_ix(self, x: dict) -> tuple:
        assert x['location'] in self._local_models.keys(), 'Model should be fitted to the location first!'
        model = self._local_models[x['location']]
        return model.predict_ix(x)

    def predict(self, x: dict) -> tuple:
        assert x['location'] in self._local_models.keys(), 'Model should be fitted to the location first!'
        model = self._local_models[x['location']]
        return model.predict(x)

    @classmethod
    def fit(cls,
            dataset: Dataset,
            method: str = 'grid',
            name: str = None,
            ) -> tuple:
        name = name or cls.__name__

        if method == 'grid':
            return cls._fit_grid(dataset, name)

        raise NotImplementedError

    @classmethod
    def _path_params_dir(cls) -> str:
        return os.path.join(config.PATH_PARAMS_DIR, cls.__name__)

    @classmethod
    def _fit_grid(cls,
                  dataset: Dataset,
                  name: str,
                  ):
        model_cls = cls.model_cls()
        assert issubclass(model_cls, BaseAccumulationModel)

        # Do a grid search for locations separately
        # Store results in a list of dataframes
        dfs = []

        # Keep a progress bar of the fitting process
        location_progress = tqdm(dataset.locations_train)

        # Iterate through all locations
        for location in location_progress:
            location_progress.set_description(f'{model_cls.__name__} Grid Search [Current: {location}]')

            # Get the data corresponding to this location
            data_local = dataset.get_local_train_data(location)
            # Perform a grid search in one location
            df_local = model_cls._fit_grid_local(location, data_local)
            # Append the local grid to the results
            dfs.append(df_local)

        # Concatenate all local dataframes
        df = pd.concat(dfs)

        # Save the grid
        fn = f'{name}_grid_parameter_fits.csv.gz'
        path = os.path.join(cls._path_params_dir(), name)
        os.makedirs(path, exist_ok=True)
        df.to_csv(os.path.join(path, fn), compression='gzip')

        # Save the best parameter configuration per location
        dfs = [df.sort_index(level=[1, 2, 3]) for df in dfs]
        best_fits = pd.DataFrame([df.loc[df['mse'].idxmin()] for df in dfs])
        best_fits.to_csv(os.path.join(path, f'{name}_grid_parameter_fits_best_mse.csv'))

        # Initialize a model from the saved parameters and return it
        model = cls.load(name)
        return model, {}

    @classmethod
    def load(cls, name: str):

        fn = f'{name}_grid_parameter_fits_best_mse.csv'
        path = os.path.join(cls._path_params_dir(), name, fn)

        df = pd.read_csv(path,
                         index_col=[0, 1, 2, 3],
                         )

        local_models = dict()
        for i in df.index.values:
            location, _, _, _ = i

            tb = df.loc[i]['t_base']
            cr = df.loc[i]['chill_req']
            gr = df.loc[i]['growth_req']

            # Initialize a model using parameters
            model = cls.model_cls()(
                threshold_chill=cr,
                threshold_growth=gr,
                t_base=tb,
            )

            local_models[location] = model

        return cls(local_models)

