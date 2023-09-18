import os

from collections import defaultdict

import pandas as pd

import config
from datasets.dataset import Dataset
from models.base import BaseModel


class MeanModel(BaseModel):

    """

        Model that outputs a location specific mean day-of-year based on the train data

    """

    PATH_PARAMS_DIR = os.path.join(config.PATH_PARAMS_DIR, 'MeanModel')

    def __init__(self, means: dict):
        super().__init__()
        self._loc_means = means

    def predict_ix(self, x: dict) -> tuple:
        raise NotImplementedError

    def predict(self, x: dict) -> tuple:
        location = x['location']
        return self._loc_means[location], True, {}

    @classmethod
    def fit(cls, dataset: Dataset, method: str = None) -> tuple:

        data_train = dataset.get_train_data()

        loc_doys = defaultdict(list)
        for x in data_train:
            location = x['location']
            loc_doys[location].append(x['bloom_doy'])

        loc_means = {
            loc: int((sum(doys) / len(doys)) + 1)
            for loc, doys in loc_doys.items()
        }

        model = cls(loc_means)

        return model, {}

    def save(self, fn: str = 'location_mean_doys.csv'):
        entries = [[loc, mean] for loc, mean in self._loc_means.items()]
        df = pd.DataFrame(entries, columns=['location', 'mean'])
        os.makedirs(MeanModel.PATH_PARAMS_DIR, exist_ok=True)
        df.to_csv(os.path.join(MeanModel.PATH_PARAMS_DIR, fn))

    @staticmethod
    def load(fn: str = 'location_mean_doys.csv'):
        df = pd.read_csv(os.path.join(MeanModel.PATH_PARAMS_DIR, fn), index_col=0)
        loc_means = {loc: mean for loc, mean in df.values}
        model = MeanModel(loc_means)
        return model
