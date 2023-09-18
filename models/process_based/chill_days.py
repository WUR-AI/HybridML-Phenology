import numpy as np

from phenology.chill.chilldays import chill_days, anti_chill_days
from models.process_based.base_accumulation import BaseAccumulationModel, BaseLocalAccumulationModel


class ChillDaysModel(BaseAccumulationModel):

    def __init__(self, threshold_chill: float, threshold_growth: float, t_base: float):
        super().__init__(threshold_chill, threshold_growth, t_base)

    def chill_units(self, ts: np.ndarray) -> np.ndarray:
        return np.maximum(-chill_days(ts, self.t_base), 0)

    def growth_units(self, ts: np.ndarray) -> np.ndarray:
        return anti_chill_days(ts, self.t_base)

    @classmethod
    def _grid_tbs(cls):
        """
        :return: a list of base temperatures that will be evaluated during grid search
        """
        t_min = 0
        t_max = 16
        t_step = 0.2
        # t_min = 0
        # t_max = 15
        # t_step = 1
        return np.arange(t_min, t_max, t_step)

    @classmethod
    def _grid_crs(cls) -> np.ndarray:
        """
        :return: a list of chill requirements that will be evaluated during grid search
        """
        cr_min = 0
        cr_max = 400
        cr_step = 5
        # cr_min = 0
        # cr_max = 300
        # cr_step = 50
        return np.arange(cr_min, cr_max, cr_step)

    @classmethod
    def _grid_grs(cls) -> np.ndarray:
        """
        :return: a list of growth requirements that will be evaluated during grid search
        """
        acr_min = 0
        acr_max = 400
        acr_step = 5
        # acr_min = 0
        # acr_max = 300
        # acr_step = 50
        return np.arange(acr_min, acr_max, acr_step)


class LocalChillDaysModel(BaseLocalAccumulationModel):

    @classmethod
    def model_cls(cls) -> callable:
        return ChillDaysModel

