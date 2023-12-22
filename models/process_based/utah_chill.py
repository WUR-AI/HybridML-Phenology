import numpy as np

from phenology.chill.utah_model import utah_chill
from phenology.growth.gdd import f_gdd
from models.process_based.base_accumulation import BaseAccumulationModel, BaseLocalAccumulationModel


class UtahChillModel(BaseAccumulationModel):

    def __init__(self, threshold_chill: float, threshold_growth: float, t_base: float):
        super().__init__(threshold_chill, threshold_growth, t_base)

    def chill_units(self, ts: np.ndarray) -> np.ndarray:
        return np.maximum(utah_chill(ts), 0)

    def growth_units(self, ts: np.ndarray) -> np.ndarray:
        return f_gdd(ts, self.t_base)

    @classmethod
    def _grid_tbs(cls):
        """
        :return: a list of base temperatures that will be evaluated during grid search
        """
        t_min = 0
        t_max = 16
        t_step = 0.5
        return np.arange(t_min, t_max, t_step)

    @classmethod
    def _grid_crs(cls) -> np.ndarray:
        """
        :return: a list of chill requirements that will be evaluated during grid search
        """
        # cr_min = -2000
        # cr_max = 0
        cr_min = 0
        cr_max = 2000
        cr_step = 25
        return np.arange(cr_min, cr_max, cr_step)

    @classmethod
    def _grid_grs(cls) -> np.ndarray:
        """
        :return: a list of growth requirements that will be evaluated during grid search
        """
        # gr_min = 0
        # gr_max = 2000
        # gr_step = 25
        # gr_min = 0
        # gr_max = 600
        # gr_step = 2
        gr_min = 0
        gr_max = 600
        gr_step = 7.5
        return np.arange(gr_min, gr_max, gr_step)


class LocalUtahChillModel(BaseLocalAccumulationModel):

    @classmethod
    def model_cls(cls) -> callable:
        return UtahChillModel
