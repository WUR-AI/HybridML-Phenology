import torch

from models.base_torch_accumulation import BaseTorchAccumulationModel
from models.components.param_v2 import ParameterModel
# from models.components.param import ParameterModel
from models.components.torch_phenology_pb import TorchGDD, LogisticUtahChillModule


class DiffUtahModel(BaseTorchAccumulationModel):

    def __init__(self,
                 param_model: ParameterModel,
                 ):
        super().__init__(param_model)
        self._chill_model = LogisticUtahChillModule()

    def f_units_chill_growth(self, xs: dict, tb: torch.Tensor):
        cus = self._chill_model(xs)
        gus = TorchGDD.f_gdd(xs, tb)
        return cus, gus
