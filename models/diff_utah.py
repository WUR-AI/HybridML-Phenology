import torch

from models.base_torch_accumulation import BaseTorchAccumulationModel
from models.components.param_v3 import ParameterModel
from models.components.torch_phenology_pb import TorchGDD, LogisticUtahChillModule


class DiffUtahModel(BaseTorchAccumulationModel):  # TODO

    def __init__(self,
                 parameter_model_thc: ParameterModel,
                 parameter_model_thg: ParameterModel,
                 parameter_model_tbg: ParameterModel,
                 parameter_model_slc: ParameterModel,
                 parameter_model_slg: ParameterModel,
                 inference_mode_train: str = BaseTorchAccumulationModel.INFERENCE_MODES[0],
                 inference_mode_test: str = BaseTorchAccumulationModel.INFERENCE_MODES[0],
                 loss_f: str = BaseTorchAccumulationModel.LOSSES[0],
                 ):
        super().__init__(
            parameter_model_thc=parameter_model_thc,
            parameter_model_thg=parameter_model_thg,
            parameter_model_tbg=parameter_model_tbg,
            parameter_model_slc=parameter_model_slc,
            parameter_model_slg=parameter_model_slg,
            inference_mode_train=inference_mode_train,
            inference_mode_test=inference_mode_test,
            loss_f=loss_f,
        )
        self._chill_model = LogisticUtahChillModule()

    def f_units(self, xs: dict, tb: torch.Tensor):
        cus = self._chill_model(xs)
        gus = TorchGDD.f_gdd(xs, tb)
        return cus, gus
