import torch

import config
from models.base_torch_accumulation import BaseTorchAccumulationModel
from models.components.param_v3 import ParameterModel
from models.components.torch_phenology import DegreeDaysCNN, DegreeDaysDNN_PP, DegreeDaysDNN, DegreeDaysDNN_Coord, \
    DegreeDaysV
from models.components.torch_phenology_pb import TorchGDD, LogisticUtahChillModule
# from models.components.param import ParameterModel


class NNChillModel(BaseTorchAccumulationModel):

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
        # self._chill_model = DegreeDaysCNN()
        self._chill_model = DegreeDaysDNN(hidden_size=64)
        # self._chill_model = DegreeDaysV(hidden_size=64)
        # self._chill_model = DegreeDaysDNN_PP()
        # self._chill_model = DegreeDaysDNN_Coord()

    def f_units(self, xs: dict, tb: torch.Tensor):
        cus = self._chill_model(xs)
        gus = TorchGDD.f_gdd(xs, tb)
        return cus, gus

    def freeze_operator_weights(self):
        for p in self._chill_model.parameters():
            p.requires_grad = False

    def unfreeze_operator_weights(self):
        for p in self._chill_model.parameters():
            p.requires_grad = True


# class NNGrowthModel(BaseTorchAccumulationModel):
#
#     def __init__(self,
#                  param_model: ParameterModel,
#                  ):
#         super().__init__(param_model)
#         self._growth_model = DegreeDaysCNN()
#         self._chill_model = LogisticUtahChillModule()
#
#     def f_units_chill_growth(self, xs: dict, tb: torch.Tensor):
#
#         cus = self._chill_model(xs)
#         gus = self._growth_model(xs)
#
#         return cus, gus
#
#
# class NNChillGrowthModel(BaseTorchAccumulationModel):
#
#     def __init__(self,
#                  param_model: ParameterModel,
#                  ):
#         super().__init__(param_model)
#         self._model = DegreeDaysCNN(num_out_channels=2)
#
#     def f_units_chill_growth(self, xs: dict, tb: torch.Tensor):
#
#         us = self._model(xs)
#
#         # print(us.shape)
#
#         cus, gus = torch.tensor_split(us, 2, dim=1)
#
#         # print(cus.shape)
#
#         return cus, gus

# if __name__ == '__main__':
#
#     import torch.optim
#
#     import main
#
#     _dataset = main.get_configured_dataset()  # TODO -- delete
#
#     _run_name = 'nn_chill'
#     # _run_name = 'nn_chill_lat'
#     # _run_name = 'nn_chill_no_penalty'
#     # _run_name = 'nn_chill_and_param'
#     _train_model = True
#     # _train_model = False
#
#     if _train_model:
#
#         _model, _ = NNChillModel.fit(_dataset,
#                                      device='cuda',
#                                      num_epochs=20000,
#                                      # num_epochs=2000,
#                                      f_optim=torch.optim.Adam,
#                                      # f_optim=torch.optim.SGD,
#                                      optim_kwargs={'lr': 1e-3},
#                                      # optim_kwargs={'lr': 1e-2},
#                                      # scheduler_step_size=200,
#                                      scheduler_step_size=400,
#                                      # scheduler_step_size=50,
#                                      # scheduler_step_size=25,
#                                      scheduler_decay=0.9,
#                                      # scheduler_decay=0.5,
#                                      # scheduler_decay=0.1,
#                                      clip_gradient=1e-0,
#                                      )
#
#         _model.save(f'{_run_name}.pth')
#
#     else:
#         _model = NNChillModel.load(
#             f'{_run_name}.pth'
#         ).cpu()
#
#         # # _data = torch.arange(-10, 20, 0.1).view(-1, 1).expand(-1, 24)
#         # _data = torch.arange(-10, 30, 0.1).view(-1, 1).expand(-1, 24)
#         # _data = _data.unsqueeze(0).to(config.TORCH_DTYPE)
#         #
#         # _units = _model._chill_model(_data) / 24
#         #
#         # print(_units)
#
#     _model.cpu()
#
#     from evaluation import evaluation
#
#     evaluation.eval(_model,
#                     _dataset,
#                     dirname=f'evaluation_{_run_name}',
#                     )
#
