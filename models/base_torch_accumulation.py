import os
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import multiprocessing as mp

import config
from datasets.dataset import Dataset
from models.base_torch import BaseTorchModel
from models.components.logistic import SoftThreshold
from models.components.param_v2 import ParameterModel
# from models.components.param import ParamNet, LocalParams, ParameterModel
from util.torch import batch_tensors


class BaseTorchAccumulationModel(BaseTorchModel):

    SLOPE = 50
    SCALE_CHILL = Dataset.SEASON_LENGTH * 24  # TODO -- same scale
    SCALE_GROWTH = Dataset.SEASON_LENGTH

    TH_AUG_GRAD = False  # TODO -- remove

    def __init__(self, param_model: ParameterModel):
        super().__init__()
        self._param_model = param_model

        self._debug_mode = False  # TODO

    @property
    def parameter_model(self) -> ParameterModel:
        return self._param_model

    def f_parameters(self, xs: dict) -> tuple:
        return self._param_model.get_parameters(xs)

    def f_units_chill_growth(self, xs: dict, tb: torch.Tensor):
        raise NotImplementedError

    def forward(self,
                xs: dict,
                soft: bool = True,
                ) -> tuple:

        debug_info = dict()

        th_c, th_g, tb_g = self.f_parameters(xs)

        units_c, units_g = self.f_units_chill_growth(xs, tb_g)

        units_c /= BaseTorchAccumulationModel.SCALE_CHILL
        units_g /= BaseTorchAccumulationModel.SCALE_GROWTH

        units_c_cs = units_c.cumsum(dim=-1)

        req_c = SoftThreshold.f_soft_threshold(units_c_cs,
                                               BaseTorchAccumulationModel.SLOPE,
                                               th_c,
                                               augment_gradient=self.TH_AUG_GRAD,
                                               )

        units_g_masked = units_g * req_c
        units_g_cs = units_g_masked.cumsum(dim=-1)

        req_g = SoftThreshold.f_soft_threshold(units_g_cs,
                                               BaseTorchAccumulationModel.SLOPE,
                                               th_g,
                                               augment_gradient=self.TH_AUG_GRAD,
                                               )

        """
            Compute the blooming ix 
        """
        # soft=False  # TODO -- remove? make instance variable?

        if soft:
            ix = (1 - req_g).sum(dim=-1)
        else:
            mask = (units_g_cs >= th_g).to(torch.int)
            ix = (1 - mask).sum(dim=-1)

        # If blooming never occurred -> set it to the end of the season
        ix = ix.clamp(min=0, max=Dataset.SEASON_LENGTH - 1)

        # ix = (1 - req_g).sum(dim=-1)
        # ix = ix.clamp(min=0, max=Dataset.SEASON_LENGTH - 1)

        optional_info = dict()
        if self._debug_mode:

            debug_info['req_c'] = req_c.cpu().detach().numpy()
            debug_info['req_g'] = req_g.cpu().detach().numpy()
            debug_info['units_c'] = units_c.cpu().detach().numpy()
            debug_info['units_g'] = units_g.cpu().detach().numpy()
            debug_info['units_g_masked'] = units_g_masked.cpu().detach().numpy()
            debug_info['th_c'] = th_c.detach().cpu().detach().numpy()
            debug_info['th_g'] = th_g.detach().cpu().detach().numpy()
            debug_info['tb_g'] = tb_g.detach().cpu().detach().numpy()

            optional_info['debug'] = debug_info

        return ix, {
            # 'units_c': units_c,
            # 'units_g': units_g,
            'req_c': req_c,
            'req_g': req_g,
            # 'units_g_masked': units_g_masked,

            **optional_info,
        }

    # def loss(self, xs: dict, scale: float = 1e-3) -> tuple:
    #     _, info = self(xs)
    #
    #     req_g_pred = info['req_g']
    #     bs = req_g_pred.size(0)
    #
    #     ys_true = xs['bloom_ix'].to(config.TORCH_DTYPE).to(req_g_pred.device)
    #
    #     req_g_true = torch.cat(
    #         [torch.arange(Dataset.SEASON_LENGTH).unsqueeze(0) for _ in range(bs)], dim=0
    #     ).to(req_g_pred.device)
    #     req_g_true = (req_g_true >= ys_true.view(-1, 1)).to(config.TORCH_DTYPE)
    #
    #     loss = F.binary_cross_entropy(req_g_pred, req_g_true)
    #
    #     return loss, {
    #         'forward_pass': info,
    #     }

    # @staticmethod
    # def _compute_ix(units_g_cs: torch.Tensor, req_g: torch.Tensor, beta: torch.Tensor, soft: bool = True,):
    #
    #     if soft:
    #         ix = (1 - req_g).sum(dim=-1)
    #     else:
    #         ix = (1 - torch.where(units_g_cs >= beta)).sum(dim=-1)
    #
    #     # If blooming never occurred -> set it to the end of the season
    #     ix = ix.clamp(min=0, max=Dataset.SEASON_LENGTH - 1)
    #
    #     return ix

    @classmethod
    def _path_params_dir(cls) -> str:
        return os.path.join(config.PATH_PARAMS_DIR, cls.__name__)

    def freeze_operator_weights(self):
        for p in self.parameters():
            p.requires_grad = False
        for p in self._param_model.parameters():
            p.requires_grad = True

    def unfreeze_operator_weights(self):
        for p in self.parameters():
            p.requires_grad = True

    # def loss(self, xs: dict, scale: float = 1e-3) -> tuple:
    #     loss, info = super().loss(xs, scale)
    #     req_c = info['forward_pass']['req_c']
    #     req_g = info['forward_pass']['req_g']
    #
    #     c_penalty = (req_c[:, -1] - req_c[:, 0]).mean()
    #     g_penalty = (req_g[:, -1] - req_g[:, 0]).mean()
    #
    #     # print(loss)
    #
    #     w = 1
    #
    #     loss -= w * (c_penalty + g_penalty)
    #
    #     info['c_loss'] = c_penalty.item()
    #     info['g_loss'] = g_penalty.item()
    #
    #     return loss, info









# class BaseTorchAccumulationModelNN(BaseTorchAccumulationModel):
#
#     TH_AUG_GRAD = False
#
#     def __init__(self, parameter_model: ParamNet = None):
#         super().__init__()
#
#         self._params_nn = parameter_model or ParamNet()
#
#     def f_parameters(self, xs: dict) -> tuple:
#         return self._params_nn(xs)
#
#     def loss(self, xs: dict, scale: float = 1e-3) -> tuple:
#         loss, info = super().loss(xs, scale)
#         req_c = info['forward_pass']['req_c']
#         req_g = info['forward_pass']['req_g']
#
#         c_penalty = (req_c[:, -1] - req_c[:, 0]).mean()
#         g_penalty = (req_g[:, -1] - req_g[:, 0]).mean()
#
#         # print(loss)
#
#         w = 1
#
#         loss -= w * (c_penalty + g_penalty)
#
#         info['c_loss'] = c_penalty.item()
#         info['g_loss'] = g_penalty.item()
#
#         return loss, info
#
#
# class BaseTorchAccumulationModelLocal(BaseTorchAccumulationModel):
#
#     TH_AUG_GRAD = True
#
#     def __init__(self,
#                  locations: list = None,
#                  location_params: dict = None,
#                  ):
#         super().__init__()
#
#         self._params_local = LocalParams(
#             locations=locations,
#             location_params=location_params,
#         )
#
#     def f_parameters(self, xs: dict) -> tuple:
#         return self._params_local(xs)
#
#     @classmethod
#     def fit(cls, *args, **kwargs) -> tuple:
#         model, info = super().fit(*args, **kwargs)
#         model._params_local.save_param_df()
#         return model, info
#
#     @classmethod
#     def _fit_grid(cls,
#                   dataset: BaseDataset,
#                   name: str,
#                   ):
#
#         # Depending on the dataset class, implement a different optimization procedure
#         if isinstance(dataset, TemperatureBlossomDataset):
#
#             # Do a grid search for locations separately
#             # Store results in a list of dataframes
#             dfs = []
#
#             # Keep a progress bar of the fitting process
#             location_progress = tqdm(dataset.locations_train)
#
#             # Iterate through all locations
#             for location in location_progress:
#                 location_progress.set_description(f'{cls.__name__} Grid Search [Current: {location}]')
#
#                 # Get the data corresponding to this location
#                 data_local = dataset.get_local_train_data(location)
#                 # Perform a grid search in one location
#                 df_local = cls._fit_grid_local(location, data_local)
#                 # Append the local grid to the results
#                 dfs.append(df_local)
#
#             # Concatenate all local dataframes
#             df = pd.concat(dfs)
#
#             # Save the grid
#             fn = f'{name}_grid_parameter_fits.csv.gz'
#             path = os.path.join(cls._path_params_dir(), name)
#             os.makedirs(path, exist_ok=True)
#             df.to_csv(os.path.join(path, fn), compression='gzip')
#
#             # Save the best parameter configuration per location
#             best_fits = pd.DataFrame([df.loc[df['mse'].idxmin()] for df in dfs])
#             best_fits.to_csv(os.path.join(path, f'{name}_grid_parameter_fits_best_mse.csv'))
#
#             # Initialize a model from the saved parameters and return it
#             # model = cls.load(name)  # TODO -- load proper
#             model = None
#             return model, {}
#
#         raise Exception('Unknown dataset class')
#
#     @classmethod
#     def _fit_grid_local(cls,
#                         location: str,
#                         xs: list,
#                         ) -> pd.DataFrame:
#
#         pool = mp.Pool(processes=mp.cpu_count() - 1)
#
#         def _param_iter() -> iter:
#             for params in cls._parameter_grid():
#                 yield xs, params
#
#         local_grid = pool.starmap(cls._eval_samples, _param_iter())
#         for entry in local_grid:
#             entry['location'] = location
#
#         # Store all entries in a DataFrame
#         df = pd.DataFrame(local_grid)
#         # Set the index
#         df.set_index(['location', 'i_t_base', 'i_chill_req', 'i_growth_req'], inplace=True)
#
#         return df
#
#     @classmethod
#     def _eval_samples(cls, xs: list, params: tuple) -> dict:
#
#         # Obtain the data that is required to fit the model
#         doys_true = [x['bloom_ix'] for x in xs]
#
#         # Unpack parameter values
#         (i_tb, tb), (i_cr, cr), (i_gr, gr) = params
#
#         # Initialize a model using parameters
#         model = cls()
#
#         model._fit_info = info = dict()
#
#         info['dataset_type'] = TemperatureBlossomDataset.__name__
#
#         t_tb = torch.tensor([tb]).view(1, 1)
#         t_cr = torch.tensor([cr]).view(1, 1) / cls.SCALE_CHILL
#         t_gr = torch.tensor([gr]).view(1, 1) / cls.SCALE_GROWTH
#
#         def _f_parameters(x):
#             return t_cr, t_gr, t_tb
#
#         model.f_parameters = _f_parameters
#
#         with torch.no_grad():
#             # Make predictions
#             results = model.batch_predict_ix(xs)
#             doys_pred = [doy for doy, _, _ in results]
#
#         # Compute metrics
#         mse = mean_squared_error(doys_true, doys_pred)
#         r2 = r2_score(doys_true, doys_pred)
#
#         return {
#             'i_t_base': i_tb,
#             't_base': tb,
#             'i_chill_req': i_cr,
#             'chill_req': cr,
#             'i_growth_req': i_gr,
#             'growth_req': gr,
#             'mse': mse,
#             'r2': r2,
#             'n': len(xs),
#         }
#
#     @classmethod
#     def _parameter_grid(cls) -> iter:
#         """
#         Iterate through all possible combinations of parameters
#         """
#         return product(
#             enumerate(cls._grid_tbs()),
#             enumerate(cls._grid_crs()),
#             enumerate(cls._grid_grs()),
#         )
#
#     @classmethod
#     def _grid_tbs(cls) -> np.ndarray:
#         """
#         :return: a list of base temperatures that will be evaluated during grid search
#         """
#         raise NotImplementedError
#
#     @classmethod
#     def _grid_crs(cls) -> np.ndarray:
#         """
#         :return: a list of chill requirements that will be evaluated during grid search
#         """
#         raise NotImplementedError
#
#     @classmethod
#     def _grid_grs(cls) -> np.ndarray:
#         """
#         :return: a list of growth requirements that will be evaluated during grid search
#         """
#         raise NotImplementedError
#
#     @classmethod
#     def load_from_grid_search(cls, run_name: str) -> 'BaseTorchAccumulationModelLocal':
#
#         path = os.path.join(cls._path_params_dir(), run_name, f'{run_name}_grid_parameter_fits_best_mse.csv')
#
#         df = pd.read_csv(path,
#                          index_col=[0],
#                          )
#
#         param_dict = dict()
#
#         for row in df.itertuples():
#             param_dict[row.index] = (row.chill_req, row.growth_req, row.t_base)
#
#         model = cls(
#             location_params=param_dict,
#         )
#
#         return model
