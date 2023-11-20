import os
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import multiprocessing as mp

import config
from datasets.dataset import Dataset
from datasets.dataset_torch import TorchDatasetWrapper, TorchDataset
from models.base_torch import BaseTorchModel
from models.components.logistic import SoftThreshold
from models.components.param_v2 import ParameterModel, ParameterMapping
# from models.components.param import ParamNet, LocalParams, ParameterModel
from util.torch import batch_tensors


class BaseTorchAccumulationModel(BaseTorchModel):

    SLOPE = 50
    # SLOPE = 1
    SCALE_CHILL = Dataset.SEASON_LENGTH
    SCALE_GROWTH = Dataset.SEASON_LENGTH

    LOSSES = [
        'mse',  # Mean Squared Error
        'bce',  # Binary Cross-Entropy
        'nll',  # Negative Log-Likelihood
    ]

    INFERENCE_MODES = [
        'max_p',
        'count_soft',
        'count_hard',
    ]

    def __init__(self,
                 param_model: ParameterModel,
                 soft_threshold_at_eval: bool = True,
                 loss_f: str = LOSSES[0],
                 inference_mode_train: str = INFERENCE_MODES[0],
                 inference_mode_test: str = INFERENCE_MODES[0],
                 ):
        assert loss_f in BaseTorchAccumulationModel.LOSSES
        super().__init__()
        self._param_model = param_model

        # When set to False -> hard threshold (i.e. step function) is used during inference in evaluation mode
        self._soft_threshold_at_eval = soft_threshold_at_eval
        # When set to False -> hard threshold is used in the forward pass
        # This variable is set in the functions setting train/test modes for this model
        # Its behaviour depends on the value of self._soft_threshold_at_eval
        self._soft_threshold = True

        # Set the loss function that is to be used for optimization
        self._loss_f = loss_f

        # The model outputs intermediate variables during debug mode
        self._debug_mode = False  # TODO

        # self._slope = torch.nn.Parameter(torch.tensor(1.))  # TODO
        from data.bloom_doy import get_locations_japan, get_locations_switzerland  # TODO -- clean
        # locations = get_locations_japan()
        locations = get_locations_switzerland()
        self._slope = ParameterMapping(
            {loc: i for i, loc in enumerate(locations)},
            init_val=1.0,
        )

    @property
    def parameter_model(self) -> ParameterModel:
        return self._param_model

    def f_parameters(self, xs: dict) -> tuple:
        return self._param_model.get_parameters(xs)

    def f_units_chill_growth(self, xs: dict, tb: torch.Tensor):
        raise NotImplementedError

    def forward(self,
                xs: dict,
                ) -> tuple:

        debug_info = dict()

        th_c, th_g, tb_g = self.f_parameters(xs)

        units_c, units_g = self.f_units_chill_growth(xs, tb_g)

        units_c = units_c / BaseTorchAccumulationModel.SCALE_CHILL
        units_g = units_g / BaseTorchAccumulationModel.SCALE_GROWTH

        units_c_cs = units_c.cumsum(dim=-1)

        req_c = SoftThreshold.f_soft_threshold(units_c_cs,
                                               BaseTorchAccumulationModel.SLOPE,
                                               th_c,
                                               )

        units_g_masked = units_g * req_c
        # units_g_masked = units_g * req_c + 1e-3
        units_g_cs = units_g_masked.cumsum(dim=-1)

        req_g = SoftThreshold.f_soft_threshold(units_g_cs,
                                               # BaseTorchAccumulationModel.SLOPE,
                                               # 1 / torch.clamp(self._slope.to(units_g_cs.device), min=0.01),  # TODO
                                               # 1 / (self._slope**2 + 0.01),
                                               1 / (self._slope(xs) ** 2 + 0.01),
                                               th_g,
                                               )

        # ix_neg = (req_g - torch.roll(req_g, 1, dims=-1)) < 0
        # eps = 1e-6
        # req_g = req_g + (ix_neg * eps).cumsum(dim=-1)

        req_g = req_g + torch.arange(274).view(1, -1).expand(req_g.size(0), -1).to(req_g.device) * 1e-6

        # print((units_g_masked < 0).sum())
        #
        # bixs = torch.arange(req_g.size(0)).to(req_g.device)

        # print(((req_g - torch.roll(req_g, 1, dims=-1))[bixs, 1:] < 0).sum())

        # rtrt = 0
        # for e in (req_g - torch.roll(req_g, 1, dims=-1))[bixs, 1:].view(-1):
        #
        #     if e < 0:
        #         print(e.item())
        #         rtrt += 1
        #
        #     if rtrt > 10:
        #         exit()

        # print(self._slope.item())

        """
            Compute the blooming ix 
        """

        if self._soft_threshold:
            ix = (1 - req_g).sum(dim=-1)
        else:
            mask = (units_g_cs >= th_g).to(torch.int)
            ix = (1 - mask).sum(dim=-1)

        # If blooming never occurred -> set it to the end of the season
        ix = ix.clamp(min=0, max=Dataset.SEASON_LENGTH - 1)

        ix = torch.argmax(req_g - torch.roll(req_g, 1, dims=-1), dim=-1)

        # ix = (1 - req_g).sum(dim=-1)
        # ix = ix.clamp(min=0, max=Dataset.SEASON_LENGTH - 1)

        # TODO -- remove
        # dist = torch.distributions.LogNormal(tb_g, torch.ones_like(tb_g) * self._slope)
        # dist = torch.distributions.Normal(tb_g, torch.ones_like(tb_g) * (self._slope**2 + 0.01))
        # _ps = dist.log_prob(units_g_cs)
        # _sigma = self._slope(xs)**2 + 0.01
        # _sigma = 1
        # _p_bloom = (1 / (_sigma * 2 * torch.pi) * torch.exp(-.5 * ((units_g_cs - tb_g) / _sigma) ** 2))

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

            # 'tb_g': th_g,

            # 'units_g_cs': units_g_cs,


            # '_p_bloom': _p_bloom,
            # '_p_not_bloom': 1 - _p_bloom,  # probability of not blooming given that there has been no bloom yet
            # '_survival': (1 - _p_bloom).cumprod(dim=-1),
            # '_p_bloom_t': (1 - _p_bloom).cumprod(dim=-1) * torch.roll(_p_bloom, 1, dims=-1),  # TODO -- first element not valid

            **optional_info,
        }

    def loss(self, xs: dict, scale: float = 1e-3) -> tuple:
        if self._loss_f == 'mse':
            return super().loss(xs, scale=scale)
        if self._loss_f == 'bce':
            return self._bce_loss(xs, scale=scale)
        if self._loss_f == 'nll':
            return self._nll_loss(xs, scale=scale)
        raise Exception('Unrecognized loss function')

    def _bce_loss(self, xs: dict, scale: float = 1e-3) -> tuple:
        _, info = self(xs)

        req_g_pred = info['req_g']
        bs = req_g_pred.size(0)

        ys_true = xs['bloom_ix'].to(config.TORCH_DTYPE).to(req_g_pred.device)

        req_g_true = torch.cat(
            [torch.arange(Dataset.SEASON_LENGTH).unsqueeze(0) for _ in range(bs)], dim=0
        ).to(req_g_pred.device)
        req_g_true = (req_g_true >= ys_true.view(-1, 1)).to(config.TORCH_DTYPE)

        loss = F.binary_cross_entropy(req_g_pred, req_g_true)

        return loss, {
            'forward_pass': info,
        }

    def _nll_loss(self, xs: dict, scale: float = 1) -> tuple:
        _, info = self(xs)

        req_g_pred = info['req_g']

        ys_true = xs['bloom_ix']

        bixs = torch.arange(ys_true.size(0)).to(ys_true.device)

        # print('req g')
        # print(req_g_pred)
        # print('ys')
        # print(ys_true)
        # print('r[t]')
        # print(req_g_pred[bixs, ys_true])
        # print('r[t-1]')
        # print(req_g_pred[bixs, ys_true - 1])

        # print('hoi')
        # print(req_g_pred.shape)  # bs x 274
        # print(ys_true.shape)
        # print(req_g_pred[:, ys_true].shape)
        # print(req_g_pred[bixs, ys_true].shape)

        # _ps_t = (1 - info['_ps']).cumprod(dim=-1)[bixs, ys_true]
        # _ps_tmin1 = (1 - info['_ps']).cumprod(dim=-1)[bixs, ys_true - 1]
        #
        # ps = _ps_tmin1 - _ps_t

        # ps = torch.index_select(req_g_pred, dim=1, index=ys_true) - torch.index_select(req_g_pred, dim=1, index=ys_true - 1)

        ps = req_g_pred[bixs, ys_true] - req_g_pred[bixs, ys_true - 1]

        # print(' '.join([f'{e.item():.2f}' for e in req_g_pred[0] - torch.roll(req_g_pred[0], 1, dims=-1)]))

        # ps = (1 - info['_ps']).cumprod(dim=-1)[bixs, ys_true - 1] * info['_ps'][bixs, ys_true]

        # ps = info['_p_bloom_t'][bixs, ys_true + 1]

        # ps = (1 - req_g_pred[ys_true - 1]) - (1 - req_g_pred[ys_true])

        # print(((req_g_pred[bixs, ys_true - 1] > req_g_pred[bixs, ys_true]) * req_g_pred[bixs, ys_true]).sum())
        # print(((req_g_pred[bixs, ys_true - 1] > req_g_pred[bixs, ys_true]) * req_g_pred[bixs, ys_true - 1]).sum())

        # for _i, (_p1, _p2) in enumerate(zip(req_g_pred[bixs, ys_true], req_g_pred[bixs, ys_true - 1])):
        #     if _p1.item() < _p2.item():
        #         print('Culprit:')
        #         print(_p1.item())
        #         print(_p2.item())
        #         print(ys_true[_i].item())
        #         print(info['tb_g'][_i].item())
        #         print()

        # print(req_g_pred[bixs, ys_true - 1])
        # print(req_g_pred[bixs, ys_true])

        # print(ps.max())
        # print(ps.min())
        # print(torch.isnan(ps).sum())

        # print('p')
        # print(ps)

        # loss = F.binary_cross_entropy(req_g_pred[bixs, ys_true + 5], torch.ones_like(req_g_pred[bixs, ys_true]))\
        # + F.binary_cross_entropy(req_g_pred[bixs, ys_true - 1 - 5], torch.zeros_like(req_g_pred[bixs, ys_true - 1]))

        # loss = -torch.log(ps + 1e-5).mean()
        loss = F.binary_cross_entropy(ps, torch.ones_like(ps))
        # loss = F.binary_cross_entropy(torch.clamp(ps, min=0, max=1), torch.ones_like(ps))
        # z = input()

        return loss, {
            'forward_pass': info,
        }

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

    def set_mode_train(self):
        self._soft_threshold = True
        super().set_mode_train()

    def set_mode_test(self):
        if not self._soft_threshold_at_eval:
            self._soft_threshold = False
        super().set_mode_test()

    @classmethod
    def _fit_grid(cls,
                  dataset: Dataset,
                  name: str,
                  model_kwargs: dict = None,
                  model: 'BaseTorchAccumulationModel' = None,
                  ):
        model_kwargs = model_kwargs or dict()

        model = model or cls(**model_kwargs)

        # Do a grid search for locations separately
        # Store results in a list of dataframes
        dfs = []

        model._fit_info = info = dict()  # TODO -- this is deprecated
        info['dataset_type'] = type(dataset).__name__

        # Keep a progress bar of the fitting process
        location_progress = tqdm(dataset.locations_train)

        # Iterate through all locations
        for location in location_progress:
            location_progress.set_description(f'{cls.__name__} Grid Search [Current: {location}]')

            # Get the data corresponding to this location
            data_local = dataset.get_local_train_data(location)
            # Perform a grid search in one location
            df_local = cls._fit_grid_local(model, location, data_local)
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
        best_fits = pd.DataFrame([df.loc[df['mse'].idxmin()] for df in dfs])
        best_fits.to_csv(os.path.join(path, f'{name}_grid_parameter_fits_best_mse.csv'))

        # Initialize a model from the saved parameters and return it
        # model = cls.load(name)  # TODO -- load proper
        model = None
        # TODO -- set loss f back to original
        return model, {}

    @classmethod
    def _fit_grid_local(cls,
                        model: 'BaseTorchAccumulationModel',
                        location: str,
                        xs: list,
                        ) -> pd.DataFrame:

        with torch.no_grad():
            pool = mp.Pool(processes=mp.cpu_count() - 1)

            def _param_iter() -> iter:
                for params in cls._parameter_grid():
                    yield model, xs, params

            # cls._eval_samples(*next(_param_iter()))  # For obtaining a stacktrace while debugging

            local_grid = pool.starmap(cls._eval_samples, _param_iter())
            for entry in local_grid:
                entry['location'] = location

            # Store all entries in a DataFrame
            df = pd.DataFrame(local_grid)
            # Set the index
            df.set_index(['location', 'i_t_base', 'i_chill_req', 'i_growth_req'], inplace=True)

        return df

    @classmethod
    def _eval_samples(cls, model: 'BaseTorchAccumulationModel', xs: list, params: tuple) -> dict:

        # Unpack parameter values
        (i_tb, tb), (i_cr, cr), (i_gr, gr) = params

        t_tb = torch.tensor([tb]).view(1, 1)
        t_cr = torch.tensor([cr]).view(1, 1) / cls.SCALE_CHILL
        t_gr = torch.tensor([gr]).view(1, 1) / cls.SCALE_GROWTH

        doys_true = [x['bloom_ix'] for x in xs]

        def _get_params(*_) -> tuple:
            return t_cr, t_gr, t_tb

        model.f_parameters = _get_params

        results = model.batch_predict_ix(xs)
        doys_pred = [doy for doy, _, _ in results]

        # Compute metrics
        mse = mean_squared_error(doys_true, doys_pred)
        r2 = r2_score(doys_true, doys_pred)

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
    def _parameter_grid(cls) -> iter:
        """
        Iterate through all possible combinations of parameters
        """
        return product(
            enumerate(cls._grid_tbs()),
            enumerate(cls._grid_crs()),
            enumerate(cls._grid_grs()),
        )

    @classmethod
    def _grid_tbs(cls) -> np.ndarray:
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
        cr_min = 0
        cr_max = BaseTorchAccumulationModel.SCALE_CHILL
        cr_step = BaseTorchAccumulationModel.SCALE_CHILL // 80
        return np.arange(cr_min, cr_max, cr_step)

    @classmethod
    def _grid_grs(cls) -> np.ndarray:
        """
        :return: a list of growth requirements that will be evaluated during grid search
        """
        gr_min = 0
        gr_max = BaseTorchAccumulationModel.SCALE_GROWTH
        gr_step = BaseTorchAccumulationModel.SCALE_GROWTH // 80
        return np.arange(gr_min, gr_max, gr_step)


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
