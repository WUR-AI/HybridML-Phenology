import os
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import multiprocessing as mp

import config
from datasets.dataset import Dataset
from models.base_torch import BaseTorchModel
from models.components.logistic import SoftThreshold
from models.components.param_v3 import ParameterModel
from util.torch import batch_tensors


class BaseTorchAccumulationModel(BaseTorchModel):

    SLOPE_CHILL = 50
    SLOPE_GROWTH = 50

    SCALE_CHILL = Dataset.SEASON_LENGTH
    SCALE_GROWTH = Dataset.SEASON_LENGTH

    LOSSES = [
        'nll',  # Negative Log-Likelihood
        'mse',  # Mean Squared Error
        'bce',  # Binary Cross-Entropy
    ]

    INFERENCE_MODES = [
        'max_p',
        'count_soft',
        'count_hard',
    ]

    def __init__(self,
                 parameter_model_thc: ParameterModel,
                 parameter_model_thg: ParameterModel,
                 parameter_model_tbg: ParameterModel,
                 parameter_model_slc: ParameterModel,
                 parameter_model_slg: ParameterModel,
                 loss_f: str = LOSSES[0],
                 inference_mode_train: str = INFERENCE_MODES[0],
                 inference_mode_test: str = INFERENCE_MODES[0],
                 ):
        assert loss_f in BaseTorchAccumulationModel.LOSSES
        assert inference_mode_train in BaseTorchAccumulationModel.INFERENCE_MODES
        assert inference_mode_test in BaseTorchAccumulationModel.INFERENCE_MODES
        super().__init__()

        self._pm_thc = parameter_model_thc
        self._pm_thg = parameter_model_thg
        self._pm_tbg = parameter_model_tbg
        self._pm_slc = parameter_model_slc
        self._pm_slg = parameter_model_slg

        # Set the loss function that is to be used for optimization
        self._loss_f = loss_f

        # The model outputs intermediate variables during debug mode
        self._debug_mode = False

        self._inf_mode_train = inference_mode_train
        self._inf_mode_test = inference_mode_test
        self._inf_mode = inference_mode_train

        self._clip_slope = 0.01

    def f_units(self, xs: dict, tb: torch.Tensor):
        raise NotImplementedError

    def forward(self,
                xs: dict,
                ) -> tuple:

        debug_info = dict()

        th_c = self._pm_thc.get_parameters(xs)
        th_g = self._pm_thg.get_parameters(xs)
        tb_g = self._pm_tbg.get_parameters(xs)
        sl_c = self._pm_slc.get_parameters(xs)
        sl_g = self._pm_slg.get_parameters(xs)

        units_c, units_g = self.f_units(xs, tb_g)

        units_c = units_c / BaseTorchAccumulationModel.SCALE_CHILL
        units_g = units_g / BaseTorchAccumulationModel.SCALE_GROWTH

        units_c_cs = units_c.cumsum(dim=-1)

        req_c = SoftThreshold.f_soft_threshold(units_c_cs,
                                               1 / (sl_c ** 2 + self._clip_slope),
                                               th_c,
                                               )

        units_g_masked = units_g * req_c
        # units_g_masked = units_g * req_c + 1e-3
        units_g_cs = units_g_masked.cumsum(dim=-1)

        req_g = SoftThreshold.f_soft_threshold(units_g_cs,
                                               1 / (sl_g ** 2 + self._clip_slope),
                                               th_g,
                                               )

        req_g = req_g + torch.arange(274).view(1, -1).expand(req_g.size(0), -1).to(req_g.device) * 1e-6

        """
            Compute the blooming ix 
        """
        if self._inf_mode == 'max_p':
            ix = torch.argmax(req_g - torch.roll(req_g, 1, dims=-1), dim=-1)
        elif self._inf_mode == 'count_soft':
            ix = (1 - req_g).sum(dim=-1)
        elif self._inf_mode == 'count_hard':
            mask = (units_g_cs >= th_g).to(torch.int)
            ix = (1 - mask).sum(dim=-1)
        else:
            raise Exception(f'Unknown inference mode: {self._inf_mode}')

        # If blooming never occurred -> set it to the end of the season
        ix = ix.clamp(min=0, max=Dataset.SEASON_LENGTH - 1)

        # Optionally save debug info
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

        ps = req_g_pred[bixs, ys_true] - req_g_pred[bixs, ys_true - 1]

        loss = F.binary_cross_entropy(ps, torch.ones_like(ps))

        return loss, {
            'forward_pass': info,
        }

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
        self._inf_mode = self._inf_mode_train
        super().set_mode_train()

    def set_mode_test(self):
        self._inf_mode = self._inf_mode_test
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

