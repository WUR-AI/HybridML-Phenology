import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd

import config
from data.bloom_doy import get_locations


class ParameterModel:

    def get_parameters(self, xs: dict):
        raise NotImplementedError


class ParamNet(ParameterModel, nn.Module):

    def __init__(self,
                 hidden_size: int = 1024,
                 apply_scaling: bool = True,
                 ):
        super().__init__()

        self._apply_scaling = apply_scaling

        self._lin1 = nn.Linear(in_features=3,
                               out_features=hidden_size,
                               )

        self._lin_a = nn.Linear(in_features=hidden_size,
                                out_features=hidden_size,
                                )

        self._lin_e = nn.Linear(in_features=hidden_size,
                                out_features=hidden_size,
                                )

        self._lin2 = nn.Linear(in_features=hidden_size,
                               out_features=3,
                               )

    def forward(self, xs: dict):
        lat = xs['lat']
        lon = xs['lon']
        alt = xs['alt']

        xs = torch.cat(
            [
                lat.unsqueeze(-1),
                lon.unsqueeze(-1),
                alt.unsqueeze(-1),
            ],
            dim=-1,
        )

        xs = self._lin1(xs)
        xs = F.relu(xs)

        xs = self._lin_a(xs)
        xs = F.relu(xs)

        xs = self._lin_e(xs)
        xs = F.relu(xs)

        xs = self._lin2(xs)

        th_c, th_g, tb_g = torch.tensor_split(xs, 3, dim=-1)

        if self._apply_scaling:

            th_c = self._scale_thc(th_c)
            th_g = self._scale_thg(th_g)
            tb_g = self._scale_tbg(tb_g)

        return th_c, th_g, tb_g

    def get_parameters(self, xs: dict):
        return self(xs)

    @staticmethod
    def _scale_thc(thc: torch.Tensor):
        w_thc = 1
        # w_thc = 4000
        thc = F.sigmoid(thc)
        thc = thc * w_thc
        return thc

    @staticmethod
    def _scale_thg(thg: torch.Tensor):
        w_thg = 1
        # w_thg = 2000
        thg = F.sigmoid(thg)
        thg = thg * w_thg
        return thg

    @staticmethod
    def _scale_tbg(tbg: torch.Tensor):
        w_tbg = 20
        tbg = F.sigmoid(tbg)
        tbg = tbg * w_tbg
        return tbg


class LocalParams(ParameterModel, nn.Module):

    def __init__(self,
                 init_th_c: float = 0.0,
                 init_th_g: float = 0.0,
                 init_tb_g: float = 0.0,
                 locations: list = None,
                 location_params: dict = None,
                 ):
        super().__init__()

        if location_params is None:

            locations = locations or get_locations()
            locations = list(set(locations))
            self._loc_ixs = {loc: i for i, loc in enumerate(locations)}

            self._th_c = nn.ParameterList(
                [torch.tensor(float(init_th_c), requires_grad=True) for _ in locations]
            )
            self._th_g = nn.ParameterList(
                [torch.tensor(float(init_th_g), requires_grad=True) for _ in locations]
            )
            self._tb_g = nn.ParameterList(
                [torch.tensor(float(init_tb_g), requires_grad=True) for _ in locations]
            )
        else:

            self._loc_ixs = dict()

            self._th_c = nn.ParameterList()
            self._th_g = nn.ParameterList()
            self._tb_g = nn.ParameterList()

            for i, (loc, params) in enumerate(location_params.items()):

                self._loc_ixs[loc] = i

                th_c, th_g, tb_g = params

                self._th_c.append(torch.tensor(float(th_c), requires_grad=True))
                self._th_g.append(torch.tensor(float(th_g), requires_grad=True))
                self._tb_g.append(torch.tensor(float(tb_g), requires_grad=True))

            self._loc_ixs = {loc: i for i, loc in enumerate(locations)}

    def get_parameters(self, xs: dict):
        return self(xs)

    def forward(self, xs: dict):
        locations = xs['location']

        th_c = torch.cat([self._th_c[self._loc_ixs[loc]].view(1, 1) for loc in locations],
                         dim=0,
                         )
        th_g = torch.cat([self._th_g[self._loc_ixs[loc]].view(1, 1) for loc in locations],
                         dim=0,
                         )
        tb_g = torch.cat([self._tb_g[self._loc_ixs[loc]].view(1, 1) for loc in locations],
                         dim=0,
                         )

        th_c = self._scale_thc(th_c)
        th_g = self._scale_thg(th_g)
        tb_g = self._scale_tbg(tb_g)

        return th_c, th_g, tb_g

    @staticmethod
    def _scale_thc(thc: torch.Tensor):
        w_thc = 1
        thc = _modified_relu(thc)
        thc = thc * w_thc
        return thc

    @staticmethod
    def _scale_thg(thg: torch.Tensor):
        w_thg = 1
        thg = _modified_relu(thg)
        thg = thg * w_thg
        return thg

    @staticmethod
    def _scale_tbg(tbg: torch.Tensor):
        w_tbg = 20
        tbg = _modified_relu(tbg)
        tbg = tbg * w_tbg
        return tbg

    def to_dataframe(self, locations: list = None):

        rows = []
        for loc, i in self._loc_ixs.items():

            th_c = self._scale_thc(self._th_c[i]).item()
            th_g = self._scale_thg(self._th_g[i]).item()
            tb_g = self._scale_tbg(self._tb_g[i]).item()

            rows.append({
                'location': loc,
                'th_c': th_c,
                'th_g': th_g,
                'tb_g': tb_g,
            })

        df = pd.DataFrame(rows)

        return df

    @classmethod
    def _path_params_dir(cls) -> str:
        return os.path.join(config.PATH_PARAMS_DIR, cls.__name__)

    def save_param_df(self, fn: str = f'local_params.csv'):
        path = self._path_params_dir()
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, fn)
        df = self.to_dataframe()
        df.to_csv(path)
        return df

    @classmethod
    def load_param_df(cls, fn: str = f'local_params.csv') -> pd.DataFrame:
        path = os.path.join(cls._path_params_dir(), fn)
        df = pd.read_csv(path, index_col=[0])
        df.set_index('location', inplace=True)
        return df


def _modified_relu(x: torch.Tensor):
    w = 1
    return F.relu(x) + w * (x - x.detach().clone())


if __name__ == '__main__':

    _net = ParamNet()
    _lps = LocalParams()

    _xs = {
        'location': ['Japan/Abashiri'],
        'lat': torch.tensor([1.0]),
        'lon': torch.tensor([1.0]),
        'alt': torch.tensor([1.0]),
    }

    _th_c, _th_g, _tb_g = _net(_xs)
    print(_th_c.shape)
    print(_th_g.shape)
    print(_tb_g.shape)

    _th_c, _th_g, _tb_g = _lps(_xs)
    print(_th_c.shape)
    print(_th_g.shape)
    print(_tb_g.shape)

    print(_lps.to_dataframe())
    _lps.save_param_df()

    print(_lps.load_param_df())