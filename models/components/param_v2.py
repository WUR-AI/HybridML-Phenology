import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd

import config
from data.bloom_doy import get_locations


class ParameterModel(nn.Module):

    def get_parameters(self, xs: dict) -> tuple:
        raise NotImplementedError


class AccumulationParameterMapping(ParameterModel):

    def __init__(self,
                 location_groups: dict,
                 init_val_th_c: float = 0.0,
                 init_val_th_g: float = 0.0,
                 init_val_tb_g: float = 0.0,
                 ):
        super().__init__()

        self._thc_map = ParameterMapping(
            location_groups,
            init_val=init_val_th_c,
        )

        self._thg_map = ParameterMapping(
            location_groups,
            init_val=init_val_th_g,
        )

        self._tbg_map = ParameterMapping(
            location_groups,
            init_val=init_val_tb_g,
        )

    def forward(self, xs: dict):

        thc = self._thc_map(xs)
        thg = self._thg_map(xs)
        tbg = self._tbg_map(xs)

        thc = self._scale_param(thc, 1)
        thg = self._scale_param(thg, 1)
        tbg = self._scale_param(tbg, 20)

        return thc, thg, tbg

    @staticmethod
    def _scale_param(p: torch.Tensor, c) -> torch.Tensor:
        p = _modified_relu(p)
        return p * c


class LocalAccumulationParameterMapping(AccumulationParameterMapping):

    def __init__(self,
                 locations: list,
                 init_val_th_c: float = 0.0,
                 init_val_th_g: float = 0.0,
                 init_val_tb_g: float = 0.0,
                 ):
        super().__init__(
            {loc: i for i, loc in enumerate(locations)},
            init_val_th_c,
            init_val_th_g,
            init_val_tb_g,
        )


class GlobalAccumulationParameterMapping(AccumulationParameterMapping):

    def __init__(self,
                 locations: list,
                 init_val_th_c: float = 0.0,
                 init_val_th_g: float = 0.0,
                 init_val_tb_g: float = 0.0,
                 ):
        super().__init__(
            {loc: 0 for loc in locations},
            init_val_th_c,
            init_val_th_g,
            init_val_tb_g,
        )


class ParameterMapping(nn.Module):

    def __init__(self,
                 location_groups: dict,
                 init_val: float = 0.0,
                 ):
        super().__init__()

        n_groups = len(set(location_groups.values()))

        group_to_ix = {
            g: i for i, g in enumerate(set(location_groups.values()))
        }

        self._loc_ixs = {
            loc: group_to_ix[g] for loc, g in location_groups.items()
        }

        self._ps = nn.ParameterList(
            [torch.tensor(float(init_val), requires_grad=True) for _ in range(n_groups)]
        )

    def forward(self, xs: dict):
        locations = xs['location']

        ps = torch.cat([self._ps[self._loc_ixs[loc]].view(1, 1) for loc in locations],
                       dim=0,
                       )

        return ps


def _modified_relu(x: torch.Tensor):
    w = 1
    return F.relu(x) + w * (x - x.detach().clone())
