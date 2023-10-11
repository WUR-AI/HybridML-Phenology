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


class ParameterMapping(ParameterModel, nn.Module):

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

    def get_parameters(self, xs: dict):
        return self(xs)

    def forward(self, xs: dict):
        locations = xs['location']

        ps = torch.cat([self._ps[self._loc_ixs[loc]].view(1, 1) for loc in locations],
                       dim=0,
                       )

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


def _modified_relu(x: torch.Tensor):
    w = 1
    return F.relu(x) + w * (x - x.detach().clone())
