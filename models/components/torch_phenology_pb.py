

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from datasets.dataset_torch import TorchDatasetWrapper
from models.components.logistic import GeneralizedLogistic, SoftThreshold
from util.torch import batch_tensors

from util.normalization import mean_std_normalize


class TorchGDD(nn.Module):

    def __init__(self,
                 threshold: float,
                 threshold_req_grad: bool = True,
                 dtype=config.TORCH_DTYPE,
                 ):
        super().__init__()

        tb = torch.tensor(float(threshold)).to(dtype)
        self._tb = nn.Parameter(tb, requires_grad=threshold_req_grad)

    def forward(self, xs: dict) -> torch.Tensor:
        return self.f_gdd(xs, self._tb)

    @staticmethod
    def f_gdd(xs: dict, tb: torch.Tensor):
        ts = _get_unnormalized_temperature(xs)
        return F.relu(ts - tb.view(-1, 1, 1)).sum(dim=-1) / ts.shape[-1]


class UtahChillModule(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, xs: dict) -> torch.Tensor:
        ts = _get_unnormalized_temperature(xs)
        return self.f_chill(ts)

    @staticmethod
    def f_chill(ts: torch.Tensor) -> torch.Tensor:
        bin_0 = (ts <= 1.4).sum(dim=-1).to(ts.dtype)
        bin_1 = ((1.4 < ts) & (ts <= 2.4)).sum(dim=-1).to(ts.dtype)
        bin_2 = ((2.4 < ts) & (ts <= 9.1)).sum(dim=-1).to(ts.dtype)
        bin_3 = ((9.1 < ts) & (ts <= 12.4)).sum(dim=-1).to(ts.dtype)
        bin_4 = ((11.4 < ts) & (ts <= 15.9)).sum(dim=-1).to(ts.dtype)
        bin_5 = ((15.9 < ts) & (ts <= 18)).sum(dim=-1).to(ts.dtype)
        bin_6 = (18 < ts).sum(dim=-1).to(ts.dtype)

        bin_0 *= 0.
        bin_1 *= 0.5
        bin_2 *= 1.
        bin_3 *= 0.5
        bin_4 *= 0.
        bin_5 *= -0.5
        bin_6 *= -1

        return bin_0 + bin_1 + bin_2 + bin_3 + bin_4 + bin_5 + bin_6


class LogisticUtahChillModule(nn.Module):

    ALPHA1 = 5
    ALPHA2 = 1
    BETA1 = 2
    BETA2 = 14
    OMEGA1 = 1
    OMEGA2 = -2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._f1 = GeneralizedLogistic(
            LogisticUtahChillModule.ALPHA1,
            LogisticUtahChillModule.BETA1,
            LogisticUtahChillModule.OMEGA1,
            alpha_req_grad=False,
            beta_req_grad=False,
            omega_req_grad=False,
        )

        self._f2 = GeneralizedLogistic(
            LogisticUtahChillModule.ALPHA2,
            LogisticUtahChillModule.BETA2,
            LogisticUtahChillModule.OMEGA2,
            alpha_req_grad=False,
            beta_req_grad=False,
            omega_req_grad=False,
        )

    def forward(self, xs: dict) -> torch.Tensor:
        ts = _get_unnormalized_temperature(xs)
        return F.relu((self._f1(ts) + self._f2(ts)).sum(dim=-1))


class LogisticChillHoursModule(nn.Module):

    ALPHA1 = 3
    ALPHA2 = 3
    BETA1 = 0
    BETA2 = 7.2
    OMEGA1 = 1
    OMEGA2 = -1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._f1 = GeneralizedLogistic(
            LogisticUtahChillModule.ALPHA1,
            LogisticUtahChillModule.BETA1,
            LogisticUtahChillModule.OMEGA1,
            alpha_req_grad=False,
            beta_req_grad=False,
            omega_req_grad=False,
        )

        self._f2 = GeneralizedLogistic(
            LogisticUtahChillModule.ALPHA2,
            LogisticUtahChillModule.BETA2,
            LogisticUtahChillModule.OMEGA2,
            alpha_req_grad=False,
            beta_req_grad=False,
            omega_req_grad=False,
        )

    def forward(self, xs: dict) -> torch.Tensor:
        ts = _get_unnormalized_temperature(xs)
        return F.relu((self._f1(ts) + self._f2(ts)).sum(dim=-1))


def _get_unnormalized_temperature(xs: dict) -> torch.Tensor:
    ts = TorchDatasetWrapper.normalize_temperature(xs['temperature'], revert=True)

    #
    # ts = xs['temperature']
    # device = ts.device
    # dtype = ts.dtype
    # ts = [x['temperature'] for x in xs['original']]
    # ts = [torch.tensor(x).to(device).to(dtype) for x in ts]
    # ts = batch_tensors(*ts)
    return ts
