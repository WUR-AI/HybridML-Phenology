import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class GeneralizedLogistic(nn.Module):

    def __init__(self,
                 alpha: float,
                 beta: float,
                 omega: float,
                 alpha_req_grad: bool = True,
                 beta_req_grad: bool = True,
                 omega_req_grad: bool = True,
                 dtype=config.TORCH_DTYPE,
                 ):
        super().__init__()

        a = torch.tensor(float(alpha)).to(dtype)
        b = torch.tensor(float(beta)).to(dtype)
        o = torch.tensor(float(omega)).to(dtype)

        self._a = nn.Parameter(a, requires_grad=alpha_req_grad)
        self._b = nn.Parameter(b, requires_grad=beta_req_grad)
        self._o = nn.Parameter(o, requires_grad=omega_req_grad)

    def forward(self, ts: torch.Tensor):
        return self.f_logistic(ts, self._a, self._b, self._o)

    def __str__(self):
        return f'logistic(x ; {self._a.data.item():.2f}, {self._b.data.item():.2f}, {self._o.data.item():.2f})'

    @staticmethod
    def f_logistic(ts: torch.Tensor,
                   a,
                   b,
                   o,
                   ):
        out = o * F.sigmoid(a * (ts - b))
        return out
        # return o / (1 + torch.exp(-a * (ts - b.view(-1, 1))))


class SoftThreshold(GeneralizedLogistic):

    def __init__(self,
                 alpha: float,
                 beta: float,
                 alpha_req_grad: bool = True,
                 beta_req_grad: bool = True,
                 ):
        super().__init__(
            alpha,
            beta,
            1,
            alpha_req_grad=alpha_req_grad,
            beta_req_grad=beta_req_grad,
            omega_req_grad=False,
        )

    @staticmethod
    def f_soft_threshold(ts: torch.Tensor,
                         a,
                         b,
                         ):
        return SoftThreshold.f_logistic(ts, a, b, 1)
