import torch
from torch import nn
from torch.nn import LSTM
import torch.nn.functional as F

import config
from datasets.dataset import Dataset
from models.base_torch import BaseTorchModel
from util.normalization import normalize_latitude, normalize_longitude


class LSTMModel(BaseTorchModel):

    def __init__(self):
        super().__init__()

        self._hidden_size = 128

        self._rnn = LSTM(input_size=24,
                         hidden_size=self._hidden_size,
                         batch_first=True,
                         num_layers=2,
                         )

        self._lin = nn.Linear(in_features=self._hidden_size,
                              out_features=1,
                              )

    def forward(self, xs: dict) -> tuple:

        ts = xs['temperature']

        out, _ = self._rnn(ts)

        ixs = self._lin(out[:, -1, :]).view(-1) * Dataset.SEASON_LENGTH

        ixs = ixs.clamp(min=0, max=Dataset.SEASON_LENGTH - 1)

        return ixs, {}


class LSTMModel2(BaseTorchModel):

    def __init__(self):
        super().__init__()

        self._hidden_size = 128

        self._rnn = LSTM(input_size=24,
                         hidden_size=self._hidden_size,
                         batch_first=True,
                         num_layers=2,
                         )

        self._lin = nn.Conv2d(in_channels=1,
                              out_channels=1,
                              kernel_size=(1, self._hidden_size),
                              )

        # self._lin = nn.Linear(in_features=self._hidden_size,
        #                       out_features=1,
        #                       )

    def forward(self, xs: dict) -> tuple:

        ts = xs['temperature']

        out, _ = self._rnn(ts)  # Shape: (batch_size, series length, hidden_size)

        out = out.unsqueeze(1)

        ps = F.sigmoid(self._lin(out))

        # Remove the final dimension
        ps = ps.squeeze(-1)
        # Remove the channel dimension
        ps = ps.squeeze(1)

        ixs = torch.argmax(ps - torch.roll(ps, 1, dims=-1), dim=-1)
        ixs = ixs.clamp(min=0, max=Dataset.SEASON_LENGTH - 1)

        return ixs, {
            'ps': ps,
        }

    def loss(self, xs: dict, scale: float = 1e-3) -> tuple:
        _, info = self(xs)

        ps = info['ps']
        bs = ps.size(0)

        ys_true = xs['bloom_ix'].to(config.TORCH_DTYPE).to(ps.device)

        ps_true = torch.cat(
            [torch.arange(Dataset.SEASON_LENGTH).unsqueeze(0) for _ in range(bs)], dim=0
        ).to(ps.device)
        ps_true = (ps_true >= ys_true.view(-1, 1)).to(config.TORCH_DTYPE)

        loss = F.binary_cross_entropy(ps, ps_true)

        return loss, {
            'forward_pass': info,
        }


class LSTMModel3(BaseTorchModel):

    def __init__(self):
        super().__init__()

        self._hidden_size = 128

        self._rnn = LSTM(input_size=24 + 3,
                         hidden_size=self._hidden_size,
                         batch_first=True,
                         num_layers=2,
                         )

        self._lin = nn.Conv2d(in_channels=1,
                              out_channels=1,
                              kernel_size=(1, self._hidden_size),
                              )

    def forward(self, xs: dict) -> tuple:

        ts = xs['temperature']

        lats = xs['lat'].view(-1, 1).expand(-1, ts.size(1)).unsqueeze(-1)  # shape: (batch_size, season length, 1)
        lons = xs['lon'].view(-1, 1).expand(-1, ts.size(1)).unsqueeze(-1)  # shape: (batch_size, season length, 1)

        lats = normalize_latitude(lats, revert=True)
        lons = normalize_longitude(lons, revert=True)

        # https://datascience.stackexchange.com/questions/13567/ways-to-deal-with-longitude-latitude-feature
        xs = torch.cat(
            [
                ts,
                torch.cos(lats) * torch.cos(lons),
                torch.cos(lats) * torch.sin(lons),
                torch.sin(lats),
            ],
            dim=-1,
        )

        out, _ = self._rnn(xs)  # Shape: (batch_size, series length, hidden_size)

        out = out.unsqueeze(1)

        ps = F.sigmoid(self._lin(out))

        # Remove the final dimension
        ps = ps.squeeze(-1)
        # Remove the channel dimension
        ps = ps.squeeze(1)

        ixs = torch.argmax(ps - torch.roll(ps, 1, dims=-1), dim=-1)
        ixs = ixs.clamp(min=0, max=Dataset.SEASON_LENGTH - 1)

        return ixs, {
            'ps': ps,
        }

    def loss(self, xs: dict, scale: float = 1e-3) -> tuple:
        _, info = self(xs)

        ps = info['ps']
        bs = ps.size(0)

        ys_true = xs['bloom_ix'].to(config.TORCH_DTYPE).to(ps.device)

        ps_true = torch.cat(
            [torch.arange(Dataset.SEASON_LENGTH).unsqueeze(0) for _ in range(bs)], dim=0
        ).to(ps.device)
        ps_true = (ps_true >= ys_true.view(-1, 1)).to(config.TORCH_DTYPE)

        loss = F.binary_cross_entropy(ps, ps_true)

        return loss, {
            'forward_pass': info,
        }
