import torch
from torch import nn
from torch.nn import LSTM

from datasets.dataset import Dataset
from models.base_torch import BaseTorchModel


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
