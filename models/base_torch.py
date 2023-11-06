import os

from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import config
from datasets.dataset import Dataset
from datasets.dataset_torch import TorchDatasetWrapper
from models.base import BaseModel


class BaseTorchModel(BaseModel, nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()
        super(nn.Module, self).__init__()

        # Info about fitting process is required to be stored to be able to make predictions for single data points
        # For example, depending on the dataset that was used, different normalization/data conversion procedures are
        # required.
        # When fitting the model the following variable will store a dict containing this info
        self._fit_info = None

        # TODO -- store train locations/years -- use for checks for data leakage when evaluating

    def forward(self, xs: dict) -> tuple:
        raise NotImplementedError

    def predict_ix(self, x: dict) -> tuple:
        assert self._fit_info is not None

        with torch.no_grad():

            x = self._transform(x)
            x = self._normalize_sample(x)
            x = self._collate_fn([x])

            ix, info = self(x)

            [ix] = self._ixs_to_int(ix)

        return ix, True, {'forward_pass': info}

    def batch_predict_ix(self, xs: list) -> list:
        assert self._fit_info is not None

        with torch.no_grad():
            xs = [self._transform(x) for x in xs]
            xs = [self._normalize_sample(x) for x in xs]
            xs = self._collate_fn(xs)
            ixs, info = self(xs)
            # ixs = [int(ix.item()) for ix in ixs]
            ixs = self._ixs_to_int(ixs)

        return [(ix, True, info) for ix in ixs]

    @staticmethod
    def _ixs_to_int(ixs: torch.Tensor) -> list:
        ixs = [int(ix.item() + 0.5) for ix in ixs]
        return ixs

    def batch_predict(self, xs: list) -> list:
        return [(Dataset.index_to_doy(ix), b, i) for ix, b, i in self.batch_predict_ix(xs)]

    def _normalize_sample(self, xs: dict):
        assert self._fit_info is not None  # TODO -- is this still required?
        return TorchDatasetWrapper.normalize(xs)

    def _collate_fn(self, samples: list):
        assert self._fit_info is not None
        return TorchDatasetWrapper.collate_fn(samples)

    def _transform(self, xs: dict):
        assert self._fit_info is not None
        return TorchDatasetWrapper.cast_sample_to_tensors(xs)

    def loss(self, xs: dict, scale: float = 1e-3) -> tuple:
        ys_pred, info = self(xs)
        ys_true = xs['bloom_ix'].to(config.TORCH_DTYPE).to(ys_pred.device)
        loss = F.mse_loss(ys_pred, ys_true) * scale
        # loss = F.l1_loss(ys_pred, ys_true) * scale
        return loss, {
            'forward_pass': info,
        }

    @classmethod
    def fit(cls,
            dataset: Dataset,
            num_epochs: int = 1,
            batch_size: int = None,
            scheduler_step_size: int = None,
            scheduler_decay: float = 0.5,
            clip_gradient: float = None,
            f_optim: callable = torch.optim.SGD,  # TODO -- default is none -> init sgd with lr 0.1 or adam
            optim_kwargs: dict = None,
            model: 'BaseTorchModel' = None,
            model_kwargs: dict = None,
            # optim_kwargs: dict = {'lr': 0.1},
            device: str = torch.device('cpu'),
            ) -> tuple:
        assert num_epochs >= 0
        model_kwargs = model_kwargs or dict()
        optim_kwargs = optim_kwargs or dict()

        model = (model or cls(**model_kwargs)).to(device).to(config.TORCH_DTYPE)

        # TODO -- show model name in progress bar?

        model._fit_info = info = dict()

        info['dataset_type'] = type(dataset).__name__
        # TODO -- store normalization parameters

        dataset_wrapped = TorchDatasetWrapper(dataset)

        data_train = dataset_wrapped.get_train_data(device=device)

        batch_size = batch_size or len(data_train)

        dataloader_train = torch.utils.data.DataLoader(data_train,
                                                       batch_size=batch_size,
                                                       collate_fn=dataset_wrapped.collate_fn,
                                                       shuffle=True,
                                                       )

        optimizer = f_optim(model.parameters(), **optim_kwargs)

        scheduler_step_size = scheduler_step_size or num_epochs
        scheduler = StepLR(optimizer,
                           step_size=scheduler_step_size,
                           gamma=scheduler_decay,
                           )

        for epoch in range(num_epochs):

            train_iter = tqdm(dataloader_train,
                              total=len(dataloader_train),
                              )

            losses = []

            model.train()
            for xs in train_iter:
                xs = model._normalize_sample(xs)

                optimizer.zero_grad()

                loss, _ = model.loss(xs)

                loss.backward()

                if clip_gradient is not None:
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_gradient)

                optimizer.step()

                losses.append(loss.item())
                loss_mean = sum(losses) / len(losses)

                lr = scheduler.get_last_lr()[0]

                train_iter.set_description(
                    f'{cls.__name__} training epoch [{epoch + 1:6d}/{num_epochs}] | lr: {lr:.7f} | Batch Loss: {loss.item():8.5f} | Loss Mean: {loss_mean:8.5f}',
                )
            scheduler.step()

        return model, {}

    def save_state(self, model_name: str):
        fn = f'{model_name}_state.pth'
        os.makedirs(os.path.join(config.PATH_PARAMS_DIR, type(self).__name__), exist_ok=True)
        path = os.path.join(config.PATH_PARAMS_DIR, type(self).__name__, fn)
        torch.save((self.state_dict(), self._fit_info), path)

    def save(self, model_name: str):
        fn = f'{model_name}.pth'
        os.makedirs(os.path.join(config.PATH_PARAMS_DIR, type(self).__name__), exist_ok=True)
        path = os.path.join(config.PATH_PARAMS_DIR, type(self).__name__, fn)
        self.cpu()
        torch.save(self, path)

    @classmethod
    def load_from_state_dict(cls, model_kwargs: dict, model_name: str) -> 'BaseTorchModel':
        fn = f'{model_name}_state.pth'
        path = os.path.join(config.PATH_PARAMS_DIR, cls.__name__, fn)
        model = cls(**model_kwargs)
        state, fit_info = torch.load(path)
        model.load_state_dict(state)
        model._fit_info = fit_info
        return model

    @classmethod
    def load(cls, model_name: str) -> 'BaseTorchModel':
        fn = f'{model_name}.pth'
        path = os.path.join(config.PATH_PARAMS_DIR, cls.__name__, fn)
        model = torch.load(path)
        return model

    def set_mode_train(self):
        super().set_mode_train()
        self.train()

    def set_mode_test(self):
        super().set_mode_test()
        self.eval()


"""
    Dummy model for testing purposes
"""


class DummyTorchModel(BaseTorchModel):

    def __init__(self):
        super().__init__()

        self._l1 = nn.Linear(24 * Dataset.SEASON_LENGTH, 32)
        self._l2 = nn.Linear(32, 1)

    def forward(self, xs: dict) -> tuple:
        ts = xs['temperature']
        bs = ts.size(0)

        ts = self._l1(ts.view(bs, -1))
        ts = F.relu(ts)

        ts = self._l2(ts)
        ts = F.sigmoid(ts)

        return ts.view(bs) * (Dataset.SEASON_LENGTH - 1), {}
