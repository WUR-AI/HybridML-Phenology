import argparse
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.utils.data

import config
from datasets.dataset import Dataset
from datasets.dataset_torch import TorchDatasetWrapper
from models.base import BaseModel
from models.nn_chill_operator import NNChillModel
from runs.fit_eval_util import get_configured_dataset, configure_argparser_main, configure_argparser_dataset

if __name__ == '__main__':

    min_temperature = -40
    max_temperature = 60
    step_temperature = 1

    range_temperature = list(range(min_temperature, max_temperature, step_temperature))
    range_ix_temperature = {v: i for i, v in enumerate(range_temperature)}

    parser = argparse.ArgumentParser()
    configure_argparser_main(parser)
    configure_argparser_dataset(parser)

    args = parser.parse_args()

    model_cls = NNChillModel
    model_name = model_cls.__name__ + '_GlobalJapanYedoenis'

    model_name = model_name or model_cls.__name__

    assert issubclass(model_cls, BaseModel)

    # Configure the dataset based on the provided arguments
    dataset, _ = get_configured_dataset(args)
    assert isinstance(dataset, Dataset)

    model = model_cls.load(model_name)

    chill_operator_model = model._chill_model

    dataset_wrapped = TorchDatasetWrapper(dataset)

    use_test_data = True
    # use_test_data = False
    if use_test_data:
        data = dataset_wrapped.get_test_data()
    else:
        data = dataset_wrapped.get_train_data()

    dataloader = torch.utils.data.DataLoader(data,
                                             batch_size=1,
                                             collate_fn=dataset_wrapped.collate_fn,
                                             shuffle=True,
                                             )

    tss = []  # Temperatures
    uss = []  # Units

    entries = defaultdict(list)

    for x in dataloader:
        x = TorchDatasetWrapper.normalize(x)

        us = chill_operator_model(x).squeeze(0) / 24

        ts = x['temperature']

        ts = TorchDatasetWrapper.normalize_temperature(ts, revert=True).squeeze(0)

        tss.append(ts)
        uss.append(us)

    for ts, us in zip(tss, uss):

        # print(ts.shape)
        # print(us.shape)

        for t, u in zip(ts, us):

            # print(t.shape)
            # print(t)

            t_min = int(t.min().item() + .5)
            t_max = int(t.max().item() + .5)

            # print(t_min, t_max)
            # print(u.item())

            entries[t_min, t_max].append(u.item())

    entries_mean = {
        k: sum(v) / len(v) if len(v) != 0 else 0 for k, v in entries.items()
    }

    entries_std = {
        k: np.std(v) if len(v) != 0 else 0 for k, v in entries.items()
    }

    grid = np.zeros((len(range_temperature), len(range_temperature)))
    # for (t_min, t_max), v in entries_mean.items():
    for (t_min, t_max), v in entries_std.items():
        grid[range_ix_temperature[t_min], range_ix_temperature[t_max]] = v

    # grid[range_ix_temperature[-30], range_ix_temperature[30]] = 1

    # plt.xlim((min_temperature, max_temperature))
    # plt.ylim((min_temperature, max_temperature))
    # plt.ylim((max_temperature, min_temperature))

    plt.xlabel('t min')
    plt.ylabel('t max')

    print(grid.max())

    # left = min_temperature
    # right = max_temperature
    # bottom = max_temperature
    # top = min_temperature

    left = max_temperature
    right = min_temperature
    bottom = min_temperature
    top = max_temperature

    extent = [left, right, bottom, top]

    plt.imshow(grid, extent=extent)
    plt.savefig('temp.png')



        