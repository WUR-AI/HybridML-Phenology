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
from runs.args_util.args_dataset import configure_argparser_dataset, get_configured_dataset
from runs.args_util.args_main import configure_argparser_main

if __name__ == '__main__':

    min_temperature = -40
    max_temperature = 60
    step_temperature = .5  # dont change! -- values are rounded to nearest halves

    # range_temperature = list(range(min_temperature, max_temperature, step_temperature))
    range_temperature = np.arange(min_temperature, max_temperature, step_temperature)
    range_ix_temperature = {v: i for i, v in enumerate(range_temperature)}

    parser = argparse.ArgumentParser()
    configure_argparser_main(parser)
    configure_argparser_dataset(parser)

    args = parser.parse_args()

    model_cls = NNChillModel
    model_name = model_cls.__name__ + '_japan_seed18'

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

        us = chill_operator_model(x).squeeze(0)

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

            # t_min = int(t.min().item() + .5)
            # t_max = int(t.max().item() + .5)

            t_min = round(t.min().item() * 2) / 2
            t_max = round(t.max().item() * 2) / 2


            # print(t_min, t_max)
            # print(u.item())

            entries[t_min, t_max].append(u.item())

    entries_mean = {
        k: sum(v) / len(v) if len(v) != 0 else 0 for k, v in entries.items()
    }

    entries_std = {
        k: np.std(v) if len(v) != 0 else 0 for k, v in entries.items()
    }

    entries_count = {
        k: len(v) for k, v in entries.items()
    }

    grid_mean = np.zeros((len(range_temperature), len(range_temperature)))
    grid_std = np.zeros((len(range_temperature), len(range_temperature)))
    grid_count = np.zeros((len(range_temperature), len(range_temperature)))

    for (t_min, t_max), v in entries_mean.items():
        grid_mean[range_ix_temperature[t_min], range_ix_temperature[t_max]] = v

    for (t_min, t_max), v in entries_std.items():
        grid_std[range_ix_temperature[t_min], range_ix_temperature[t_max]] = v

    for (t_min, t_max), v in entries_count.items():
        grid_count[range_ix_temperature[t_min], range_ix_temperature[t_max]] = v

    # grid[range_ix_temperature[-30], range_ix_temperature[30]] = 1

    # plt.xlim((min_temperature, max_temperature))
    # plt.ylim((min_temperature, max_temperature))
    # plt.ylim((max_temperature, min_temperature))

    # grid = grid_mean
    # grid = grid_std
    # grid = grid_count

    fig, axs = plt.subplots(1, 3)



    # print(grid.max())

    # left = min_temperature
    # right = max_temperature
    # bottom = max_temperature
    # top = min_temperature

    left = max_temperature
    right = min_temperature
    bottom = min_temperature
    top = max_temperature

    extent = [left, right, bottom, top]

    cax0 = axs[0].imshow(grid_mean, extent=extent)
    cax1 = axs[1].imshow(grid_std, extent=extent)
    cax2 = axs[2].imshow(grid_count, extent=extent)

    fontsize = 5
    axs[0].set_xlabel('t min', fontsize=fontsize)
    axs[0].set_ylabel('t max', fontsize=fontsize)
    axs[1].set_xlabel('t min', fontsize=fontsize)
    axs[1].set_ylabel('t max', fontsize=fontsize)
    axs[2].set_xlabel('t min', fontsize=fontsize)
    axs[2].set_ylabel('t max', fontsize=fontsize)

    axs[0].set_title('mean')
    axs[1].set_title('std')
    axs[2].set_title('count')

    # plt.colorbar(cax=cax0)

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.savefig('temp_response.png', bbox_inches='tight')




        