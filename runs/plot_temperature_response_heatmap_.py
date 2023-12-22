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
from phenology.chill.utah_model import utah_chill
from runs.args_util.args_dataset import configure_argparser_dataset, get_configured_dataset
from runs.args_util.args_main import configure_argparser_main


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    configure_argparser_main(parser)
    configure_argparser_dataset(parser)

    args = parser.parse_args()

    model_cls = NNChillModel
    model_name = model_cls.__name__ + '_japan_seed18'
    # model_name = model_cls.__name__ + '_japan_seed5'
    # model_name = model_cls.__name__ + '_japan_seed18_yedoensis'

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

    """
        Evaluate the model
    """

    tss = []  # Temperatures
    uss = []  # Units

    entries = defaultdict(list)

    for x in dataloader:
        x = TorchDatasetWrapper.normalize(x)

        us = chill_operator_model(x).squeeze(0)

        ts = x['temperature']

        ts = TorchDatasetWrapper.normalize_temperature(ts, revert=True).squeeze(0)

        tss.append(ts.detach().cpu().numpy())
        uss.append(us.detach().cpu().numpy())

    """
        Bin the results
    """

    # results_mlp = defaultdict(list)
    # results_utah = defaultdict(list)
    #
    # values_x = []
    # values_y = []
    # values_z = []
    #
    # for ts, us in zip(tss, uss):
    #
    #     for t, u in zip(ts, us):
    #
    #         t_mean = t.mean()
    #
    #         t_mean = round(t_mean)
    #
    #         results_mlp[t_mean].append(u)
    #         results_utah[t_mean].append(max(utah_chill(t), 0) / 24)
    #
    #         values_x.append(t.mean())
    #         values_y.append(u)
    #         values_z.append(max(utah_chill(t), 0) / 24)


    tmin = -30
    tmax = 40

    n_bins_temp = (tmax - tmin) * 2
    n_bins_resp = 80

    resps = np.arange(0, 1, 1 / n_bins_resp)

    grid = np.zeros(shape=(n_bins_resp, n_bins_temp))

    t_to_ix = {v: i for i, v in zip(range(n_bins_temp), np.arange(tmin, tmax, 0.5))}
    u_to_ix = {v: i for i, v in enumerate(resps)}

    for ts, us in zip(tss, uss):

        for t, u in zip(ts, us):

            t_mean = t.mean()

            # Round to nearest 0.5
            t_mean = round(t_mean * 2) / 2

            u_closest = min(resps, key=lambda v: abs(v-u))

            t_ix = t_to_ix[t_mean]
            u_ix = u_to_ix[u_closest]

            grid[u_ix, t_ix] += 1

    for i in range(n_bins_temp):
        grid[:, i] /= max(grid[:, i]) + 1e-6
    #     # grid[:, i] /= sum(grid[:, i]) + 1e-6
    #     # print(sum(grid[:, i]))

    fig, axs = plt.subplots(figsize=(6, 6))

    plt.imshow(grid, origin='lower')

    plt.savefig('temp.png', bbox_inches='tight')
