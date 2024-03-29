import argparse
import os.path
from collections import defaultdict

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

import torch
import torch.utils.data

import config
from datasets.dataset import Dataset
from datasets.dataset_torch import TorchDatasetWrapper
from models.base import BaseModel
from models.nn_chill_operator import NNChillModel
from phenology.chill.utah_model import utah_chill
from phenology.chill.chill_hours import f_chill_hours
from phenology.chill.chilldays import chill_days
from runs.args_util.args_dataset import configure_argparser_dataset, get_configured_dataset
from runs.args_util.args_main import configure_argparser_main
from runs.args_util.args_model import MODELS_KEYS_TO_CLS

if __name__ == '__main__':

    # TODO -- implement for pb models as well

    parser = argparse.ArgumentParser()
    configure_argparser_main(parser)
    configure_argparser_dataset(parser)

    parser.add_argument('--model_cls',
                        type=str,
                        choices=list(MODELS_KEYS_TO_CLS.keys()),
                        required=True,
                        help='Specify the model class that is to be trained/evaluated',
                        )
    parser.add_argument('--model_name',
                        type=str,
                        help='Optionally specify a name for the model. If none is provided the model class will be '
                             'used. The name is used for storing model weights and evaluation files',
                        )

    args = parser.parse_args()

    seed = args.seed

    # model_cls = NNChillModel
    # # model_name = model_cls.__name__ + f'_japan_seed{seed}'
    # # model_name = model_cls.__name__ + '_japan_seed5'
    # # model_name = model_cls.__name__ + f'_japan_seed{seed}_yedoensis'
    #
    # model_name = 'NNChillModel_japan_seed79_decay'

    model_cls = MODELS_KEYS_TO_CLS[args.model_cls]
    model_name = args.model_name or model_cls.__name__

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

    uss_hour = []  # Units
    uss_utah = []  # Units
    uss_days = []  # Units

    entries = defaultdict(list)

    for x in dataloader:
        x = TorchDatasetWrapper.normalize(x)

        us = chill_operator_model(x).squeeze(0)

        ts = x['temperature']

        ts = TorchDatasetWrapper.normalize_temperature(ts, revert=True).squeeze(0)

        us_hour = f_chill_hours(ts.detach().cpu().numpy()) / 24
        us_utah = utah_chill(ts.detach().cpu().numpy()) / 24
        us_days = -1 * chill_days(ts.detach().cpu().numpy(), t_C=7) / 24

        tss.append(ts.detach().cpu().numpy())
        uss.append(us.detach().cpu().numpy())

        uss_hour.append(us_hour)
        uss_utah.append(us_utah)
        uss_days.append(us_days)

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

    grid_hour = np.zeros(shape=(n_bins_resp, n_bins_temp))
    grid_utah = np.zeros(shape=(n_bins_resp, n_bins_temp))
    grid_days = np.zeros(shape=(n_bins_resp, n_bins_temp))

    t_to_ix = {v: i for i, v in zip(range(n_bins_temp), np.arange(tmin, tmax, 0.5))}
    u_to_ix = {v: i for i, v in enumerate(resps)}

    for ts, us, us_hour, us_utah, us_days in zip(tss, uss, uss_hour, uss_utah, uss_days):

        for t, u, u_hour, u_utah, u_days in zip(ts, us, us_hour, us_utah, us_days):

            t_mean = t.mean()

            # Round to nearest 0.5
            t_mean = round(t_mean * 2) / 2

            u_closest = min(resps, key=lambda v: abs(v-u))

            u_closest_hour = min(resps, key=lambda v: abs(v-u_hour))
            u_closest_utah = min(resps, key=lambda v: abs(v-u_utah))
            u_closest_days = min(resps, key=lambda v: abs(v-u_days))

            t_ix = t_to_ix[t_mean]
            u_ix = u_to_ix[u_closest]

            u_ix_hour = u_to_ix[u_closest_hour]
            u_ix_utah = u_to_ix[u_closest_utah]
            u_ix_days = u_to_ix[u_closest_days]

            grid[u_ix, t_ix] += 1

            grid_hour[u_ix_hour, t_ix] += 1
            grid_utah[u_ix_utah, t_ix] += 1
            grid_days[u_ix_days, t_ix] += 1

    for i in range(n_bins_temp):
        grid[:, i] /= max(grid[:, i]) + 1e-6

        grid_hour[:, i] /= max(grid_hour[:, i]) + 1e-6
        grid_utah[:, i] /= max(grid_utah[:, i]) + 1e-6
        grid_days[:, i] /= max(grid_days[:, i]) + 1e-6

    #     # grid[:, i] /= sum(grid[:, i]) + 1e-6
    #     # print(sum(grid[:, i]))

    fig, axs = plt.subplots(figsize=(6, 6))

    plt.imshow(grid,
               origin='lower',
               cmap=matplotlib.colormaps['Blues'],
               )

    axs.set_xticks(np.arange(0, (tmax-tmin), 10) * 2, np.arange(tmin, tmax, 10))
    axs.set_yticks([0, (n_bins_resp - 1) // 2, n_bins_resp - 1], [0, 0.5, 1])

    # plt.xlabel('Mean Temperature (°C)')
    # plt.ylabel('$u^{(c)}$')

    fn = 'response_over_mean_temperature.png'
    path = os.path.join(config.PATH_FIGURES_DIR,
                        model_cls.__name__,
                        model_name,
                        'plot_temperature_response',
                        )
    os.makedirs(path, exist_ok=True)

    plt.savefig(os.path.join(path, fn), bbox_inches='tight')

    plt.cla()
    plt.close()

    """
        pb models
    """

    fig, axs = plt.subplots(figsize=(6, 6))

    plt.imshow(grid_hour,
               origin='lower',
               cmap=matplotlib.colormaps['Blues'],
               )

    axs.set_xticks(np.arange(0, (tmax - tmin), 10) * 2, np.arange(tmin, tmax, 10))
    axs.set_yticks([0, (n_bins_resp - 1) // 2, n_bins_resp - 1], [0, 0.5, 1])

    # plt.xlabel('Mean Temperature (°C)')
    # plt.ylabel('$u^{(c)}$')

    fn = 'pb_hour_response_over_mean_temperature.png'
    path = os.path.join(config.PATH_FIGURES_DIR,
                        model_cls.__name__,
                        model_name,
                        'plot_temperature_response',
                        )
    os.makedirs(path, exist_ok=True)

    plt.savefig(os.path.join(path, fn), bbox_inches='tight')

    plt.cla()
    plt.close()




    fig, axs = plt.subplots(figsize=(6, 6))

    plt.imshow(grid_utah,
               origin='lower',
               cmap=matplotlib.colormaps['Blues'],
               )

    axs.set_xticks(np.arange(0, (tmax - tmin), 10) * 2, np.arange(tmin, tmax, 10))
    axs.set_yticks([0, (n_bins_resp - 1) // 2, n_bins_resp - 1], [0, 0.5, 1])

    # plt.xlabel('Mean Temperature (°C)')
    # plt.ylabel('$u^{(c)}$')

    fn = 'pb_utah_response_over_mean_temperature.png'
    path = os.path.join(config.PATH_FIGURES_DIR,
                        model_cls.__name__,
                        model_name,
                        'plot_temperature_response',
                        )
    os.makedirs(path, exist_ok=True)

    plt.savefig(os.path.join(path, fn), bbox_inches='tight')

    plt.cla()
    plt.close()




    fig, axs = plt.subplots(figsize=(6, 6))

    plt.imshow(grid_days,
               origin='lower',
               cmap=matplotlib.colormaps['Blues'],
               )

    axs.set_xticks(np.arange(0, (tmax - tmin), 10) * 2, np.arange(tmin, tmax, 10))
    axs.set_yticks([0, (n_bins_resp - 1) // 2, n_bins_resp - 1], [0, 0.5, 1])

    # plt.xlabel('Mean Temperature (°C)')
    # plt.ylabel('$u^{(c)}$')

    fn = 'pb_days_response_over_mean_temperature.png'
    path = os.path.join(config.PATH_FIGURES_DIR,
                        model_cls.__name__,
                        model_name,
                        'plot_temperature_response',
                        )
    os.makedirs(path, exist_ok=True)

    plt.savefig(os.path.join(path, fn), bbox_inches='tight')

    plt.cla()
    plt.close()
