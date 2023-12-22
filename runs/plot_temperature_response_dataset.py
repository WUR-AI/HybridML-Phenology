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

    results_mlp = defaultdict(list)
    results_utah = defaultdict(list)

    values_x = []
    values_y = []
    values_z = []

    for ts, us in zip(tss, uss):

        for t, u in zip(ts, us):

            t_mean = t.mean()

            # Round to nearest 0.5
            # t_mean = round(t_mean * 2) / 2

            t_mean = round(t_mean)

            results_mlp[t_mean].append(u)
            results_utah[t_mean].append(max(utah_chill(t), 0) / 24)

            values_x.append(t.mean())
            values_y.append(u)
            values_z.append(max(utah_chill(t), 0) / 24)

    """
        Plot the result
    """

    # x = []
    # y = []
    # for ts_mean, uss in results.items():
    #     x.append(ts_mean)
    #     y.append(np.array(uss))

    tmin = -20
    tmax = 40

    # data = [np.array(results[key]) for key in np.arange(tmin, tmax, 0.5)]
    data = [np.array(results_mlp[key]) for key in np.arange(tmin, tmax, 1)]
    data_utah = [np.array(results_utah[key]) for key in np.arange(tmin, tmax, 1)]
    # data = list(results.values())
    nans = [float('nan'), float('nan')]  # requires at least 2 nans

    fig, axs = plt.subplots(figsize=(6, 6))

    # gs = fig.add_gridspec(2, 2,
    #                       width_ratios=(4, 1), height_ratios=(1, 4),
    #                       left=0.1, right=0.9, bottom=0.1, top=0.9,
    #                       wspace=0.05, hspace=0.05,
    #                       )

    # plt.violinplot([d if len(d) > 0 else nans for d in data], positions=np.arange(tmin, tmax, 0.5))


    plt.violinplot([d if len(d) > 0 else nans for d in data], positions=np.arange(tmin, tmax, 1), showextrema=False)
    # plt.violinplot([d if len(d) > 0 else nans for d in data_utah], positions=np.arange(tmin, tmax, 1), showextrema=False)

    # plt.scatter(values_x, values_y, alpha=0.005, s=2, c='darkblue')

    # plt.scatter(values_x, values_z, alpha=0.005, s=2, c='darkblue')

    # ax_histx = fig.add_subplot(gs[0, 0], sharex=axs)

    # ax_histx.tick_params(axis="x", labelbottom=False)
    # ax_histx.tick_params(axis="y", labelleft=False)

    # ax_histx.bar(np.arange(tmin, tmax, 1), [len(d) for d in data])

    plt.savefig('temp_response_violinplot.png', bbox_inches='tight')
    # plt.savefig('temp_response_scatter.png', bbox_inches='tight')


