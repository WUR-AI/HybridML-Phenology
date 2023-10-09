import argparse

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

    # min_temperature = -30
    # max_temperature = 60
    # step_temperature = 1
    #
    # bins_temperature = list(range(min_temperature, max_temperature, step_temperature))

    parser = argparse.ArgumentParser()
    configure_argparser_main(parser)
    configure_argparser_dataset(parser)

    args = parser.parse_args()

    model_cls = NNChillModel
    # model_cls = LocalUtahChillModel
    model_name = None

    model_name = model_name or model_cls.__name__

    assert issubclass(model_cls, BaseModel)

    # Configure the dataset based on the provided arguments
    dataset, _ = get_configured_dataset(args)
    assert isinstance(dataset, Dataset)

    model = model_cls.load(model_name)

    chill_operator_model = model._chill_model

    dataset_wrapped = TorchDatasetWrapper(dataset)

    use_test_data = True
    if use_test_data:
        data = dataset_wrapped.get_test_data()
    else:
        data = dataset_wrapped.get_train_data()

    dataloader = torch.utils.data.DataLoader(data,
                                             batch_size=1,
                                             collate_fn=dataset_wrapped.collate_fn,
                                             shuffle=True,
                                             )

    ts = []  # Temperatures
    us = []  # Units

    for x in dataloader:
        x = TorchDatasetWrapper.normalize(x)

        u = chill_operator_model(x).squeeze(0)

        t = x['temperature']

        t = TorchDatasetWrapper.normalize_temperature(t, revert=True).squeeze(0)

        ts.append(t)
        us.append(u)


        # TODO -- continue
        