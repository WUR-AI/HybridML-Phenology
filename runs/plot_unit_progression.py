import argparse

import numpy as np
from matplotlib import pyplot as plt

import torch

import config
from datasets.dataset import Dataset
from datasets.dataset_torch import TorchDatasetWrapper
from models.base import BaseModel
from models.base_torch_accumulation import BaseTorchAccumulationModel
from models.nn_chill_operator import NNChillModel
from runs.args_util.args_dataset import configure_argparser_dataset, get_configured_dataset
from runs.args_util.args_main import configure_argparser_main
from runs.args_util.args_model import MODELS_KEYS_TO_CLS

if __name__ == '__main__':

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

    model_cls = NNChillModel

    model_name = args.model_name or model_cls.__name__

    assert issubclass(model_cls, BaseTorchAccumulationModel)

    dataset, _ = get_configured_dataset(args)
    assert isinstance(dataset, Dataset)

    model = model_cls.load(model_name)

    model._debug_mode = True
    model.set_mode_test()

    # location = dataset.locations_test[0]
    location = 'Switzerland/Cevio-Cavergno'
    year = dataset.years_test[0]

    data_ix = (year, location)

    sample = dataset[data_ix]

    ix_pred, _, info = model.predict_ix(sample)

    ix_true = sample['bloom_ix']

    debug_info = info['forward_pass']['debug']

    temperatures = np.reshape(sample['temperature'], newshape=-1)

    units_chill = np.reshape(debug_info['units_c'], newshape=-1)
    units_growth = np.reshape(debug_info['units_g_masked'], newshape=-1)

    req_c = np.reshape(debug_info['req_c'], newshape=-1)
    req_g = np.reshape(debug_info['req_g'], newshape=-1)

    th_c = debug_info['th_c'][0]
    th_g = debug_info['th_g'][0]
    tb_g = debug_info['tb_g'][0]

    """
        Create plot
    """

    fig, axs = plt.subplots(6)
    fig.set_size_inches(18.5, 18.5)

    fig.suptitle(f'loc: {location}, year: {year}')

    axs[0].plot(temperatures, label='temp')

    axs[1].plot(units_chill, label='units_chill')
    axs[2].plot(units_growth, label=f'units_growth')

    axs[3].plot(np.cumsum(units_chill, axis=-1), label='units sum chill')
    axs[3].plot([th_c for _ in range(len(units_chill))], '--')

    axs[4].plot(np.cumsum(units_growth, axis=-1), label='units sum growth')
    axs[4].plot([th_g for _ in range(len(units_growth))], '--')

    axs[5].plot(req_c, label='req chill')
    axs[5].plot(req_g, label='req growth')

    axs[5].axvline(ix_true, c='red')
    axs[5].axvline(ix_pred, c='blue')

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[3].legend()
    axs[4].legend()
    axs[5].legend()

    plt.savefig('temp_unit_progression_bce.png')
    plt.cla()
    plt.close()
