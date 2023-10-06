import numpy as np
from matplotlib import pyplot as plt

import torch

import config
from datasets.dataset_torch import TorchDatasetWrapper
from models.nn_chill_operator import NNChillModel

if __name__ == '__main__':

    # model_name = NNChillModel.__name__
    # model_name = NNChillModel.__name__ + '_DNN_size64'
    model_name = NNChillModel.__name__ + '_GlobalYedoenis'

    # _data = torch.arange(-10, 20, 0.1).view(-1, 1).expand(-1, 24)
    data = torch.arange(-10, 30, 0.1).view(-1, 1).expand(-1, 24)
    data = data.unsqueeze(0).to(config.TORCH_DTYPE)

    data = TorchDatasetWrapper.normalize_temperature(data)

    model = NNChillModel.load(model_name)

    xs = {
        'temperature': data,
    }

    cus, gus = model.f_units_chill_growth(xs, torch.tensor(7.0))

    cus = cus.squeeze(0) / 24

    from phenology.chill.utah_model import utah_chill

    # fig, ax = plt.subplots(3, figsize=(8, 15))
    fig, ax = plt.subplots()

    x = np.arange(-10, 30, 0.1)

    ax.plot(x, utah_chill(np.reshape(x, newshape=(-1, 1))), label='utah chill', c='black')

    ax.plot(x, cus.cpu().detach().numpy(), label='nn chill', c='r')

    plt.savefig('temp.png')


