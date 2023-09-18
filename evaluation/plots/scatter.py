import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error


def savefig_scatter_doys(doys_true: list,
                         doys_pred: list,
                         path: str,
                         fn: str,
                         title: str = '',
                         colors: list = None,
                         ):
    assert len(doys_true) == len(doys_pred)
    colors = colors or ['black'] * len(doys_true)
    os.makedirs(path, exist_ok=True)

    fig, ax = plt.subplots()

    r2 = r2_score(doys_true, doys_pred)
    rmse = mean_squared_error(doys_true, doys_pred, squared=False)

    ax.plot(np.arange(0, 200), np.arange(0, 200), '--', color='grey')

    ax.scatter(doys_pred,
               doys_true,
               c=colors,
               label=f'r2 {r2:.2f}, rmse {rmse:.2f}, n={len(doys_true)})',
               s=3,
               alpha=0.3,
               )

    ax.set_xlabel('DOY pred')
    ax.set_ylabel('DOY true')

    plt.legend()

    plt.title(title)

    plt.savefig(os.path.join(path, fn))

    plt.cla()
    plt.close()
