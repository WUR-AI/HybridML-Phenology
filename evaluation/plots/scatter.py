import os
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm

from datasets.dataset import split_location_token
from evaluation.util import get_countries_occurring_in


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
    mae = mean_absolute_error(doys_true, doys_pred)

    ax.plot(np.arange(0, 200), np.arange(0, 200), '--', color='grey')

    ax.scatter(doys_pred,
               doys_true,
               c=colors,
               label=f'r2 {r2:.2f}, rmse {rmse:.2f}, mae {mae:.2f}, n={len(doys_true)})',
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


def savefig_scatter_doys_global(doys_true: list,
                                doys_pred: list,
                                locations: list,
                                path: str,
                                ):
    assert len(doys_true) == len(doys_pred)
    assert len(doys_true) == len(locations)
    os.makedirs(path, exist_ok=True)

    # Group predictions based on country
    doys_true_nat = defaultdict(list)
    doys_pred_nat = defaultdict(list)
    locations_nat = defaultdict(list)
    for location, doy_true, doy_pred in zip(locations, doys_true, doys_pred):
        country, site = split_location_token(location)

        doys_true_nat[country].append(doy_true)
        doys_pred_nat[country].append(doy_pred)
        locations_nat[country].append(location)

    """
        Making global scatter plot grouped by country
    """

    fig, ax = plt.subplots()
    path_global = os.path.join(path, 'plots_scatter_global')
    os.makedirs(path_global, exist_ok=True)
    fn_global = 'scatterplot_doys_global.png'

    r2 = r2_score(doys_true, doys_pred)
    rmse = mean_squared_error(doys_true, doys_pred, squared=False)
    mae = mean_absolute_error(doys_true, doys_pred)

    ax.plot(np.arange(0, 200), np.arange(0, 200), '--', color='grey')

    for country in tqdm(doys_true_nat.keys(), desc='Making global scatter plot'):
        r2_nat = r2_score(doys_true_nat[country], doys_pred_nat[country])
        rmse_nat = mean_squared_error(doys_true_nat[country], doys_pred_nat[country], squared=False)
        mae_nat = mean_absolute_error(doys_true_nat[country], doys_pred_nat[country])

        ax.scatter(doys_pred_nat[country],
                   doys_true_nat[country],
                   label=f'{country} (r2 {r2_nat:.2f}, rmse {rmse_nat:.2f}, mae {mae_nat:.2f}, n={len(doys_true_nat[country])})',
                   s=3,
                   alpha=0.3,
                   )

    ax.set_xlabel('DOY pred')
    ax.set_ylabel('DOY true')

    plt.legend()

    plt.title(f'Global DOY predictions (r2 {r2:.2f}, rmse {rmse:.2f}, mae {mae:.2f}, n={len(doys_true)})')

    plt.savefig(os.path.join(path_global, fn_global))

    plt.cla()
    plt.close()

    """
        Making national scatter plots
    """

    for country in tqdm(doys_true_nat.keys(), desc='Making national scatter plots'):
        fn_nat = f'scatterplot_doys_{country.lower()}.png'

        r2_nat = r2_score(doys_true_nat[country], doys_pred_nat[country])
        rmse_nat = mean_squared_error(doys_true_nat[country], doys_pred_nat[country], squared=False)
        mae_nat = mean_absolute_error(doys_true_nat[country], doys_pred_nat[country])

        savefig_scatter_doys(
            doys_true_nat[country],
            doys_pred_nat[country],
            path=path_global,
            fn=fn_nat,
            title=f'{country} DOY predictions (r2 {r2_nat:.2f}, rmse {rmse_nat:.2f}, mae {mae_nat:.2f}, n={len(doys_true_nat[country])})',
        )


def savefig_scatter_doys_local(doys_true: list,
                               doys_pred: list,
                               locations: list,
                               path: str,
                               ):
    assert len(doys_true) == len(doys_pred)
    assert len(doys_true) == len(locations)
    os.makedirs(path, exist_ok=True)

    """
    Group doys based on location
    """

    doys_true_loc = defaultdict(list)
    doys_pred_loc = defaultdict(list)
    for location, doy_true, doy_pred in zip(locations, doys_true, doys_pred):
        doys_true_loc[location].append(doy_true)
        doys_pred_loc[location].append(doy_pred)

    """

    Make a scatter plot for each location

    """
    path_local = os.path.join(path, 'plots_scatter_local')

    countries = get_countries_occurring_in(locations)
    for country in countries:
        os.makedirs(os.path.join(path_local, country.lower()), exist_ok=True)

    for location in tqdm(doys_true_loc.keys(), 'Making local scatter plots'):
        country, site = split_location_token(location)

        fn_loc = f'scatterplot_doys_{site}.png'

        r2_loc = r2_score(doys_true_loc[location], doys_pred_loc[location])
        rmse_loc = mean_squared_error(doys_true_loc[location], doys_pred_loc[location], squared=False)
        mae_loc = mean_absolute_error(doys_true_loc[location], doys_pred_loc[location],)

        savefig_scatter_doys(
            doys_true_loc[location],
            doys_pred_loc[location],
            path=os.path.join(path_local, country.lower()),
            fn=fn_loc,
            title=f'{location} DOY predictions (r2 {r2_loc:.2f}, rmse {rmse_loc:.2f}, mae {mae_loc:.2f}, n={len(doys_true_loc[location])})',
        )
