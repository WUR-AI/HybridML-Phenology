
import os
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm

from data.bloom_doy import get_locations_coordinates
from datasets.dataset import split_location_token


def savefig_mae_over_coords(doys_true: list,
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
        Making national plots
    """

    loc_coords = get_locations_coordinates().to_dict(orient='index')

    for country in tqdm(doys_true_nat.keys(), desc='Making national scatter plots'):

        # Group based on location within the country
        doys_true_loc = defaultdict(list)
        doys_pred_loc = defaultdict(list)
        for location, doy_true, doy_pred in zip(locations_nat[country], doys_true_nat[country], doys_pred_nat[country]):
            doys_true_loc[location].append(doy_true)
            doys_pred_loc[location].append(doy_pred)

        loc_maes = []
        loc_lats = []
        loc_lons = []
        loc_alts = []
        for loc in doys_true_loc.keys():

            r2_loc = r2_score(doys_true_loc[loc], doys_pred_loc[loc])
            rmse_loc = mean_squared_error(doys_true_loc[loc], doys_pred_loc[loc], squared=False)
            mae_loc = mean_absolute_error(doys_true_loc[loc], doys_pred_loc[loc])

            lat_loc = loc_coords[loc]['lat']
            lon_loc = loc_coords[loc]['long']
            alt_loc = loc_coords[loc]['alt']

            loc_maes.append(mae_loc)
            loc_lats.append(lat_loc)
            loc_lons.append(lon_loc)
            loc_alts.append(alt_loc)

        """
            Plot MAE over ALT
        """

        fig, ax = plt.subplots()

        ax.scatter(loc_alts,
                   loc_maes,
                   s=3,
                   alpha=0.3,
                   )

        ax.set_xlabel('alt')
        ax.set_ylabel('mae')

        plt.title(f'Local MAE over alt in {country}')

        plt.savefig(os.path.join(path, f'plot_{country.lower()}_mae_over_alt.png'))

        plt.cla()
        plt.close()

        """
            Plot MAE over LAT
        """

        fig, ax = plt.subplots()

        ax.scatter(loc_lats,
                   loc_maes,
                   s=3,
                   alpha=0.3,
                   )

        ax.set_xlabel('lat')
        ax.set_ylabel('mae')

        plt.title(f'Local MAE over lat in {country}')

        plt.savefig(os.path.join(path, f'plot_{country.lower()}_mae_over_lat.png'))

        plt.cla()
        plt.close()

        """
            Plot MAE over LON
        """

        fig, ax = plt.subplots()

        ax.scatter(loc_lons,
                   loc_maes,
                   s=3,
                   alpha=0.3,
                   )

        ax.set_xlabel('lon')
        ax.set_ylabel('mae')

        plt.title(f'Local MAE over lon in {country}')

        plt.savefig(os.path.join(path, f'plot_{country.lower()}_mae_over_lon.png'))

        plt.cla()
        plt.close()
