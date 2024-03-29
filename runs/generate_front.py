import argparse
import os.path
from collections import defaultdict

from matplotlib import pyplot as plt
from tqdm import tqdm

from data.bloom_doy import get_locations_coordinates
from datasets.dataset import Dataset
from evaluation.plots.maps import _get_countries_in_location_list, _filter_country_data, _get_country_basemap
from models.base import BaseModel
from models.nn_chill_operator import NNChillModel
from models.process_based.utah_chill import LocalUtahChillModel
from runs.args_util.args_dataset import get_configured_dataset, configure_argparser_dataset
from runs.args_util.args_main import configure_argparser_main


# def _savefig_locations_front(locations_ixs_true: dict,
#                              locations_ixs_pred: dict,
#                              ):
#     assert len(locations_ixs_true) == len(locations_ixs_pred)
#
#     path_folder = os.path.join('temp')  # TODO
#     os.makedirs(path_folder, exist_ok=True)
#
#     # Get all location coordinates
#     location_data = get_locations_coordinates()
#
#     locations = list(locations_ixs_true.keys())
#     lats = location_data.loc[locations]['lat'].values
#     lons = location_data.loc[locations]['long'].values
#
#     locations_lats = {loc: lat for loc, lat in zip(locations, lats)}
#     locations_lons = {loc: lon for loc, lon in zip(locations, lons)}
#
#     ixs_locations_true = defaultdict(list)
#     for loc, ix in locations_ixs_true.items():
#         ixs_locations_true[ix].append(loc)
#
#     ixs_locations_pred = defaultdict(list)
#     for loc, ix in locations_ixs_pred.items():
#         ixs_locations_pred[ix].append(loc)
#
#     # Get all countries that are included in the location tokens
#     # These are needed to know for which country we should make maps
#     countries = list(_get_countries_in_location_list(locations))
#     n_countries = len(countries)
#
#     # print(countries)
#
#     for doy in tqdm(range(1, Dataset.SEASON_END_DOY + 1)):
#
#         t = Dataset.doy_to_index(doy)
#
#         # if t < Dataset.doy_to_index(1):
#         #     continue
#
#         fig = plt.figure()
#
#         i = 1
#         for i_country, country in enumerate(countries):
#
#             # locs_in_country = _filter_country_data(country, locations)
#             # locs_in_country_ixs_true = {loc: locations_ixs_true[loc] for loc in locs_in_country}
#             # locs_in_country_ixs_pred = {loc: locations_ixs_true[loc] for loc in locs_in_country}
#
#             blooming_locs_true = _filter_country_data(country, ixs_locations_true[t])
#             blooming_locs_pred = _filter_country_data(country, ixs_locations_pred[t])
#
#             # print(blooming_locs_true)
#
#             ax = fig.add_subplot(n_countries, 2, i)
#             ax.set_title('True')
#             i += 1
#
#             m = _get_country_basemap(country)
#
#             m.scatter(
#                 x=location_data.loc[blooming_locs_true]['long'].values,
#                 y=location_data.loc[blooming_locs_true]['lat'].values,
#                 latlon=True,
#                 c='red',
#                 s=2,
#             )
#
#             ax = fig.add_subplot(n_countries, 2, i)
#             ax.set_title('Pred')
#             i += 1
#
#             m = _get_country_basemap(country)
#
#             m.scatter(
#                 x=location_data.loc[blooming_locs_pred]['long'].values,
#                 y=location_data.loc[blooming_locs_pred]['lat'].values,
#                 latlon=True,
#                 c='red',
#                 s=2,
#             )
#
#         fn = f'plot_t{doy - 1:03d}.png'
#
#         # Save the figure
#         plt.savefig(os.path.join(path_folder, fn),
#                     dpi=100,
#                     )
#         # Clear results for constructing the next figure
#         plt.cla()
#         plt.close()


def doy_to_color(doy: int, bloom_doy: int) -> tuple:

    # doy_max = Dataset.SEASON_END_DOY

    c = 0.95

    a = 0. if doy < bloom_doy else c ** (doy - bloom_doy)

    rgba = (1., 0., 0., a)

    return rgba


def _savefig_locations_front(locations_ixs_true: dict,
                             locations_ixs_pred: dict,
                             year: int,
                             ):
    assert len(locations_ixs_true) == len(locations_ixs_pred)

    path_folder = os.path.join('temp')  # TODO
    os.makedirs(path_folder, exist_ok=True)

    # Get all location coordinates
    location_data = get_locations_coordinates()

    locations = list(locations_ixs_true.keys())
    lats = location_data.loc[locations]['lat'].values
    lons = location_data.loc[locations]['long'].values

    # locations_lats = {loc: lat for loc, lat in zip(locations, lats)}
    # locations_lons = {loc: lon for loc, lon in zip(locations, lons)}

    # ixs_locations_true = defaultdict(list)
    # for loc, ix in locations_ixs_true.items():
    #     ixs_locations_true[ix].append(loc)
    #
    # ixs_locations_pred = defaultdict(list)
    # for loc, ix in locations_ixs_pred.items():
    #     ixs_locations_pred[ix].append(loc)

    # Get all countries that are included in the location tokens
    # These are needed to know for which country we should make maps
    countries = list(_get_countries_in_location_list(locations))
    n_countries = len(countries)

    # print(countries)

    for doy in tqdm(range(1, Dataset.SEASON_END_DOY + 1)):

        t = Dataset.doy_to_index(doy)

        # if t < Dataset.doy_to_index(1):
        #     continue

        fig = plt.figure()

        i = 1
        for i_country, country in enumerate(countries):

            # locs_in_country = _filter_country_data(country, locations)
            # locs_in_country_ixs_true = {loc: locations_ixs_true[loc] for loc in locs_in_country}
            # locs_in_country_ixs_pred = {loc: locations_ixs_true[loc] for loc in locs_in_country}

            country_locs, = _filter_country_data(country, locations)

            # print(country_locs)

            ax = fig.add_subplot(n_countries, 2, i)
            ax.set_title('True')
            i += 1

            m = _get_country_basemap(country)

            # cs = [doy_to_color(doy, locations_ixs_true[loc]) for loc in country_locs]

            m.scatter(
                x=location_data.loc[country_locs]['long'].values,
                y=location_data.loc[country_locs]['lat'].values,
                latlon=True,
                c=[doy_to_color(doy, Dataset.index_to_doy(locations_ixs_true[loc])) for loc in country_locs],
                s=2,
            )

            ax.set_xlabel(f'DOY: {doy}')

            ax = fig.add_subplot(n_countries, 2, i)
            ax.set_title('Pred')
            i += 1

            m = _get_country_basemap(country)

            m.scatter(
                x=location_data.loc[country_locs]['long'].values,
                y=location_data.loc[country_locs]['lat'].values,
                latlon=True,
                c=[doy_to_color(doy, Dataset.index_to_doy(locations_ixs_pred[loc])) for loc in country_locs],
                s=2,
            )

            ax.set_xlabel(f'Date: {Dataset.doy_to_date_in_year(year, doy)}')

        fn = f'plot_t{doy - 1:03d}.png'

        # plt.xlabel(doy)  # TODO -- date
        fig.tight_layout()

        # Save the figure
        plt.savefig(os.path.join(path_folder, fn),
                    dpi=100,
                    )
        # Clear results for constructing the next figure
        plt.cla()
        plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    configure_argparser_main(parser)
    configure_argparser_dataset(parser)

    args = parser.parse_args()

    model_cls = NNChillModel
    # model_cls = LocalUtahChillModel
    # model_name = None
    model_name = 'NNChillModel_japan_seed18'

    model_name = model_name or model_cls.__name__

    assert issubclass(model_cls, BaseModel)

    # Configure the dataset based on the provided arguments
    dataset, _ = get_configured_dataset(args)
    assert isinstance(dataset, Dataset)

    model = model_cls.load(model_name)

    # year = dataset._years_test[1]
    year = 2000

    local_ixs_true = dict()
    local_ixs_pred = dict()

    # for x in dataset.get_test_data_in_years([year]):
    for x in dataset.get_train_data_in_years([year]):

        ix_true = x['bloom_ix']
        ix_pred, _, _ = model.predict_ix(x)

        location = x['location']

        local_ixs_true[location] = ix_true
        local_ixs_pred[location] = ix_pred

    print(local_ixs_true)
    print(local_ixs_pred)

    _savefig_locations_front(local_ixs_true,
                             local_ixs_pred,
                             year,
                             )

    # ffmpeg -i plot_t%03d.png video.mp4
