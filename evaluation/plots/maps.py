import os

from matplotlib import pyplot as plt
import matplotlib.colors
from mpl_toolkits.basemap import Basemap

import config
from data.bloom_doy import get_locations_coordinates

from datasets.dataset import split_location_token


_MAP_DPI = 500
_ANN_FONT_SIZE = 3


def _get_basemap(min_lat: float,
                 max_lat: float,
                 min_lon: float,
                 max_lon: float,
                 ):
    """
    Get a mpl_toolkits.basemap.BaseMap with the specified parameters
    :param min_lat: Min. latitude
    :param max_lat: Max. latitude
    :param min_lon: Min. longitude
    :param max_lon: Max. longitude
    :return:
    """

    m = Basemap(projection='merc',
                llcrnrlat=min_lat,
                urcrnrlat=max_lat,
                llcrnrlon=min_lon,
                urcrnrlon=max_lon,
                resolution='i',
                )

    m.drawcoastlines(color='grey',
                     zorder=0,
                     linewidth=0.7,
                     )

    m.drawcountries(color='grey',
                    zorder=0,
                    linewidth=0.7,
                    )

    return m


def _get_basemap_japan():

    lat_min = 24.3366666666667
    lat_max = 45.415
    lon_min = 123.010555555556
    lon_max = 145.585555555556

    lat_border = 2
    lon_border = 4

    return _get_basemap(
        lat_min - lat_border,
        lat_max + lat_border,
        lon_min - lon_border,
        lon_max + lon_border,
    )


def _get_basemap_south_korea():

    lat_min = 33.24613
    lat_max = 38.25085
    lon_min = 126.38121
    lon_max = 130.89864

    lat_border = 1
    lon_border = 2

    return _get_basemap(
        lat_min - lat_border,
        lat_max + lat_border,
        lon_min - lon_border,
        lon_max + lon_border,
    )


def _get_basemap_switzerland():

    lat_min = 45.857811
    lat_max = 47.763636
    lon_min = 6.014089
    lon_max = 10.464608

    lat_border = 0.5
    lon_border = 1

    return _get_basemap(
        lat_min - lat_border,
        lat_max + lat_border,
        lon_min - lon_border,
        lon_max + lon_border,
    )


def _get_basemap_usa():

    lat_min = 38.8853496
    lat_max = 38.8853496
    lon_min = -77.0386278
    lon_max = -77.0386278

    lat_border = 2
    lon_border = 4

    return _get_basemap(
        lat_min - lat_border,
        lat_max + lat_border,
        lon_min - lon_border,
        lon_max + lon_border,
    )


def _get_country_basemap(country: str):
    if country == 'Japan':
        return _get_basemap_japan()
    if country == 'South Korea':
        return _get_basemap_south_korea()
    if country == 'Switzerland':
        return _get_basemap_switzerland()
    if country == 'USA':
        return _get_basemap_usa()
    raise Exception(f'No basemap for country: {country}')


def savefig_location_on_map(location: str,
                            lat: float,
                            lon: float,
                            path: str,
                            ):

    """
    Plot a single location on a map and save the figure
    :param location: location token
    :param lat: latitude
    :param lon: longitude
    :param path:
    """

    country, site = split_location_token(location)

    fig, ax = plt.subplots()

    m = _get_country_basemap(country)

    m.scatter(
        x=[lon],
        y=[lat],
        latlon=True,
        c='red',
        s=2,
    )

    ax.annotate(site,
                m(lon, lat),
                fontsize=_ANN_FONT_SIZE,
                )

    plt.savefig(path,
                dpi=_MAP_DPI,
                )

    plt.cla()
    plt.close()


def _get_countries_in_location_list(locations: list) -> set:
    # Get all countries that occur in a list of location tokens
    return {split_location_token(location)[0] for location in locations}


def _filter_country_data(country: str, *series) -> tuple:
    # Filter data that corresponds to locations in a certain country
    # Data features are assumed to be stored in lists
    # The first list (2nd function argument) is assumed to contain location tags

    # Create empty lists to store the filtered results
    f_series = tuple(list() for _ in series)

    # Iterate through the data and store relevant entries
    for i, location in enumerate(series[0]):
        # Split location tags to obtain the country
        loc_country, _ = split_location_token(location)
        if loc_country == country:
            for s, fs in zip(series, f_series):
                fs.append(s[i])

    return f_series


def savefig_locations_on_map(locations: list,
                             path: str,  # TODO
                             annotate: bool = True,
                             ):
    """
    Save maps showing all the locations included in the data
    :param locations: a list of location tokens (as they appear in the dataset)
    :param path:
    """

    os.makedirs(path, exist_ok=True)

    # Get all location coordinates
    location_data = get_locations_coordinates()
    lats = location_data.loc[locations]['lat'].values
    lons = location_data.loc[locations]['long'].values

    # Get all countries that are included in the location tokens
    # These are needed to know for which country we should make maps
    countries = _get_countries_in_location_list(locations)

    # Make a map for each country included in the data
    for country in countries:

        # Filter this country's data
        c_locations, c_lats, c_lons = _filter_country_data(country, locations, lats, lons)

        # # Create a path where the figure should be stored
        # path = os.path.join(PATH_FIGURES_DIR, country, '_global', 'locations_on_map.png')
        # _ensure_country_dirs(country)

        # Create the figure
        fig, ax = plt.subplots()
        # Obtain this country's basemap
        m = _get_country_basemap(country)
        # Plot the locations on the map
        m.scatter(
            x=c_lons,
            y=c_lats,
            latlon=True,
            c='red',
            s=2,
        )
        if annotate:
            # Annotate all locations with their name
            for lat, lon, location in zip(c_lats, c_lons, c_locations):
                _, site = split_location_token(location)
                ax.annotate(site,
                            m(lon, lat),
                            fontsize=_ANN_FONT_SIZE,
                            )
        # Save the figure
        plt.savefig(os.path.join(path, f'{country}.png'),
                    dpi=_MAP_DPI,
                    )
        # Clear results for constructing the next figure
        plt.cla()
        plt.close()


def savefig_location_r2s_on_map(r2s: list,
                                locations: list,
                                path: str,
                                filename: str,
                                ):
    """
    Save maps showing r2 values
    :param r2s: a list of r2 values (as floats)
    :param locations: a list of location tokens (as they appear in the dataset)
    :param path: path to folder where the maps will be saved
    :param filename: name of the file in which the maps will be saved
    """
    assert len(r2s) == len(locations)

    # Get all location coordinates
    location_data = get_locations_coordinates()
    lats = location_data.loc[locations]['lat'].values
    lons = location_data.loc[locations]['long'].values

    # Get all countries that are included in the location tokens
    # These are needed to know for which country we should make maps
    countries = _get_countries_in_location_list(locations)

    # Make a map for each country included in the data
    for country in countries:  # TODO -- tqdm

        # Filter this country's data
        c_locations, c_lats, c_lons, c_r2s = _filter_country_data(country, locations, lats, lons, r2s)

        # # Create a path where the figure should be stored
        c_path = os.path.join(path, country)
        os.makedirs(c_path, exist_ok=True)

        # Define a color scheme for plotting the r2 values
        # Values between 0 and 1 are scaled linearly from 'light gray' (0.8, 0.8, 0.8) to black (0, 0, 0)
        # Values below 0 are clipped to 0 for color representation
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
        colors = [(1 - (norm(r2) * 0.8 + 0.2),) * 3 for r2 in c_r2s]

        # Create the figure
        fig, ax = plt.subplots()
        # Obtain this country's basemap
        m = _get_country_basemap(country)
        # Plot the locations on the map
        m.scatter(
            x=c_lons,
            y=c_lats,
            latlon=True,
            c=colors,
            s=2,
        )
        # Annotate all locations with their r2
        for lat, lon, r2 in zip(c_lats, c_lons, c_r2s):
            ann = f'{r2:.2f}'
            ax.annotate(ann,
                        m(lon, lat),
                        fontsize=_ANN_FONT_SIZE,
                        )
        # Save the figure
        plt.savefig(os.path.join(c_path, filename),
                    dpi=_MAP_DPI,
                    )
        # Clear results for constructing the next figure
        plt.cla()
        plt.close()


def savefig_location_rmse_on_map(rmses: list,
                                 locations: list,
                                 path: str,
                                 filename: str,
                                 ):
    """
    Save maps showing rmse values
    :param rmses: a list of rmse values (as floats)
    :param locations: a list of location tokens (as they appear in the dataset)
    :param path: path to folder where the maps will be saved
    :param filename: name of the file in which the maps will be saved
    """
    assert len(rmses) == len(locations)

    # Get all location coordinates
    location_data = get_locations_coordinates()
    lats = location_data.loc[locations]['lat'].values
    lons = location_data.loc[locations]['long'].values

    # Get all countries that are included in the location tokens
    # These are needed to know for which country we should make maps
    countries = _get_countries_in_location_list(locations)

    # Make a map for each country included in the data
    for country in countries:  # TODO -- tqdm

        # Filter this country's data
        c_locations, c_lats, c_lons, c_rmses = _filter_country_data(country, locations, lats, lons, rmses)

        # # Create a path where the figure should be stored
        c_path = os.path.join(path, country)
        os.makedirs(c_path, exist_ok=True)

        # Define a color scheme for plotting the rmse values
        # Values between 7 and 0 are scaled linearly from 'light gray' (0.8, 0.8, 0.8) to black (0, 0, 0)
        # Values above 7 are clipped to 7 for color representation
        norm = matplotlib.colors.Normalize(vmin=0, vmax=7, clip=True)
        colors = [((norm(rmse) * 0.8),) * 3 for rmse in c_rmses]

        # Create the figure
        fig, ax = plt.subplots()
        # Obtain this country's basemap
        m = _get_country_basemap(country)
        # Plot the locations on the map
        m.scatter(
            x=c_lons,
            y=c_lats,
            latlon=True,
            c=colors,
            s=2,
        )
        # Annotate all locations with their rmse
        for lat, lon, r2 in zip(c_lats, c_lons, c_rmses):
            ann = f'{r2:.1f}'
            ax.annotate(ann,
                        m(lon, lat),
                        fontsize=_ANN_FONT_SIZE,
                        )
        # Save the figure
        plt.savefig(os.path.join(c_path, filename),
                    dpi=_MAP_DPI,
                    )
        # Clear results for constructing the next figure
        plt.cla()
        plt.close()


def savefig_location_mae_on_map(maes: list,
                                locations: list,
                                path: str,
                                filename: str,
                                ):
    """
    Save maps showing mae values
    :param maes: a list of mae values (as floats)
    :param locations: a list of location tokens (as they appear in the dataset)
    :param path: path to folder where the maps will be saved
    :param filename: name of the file in which the maps will be saved
    """
    assert len(maes) == len(locations)

    # Get all location coordinates
    location_data = get_locations_coordinates()
    lats = location_data.loc[locations]['lat'].values
    lons = location_data.loc[locations]['long'].values

    # Get all countries that are included in the location tokens
    # These are needed to know for which country we should make maps
    countries = _get_countries_in_location_list(locations)

    # Make a map for each country included in the data
    for country in countries:  # TODO -- tqdm

        # Filter this country's data
        c_locations, c_lats, c_lons, c_maes = _filter_country_data(country, locations, lats, lons, maes)

        # # Create a path where the figure should be stored
        c_path = os.path.join(path, country)
        os.makedirs(c_path, exist_ok=True)

        # Define a color scheme for plotting the mae values

        # Values between 7 and 0 are scaled linearly from 'light gray' (0.8, 0.8, 0.8) to black (0, 0, 0)
        # Values above 7 are clipped to 7 for color representation
        # norm = matplotlib.colors.Normalize(vmin=0, vmax=7, clip=True)
        # colors = [((norm(mae) * 0.8),) * 3 for mae in c_maes]

        c_1 = (0, 76 / 255, 153 / 255,)
        c_2 = (0, 128 / 255, 1,)
        c_3 = (102 / 255, 178 / 255, 1,)
        c_4 = (204 / 255, 229 / 255, 1,)

        def _mae_c_map(_mae):
            if 0 <= _mae < 5:
                return c_1
            if 5 <= _mae < 10:
                return c_2
            if 10 <= _mae < 20:
                return c_3
            return c_4

        colors = [_mae_c_map(mae) for mae in c_maes]

        # Create the figure
        fig, ax = plt.subplots()
        # Obtain this country's basemap
        m = _get_country_basemap(country)
        # Plot the locations on the map
        m.scatter(
            x=c_lons,
            y=c_lats,
            latlon=True,
            c=colors,
            s=2,
        )
        # Annotate all locations with their mae
        # for lat, lon, mae in zip(c_lats, c_lons, c_maes):
        #     ann = f'{mae:.1f}'
        #     ax.annotate(ann,
        #                 m(lon, lat),
        #                 fontsize=_ANN_FONT_SIZE,
        #                 )
        # Save the figure
        plt.savefig(os.path.join(c_path, filename),
                    dpi=_MAP_DPI,
                    )
        # Clear results for constructing the next figure
        plt.cla()
        plt.close()


def savefig_location_annotations_on_map(annotations: list,
                                        locations: list,
                                        # filename: str,
                                        path: str,
                                        colors=None,
                                        ):
    """
    Save maps showing the specified annotations
    :param annotations: a list of the annotations to use
    :param locations: a list of location tokens (as they appear in the dataset)
    :param path:
    :param colors:

    # :param filename: name of the file in which the maps will be saved
    """
    assert len(annotations) == len(locations)
    colors = colors or 'red'

    # Get all location coordinates
    location_data = get_locations_coordinates()
    lats = location_data.loc[locations]['lat'].values
    lons = location_data.loc[locations]['long'].values

    # Get all countries that are included in the location tokens
    # These are needed to know for which country we should make maps
    countries = _get_countries_in_location_list(locations)

    # Make a map for each country included in the data
    for country in countries:

        # Filter this country's data
        c_locations, c_lats, c_lons, c_anns = _filter_country_data(country, locations, lats, lons, annotations)

        # Create the figure
        fig, ax = plt.subplots()
        # Obtain this country's basemap
        m = _get_country_basemap(country)
        # Plot the locations on the map
        m.scatter(
            x=c_lons,
            y=c_lats,
            latlon=True,
            c=colors,  # TODO -- this doesnt work with multiple countries
            s=2,
        )
        # Annotate all locations as specified by the input
        for lat, lon, ann in zip(c_lats, c_lons, c_anns):
            ann = str(ann)  # cast to string if required
            ax.annotate(ann,
                        m(lon, lat),
                        fontsize=_ANN_FONT_SIZE,
                        )

        # Save the figure
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, f'{country}.png'),
                    dpi=_MAP_DPI,
                    )
        # Clear results for constructing the next figure
        plt.cla()
        plt.close()


# TODO -- regional plots


def plot_cultivar_maps() -> None:
    from data.regions import LOCATION_VARIETY_JAPAN, LOCATION_VARIETY_SWITZERLAND, LOCATION_VARIETY_SOUTH_KOREA, \
        LOCATION_VARIETY

    colors_map = {
        0: 'red',
        1: 'blue',
        2: 'green',
        3: 'purple',
        4: 'orange',
        5: 'brown',
    }
    # color = 'red'
    # colors_map = {
    #     0: color,
    #     1: color,
    #     2: color,
    #     3: color,
    #     4: color,
    #     5: color,
    # }
    marker_map = {
        0: '.',
        1: 'x',
        2: '*',
        3: 's',
        4: 'D',
        5: '^',
    }

    # Get all location coordinates
    location_data = get_locations_coordinates()

    for country, locations_varieties in zip(
            [
                'Japan',
                'Switzerland',
                'South Korea',
            ],
            [
                # LOCATION_VARIETY_JAPAN,
                {**LOCATION_VARIETY_JAPAN, **LOCATION_VARIETY_SOUTH_KOREA},
                LOCATION_VARIETY_SWITZERLAND,
                LOCATION_VARIETY_SOUTH_KOREA,
            ]
    ):

        locations = list(locations_varieties.keys())

        # Create the figure
        fig, ax = plt.subplots()
        # Obtain this country's basemap
        m = _get_country_basemap(country)

        for variety in set(locations_varieties.values()):

            v_locs = [loc for loc in locations if locations_varieties[loc] == variety]

            lats = location_data.loc[v_locs]['lat'].values
            lons = location_data.loc[v_locs]['long'].values

            m.scatter(
                x=lons,
                y=lats,
                latlon=True,
                c=colors_map[variety],
                s=8,
                marker=marker_map[variety],
            )

        path = os.path.join(config.PATH_FIGURES_DIR, 'variety_distribution')
        os.makedirs(path, exist_ok=True)

        # Save the figure
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, f'{country}.png'),
                    dpi=_MAP_DPI,
                    bbox_inches='tight',
                    )
        # Clear results for constructing the next figure
        plt.cla()
        plt.close()


if __name__ == '__main__':
    # from data.bloom_doy import get_locations
    # savefig_locations_on_map(
    #     get_locations(),
    #     os.path.join(config.PATH_FIGURES_DIR, 'location_maps'),
    #     annotate=False,
    # )

    plot_cultivar_maps()


# if __name__ == '__main__':
#     import sklearn.model_selection
#
#     from data.locations import get_data as get_location_data
#     _locs = get_location_data(set_index=True).index.values
#
#     from data.regions_japan import LOCATIONS_WO_OKINAWA
#
#     seed_location_split = config.SEED
#     _locs = list(LOCATIONS_WO_OKINAWA.keys())
#     train_locations, test_locations = sklearn.model_selection.train_test_split(_locs,
#                                                                                random_state=seed_location_split,
#                                                                                shuffle=True,
#                                                                                train_size=0.9,
#                                                                                )
#
#     savefig_location_annotations_on_map(
#         [''] * len(_locs),
#         train_locations + test_locations,
#         path='temp',
#         colors=['black'] * len(train_locations) + ['red'] * len(test_locations),
#     )
#
#
#     # # savefig_location_annotations_on_map(locations=_locs,
#     # savefig_location_annotations_on_map(locations=_locs,
#     #                                     path='temp',
#     #                                     annotations=[''] * len(_locs))
#     #
#     # fig, ax = plt.subplots()
#     #
#     # m = _get_basemap_japan()
#     #
#     # ax.spines['top'].set_visible(False)
#     # ax.spines['right'].set_visible(False)
#     # ax.spines['left'].set_visible(False)
#     # ax.spines['bottom'].set_visible(False)
#     #
#     # plt.savefig('temp2.png',
#     #             dpi=1000,
    #             )