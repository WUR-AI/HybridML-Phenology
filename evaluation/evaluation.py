import os.path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tabulate import tabulate
from tqdm import tqdm
import openpyxl

import config
from data import regions_japan

from datasets.dataset import Dataset, split_location_token
from evaluation.plots.coordinates import savefig_mae_over_coords
from evaluation.plots.maps import savefig_location_r2s_on_map, savefig_location_rmse_on_map, savefig_location_mae_on_map
from evaluation.plots.scatter import savefig_scatter_doys, savefig_scatter_doys_global, savefig_scatter_doys_local
from evaluation.util import group_items_per_location, group_items_per_country, group_items_per_region_japan
from models.base import BaseModel


def evaluate(model: BaseModel,
             dataset: Dataset,
             model_name: str = None,

             generate_plots_scatter_global: bool = True,
             generate_plots_scatter_local: bool = True,
             generate_plots_mae_over_coords: bool = True,  # TODO

             # generate_plots_residual: bool = False,  # TODO
             generate_plots_maps: bool = True,
             ):
    # If no model name is provided, use class name by default
    model_name = model_name or type(model).__name__

    # Set location for storing model evaluation
    path_eval = os.path.join(config.PATH_MODEL_EVAL_DIR, type(model).__name__, model_name)
    os.makedirs(path_eval, exist_ok=True)

    # Set location for storing figures
    path_figures = os.path.join(config.PATH_FIGURES_DIR, type(model).__name__, model_name)
    os.makedirs(path_figures, exist_ok=True)

    """
        Obtain the data
    """

    data_train = dataset.get_train_data()
    data_test = dataset.get_test_data()

    included_countries = dataset.countries
    included_countries_train = dataset.countries_train
    included_countries_test = dataset.countries_test

    included_locations = dataset.locations
    included_locations_train = dataset.locations_train
    included_locations_test = dataset.locations_test

    """
        ########################################################################
        #  Run the model on the train and test data                            #
        ########################################################################
    """

    model.set_mode_test()

    """
        Run model on train data
    """

    doys_true_train = []
    doys_pred_train = []
    locations_train = []

    progress_train = tqdm(data_train, desc=f'Running {type(model).__name__} on train data')

    for x in progress_train:
        doy_true = x['bloom_doy']
        doy_pred, _, _ = model.predict(x)
        location = x['location']

        doys_true_train.append(doy_true)
        doys_pred_train.append(doy_pred)
        locations_train.append(location)

    """
        Run model on test data
    """

    doys_true_test = []
    doys_pred_test = []
    locations_test = []

    progress_test = tqdm(data_test, desc=f'Running {type(model).__name__} on test data')

    for x in progress_test:
        doy_true = x['bloom_doy']
        doy_pred, _, _ = model.predict(x)
        location = x['location']

        doys_true_test.append(doy_true)
        doys_pred_test.append(doy_pred)
        locations_test.append(location)

    """
        #########################################################################
        #  Group predictions based on country / region / location               #
        #########################################################################
    """

    doys_true_per_location_train = group_items_per_location(locations_train, doys_true_train)
    doys_pred_per_location_train = group_items_per_location(locations_train, doys_pred_train)

    doys_true_per_location_test = group_items_per_location(locations_test, doys_true_test)
    doys_pred_per_location_test = group_items_per_location(locations_test, doys_pred_test)

    doys_true_per_country_train = group_items_per_country(locations_train, doys_true_train)
    doys_pred_per_country_train = group_items_per_country(locations_train, doys_pred_train)

    doys_true_per_country_test = group_items_per_country(locations_test, doys_true_test)
    doys_pred_per_country_test = group_items_per_country(locations_test, doys_pred_test)

    doys_true_per_region_train = group_items_per_region_japan(locations_train, doys_true_train)
    doys_pred_per_region_train = group_items_per_region_japan(locations_train, doys_pred_train)

    doys_true_per_region_test = group_items_per_region_japan(locations_test, doys_true_test)
    doys_pred_per_region_test = group_items_per_region_japan(locations_test, doys_pred_test)

    """
        #########################################################################
        #  Compute global/national/regional/local model performance statistics  #
        #########################################################################
    """

    metrics = {
        'mse': lambda _doy_true, _doy_pred: mean_squared_error(_doy_true, _doy_pred, squared=True),
        'rmse': lambda _doy_true, _doy_pred: mean_squared_error(_doy_true, _doy_pred, squared=False),
        'mae': lambda _doy_true, _doy_pred: mean_absolute_error(_doy_true, _doy_pred),
        'r2': lambda _doy_true, _doy_pred: r2_score(_doy_true, _doy_pred),
        'kendall_tau': lambda _doy_true, _doy_pred: kendalltau(_doy_true, _doy_pred).statistic,
        'mean': lambda _doy_true, _doy_pred: np.mean(_doy_pred),
        'std': lambda _doy_true, _doy_pred: np.std(_doy_pred),
        'count': lambda _doy_true, _doy_pred: len(_doy_true),
    }

    tables = []  # TODO dont overwrite column names

    """
        Compute global & national metrics
    """

    table_global_train, column_names = _create_table(
        metrics,
        {'global': doys_true_train, **doys_true_per_country_train},  # Extend grouped data with global doys
        {'global': doys_pred_train, **doys_pred_per_country_train},
        name='global_train',
    )

    table_global_test, column_names = _create_table(
        metrics,
        {'global': doys_true_test, **doys_true_per_country_test},
        {'global': doys_pred_test, **doys_pred_per_country_test},
        name='global_test',
    )

    df_global_train = _create_dataframe(
        metrics,
        {'global': doys_true_train, **doys_true_per_country_train},  # Extend grouped data with global doys
        {'global': doys_pred_train, **doys_pred_per_country_train},
    )

    df_global_test = _create_dataframe(
        metrics,
        {'global': doys_true_test, **doys_true_per_country_test},
        {'global': doys_pred_test, **doys_pred_per_country_test},
    )

    tables.append(table_global_train)
    tables.append(table_global_test)

    _save_df(df_global_train,
             path_eval,
             'global_train',
             )

    _save_df(df_global_test,
             path_eval,
             'global_test',
             )

    if 'Japan' in included_countries:

        """
            Compute regional metrics
        """

        table_regional_train, column_names = _create_table(
            metrics,  # Replace region id with region names
            {regions_japan.REGIONS[region_id]: vals for region_id, vals in doys_true_per_region_train.items()},
            {regions_japan.REGIONS[region_id]: vals for region_id, vals in doys_pred_per_region_train.items()},
            name='regional_japan_train',
        )

        table_regional_test, column_names = _create_table(
            metrics,  # Replace region id with region names
            {regions_japan.REGIONS[region_id]: vals for region_id, vals in doys_true_per_region_test.items()},
            {regions_japan.REGIONS[region_id]: vals for region_id, vals in doys_pred_per_region_test.items()},
            name='regional_japan_test',
        )

        tables.append(table_regional_train)
        tables.append(table_regional_test)

        df_regional_train = _create_dataframe(
            metrics,  # Replace region id with region names
            {regions_japan.REGIONS[region_id]: vals for region_id, vals in doys_true_per_region_train.items()},
            {regions_japan.REGIONS[region_id]: vals for region_id, vals in doys_pred_per_region_train.items()},
        )

        df_regional_test = _create_dataframe(
            metrics,  # Replace region id with region names
            {regions_japan.REGIONS[region_id]: vals for region_id, vals in doys_true_per_region_test.items()},
            {regions_japan.REGIONS[region_id]: vals for region_id, vals in doys_pred_per_region_test.items()},
        )

        _save_df(df_regional_train,
                 path_eval,
                 'regional_japan_train',
                 )

        _save_df(df_regional_test,
                 path_eval,
                 'regional_japan_test',
                 )

    """
        Compute local metrics -- grouped by country
    """

    for country in included_countries_train:

        table_local_country_train, column_names = _create_table(
            metrics,
            {loc: vals for loc, vals in doys_true_per_location_train.items() if split_location_token(loc)[0] == country},
            {loc: vals for loc, vals in doys_pred_per_location_train.items() if split_location_token(loc)[0] == country},
            name=f'local_{country.lower()}_train',
        )

        tables.append(table_local_country_train)

        df_local_country_train = _create_dataframe(
            metrics,
            {loc: vals for loc, vals in doys_true_per_location_train.items() if split_location_token(loc)[0] == country},
            {loc: vals for loc, vals in doys_pred_per_location_train.items() if split_location_token(loc)[0] == country},
        )

        _save_df(df_local_country_train,
                 path_eval,
                 f'local_{country.lower()}_train',
                 )

    for country in included_countries_test:

        table_local_country_test, column_names = _create_table(
            metrics,
            {loc: vals for loc, vals in doys_true_per_location_test.items() if split_location_token(loc)[0] == country},
            {loc: vals for loc, vals in doys_pred_per_location_test.items() if split_location_token(loc)[0] == country},
            name=f'local_{country.lower()}_test',
        )

        tables.append(table_local_country_test)

        df_local_country_test = _create_dataframe(
            metrics,
            {loc: vals for loc, vals in doys_true_per_location_test.items() if split_location_token(loc)[0] == country},
            {loc: vals for loc, vals in doys_pred_per_location_test.items() if split_location_token(loc)[0] == country},
        )

        _save_df(df_local_country_test,
                 path_eval,
                 f'local_{country.lower()}_test',
                 )

    # TODO -- save tables -- fix

    # _save_tables(
    #     tables,
    #     column_names,
    #     path_eval,
    #     'table_metrics'
    # )

    """
        #########################################################################
        #  Optionally generate plots                                            #
        #########################################################################
    """

    """
        Scatter DOYs globally
    """
    if generate_plots_scatter_global:

        savefig_scatter_doys_global(
            doys_true_train,
            doys_pred_train,
            locations_train,
            path=os.path.join(path_figures, 'plots_scatter_global_train')
        )

        savefig_scatter_doys_global(
            doys_true_test,
            doys_pred_test,
            locations_test,
            path=os.path.join(path_figures, 'plots_scatter_global_test')
        )

    """
        Scatter DOYs locally
    """
    if generate_plots_scatter_local:

        savefig_scatter_doys_local(
            doys_true_train,
            doys_pred_train,
            locations_train,
            path=os.path.join(path_figures, 'plots_scatter_local_train')
        )

        savefig_scatter_doys_local(
            doys_true_test,
            doys_pred_test,
            locations_test,
            path=os.path.join(path_figures, 'plots_scatter_local_test')
        )

    """
        Plot model performance over coordinates
    """
    if generate_plots_mae_over_coords:
        savefig_mae_over_coords(
            doys_true_test,
            doys_pred_test,
            locations_test,
            path=os.path.join(path_figures, 'plots_mae_over_coords')
        )


def _save_tables(tables: list,
                 column_names: list,  # Assumed to be equal for all tables
                 path: str,
                 name: str,
                 ):

    with open(os.path.join(path, f'{name}.md'), 'w') as f:

        for table in tables:
            f.write(tabulate(table, headers=column_names, floatfmt='.2f', tablefmt='github'))
            f.write('\n\n\n\n')

    with open(os.path.join(path, f'{name}.txt'), 'w') as f:

        for table in tables:
            f.write(tabulate(table, headers=column_names, floatfmt='.2f', tablefmt='pretty'))
            f.write('\n\n\n\n')

    with open(os.path.join(path, f'{name}.tsv'), 'w') as f:

        for table in tables:
            f.write(tabulate(table, headers=column_names, floatfmt='.2f', tablefmt='tsv'))


def _save_df(df: pd.DataFrame,
             path: str,
             name: str,
             ):
    df.to_csv(os.path.join(path, f'{name}.csv'))
    df.to_excel(os.path.join(path, f'{name}.xlsx'))


def _create_table(metrics: dict,
                  grouped_doys_true: dict,
                  grouped_doys_pred: dict,
                  name: str = None,
                  ) -> tuple:
    """
        Helper function to create a table containing scores for grouped day-of-year-prediction results
        :param metrics: a dict of metrics that will be used to evaluate the predictions
                        - key: metric name that will be used as column name
                        - val: callable for computing the metric
                               assumes to accept arguments as follows: f(doys_true, doys_pred)
        :param grouped_doys_true: grouped doy values in a dict with
                                            - key: group name (str)
                                            - val: list of doy values
        :param grouped_doys_pred:  "
        :return: a two-tuple consisting of:
                    - a list of lists forming a table
                    - a list of column names
        """
    name = name or 'eval score'
    column_names = [name]
    column_names += [metric_name for metric_name in metrics.keys()]

    table = []

    for group in grouped_doys_true.keys():
        doys_true = grouped_doys_true[group]
        doys_pred = grouped_doys_pred[group]

        line = [group]
        line += [f_metric(doys_true, doys_pred) for _, f_metric in metrics.items()]

        table.append(line)

    return table, column_names


def _create_dataframe(metrics: dict,
                      grouped_doys_true: dict,
                      grouped_doys_pred: dict,
                      ) -> pd.DataFrame:
    df = []
    for area in grouped_doys_true.keys():
        line = {'area': area,
                **{metric: f_metric(grouped_doys_true[area], grouped_doys_pred[area])
                   for metric, f_metric in metrics.items()},
                }
        df.append(line)

    df = pd.DataFrame(df,
                      columns=['area',
                               *[metric for metric in metrics.keys()],
                               ]  # Column names need to be set in case the df is empty
                      )
    df = df.set_index('area')
    return df


def get_evaluation_result(model_cls: callable,
                          model_name: str = None,
                          ) -> dict:

    # If no model name is provided, use class name by default
    model_name = model_name or model_cls.__name__

    # Location where model evaluation is stored
    path_eval = os.path.join(config.PATH_MODEL_EVAL_DIR, model_cls.__name__, model_name)

    dfs = dict()

    df_global_train = pd.read_csv(os.path.join(path_eval, f'global_train.csv'))
    dfs['global_train'] = df_global_train
    df_global_test = pd.read_csv(os.path.join(path_eval, f'global_test.csv'))
    dfs['global_test'] = df_global_test

    # TODO


    pass  # TODO

#     """
#         #########################################################################
#         #  Create scatter plots of the predictions                              #
#         #########################################################################
#     """
#     if generate_plots_scatter:
#         """
#             Of all data
#         """
#
#         savefig_scatter_doys(doys_true_train,
#                              doys_pred_train,
#                              path,
#                              fn='scatter_doys_train',
#                              title=f'Scatter plot (global, train) of {type(model).__name__}')
#
#         savefig_scatter_doys(doys_true_test,
#                              doys_pred_test,
#                              path,
#                              fn='scatter_doys_test',
#                              title=f'Scatter plot (global, test) of {type(model).__name__}')
#
#         """
#             Grouped per country
#         """
#
#         # TODO -- copy the rest
#
#     """
#         #########################################################################
#         #  Create scatter plots of the predictions                              #
#         #########################################################################
#     """
#     if generate_plots_residual:
#         pass  # TODO -- copy
#
#     """
#         #########################################################################
#         #  Create maps of the prediction scores                                 #
#         #########################################################################
#     """
#     if generate_plots_maps:
#         savefig_location_r2s_on_map(
#             [r2_score(doys_true_per_location_train[loc], doys_pred_per_location_train[loc])
#              for loc in doys_true_per_location_train.keys()],
#             list(doys_true_per_location_train.keys()),
#             path,
#             'map_r2_train.png',
#         )
#
#         savefig_location_r2s_on_map(
#             [r2_score(doys_true_per_location_test[loc], doys_pred_per_location_test[loc])
#              for loc in doys_true_per_location_test.keys()],
#             list(doys_true_per_location_test.keys()),
#             path,
#             'map_r2_test.png',
#         )
#
#         savefig_location_rmse_on_map(
#             [mean_squared_error(doys_true_per_location_train[loc], doys_pred_per_location_train[loc], squared=False)
#              for loc in doys_true_per_location_train.keys()],
#             list(doys_true_per_location_train.keys()),
#             path,
#             'map_rmse_train.png',
#         )
#
#         savefig_location_rmse_on_map(
#             [mean_squared_error(doys_true_per_location_test[loc], doys_pred_per_location_test[loc], squared=False)
#              for loc in doys_true_per_location_test.keys()],
#             list(doys_true_per_location_test.keys()),
#             path,
#             'map_rmse_test.png',
#         )
#
#         savefig_location_mae_on_map(
#             [mean_absolute_error(doys_true_per_location_train[loc], doys_pred_per_location_train[loc])
#              for loc in doys_true_per_location_train.keys()],
#             list(doys_true_per_location_train.keys()),
#             path,
#             'map_mae_train.png',
#         )
#
#         # savefig_location_mae_on_map(
#         #     [mean_absolute_error(doys_true_per_location_test[loc], doys_pred_per_location_test[loc])
#         #      for loc in doys_true_per_location_test.keys()],
#         #     list(doys_true_per_location_test.keys()),
#         #     path,
#         #     'map_mae_test.png',
#         # )
#
