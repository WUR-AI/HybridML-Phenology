import argparse

from sklearn.model_selection import train_test_split

import config
from datasets.dataset import load_debug_dataset, Dataset
from runs.args_util.location_groups import LOCATION_GROUPS


""" 
    
    This file contains:

        - Functions for configuring an ArgumentParser to parse arguments relating to the dataset
    
        - Functions for constructing a dataset based on the parsed arguments

"""


def configure_argparser_dataset(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Configure the provided argument parser to parse arguments that relate to the dataset
    The configuration is in-place
    :param parser: the parser that should be configured
    :return: a reference to the provided parser, now configured
    """
    parser.add_argument('--debug',
                        action='store_true',  # TODO -- check if saved debug dataset already exists, now it is assumed it does exist
                        help='If set, a small subset of the data is loaded (which is much faster)',
                        )

    parser.add_argument('--include_temperature',
                        action='store_true',
                        help='If set, the dataset will contain temperature data',
                        )
    parser.add_argument('--include_photoperiod',
                        action='store_true',
                        help='If set, the dataset will contain photoperiod data',
                        )
    parser.add_argument('--locations',
                        type=str,
                        choices=list(LOCATION_GROUPS.keys()),
                        default='all',
                        help='Location group that should be included in the dataset',
                        )
    parser.add_argument('--seed_year_split',
                        type=int,
                        help='Seed for controlling the year split. Overwritten if --seed was set',
                        )
    parser.add_argument('--train_size_years',
                        type=float,
                        default=0.75,
                        help='Percentage (between 0-1) of years that is used for training. Remainder is used for '
                             'testing',
                        )
    parser.add_argument('--hold_out_locations',
                        action='store_true',
                        help='If set, an additional train/test split will be done on the specified locations')
    parser.add_argument('--seed_location_split',
                        type=int,
                        help='Seed that is used when splitting locations. Set to config.seed by default. Ignored if '
                             'no split is made'
                        )
    parser.add_argument('--train_size_locations',
                        type=float,
                        default=0.75,
                        help='Percentage (between 0-1) of locations that is used for training. Remainder is used for '
                             'testing',
                        )

    return parser


def get_configured_dataset(args: argparse.Namespace) -> tuple:
    """
    Configure the dataset based on the parsed arguments
    :param args: the parsed arguments
    :return: the configured dataset
    """

    # Info dict for providing info about the process
    info = dict()

    # Load a small subset of the data for debugging
    if args.debug:
        return load_debug_dataset(), info

    # Determine the seed used for splitting years into train/test years
    # Different cases are distinguished:
    # (1) the `seed` argument was set -> use this as seed and ignore the `seed_year_split` argument
    if args.seed is not None:
        seed_year_split = args.seed
    # (2) only the `seed_year_split` was set -> use this one
    elif args.seed_year_split is not None:
        seed_year_split = args.seed_year_split
    # (3) no seed was explicitly given -> use the default one in the config
    else:
        seed_year_split = config.SEED

    # Determine the seed used for splitting locations into train/test locations
    # Different cases are distinguished:
    # (1) the `seed` argument was set -> use this as seed and ignore the `seed_location_split` argument
    if args.seed is not None:
        seed_location_split = args.seed
    # (2) only the `seed_location_split` was set -> use this one
    elif args.seed_location_split is not None:
        seed_location_split = args.seed_location_split
    # (3) no seed was explicitly given -> use the default one in the config
    else:
        seed_location_split = config.SEED

    # Get the locations used based on the provided group name
    locations = LOCATION_GROUPS[args.locations]

    # Determine the location train/test split (if any)
    if args.hold_out_locations:
        locations_train, locations_test = train_test_split(locations,
                                                           train_size=args.train_size_locations,
                                                           random_state=seed_location_split,
                                                           shuffle=True,
                                                           )
    else:
        locations_train, locations_test = locations, locations

    # Define the named arguments used for initializing the Dataset object
    kwargs = {
        'year_split': train_test_split(Dataset.YEAR_RANGE,
                                       train_size=args.train_size_years,
                                       random_state=seed_year_split,
                                       shuffle=True,
                                       ),
        'locations_train': locations_train,
        'locations_test': locations_test,
        'include_temperature': args.include_temperature,
        'include_photoperiod': args.include_photoperiod,
    }

    # Initialize the dataset
    dataset = Dataset(
        **kwargs,
    )

    return dataset, info

