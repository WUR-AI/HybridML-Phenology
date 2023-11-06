import argparse

import torch

import config


"""

    This file contains:

        - Functions for configuring an ArgumentParser to parse arguments relating to global project configurations
    
        - Functions for setting global project configurations based on the parsed arguments
    
"""


def configure_argparser_main(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Configure the provided argument parser to parse arguments that relate to global "project-level" config
    The configuration is in-place
    :param parser: the parser that should be configured
    :return: a reference to the provided parser, now configured
    """
    parser.add_argument('--seed',
                        type=int,
                        help='One seed to rule them all',
                        )
    parser.add_argument('--seed_torch',
                        type=int,
                        help='seed that is specifically used to fix PyTorch randomness (e.g. model weight '
                             'initialization, data shuffling when training, ..). Is ignored if `seed` was set')
    return parser


def set_config_using_args(args: argparse.Namespace) -> None:
    set_randomness_seed(args)


def set_randomness_seed(args: argparse.Namespace) -> None:

    # Determine the seed used for PyTorch models
    # Different cases are distinguished:
    # (1) the `seed` argument was set -> use this as seed and ignore the `seed_torch` argument
    if args.seed is not None:
        seed_torch = args.seed
    # (2) only the `seed_torch` was set -> use this one
    elif args.seed_torch is not None:
        seed_torch = args.seed_torch
    # (3) no seed was explicitly given -> use a random seed previously generated in config
    else:
        seed_torch = config.SEED

    torch.manual_seed(seed_torch)

    # np.random.seed(seed)


class ConfigException(Exception):
    pass  # Exception for misconfigured runs

