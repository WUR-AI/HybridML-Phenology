import argparse

from datasets.dataset import Dataset
from evaluation.evaluation import evaluate
from models.base import BaseModel


"""

    This file contains:

        - Functions for configuring an ArgumentParser to parse arguments relating to the evaluation procedure
    
        - Functions for evaluating a model based on the parsed arguments
        
"""


def configure_argparser_evaluation(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Configure an ArgumentParser to parse arguments relating to the evaluation procedure
    The configuration is in-place
    :param parser: the parser that should be configured
    :return: a reference to the provided parser, now configured
    """
    parser.add_argument('--plot_level',
                        type=str,
                        default='global',
                        choices=['global', 'local'],
                        help='Spatial level at which plots should be generated. Global/national by default. If set to '
                             '`local`, plots will also be generated per location',
                        )

    parser.add_argument('--generate_maps',
                        action='store_true',
                        help='If set, generate maps of local model performances',
                        )

    return parser


def evaluate_model_using_args(args: argparse.Namespace,
                              model: BaseModel,
                              model_name: str,
                              dataset: Dataset,
                              ) -> None:
    """
    Evaluate a model based on the parsed arguments
    :param args: the parsed arguments
    :param model: the model that is to be evaluated
    :param model_name: the name of the model that is to be evaluated
    :param dataset: the dataset that is used for evaluation
    """
    generate_local_plots = args.plot_level == 'local'
    generate_maps = args.generate_maps

    evaluate(
        model,
        dataset,
        model_name=model_name,
        generate_plots_scatter_global=True,
        generate_plots_scatter_local=generate_local_plots,
        generate_plots_maps=generate_maps,
    )


