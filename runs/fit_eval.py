import argparse

from runs.args_util.args_dataset import get_configured_dataset, configure_argparser_dataset
from runs.args_util.args_evaluation import evaluate_model_using_args, configure_argparser_evaluation
from runs.args_util.args_main import set_config_using_args, configure_argparser_main
from runs.args_util.args_model import fit_model, MODELS_KEYS_TO_CLS, configure_argparser_model

"""

    Fit and evaluate a model
    
"""


def _configure_argparser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Configure an argument parser to parse arguments for the fit/eval run
    """
    # Configure it to parse arguments related to the main flow of the program
    # (including the selection of which model to train)
    configure_argparser_main(parser)
    # Configure it to parse arguments related to building the dataset
    configure_argparser_dataset(parser)
    # Configure it to parse arguments related to the evaluation process
    configure_argparser_evaluation(parser)
    # Configure it to parse arguments related to the run procedure
    parser.add_argument('--skip_fit',
                        action='store_true',
                        help='If set, no model will be trained but instead will be loaded from disk',
                        )
    parser.add_argument('--skip_eval',
                        action='store_true',
                        help='If set, the trained/loaded model will not be evaluated',
                        )
    parser.add_argument('--model_cls',
                        type=str,
                        choices=list(MODELS_KEYS_TO_CLS.keys()),
                        required=True,
                        help='Specify the model class that is to be trained/evaluated',
                        )
    parser.add_argument('--model_name',
                        type=str,
                        help='Optionally specify a name for the model. If none is provided the model class will be '
                             'used. The name is used for storing model weights and evaluation files',
                        )

    # Parse the arguments that are known so far -- the remaining configuration is based on these initial arguments
    known_args, _ = parser.parse_known_args()

    # Configure the parser based on the model selection
    configure_argparser_model(MODELS_KEYS_TO_CLS[known_args.model_cls], parser)

    return parser


def _get_args() -> argparse.Namespace:
    """
    Configure an argument parser and use it to obtain arguments for configuring the run
    """

    description = 'Fit and evaluate a model'

    # Initialize a parser
    parser = argparse.ArgumentParser(description=description)
    # Configure it
    _configure_argparser(parser)
    # Parse all arguments
    args = parser.parse_args()

    # Obtain the model that was selected to be trained/evaluated
    model_cls = MODELS_KEYS_TO_CLS[args.model_cls]
    # Overwrite the model class keyword with the class itself
    args.model_cls = model_cls

    # Perform some checks for validity of the run configuration
    # validate_args(args)  # TODO

    # Return the final arguments
    return args


if __name__ == '__main__':

    # Get the provided program arguments
    args = _get_args()

    # Set global config (e.g. seeds)
    set_config_using_args(args)

    # Obtain the model class that will be trained/evaluated
    model_cls = args.model_cls
    # Get the model name (if specified, otherwise use the class name)
    model_name = args.model_name or model_cls.__name__

    # Configure the dataset based on the provided arguments
    dataset, _ = get_configured_dataset(args)

    # If training should be skipped -> load the model from disk
    if args.skip_fit:
        model = model_cls.load(model_name)  # TODO -- move to util
    # Otherwise, fit a new model
    else:
        model, _ = fit_model(args, dataset)
        # Save the trained model
        model.save(model_name)

    # Evaluate the model if not disabled
    if not args.skip_eval:
        evaluate_model_using_args(
            args,
            model,
            model_name,
            dataset,
        )
