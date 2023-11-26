import argparse

import torch

from models.base_torch_accumulation import BaseTorchAccumulationModel
from models.components.param_v3 import GroupedParameterMapping
from models.nn_chill_operator import NNChillModel
from runs.args_util.args_dataset import get_configured_dataset, configure_argparser_dataset
from runs.args_util.args_evaluation import configure_argparser_evaluation, evaluate_model_using_args
from runs.args_util.args_main import configure_argparser_main
from runs.args_util.args_model import MODELS_KEYS_TO_CLS, configure_argparser_model, TORCH_OPTIMIZERS


"""
    Fine-tune a pretrained model
    
    (i.e. fix the learned chill operator weights and solely update process-based model weights)
    Any parameter grouping constraints are removed
"""


def _configure_argparser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # Configure it to parse arguments related to the main flow of the program
    # (including the selection of which model to train)
    configure_argparser_main(parser)
    # Configure it to parse arguments related to building the dataset
    configure_argparser_dataset(parser)
    # Configure it to parse arguments related to the evaluation process
    configure_argparser_evaluation(parser)

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
    description = 'Fine-tune and evaluate a pre-trained model'

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

    # Return the final arguments
    return args


if __name__ == '__main__':

    args = _get_args()

    model_name = args.model_name or args.model_cls.__name__

    # Configure the dataset based on the provided arguments
    dataset, _ = get_configured_dataset(args)

    model = args.model_cls.load(model_name)

    assert isinstance(model, NNChillModel)

    if isinstance(model._pm_thc, GroupedParameterMapping):
        model._pm_thc = model._pm_thc.as_ungrouped()
    if isinstance(model._pm_thg, GroupedParameterMapping):
        model._pm_thg = model._pm_thg.as_ungrouped()
    if isinstance(model._pm_tbg, GroupedParameterMapping):
        model._pm_tbg = model._pm_tbg.as_ungrouped()
    if isinstance(model._pm_slc, GroupedParameterMapping):
        model._pm_slc = model._pm_slc.as_ungrouped()
        model._pm_slc.freeze()
    if isinstance(model._pm_slg, GroupedParameterMapping):
        model._pm_slg = model._pm_slg.as_ungrouped()
        model._pm_slg.freeze()

    model.freeze_operator_weights()

    device = torch.device('cuda') if torch.cuda.is_available() and not args.disable_cuda else torch.device('cpu')

    model_kwargs = dict()
    kwargs = {
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'scheduler_step_size': args.scheduler_step_size,
        'scheduler_decay': args.scheduler_decay,
        'clip_gradient': args.clip_gradient,
        'f_optim': TORCH_OPTIMIZERS[args.optimizer],
        'optim_kwargs': {
            'lr': args.lr,
        },
        'device': device,
        'model': model,
        'model_kwargs': model_kwargs,
    }

    model, _ = type(model).fit(
        dataset,
        **kwargs,
    )

    evaluate_model_using_args(
        args,
        model,
        f'{model_name}_finetuned',
        dataset,
    )
