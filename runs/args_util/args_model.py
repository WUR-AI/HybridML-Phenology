import argparse

import torch

import data.regions
from datasets.dataset import Dataset
from models.base_torch import DummyTorchModel, BaseTorchModel
from models.base_torch_accumulation import BaseTorchAccumulationModel
from models.components.param_v3 import LocalParameterMapping, GlobalParameterMapping, GroupedParameterMapping
from models.diff_utah import DiffUtahModel
from models.lstm import LSTMModel, LSTMModel2, LSTMModel3
from models.mean import MeanModel
from models.nn_chill_operator import NNChillModel
from models.process_based.base_accumulation import BaseAccumulationModel, BaseLocalAccumulationModel
from models.process_based.chill_days import ChillDaysModel, LocalChillDaysModel
from models.process_based.chill_hours import ChillHoursModel, LocalChillHoursModel
from models.process_based.utah_chill import UtahChillModel, LocalUtahChillModel
from runs.args_util.args_main import ConfigException
from runs.args_util.location_groups import LOCATION_GROUPS

""" 

    This file contains:

        - Functions for configuring an ArgumentParser to parse arguments relating to various models

        - Functions for constructing and fitting a model based on the parsed arguments

"""

"""
    Registered models that have been configured to be trained using this code

    Adding a model requires the following steps:
        - Configure any additional arguments that need to be specified in `configure_argparser_model`
        - Optionally set checks in `validate_args` to validate the arguments for correctness before running everything
        - Specify how a model should be trained with the provided arguments in `fit_model`
"""
MODELS = [
    # Fits a mean DOY per location
    MeanModel,  # For testing
    # MLP that predicts DOY based on a flattened temperature input
    DummyTorchModel,  # For testing

    # Process Based models that use a single parameter set
    ChillHoursModel,
    UtahChillModel,
    ChillDaysModel,

    # Process Based models that use a parameter set per location
    LocalChillHoursModel,
    LocalUtahChillModel,
    LocalChillDaysModel,

    # Differentiable approximation to the Utah model
    DiffUtahModel,

    # Learned chill operator
    NNChillModel,

    # LSTM baseline
    # LSTMModel,
    LSTMModel2,
    LSTMModel3,

]

"""
    Mapping of keywords to model classes
    The user will specify one of these keywords to select which model will be trained/evaluated
    The mapping allows for calling class-specific methods for the selected model
"""
MODELS_KEYS_TO_CLS = {
    cls.__name__: cls for cls in MODELS
}


def configure_argparser_model(model_cls: callable, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Configure the provided argument parser to parse arguments that relate to constructing various models
    The configuration is in-place
    :param model_cls: the class of the model
    :param parser: the parser that should be configured
    :return: a reference to the provided parser, now configured
    """
    if model_cls == MeanModel:
        return parser
    if model_cls == DummyTorchModel:
        configure_argparser_fit_torch(parser)
        return parser

    if issubclass(model_cls, BaseAccumulationModel):
        return parser

    if issubclass(model_cls, BaseLocalAccumulationModel):
        return parser

    if issubclass(model_cls, BaseTorchModel):  # TODO -- more explicit?
        configure_argparser_fit_torch(parser)
        if issubclass(model_cls, BaseTorchAccumulationModel):
            parser.add_argument('--loss_f',
                                type=str,
                                choices=BaseTorchAccumulationModel.LOSSES,
                                default=BaseTorchAccumulationModel.LOSSES[0],
                                help='Loss function that is to be used for training the model',
                                )
            parser.add_argument('--inference_mode_train',
                                type=str,
                                choices=BaseTorchAccumulationModel.INFERENCE_MODES,
                                default=BaseTorchAccumulationModel.INFERENCE_MODES[0],
                                help='Specify how to compute the blooming index',
                                )
            parser.add_argument('--inference_mode_test',
                                type=str,
                                choices=BaseTorchAccumulationModel.INFERENCE_MODES,
                                default=BaseTorchAccumulationModel.INFERENCE_MODES[0],
                                help='Specify how to compute the blooming index',
                                )
            configure_argument_parser_parameter_models(parser)

        return parser

    raise ConfigException('Model class not recognized while configuring the argument parser!')


TORCH_OPTIMIZERS = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam,
}


def configure_argparser_fit_torch(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Configure the provided argument parser to parse arguments that relate to training a PyTorch model
    The configuration is in-place
    :param parser: the parser that should be configured
    :return: a reference to the provided parser, now configured
    """
    parser.add_argument('--num_epochs',
                        type=int,
                        default=1,
                        help='Number of iterations of training over the entire training dataset',
                        )
    parser.add_argument('--batch_size',
                        type=int,
                        default=None,
                        help='Batch size used during the optimization process. Default is None, meaning that '
                             'gradients will be computed over the entire dataset',
                        )
    parser.add_argument('--scheduler_step_size',
                        type=int,
                        default=None,
                        help='Learning rate is decayed every `scheduler_step_size` iterations. If not specified, '
                             'the learning rate will never be decayed',
                        )
    parser.add_argument('--scheduler_decay',
                        type=float,
                        default=0.5,
                        help='Factor by which the learning rate will be scaled when decaying the learning rate',
                        )
    parser.add_argument('--clip_gradient',
                        type=float,
                        default=None,
                        help='Value by which the gradient values will be clipped. No clipping is done when left '
                             'unspecified',
                        )
    parser.add_argument('--optimizer',
                        type=str,
                        choices=list(TORCH_OPTIMIZERS.keys()),
                        default='sgd',
                        help='Optimizer that is used when fitting the model',
                        )
    parser.add_argument('--lr',
                        type=float,
                        default=1e-2,
                        help='Learning rate',
                        )
    parser.add_argument('--weight_decay',
                        type=float,
                        default=None,
                        help='Optimizer weight decay regularization')
    parser.add_argument('--disable_cuda',
                        action='store_true',
                        help='Flag that controls whether GPU usage should be disabled',
                        )

    return parser


# Options for configuring a parameter model
PARAMETER_MODELS_KEYS = [
    # Constant value that is not learned
    'constant',
    # Map parameter sets to each location individually
    'local',
    # Use one set of parameters
    'global',
    # Map parameter sets to each cultivar restricted to Japan
    'japan_cultivars',
    # Map parameter sets to each cultivar
    'known_cultivars',
]


def configure_argument_parser_parameter_models(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Configure the provided argument parser to parse arguments that relate to configuring a model to provide
    parameters based on the location
    The configuration is in-place
    :param parser: the parser that should be configured
    :return: a reference to the provided parser, now configured
    """
    # parser.add_argument('--parameter_model',
    #                     type=str,
    #                     default='local',
    #                     choices=PARAMETER_MODELS_KEYS,
    #                     help='Model that should be used to map locations to parameter values. If none is specified, '
    #                          'each location is mapped to a tuple of parameters.',
    #                     )

    parser.add_argument('--parameter_model_thc',
                        type=str,
                        default='local',
                        choices=PARAMETER_MODELS_KEYS[1:],
                        help='Model that is used to map locations to "chill threshold" parameter values. If none is '
                             'specified, each location is mapped to one parameter',
                        )
    parser.add_argument('--parameter_model_thg',
                        type=str,
                        default='local',
                        choices=PARAMETER_MODELS_KEYS[1:],
                        help='Model that is used to map locations to "growth threshold" parameter values. If none is '
                             'specified, each location is mapped to one parameter',
                        )
    parser.add_argument('--parameter_model_tbg',
                        type=str,
                        default='local',
                        choices=PARAMETER_MODELS_KEYS[1:],
                        help='Model that is used to map locations to "base temperature" parameter values. If none is '
                             'specified, each location is mapped to one parameter',
                        )

    parser.add_argument('--parameter_model_slc',
                        type=str,
                        default='constant',
                        choices=PARAMETER_MODELS_KEYS,
                        help='Model that is used to map locations to parameter values controlling the slope of the '
                             'soft chill threshold. If none is specified, a constant is used'
                        )
    parser.add_argument('--parameter_model_slg',
                        type=str,
                        default='local',
                        choices=PARAMETER_MODELS_KEYS[1:],
                        help='Model that is used to map locations to parameter values controlling the slope of the '
                             'soft chill threshold. If none is specified, each location is mapped to one parameter'
                        )

    return parser


"""
    
    Create and fit models based on the parsed arguments

"""


def fit_model(args: argparse.Namespace, dataset: Dataset) -> tuple:
    """
    Fit a model based on the parsed arguments
    :param args: the parsed arguments
    :param dataset: the dataset to be used for fitting the model
    :return: a two-tuple consisting of
        - the trained model
        - a dict containing info about the fitting procedure
    """
    # Obtain the model class that will be trained/evaluated
    model_cls = args.model_cls
    # Get the model name (if specified, otherwise use the class name)
    model_name = args.model_name or model_cls.__name__

    # TODO -- fit differentiable utah model using grid search
    if issubclass(model_cls, BaseAccumulationModel):
        return model_cls.fit(dataset, name=model_name)
    if issubclass(model_cls, BaseLocalAccumulationModel):
        return model_cls.fit(dataset, name=model_name)
    if issubclass(model_cls, BaseTorchModel):
        return fit_torch_model_using_args(model_cls,
                                          dataset,
                                          args,
                                          )

    raise ConfigException('Model class not recognized while trying to fit the model!')


def fit_torch_model_using_args(model_cls: callable,
                               dataset: Dataset,
                               args: argparse.Namespace,
                               # model_kwargs: dict = None,
                               ) -> tuple:
    """
    Fit a PyTorch model using the parsed arguments
    :param model_cls: the class of the model
    :param dataset: the dataset to be used for fitting the model
    :param args: the parsed arguments
    # :param model_kwargs:
    :return:
    """
    # model_kwargs = model_kwargs or dict()
    device = torch.device('cuda') if torch.cuda.is_available() and not args.disable_cuda else torch.device('cpu')

    model_kwargs = dict()
    if issubclass(model_cls, BaseTorchAccumulationModel):

        model_kwargs['loss_f'] = args.loss_f
        model_kwargs['inference_mode_train'] = args.inference_mode_train
        model_kwargs['inference_mode_test'] = args.inference_mode_test

        model_kwargs['parameter_model_thc'] = _param_model_from_key(args, args.parameter_model_thc)
        model_kwargs['parameter_model_thg'] = _param_model_from_key(args, args.parameter_model_thg)
        model_kwargs['parameter_model_tbg'] = _param_model_from_key(args, args.parameter_model_tbg)

        model_kwargs['parameter_model_slc'] = _param_model_from_key(args, args.parameter_model_tbg, init_value=1.0)
        model_kwargs['parameter_model_slg'] = _param_model_from_key(args, args.parameter_model_tbg, init_value=1.0)

    kwargs_optim = {
        'lr': args.lr,
    }
    if args.weight_decay is not None:
        kwargs_optim['weight_decay'] = args.weight_decay

    kwargs = {
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'scheduler_step_size': args.scheduler_step_size,
        'scheduler_decay': args.scheduler_decay,
        'clip_gradient': args.clip_gradient,
        'f_optim': TORCH_OPTIMIZERS[args.optimizer],
        'optim_kwargs': kwargs_optim,
        'device': device,
        'model_kwargs': model_kwargs,
    }

    return model_cls.fit(dataset,
                         **kwargs,
                         )


def _param_model_from_key(args: argparse.Namespace, key: str, init_value: float = 0,):
    locations = LOCATION_GROUPS[args.locations]

    if key == 'constant':
        return None
    if key == 'local':
        return LocalParameterMapping(locations, init_val=init_value,)
    if key == 'global':
        return GlobalParameterMapping(locations, init_val=init_value,)
    if key == 'japan_cultivars':
        return GroupedParameterMapping(data.regions.LOCATION_VARIETY_JAPAN, init_val=init_value,)
    if key == 'known_cultivars':
        return GroupedParameterMapping(data.regions.LOCATION_VARIETY, init_val=init_value,)

    raise ConfigException(f'Cannot configure parameter model "{args.parameter_model}"')
