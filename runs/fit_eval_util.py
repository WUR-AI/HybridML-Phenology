import argparse

import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.optim

import config
import data.bloom_doy
import data.regions_japan
from datasets.dataset import Dataset, load_debug_dataset
from evaluation.evaluation import evaluate
from models.base import BaseModel
from models.base_torch import BaseTorchModel, DummyTorchModel
from models.base_torch_accumulation import BaseTorchAccumulationModel
from models.components.param import ParamNet, LocalParams, GlobalParams
from models.diff_utah import DiffUtahModel
from models.mean import MeanModel
from models.nn_chill_operator import NNChillModel
from models.process_based.base_accumulation import BaseLocalAccumulationModel, BaseAccumulationModel
from models.process_based.chill_days import LocalChillDaysModel, ChillDaysModel
from models.process_based.chill_hours import LocalChillHoursModel, ChillHoursModel
from models.process_based.utah_chill import LocalUtahChillModel, UtahChillModel

"""
    Pre-configured location groups on which the models can be run
    Defines a mapping from
        group_name -> list of location tokens in group
"""
_LOCATION_GROUPS = {
    'all': data.bloom_doy.get_locations(),
    'japan': data.bloom_doy.get_locations_japan(),
    'switzerland': data.bloom_doy.get_locations_switzerland(),
    'south_korea': data.bloom_doy.get_locations_south_korea(),
    'usa': data.bloom_doy.get_locations_usa(),
    'japan_wo_okinawa': list(data.regions_japan.LOCATIONS_WO_OKINAWA.keys()),
    'japan_yedoenis': list(data.regions_japan.LOCATIONS_JAPAN_YEDOENIS.keys()),
    'japan_south_korea': data.bloom_doy.get_locations_japan() + data.bloom_doy.get_locations_south_korea(),
    'japan_switzerland': data.bloom_doy.get_locations_japan() + data.bloom_doy.get_locations_switzerland(),
    'no_us': data.bloom_doy.get_locations_japan() + data.bloom_doy.get_locations_switzerland() + data.bloom_doy.get_locations_south_korea(),
    'japan_hokkaido': list(data.regions_japan.LOCATIONS_HOKKAIDO.keys()),
    'japan_tohoku': list(data.regions_japan.LOCATIONS_TOHOKU.keys()),
    'japan_hokuriku': list(data.regions_japan.LOCATIONS_HOKURIKU.keys()),
    'japan_kanto_koshin': list(data.regions_japan.LOCATIONS_KANTO_KOSHIN.keys()),
    'japan_kinki': list(data.regions_japan.LOCATIONS_KINKI.keys()),
    'japan_chugoku': list(data.regions_japan.LOCATIONS_CHUGOKU.keys()),
    'japan_tokai': list(data.regions_japan.LOCATIONS_TOKAI.keys()),
    'japan_shikoku': list(data.regions_japan.LOCATIONS_SHIKOKU.keys()),
    'japan_kyushu_north': list(data.regions_japan.LOCATIONS_KYUSHU_NORTH.keys()),
    'japan_kyushu_south_amami': list(data.regions_japan.LOCATIONS_KYUSHU_SOUTH_AMAMI.keys()),
    'japan_okinawa': list(data.regions_japan.LOCATIONS_OKINAWA.keys()),
}

"""
    Registered models that have been configured to be trained using this code
    
    Adding a model requires the following steps:
        - Configure any additional arguments that need to be specified in `configure_argparser_model`
        - Optionally set checks in `validate_args` to validate the arguments for correctness before running everything
        - Specify how a model should be trained with the provided arguments in `fit_model`
"""
_MODELS = [
    MeanModel,  # For testing
    DummyTorchModel,  # For testing

    ChillHoursModel,
    UtahChillModel,
    ChillDaysModel,

    LocalChillHoursModel,
    LocalUtahChillModel,
    LocalChillDaysModel,

    DiffUtahModel,

    # Learned chill operator cnn

    NNChillModel,


]

"""
    Mapping of keywords to model classes
    The user will specify one of these keywords to select which model will be trained/evaluated
    The mapping allows for calling class-specific methods for the selected model
"""
_MODELS_KEYS_TO_CLS = {
    cls.__name__: cls for cls in _MODELS
}


"""

    CODE FOR PARSING ARGUMENTS

"""


def get_args() -> argparse.Namespace:
    """
    Parse the arguments that have been provided
    :return: an argparse.Namespace object containing the arguments
    """
    # General run description
    description = 'Fit and evaluate a model'

    # Initialize a parser
    parser = argparse.ArgumentParser(description=description)
    # Configure it to parse arguments related to the main flow of the program
    # (including the selection of which model to train)
    configure_argparser_main(parser)
    # Configure it to parse arguments related to building the dataset
    configure_argparser_dataset(parser)
    # Configure it to parse arguments related to the evaluation process
    configure_argparser_evaluation(parser)

    # Parse the arguments that are known so far -- the remaining configuration is based on these initial arguments
    known_args, _ = parser.parse_known_args()
    # Obtain the model that was selected to be trained/evaluated
    model_cls = _MODELS_KEYS_TO_CLS[known_args.model_cls]

    # Configure the parser based on the model selection
    configure_argparser_model(model_cls, parser)

    # Parse all arguments
    args = parser.parse_args()
    # Overwrite the model class keyword with the class itself
    args.model_cls = model_cls

    # Return the final arguments
    return args


def configure_argparser_main(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Configure the provided argument parser to parse arguments that relate to the main flow of the run
    The configuration is in-place
    :param parser: the parser that should be configured
    :return: a reference to the provided parser, now configured
    """

    parser.add_argument('--model_cls',
                        type=str,
                        choices=list(_MODELS_KEYS_TO_CLS.keys()),
                        required=True,
                        help='Specify the model class that is to be trained/evaluated',
                        )

    parser.add_argument('--model_name',
                        type=str,
                        help='Optionally specify a name for the model. If none is provided the model class will be '
                             'used. The name is used for storing model weights and evaluation files',
                        )

    parser.add_argument('--seed',
                        type=int,
                        help='One seed to rule them all',
                        )
    parser.add_argument('--seed_torch',
                        type=int,
                        help='seed that is specifically used to fix PyTorch randomness (e.g. model weight '
                             'initialization, data shuffling when training, ..). Is ignored if `seed` was set')

    parser.add_argument('--skip_fit',
                        action='store_true',
                        help='If set, no model will be trained but instead will be loaded from disk',
                        )

    parser.add_argument('--skip_eval',
                        action='store_true',
                        help='If set, the trained/loaded model will not be evaluated',
                        )

    return parser


def configure_argparser_dataset(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Configure the provided argument parser to parse arguments that relate to the dataset
    The configuration is in-place
    :param parser: the parser that should be configured
    :return: a reference to the provided parser, now configured
    """
    parser.add_argument('--debug',
                        action='store_true',  # TODO -- check if saved debug dataset already exists, now it is assumed it does exist
                        help='If set, a small subset of the data is loaded (which is much faster)')

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
                        choices=list(_LOCATION_GROUPS.keys()),
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


def configure_argparser_evaluation(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """

    :param parser:
    :return:
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


def configure_argparser_model(model_cls: callable, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """

    :param model_cls:
    :param parser:
    :return:
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
            configure_argument_parser_parameter_model(parser)

        return parser

    raise ConfigException('Model class not recognized while configuring the argument parser!')


_OPTIMIZERS = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam,
}


def configure_argparser_fit_torch(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """

    :param parser:
    :return:
    """
    parser.add_argument('--num_epochs',
                        type=int,
                        default=1,
                        )
    parser.add_argument('--batch_size',
                        type=int,
                        default=None,
                        )
    parser.add_argument('--scheduler_step_size',
                        type=int,
                        default=None,
                        )
    parser.add_argument('--scheduler_decay',
                        type=float,
                        default=0.5,
                        )
    parser.add_argument('--clip_gradient',
                        type=float,
                        default=None,
                        )
    parser.add_argument('--optimizer',
                        type=str,
                        choices=list(_OPTIMIZERS.keys()),
                        default='sgd',
                        )
    parser.add_argument('--lr',
                        type=float,
                        default=1e-2,
                        )
    parser.add_argument('--disable_cuda',
                        action='store_true',
                        )

    return parser


_PARAMETER_MODELS = [
    LocalParams,  # Parameter set per location
    GlobalParams,  # Shared parameter set for all locations
    ParamNet,  # Parameterize using NN
]

_PARAMETER_MODELS_DEFAULT = _PARAMETER_MODELS[0]

_PARAMETER_MODELS_KEYS_TO_CLS = {
    cls.__name__: cls for cls in _PARAMETER_MODELS
}


def configure_argument_parser_parameter_model(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """

    :param parser:
    :return:
    """
    parser.add_argument('--parameter_model',
                        type=str,
                        default=_PARAMETER_MODELS_DEFAULT.__name__,
                        choices=list(_PARAMETER_MODELS_KEYS_TO_CLS.keys()),
                        help='Model that should be used to map locations to parameter values. If none is specified, '
                             'each location is mapped to a tuple of parameters.',
                        )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    """
    Apply some (non-exhaustive) checks for correct configuration of the arguments
    :param args: the parsed arguments
    """

    model_cls = args.model_cls

    if model_cls == MeanModel:
        assert not args.hold_out_locations, 'This model does not support held-out locations'
    if issubclass(model_cls, BaseLocalAccumulationModel):
        assert not args.hold_out_locations, 'This model does not support held-out locations'
        assert args.include_temperature

    pass  # TODO


"""

    CODE FOR CONSTRUCTING THE EXPERIMENT BASED ON THE PARSED ARGUMENTS

"""


def set_config_using_args(args: argparse.Namespace) -> None:
    set_randomness_seed(args)


def set_randomness_seed(args: argparse.Namespace) -> None:

    # Determine the seed used for PyTorch models
    # Different cases are distinguished:
    # (1) the `seed` argument was set -> use this as seed and ignore the `seed_torch` argument
    if args.seed is not None:
        seed_torch = args.seed
    # (2) only the `seed_year_split` was set -> use this one
    elif args.seed_torch is not None:
        seed_torch = args.seed_torch
    # (3) no seed was explicitly given -> use the default one in the config
    else:
        seed_torch = config.SEED

    torch.manual_seed(seed_torch)

    # np.random.seed(seed)


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
    locations = _LOCATION_GROUPS[args.locations]

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


def fit_model(args: argparse.Namespace, dataset: Dataset) -> tuple:
    """

    :param args:
    :param dataset:
    :return:
    """
    model_cls = args.model_cls

    # TODO -- fit differentiable utah model using grid search
    if issubclass(model_cls, BaseAccumulationModel):
        return model_cls.fit(dataset)
    if issubclass(model_cls, BaseLocalAccumulationModel):
        return model_cls.fit(dataset)
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

    :param model_cls:
    :param dataset:
    :param args:
    # :param model_kwargs:
    :return:
    """
    # model_kwargs = model_kwargs or dict()
    device = torch.device('cuda') if torch.cuda.is_available() and not args.disable_cuda else torch.device('cpu')

    model_kwargs = dict()
    if issubclass(model_cls, BaseTorchAccumulationModel):
        if args.parameter_model is None or _PARAMETER_MODELS_KEYS_TO_CLS[args.parameter_model] == LocalParams:
            locations = _LOCATION_GROUPS[args.locations]
            model_kwargs['param_model'] = LocalParams(locations=locations)
        elif _PARAMETER_MODELS_KEYS_TO_CLS[args.parameter_model] == GlobalParams:
            model_kwargs['param_model'] = GlobalParams()
        elif _PARAMETER_MODELS_KEYS_TO_CLS[args.parameter_model] == ParamNet:
            model_kwargs['param_model'] = ParamNet()
        else:
            raise ConfigException(f'Cannot configure parameter model "{args.parameter_model}"')

    kwargs = {
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'scheduler_step_size': args.scheduler_step_size,
        'scheduler_decay': args.scheduler_decay,
        'clip_gradient': args.clip_gradient,
        'f_optim': _OPTIMIZERS[args.optimizer],
        'optim_kwargs': {
            'lr': args.lr,
        },
        'device': device,
        'model_kwargs': model_kwargs,
    }

    return model_cls.fit(dataset,
                         **kwargs,
                         )


def evaluate_model_using_args(args: argparse.Namespace,
                              model: BaseModel,
                              model_name: str,
                              dataset: Dataset,
                              ):
    """

    :param args:
    :param model:
    :param model_name:
    :param dataset:
    :return:
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


class ConfigException(Exception):
    pass  # Exception for misconfigured runs
