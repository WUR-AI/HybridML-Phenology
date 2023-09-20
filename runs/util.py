import argparse

from sklearn.model_selection import train_test_split

import torch
import torch.optim

import config
import data.bloom_doy
import data.regions_japan
from datasets.dataset import Dataset
from models.base_torch import BaseTorchModel

_LOCATION_GROUPS = {
    'all': data.bloom_doy.get_locations(),
    'japan': data.bloom_doy.get_locations_japan(),
    'switzerland': data.bloom_doy.get_locations_switzerland(),
    'south_korea': data.bloom_doy.get_locations_south_korea(),
    'usa': data.bloom_doy.get_locations_usa(),
    'japan_wo_okinawa': list(data.regions_japan.LOCATIONS_WO_OKINAWA.keys()),
    'japan_yedoenis': list(data.regions_japan.LOCATIONS_JAPAN_YEDOENIS.keys()),
}


def configure_argparser_main(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--model_name',
                        type=str,
                        )

    parser.add_argument('--seed',  # TODO -- option to generate randomly
                        type=int,
                        )

    parser.add_argument('--skip_fit',
                        action='store_true',
                        )

    parser.add_argument('--skip_eval',
                        action='store_true',
                        )

    return parser


def configure_argparser_dataset(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:  # TODO -- debug option

    parser.add_argument('--include_temperature',
                        action='store_true',
                        )
    parser.add_argument('--include_photoperiod',
                        action='store_true',
                        )
    parser.add_argument('--locations',
                        type=str,
                        choices=list(_LOCATION_GROUPS.keys()),
                        default='all',
                        )
    parser.add_argument('--seed_year_split',
                        type=int,
                        )
    parser.add_argument('--train_size_years',
                        type=float,
                        default=0.75,
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
                        )

    return parser


def get_configured_dataset(args: argparse.Namespace) -> tuple:
    if args.seed is not None:
        seed_year_split = args.seed
    elif args.seed_year_split is not None:
        seed_year_split = args.seed_year_split
    else:
        seed_year_split = config.SEED

    if args.seed is not None:
        seed_location_split = args.seed
    elif args.seed_year_split is not None:
        seed_location_split = args.seed_location_split
    else:
        seed_location_split = config.SEED

    locations = _LOCATION_GROUPS[args.locations]

    if args.hold_out_locations:
        locations_train, locations_test = train_test_split(locations,
                                                           train_size=args.train_size_locations,
                                                           random_state=seed_location_split,
                                                           shuffle=True,
                                                           )
    else:
        locations_train, locations_test = locations, locations

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

    dataset = Dataset(
        **kwargs,
    )

    info = dict()

    return dataset, info


_OPTIMIZERS = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam,
}


def configure_argparser_fit_torch(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
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


def fit_torch_model_using_args(model_cls: callable,
                               dataset: Dataset,
                               args: argparse.Namespace,
                               model_kwargs: dict = None,
                               ) -> tuple:
    model_kwargs = model_kwargs or dict()
    device = torch.device('cuda') if torch.cuda.is_available() and not args.disable_cuda else torch.device('cpu')

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
